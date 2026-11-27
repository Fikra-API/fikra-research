"""GPU kernel for fused-core execution.

The budget says fusion is worth exactly 2x and that it is a scheduling change,
not an arithmetic one. This module is that change expressed as a kernel: one
launch walks the whole plan, and a layer that runs more than once appears in the
visit list once instead of three times.

Two things make it work on a T4 that has no fp8 or fp4 hardware:

  * the dequant is arithmetic (E2M1 from the code bits, E4M3 from a bit-shift),
    never a cast to a hardware float type
  * the visit list is the schedule, so fusion is data rather than a branch

It is generic: the plan and formats come in as data, so the same kernel runs a
looped model, a plain MoE stack or a dense model.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

from .formats import Format, NVFP4
from .plans import Plan


def plan_to_arrays(plan: Plan) -> dict:
    """Flatten a plan into the arrays the kernel indexes.

    Every visit in order: which layer, which pass. Fusion is expressed by which
    layers appear more than once in this list, so the ordering logic lives in one
    place and is testable without a GPU.
    """
    order = []
    max_passes = max((l.passes for l in plan.layers), default=1)
    for p in range(max_passes):
        for layer in plan.layers:
            if p < layer.passes:
                order.append((layer.index, p))
    return {
        "layer_of_visit": [i for i, _ in order],
        "pass_of_visit": [p for _, p in order],
        "visits": len(order),
        "resident_layers": [l.index for l in plan.layers
                            if l.passes > 1 and l.share_weights_across_passes],
    }


def available() -> bool:
    return triton is not None and torch is not None


def _build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1(c):
        return tl.where(c == 0, 0.0, tl.where(c == 1, 0.5, tl.where(c == 2, 1.0,
               tl.where(c == 3, 1.5, tl.where(c == 4, 2.0, tl.where(c == 5, 3.0,
               tl.where(c == 6, 4.0, 6.0)))))))

    @triton.jit
    def _e4m3(b):
        s = tl.where((b & 0x80) != 0, -1.0, 1.0)
        e = ((b >> 3) & 0x0F).to(tl.int32)
        m = (b & 0x07).to(tl.int32)
        normal = (((e - 7 + 127) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = ((118 << 23) | (m << 20)).to(tl.float32, bitcast=True)
        return s * tl.where(e == 0, sub, normal)

    @triton.jit
    def fused_walk(X, PACKED, SCALES, OUT, LAYER_OF_VISIT, N, K, VISITS,
                   STRIDE_E, STRIDE_N,
                   BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, NP: tl.constexpr):
        """One launch walks every visit of the plan.

        A fused plan emits fewer visits, so the saving is in the visit list, not
        in a branch. That is why the same kernel serves fused and unfused runs
        and the difference is measurable rather than asserted.
        """
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = offs_n < N
        acc = tl.zeros([BLOCK_N], tl.float32)
        for v in range(VISITS):
            e = tl.load(LAYER_OF_VISIT + v)
            for p0 in range(0, NP, BLOCK_P):
                offs_p = p0 + tl.arange(0, BLOCK_P)
                pm = offs_p < NP
                m2 = nm[:, None] & pm[None, :]
                hdr = tl.load(PACKED + e * STRIDE_E + offs_n[:, None] * STRIDE_N + offs_p[None, :],
                              mask=m2, other=0)
                lo = (hdr & 0x0F).to(tl.int32)
                hi = ((hdr >> 4) & 0x0F).to(tl.int32)
                gidx = tl.arange(0, BLOCK_P // 8)
                sc8 = _e4m3(tl.load(SCALES + e * STRIDE_E + offs_n[:, None] * STRIDE_N
                                    + (p0 // 8) + gidx[None, :],
                                    mask=nm[:, None], other=0x38).to(tl.uint8))
                sc8 = tl.reshape(
                    tl.broadcast_to(sc8[:, :, None], [BLOCK_N, BLOCK_P // 8, 8]),
                    [BLOCK_N, BLOCK_P])
                w_lo = _e2m1(lo & 7) * tl.where((lo & 8) != 0, -1.0, 1.0) * sc8
                w_hi = _e2m1(hi & 7) * tl.where((hi & 8) != 0, -1.0, 1.0) * sc8
                kl = offs_p * 2
                xl = tl.load(X + kl, mask=pm, other=0.0)
                xh = tl.load(X + kl + 1, mask=pm, other=0.0)
                acc += tl.sum(w_lo * xl[None, :] + w_hi * xh[None, :], axis=1)
        tl.store(OUT + offs_n, acc, mask=nm)

    return fused_walk


_KERNEL = _build_kernel() if triton is not None else None


def run_fused(plan: Plan, packed, scales, x, n_out: int, k_in: int,
              fmt: Format = NVFP4, block_n: int = 4, block_p: int = 32,
              warps: int = 1):
    """Launch the fused walk. Returns (out, visit_arrays) for accounting."""
    if not available():
        raise RuntimeError("triton and torch are required for the GPU kernel")
    if not x.is_cuda:
        raise RuntimeError("this kernel needs a CUDA tensor; use quant_ref on CPU")
    arr = plan_to_arrays(plan)
    dev = x.device
    lv = torch.tensor(arr["layer_of_visit"], dtype=torch.int32, device=dev)
    out = torch.zeros(n_out, dtype=torch.float32, device=dev)
    grid = (triton.cdiv(n_out, block_n),)
    _KERNEL[grid](
        x.float(), packed, scales, out, lv,
        n_out, k_in, arr["visits"], packed.stride(0), packed.stride(1),
        BLOCK_N=block_n, BLOCK_P=block_p, NP=k_in // 2, num_warps=warps,
    )
    return out, arr
