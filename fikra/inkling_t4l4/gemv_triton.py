# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Kernel #1: NVFP4-dequant MoE GEMV for T4 (sm75) and L4 (sm89).

Decode reads 8 of 128 experts per layer, so the matmul is a GEMV and the cost
is bytes moved, not FLOPs. The kernel therefore never materialises BF16
weights: each program streams a 16-column tile of packed FP4 plus one FP8
scale per row, dequantises in registers, and accumulates with the activation
vector. BLOCK_N=16 matches the NVFP4 scale group exactly, so no scale is
loaded twice and no cross-lane reduce is needed; 16-wide tiles also keep 2 KB
of nibbles in flight instead of the 16 KB a BLOCK_N=128 tile would need.

Formats (verified against the released checkpoint in ``model_spec``):
  packed  uint8 [E, N, K/2]  low nibble = even k, high nibble = odd k
  scales  fp8   [E, N, K/16] one E4M3 scale per 16 k (k-major, contiguous)
  global  fp32  [E]          amax / outer_scale_denominator, 2688 for every layer

The global factor is not optional: without it each expert is off by its own
amax/2688, which ranges over experts and would look like a routing bug.

Layers 20-23 store experts as MXFP8; ``mxfp8_gemv`` covers them with a 32x32
block scale. Neither path uses FP8/FP4 tensor cores, which is deliberate:
sm75 has neither and sm89 lacks block scaling, so the decode win has to come
from not reading BF16 weights, not from a faster matmul unit.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # CPU-only / numpy-only environments still import this module
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments still import this module
    triton = None
    tl = None

from .kernel_spec import BLOCK_K, BLOCK_N, NUM_WARPS


def _build_fp4_lut_values():
    return [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


if triton is not None:

    @triton.jit
    def _e4m3_to_float(b):
        """E4M3 byte -> float, by bit assembly (sm75 has no fp8 datatype)."""
        b32 = b.to(tl.uint8, bitcast=False).to(tl.int32)
        sign = tl.where((b32 & 0x80) != 0, -1.0, 1.0)
        e = (b32 >> 3) & 0x0F
        m = b32 & 0x07
        normal = (((e - 7 + 127) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = ((118 << 23) | (m << 20)).to(tl.float32, bitcast=True)
        return sign * tl.where(e == 0, sub, normal)

    @triton.jit
    def _nvfp4_gemv_kernel(
        X,                 # fp32 [K]
        PACKED,            # uint8 [E, N, K/2]
        SCALES,            # fp8   [E, N, K/16]
        OUT,               # fp32 [topk, N]
        EXPERTS,           # int32 [topk] selected expert ids
        GLOBAL_SCALE,      # fp32  [E] amax / outer_scale_denominator
        N, K,
        stride_pe, stride_pn,
        stride_se, stride_sn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)   # column tile
        pid_e = tl.program_id(1)   # slot in the top-k list
        expert = tl.load(EXPERTS + pid_e)
        gscale = tl.load(GLOBAL_SCALE + expert)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # E2M1 magnitudes 0,0.5,1,1.5,2,3,4,6 for codes 0..7; low 3 bits index
        # the table, bit 3 is the sign (two's-complement negate of the magnitude).
        code = tl.arange(0, 16)
        absc = code & 7
        mag = (absc == 1) * 0.5 + (absc == 2) * 1.0 + (absc == 3) * 1.5 \
            + (absc == 4) * 2.0 + (absc == 5) * 3.0 + (absc == 6) * 4.0 + (absc == 7) * 6.0
        fp4_table = tl.where(code & 8, -mag, mag).to(tl.float32)  # [16]

        acc = tl.zeros([BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            sb = tl.load(SCALES + expert * stride_se + offs_n[:, None] * stride_sn
                         + offs_k[None, :] // 16,
                         mask=n_mask[:, None] & k_mask[None, :], other=0x38)
            sc = _e4m3_to_float(sb)
            hdr = tl.load(PACKED + expert * stride_pe + offs_n[:, None] * stride_pn + offs_k[None, :] // 2,
                          mask=n_mask[:, None] & k_mask[None, :], other=0)
            lo = (hdr & 0x0F).to(tl.int32)
            hi = ((hdr >> 4) & 0x0F).to(tl.int32)
            codes = tl.where(offs_k[None, :] % 2 == 0, lo, hi)        # [BLOCK_N, BLOCK_K]
            w = tl.sum(tl.where(codes[:, :, None] == code[None, None, :],
                                fp4_table[None, None, :], 0.0), axis=2)
            w = w * sc * gscale
            x = tl.load(X + offs_k, mask=k_mask, other=0.0)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + pid_e * N + offs_n, acc, mask=n_mask)


def nvfp4_gemv(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               experts: torch.Tensor, n: int, k: int,
               global_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Reference dispatch: Triton when a GPU is present, otherwise torch.

    Shapes: packed [E, N, K/2] uint8, scales [E, N, K/16] fp8 (E4M3),
    global_scale [E] fp32 (amax / outer_scale_denominator), x [K] fp32,
    experts [topk] int32. Returns [topk, N] fp32.

    ``global_scale`` defaults to ones so shape-level tests still run, but a
    real checkpoint result without it is wrong.
    """
    if triton is not None and x.is_cuda:
        # The scales arrive as raw E4M3 bytes. A float8 tensor cannot even be
        # handed to Triton on sm75, so reinterpret before it crosses.
        if scales.dtype != torch.uint8:
            scales = scales.view(torch.uint8)
        if packed.dtype != torch.uint8:
            packed = packed.view(torch.uint8)
        out = torch.zeros((experts.numel(), n), dtype=torch.float32, device=x.device)
        if global_scale is None:
            global_scale = torch.ones(packed.shape[0], dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(n, BLOCK_N), experts.numel())
        _nvfp4_gemv_kernel[grid](
            x.contiguous(), packed.contiguous(), scales.contiguous(), out,
            experts.to(torch.int32), global_scale.contiguous(),
            n, k, packed.stride(0), packed.stride(1), scales.stride(0), scales.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=NUM_WARPS,
        )
        return out
    from .gemv_torch import dequant_nvfp4
    rows = []
    for slot, e in enumerate(experts.tolist()):
        # Bank rows are [N, K//2] and [N, K//16]: the transposed reference layout.
        g = None if global_scale is None else float(global_scale[e])
        w = dequant_nvfp4(packed[e], scales[e].to(torch.float32), outer_scale=g)
        rows.append(w @ x.to(torch.float32))
    return torch.stack(rows) if rows else torch.zeros((0, n))


# ---------------------------------------------------------------------------
# Kernel #1b: batched dequant GEMM (N tokens verified per weight read)
# ---------------------------------------------------------------------------
# The GEMV above is correct but it makes speculative decoding pay nothing on T4.
# Verifying a block of N draft tokens should cost about the same as decoding one
# token, because every expert weight is read once for the whole block and decode
# is memory-bound. A GEMV reads the weights N times for N tokens, so it throws
# that property away.
#
# This kernel dequantises each FP4 block once into registers and multiplies it
# against all N activations before moving on, so the dequant work per weight is
# amortised over the block. If that makes the per-token cost flat in N, spec
# decode becomes a real multiplier on a card with no FP4 hardware, which is the
# paper's claim to test.
#
# Shapes: packed [E, N_OUT, K/2], scales [E, N_OUT, K/16], X [T, K] fp32,
# experts [topk] int32, OUT [topk, T, N_OUT] fp32.


def _build_batched_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _nvfp4_gemm_kernel(
        X,                 # fp32 [T, K]
        PACKED,            # uint8 [E, N, K/2]
        SCALES,            # fp8   [E, N, K/16]
        OUT,               # fp32 [topk, T, N]
        EXPERTS,           # int32 [topk]
        GLOBAL_SCALE,      # fp32 [E]
        N, K, T,
        stride_pe, stride_pn,
        stride_se, stride_sn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        T_MAX: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_e = tl.program_id(1)
        expert = tl.load(EXPERTS + pid_e)
        gscale = tl.load(GLOBAL_SCALE + expert)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        code = tl.arange(0, 16)
        absc = code & 7
        mag = (absc == 1) * 0.5 + (absc == 2) * 1.0 + (absc == 3) * 1.5 \
            + (absc == 4) * 2.0 + (absc == 5) * 3.0 + (absc == 6) * 4.0 + (absc == 7) * 6.0
        fp4_table = tl.where(code & 8, -mag, mag).to(tl.float32)

        offs_t = tl.arange(0, T_MAX)
        t_mask = offs_t < T
        acc = tl.zeros([T_MAX, BLOCK_N], tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            sb = tl.load(SCALES + expert * stride_se + offs_n[:, None] * stride_sn
                         + offs_k[None, :] // 16,
                         mask=n_mask[:, None] & k_mask[None, :], other=0x38)
            sc = _e4m3_to_float(sb)
            hdr = tl.load(PACKED + expert * stride_pe + offs_n[:, None] * stride_pn
                          + offs_k[None, :] // 2,
                          mask=n_mask[:, None] & k_mask[None, :], other=0)
            lo = (hdr & 0x0F).to(tl.int32)
            hi = ((hdr >> 4) & 0x0F).to(tl.int32)
            codes = tl.where(offs_k[None, :] % 2 == 0, lo, hi)
            # Dequantise ONCE for this block: w is [BLOCK_N, BLOCK_K].
            w = tl.sum(tl.where(codes[:, :, None] == code[None, None, :],
                                fp4_table[None, None, :], 0.0), axis=2)
            w = w * sc * gscale
            # All T activation rows reuse the same w.
            x = tl.load(X + offs_t[:, None] * K + offs_k[None, :],
                        mask=t_mask[:, None] & k_mask[None, :], other=0.0)   # [T_MAX, BLOCK_K]
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")

        tl.store(OUT + pid_e * T * N + offs_t[:, None] * N + offs_n[None, :],
                 acc, mask=t_mask[:, None] & n_mask[None, :])

    return _nvfp4_gemm_kernel


_BATCHED = _build_batched_kernel() if triton is not None else None


def nvfp4_gemm_batched(x, packed, scales, experts, n, k, t,
                       global_scale=None, t_max=16):
    """N-token dequant GEMM. Returns [topk, T, N] fp32.

    Falls back to the torch reference without CUDA, so the cost curve can be
    compared against the same arithmetic the kernel implements.
    """
    if triton is not None and x.is_cuda:
        out = torch.zeros((experts.numel(), t, n), dtype=torch.float32, device=x.device)
        if global_scale is None:
            global_scale = torch.ones(packed.shape[0], dtype=torch.float32, device=x.device)
        tm = 1
        while tm < t:
            tm *= 2
        tm = max(tm, 2)
        grid = (triton.cdiv(n, BLOCK_N), experts.numel())
        _BATCHED[grid](
            x.contiguous(), packed.contiguous(), scales.contiguous(), out,
            experts.to(torch.int32), global_scale.contiguous(),
            n, k, t, packed.stride(0), packed.stride(1), scales.stride(0), scales.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, T_MAX=tm, num_warps=NUM_WARPS,
        )
        return out
    from .gemv_torch import dequant_nvfp4
    rows = []
    for e in experts.tolist():
        g = None if global_scale is None else float(global_scale[e])
        w = dequant_nvfp4(packed[e], scales[e].to(torch.float32), outer_scale=g)
        rows.append(x.to(torch.float32) @ w.T)   # [T, N]
    return torch.stack(rows) if rows else torch.zeros((0, t, n))
