# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Expert-side kernels for both on-disk formats, plus the capped SwiGLU.

Kernel #1 in ``gemv_triton`` covers NVFP4 only. Inkling's coda layers (20-23)
store their experts as MXFP8: FP8 E4M3 codes with one ue8m0 power-of-two scale
per 32x32 block. The arithmetic differs enough that a format branch inside one
kernel body would be a compile-time fork anyway, so it is a second kernel.

Both kernels keep the property that made the NVFP4 one work: they stream packed
weights and never materialise a float weight matrix. The win on a T4 is bytes not
moved, because the card has no FP4 or block-scaled FP8 tensor cores at all.

The activation is the checkpoint's, not a textbook SwiGLU. Inkling caps the gate
and the up projection (``mlp_gate_softcap`` 4.0, ``mlp_up_softcap`` 25.0) and
multiplies the sigmoid of the *uncapped* gate. Applying a plain silu(gate)*up
gives numbers that look plausible and are wrong, which is the worst failure mode
available, so the caps are kernel arguments with no default.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - numpy-only environments still import
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

MXFP8_BLOCK = 32


def available() -> bool:
    return triton is not None and torch is not None


if triton is not None:

    @triton.jit
    def _mxfp8_gemv_kernel(
        X,                  # fp32 [K] or [topk, K]
        CODES,              # fp8 e4m3 uint8 [E, N, K]
        SCALES,             # uint8 ue8m0 [E, N, K/32] (one exponent per 32 K, per row)
        OUT,                # fp32 [topk, N]
        EXPERTS,            # int32 [topk]
        N, K,
        stride_ce, stride_cn,
        stride_se, stride_sn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        X_2D: tl.constexpr,
    ):
        """y[n] = sum_k e4m3(codes[n,k]) * 2**(scale[n//32, k//32] - 127) * x[k].

        The checkpoint's MXFP8 scale tensor is [E, N, K/32] -- one ue8m0
        exponent per 32 K per output row, not a 32x32 block -- so each row's
        exponent is a single broadcast multiply over its own 32-wide k group.
        """
        pid_n = tl.program_id(0)
        pid_e = tl.program_id(1)
        expert = tl.load(EXPERTS + pid_e)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        x_row = X + (pid_e * K if X_2D else 0)

        acc = tl.zeros([BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            k_blk = offs_k // SCALE_BLOCK

            raw = tl.load(CODES + expert * stride_ce + offs_n[:, None] * stride_cn + offs_k[None, :],
                          mask=n_mask[:, None] & k_mask[None, :], other=0)
            w = _e4m3_to_float(raw.to(tl.uint8, bitcast=False))

            sb = tl.load(SCALES + expert * stride_se + offs_n[:, None] * stride_sn + k_blk[None, :],
                         mask=n_mask[:, None] & k_mask[None, :], other=127)
            w = w * tl.exp2((sb.to(tl.float32) - 127.0))

            x = tl.load(x_row + offs_k, mask=k_mask, other=0.0)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + pid_e * N + offs_n, acc, mask=n_mask)

    @triton.jit
    def _e4m3_to_float(b):
        """E4M3 (finite, no inf) from its bytes, by bit assembly.

        The T4 has no fp8 datatype, so this is the only way to get a float out
        of an E4M3 byte on this card. Subnormals (exp 0) become (m/8) * 2**-6.
        """
        b32 = b.to(tl.int32)
        sign = tl.where((b32 & 0x80) != 0, -1.0, 1.0)
        e = (b32 >> 3) & 0x0F
        m = b32 & 0x07
        normal = (((e - 7 + 127) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = (((118 << 23) | (m << 20)).to(tl.float32, bitcast=True))
        return sign * tl.where(e == 0, sub, normal)


def mxfp8_gemv(x, codes, scales, experts, n, k):
    """Triton MXFP8 GEMV when CUDA is present, otherwise the numpy oracle.

    Shapes: codes [E, N, K] uint8 (E4M3), scales [E, N, K/32] uint8 (ue8m0),
    x [K] fp32, experts [topk] int32. Returns [topk, N] fp32.
    """
    if triton is not None and x.is_cuda:
        out = torch.zeros((experts.numel(), n), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(n, 16), experts.numel())
        _mxfp8_gemv_kernel[grid](
            x.contiguous(), codes.contiguous(), scales.contiguous(), out,
            experts.to(torch.int32),
            n, k, codes.stride(0), codes.stride(1),
            scales.stride(0), scales.stride(1),
            BLOCK_N=16, BLOCK_K=256, SCALE_BLOCK=MXFP8_BLOCK,
            X_2D=(x.dim() == 2), num_warps=4,
        )
        return out

    import numpy as np
    from .nvfp4 import dequant_mxfp8

    x2 = x if x.dim() == 2 else x.unsqueeze(0).expand(experts.numel(), -1)
    rows = []
    for slot, e in enumerate(experts.tolist()):
        w = dequant_mxfp8(codes[e].cpu().numpy(), scales[e].cpu().numpy(), MXFP8_BLOCK)
        rows.append(w @ x2[slot].cpu().numpy().astype(np.float32))
    return torch.tensor(np.stack(rows), dtype=torch.float32) if rows else torch.zeros((0, n))


def capped_swiglu(gate_up, gate_cap: float, up_cap: float):
    """Inkling's SiTU: interleaved (gate, up) columns, tanh caps, sigmoid of the
    uncapped gate, no extra scale.

    ``gate_up`` is [..., 2 * I] with gate at even and up at odd positions, which
    is what w13 produces: its rows are interleaved rather than concatenated.
    """
    g, u = gate_up.float().unflatten(-1, (-1, 2)).unbind(-1)
    sig = torch.sigmoid(g)
    if gate_cap and gate_cap > 0:
        g = gate_cap * torch.tanh(g / gate_cap)
    if up_cap and up_cap > 0:
        u = up_cap * torch.tanh(u / up_cap)
    return g * sig * u


def _self_check() -> None:
    """Shapes and the cap arithmetic. Kept out of import: a broken assumption
    here must fail a test, not kill a training run at import time."""
    if torch is None:
        return
    # gate 1e4 saturates to 4, up 1e4 saturates to 25, sigmoid(1e4) is 1
    out = capped_swiglu(torch.tensor([[1e4, 1e4]]), 4.0, 25.0)
    assert abs(float(out[0, 0]) - 100.0) < 0.1, float(out[0, 0])
    # a large negative gate saturates to -4 and its sigmoid is ~0, so the whole
    # product vanishes -- that is the property that distinguishes SiTU from
    # silu(gate)*up, which would come out near -4 * 25 * 1
    neg = capped_swiglu(torch.tensor([[-1e4, 1e4]]), 4.0, 25.0)
    assert abs(float(neg[0, 0])) < 1e-6, float(neg[0, 0])
    # zero in, zero out
    assert float(capped_swiglu(torch.zeros(1, 2), 4.0, 25.0).sum()) == 0.0
