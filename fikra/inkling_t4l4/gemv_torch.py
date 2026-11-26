# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Torch reference (and CPU path) for the FP4-dequant expert GEMV.

This is the oracle the Triton kernel is checked against, and it is also the
only portable path when no GPU is present. It dequantises per expert tile,
never materialising the whole expert bank.
"""
from __future__ import annotations

import torch


def _fp4_lut(device, dtype=torch.float32) -> torch.Tensor:
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=dtype)
    return torch.cat([lut, -lut])


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [.., K, N/2] -> float32 [.., K, N]; low nibble = even column."""
    lut = _fp4_lut(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    out = torch.stack([lo, hi], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequant_nvfp4(packed: torch.Tensor, scales: torch.Tensor, block: int = 16,
                  outer_scale: float | None = None) -> torch.Tensor:
    """[N, K/2] uint8 + [N, K/16] FP8 -> float32 [N, K] (weights on last axis).

    ``outer_scale`` is the per-expert amax / outer_scale_denominator. The kernel
    needs it (see gemv_triton) and the reference must apply it identically, or a
    CPU/GPU comparison silently disagrees by that factor.
    """
    vals = unpack_fp4(packed)
    s = scales.to(torch.float32)
    rep = s.repeat_interleave(block, dim=-1)
    out = vals * rep
    if outer_scale is not None:
        out = out * float(outer_scale)
    return out


def gemv_nvfp4(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               expert: int, block: int = 16) -> torch.Tensor:
    """y[slot] = W_expert x, for a [E, N/2, K/2]-shaped bank row."""
    w = dequant_nvfp4(packed[expert], scales[expert], block)  # [N, K]
    return w @ x.to(torch.float32)


def moe_forward(x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor,
                w13: torch.Tensor, s13: torch.Tensor,
                w2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """One token through the expert bank.

    x: [K], topk_idx/topk_w: [T]. w13: [E, 2I/2, K/2], s13: [E, 2I/2, K/16],
    w2: [E, H/2, I/2], s2: [E, H/2, I/16]. Returns [H] (already router-weighted).
    """
    hidden = w2.shape[1] * 2
    out = torch.zeros(hidden, dtype=torch.float32, device=x.device)
    for slot in range(topk_idx.numel()):
        e = int(topk_idx[slot])
        gu = gemv_nvfp4(x, w13, s13, e)          # [2I]
        gate, up = gu.chunk(2)
        act = torch.nn.functional.silu(gate) * up  # [I]
        y = gemv_nvfp4(act, w2, s2, e)            # [H]
        out = out + topk_w[slot].to(torch.float32) * y
    return out


def reference_fp4_gemv(w_codes_int: torch.Tensor, scales_f32: torch.Tensor, x: torch.Tensor,
                       block: int = 16) -> torch.Tensor:
    """Pure-float oracle: given already-decoded int codes 0..15, compute y = W x."""
    lut = _fp4_lut(w_codes_int.device)
    w = lut[w_codes_int.long()] * scales_f32.repeat_interleave(block, dim=1)
    return w @ x
