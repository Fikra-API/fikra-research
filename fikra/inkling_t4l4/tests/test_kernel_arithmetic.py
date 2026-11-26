# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Validate the FP4-dequant GEMV arithmetic on CPU, before any GPU minute.

The Triton kernel cannot run without a GPU, so the expensive risk is not
scheduling, it is arithmetic: nibble order, scale grouping and the per-expert
global factor. This test reimplements the kernel's arithmetic in torch and
compares it against the pure-float oracle on identical codes. If these two
disagree the kernel is wrong; if they agree, a GPU run is only measuring speed.

Skipped without torch. Never needs a GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fikra.inkling_t4l4.gemv_torch import reference_fp4_gemv, unpack_fp4  # noqa: E402
from fikra.inkling_t4l4.gemv_triton import nvfp4_gemv  # noqa: E402

N, K, E = 64, 128, 8
OUTER_DENOM = 2688.0


@pytest.fixture
def bank():
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (E, N, K), dtype=torch.int64)
    scales = torch.rand(E, N, K // 16, dtype=torch.float32) + 0.5
    amax = torch.rand(E, dtype=torch.float32) * 2 + 0.5
    x = torch.randn(K, dtype=torch.float32)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return codes, scales, amax, x, packed


def test_arithmetic_matches_oracle(bank):
    codes, scales, amax, x, packed = bank
    e = 0
    w = unpack_fp4(packed[e]) * scales[e].repeat_interleave(16, dim=-1)
    got = (w @ x) * (amax[e] / OUTER_DENOM)
    expected = reference_fp4_gemv(codes[e], scales[e], x) * (amax[e] / OUTER_DENOM)
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_nibble_order_matters(bank):
    """Low nibble is even k. Swapping must change the answer."""
    codes, scales, amax, x, packed = bank
    e = 0
    base = unpack_fp4(packed[e]) * scales[e].repeat_interleave(16, dim=-1) @ x
    swapped = (codes[e][:, 1::2] | (codes[e][:, 0::2] << 4)).to(torch.uint8)
    other = unpack_fp4(swapped) * scales[e].repeat_interleave(16, dim=-1) @ x
    assert not torch.allclose(base, other, atol=1e-3)


def test_scale_grouping_is_per_16_not_per_row(bank):
    """One scale per 16 contiguous k. Per-row scales must differ."""
    codes, scales, amax, x, packed = bank
    e = 0
    base = unpack_fp4(packed[e]) * scales[e].repeat_interleave(16, dim=-1) @ x
    row = scales[e, :, :1].repeat(1, K // 16)
    other = unpack_fp4(packed[e]) * row.repeat_interleave(16, dim=-1) @ x
    assert not torch.allclose(base, other, atol=1e-3)


def test_cpu_dispatch_applies_global_scale(bank):
    """The fallback path must include amax/2688, or every expert is off."""
    codes, scales, amax, x, packed = bank
    experts = torch.tensor([0, 3, 5], dtype=torch.int64)
    got = nvfp4_gemv(x, packed, scales, experts, n=N, k=K, global_scale=amax / OUTER_DENOM)
    want = torch.stack([
        reference_fp4_gemv(codes[e], scales[e], x) * (amax[e] / OUTER_DENOM) for e in experts
    ])
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)
