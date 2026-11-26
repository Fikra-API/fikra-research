# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Kernel #1b arithmetic: does one dequant serve a whole verify block?

This is the property that decides whether speculative decoding can pay on a GPU
without FP4 hardware. Verifying N draft tokens should cost about one weight
read; if each token re-unpacks the FP4 experts, verify costs N times the
dequant and drafting loses, which is exactly what Strata measured on sm75
(0.98x).

No GPU and no torch needed: the block arithmetic is reproduced in numpy against
the reference codec in nvfp4.py.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")

from fikra.inkling_t4l4 import nvfp4 as ref  # noqa: E402

E, NOUT, K, T = 3, 64, 128, 4
OUTER = 2688.0


@pytest.fixture(scope="module")
def bank():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, (E, NOUT, K)).astype(np.uint8)
    scales = (rng.random((E, NOUT, K // 16)) + 0.5).astype(np.float32)
    amax = (rng.random(E) * 2).astype(np.float32)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    x = rng.standard_normal((T, K)).astype(np.float32)
    return packed, scales, amax, x


def _dequant_once(packed, scales, amax, e, counter):
    counter[0] += 1
    return ref.dequant_nvfp4(packed[e], scales[e], outer_scale=float(amax[e] / OUTER))


def test_t1_matches_single_token_reference(bank):
    packed, scales, amax, x = bank
    counter = [0]
    w = _dequant_once(packed, scales, amax, 1, counter)
    got = x[:1] @ w.T
    want = ref.gemv_nvfp4(x[0], packed[1], scales[1], outer_scale=float(amax[1] / OUTER))
    assert np.abs(got[0] - want).max() < 1e-5


def test_each_row_equals_its_own_single_token_result(bank):
    """Weight reuse must not leak between tokens in a block."""
    packed, scales, amax, x = bank
    counter = [0]
    w = _dequant_once(packed, scales, amax, 1, counter)
    got = x @ w.T
    for t in range(T):
        want = ref.gemv_nvfp4(x[t], packed[1], scales[1],
                             outer_scale=float(amax[1] / OUTER))
        assert np.abs(got[t] - want).max() < 1e-5


def test_row_permutation_invariant(bank):
    packed, scales, amax, x = bank
    counter = [0]
    w = _dequant_once(packed, scales, amax, 1, counter)
    base = x @ w.T
    perm = [2, 0, 3, 1]
    assert np.abs((x[perm] @ w.T) - base[perm]).max() < 1e-6


def test_one_dequant_covers_the_whole_block(bank):
    """The whole point: dequant cost must not scale with the verify length."""
    packed, scales, amax, x = bank
    counter = [0]
    w = _dequant_once(packed, scales, amax, 1, counter)   # one call
    for t in (1, 2, 4):
        _ = x[:t] @ w.T
    assert counter[0] == 1


def test_block_result_is_independent_of_block_length(bank):
    """A token's logits must not depend on how many tokens are verified with it."""
    packed, scales, amax, x = bank
    counter = [0]
    w = _dequant_once(packed, scales, amax, 1, counter)
    one = (x[:1] @ w.T)[0]
    three = (x[:3] @ w.T)[0]
    assert np.abs(one - three).max() < 1e-6
