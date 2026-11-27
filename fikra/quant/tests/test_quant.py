# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tests for the model-agnostic quant layer.

These are the contracts a new model relies on: a Format describes packing, a
Plan describes execution, and the numpy path defines the arithmetic every GPU
kernel is checked against. If any of this drifts, every model built on it
silently produces wrong numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from fikra.quant import formats as F
from fikra.quant import plans as P
from fikra.quant import quant_ref as Q


# --- formats ---------------------------------------------------------------

def test_e2m1_table_is_the_published_one():
    got = [float(v) for v in F.e2m1_code_values()]
    want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    assert got == want


@pytest.mark.parametrize("raw,want", [(0x38, 1.0), (0x3C, 1.5), (0x40, 2.0),
                                      (0x00, 0.0), (0xB8, -1.0), (0x30, 0.5)])
def test_e4m3_decode_matches_known_values(raw, want):
    assert float(F.e4m3_to_float(raw)) == pytest.approx(want)


def test_e4m3_subnormal_is_smallest_step():
    # e=0, m=1 -> (1/8) * 2^-6 = 2^-9
    assert float(F.e4m3_to_float(0x01)) == pytest.approx(2 ** -9)


def test_formats_are_registered_and_retrievable():
    names = [f.name for f in F.formats()]
    for expected in ("nvfp4", "mxfp8", "bf16"):
        assert expected in names
        assert F.get_format(expected).name == expected


def test_unknown_format_raises_with_the_list():
    with pytest.raises(KeyError) as e:
        F.get_format("int3")
    assert "nvfp4" in str(e.value)


def test_new_format_can_be_registered_without_new_code():
    fmt = F.register_format(F.Format(name="int3_test", bits=3, block=8))
    assert F.get_format("int3_test") is fmt
    assert fmt.bytes_per_block == pytest.approx(3.0)


# --- plans -----------------------------------------------------------------

def test_plan_every_layer_once_is_identity():
    p = P.Plan.every_layer_once(32)
    assert p.layer_visits() == 32
    assert p.weight_reads_per_token() == 32


def test_looped_plan_matches_inkling_shape():
    p = P.Plan.looped(24, 6, 18, 3)
    d = p.describe()
    assert d["n_layers"] == 24
    assert d["layer_visits"] == 48        # 6 + 12*3 + 6
    assert d["weight_reads"] == 48
    assert set(d["repeats"]) == {str(i) for i in range(6, 18)}


def test_fusing_the_core_halves_weight_reads():
    unfused = P.Plan.looped(24, 6, 18, 3).weight_reads_per_token()
    fused = P.Plan.looped(24, 6, 18, 3, fuse_core=True).weight_reads_per_token()
    assert unfused == 48
    assert fused == 24, "fusion is the largest single lever, so it must halve reads"


def test_plan_rejects_bad_core_range():
    with pytest.raises(ValueError):
        P.Plan.looped(24, 18, 6, 3)
    with pytest.raises(ValueError):
        P.Plan.looped(24, 6, 18, 0)


def test_plan_marks_expert_layers_only_where_asked():
    p = P.Plan.looped(24, 6, 18, 3, expert_layers=range(20))
    assert p.expert_layer_indices() == list(range(20))


# --- reference arithmetic --------------------------------------------------

def _bank(n=8, k=32, fmt=F.NVFP4, seed=0):
    rng = np.random.default_rng(seed)
    if fmt.bits == 4:
        packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    else:
        packed = rng.integers(0, 256, (n, k), dtype=np.uint8)
    nsc = k // fmt.block
    # normals: exponent in 7..12, random mantissa
    exp = rng.integers(7, 13, (n, nsc), dtype=np.uint8) << 3
    man = rng.integers(0, 8, (n, nsc), dtype=np.uint8)
    return packed, (exp | man)


def test_nvfp4_decode_shape_and_dtype():
    packed, scales = _bank()
    w = Q.dequant_block_numpy(packed, scales, F.NVFP4)
    assert w.shape == (8, 32)
    assert w.dtype == np.float32


def test_nibble_order_low_is_even_k():
    """Low nibble is even k. Swapping changes the answer, so the test is real."""
    # E2M1 code 1 = 0.5, code 2 = 1.0. byte 0x21 -> low 1, high 2.
    packed = np.zeros((1, 16), dtype=np.uint8)
    packed[0, 0] = 0x21
    scales = np.full((1, 2), 0x38, dtype=np.uint8)           # scale 1.0
    w = Q.dequant_block_numpy(packed, scales, F.NVFP4)
    assert float(w[0, 0]) == pytest.approx(0.5), "low nibble must be even k"
    assert float(w[0, 1]) == pytest.approx(1.0), "high nibble must be odd k"


def test_scale_is_per_block_not_per_row():
    packed = np.zeros((1, 16), dtype=np.uint8)
    packed[0, :] = 0x22                                       # every nibble = 1.0
    scales = np.zeros((1, 2), dtype=np.uint8)
    scales[0, 0] = 0x38   # 1.0
    scales[0, 1] = 0x40   # 2.0
    w = Q.dequant_block_numpy(packed, scales, F.NVFP4)
    assert np.allclose(w[0, :16], 1.0)
    assert np.allclose(w[0, 16:], 2.0)


def test_global_scale_applies_once():
    packed, scales = _bank()
    plain = Q.dequant_block_numpy(packed, scales, F.NVFP4)
    scaled = Q.dequant_block_numpy(packed, scales, F.NVFP4, outer_scale=0.5)
    assert np.allclose(scaled, plain * 0.5)


def test_gemv_matches_explicit_matrix_product():
    packed, scales = _bank(4, 32)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32).astype(np.float32)
    got = Q.quantized_gemv_numpy(x, packed, scales, F.NVFP4)
    w = Q.dequant_block_numpy(packed, scales, F.NVFP4)
    assert np.allclose(got, w @ x, rtol=1e-5, atol=1e-5)


def test_batch_gemv_decodes_weights_once():
    """The verify-block property: T tokens, one decode, rows agree with solo."""
    packed, scales = _bank(4, 32)
    rng = np.random.default_rng(2)
    xs = rng.standard_normal((4, 32)).astype(np.float32)
    batch = Q.batch_gemv_numpy(xs, packed, scales, F.NVFP4)
    assert batch.shape == (4, 4)
    for t in range(4):
        solo = Q.quantized_gemv_numpy(xs[t], packed, scales, F.NVFP4)
        assert np.allclose(batch[t], solo, rtol=1e-5, atol=1e-5)


def test_mxfp8_path_uses_32_block():
    packed, scales = _bank(2, 64, F.MXFP8)
    w = Q.dequant_block_numpy(packed, scales, F.MXFP8)
    assert w.shape == (2, 64)
    assert F.MXFP8.block == 32


def test_unsupported_bit_width_is_loud():
    odd = F.Format(name="int3", bits=3, block=8)
    with pytest.raises(NotImplementedError):
        Q.dequant_block_numpy(np.zeros((1, 8), np.uint8), np.zeros((1, 1), np.uint8), odd)


# --- dense formats ---------------------------------------------------------

from fikra.quant.quant_ref import dequant_dense_numpy  # noqa: E402


def test_dense_formats_are_registered():
    for name in ("int8_per_tensor", "int8_per_block", "int8_per_channel", "int4_per_channel"):
        assert F.get_format(name).name == name


def test_per_channel_scale_applies_per_row():
    codes = np.arange(8, dtype=np.uint8).reshape(1, 8) + 10
    scales = np.array([0x38], dtype=np.uint8)      # 1.0
    got = dequant_dense_numpy(codes, scales, F.INT8_PER_CHANNEL)
    assert np.allclose(got[0], codes[0].astype(np.float32))


def test_per_tensor_scale_applies_everywhere():
    codes = np.full((2, 8), 3, dtype=np.uint8)
    got = dequant_dense_numpy(codes, np.array([0x40], np.uint8), F.INT8_PER_TENSOR)
    assert np.allclose(got, 2.0 * 3)


def test_per_block_scale_changes_mid_row():
    codes = np.ones((1, 64), dtype=np.uint8)
    scales = np.array([0x38, 0x40], dtype=np.uint8)   # 1.0 then 2.0
    got = dequant_dense_numpy(codes, scales, F.INT8_PER_BLOCK)
    assert np.allclose(got[0, :32], 1.0)
    assert np.allclose(got[0, 32:], 2.0)


def test_dense_and_expert_formats_do_not_collide():
    """mxfp8 is expert-side, int8_per_block is dense-side; same block, different path."""
    assert F.MXFP8.name != F.INT8_PER_BLOCK.name
    assert F.INT8_PER_BLOCK.block == F.MXFP8.block == 32


def test_int4_per_channel_unpacks_two_per_byte():
    """Dense int4 is INTEGER codes, not the E2M1 float codebook.

    Code 1 decoded through the expert-side path would be 0.5; here it is the
    integer 1.0 times the scale. The two 4-bit families must not be confused.
    """
    packed = np.full((1, 8), 0x11, dtype=np.uint8)   # eight nibbles of code 1
    scales = np.array([0x38], dtype=np.uint8)        # 1.0
    got = dequant_dense_numpy(packed, scales, F.INT4_PER_CHANNEL)
    assert got.shape == (1, 16)
    assert np.allclose(got, 1.0)


def test_int4_dense_differs_from_e2m1_expert_decode():
    """A code of 6 is 4.0 as E2M1 and 6.0 as an integer."""
    assert float(F.e2m1_code_values()[6]) == 4.0
    packed = np.full((1, 8), 0x66, dtype=np.uint8)
    scales = np.array([0x38], dtype=np.uint8)
    got = dequant_dense_numpy(packed, scales, F.INT4_PER_CHANNEL)
    assert np.allclose(got, 6.0, atol=1e-6)
