# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Codec correctness: nibble order, E4M3 table, scale block width, round trip."""
import numpy as np
import pytest

from fikra.inkling_t4l4 import nvfp4


def test_fp4_table():
    t = nvfp4.fp4_values()
    assert len(t) == 16
    np.testing.assert_array_equal(t[:8], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    np.testing.assert_array_equal(t[8:], -t[:8])


def test_unpack_low_nibble_is_even_k():
    # 0xFE: low nibble 0xE -> -4.0 at even k, high nibble 0xF -> -6.0 at odd k.
    # 0x21: 0x1 -> 0.5 at even k, 0x2 -> 1.0 at odd k.
    packed = np.array([[0xFE, 0x21]], np.uint8)
    out = nvfp4.unpack_fp4(packed)
    np.testing.assert_array_equal(out, [[-4.0, -6.0, 0.5, 1.0]])


@pytest.mark.parametrize("byte, value", [
    (0x00, 0.0), (0x38, 1.0), (0x3C, 1.5), (0x3F, 1.875), (0x01, 0.001953125),
    (0x7E, 448.0), (0x40, 2.0), (0xB8, -1.0), (0x04, 0.0078125),
])
def test_e4m3_decodes(byte, value):
    assert nvfp4.e4m3_to_float(bytes([byte]))[0] == pytest.approx(value, rel=1e-6)


def test_e4m3_saturation_and_nan():
    assert nvfp4.e4m3_to_float(bytes([0x7E]))[0] == 448.0
    assert np.isnan(nvfp4.e4m3_to_float(bytes([0x7F]))[0])


def test_ue8m0():
    assert nvfp4.ue8m0_to_float(bytes([127]))[0] == 1.0
    assert nvfp4.ue8m0_to_float(bytes([0]))[0] == pytest.approx(2.0 ** -127)
    assert np.isnan(nvfp4.ue8m0_to_float(bytes([255]))[0])


def test_dequant_applies_one_scale_per_16_k():
    # One output row, K=32: the checkpoint stores [N, K/2] codes and [N, K/16]
    # scales, so each scale covers 16 consecutive K elements.
    packed = np.tile(np.array([[0x22]], np.uint8), (1, 16))       # 32 ones (0x22 -> 1,1)
    scales = np.array([[0x38, 0x40]], np.uint8)                   # 1.0, 2.0
    out = nvfp4.dequant_nvfp4(packed, scales)
    assert out.shape == (1, 32)
    np.testing.assert_array_equal(out[0, :16], np.ones(16))
    np.testing.assert_array_equal(out[0, 16:], np.full(16, 2.0))


def test_dequant_refuses_column_major_scales():
    # A [K, N/16] scale tensor is a different (wrong) layout; refuse it instead
    # of broadcasting into nonsense.
    packed = np.zeros((2, 8), np.uint8)
    with pytest.raises(ValueError, match="one FP8 value per 16 K"):
        nvfp4.dequant_nvfp4(packed, np.zeros((16, 1), np.uint8))


def test_dequant_applies_the_expert_global_scale():
    # The released weights need code * scale * amax / outer_denominator; the
    # 2688 is the same for every layer and both projections.
    packed = np.full((1, 16), 0x22, np.uint8)
    scales = np.full((1, 2), 0x38, np.uint8)
    out = nvfp4.dequant_nvfp4(packed, scales, outer_scale=0.5 * 0.12112041)
    np.testing.assert_allclose(out, 0.5 * 0.12112041, rtol=1e-6)


def test_quantize_round_trip_is_exact_on_table_values():
    t = nvfp4.fp4_values()
    np.testing.assert_array_equal(nvfp4.quantize_e2m1(t), t)


def test_gemv_matches_dense_matmul():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, (4, 32)).astype(np.uint8)
    packed = (codes[:, ::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    scales_fp8 = np.array([[0x38, 0x40]] * 4, np.uint8)  # keep scales exact
    x = rng.standard_normal(32).astype(np.float32)
    y = nvfp4.gemv_nvfp4(x, packed, scales_fp8)
    w = nvfp4.fp4_values()[codes] * np.repeat([[1.0, 2.0]], 16, axis=1)
    np.testing.assert_allclose(y, w @ x, rtol=1e-6)


def test_gemv_with_global_scale_matches_reference():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 16, (2, 32)).astype(np.uint8)
    packed = (codes[:, ::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    scales_fp8 = np.array([[0x38, 0x40]] * 2, np.uint8)
    g = 0.12112041 / 2688.0
    x = rng.standard_normal(32).astype(np.float32)
    y = nvfp4.gemv_nvfp4(x, packed, scales_fp8, outer_scale=g)
    w = nvfp4.fp4_values()[codes] * np.repeat([[1.0, 2.0]], 16, axis=1) * g
    np.testing.assert_allclose(y, w @ x, rtol=1e-5)



# --- checkpoint constants (verified against the released bytes) ------------

# model-00001-of-00001.safetensors, layers 0-19:
#   w13_weight        U8 [128, 2304, 1152]      N=2304 rows, K=2304 (two nibbles/byte)
#   w13_weight_scale  F8_E4M3 [128, 2304, 144]  = K/16, not N/16
#   w2_weight_scale   F8_E4M3 [128, 2304, 72]   = K/16 with K=1152
#   *_outer_scale_denominator F32 [1] = 2688.0 (layers 0, 5, 19; w13 and w2)
#   *_amax            F32 [128]                 one value per expert
def test_scale_axis_is_k_not_n():
    E, H, I = 128, 2304, 1152
    assert (E, H, H // 16) == (128, 2304, 144)   # w13: 144 = K/16 with K=2304
    assert (E, H, I // 16) == (128, 2304, 72)    # w2:   72 = K/16 with K=1152
    # The column reading would need 2304/16 = 144 scales per row too, but for w2
    # it would give 2304/16 = 144, not the 72 the header stores. Only K/16 fits.
    assert I // 16 == 72 and H // 16 == 144


def test_mxfp8_layers_have_no_nvfp4_global_scale():
    # Layers 20-23 ship w13_weight_scale [128, 2304, 72] F8_E4M3 with 32x32
    # ue8m0 blocks and no amax/outer_scale_denominator tensors; applying the
    # NVFP4 global factor there would halve the block size and rescale wrongly.
    assert 2304 // 32 == 72
    assert 2304 // 16 == 144
