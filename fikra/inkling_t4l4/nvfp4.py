# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""NVFP4 / MXFP8 reference codecs and a CPU GEMV, for correctness only.

Expert weights in ``model-00001.safetensors`` are packed two FP4 codes per
byte along the contraction axis (``uint8 [E, N, K/2]``) with one FP8 E4M3
scale per 16 K elements per output row (``[E, N, K/16]``). Each expert also
carries ``amax`` (F32, one value per expert) and
``outer_scale_denominator`` (F32, 2688 for every layer we checked). The stored
value of a weight is ``code * scale * amax / outer_scale_denominator``.

Layers 20-23 use MXFP8: ``float8_e4m3fn`` codes with one ue8m0 scale per 32x32
block. All of this is read off the released header and verified against the
bytes (see ``research/nvfp4_probe.py``), not assumed from the format name.
"""
from __future__ import annotations

from functools import lru_cache

try:  # numpy is a test-time dependency only; the kernels themselves use torch
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _require_numpy():
    if np is None:  # pragma: no cover
        raise RuntimeError("the reference codec needs numpy (pip install numpy)")
    return np


@lru_cache(maxsize=1)
def fp4_values():
    """E2M1 magnitudes for codes 0..7; sign lives in bit 3 of each nibble."""
    n = _require_numpy()
    lut = n.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], n.float32)
    return n.concatenate([lut, -lut])


def e4m3_to_float(raw) -> "np.ndarray":
    """Decode FP8 E4M3FN bytes (no infinities; 0x7f/0xff are NaN) to float32."""
    n = _require_numpy()
    b = n.frombuffer(raw, n.uint8) if isinstance(raw, (bytes, bytearray)) else n.asarray(raw, n.uint8)
    b = b.astype(n.int32)
    sign = n.where(b & 0x80, -1.0, 1.0).astype(n.float32)
    mag = b & 0x7F
    exp = (mag >> 3) & 0xF
    man = mag & 0x7
    sub = (man.astype(n.float32) / 8.0) * n.exp2(-6.0)
    norm = (1.0 + man.astype(n.float32) / 8.0) * n.exp2((exp - 7).astype(n.float32))
    val = n.where(exp == 0, sub, norm)
    val = n.where(mag == 0x7F, n.nan, val)
    return (sign * val).astype(n.float32)


def ue8m0_to_float(raw) -> "np.ndarray":
    """Decode an MXFP8 block exponent: 2**(x-127), 0xff means NaN."""
    n = _require_numpy()
    b = n.frombuffer(raw, n.uint8) if isinstance(raw, (bytes, bytearray)) else n.asarray(raw, n.uint8)
    b = b.astype(n.int32)
    val = n.exp2((b - 127).astype(n.float32))
    return n.where(b == 0xFF, n.nan, val).astype(n.float32)


def unpack_fp4(packed) -> "np.ndarray":
    """Unpack a uint8 [.., N, K/2] tile to float32 [.., N, K].

    Two codes share a byte along the contraction axis: the low nibble is the
    element at even K, the high nibble at odd K.
    """
    n = _require_numpy()
    lut = fp4_values()
    packed = n.asarray(packed, n.uint8)
    lo = lut[packed & 0x0F]
    hi = lut[packed >> 4]
    out = n.stack([lo, hi], axis=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequant_nvfp4(packed, scales, block: int = 16, outer_scale: float | None = None) -> "np.ndarray":
    """[N, K/2] uint8 + [N, K/16] FP8 scales -> float32 [N, K].

    ``outer_scale`` is the expert's ``amax / outer_scale_denominator``; the
    released checkpoint needs it to reproduce the BF16 weights, so leaving it
    out silently scales every expert by a different factor.
    """
    n = _require_numpy()
    vals = unpack_fp4(packed)
    s = e4m3_to_float(n.asarray(scales, n.uint8)).astype(n.float32)
    rep = n.repeat(s, block, axis=-1)
    if rep.shape != vals.shape:
        raise ValueError(f"scale expansion {rep.shape} does not match codes {vals.shape}: "
                         f"scales must be one FP8 value per {block} K elements")
    out = vals * rep
    if outer_scale is not None:
        out = out * outer_scale
    return out.astype(n.float32)


def dequant_mxfp8(codes, scales, block: int = 32) -> "np.ndarray":
    """[K, N] FP8 E4M3 + [ceil(K/32), ceil(N/32)] ue8m0 -> float32 [K, N]."""
    n = _require_numpy()
    vals = e4m3_to_float(n.asarray(codes, n.uint8))
    k, cols = vals.shape
    s = ue8m0_to_float(n.asarray(scales, n.uint8))
    rep = n.repeat(n.repeat(s, block, axis=0), block, axis=1)[:k, :cols]
    return (vals * rep).astype(n.float32)


def gemv_nvfp4(x, packed, scales, block: int = 16, outer_scale: float | None = None) -> "np.ndarray":
    """y = W x for W stored as [N, K/2] uint8 with [N, K/16] scales."""
    n = _require_numpy()
    w = dequant_nvfp4(packed, scales, block, outer_scale)  # [N, K]
    return w @ n.asarray(x, n.float32)


def quantize_e2m1(x) -> "np.ndarray":
    """Round to nearest E2M1 magnitudes (ties to even index), for tests."""
    n = _require_numpy()
    lut = fp4_values()[:8]
    a = n.abs(n.asarray(x, n.float32))
    idx = n.searchsorted(lut, a, side="left")
    idx = n.clip(idx, 1, 7)
    lo = lut[idx - 1]
    hi = lut[idx]
    closer_low = (a - lo) < (hi - a)
    idx = n.where(closer_low, idx - 1, idx)
    return n.sign(x) * lut[idx]
