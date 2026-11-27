"""Reference decode path in numpy, for any Format.

This is the oracle every GPU kernel is checked against, and it is deliberately
written once, generically, so it is not duplicated per model. A kernel that
disagrees with this is wrong; this is the definition of the arithmetic.

Two operations:

  dequant_block_numpy - one packed block to floats
  quantized_gemv_numpy - y = W x over a packed bank, one expert at a time

Both take a Format, so NVFP4 Inkling experts, MXFP8 Inkling coda layers and a
hypothetical new 8-bit MoE all use the same code path.
"""
from __future__ import annotations

import numpy as np

from .formats import Format, NVFP4, MXFP8, BF16, e2m1_code_values, e4m3_to_float


def unpack_nvfp4_codes(packed: np.ndarray) -> np.ndarray:
    """[..., K/2] uint8 -> [..., K] int codes. Low nibble is the even k."""
    p = np.asarray(packed, dtype=np.uint8)
    lo = (p & 0x0F).astype(np.int16)
    hi = ((p >> 4) & 0x0F).astype(np.int16)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.int16)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _decode_e2m1(codes: np.ndarray) -> np.ndarray:
    return e2m1_code_values()[np.asarray(codes, dtype=np.int16) & 0x0F]


def _e4m3_signed_scale_or_int8(scales: np.ndarray, signed: bool) -> np.ndarray:
    """Interpret scale bytes as e4m3 (default) or as signed int8 fractions.

    Dense checkpoints in the wild use both; the format, not the kernel, decides.
    """
    if signed:
        b = np.asarray(scales, dtype=np.int8).astype(np.float32)
        return b / 127.0
    return e4m3_to_float(scales)


def dequant_dense_numpy(
    packed: np.ndarray,
    scales: np.ndarray,
    fmt: Format,
    signed_codes: bool = False,
    signed_scales: bool = False,
) -> np.ndarray:
    """Decode a DENSE weight matrix for the int8/int4 per-channel families.

    ``packed`` [N, K] uint8 codes (or [N, K/2] for 4-bit)
    ``scales`` [N] for per-channel, [K/block] for per-block, [1] for per-tensor
    Returns [N, K] float32

    Kept separate from the expert path because the scale layout is genuinely
    different: experts scale along K, dense weights scale per output channel.
    """
    p = np.asarray(packed)

    if fmt.bits == 4:
        codes = unpack_nvfp4_codes(p).astype(np.int16)
        vals = np.where(codes & 8, codes - 16, codes).astype(np.float32) if signed_codes \
            else (codes & 0x0F).astype(np.float32)
    elif fmt.bits == 8:
        vals = p.astype(np.int8).astype(np.float32) if signed_codes else p.astype(np.float32)
    else:
        raise NotImplementedError(f"dequant_dense_numpy does not handle {fmt.bits}-bit")

    s = _e4m3_signed_scale_or_int8(scales, signed_scales).reshape(-1)

    if fmt.block == -1:                      # per output channel
        out = vals * s[:, None]
    elif fmt.block == 0:                     # per tensor
        out = vals * float(s[0])
    else:                                    # per block along K
        rep = np.repeat(s, fmt.block)[: vals.shape[-1]]
        out = vals * rep[None, :]
    return out.astype(np.float32)


def dequant_block_numpy(
    packed: np.ndarray,
    scales: np.ndarray,
    fmt: Format = NVFP4,
    outer_scale: float | None = None,
) -> np.ndarray:
    """One row or bank row of packed weights to float32.

    ``packed``  [..., K/2] (4-bit) or [..., K] (8-bit) raw codes
    ``scales``  [..., K/block] raw scale bytes
    Returns     [..., K] float32

    NVFP4 additionally carries a per-expert ``amax / outer_scale_denominator``;
    omitting it leaves each expert off by its own factor, which looks like a
    routing bug rather than a quantisation one.
    """
    packed = np.asarray(packed)
    scales = np.asarray(scales)

    if fmt.name == BF16:
        return packed.astype(np.float32)

    if fmt.block in (-1, 0) or fmt.bits in (4, 8) and fmt.block in (-1, 0):
        # per-channel / per-tensor families belong on the dense path
        return dequant_dense_numpy(packed, scales, fmt)

    if fmt.bits == 4:
        codes = unpack_nvfp4_codes(packed)
        vals = _decode_e2m1(codes)
    elif fmt.bits == 8:
        vals = e4m3_to_float(packed)
        codes = None
    else:
        raise NotImplementedError(f"reference decode for {fmt.bits}-bit is not implemented")

    s = e4m3_to_float(scales)
    rep = np.repeat(s, fmt.block, axis=-1)
    if rep.shape != vals.shape:
        # tolerate a scale row that is one block long at the tail
        k = min(rep.shape[-1], vals.shape[-1])
        rep = rep[..., :k]
        vals = vals[..., :k]
    out = vals * rep
    if fmt.has_global_scale and outer_scale is not None:
        out = out * float(outer_scale)
    del codes
    return out.astype(np.float32)


def quantized_gemv_numpy(
    x: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    fmt: Format = NVFP4,
    outer_scale: float | None = None,
) -> np.ndarray:
    """y = W x for one packed matrix. Shapes: W is [N, K] after decode."""
    w = dequant_block_numpy(packed, scales, fmt, outer_scale)
    return (w @ np.asarray(x, dtype=np.float32)).astype(np.float32)


def batch_gemv_numpy(xs: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Several token vectors against one weight read: the verify-block case.

    xs is [T, K]; the result is [T, N]. The weight is decoded once, which is the
    whole point of the batched path.
    """
    w = dequant_block_numpy(*args, **kwargs)
    return (np.asarray(xs, dtype=np.float32) @ w.T).astype(np.float32)
