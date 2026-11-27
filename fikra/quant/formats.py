"""Weight-block formats, described as data so any model can declare its own.

A format answers three questions and nothing else:

1. how many K elements share one scale  (block size along the contraction axis)
2. how many bits one weight occupies     (packing density)
3. how a block's bytes become floats     (the decode)

Keeping the decode here rather than in a kernel is what makes the kernels
model-agnostic: a new checkpoint with a new packing needs a new Format, not a
new kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# E2M1 is the 4-bit float used by NVFP4: 1 sign, 2 exponent, 1 mantissa.
# The sixteen codes in order are 0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives.
_E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def e2m1_code_values() -> np.ndarray:
    """The 16 E2M1 values, indexable by the raw 4-bit code."""
    return _E2M1.copy()


def e4m3_to_float(raw: np.ndarray | int) -> np.ndarray:
    """Decode E4M3 bytes to float32 by bit arithmetic, no table.

    Works on any host: no dependency on a GPU fp8 dtype, which is the reason
    this exists (T4 has none).
    """
    b = np.asarray(raw, dtype=np.uint8).astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 0x0F
    man = b & 0x07
    normal = (1.0 + man * 0.125) * np.exp2(exp - 7.0)
    subnormal = (man * 0.125) * np.exp2(-6.0)
    return (sign * np.where(exp == 0, subnormal, normal)).astype(np.float32)


@dataclass(frozen=True)
class Format:
    """How one weight matrix is packed.

    ``block`` is the number of contraction-axis elements per scale, so a tile
    should be a multiple of it for the scale load to be reusable, which is the
    single largest kernel win measured so far.
    """

    name: str
    bits: int
    block: int
    has_global_scale: bool = False
    notes: str = ""

    @property
    def bytes_per_block(self) -> float:
        return self.bits * self.block / 8.0

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


NVFP4 = Format(
    name="nvfp4", bits=4, block=16, has_global_scale=True,
    notes="E2M1 codes, one fp8-e4m3 scale per 16 K, plus a per-tensor amax "
          "divisor. Used by Inkling experts on layers 0-19.",
)

MXFP8 = Format(
    name="mxfp8", bits=8, block=32, has_global_scale=False,
    notes="e4m3 codes with one ue8m0 scale per 32x32 block. Used by Inkling "
          "experts on layers 20-23.",
)

FP8_E4M3_BLOCK = Format(
    name="fp8_e4m3_block", bits=8, block=32, has_global_scale=False,
    notes="e4m3 codes with a per-block scale; the shape many newer MoE "
          "checkpoints ship.",
)

BF16 = Format(
    name="bf16", bits=16, block=1, has_global_scale=False,
    notes="unquantised reference path, so a model can mix quantised experts "
          "with dense bf16 weights.",
)

_REGISTRY: dict[str, Format] = {f.name: f for f in (NVFP4, MXFP8, FP8_E4M3_BLOCK, BF16)}


def register_format(fmt: Format) -> Format:
    """Add a format at runtime, so a new checkpoint needs no kernel change."""
    _REGISTRY[fmt.name] = fmt
    return fmt


def get_format(name: str) -> Format:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown format {name!r}; known: {', '.join(sorted(_REGISTRY))}") from None


def formats() -> list[Format]:
    return [ _REGISTRY[k] for k in sorted(_REGISTRY)]
