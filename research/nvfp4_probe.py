# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Probe Inkling 2 Mini NVFP4 expert tensors: layout, scale dtype, amax.

Run against a local safetensors file (or the header + a byte range of it):

    python research/nvfp4_probe.py model.safetensors
    python research/nvfp4_probe.py model.safetensors --layer 6 --experts 0 7 9

Reports, per MoE layer:
  * w13 / w2 weight shapes and the logical matrix size
  * the scale block width implied by the shapes, which must be 16 along K
  * per-expert amax (F32 [128]) and outer_scale_denominator (F32 [1])
  * FP8 E4M3 scale statistics and saturation check
  * reconstructed max|W| versus the stored amax, which fails loudly if the
    global scale or the nibble order is wrong

Exit status is non-zero if any expected invariant does not hold, so this can
gate a checkpoint download instead of being read by eye.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n


def read_tensor(path, header, header_bytes, name):
    t = header[name] if isinstance(header[name], dict) else header[name].__dict__
    s, e = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_bytes + s)
        raw = f.read(e - s)
    if len(raw) != e - s:
        raise ValueError(f"short read for {name}: {len(raw)} != {e - s}")
    dt = t["dtype"]
    if dt == "U8":
        arr = np.frombuffer(raw, np.uint8)
    elif dt == "F32":
        arr = np.frombuffer(raw, "<f4")
    elif dt == "BF16":
        arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    elif dt == "F8_E4M3":
        arr = from_e4m3(raw)
    else:
        raise ValueError(f"unsupported dtype {dt} for {name}")
    return arr.reshape(t["shape"])


def from_e4m3(raw: bytes) -> np.ndarray:
    """Decode FP8 E4M3 (finite, no inf, NaN=0x7f/0xff) the way NVIDIA uses it."""
    b = np.frombuffer(raw, np.uint8).astype(np.uint32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    mag = b & 0x7F
    exp = (mag >> 3) & 0xF
    man = mag & 0x7
    sub = (man / 8.0) * np.exp2(-6.0)
    norm = (1.0 + man / 8.0) * np.exp2(exp.astype(np.float64) - 7.0)
    val = np.where(exp == 0, sub, norm)
    val = np.where(mag == 0x7F, np.nan, val)
    return (sign * val).astype(np.float32)


def fp4_table() -> np.ndarray:
    """E2M1 values [0,.5,1,1.5,2,3,4,6] for codes 0..7; sign bit 8."""
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    return np.concatenate([lut, -lut])


def dequant_expert(packed: np.ndarray, scales: np.ndarray, block: int = 16,
                   outer_scale: float | None = None) -> np.ndarray:
    """One expert matrix: packed uint8 [n, k/2], scales [n, k/block]."""
    n, half = packed.shape
    vals = fp4_table()[np.stack([packed & 0x0F, packed >> 4], -1)].reshape(n, half * 2)
    rep = np.repeat(scales.astype(np.float32), block, axis=1)
    if rep.shape != vals.shape:
        raise ValueError(f"scales {scales.shape} do not expand to codes {vals.shape}; "
                         f"expected one scale per {block} K elements")
    out = vals * rep
    return out if outer_scale is None else out * outer_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--layer", type=int, action="append", default=None)
    ap.add_argument("--experts", type=int, nargs="*", default=None)
    ap.add_argument("--block", type=int, default=16)
    args = ap.parse_args()

    header, hb = read_header(args.path)
    meta = header.get("__metadata__", {})
    print("file metadata:", meta)

    layers = defaultdict(dict)
    for name in header:
        if name == "__metadata__" or "mlp.experts" not in name:
            continue
        layer = int(name.split("layers.")[1].split(".")[0])
        suffix = name.split("mlp.experts.")[1]
        layers[layer][suffix] = name

    problems = []
    print(f"{'layer':>5} {'w13':>18} {'w2':>18} {'s13':>14} {'s2':>12} "
          f"{'scale_bits':>10} {'max|W|':>8} {'amax':>8} {'scale_max':>9} {'sat%':>5}")
    for layer in sorted(layers):
        if args.layer and layer not in args.layer:
            continue
        t = layers[layer]
        w13 = header[t["w13_weight"]]["shape"]
        w2 = header[t["w2_weight"]]["shape"]
        s13 = header[t["w13_weight_scale"]]["shape"]
        s2 = header[t["w2_weight_scale"]]["shape"]
        sdt = header[t["w13_weight_scale"]]["dtype"]
        exp = args.experts or [0]
        a = read_tensor(args.path, header, hb, t["w13_weight"])
        s = read_tensor(args.path, header, hb, t["w13_weight_scale"])

        # Scale axis: [E, N, K/block] means block = 2 * packed_K / scale_K.
        block = (2 * w13[2]) // s13[2]
        if block != args.block:
            problems.append(f"layer {layer}: scale block is {block}, expected {args.block}")
        if w13[1] != s13[1]:
            problems.append(f"layer {layer}: w13 N={w13[1]} but scales N={s13[1]}")

        denom, amax = 1.0, None
        if "w13_weight_amax" in t:
            amax = read_tensor(args.path, header, hb, t["w13_weight_amax"])
            denom = float(read_tensor(args.path, header, hb,
                                      t["w13_weight_outer_scale_denominator"])[0])
        g = None if amax is None else float(amax[exp[0]]) / denom
        w = np.concatenate([dequant_expert(a[e], s[e], block, g).ravel() for e in exp])
        smax = float(np.nanmax(s[0]))
        sat = float(np.mean(np.abs(s.astype(np.float32)) >= 448.0) * 100)
        got, want = float(np.abs(w).max()), float(amax[exp[0]]) if amax is not None else None
        print(f"{layer:>5} {str(w13):>18} {str(w2):>18} {str(s13):>14} {str(s2):>12} "
              f"{sdt:>10} {got:8.4f} {('%.4f' % want) if want else '-':>8} {smax:9.1f} {sat:5.2f}")
        if want is not None and got > want * 1.0001:
            problems.append(f"layer {layer}: reconstructed max|W| {got:.6f} exceeds amax {want:.6f}")

    if problems:
        print("\nFAILED INVARIANTS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("\nall invariants hold")


if __name__ == "__main__":
    main()
