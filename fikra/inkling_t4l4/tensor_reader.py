# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Header-only reader for an HF safetensors file.

We never download the 19.5 GB checkpoint just to learn its layout: the
safetensors header is a JSON blob at the front of the file, and HTTP range
requests are enough. ``read_tensor`` also serves tensor bodies by range, so
analysis (amax histograms, routing probes) works from a laptop.
"""
from __future__ import annotations

import json
import struct
import urllib.request
from dataclasses import dataclass
from typing import BinaryIO

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def expected_nbytes(self) -> int | None:
        unit = DTYPE_BYTES.get(self.dtype)
        return None if unit is None else unit * self.numel


def parse_header(raw: bytes) -> tuple[dict[str, TensorInfo], dict]:
    """Parse a safetensors header (the JSON bytes, no length prefix)."""
    header = json.loads(raw.decode("utf-8"))
    meta = header.pop("__metadata__", {}) or {}
    tensors = {}
    for name, spec in header.items():
        begin, end = spec["data_offsets"]
        tensors[name] = TensorInfo(name, spec["dtype"], tuple(spec["shape"]), begin, end)
    return tensors, meta


def read_header(stream: BinaryIO) -> tuple[dict[str, TensorInfo], dict]:
    stream.seek(0)
    n = struct.unpack("<Q", stream.read(8))[0]
    if n <= 0 or n > 200_000_000:
        raise ValueError(f"implausible safetensors header length {n}")
    return parse_header(stream.read(n))


def read_tensor(stream: BinaryIO, info: TensorInfo, header_bytes: int) -> bytes:
    stream.seek(8 + header_bytes + info.begin)
    data = stream.read(info.nbytes)
    if len(data) != info.nbytes:
        raise ValueError(f"short read for {info.name}: {len(data)} != {info.nbytes}")
    return data


def http_url(url: str) -> str:
    """A stable, range-capable URL for a HF file (follows the CDN redirect)."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.length is not None and resp.length < 4096 and resp.url != url:
            redirect = resp.headers.get("Location")
            if redirect:
                return redirect
    return url


class RangeReader:
    """Seekable file-like object over an HTTP(S) URL using range requests."""

    def __init__(self, url: str, block: int = 8 << 20):
        self.url = url
        self.block = block
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])
        self._start = 0
        self._buf = b""

    def seek(self, pos: int) -> int:
        self.pos = pos
        return pos

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        out = bytearray()
        while len(out) < n:
            if not (self._start <= self.pos < self._start + len(self._buf)):
                self._fill(self.pos, n - len(out))
            off = self.pos - self._start
            chunk = self._buf[off:off + (n - len(out))]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def _fill(self, pos: int, need: int) -> None:
        if pos >= self.size:
            raise EOFError(pos)
        end = min(self.size, pos + max(need, self.block))
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={pos}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            self._buf = resp.read()
        self._start = pos
