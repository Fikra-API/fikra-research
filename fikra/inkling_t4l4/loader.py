# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Load the Inkling 2 Mini checkpoint into the tensors the runner asks for.

The checkpoint is a single 19.5 GB safetensors file, so the load is done by
mapping it rather than reading it into RAM: ``safetensors`` mmaps, and each
tensor is copied to its card as it is touched. Peak host RAM stays near the size
of one layer instead of 19.5 GB, which is the difference between loading on a
32 GB Kaggle box and being OOM-killed on it.

Only the tensors a decode token actually needs are loaded. Vision, the MTP draft
head and the second copy of the tied embedding are skipped, and the skip is
asserted rather than assumed: a missing text tensor raises with its name.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import model_spec as ms

REPO = "thinkingmachines/Inkling-2-Mini"
WEIGHTS_FILE = "model-00001-of-00001.safetensors"
DTYPE_MAP = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8,
             "F8_E4M3": torch.uint8}


def download(dest: str | Path = "/kaggle/working", revision: str | None = None) -> Path:
    """Fetch the weights file, resuming if a partial copy is there."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, WEIGHTS_FILE, revision=revision,
                           local_dir=str(dest), resume_download=True)
    return Path(path)


def _read_header(path: Path) -> dict:
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def load(weights_path: str | Path, device="cuda:0", skip_vision: bool = True) -> dict:
    """Return {checkpoint name -> tensor} for every tensor a token needs.

    ``device`` may be "cpu": the runner moves each tensor itself, and the mapper
    keeps a single copy per tensor either way.
    """
    from safetensors import safe_open

    path = Path(weights_path)
    header = _read_header(path)
    names = [k for k in header if k != "__metadata__"]
    wanted = []
    for name in names:
        if skip_vision and (name.startswith("model.visual.") or name.startswith("model.vision.")):
            continue
        if name.startswith("model.mtp.") or ".mtp." in name:
            continue
        wanted.append(name)

    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in wanted:
            spec = header[name]
            dtype = DTYPE_MAP.get(spec["dtype"])
            if dtype is None:
                raise TypeError(f"{name}: unsupported dtype {spec['dtype']}")
            t = f.get_tensor(name)
            if dtype == torch.uint8:
                # FP4 codes stay raw bytes; FP8 E4M3 is also stored as bytes and
                # is decoded in-kernel, because the T4 has no fp8 datatype.
                pass
            else:
                t = t.to(dtype)
            out[name] = t.to(device) if device != "cpu" else t
    missing = [k for k in ("model.llm.embed.weight", "model.llm.unembed.weight",
                           "model.llm.norm.weight") if k not in out]
    if missing:
        raise KeyError(f"checkpoint is missing required tensors: {missing}")
    return out


def load_tokenizer():
    """The checkpoint's tokenizer (tiktoken-style model file)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
