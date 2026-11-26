# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Inkling 2 Mini on 16 GB-class GPUs (T4 sm75, L4 sm89).

The shipped checkpoint is NVFP4 for layers 0-19 and MXFP8 for layers 20-23;
neither T4 nor L4 has NVFP4 block scaling in hardware, so the expert GEMMs
unpack the formats in software. See ``model_spec`` for the facts we rely on.
"""
from . import model_spec

__all__ = ["model_spec"]
