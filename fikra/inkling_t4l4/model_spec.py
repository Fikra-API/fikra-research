# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Inkling 2 Mini shapes and quant facts, as read from the released checkpoint.

Sources (all public, 2026-11-24 revision c9d02830):
  * config.json        -> attention / MoE / loop layout
  * hf_quant_config.json -> which tensors are NVFP4, MXFP8, BF16, FP32
  * safetensors header -> every tensor's dtype, shape and byte offset
  * model card + vLLM recipe -> prelude 0-5, core 6-17 x3, coda 18-23

This module is the single place the rest of the package learns the model from,
so a shape change shows up as a failing test rather than a wrong kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HIDDEN = 2304
NUM_LAYERS = 24
NUM_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 1152
NUM_EXPERTS = 128
TOP_K = 8
VOCAB = 201024
SLIDING_WINDOW = 1024
LOCAL_LAYER_IDS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
CORE_START, CORE_END, NUM_PASSES = 6, 18, 3
GATE_RANK = 64
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
NVFP4_LAYERS = tuple(range(0, 20))   # layers 0-19: experts NVFP4
MXFP8_LAYERS = tuple(range(20, 24))  # layers 20-23: experts MXFP8
# MTP (mtp.safetensors, 635,801,416 B, a separate file next to the main weights)
MTP_LAYERS = 3
MTP_BLOCK = 8
MTP_MARKOV_RANK = 256
MTP_INTERMEDIATE = 9216
MTP_TARGET_LAYER_IDS = (5, 23, 41, 48)
MTP_LOCAL_LAYER_IDS = (0, 2)
VISION_LAYERS = 23          # config says 23; the tensors are under model.visual.*
VISION_HIDDEN = 1024

# Byte budgets summed from the safetensors header of model-00001
# (19,525,870,752 B total; the sum below matches it exactly).
BYTES = {
    "experts_nvfp4": 11_466_199_200,   # layers 0-19, weights + scales + amax
    "experts_mxfp8": 4_204_265_472,    # layers 20-23
    "dense_bf16": 1_366_252_032,       # attention, norms, router, controllers
    "embed": 1_852_641_792,            # token embed + unembed
    "vision": 636_512_256,             # model.visual.*, skipped in Phase 1
}
BYTES_ATTENTION_BF16 = BYTES["dense_bf16"]
BYTES_EMBED_UNEMBED = BYTES["embed"]
BYTES_VISION = BYTES["vision"]
BYTES_WEIGHTS_TOTAL = 19_525_870_752
# Every text weight a decode token touches, minus the second copy of the tied
# embedding and minus vision: 19.53 GB - 0.64 GB vision - 0.93 GB unembed.
WEIGHTS_READ_PER_DECODE_TOKEN = (
    BYTES["experts_nvfp4"] + BYTES["experts_mxfp8"] + BYTES["dense_bf16"]
    + BYTES["embed"] // 2)

# Hardware ceilings used by the planner (GB/s, SM count, shared memory per SM).
DEVICES = {
    "T4": {"sm": 75, "vram_gb": 16.0, "bw_gbps": 320.0, "smem_kb": 64, "fp8": False, "bf16": True},
    "P100": {"sm": 60, "vram_gb": 16.0, "bw_gbps": 732.0, "smem_kb": 64, "fp8": False, "bf16": False},
    "L4": {"sm": 89, "vram_gb": 24.0, "bw_gbps": 300.0, "smem_kb": 100, "fp8": True, "bf16": True},
}


@dataclass(frozen=True)
class ExpertTier:
    layers: tuple[int, ...]
    storage: str
    weight_bytes_per_expert: int  # w13 + w2, packed
    scale_bytes_per_expert: int

    @property
    def weights_per_expert(self) -> int:
        """Logical elements per expert: w13 (2I x H) + w2 (H x I)."""
        return 2 * HIDDEN * INTERMEDIATE + HIDDEN * INTERMEDIATE

    @property
    def bits_per_weight(self) -> float:
        """Stored bits per logical weight, scales included.

        NVFP4 measured from the header is 4.375: 4 payload bits plus one FP8
        E4M3 scale per 16 K elements, i.e. 4 + 8/16. MXFP8 is 8 + 8/32 = 8.25.
        """
        return 8.0 * (self.weight_bytes_per_expert + self.scale_bytes_per_expert) / self.weights_per_expert


def tiers() -> tuple[ExpertTier, ...]:
    # NVFP4: w13 U8 [128, 2304, 1152] and w2 U8 [128, 2304, 576] per expert,
    # i.e. one byte per two K elements. Scales are one FP8 per 16 K per row:
    # w13 [128, 2304, 144], w2 [128, 2304, 72].
    w13_packed = HIDDEN * (2 * INTERMEDIATE) // 2
    w2_packed = HIDDEN * INTERMEDIATE // 2
    nv_scales = HIDDEN * (2 * INTERMEDIATE) // NVFP4_BLOCK + HIDDEN * INTERMEDIATE // NVFP4_BLOCK
    # MXFP8: F8 [128, 2304, 2304] and [128, 2304, 1152], one ue8m0 byte per 32 K.
    mx_scales = HIDDEN * (2 * INTERMEDIATE) // MXFP8_BLOCK + HIDDEN * INTERMEDIATE // MXFP8_BLOCK
    return (
        ExpertTier(NVFP4_LAYERS, "nvfp4", w13_packed + w2_packed, nv_scales),
        ExpertTier(MXFP8_LAYERS, "mxfp8", HIDDEN * (2 * INTERMEDIATE) + HIDDEN * INTERMEDIATE, mx_scales),
    )


@dataclass(frozen=True)
class LayerVisit:
    """One application of one layer in one loop pass."""
    layer: int
    loop_pass: int  # 0, 1, 2 for core layers; always 0 otherwise

    @property
    def cache_index(self) -> int:
        """Slot in the KV + conv-state cache for this visit.

        Each core layer attends over its own KV, so the three passes need three
        slots even though the weights are shared: 12 core layers x 3 passes = 36,
        plus 12 non-core layers = 48 slots for 48 visits.
        """
        if CORE_START <= self.layer < CORE_END:
            return NUM_LAYERS + (self.layer - CORE_START) * NUM_PASSES + self.loop_pass
        return self.layer


def schedule() -> list[LayerVisit]:
    """The 48 layer visits a token makes, in order."""
    out = [LayerVisit(i, 0) for i in range(CORE_START)]
    out += [LayerVisit(i, p) for p in range(NUM_PASSES) for i in range(CORE_START, CORE_END)]
    out += [LayerVisit(i, 0) for i in range(CORE_END, NUM_LAYERS)]
    return out


def is_local(layer: int) -> bool:
    return layer in LOCAL_LAYER_IDS


@dataclass
class DecodeBudget:
    """Analytic decode-token budget: bytes read / achieved bandwidth."""
    bytes_read: int = WEIGHTS_READ_PER_DECODE_TOKEN
    kv_bytes: int = 0
    attention_flops: float = 0.0
    expert_flops: float = 0.0
    dense_flops: float = 0.0
    notes: list[str] = field(default_factory=list)


def qkv_dims(layer: int) -> dict:
    """Per-layer attention projection sizes.

    Local layers carry the short convs (k/v sconv static weights) and a 16-dim
    relative-bias projection; global layers project their conv weights. All 24
    layers use 24 Q heads x 128 and 8 KV heads x 128, with fused Du/Dv widths.
    """
    return {
        "wq_du": (NUM_HEADS * HEAD_DIM, HIDDEN),          # 3072 x 2304
        "wk_dv": (NUM_KV_HEADS * HEAD_DIM, HIDDEN),       # 1024 x 2304
        "wv_dv": (NUM_KV_HEADS * HEAD_DIM, HIDDEN),       # 1024 x 2304
        "wo_ud": (HIDDEN, NUM_HEADS * HEAD_DIM),          # 2304 x 3072
        "wr_du": ((HEAD_DIM // 8) * NUM_HEADS, HIDDEN),   # 384 x 2304 relative bias
        "gate_attn": (NUM_HEADS * HEAD_DIM, HIDDEN),      # gate on attention output
        "sink": (NUM_HEADS, HIDDEN),
        "sconv": HIDDEN if not is_local(layer) else NUM_KV_HEADS * HEAD_DIM,
        "local": is_local(layer),
    }


def mlp_bytes_per_token_per_layer(top_k: int = TOP_K) -> int:
    """Bytes of expert weights read per token in one MoE layer, at a given top-k."""
    t = tiers()
    per = {tier.storage: tier.weight_bytes_per_expert + tier.scale_bytes_per_expert for tier in t}
    # Same per-expert size on both sides (storage differs, logical matrix identical).
    return top_k * next(iter(per.values()))


def gqa_kv_bytes_per_token(context: int, layers: int = 48) -> int:
    """KV bytes per token for `layers` cache slots: K and V, 8 heads x 128, bf16."""
    return layers * context * 2 * NUM_KV_HEADS * HEAD_DIM * 2
