# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Analytic decode budget: is 60 tok/s (T4) or 100 tok/s (L4) reachable at all?

Single-stream decode is bandwidth bound. Per token we must read every weight a
token touches: 48 layer visits, but the core layers' weights are read once and
reused across their three passes only if the pass is fused; each visit touches
top-8 of 128 experts. This module states the arithmetic the project must beat,
so a measured number can be checked against a calculation instead of a hunch.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import model_spec as ms


@dataclass
class Scenario:
    name: str
    device: str
    top_k: int
    core_passes_fused: bool  # True: core weights resident for all 3 passes
    context: int
    weights_scale: float = 1.0  # 1.0 as shipped; <1 with hot-expert residency
    spec_accept: float = 1.0    # >1 with a draft head: accepted tokens per weight read

    def expert_bytes(self) -> int:
        # 24 layers of expert weights; a visit reads top_k/128 of its layer.
        all_experts = ms.BYTES["experts_nvfp4"] + ms.BYTES["experts_mxfp8"]
        frac = self.top_k / ms.NUM_EXPERTS
        core = ms.CORE_END - ms.CORE_START
        core_visits = 1 if self.core_passes_fused else ms.NUM_PASSES
        # Prelude + coda always run once: top_k/128 of 12 layers. Core runs 3x.
        weights = all_experts * frac * core / ms.NUM_LAYERS * core_visits \
            + all_experts * frac * (ms.NUM_LAYERS - core) / ms.NUM_LAYERS
        return int(self.weights_scale * weights)

    def dense_bytes(self) -> int:
        # Attention and controllers are read once per layer, not once per core
        # pass: fusing the core means reusing the same dense weights, so the
        # visit count above only applies to expert weights.
        attn_per_layer = ms.BYTES_ATTENTION_BF16 / ms.NUM_LAYERS
        return int(attn_per_layer * ms.NUM_LAYERS + ms.BYTES_EMBED_UNEMBED / 2)

    def kv_bytes(self) -> int:
        # One KV per layer visit (48), not 42: core passes are distinct states.
        return ms.gqa_kv_bytes_per_token(self.context, layers=len(ms.schedule()))

    def total_bytes(self) -> int:
        return self.expert_bytes() + self.dense_bytes() + self.kv_bytes()

    def tok_s(self, efficiency: float = 1.0) -> float:
        hw = ms.DEVICES[self.device]
        return hw["bw_gbps"] * 1e9 * efficiency * self.spec_accept / self.total_bytes()


def report() -> dict:
    scenarios = [
        Scenario("shipped, 48 visits, bf16 attn", "L4", ms.TOP_K, False, 4096),
        Scenario("fused core, 24 visits", "L4", ms.TOP_K, True, 4096),
        Scenario("fused core + 32k ctx", "L4", ms.TOP_K, True, 32768),
        Scenario("fused core, top-4 (quality cost)", "L4", 4, True, 32768),
        Scenario("fused core, 50% experts resident", "L4", ms.TOP_K, True, 32768, weights_scale=0.6),
        Scenario("fused core + 2x spec accept", "L4", ms.TOP_K, True, 32768, spec_accept=2.0),
        Scenario("shipped, 48 visits, bf16 attn", "T4", ms.TOP_K, False, 2048),
        Scenario("fused core", "T4", ms.TOP_K, True, 2048),
        Scenario("fused core, top-4", "T4", 4, True, 2048),
        Scenario("fused core + 2x spec accept", "T4", ms.TOP_K, True, 2048, spec_accept=2.0),
        Scenario("fused core + 2x spec accept + 40% resident", "T4", ms.TOP_K, True, 2048,
                 weights_scale=0.6, spec_accept=2.0),
    ]
    return {
        "scenarios": [
            {
                "name": s.name,
                "device": s.device,
                "top_k": s.top_k,
                "context": s.context,
                "weights_GB": s.total_bytes() / 1e9,
                "expert_GB": s.expert_bytes() / 1e9,
                "kv_GB": s.kv_bytes() / 1e9,
                "tok_s_roofline": s.tok_s(),
                "tok_s_at_60pct_bw": s.tok_s(0.6),
            }
            for s in scenarios
        ],
        "targets": {"T4": 60, "L4": 100},
    }
