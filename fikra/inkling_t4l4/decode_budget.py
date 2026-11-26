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


# ---------------------------------------------------------------------------
# Residency planning from measured routing
# ---------------------------------------------------------------------------

# Bits per weight of a demoted ("cold") expert, relative to its shipped 4-bit
# NVFP4 / 8-bit MXFP8. A 3-bit NVFP4-style code plus the same 16-wide scale
# group costs 3/4 of the shipped bytes; resident experts keep their full code.
COLD_BITS = 3
SHIPPED_BITS = {"nvfp4": 4, "mxfp8": 8}


def residency_scale(hot_experts: dict, frac_resident: float, cold_bits: int = COLD_BITS) -> float:
    """Multiplier on expert bytes read per token, given a residency plan.

    ``hot_experts`` is ``RoutingStats.heatmap()['hot_experts']``: per layer, the
    experts covering the chosen share of traffic, hottest first.
    ``frac_resident`` is the share of the 128 experts per layer kept at shipped
    precision; the rest are demoted to ``cold_bits``.

    Shipped cost per layer visit is top_k experts at full precision. Planned
    cost is: the k experts the token picks, charged 1.0 if resident and
    cold_bits/4 if not. Because only picks landing outside the resident set get
    cheaper, the multiplier is monotonic in frac_resident and reaches exactly
    1.0 when everything is resident, which is the floor test.

    A plan can only lower bytes, so anything above 1.0 means the inputs are
    inconsistent and this raises instead of inflating a budget.
    """
    if not 0.0 <= frac_resident <= 1.0:
        raise ValueError("frac_resident must be in [0, 1]")
    if not hot_experts:
        return 1.0
    keep = max(1, min(ms.NUM_EXPERTS, int(round(frac_resident * ms.NUM_EXPERTS))))
    cold_ratio = cold_bits / SHIPPED_BITS["nvfp4"]

    planned = 0.0
    shipped = 0.0
    for layer in range(ms.NUM_LAYERS):
        hot = set(list(hot_experts.get(str(layer), []))[:keep])
        if not hot:
            continue
        # A token picks top_k of 128 experts. The expected number landing on
        # the resident set is top_k * resident_fraction.
        resident_picks = min(float(ms.TOP_K), ms.TOP_K * (len(hot) / ms.NUM_EXPERTS))
        planned += resident_picks * 1.0 + (ms.TOP_K - resident_picks) * cold_ratio
        shipped += ms.TOP_K
    if shipped == 0:
        return 1.0
    scale = planned / shipped
    if scale > 1.0 + 1e-9 or scale <= 0:
        raise ValueError(
            f"residency_scale produced {scale:.4f}; a residency plan cannot "
            "increase bytes read. Check hot_experts and cold_bits.")
    return scale


def plan_from_routing(stats, devices=("T4", "L4"), contexts=(2048, 32768),
                      frac_resident: float = 0.5, spec_accept: float = 1.0) -> dict:
    """Turn measured routing into a decode budget a memory planner can act on.

    Separated from ``report()`` because ``report()`` states the arithmetic with
    assumed residency; this one uses the hot-expert lists we actually measured.
    """
    heat = stats.heatmap()["hot_experts"] if hasattr(stats, "heatmap") else stats
    scale = residency_scale(heat, frac_resident)
    rows = []
    for device in devices:
        for ctx in contexts:
            for fused in (False, True):
                sc = Scenario(
                    name=f"measured residency {int(frac_resident * 100)}%, "
                         f"{'fused' if fused else 'unfused'}, spec x{spec_accept:g}",
                    device=device, top_k=ms.TOP_K, core_passes_fused=fused,
                    context=ctx, weights_scale=scale, spec_accept=spec_accept)
                hw = ms.DEVICES[device]
                rows.append({
                    "device": device, "context": ctx, "fused": fused,
                    "frac_resident": frac_resident, "weights_scale": round(scale, 4),
                    "expert_GB": round(sc.expert_bytes() / 1e9, 3),
                    "total_GB": round(sc.total_bytes() / 1e9, 3),
                    "tok_s_roofline": round(sc.tok_s(1.0), 1),
                    "tok_s_at_60pct_bw": round(sc.tok_s(0.6), 1),
                    "fits_vram": sc.total_bytes() <= hw["vram_gb"] * 1e9,
                })
    return {"frac_resident": frac_resident, "spec_accept": spec_accept,
            "weights_scale": round(scale, 4), "rows": rows}
