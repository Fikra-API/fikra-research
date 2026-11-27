"""Execution planning: turn a Plan plus model shapes into a byte budget.

Generic on purpose. Given a Plan (which layers run how often) and a description
of one layer's traffic, this says how many bytes a token must move, and
therefore what tok/s any given bandwidth buys. That is the calculation that
decided Phase 1's schedule, and it works the same for a looped model as for a
plain MoE stack.

The central fact it encodes: a layer visited three times reads its weights three
times unless the weights are kept resident across passes. Resident means the
whole layer fits somewhere fast enough to be reused; on a 15 GB T4 with 6 MB of
L2 and one 107 MB layer, nothing survives between passes, so fusion is all or
nothing per layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .plans import Plan


@dataclass(frozen=True)
class LayerFootprint:
    """Bytes a single layer visit must read.

    ``dense_bytes`` are read once per visit and are never shared.
    ``expert_bytes`` are read once per visit for the experts a token selects.
    """

    dense_bytes: int
    expert_bytes: int
    quantised_expert_bytes: int = 0   # the part that is 4- or 8-bit

    @property
    def per_visit_bytes(self) -> int:
        return self.dense_bytes + self.expert_bytes


@dataclass
class Budget:
    visits: int
    weight_reads: int
    bytes_per_token: int
    fused_layers: int

    def tok_s(self, bandwidth_gbs: float) -> float:
        if self.bytes_per_token <= 0:
            return 0.0
        return bandwidth_gbs * 1e9 / self.bytes_per_token

    def describe(self) -> dict:
        return {
            "visits": self.visits,
            "weight_reads": self.weight_reads,
            "GB_per_token": round(self.bytes_per_token / 1e9, 3),
            "fused_layers": self.fused_layers,
        }


def budget_for(plan: Plan, footprint: LayerFootprint) -> Budget:
    """Bytes per token for this plan.

    A fused layer reads its footprint once regardless of how many times it runs.
    An unfused layer reads it once per pass, because the weights do not survive
    between passes on this hardware.
    """
    total = 0
    fused = 0
    for layer in plan.layers:
        if layer.share_weights_across_passes:
            total += footprint.per_visit_bytes
            fused += 1
        else:
            total += footprint.per_visit_bytes * layer.passes
    return Budget(visits=plan.layer_visits(),
                  weight_reads=plan.weight_reads_per_token(),
                  bytes_per_token=total, fused_layers=fused)


def compare(plan_unfused: Plan, plan_fused: Plan, footprint: LayerFootprint,
            bandwidths=(64.4, 150.0, 205.8, 243.8)) -> dict:
    """Side-by-side, which is the shape the paper and the README both want."""
    bu = budget_for(plan_unfused, footprint)
    bf = budget_for(plan_fused, footprint)
    return {
        "unfused": {**bu.describe(),
                    "tok_s": {f"{b:g}": round(bu.tok_s(b), 1) for b in bandwidths}},
        "fused": {**bf.describe(),
                  "tok_s": {f"{b:g}": round(bf.tok_s(b), 1) for b in bandwidths}},
        "speedup_from_fusion": round(bu.bytes_per_token / bf.bytes_per_token, 2),
    }
