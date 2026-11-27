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

    ``dense_bits`` is the precision the dense weights are stored at. It exists
    because measurement showed the dense half is two thirds of the traffic, so
    quantising it moves the needle far more than any expert-side change. 16 is
    bf16, i.e. as shipped.
    """

    dense_bytes: int
    expert_bytes: int
    quantised_expert_bytes: int = 0   # the part that is 4- or 8-bit
    dense_bits: int = 16              # 16 = bf16 as shipped

    def with_dense_bits(self, bits: int) -> "LayerFootprint":
        """The same layer stored with its dense weights at ``bits``."""
        if bits not in (4, 8, 16):
            raise ValueError("dense_bits must be 4, 8 or 16")
        return LayerFootprint(
            dense_bytes=int(self.dense_bytes * bits / 16),
            expert_bytes=self.expert_bytes,
            quantised_expert_bytes=self.quantised_expert_bytes,
            dense_bits=bits,
        )

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


# ---------------------------------------------------------------------------
# Fused execution
# ---------------------------------------------------------------------------

class CountingStore:
    """A weight store that counts bytes actually read.

    Fusion's entire claim is that a layer read once and reused across passes
    moves fewer bytes than the same layer read three times. That claim is only
    verifiable if reads are counted, so the reference executor does exactly that
    rather than asserting the saving.
    """

    def __init__(self):
        self.reads: dict[str, int] = {}
        self.total_bytes = 0

    def read(self, name: str, size: int) -> str:
        self.reads[name] = self.reads.get(name, 0) + 1
        self.total_bytes += size
        return name

    def report(self) -> dict:
        return {"distinct": len(self.reads), "total_reads": sum(self.reads.values()),
                "total_bytes": self.total_bytes,
                "GB": round(self.total_bytes / 1e9, 3),
                "hot": sorted(self.reads.items(), key=lambda kv: -kv[1])[:3]}


def execute(plan, store: CountingStore, layer_size: int, layer_fn,
            fuse: bool = False) -> dict:
    """Run every layer visit in plan order, honouring fusion.

    ``layer_fn(layer_index, pass_index)`` does the layer's work and is where a
    real kernel goes. Fusion changes only whether the weights are read again on
    later passes: with ``fuse`` the layer's weights are read on pass 0 and reused
    for the remaining passes, which is what a resident-block scheduler does.

    Note this is the reference executor, not a GPU scheduler. It makes the byte
    accounting exact so a GPU implementation can be checked against it.
    """
    resident: set[int] = set()
    visits = 0
    max_passes = max((l.passes for l in plan.layers), default=1)
    for pass_index in range(max_passes):
        for layer in plan.layers:
            if pass_index >= layer.passes:
                continue
            visits += 1
            # A layer is reusable if it runs more than once. The plan can state
            # this itself (share_weights_across_passes), but the executor also
            # honours the caller's fuse flag, so the same plan can be measured
            # both ways, which is how the 2x is demonstrated rather than assumed.
            reusable = layer.passes > 1
            reuse = fuse and reusable and pass_index > 0
            if not reuse:
                store.read(f"layer{layer.index}", layer_size)
                if fuse and reusable:
                    resident.add(layer.index)
            layer_fn(layer.index, pass_index)
    return {"visits": visits, "resident_layers": sorted(resident), "reads": store.report()}
