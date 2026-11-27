"""Execution plans: which layers run, in what order, and how many times.

This is the second half of being model-agnostic. A plain MoE (Qwen 3.8 27B) runs
each layer once. A looped model (Inkling 2 Mini) runs a middle block three times
per token. A dense model has no expert routing at all. All three are the same
data structure here, so the kernels never branch on model family.

The plan is also where the savings live for a looped model: layers that repeat
re-read their weights unless the pass is fused, and the budget for a T4 says
that fusion is worth more than any kernel micro-optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerPlan:
    """One layer and how many times a token passes through it."""

    index: int
    passes: int = 1
    is_expert_layer: bool = False
    share_weights_across_passes: bool = False
    notes: str = ""

    @property
    def weight_reads(self) -> int:
        """How many times this layer's weights are read per token.

        Sharing across passes means the weights are read once and kept, which is
        the difference between 48 weight reads per token and 24 on Inkling.
        """
        return 1 if self.share_weights_across_passes else self.passes


@dataclass
class Plan:
    layers: list[LayerPlan] = field(default_factory=list)

    @classmethod
    def every_layer_once(cls, n_layers: int, expert_layers=()) -> "Plan":
        """A plain stack: every layer once. Qwen and most MoE models."""
        return cls([LayerPlan(i, 1, i in set(expert_layers)) for i in range(n_layers)])

    @classmethod
    def looped(
        cls,
        n_layers: int,
        core_start: int,
        core_end: int,
        passes: int,
        expert_layers=(),
        fuse_core: bool = False,
    ) -> "Plan":
        """A looped stack: [0, core_start) once, [core_start, core_end) x passes,
        [core_end, n_layers) once. Inkling 2 Mini is 6 / 6-17 x3 / 18-23.

        ``fuse_core`` says the middle block keeps its weights resident across
        passes. It is off by default because it is the expensive path to get
        right, but it is the single largest lever on the budget.
        """
        if not 0 <= core_start < core_end <= n_layers:
            raise ValueError("core range must satisfy 0 <= start < end <= n_layers")
        if passes < 1:
            raise ValueError("passes must be >= 1")
        experts = set(expert_layers)
        out = []
        for i in range(n_layers):
            if core_start <= i < core_end:
                out.append(LayerPlan(i, passes, i in experts, fuse_core))
            else:
                out.append(LayerPlan(i, 1, i in experts))
        return cls(out)

    # --- what the kernels and the budget ask it -----------------------------

    def weight_reads_per_token(self) -> int:
        return sum(l.weight_reads for l in self.layers)

    def layer_visits(self) -> int:
        return sum(l.passes for l in self.layers)

    def expert_layer_indices(self) -> list[int]:
        return [l.index for l in self.layers if l.is_expert_layer]

    def describe(self) -> dict:
        return {
            "n_layers": len(self.layers),
            "layer_visits": self.layer_visits(),
            "weight_reads": self.weight_reads_per_token(),
            "expert_layers": len(self.expert_layer_indices()),
            "fused_layers": sum(1 for l in self.layers if l.share_weights_across_passes),
            "repeats": {str(l.index): l.passes for l in self.layers if l.passes > 1},
        }
