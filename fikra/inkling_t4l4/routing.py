# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Expert-routing statistics: who is hot, and does a loop pass pick again?

The 16 GB plan depends on two numbers we cannot guess: how concentrated expert
usage is across the 128 experts of each layer, and how much of pass-2/pass-3
routing repeats pass 1. This accumulator answers both from logged top-k
indices; ``heatmap`` turns it into residency decisions.

Counts are kept as a dense [layers x passes x experts] int64 array because 128
experts x 24 layers x 3 passes is 9k counters, far cheaper than a dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import model_spec as ms


@dataclass
class RoutingStats:
    """Accumulated top-k routing observations.

    ``counts[layer, loop_pass, expert]`` is the number of times a token routed
    to that expert on that visit. ``pairs`` counts (layer, pass, pass-1) joint
    selections for the reuse question.
    """
    counts: object  # numpy int64 [NUM_LAYERS, NUM_PASSES, NUM_EXPERTS]
    tokens: int = 0
    passes_seen: int = 0
    loop_reuse: dict = field(default_factory=dict)  # (layer, pass) -> [tokens, same_as_prev_pass]

    @classmethod
    def empty(cls):
        import numpy as np
        return cls(np.zeros((ms.NUM_LAYERS, ms.NUM_PASSES, ms.NUM_EXPERTS), np.int64))

    def log(self, layer: int, loop_pass: int, topk) -> None:
        """Record one token's selected experts for one layer visit."""
        import numpy as np
        idx = np.asarray(topk, np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("topk must be a non-empty 1-D list of expert ids")
        if idx.min() < 0 or idx.max() >= ms.NUM_EXPERTS:
            raise ValueError("expert id out of range")
        self.counts[layer, loop_pass, idx] += 1
        self.tokens += 1

        if idx.size != 1 and idx.size != len(set(idx.tolist())):
            raise ValueError("a token must not select the same expert twice")

    def log_pass(self, loop_pass: int, per_layer_topk) -> None:
        """Record one whole pass: per_layer_topk[layer] = selected expert ids.

        For core layers on passes 1-2 this also scores overlap with the same
        layer's previous pass, which decides whether a loop pass can reuse the
        previous pass's expert weights (fused pass) or must re-read them.
        """
        prev = getattr(self, "_prev", {})
        this = {}
        for layer, topk in enumerate(per_layer_topk):
            if topk is None:
                continue
            self.log(layer, loop_pass, topk)
            this[layer] = set(int(i) for i in topk)
            if loop_pass > 0 and layer in prev:
                slot = self.loop_reuse.setdefault((layer, loop_pass), [0, 0])
                slot[0] += len(this[layer])
                slot[1] += len(this[layer] & prev[layer])
        self._prev = this
        self.passes_seen += 1

    def reuse_rate(self, layer: int, loop_pass: int) -> float:
        """Share of this visit's expert picks that repeat the previous pass."""
        sel, same = self.loop_reuse.get((layer, loop_pass), [0, 0])
        return same / sel if sel else 0.0

    def summary(self, frac: float = 0.8) -> dict:
        """Per layer: experts covering ``frac`` of activations; plus loop reuse."""
        per_layer = {}
        for layer in range(ms.NUM_LAYERS):
            passes = {}
            for p in range(ms.NUM_PASSES):
                need, cover = self.concentration(layer, p, frac)
                passes[p] = {"experts_for_cover": need, "coverage": round(cover, 4)}
            per_layer[layer] = passes
        reuse = {(l, p): round(self.reuse_rate(l, p), 4) for l in range(ms.CORE_START, ms.CORE_END)
                 for p in range(1, ms.NUM_PASSES)}
        return {"tokens": self.tokens, "passes": self.passes_seen,
                "cover_frac": frac, "per_layer": per_layer,
                "loop_reuse": {f"{l}/{p}": v for (l, p), v in sorted(reuse.items())}}

    def heatmap(self, frac: float = 0.8) -> dict:
        """Sorted hot-expert lists per layer, unioned across passes.

        Gives the memory planner an explicit residency order: hot experts of a
        layer are the ones to keep at 4 bits/NVFP4, the rest can be demoted.
        """
        import numpy as np
        hot = {}
        for layer in range(ms.NUM_LAYERS):
            c = self.counts[layer].sum(0)
            if c.sum() == 0:
                continue
            order = np.argsort(-c)
            need = max(1, int(np.searchsorted(np.cumsum(c[order]) / c.sum(), frac) + 1))
            hot[str(layer)] = [int(i) for i in order[:need]]
        return {"cover_frac": frac, "hot_experts": hot}

    def concentration(self, layer: int, loop_pass: int, frac: float = 0.5):
        """Experts covering ``frac`` of token selections in a layer/pass.

        Returns (experts_needed, their share of all activations). Lower is
        better for residency: if 32 of 128 experts serve 80% of tokens, the
        other 96 can stay cold at 3-4 bits or on the host.
        """
        import numpy as np
        c = self.counts[layer, loop_pass]
        total = int(c.sum())
        if total == 0:
            return 0, 0.0
        order = np.argsort(-c)
        cum = np.cumsum(c[order]) / total
        need = int(np.searchsorted(cum, frac) + 1)
        return need, float(cum[need - 1])
