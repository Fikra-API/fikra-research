# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Memory layout and grid arithmetic for the FP4-dequant MoE GEMV.

Decode is one token at a time, so every expert matmul is a GEMV:

    down_proj(silu(gate(x)) * up(x)) with gate/up stored as one fused w13.

The GPU-side kernel is Triton (``gemv_triton.py``). Everything that can be
wrong without a GPU -- nibble order, scale indexing, tile sizes, grid shape,
which experts a block handles -- lives here and is unit-tested on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

BLOCK_N = 16     # output rows per program
BLOCK_K = 256    # K elements walked per inner loop; w13 (K=2304) divides, w2 (1152) masks tail
SCALE_BLOCK = 16  # NVFP4 scale group along K; K must be a multiple of it
NUM_WARPS = 4


@dataclass(frozen=True)
class GemvPlan:
    """Grid and tile shape for one (expert tile, top-k) GEMV."""
    rows: int          # N of w13 (2 * intermediate) or w2 (hidden)
    k: int
    experts: int       # number of selected experts the kernel processes
    block_n: int = BLOCK_N
    block_k: int = BLOCK_K
    num_warps: int = NUM_WARPS

    @property
    def n_programs(self) -> int:
        if self.rows % self.block_n:
            raise ValueError(f"N={self.rows} not divisible by BLOCK_N={self.block_n}")
        return self.rows // self.block_n

    @property
    def grid(self) -> tuple[int, int]:
        """(column tiles, selected experts). One program = one 16-wide tile."""
        return self.n_programs, self.experts

    @property
    def k_steps(self) -> int:
        """Inner-loop steps; the last one is masked when K is not a multiple.

        K still has to be a multiple of the NVFP4 scale block, because the
        checkpoint stores one FP8 scale per 16 K elements and a ragged group
        would need a scale the weights do not carry.
        """
        if self.k <= 0 or self.k % SCALE_BLOCK:
            raise ValueError(f"K={self.k} is not a positive multiple of the "
                             f"NVFP4 scale block {SCALE_BLOCK}")
        return -(-self.k // self.block_k)

    @property
    def scales_per_row(self) -> int:
        """FP8 scales one output row needs: one per 16 K elements."""
        return self.k // SCALE_BLOCK


def w13_plan(num_experts: int = 8) -> GemvPlan:
    from . import model_spec as ms
    return GemvPlan(rows=2 * ms.INTERMEDIATE, k=ms.HIDDEN, experts=num_experts)


def w2_plan(num_experts: int = 8) -> GemvPlan:
    from . import model_spec as ms
    return GemvPlan(rows=ms.HIDDEN, k=ms.INTERMEDIATE, experts=num_experts)


def tile_bytes(plan: GemvPlan) -> int:
    """Bytes a BLOCK_N-row tile streams per expert: packed weights + FP8 scales.

    Scales run along K (``[N, K/16]``), so a tile reads ``block_n`` rows x
    ``k/16`` scales, not one per row.
    """
    packed = plan.k * plan.block_n // 2
    scales = plan.block_n * plan.scales_per_row
    return packed + scales


def gather_offsets(selected, rows_per_expert: int, block_n: int = BLOCK_N) -> list[int]:
    """Flatten [(expert, row)] x column tile order into expert*N + row.

    The Triton program maps ``pid0`` to a column tile and ``pid1`` to a slot in
    the top-k list, so the offsets are expert-major. Kept here so the host-side
    gather and the kernel index math cannot drift apart.
    """
    if len(selected) == 0:
        return []
    out = []
    for slot, expert in enumerate(selected):
        if expert < 0:
            raise ValueError(f"slot {slot}: negative expert id")
        out.append(expert * rows_per_expert)
    return out


def combined_output_offsets(selected, hidden: int) -> list[int]:
    """Where each expert's hidden-vector output lands in the [topk, hidden] buffer."""
    return [slot * hidden for slot in range(len(selected))]
