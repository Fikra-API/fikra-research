# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Grid math for the GEMV: divisibility, tile bytes, gather offsets."""
import pytest

from fikra.inkling_t4l4 import kernel_spec as ks


def test_w13_and_w2_plans_divide_cleanly():
    # w13 K = hidden 2304 = 9 x 256; w2 K = intermediate 1152, which needs the
    # masked final k step (1152 = 4 x 256 + 128).
    for plan in (ks.w13_plan(), ks.w2_plan()):
        assert plan.grid[0] == plan.rows // 16
        assert -(-plan.k // plan.block_k) * plan.block_k >= plan.k
    assert ks.w13_plan().k_steps == 9


def test_block_n_is_exactly_one_scale_group():
    assert ks.BLOCK_N == 16


def test_tile_bytes_include_scales_per_k_block():
    plan = ks.w2_plan()
    # 16 output rows x 1152 K nibble-pairs (/2) plus 16 rows x 72 FP8 scales.
    assert ks.tile_bytes(plan) == plan.k * 16 // 2 + 16 * (plan.k // 16)


def test_k_must_be_a_whole_number_of_scale_blocks():
    assert ks.w2_plan().scales_per_row == 72
    with pytest.raises(ValueError, match="scale block"):
        ks.GemvPlan(rows=16, k=24, experts=1).k_steps


def test_grid_covers_every_selected_expert():
    plan = ks.w13_plan(num_experts=8)
    assert plan.grid == (2304 // 16, 8)


def test_bad_shapes_refused():
    with pytest.raises(ValueError):
        ks.GemvPlan(rows=10, k=64, experts=1).n_programs
    with pytest.raises(ValueError):
        ks.GemvPlan(rows=16, k=100, experts=1).k_steps
    with pytest.raises(ValueError):
        ks.GemvPlan(rows=16, k=0, experts=1).k_steps


def test_gather_offsets_are_expert_major():
    assert ks.gather_offsets([0, 3], rows_per_expert=1152) == [0, 3456]
    with pytest.raises(ValueError):
        ks.gather_offsets([-1], 1152)
    assert ks.combined_output_offsets([5, 7], 2304) == [0, 2304]
