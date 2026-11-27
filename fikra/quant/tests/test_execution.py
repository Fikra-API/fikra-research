# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Execution-budget contracts.

This module decides the project's schedule, so its arithmetic has to be right:
it is the thing that says fusion is worth 2x and that 60 tok/s needs 154 GB/s
rather than the 205.8 a bad earlier budget implied.
"""
from __future__ import annotations

import pytest

from fikra.quant.execution import LayerFootprint, budget_for, compare
from fikra.quant.plans import Plan

FP = LayerFootprint(dense_bytes=100, expert_bytes=50, quantised_expert_bytes=50)
PLAIN = Plan.every_layer_once(4)


def test_unfused_counts_every_visit():
    """A layer visited three times pays three times when not fused."""
    plan = Plan.looped(4, 1, 3, 3)
    b = budget_for(plan, FP)
    # visits: 1 + (2 layers x 3) + 1 = 8
    assert b.visits == 8
    assert b.bytes_per_token == 150 * 8
    assert b.fused_layers == 0


def test_fused_counts_each_layer_once():
    plan = Plan.looped(4, 1, 3, 3, fuse_core=True)
    b = budget_for(plan, FP)
    # still 8 visits, but the core's two layers read once each
    assert b.visits == 8
    assert b.fused_layers == 2
    # 2 outer layers at 3 visits each? no: outer layers run once -> 1+1, core once
    assert b.bytes_per_token == 150 * 4


def test_fusion_halves_inkling_traffic():
    """The headline: 48 reads becomes 24, so bytes per token halves."""
    ink = Plan.looped(24, 6, 18, 3, expert_layers=range(24))
    ink_f = Plan.looped(24, 6, 18, 3, expert_layers=range(24), fuse_core=True)
    footprint = LayerFootprint(dense_bytes=int(71.2e6), expert_bytes=int(35.8e6))
    u = budget_for(ink, footprint)
    f = budget_for(ink_f, footprint)
    assert round(u.bytes_per_token / f.bytes_per_token, 2) == 2.0
    assert round(u.tok_s(64.4), 1) == 12.5
    assert round(f.tok_s(64.4), 1) == 25.1


def test_plain_moe_has_nothing_to_fuse():
    b = budget_for(PLAIN, FP)
    assert b.fused_layers == 0
    assert b.bytes_per_token == 150 * 4


def test_tok_s_scales_linearly_with_bandwidth():
    b = budget_for(PLAIN, FP)
    assert b.tok_s(100) == pytest.approx(b.tok_s(50) * 2)


def test_zero_footprint_does_not_divide_by_zero():
    empty = LayerFootprint(dense_bytes=0, expert_bytes=0)
    assert budget_for(PLAIN, empty).tok_s(100.0) == 0.0


def test_compare_reports_the_speedup_and_both_tables():
    ink = Plan.looped(24, 6, 18, 3)
    ink_f = Plan.looped(24, 6, 18, 3, fuse_core=True)
    footprint = LayerFootprint(dense_bytes=int(71.2e6), expert_bytes=int(35.8e6))
    out = compare(ink, ink_f, footprint)
    assert out["speedup_from_fusion"] == 2.0
    assert set(out["fused"]["tok_s"]) == {"64.4", "150", "205.8", "243.8"}
    assert out["fused"]["tok_s"]["150"] > out["unfused"]["tok_s"]["150"]


def test_fusion_plus_full_bandwidth_reaches_the_60_target():
    """The corrected plan: 60 tok/s needs fusion and ~154 GB/s, not 205.8."""
    ink_f = Plan.looped(24, 6, 18, 3, fuse_core=True)
    footprint = LayerFootprint(dense_bytes=int(71.2e6), expert_bytes=int(35.8e6))
    b = budget_for(ink_f, footprint)
    assert b.tok_s(154) >= 59
    assert b.tok_s(154) < 60.5


# --- fused executor --------------------------------------------------------

from fikra.quant.execution import CountingStore, execute  # noqa: E402

LAYER = int(107e6)


def _noop(_i, _p):
    pass


def test_executor_visits_the_same_layers_either_way():
    """Fusion must not change what runs, only what is re-read."""
    ink = Plan.looped(24, 6, 18, 3)
    a = execute(ink, CountingStore(), LAYER, _noop, fuse=False)
    b = execute(ink, CountingStore(), LAYER, _noop, fuse=True)
    assert a["visits"] == b["visits"] == 48


def test_executor_fused_reads_half_the_bytes():
    ink = Plan.looped(24, 6, 18, 3)
    unfused = execute(ink, CountingStore(), LAYER, _noop, fuse=False)["reads"]
    fused = execute(ink, CountingStore(), LAYER, _noop, fuse=True)["reads"]
    assert unfused["total_reads"] == 48
    assert fused["total_reads"] == 24
    assert round(unfused["GB"] / fused["GB"], 2) == 2.0


def test_executor_keeps_repeated_layers_resident():
    ink = Plan.looped(24, 6, 18, 3)
    r = execute(ink, CountingStore(), LAYER, _noop, fuse=True)
    assert r["resident_layers"] == list(range(6, 18))


def test_executor_reads_passed_layers_once():
    ink = Plan.looped(24, 6, 18, 3)
    st = CountingStore()
    execute(ink, st, LAYER, _noop, fuse=True)
    for i in list(range(0, 6)) + list(range(18, 24)):
        assert st.reads[f"layer{i}"] == 1


def test_executor_reads_core_once_when_fused():
    ink = Plan.looped(24, 6, 18, 3)
    st = CountingStore()
    execute(ink, st, LAYER, _noop, fuse=True)
    for i in range(6, 18):
        assert st.reads[f"layer{i}"] == 1


def test_executor_plain_plan_is_unaffected_by_fusion():
    plain = Plan.every_layer_once(8)
    a = execute(plain, CountingStore(), LAYER, _noop, fuse=False)["reads"]["total_reads"]
    b = execute(plain, CountingStore(), LAYER, _noop, fuse=True)["reads"]["total_reads"]
    assert a == b == 8, "a plain stack has no repeats, so fusion cannot help"


def test_executor_calls_the_layer_in_the_right_order():
    seen = []
    ink = Plan.looped(4, 1, 3, 3)
    execute(ink, CountingStore(), LAYER, lambda i, p: seen.append((p, i)), fuse=True)
    # pass 0: 0,1,2,3 ; pass 1: 1,2 ; pass 2: 1,2
    assert seen == [(0, 0), (0, 1), (0, 2), (0, 3),
                    (1, 1), (1, 2), (2, 1), (2, 2)]


# --- plan to kernel arrays -------------------------------------------------

from fikra.quant.fused_kernel import plan_to_arrays  # noqa: E402


def test_plan_to_arrays_covers_every_visit_in_order():
    arr = plan_to_arrays(Plan.looped(4, 1, 3, 2))
    assert arr["visits"] == arr["visits"]
    assert arr["layer_of_visit"] == [i for i, _ in
                                     [(0, 0), (1, 0), (2, 0), (3, 0),
                                      (1, 1), (2, 1)]]
    assert arr["pass_of_visit"] == [0, 0, 0, 0, 1, 1]


def test_plan_to_arrays_marks_only_repeated_shareable_layers_resident():
    arr = plan_to_arrays(Plan.looped(24, 6, 18, 3, fuse_core=True))
    assert arr["resident_layers"] == list(range(6, 18))
    assert arr["visits"] == 48


def test_plan_to_arrays_plain_stack_has_no_residents():
    arr = plan_to_arrays(Plan.every_layer_once(12))
    assert arr["resident_layers"] == []
    assert arr["visits"] == 12


def test_plan_to_arrays_matches_executor_visit_count():
    """The GPU schedule and the reference executor must agree on visit count."""
    ink = Plan.looped(24, 6, 18, 3)
    gpu_side = plan_to_arrays(ink)["visits"]
    cpu_side = execute(ink, CountingStore(), LAYER, _noop, fuse=False)["visits"]
    assert gpu_side == cpu_side == 48
