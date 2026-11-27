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
