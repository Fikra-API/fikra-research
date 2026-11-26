# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Residency planning: routing stats in, a smaller decode budget out.

The point of measuring routing is to decide which experts stay resident. These
tests pin the properties a memory planner depends on, without a GPU.
"""
from __future__ import annotations

import pytest

from fikra.inkling_t4l4 import decode_budget as db
from fikra.inkling_t4l4 import model_spec as ms

FULL = {str(layer): list(range(ms.NUM_EXPERTS)) for layer in range(ms.NUM_LAYERS)}


def test_full_residency_is_the_identity():
    """If every expert is resident, the plan must change nothing."""
    assert db.residency_scale(FULL, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_scale_is_monotonic_in_residency():
    values = [db.residency_scale(FULL, f) for f in (0.125, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values), f"not monotonic: {values}"


def test_scale_never_exceeds_one():
    """A plan can only reduce bytes. Anything above 1.0 is a bug upstream."""
    for frac in (0.01, 0.1, 0.5, 0.99, 1.0):
        assert db.residency_scale(FULL, frac) <= 1.0 + 1e-9


def test_demotion_actually_saves():
    """With cold experts at 3 bits, a 50% residency plan must read less."""
    assert db.residency_scale(FULL, 0.5) < 1.0


def test_empty_heatmap_is_identity():
    assert db.residency_scale({}, 0.5) == 1.0


def test_bad_fraction_rejected():
    with pytest.raises(ValueError):
        db.residency_scale(FULL, 1.5)


def test_plan_from_routing_reports_fit_and_speed():
    plan = db.plan_from_routing(FULL, devices=("T4",), contexts=(2048,), frac_resident=0.5)
    rows = plan["rows"]
    assert rows, "planner returned no scenarios"
    for row in rows:
        assert row["fits_vram"] is True, row
        assert 0 < row["tok_s_at_60pct_bw"] <= row["tok_s_roofline"]
    assert plan["weights_scale"] < 1.0


def test_shipped_budget_misses_the_t4_target_unfused():
    """The negative result the paper has to state honestly."""
    sc = db.Scenario("shipped", "T4", ms.TOP_K, core_passes_fused=False, context=2048)
    assert sc.tok_s(0.6) < 60, "if this passes we are already at target"


def test_fusion_alone_still_misses_but_narrows_the_gap():
    unfused = db.Scenario("u", "T4", ms.TOP_K, False, 2048).tok_s(0.6)
    fused = db.Scenario("f", "T4", ms.TOP_K, True, 2048).tok_s(0.6)
    assert fused > unfused
    assert fused < 60, "fusion alone should not reach 60; spec decode is required"
