# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pin the checkpoint facts the kernels and the planner depend on."""
import json
import struct

from fikra.inkling_t4l4 import model_spec as ms

CONFIG = {
    "hidden_size": 2304, "num_hidden_layers": 24, "num_attention_heads": 24,
    "num_key_value_heads": 8, "head_dim": 128, "num_routed_experts": 128,
    "num_experts_per_tok": 8, "intermediate_size": 1152, "sliding_window_size": 1024,
    "local_layer_ids": list(ms.LOCAL_LAYER_IDS),
    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64},
}


def test_shapes_match_config():
    assert CONFIG["hidden_size"] == ms.HIDDEN
    assert CONFIG["intermediate_size"] == ms.INTERMEDIATE
    assert CONFIG["num_attention_heads"] * CONFIG["head_dim"] == 3072
    assert CONFIG["num_key_value_heads"] * CONFIG["head_dim"] == 1024


def test_twenty_local_four_global():
    assert len(ms.LOCAL_LAYER_IDS) == 20
    global_layers = set(range(24)) - set(ms.LOCAL_LAYER_IDS)
    assert global_layers == {5, 11, 17, 23}


def test_schedule_is_48_visits_with_48_cache_slots():
    visits = ms.schedule()
    assert len(visits) == 48
    assert [v.layer for v in visits[:6]] == list(range(6))
    assert [v.layer for v in visits[6:12]] == list(range(6, 12))
    assert [v.loop_pass for v in visits[6:12]] == [0] * 6
    assert [v.loop_pass for v in visits[18:24]] == [1] * 6
    assert [v.layer for v in visits[-6:]] == list(range(18, 24))
    # Every visit attends over its own KV: 12 core layers x 3 passes = 36, plus
    # 12 non-core layers = 48 slots. Returning 42 aliased core pass 1 onto the
    # coda layers 18-23, which is a wrong attention state, not a saving.
    assert len({v.cache_index for v in visits}) == 48


def test_cache_indices_are_stable_per_layer_and_pass():
    by_visit = {(v.layer, v.loop_pass): v.cache_index for v in ms.schedule()}
    assert len(by_visit) == 48
    assert by_visit[(6, 0)] != by_visit[(6, 1)] != by_visit[(6, 2)]
    assert by_visit[(0, 0)] == 0 and by_visit[(23, 0)] == 23
    assert ms.gqa_kv_bytes_per_token(1024, layers=48) == 48 * 1024 * 2 * 8 * 128 * 2


def test_expert_tiers_cover_all_layers():
    tiers = ms.tiers()
    layers = sorted(layer for tier in tiers for layer in tier.layers)
    assert layers == list(range(24))
    nv, mx = tiers
    assert nv.storage == "nvfp4" and mx.storage == "mxfp8"
    # 4.5 nominal bits per weight: 4 payload + one FP8 scale per 16 K (4+0.5).
    assert abs(nv.bits_per_weight - 4.5) < 1e-6
    # MXFP8: 8 payload + one ue8m0 byte per 32 K (8+0.25).
    assert abs(mx.bits_per_weight - 8.25) < 1e-6
    # Per-expert bytes must match the header: 573,309,960 B per layer / 128
    # experts = 4,478,984.06 B. The analytic sizes differ by only the 8 F32
    # amax/denominator bytes per expert, so allow that one-off.
    nv_bytes = nv.weight_bytes_per_expert + nv.scale_bytes_per_expert
    assert 4_478_976 <= nv_bytes <= 4_478_984
    assert abs((mx.weight_bytes_per_expert + mx.scale_bytes_per_expert) - 8_211_456) == 0


def test_tensor_reader_reads_a_real_header(tmp_path):
    from fikra.inkling_t4l4.tensor_reader import read_header, read_tensor

    header = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "__metadata__": {"k": "v"}}
    blob = json.dumps(header).encode()
    path = tmp_path / "x.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + struct.pack("<ff", 1.5, -2.0))
    with path.open("rb") as f:
        tensors, meta = read_header(f)
        assert meta == {"k": "v"}
        info = tensors["w"]
        assert info.shape == (2,) and info.nbytes == 8
        assert read_tensor(f, info, len(blob)) == struct.pack("<ff", 1.5, -2.0)
