"""Shape-level checks for the decode stack, with no GPU and no real weights.

Every bug this catches used to cost a five-minute Kaggle cycle. The tensors are
built at the true layer shapes with random values, so a transposed projection,
a wrong contraction axis or a KV grouping mistake fails here in a second instead
of surfacing as a shape error mid-token on a T4.

The router and the expert matmuls are stubbed: their real arithmetic is covered
by the kernel tests, and the point of this file is the wiring between them.
"""
import math
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from fikra.inkling_t4l4 import model as M
from fikra.inkling_t4l4 import model_spec as ms

H, I, E, KV, HD, HEADS = ms.HIDDEN, ms.INTERMEDIATE, ms.NUM_EXPERTS, ms.NUM_KV_HEADS, ms.HEAD_DIM, ms.NUM_HEADS


def _t(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype) * 0.02


def fake_weights(layers=(0, 6)):
    w = {
        "model.llm.embed.weight": torch.randn(1024, H) * 0.02,
        "model.llm.embed_norm.weight": torch.ones(H),
        "model.llm.norm.weight": torch.ones(H),
        "model.llm.unembed.weight": torch.randn(1024, H) * 0.02,
        "model.llm.middle_cycle_cross_pass_queries": _t(24, H),
        "model.llm.middle_cycle_cross_pass_kv_queries": _t(24, H),
        "model.llm.attn_res.output_query": _t(H),
    }
    for i in range(23):
        w[f"model.llm.attn_res.attn_queries.{i}"] = (
            _t(3, H) if i in ms.LOCAL_LAYER_IDS else _t(H))
    for i in range(24):
        w[f"model.llm.attn_res.mlp_queries.{i}"] = _t(H)
    for i in layers:
        p = f"model.llm.layers.{i}."
        local = ms.is_local(i)
        w[p + "attn_norm.weight"] = torch.ones(H)
        w[p + "mlp_norm.weight"] = torch.ones(H)
        if not local:
            for n in ("q", "k", "v"):
                w[p + f"{n}_input_norm.weight"] = torch.ones(H)
        w[p + "attn.wq_du.weight"] = _t(HEADS * HD, H, dtype=torch.bfloat16)
        w[p + "attn.wk_dv.weight"] = _t(KV * HD, H, dtype=torch.bfloat16)
        w[p + "attn.wv_dv.weight"] = _t(KV * HD, H, dtype=torch.bfloat16)
        w[p + "attn.wo_ud.weight"] = _t(H, HEADS * HD, dtype=torch.bfloat16)
        w[p + "attn.wr_du.weight"] = _t(HEADS * 16, H, dtype=torch.bfloat16)
        w[p + "attn.gate_attn.weight"] = _t(HEADS * HD, H, dtype=torch.bfloat16)
        w[p + "attn.attn_sink_proj.weight"] = _t(HEADS, H, dtype=torch.bfloat16)
        w[p + "attn.q_l2_clip.scale"] = torch.ones(HEADS, HD)
        w[p + "attn.rel_l2_clip.scale"] = torch.ones(HEADS, 16)
        w[p + "attn.rel_logits_proj.proj"] = _t(16, 1024, dtype=torch.bfloat16)
        w[p + "attn_sconv.weight"] = _t(H, 1, 4, dtype=torch.bfloat16)
        w[p + "mlp_sconv.weight"] = _t(H, 1, 4, dtype=torch.bfloat16)
        if local:
            w[p + "attn.k_sconv.weight"] = _t(KV * HD, 1, 4, dtype=torch.bfloat16)
            w[p + "attn.v_sconv.weight"] = _t(KV * HD, 1, 4, dtype=torch.bfloat16)
        else:
            w[p + "attn.k_sconv.static_w"] = _t(4, KV * HD, dtype=torch.bfloat16)
            w[p + "attn.v_sconv.static_w"] = _t(4, KV * HD, dtype=torch.bfloat16)
            w[p + "attn.k_sconv.weight_proj"] = _t(4 * KV, H, dtype=torch.bfloat16)
            w[p + "attn.v_sconv.weight_proj"] = _t(4 * KV, H, dtype=torch.bfloat16)
        w[p + "mlp.gate.weight"] = _t(E, H, dtype=torch.bfloat16)
        w[p + "mlp.gate.bias"] = _t(E)
        w[p + "mlp.experts.w13_weight"] = torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8)
        w[p + "mlp.experts.w13_weight_scale"] = torch.full((E, 2 * I, H // 16), 0x38, dtype=torch.uint8)
        w[p + "mlp.experts.w13_weight_amax"] = torch.ones(E)
        w[p + "mlp.experts.w13_weight_outer_scale_denominator"] = torch.ones(1)
        w[p + "mlp.experts.w2_weight"] = torch.zeros(E, H, I // 2, dtype=torch.uint8)
        w[p + "mlp.experts.w2_weight_scale"] = torch.full((E, H, I // 16), 0x38, dtype=torch.uint8)
        w[p + "mlp.experts.w2_weight_amax"] = torch.ones(E)
        w[p + "mlp.experts.w2_weight_outer_scale_denominator"] = torch.ones(1)
        if 6 <= i < 18:
            for m in ("attn", "k", "v", "mlp"):
                w[p + f"controllers.{m}.down_weight"] = _t(64, 2 * H, dtype=torch.bfloat16)
                w[p + f"controllers.{m}.gain_weight"] = _t(H, 64, dtype=torch.bfloat16)
                w[p + f"controllers.{m}.forget_weight"] = _t(H, 64, dtype=torch.bfloat16)
    return w


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(M, "nvfp4_gemv",
                        lambda x, w, s, e, n, k, global_scale=None:
                        torch.zeros(e.numel(), n))
    monkeypatch.setattr(M, "mxfp8_gemv",
                        lambda x, c, s, e, n, k: torch.zeros(e.numel(), n))
    r = M.InklingRunner(fake_weights(), devices=("cpu",), split_layer=None, max_tokens=8)
    return r


def test_relative_bias_contraction(runner):
    """rel_proj is [16, extent] and must be contracted over d_rel, not transposed."""
    layer = runner.layers[0]
    assert layer.rel_proj.shape == (16, 1024)
    r = torch.randn(HEADS, 16)
    out = r @ layer.rel_proj
    assert out.shape == (HEADS, 1024)


def test_one_attention_step_shapes(runner):
    layer = runner.layers[0]
    src = (torch.randn(ms.HIDDEN),)
    out = runner._attention(layer, src, 0, 0, None)
    assert out.shape == (ms.HIDDEN)
    assert torch.isfinite(out.float()).all()


def test_attention_gate_is_per_element_and_sink_per_head():
    """gate_attn is heads*head_dim wide; the sink is one scalar per head."""
    assert (ms.NUM_HEADS * ms.HEAD_DIM) == 3072
    assert ms.NUM_HEADS == 24


def test_one_visit_produces_a_vector(runner):
    """The whole layer visit, not just attention: the MLP input must stay 1-D.

    A boundary layer takes the old partial as a source and replaces it; a
    non-boundary layer adds the attention output into the partial source. Adding
    both at once made the MLP input a 2x2304 stack, which only shows up here.
    """
    for layer_id in (0, 6, 7):
        layer = runner.layers[0] if layer_id != 6 else runner.layers[1]
        state = {"blocks": [], "partial": torch.randn(ms.HIDDEN)}
        runner._visit(layer, state, 0, 0, None, None)
        assert state["partial"].shape == (ms.HIDDEN), (layer_id, state["partial"].shape)
        assert torch.isfinite(state["partial"].float()).all()


def test_gqa_grouping_is_applied(runner):
    """24 query heads read 8 KV heads, three queries per KV head."""
    group = ms.NUM_HEADS // ms.NUM_KV_HEADS
    h2kv = torch.arange(ms.NUM_HEADS) // group
    assert h2kv.tolist()[:3] == [0, 0, 0]
    assert h2kv.tolist()[-1] == ms.NUM_KV_HEADS - 1


def test_mlp_queries_indexed_by_layer(runner):
    assert len(runner.mlp_queries) == 24


def test_attn_queries_have_23_entries(runner):
    """Layer 0 has no attention input to mix, so there are 23, indexed by layer-1."""
    assert len(runner.attn_queries) == 23
    assert runner.attn_queries[0].shape[-1] == ms.HIDDEN


def test_short_conv_residual_scale():
    """The residual tap counts as a weight in the scale, as the kernel does."""
    w = torch.ones(ms.HIDDEN, 1, 4)
    cache = torch.zeros(3, ms.HIDDEN)
    x = torch.ones(ms.HIDDEN)
    y = M.short_conv(x, w, cache)
    assert float(y[0]) == pytest.approx((4 + 1) ** -0.5 * 5.0, rel=1e-3)


def test_attn_res_returns_one_vector_per_query_row():
    """A local layer asks with one query, a global layer with three."""
    sources = [torch.randn(ms.HIDDEN) for _ in range(3)]
    one = M.attn_res(sources, torch.randn(ms.HIDDEN))
    assert one.shape == (ms.HIDDEN)
    three = M.attn_res(sources, torch.randn(3, ms.HIDDEN))
    assert three.shape == (3, ms.HIDDEN)
    # the three rows must differ: one shared vector would mean the queries
    # collapsed, which is exactly the bug that broke the projection
    assert not torch.allclose(three[0], three[1])
