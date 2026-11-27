# Copyright 2026 Fikra API
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Inkling 2 Mini, decode-only, written against the released checkpoint.

This is the first place in the project where a token goes through all 48 layer
visits with real weights and our own expert kernels. Everything upstream of it
was either one layer or an analytic budget; the number this file produces is the
first end-to-end tok/s we can quote.

What is deliberately NOT here:

* prefill. Decode-only keeps the KV cache append a single slot and removes the
  whole mask/packing surface. A token is fed one at a time; the first token is
  prefill in spirit and is excluded from the timing by the harness.
* the MTP draft head. Drafting is a separate multiplier and needs its own
  measured acceptance rate first.

Architecture, pinned from the SGLang reference (``inkling2.py`` and
``inkling_common/*``) rather than guessed:

* attention-residual stream: layer L's attention input is a softmax over
  {all completed block summaries, current partial} scored by a learned query,
  each source RMS-normalised. Every 6 layers a new block starts.
* short convolutions (kernel 4, depthwise, residual + scale) on the attention
  output, the MLP output, and (global layers only) the K/V streams. K/V weights
  are dynamic there: a projection of the layer input added to a static 4-tap.
* loop: prelude 0-5, core 6-17 three times with shared weights, coda 18-23.
  Later core passes start again from the prelude's residual state and see the
  previous pass only through message controllers (rank 64, tanh gain/forget).
* attention: GQA 24/8 heads x 128, per-head QK L2-norm clip with a learned
  per-head scale, learned relative-position bias over 1024 taps computed from a
  16-dim projection of the input, a sigmoid gate on the attention output, and a
  learned per-head attention sink.
* MoE: top-8 of 128 by sigmoid(logits) + bias, weights renormalised over the
  top-8 as log-sigmoid probabilities, times ``route_scale`` 8. Layers 0-19 are
  NVFP4, 20-23 MXFP8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import model_spec as ms
from .gemv_triton import nvfp4_gemv
from .moe_kernels import capped_swiglu, mxfp8_gemv

EPS = 1e-6
ROUTE_SCALE = 8.0
GATE_CAP = 4.0
UP_CAP = 25.0
L2_THRESHOLD = 22.627416997969522
REL_L2_THRESHOLD = 8.0
LOG_SCALING_N_FLOOR = 8192
LOG_SCALING_ALPHA = 0.1
FINAL_SOFT_CAP = 30.0
SINK_SLIDING = 1024


# ---------------------------------------------------------------------------
# small primitives
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, w: torch.Tensor | None = None, eps: float = EPS):
    """RMSNorm over the last dim; no mean subtraction, which is the whole point."""
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y = x * rstd
    return y if w is None else y * w


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None,
            eps: float = EPS):
    """Per-row L2 norm clipped from above, times a learned per-row scale.

    The reference reads ``norm = sum_sq * rsqrt(sum_sq)``, i.e. ``sqrt(sum_sq)``
    computed that way; same value, kept in that form so the rounding matches.
    """
    if scale is not None:
        x = x * scale
    sum_sq = x.pow(2).sum(-1, keepdim=True)
    norm = sum_sq * torch.rsqrt(sum_sq.clamp_min(1e-30))
    coeff = torch.clamp(threshold / (norm + eps), max=1.0)
    return x * coeff


def short_conv(x: torch.Tensor, w: torch.Tensor, cache: torch.Tensor,
               dynamic: torch.Tensor | None = None):
    """Depthwise causal conv over the token axis, kernel 4, residual + scale.

    ``w`` is [D, 1, 4] (static), ``dynamic`` is [D//head_dim, 4] per token for
    the global-layer K/V streams. Cache is [3, D] holding the last three inputs.
    The scale is ``(W * (2 if dynamic else 1) + 1) ** -0.5`` -- the ``+1`` is the
    residual tap counted as a weight, which is what the reference kernel does.
    """
    d = x.shape[-1]
    width = w.shape[-1]
    cur = x
    taps = []
    for lag in range(width - 1, -1, -1):
        if lag == 0:
            taps.append(cur)
        else:
            taps.append(cache[width - 1 - lag])
    # update cache with the new input (oldest first)
    if width > 1:
        cache[:-1] = cache[1:].clone()
        cache[-1] = cur
    acc = None
    for tap, value in enumerate(taps):
        wt = w[:, 0, tap].to(torch.float32)
        if dynamic is not None:
            wt = wt + dynamic[:, tap].repeat_interleave(d // dynamic.shape[0])
        term = value.to(torch.float32) * wt
        acc = term if acc is None else acc + term
    acc = acc + cur.to(torch.float32)
    scale = (width * (2 if dynamic is not None else 1) + 1) ** -0.5
    return (acc * scale).to(torch.bfloat16)


def attn_res(sources: list[torch.Tensor], query: torch.Tensor,
             normalize_values: bool = True):
    """One softmax mix of the residual sources scored by a learned query.

    Per source: rstd = rsqrt(mean(x^2) + eps); score = (q * rstd) . x; weights are
    a softmax over sources; output = sum w_i * x_i, optionally re-scaled by rstd
    so a source's own magnitude does not leak into the mix.
    """
    x = torch.stack(sources, 0).float()                        # [N, T, D]
    stat = torch.rsqrt(x.pow(2).mean(-1) + EPS)                # [N, T]
    logits = (x * query.view(1, 1, -1)).sum(-1) * stat          # [N, T]
    w = torch.softmax(logits, dim=0)
    if normalize_values:
        w = w * stat
    return (w.unsqueeze(-1) * x).sum(0)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

@dataclass
class LayerWeights:
    layer: int
    local: bool
    attn_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    v_norm: torch.Tensor
    mlp_norm: torch.Tensor
    wq_du: torch.Tensor
    wk_dv: torch.Tensor
    wv_dv: torch.Tensor
    wo_ud: torch.Tensor
    wr_du: torch.Tensor
    gate_attn: torch.Tensor
    sink: torch.Tensor
    q_clip_scale: torch.Tensor
    rel_clip_scale: torch.Tensor
    rel_proj: torch.Tensor
    attn_sconv: torch.Tensor
    mlp_sconv: torch.Tensor
    k_sconv: torch.Tensor
    v_sconv: torch.Tensor
    k_sconv_dyn: torch.Tensor | None
    v_sconv_dyn: torch.Tensor | None
    gate_w: torch.Tensor
    gate_b: torch.Tensor
    experts: dict            # w13/w2 packed tensors + scales + global scale
    controllers: dict | None


class InklingRunner:
    """Decode-only Inkling 2 Mini over one device."""

    def __init__(self, weights: dict, devices=("cuda:0",),
                 split_layer: int | None = None, dtype=torch.bfloat16,
                 kv_dtype=torch.bfloat16, max_tokens: int = 4096):
        """``devices`` may be one or two GPUs.

        With two, layers ``[0, split_layer)`` live on ``devices[0]`` and the rest
        on ``devices[1]``. The checkpoint is 19.5 GB and one T4 holds 15.6 GB, so
        on a single card this model does not fit at all; the split is a
        requirement, not an optimisation. The residual stream is copied between
        cards once per visit that crosses the boundary.
        """
        self.devs = [torch.device(d) for d in devices]
        self.dev = self.devs[0]
        self.split_layer = split_layer
        self.dtype = dtype
        self.w = weights
        self.max_tokens = max_tokens
        self.layers = [self._layer(i) for i in range(ms.NUM_LAYERS)]
        # attn_queries is indexed by layer - 1: the first layer has no attention
        # input to mix, so the checkpoint stores 23 entries for 24 layers.
        self.attn_queries = [weights[f"model.llm.attn_res.attn_queries.{i}"].to(self.dev)
                             for i in range(23)]
        self.mlp_queries = [weights[f"model.llm.attn_res.mlp_queries.{i}"].to(self.dev)
                            for i in range(24)]
        self.output_query = weights["model.llm.attn_res.output_query"].to(self.dev)
        self.embed = weights["model.llm.embed.weight"].to(self.dev)
        self.embed_norm = weights["model.llm.embed_norm.weight"].to(self.dev)
        self.final_norm = weights["model.llm.norm.weight"].to(self.dev)
        # the unembedding lives with the coda layers, on the last card
        self.unembed = weights["model.llm.unembed.weight"].to(self.devs[-1])
        self.cross_pass_q = weights["model.llm.middle_cycle_cross_pass_queries"].to(self.dev)
        self.cross_pass_kv_q = weights["model.llm.middle_cycle_cross_pass_kv_queries"].to(self.dev)

        # A KV slot per layer visit (48 of them): core layers get one per pass,
        # because each pass attends over its own history, not the previous pass's.
        n_slots = 48
        slot_dev = [self._slot_device(self._slot_layer(i)) for i in range(n_slots)]
        self.kv, self.kv_len, self.conv_attn, self.conv_mlp, self.conv_kv = [], [], [], [], []
        for i in range(n_slots):
            dv = slot_dev[i]
            self.kv.append(torch.zeros((2, max_tokens, ms.NUM_KV_HEADS, ms.HEAD_DIM),
                                       dtype=kv_dtype, device=dv))
            self.kv_len.append(torch.zeros((), dtype=torch.long, device=dv))
            self.conv_attn.append(torch.zeros((3, ms.HIDDEN), dtype=self.dtype, device=dv))
            self.conv_mlp.append(torch.zeros((3, ms.HIDDEN), dtype=self.dtype, device=dv))
            self.conv_kv.append(torch.zeros((2, 3, ms.NUM_KV_HEADS * ms.HEAD_DIM),
                                            dtype=self.dtype, device=dv))

    @staticmethod
    def _slot_layer(slot: int) -> int:
        """Which model layer a cache slot belongs to."""
        if slot < ms.CORE_START:
            return slot
        n_core = ms.CORE_END - ms.CORE_START
        if slot < ms.CORE_START + n_core * ms.NUM_PASSES:
            return ms.CORE_START + (slot - ms.CORE_START) % n_core
        return ms.CORE_END + (slot - ms.CORE_START - n_core * ms.NUM_PASSES)

    def _slot_device(self, layer: int) -> torch.device:
        return self.dev_for(layer)

    def dev_for(self, layer: int) -> torch.device:
        """Which card holds this layer's weights."""
        if len(self.devs) == 1 or self.split_layer is None:
            return self.devs[0]
        return self.devs[0] if layer < self.split_layer else self.devs[1]

    def _layer(self, i: int) -> LayerWeights:
        p = f"model.llm.layers.{i}."
        dev = self.dev_for(i)
        g = lambda k, d=None: (None if self.w.get(p + k, d) is None
                               else self.w.get(p + k, d).to(dev))   # noqa: E731
        local = ms.is_local(i)
        nv = i < 20
        wing = dict(
            layer=i, local=local,
            attn_norm=g("attn_norm.weight"), mlp_norm=g("mlp_norm.weight"),
            q_norm=g("q_input_norm.weight"), k_norm=g("k_input_norm.weight"),
            v_norm=g("v_input_norm.weight"),
            wq_du=g("attn.wq_du.weight"), wk_dv=g("attn.wk_dv.weight"),
            wv_dv=g("attn.wv_dv.weight"), wo_ud=g("attn.wo_ud.weight"),
            wr_du=g("attn.wr_du.weight"), gate_attn=g("attn.gate_attn.weight"),
            sink=g("attn.attn_sink_proj.weight"),
            q_clip_scale=g("attn.q_l2_clip.scale"),
            rel_clip_scale=g("attn.rel_l2_clip.scale"),
            rel_proj=g("attn.rel_logits_proj.proj"),
            attn_sconv=g("attn_sconv.weight"), mlp_sconv=g("mlp_sconv.weight"),
            k_sconv=g("attn.k_sconv.weight", g("attn.k_sconv.static_w")),
            v_sconv=g("attn.v_sconv.weight", g("attn.v_sconv.static_w")),
            k_sconv_dyn=g("attn.k_sconv.weight_proj"),
            v_sconv_dyn=g("attn.v_sconv.weight_proj"),
            gate_w=g("mlp.gate.weight"), gate_b=g("mlp.gate.bias"),
            experts={
                "fmt": "nvfp4" if nv else "mxfp8",
                "w13": g("mlp.experts.w13_weight"), "s13": g("mlp.experts.w13_weight_scale"),
                "w2": g("mlp.experts.w2_weight"), "s2": g("mlp.experts.w2_weight_scale"),
                "g13": (g("mlp.experts.w13_weight_amax") / g("mlp.experts.w13_weight_outer_scale_denominator")
                        if nv else None),
                "g2": (g("mlp.experts.w2_weight_amax") / g("mlp.experts.w2_weight_outer_scale_denominator")
                       if nv else None),
            },
            controllers=None,
        )
        if 6 <= i < 18:
            c = {}
            for m in ("attn", "k", "v", "mlp"):
                c[m] = {
                    "down": g(f"controllers.{m}.down_weight"),
                    "gain": g(f"controllers.{m}.gain_weight"),
                    "forget": g(f"controllers.{m}.forget_weight"),
                }
            wing["controllers"] = c
        return LayerWeights(**wing)

    # -- helpers ---------------------------------------------------------
    def _mix_message(self, layer: LayerWeights, kind: str, dest: torch.Tensor,
                     messages: list[torch.Tensor]):
        """(1 - tanh(h F^T)) dest + tanh(h G^T) message, h = down(rms(dest,msg))."""
        c = layer.controllers[kind]
        normed = torch.cat([rms_norm(dest.float()), rms_norm(messages[0].float())], -1)
        h = (normed.to(torch.bfloat16) @ c["down"].T).float()
        gain = torch.tanh(h @ c["gain"].T)
        forget = torch.tanh(h @ c["forget"].T)
        return (1.0 - forget) * dest.float() + gain * messages[0].float()

    def _moe(self, layer: LayerWeights, x: torch.Tensor) -> torch.Tensor:
        """top-8 routing and the expert MLP for one token."""
        e = layer.experts
        gate = layer.gate_w.to(torch.float32) @ x.float() + layer.gate_b.float()
        sel = torch.sigmoid(gate)
        top = torch.topk(sel, 8, largest=True, sorted=True).indices
        logits = gate[top]
        # log-sigmoid renormalisation over the top-8, then route_scale
        lp = torch.log(torch.sigmoid(logits) + 1e-30)
        wts = torch.softmax(lp, 0) * ROUTE_SCALE
        if e["fmt"] == "nvfp4":
            h = nvfp4_gemv(x.float(), e["w13"], e["s13"], top, 2 * ms.INTERMEDIATE,
                           ms.HIDDEN, global_scale=e["g13"])
            act = capped_swiglu(h, GATE_CAP, UP_CAP)                    # [8, I]
            y = nvfp4_gemv(act.reshape(-1).float(), e["w2"], e["s2"], top, ms.HIDDEN,
                           ms.INTERMEDIATE, global_scale=e["g2"])
            return (wts[:, None] * y).sum(0).to(self.dtype)
        h = mxfp8_gemv(x.float(), e["w13"], e["s13"], top, 2 * ms.INTERMEDIATE, ms.HIDDEN)
        act = capped_swiglu(h, GATE_CAP, UP_CAP)
        y = mxfp8_gemv(act.reshape(-1).float(), e["w2"], e["s2"], top, ms.HIDDEN, ms.INTERMEDIATE)
        return (wts[:, None] * y).sum(0).to(self.dtype)

    # -- attention -------------------------------------------------------
    def _attention(self, layer: LayerWeights, src: tuple[torch.Tensor, ...],
                   pos: int, slot: int, dynamic_kv: torch.Tensor | None):
        """One decode step of attention for one layer visit."""
        d = self.dev
        # The three attention inputs are each RMS-normalised, and a local layer
        # has one norm for all three while a global layer has one per stream.
        if layer.local:
            q_in = k_in = v_in = rms_norm(src[0].float(), layer.attn_norm).to(self.dtype)
        else:
            q_in = rms_norm(src[0].float(), layer.q_norm).to(self.dtype)
            k_in = rms_norm(src[1].float(), layer.k_norm).to(self.dtype)
            v_in = rms_norm(src[2].float(), layer.v_norm).to(self.dtype)

        q = (layer.wq_du.to(self.dtype) @ q_in)                 # 3072
        kv = (layer.wk_dv.to(self.dtype) @ k_in)               # 1024
        vv = (layer.wv_dv.to(self.dtype) @ v_in)               # 1024
        r = (layer.wr_du.to(self.dtype) @ q_in)                # 384
        gate = (layer.gate_attn.to(self.dtype) @ q_in)         # 3072
        sink = (layer.sink.to(self.dtype) @ q_in)              # 24

        k = short_conv(kv, layer.k_sconv, self.conv_kv[slot][0],
                       None if layer.local else dynamic_kv)
        v = short_conv(vv, layer.v_sconv, self.conv_kv[slot][1],
                       None if layer.local else dynamic_kv)

        # punt on the tau log-scaling below 8192 positions; it is exactly 1.0 there
        tau = None
        if not layer.local:
            eff = float(pos + 1)
            tau = 1.0 + LOG_SCALING_ALPHA * math.log(max(eff / LOG_SCALING_N_FLOOR, 1.0))

        q = l2_clip(q.view(ms.NUM_HEADS, ms.HEAD_DIM), L2_THRESHOLD, layer.q_clip_scale)
        if tau is not None and tau != 1.0:
            q = q * tau
        k = l2_clip(k.view(ms.NUM_KV_HEADS, ms.HEAD_DIM), L2_THRESHOLD, None)
        v = rms_norm(v.view(ms.NUM_KV_HEADS, ms.HEAD_DIM))
        r = l2_clip(r.view(ms.NUM_HEADS, 16), REL_L2_THRESHOLD, layer.rel_clip_scale)
        rel_bias = r.float() @ layer.rel_proj.float().T          # [24, 1024]

        kv = self.kv[slot]
        kv[0, pos] = k.to(kv.dtype)
        kv[1, pos] = v.to(kv.dtype)
        n = pos + 1
        start = 0 if not layer.local else max(0, n - SINK_SLIDING)
        keys = kv[0, start:n].float()                            # [n, 8, 128]
        vals = kv[1, start:n].float()

        qh = q.float().view(ms.NUM_HEADS, ms.HEAD_DIM)
        group = ms.NUM_HEADS // ms.NUM_KV_HEADS
        scores = torch.einsum("hd,nkd->hnk", qh, keys) / math.sqrt(ms.HEAD_DIM)
        # relative bias: distance = q_pos - kv_pos, clamped to [0, extent), then
        # the per-distance bias (same for every KV head of a group) is added
        dist = (pos - torch.arange(start, n, device=d)).clamp(0, SINK_SLIDING - 1)
        bias = rel_bias.float()[:, dist].repeat_interleave(group, 0)   # [24, n]
        scores = scores + bias[:, None, :]
        probs = torch.softmax(scores, dim=-1)                    # [24, n_kv, n]

        out = torch.einsum("hkq,qkd->hd", probs,
                           vals.repeat_interleave(group, 0))     # [24, 128]
        lse = torch.logsumexp(scores, dim=-1).flatten()           # [24 * n_kv]
        if lse.numel() != ms.NUM_HEADS:
            lse = lse.view(ms.NUM_HEADS, -1).max(-1).values
        sink_sig = torch.sigmoid(lse - sink.float())
        out = (out.view(-1) * torch.sigmoid(gate.float()) * sink_sig).to(self.dtype)
        y = (layer.wo_ud.to(self.dtype) @ out).to(self.dtype)
        return short_conv(y, layer.attn_sconv, self.conv_attn[slot])

    # -- one layer visit --------------------------------------------------
    def _visit(self, layer: LayerWeights, state: dict, pos: int, slot: int,
               messages: list[torch.Tensor] | None, cross_kv: torch.Tensor | None):
        # The residual stream follows the layer that is about to read it. On a
        # two-card split this is one 4.6 KB copy per crossing, which is noise;
        # keeping the weights where they are is what matters.
        dev = self.dev_for(layer.layer)
        if len(self.devs) > 1:
            state["partial"] = state["partial"].to(dev)
            state["blocks"] = [b.to(dev) for b in state["blocks"]]
        blocks = state["blocks"]
        partial = state["partial"]
        is_boundary = layer.layer % 6 == 0

        if not blocks:
            src = (partial,) * (1 if layer.local else 3)
        else:
            q = self.attn_queries[layer.layer - 1]
            src = attn_res(blocks + [partial], q)
            src = (src,) if layer.local else (src, src, src)
        if messages is not None:
            messages = {k: v.to(dev) for k, v in messages.items()}
            dest = torch.stack([s.float() for s in src], 0)
            mixed = self._mix_message(layer, "attn", dest, [messages["attn"]])
            if layer.local:
                src = (mixed[0].to(self.dtype),)
            else:
                src = tuple(m.to(self.dtype) for m in mixed)
        attn = self._attention(layer, src, pos, slot, cross_kv)

        # mlp input
        if is_boundary:
            mblocks, mpartial = blocks + [partial], attn
        else:
            mblocks, mpartial = blocks, [partial, attn]
        if isinstance(mpartial, list):
            sources = list(mblocks) + [mpartial[0]]
            # the reference adds the attention output into the partial source
            # before it is normed and dotted, so it belongs on the operand
            sources[-1] = sources[-1].float() + mpartial[1].float()
        else:
            sources = list(mblocks) + [mpartial]
        if not mblocks:
            mlp_in = sources[-1]
        else:
            mlp_in = attn_res(sources, self.mlp_queries[layer.layer])
        mlp_in = rms_norm(mlp_in.float().to(self.dtype), layer.mlp_norm)
        if messages is not None:
            mlp_in = self._mix_message(layer, "mlp", mlp_in.float(), [messages["mlp"]])
            mlp_in = mlp_in.to(self.dtype)

        mlp = self._moe(layer, mlp_in)
        mlp = short_conv(mlp, layer.mlp_sconv, self.conv_mlp[slot])

        if is_boundary:
            state["blocks"] = blocks + [partial]
            state["partial"] = attn + mlp
        else:
            state["partial"] = partial + attn + mlp

    # -- a token ----------------------------------------------------------
    def _embed(self, token_id: int) -> torch.Tensor:
        x = self.embed[token_id].to(self.dtype)
        return rms_norm(x.float(), self.embed_norm).to(self.dtype)

    @torch.no_grad()
    def step(self, token_id: int, pos: int) -> torch.Tensor:
        """One decode step: returns logits for the next token."""
        hidden = self._embed(token_id)
        state = {"blocks": [], "partial": hidden}

        for i in range(ms.CORE_START):
            self._visit(self.layers[i], state, pos, i, None, None)

        prelude_partial, prelude_blocks = state["partial"], list(state["blocks"])
        core = list(range(ms.CORE_START, ms.CORE_END))
        base = ms.CORE_START

        for p in range(ms.NUM_PASSES):
            if p:
                bank = [b.to(self.dev) for b in
                        state["blocks"][len(prelude_blocks) + 1:] + [state["partial"]]]
                q = self.cross_pass_q.view(ms.NUM_PASSES - 1, -1, ms.HIDDEN)[p - 1]
                qkv = self.cross_pass_kv_q.view(ms.NUM_PASSES - 1, -1, ms.HIDDEN)[p - 1]
                m_attn = attn_res(bank, q[:ms.HIDDEN])
                m_mlp = attn_res(bank, q[ms.HIDDEN:])
                m_k = attn_res(bank, qkv[:ms.HIDDEN])
                m_v = attn_res(bank, qkv[ms.HIDDEN:])
                messages = {"attn": m_attn, "mlp": m_mlp}
                state = {"blocks": list(prelude_blocks), "partial": prelude_partial}
            else:
                messages = None
            for j, li in enumerate(core):
                slot = base + p * len(core) + j
                self._visit(self.layers[li], state, pos, slot, messages, None)

        for k, li in enumerate(range(ms.CORE_END, ms.NUM_LAYERS)):
            slot = base + ms.NUM_PASSES * len(core) + k
            self._visit(self.layers[li], state, pos, slot, None, None)

        last = self.devs[-1]
        final = attn_res([b.to(last) for b in state["blocks"]] + [state["partial"].to(last)],
                         self.output_query.to(last))
        final = rms_norm(final.float(), self.final_norm.to(last)).to(self.dtype)
        logits = (self.unembed.to(self.dtype) @ final).float()
        return FINAL_SOFT_CAP * torch.tanh(logits / FINAL_SOFT_CAP)
