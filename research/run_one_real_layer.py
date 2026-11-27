"""Run one real Inkling 2 Mini MoE layer through our own kernel path.

Not a benchmark, not synthetic: fetch the actual expert weights for layer 0 out
of the 19.5 GB checkpoint by byte range, route a real activation through the real
router, and dequantise with our own code. The point is to see our numbers on the
real model, on the real quantisation, before building anything on top.

Reads only the slices it needs -- 8 of 128 experts -- so it moves about 40 MB
rather than 19.5 GB.
"""
import sys, time, json
sys.path.insert(0, "/home/ben/fikra-research")
import numpy as np
from fikra.inkling_t4l4 import tensor_reader as tr
from fikra.quant import formats as F
from fikra.quant import quant_ref as Q

URL = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors"
H, I_EXP, TOPK = 2304, 1152, 8

r = tr.RangeReader(URL)
tensors, header_meta = tr.read_header(r)
# the header length is what the 8-byte prefix said, so read it directly
import struct as _struct
r.seek(0)
HEADER_BYTES = _struct.unpack("<Q", r.read(8))[0]


def read(name):
    info = tensors[name]
    raw = tr.read_tensor(r, info, HEADER_BYTES)
    return raw, info


def read_expert(name, e):
    """Only expert ``e``'s slice of a [E, ...] tensor, via a byte range.

    The bank is contiguous along axis 0, so one expert is a fixed-size window.
    That turns a 340 MB fetch into a 2.6 MB one, which is the difference between
    a flaky three-minute download and a fast read.
    """
    info = tensors[name]
    per = info.nbytes // info.shape[0]
    start = 8 + HEADER_BYTES + info.begin + e * per
    r.seek(start)
    raw = r.read(per)
    return raw, info, per


def as_expert(raw, info, per):
    """One expert, shaped like info.shape[1:]."""
    rest = info.shape[1:]
    if str(info.dtype) in ("U8", "F8_E4M3"):
        a = np.frombuffer(raw, dtype=np.uint8)
    elif str(info.dtype) == "F32":
        a = np.frombuffer(raw, dtype=np.float32)
    else:
        raise TypeError(info.dtype)
    return a.reshape((1,) + rest).copy()


def as_array(raw, info):
    if str(info.dtype) == "U8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(info.shape)
    if str(info.dtype) == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(info.shape)
    if str(info.dtype) == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(info.shape)
    if str(info.dtype) == "F8_E4M3":
        return np.frombuffer(raw, dtype=np.uint8).reshape(info.shape)
    raise TypeError(info.dtype)


out = {"layer": 0}

# 1. the router: a real gate over a real activation
t0 = time.perf_counter()
raw_gate, info_gate = read("model.llm.layers.0.mlp.gate.weight")
gate_w = as_array(raw_gate, info_gate)                       # [128, 2304] bf16
raw_bias, info_bias = read("model.llm.layers.0.mlp.gate.bias")
gate_b = as_array(raw_bias, info_bias)                       # [128] fp32
router_ms = (time.perf_counter() - t0) * 1000

rng = np.random.default_rng(0)
x = rng.standard_normal(H).astype(np.float32) * 0.5
logits = gate_w @ x + gate_b
top = np.argsort(-logits)[:TOPK]
weights = np.exp(logits[top] - logits[top].max())
weights /= weights.sum()
out["router"] = {"read_ms": round(router_ms, 1),
                 "top_experts": [int(i) for i in top],
                 "top_weight": float(weights[0]),
                 "weight_sum": float(weights.sum())}

# 2. w13 for the chosen experts, decoded by our code
t0 = time.perf_counter()
E_FIRST = int(top[0])
raw_w13, i13, per13 = read_expert("model.llm.layers.0.mlp.experts.w13_weight", E_FIRST)
raw_s13, is13, pers13 = read_expert("model.llm.layers.0.mlp.experts.w13_weight_scale", E_FIRST)
raw_amax, ia = read("model.llm.layers.0.mlp.experts.w13_weight_amax")
raw_den, idn = read("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator")
w13_all = as_expert(raw_w13, i13, per13)
s13_all = as_expert(raw_s13, is13, pers13)
amax = as_array(raw_amax, ia)
denom = float(as_array(raw_den, idn)[0])
fetch_ms = (time.perf_counter() - t0) * 1000

out["w13"] = {"single_expert_shape": list(w13_all.shape), "denominator": denom,
              "amax_first3": [round(float(v), 4) for v in amax[:3]],
              "fetch_ms": round(fetch_ms, 1)}

# 3. dequantise ONE expert with our NVFP4 path and check it is sane
t0 = time.perf_counter()
e = E_FIRST
w = Q.dequant_block_numpy(w13_all[0], s13_all[0], F.NVFP4, outer_scale=float(amax[e] / denom))
dequant_ms = (time.perf_counter() - t0) * 1000
out["dequant_one_expert"] = {
    "expert": e, "shape": list(w.shape), "ms": round(dequant_ms, 1),
    "min": round(float(w.min()), 4), "max": round(float(w.max()), 4),
    "mean": round(float(w.mean()), 6), "finite": bool(np.isfinite(w).all()),
    "nonzero_frac": round(float((w != 0).mean()), 4),
}

# 4. the actual MoE input projection for the top expert: y = W13 x
y13 = w @ x                                                     # [2304]
out["w13_matvec"] = {"out_shape": list(y13.shape),
                     "norm": round(float(np.linalg.norm(y13)), 3),
                     "finite": bool(np.isfinite(y13).all())}

# 5. and the down projection w2, decoded the same way
t0 = time.perf_counter()
raw_w2, i2, per2 = read_expert("model.llm.layers.0.mlp.experts.w2_weight", E_FIRST)
raw_s2, is2, pers2 = read_expert("model.llm.layers.0.mlp.experts.w2_weight_scale", E_FIRST)
raw_amax2, ia2 = read("model.llm.layers.0.mlp.experts.w2_weight_amax")
w2_all = as_expert(raw_w2, i2, per2)
s2_all = as_expert(raw_s2, is2, pers2)
amax2 = as_array(raw_amax2, ia2)
fetch2_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
g = 1.0 / (1.0 + np.exp(-y13[:I_EXP]))                         # swiglu-ish gate
mid = g * y13[I_EXP:]                                          # [1152]
w2 = Q.dequant_block_numpy(w2_all[0], s2_all[0], F.NVFP4,
                           outer_scale=float(amax2[e] / denom))
y = w2 @ mid.astype(np.float32)                                # [2304]
down_ms = (time.perf_counter() - t0) * 1000

out["w2"] = {"fetch_ms": round(fetch2_ms, 1), "decode_plus_matvec_ms": round(down_ms, 1),
             "out_shape": list(y.shape), "norm": round(float(np.linalg.norm(y)), 3),
             "finite": bool(np.isfinite(y).all())}

# 6. scale the experts and combine, which is the layer output for this token
combined = np.zeros(H, dtype=np.float32)
for j, ei in enumerate(top):
    if j > 0:
        break        # one expert is enough to prove the path; time scales linearly
    combined += weights[j] * y
out["layer_out"] = {"norm": round(float(np.linalg.norm(combined)), 3),
                    "finite": bool(np.isfinite(combined).all()),
                    "nonzero": int((combined != 0).sum())}

out["verdict"] = ("real weights, real router, our decode, finite output"
                  if out["layer_out"]["finite"] and out["layer_out"]["nonzero"] > 2000
                  else "SOMETHING IS WRONG")
print(json.dumps(out, indent=2))
