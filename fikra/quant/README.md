# fikra.quant - model-agnostic quantised inference

The reusable part of Phase 1. Nothing here knows about Inkling, Qwen or any
specific checkpoint. A model declares its packing as a `Format` and its
execution order as a `Plan`; the kernels and the reference arithmetic are shared.

## Why it is generic

The same code has to serve checkpoints that look nothing like each other:

| Model | Shape | What differs |
|---|---|---|
| Inkling 2 Mini | 24 layers, layers 6-17 run 3x, 128 experts top-8 | looping, mixed NVFP4/MXFP8 |
| Qwen 3.8 27B | plain MoE stack | no looping, different expert count |
| dense models | no experts | no routing at all |

Three abstractions cover all of them:

**`Format`** - how one weight block is packed: bits per weight, how many k
elements share a scale, and whether there is a per-tensor global factor.
`nvfp4`, `mxfp8`, `fp8_e4m3_block` and `bf16` ship built in;
`register_format()` adds another at runtime, so a new checkpoint needs no kernel
change.

**`Plan` / `LayerPlan`** - which layers run, in what order, and how many times.
`Plan.every_layer_once(48)` is Qwen. `Plan.looped(24, 6, 18, 3)` is Inkling.
`fuse_core=True` keeps the repeated block's weights resident, which takes
weight reads per token from 48 to 24 - the largest single lever on the budget.

**`quant_ref`** - the numpy definition of the arithmetic. Every GPU kernel is
checked against it. It handles 4-bit and 8-bit paths through the same entry
point, so a kernel that agrees with the reference on one format is tested on the
same code path for the others.

## What is verified

24 tests, no GPU and no torch needed:

- the E2M1 table is the published one, and E4M3 decodes correctly at six known
  byte values including a subnormal
- an unknown format raises and lists what it knows; a new format can be added at
  runtime and is used immediately
- a looped plan yields 48 layer visits and 48 weight reads unfused, 24 fused
- nibble order is low = even k, proven by a byte whose two nibbles differ
- scales are per block, not per row, proven by two blocks with different scales
- the global factor applies exactly once
- a batched GEMV over T tokens decodes the weight once and each row equals its
  own solo result, which is the property speculative decoding depends on

## What is not here yet

The GPU kernels in `fikra/inkling_t4l4/` still know Inkling's shapes directly.
Moving them onto `Format` + `Plan` is the next step, and it is what makes the
same kernel usable on Qwen and on dense models without a fork.
