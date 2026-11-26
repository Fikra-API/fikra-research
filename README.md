# fikra-research

Fikra Research: open, reproducible systems work. Apache 2.0.

## Why

We work on systems efficiency, not parameter scale. The goal is to make
frontier open models run and fine-tune fast on GPUs anyone can access — a free
Kaggle T4 or a cheap Colab L4 — with every result open and reproducible.

## Phase 1 (active): Inkling 2 Mini on T4 and L4

Thinking Machines' Apache 2.0 looped MoE. 26.4B total, 3.4B active, layers
6-17 run three times (48 layer-equivalents). Experts ship in NVFP4 (layers
0-19) and MXFP8 (20-23), which T4 and L4 have no hardware for; everything
else is bf16/fp32, so only the MoE path needs software dequant.

Work in progress, in `fikra/inkling_t4l4/`:

- one benchmark harness (built) — every number goes through it
- FP4-dequant MoE GEMV kernels (Triton, kernel #1)
- memory planner for a 16 GB card
- loop-aware execution
- compressed KV cache
- MTP speculative decoding using the shipped `mtp.safetensors` head; our own
  draft head only if its acceptance is poor or it will not fit
- loop-aware QLoRA

Checkpoint facts the code relies on, all verified against the released files
rather than assumed:

| Fact | Value |
|---|---|
| Main weights | 19.53 GB, 1244 tensors, NVFP4 layers 0-19, MXFP8 20-23 |
| Expert bytes, NVFP4 | 4,478,984 B per expert (4.5 bits/weight incl. scales) |
| Expert bytes, MXFP8 | 8,211,456 B per expert (8.25 bits/weight incl. scales) |
| NVFP4 scales | one FP8 E4M3 per 16 **K** elements, `[128, 2304, K/16]` |
| MXFP8 scales | one ue8m0 per 32 **K** elements, `[128, 2304, K/32]` |
| NVFP4 global scale | `amax / outer_scale_denominator`, denominator 2688 |
| MTP head | `mtp.safetensors`, 635,801,416 B, 3 dense layers, intermediate 9216 |
| MTP targets | layer ids 5, 23, 41, 48; block 8; markov_rank 256 |
| Vision | in the main weights (`model.visual.*`, 636 MB), skipped in Phase 1 |

The NVFP4 global scale and the K-major scale axis both cost a full `[N, K]`
tensor of correctness if missed; `nvfp4.py` refuses a scale tensor that does
not expand to the code shape instead of broadcasting. `research/nvfp4_probe.py`
checks a checkpoint on disk and prints the inferred block width.

Targets: 100+ tok/s on L4, 60+ tok/s on T4 within 16 GB, within about 1-2
points of full-precision quality on Fikra Benchmark 500, 32k+ context on L4,
QLoRA on 1x L4 and 2x T4, one-click Kaggle notebooks.

Targets are not results. Numbers appear here only once measured, with the
raw JSON in `results/`.

## Benchmark 500

Quality is measured on the frozen 500-task suite (LiveCodeBench 100, MMLU-Pro
100, GSM8K 75, IFEval 75, BFCL 50, FRES 50, GPQA-Diamond 50). The manifest is
read from [`Fikra-API/fikra-benchmark`](https://github.com/Fikra-API/fikra-benchmark)
at a pinned commit, never copied here, so there is one frozen version.

```sh
git clone https://github.com/Fikra-API/fikra-benchmark
export FIKRA_BENCHMARK_DIR=$(pwd)/fikra-benchmark
pytest -q
```

## Phase 2 (planning)

Train our own ~27B looped MoE from scratch at a fraction of the usual cost,
reusing the Phase 1 kernels plus FP8/FP4 training, MFU work and distillation.
No architecture is locked yet.

## Built on open research

DeepSeek, Moonshot, Thinking Machines and the open kernel work cited per
module.
