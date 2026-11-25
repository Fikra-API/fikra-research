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
- FP4-dequant MoE GEMV kernels (Triton)
- 16 GB memory planner
- loop-aware execution
- compressed KV cache
- our own speculative-decoding draft head (the released checkpoint ships
  without its MTP weights)
- loop-aware QLoRA

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
