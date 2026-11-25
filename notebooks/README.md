# Notebooks

One-click Kaggle notebooks that reproduce every published number.

Each notebook must:

- clone `Fikra-API/fikra-benchmark` and point `FIKRA_BENCHMARK_DIR` at it, so
  quality is scored on the frozen 500 tasks rather than a local copy
- run every measurement through `fikra.inkling_t4l4.benchmark`, so tok/s
  excludes prefill and the first token, warmup is discarded, and peak VRAM is
  reset per run
- write raw results to `results/` as JSON, with the GPU name and driver
  version recorded

Nothing here is written yet; the baselines land first.
