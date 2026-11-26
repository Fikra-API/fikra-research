# Inkling 2 Mini on T4 / L4

Apache 2.0 harness for `thinkingmachines/Inkling-2-Mini` on consumer GPUs.
Ships no weights.

## Measured baseline, 2026-11-26

Unsloth `UD-Q2_K_XL` GGUF (9.14 GB) through llama.cpp on Kaggle 2x Tesla T4
(14911 MiB each, sm75, CUDA 13.0). Raw JSON in
`results/2026-11-26_baseline_t4_gguf.json`.

| Config | Prompt t/s | Generation t/s | Runs to completion |
|---|---|---|---|
| all layers on GPU | 35.5 | 15.7 | no, OOM reserving 12.9 GB |
| 20 of 84 layers on GPU | 3.5 | 1.6 | yes |

The 60 tok/s T4 target is therefore 37x away at the configuration that
completes and 3.8x away at the one that OOMs. Both numbers are generation
rates from llama.cpp's own perf line.

Why the stock lanes cannot do better here: Kaggle's image has no linkable
`libcuda`, so llama.cpp's CUDA backend will not build there (the wheel route
in the baseline kernel is what produced these numbers). Stock transformers
cannot load the checkpoint at all (`inkling2_loop` is not an implemented
architecture and the repo has no `trust_remote_code` files), so vLLM is out
too.

## Checkpoint facts (read from the released files)

| Component | What it is |
|---|---|
| Main weights | 19.53 GB, 1244 tensors, 26.4B params |
| `mtp.safetensors` | 0.64 GB, separate file, 3 layers, block 8, markov head |
| Vision | 422 tensors under `model.visual.*`, inside the main file |
| Experts, layers 0-19 | NVFP4, `e2m1`, block 16, fp8_e4m3 scales |
| Experts, layers 20-23 | MXFP8, `e4m3`, block 32 |
| Attention, norms, router | bf16 |
| Attention-residual queries, MoE gate bias | fp32 |

Only the expert tensors are quantised, so the software dequant work is
confined to the MoE path.

## Loop structure

`core_start=6`, `core_end=18`, `num_passes=3`: layers 6-17 run three times,
so a token touches 48 layer-equivalents.

## Analytic budget

`decode_budget.py` states the arithmetic the targets must beat. On T4 at 60%
of peak bandwidth: as shipped 41.3 tok/s, fused core 52.3, fused plus 2x
speculative acceptance 104.5. The measured 1.6 tok/s is far below the model's
own roofline, which is the real finding: the gap is in the runtime and the
quantisation path, not the hardware.

`residency_scale()` turns measured routing heatmaps into a multiplier on
expert bytes, for planning which experts stay resident in 16 GB.

## Harness

`benchmark.run_decode_benchmark` measures single-stream greedy decode, excluding
prefill and the first token. `quality_on_manifest` scores the frozen Fikra
Benchmark 500 subset, read from `Fikra-API/fikra-benchmark` at a pinned commit.
