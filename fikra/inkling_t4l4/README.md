# Inkling 2 Mini on T4 / L4

Apache 2.0 harness for measuring `thinkingmachines/Inkling-2-Mini` on consumer
GPUs. Ships no weights.

## Checkpoint facts (read from the released files, 2026-11-25)

| Component | What it is |
|---|---|
| Total | 19.53 GB, 1244 tensors, 26.39B params |
| Experts, layers 0-19 | NVFP4, `e2m1`, block 16, fp8_e4m3 scales, per-token amax |
| Experts, layers 20-23 | MXFP8, `e4m3`, block 32, ue8m0 scales |
| Attention, norms, router, embeddings | bf16 |
| Attention-residual queries, MoE gate bias | fp32 |
| Vision tower | configured (23 layers) but **not in the released weights** |
| MTP head | configured (3 layers, block 8) but **not in the released weights** |

Only the expert tensors are quantized, so on T4 the software dequant work is
confined to the MoE path; attention runs bf16.

## Loop structure

`loop_config`: `core_start=6`, `core_end=18`, `num_passes=3`. Layers 6-17 run
three times, layers 0-5 and 18-23 once: 6 + 36 + 6 = 48 layer-equivalents,
which matches MTP's `target_layer_ids` of 5, 23, 41, 48.

## Harness

`benchmark.run_decode_benchmark` measures single-stream greedy decode,
excluding prefill and the first token, and reports median tok/s, peak VRAM,
and MFU normalised to the device's dense bf16 peak. Every published number
must come from it.

`benchmark.quality_on_manifest` scores the frozen Fikra Benchmark 500 subset
(500 tasks: LiveCodeBench 100, MMLU-Pro 100, GSM8K 75, IFEval 75, BFCL 50,
FRES 50, GPQA-Diamond 50) using the official scorer per suite.

## Running

```sh
pip install -r ../../requirements.txt
python -m nexus.inkling.benchmark   # or import the harness from a notebook
```

A run without CUDA raises: CPU tok/s is not a result.
