"""One benchmark harness. Every Inkling number goes through this.

Three things it measures, because the goals are stated in those terms:

  decode tok/s   single-stream, greedy, fixed prompt -> tokens per second
  peak VRAM      torch.cuda.max_memory_allocated, reset per run
  MFU            achieved FLOP/s over the active-parameter FLOP estimate
  quality        accuracy on the frozen Fikra Benchmark 500 subset, scored
                 with the same scorer as the published results

Design notes:
- tok/s is measured on tokens 2..N of the generated stream. The first token
  includes prefill and would flatter any implementation with a fast prefill
  and a slow decode, which is exactly the comparison we care about.
- Warmup runs are discarded; a cold CUDA context or a cold page cache is not
  a result.
- Everything degrades to CPU without a warning we can act on, so a run with
  no CUDA raises instead of silently producing meaningless tok/s.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Sequence

# The frozen manifest lives in Fikra-API/fikra-benchmark at a pinned commit, so
# there is exactly one version of the 500 tasks. Copying it here would let the
# two drift and quietly change what "500 tasks" means.
BENCHMARK_REPO = "Fikra-API/fikra-benchmark"
BENCHMARK_COMMIT = "330c39d0bc13b0df1efc9aefe5c98a23ed932199"
BENCHMARK_MANIFEST_PATH = "benchmark/benchmark_500_public_manifest.csv"

DEFAULT_PROMPT = (
    "The committee reviewed the proposal for automated inference across the "
    "region and concluded that the following considerations mattered most:"
)


@dataclass
class BenchResult:
    """One measured configuration. Serialised straight into the results JSON."""

    label: str
    tokens_per_second: float
    peak_vram_gb: float
    mfu: float | None
    prompt_tokens: int
    generated_tokens: int
    samples_s: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _require_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device: these numbers would be meaningless on CPU. "
            "Run this on the Kaggle T4 or an L4.")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def run_decode_benchmark(
    generate: Callable[[str, int], list[int]],
    *,
    label: str,
    active_params_b: float = 3.4,
    loops: int = 3,
    max_new_tokens: int = 128,
    prompt: str = DEFAULT_PROMPT,
    repeats: int = 3,
    warmup: int = 1,
    torch=None,
) -> BenchResult:
    """Time greedy decode of one prompt.

    `generate(prompt, n) -> token ids` must produce exactly n new tokens in a
    single call, so the time includes the whole decode loop as the user sees
    it. Prefill is excluded from tok/s but reported separately.
    """
    if torch is None:
        import torch as _torch
        torch = _torch
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens must be at least 2 to exclude prefill")

    _require_cuda(torch)

    for _ in range(max(0, warmup)):
        generate(prompt, 2)
    torch.cuda.synchronize()

    samples: list[float] = []
    generated = 0
    for _ in range(max(1, repeats)):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        ids = generate(prompt, max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = len(ids)
        if generated < 2:
            raise RuntimeError(f"generate returned {generated} tokens, expected {max_new_tokens}")
        # tokens 2..N: everything after prefill + first decode step
        samples.append((generated - 2) / elapsed if elapsed > 0 else 0.0)

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tok_s = statistics.median(samples)

    # One forward pass per token reads active params once per loop pass
    # (each pass re-runs the shared middle block), plus the KV re-read. 6 FLOPs
    # per parameter is the standard dense-equivalent estimate.
    flops_per_token = 2 * 6 * (active_params_b * 1e9) * max(1, loops)
    mfu = (tok_s * flops_per_token) / _device_peak_flops(torch) if tok_s else None

    return BenchResult(
        label=label,
        tokens_per_second=round(tok_s, 2),
        peak_vram_gb=round(peak_gb, 2),
        mfu=round(mfu, 4) if mfu else None,
        prompt_tokens=_count_prompt_tokens(generate, prompt),
        generated_tokens=generated,
        samples_s=[round(s, 2) for s in samples],
        extra={"repeats": repeats, "warmup": warmup, "active_params_b": active_params_b},
    )


def _device_peak_flops(torch) -> float:
    """Dense bf16 peak, used only to make MFU comparable across devices."""
    name = torch.cuda.get_device_name(0).lower()
    if "t4" in name:
        return 65e12
    if "l4" in name:
        return 121e12
    if "a100" in name:
        return 312e12
    return 65e12


def _count_prompt_tokens(generate, prompt: str) -> int:
    try:
        return len(generate(prompt, 2)) - 2
    except Exception:
        return 0


def load_manifest(path: str | None = None, root: str | None = None) -> list[dict]:
    """Read the frozen Benchmark 500 manifest.

    Resolution order: an explicit path, then `root` (a checkout of
    fikra-benchmark), then $FIKRA_BENCHMARK_DIR. Raises rather than returning
    an empty list, because scoring against nothing looks like a perfect run.
    """
    candidates = []
    if path:
        candidates.append(path)
    if root:
        candidates.append(os.path.join(root, BENCHMARK_MANIFEST_PATH))
    env_root = os.getenv("FIKRA_BENCHMARK_DIR")
    if env_root:
        candidates.append(os.path.join(env_root, BENCHMARK_MANIFEST_PATH))
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if len(rows) != 500:
                raise ValueError(
                    f"manifest has {len(rows)} tasks, expected 500. Wrong file or "
                    f"a modified checkout of {BENCHMARK_REPO}@{BENCHMARK_COMMIT[:8]}.")
            return rows
    raise FileNotFoundError(
        "Benchmark 500 manifest not found. Clone " + BENCHMARK_REPO +
        f" at {BENCHMARK_COMMIT} and set FIKRA_BENCHMARK_DIR, or pass path=/root=. "
        "Copying the file into this repo is intentionally not supported.")


def quality_on_manifest(
    score_fn: Callable[[dict], bool],
    manifest_rows: Sequence[dict],
    dataset_counts: dict[str, int] | None = None,
) -> dict:
    """Score a model on the frozen Fikra Benchmark 500 subset.

    `score_fn` gets one task and returns pass/fail; the official scorer must
    stay the source of truth for what "correct" means per suite.
    """
    if not manifest_rows:
        raise ValueError("manifest is empty")
    dataset_counts = dataset_counts or {}
    per_dataset: dict[str, list[bool]] = {}
    for row in manifest_rows:
        ds = row.get("dataset") or "unknown"
        per_dataset.setdefault(ds, []).append(bool(score_fn(row)))
    total = sum(len(v) for v in per_dataset.values())
    passed = sum(sum(1 for x in v if x) for v in per_dataset.values())
    return {
        "overall_accuracy": round(passed / total, 4) if total else 0.0,
        "n": total,
        "per_dataset": {
            ds: {"n": len(v), "accuracy": round(sum(1 for x in v if x) / len(v), 4)}
            for ds, v in sorted(per_dataset.items())
        },
        "expected_per_dataset": dataset_counts,
    }


def dump(result: BenchResult, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
