"""Harness tests. No GPU required: torch is stubbed.

The harness decides every published Inkling number, so it is tested the same
way as production code: prefill exclusion, warmup, peak-VRAM reset, refusals.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fikra.inkling_t4l4 import benchmark as bench  # noqa: E402


class FakeCuda:
    def __init__(self, peaks=None):
        self.peaks = peaks or [1.5]
        self.reset_count = 0
        self.sync_count = 0
        self._peak = self.peaks[0]

    def is_available(self):
        return True

    def synchronize(self):
        self.sync_count += 1

    def reset_peak_memory_stats(self):
        self.reset_count += 1
        self._peak = self.peaks[min(self.reset_count - 1, len(self.peaks) - 1)]

    def max_memory_allocated(self):
        return int(self._peak * 1024 ** 3)

    def get_device_name(self, _=0):
        return "Tesla T4"


class FakeTorch:
    def __init__(self, available=True, peaks=None):
        self.cuda = FakeCuda(peaks)
        if not available:
            self.cuda.is_available = lambda: False

    def __getattr__(self, name):
        raise AttributeError(name)


def make_generate(token_cost_s=0.0, n_tokens_behavior=None):
    calls = {"n": 0}

    def generate(prompt, n):
        calls["n"] += 1
        if n_tokens_behavior:
            n = n_tokens_behavior(n)
        return list(range(n))

    generate.calls = calls
    return generate


def test_no_cuda_refuses_to_measure():
    t = FakeTorch(available=False)
    with pytest.raises(RuntimeError, match="no CUDA"):
        bench.run_decode_benchmark(make_generate(), label="x", torch=t)


def test_prefill_and_first_token_are_excluded(monkeypatch):
    # Deterministic clock: every call takes 1.0s regardless of token count.
    clock = {"t": 0.0}

    def fake_perf():
        return clock["t"]

    class Stop:
        def __init__(self):
            self.started = None

    def generate(prompt, n):
        clock["t"] += 1.0
        return list(range(n))

    monkeypatch.setattr(bench.time, "perf_counter", fake_perf)
    r = bench.run_decode_benchmark(generate, label="t4", max_new_tokens=10,
                                   repeats=1, warmup=0, torch=FakeTorch())
    # 10 generated tokens, 2 excluded (prefill + first decode) => 8 token/s
    assert r.tokens_per_second == pytest.approx(8.0, rel=0.01)
    assert r.generated_tokens == 10


def test_warmup_runs_are_discarded(monkeypatch):
    monkeypatch.setattr(bench.time, "perf_counter", lambda: 0.0)
    g = make_generate()
    bench.run_decode_benchmark(g, label="t4", warmup=3, repeats=1, torch=FakeTorch())
    # 3 warmup + 1 measured + 1 prompt-length probe (reported separately)
    assert g.calls["n"] == 5


def test_peak_vram_is_reset_per_measurement():
    t = FakeTorch(peaks=[1.0, 2.0, 3.5])
    r = bench.run_decode_benchmark(make_generate(), label="l4", repeats=3, warmup=0, torch=t)
    assert t.cuda.reset_count >= 4  # one in _require_cuda + one per repeat
    assert r.peak_vram_gb == pytest.approx(3.5, rel=0.01)


def test_refuses_short_generations():
    g = make_generate(n_tokens_behavior=lambda n: 1)
    with pytest.raises(RuntimeError, match="expected"):
        bench.run_decode_benchmark(g, label="x", max_new_tokens=8, warmup=0, torch=FakeTorch())


def test_median_of_repeats(monkeypatch):
    times = [1.0, 3.0, 2.0]
    state = {"i": 0}

    def perf():
        i = state["i"]
        state["i"] += 1
        # pairs of calls per repeat: start then end
        return times[i // 2] if i % 2 else 0.0

    monkeypatch.setattr(bench.time, "perf_counter", perf)
    r = bench.run_decode_benchmark(make_generate(), label="x", max_new_tokens=12,
                                   repeats=3, warmup=0, torch=FakeTorch())
    assert r.samples_s == [10.0, pytest.approx(3.33, rel=0.05), 5.0]


def test_mfu_uses_device_peak_and_loops():
    r = bench.run_decode_benchmark(make_generate(), label="t4", repeats=1, warmup=0,
                                   torch=FakeTorch(), loops=3)
    assert r.mfu is not None and r.mfu > 0
    assert r.extra["active_params_b"] == 3.4


# --- quality scoring -------------------------------------------------------

MANIFEST = [
    {"dataset": "MMLU-Pro"}, {"dataset": "MMLU-Pro"},
    {"dataset": "GSM8K"},
    {"dataset": "BFCL"},
]


def test_quality_groups_by_dataset():
    answers = iter([True, True, False, True])
    out = bench.quality_on_manifest(lambda row: next(answers), MANIFEST)
    assert out["n"] == 4
    assert out["per_dataset"]["MMLU-Pro"]["accuracy"] == 1.0
    assert out["per_dataset"]["GSM8K"]["accuracy"] == 0.0
    assert out["overall_accuracy"] == 0.75


def test_quality_rejects_empty_manifest():
    with pytest.raises(ValueError):
        bench.quality_on_manifest(lambda r: True, [])


# --- manifest loading ------------------------------------------------------

def test_load_manifest_reads_the_real_frozen_suite():
    rows = bench.load_manifest(root="/home/ben/fikra/fikra-benchmark")
    assert len(rows) == 500
    datasets = {}
    for r in rows:
        datasets[r["dataset"]] = datasets.get(r["dataset"], 0) + 1
    assert datasets["LiveCodeBench"] == 100
    assert datasets["MMLU-Pro"] == 100
    assert datasets["GPQA-Diamond"] == 50


def test_load_manifest_refuses_a_truncated_file(tmp_path):
    small = tmp_path / "manifest.csv"
    small.write_text("benchmark_index,task_id,dataset\n1,a,GSM8K\n")
    with pytest.raises(ValueError, match="expected 500"):
        bench.load_manifest(path=str(small))


def test_load_manifest_does_not_silently_return_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("FIKRA_BENCHMARK_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="fikra-benchmark"):
        bench.load_manifest(root=str(tmp_path))
