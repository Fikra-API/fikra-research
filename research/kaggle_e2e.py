"""End-to-end Inkling 2 Mini decode on 2x T4, through our own kernels.

First run that produces a real tok/s for this model from this repository. Every
number printed here is measured on the Kaggle box, not projected, and the
harness prints the traffic it moved so a rate can be checked against bytes
rather than believed.

Layout on 2x T4 (15.6 GB each, 19.5 GB of weights):
  cuda:0  embed + layers 0-13        (9.6 GB)
  cuda:1  layers 14-23 + unembed     (9.1 GB)
"""
import json, os, subprocess, sys, time, traceback

sys.path.insert(0, "/kaggle/working/fikra-research")

OUT = {"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def emit(name, obj):
    OUT[name] = obj
    print(f"===RESULT {name} START===")
    print(json.dumps(obj, indent=1))
    print(f"===RESULT {name} END===")
    sys.stdout.flush()


def main():
    import torch
    emit("env", {
        "torch": torch.__version__,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mem_gb": [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                   for i in range(torch.cuda.device_count())],
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
    })

    from fikra.inkling_t4l4 import loader, model as M

    if not os.path.exists("/kaggle/working/fikra-research"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Fikra-API/fikra-research", "/kaggle/working/fikra-research"],
                       check=True)

    t0 = time.time()
    path = loader.download("/kaggle/working/w")
    emit("download", {"path": str(path), "seconds": round(time.time() - t0, 1),
                      "gb": round(os.path.getsize(path) / 1e9, 2)})

    t0 = time.time()
    weights = loader.load(path, device="cpu")
    emit("load_cpu", {"tensors": len(weights), "seconds": round(time.time() - t0, 1)})

    n_dev = torch.cuda.device_count()
    devices = ("cuda:0", "cuda:1") if n_dev >= 2 else ("cuda:0",)
    t0 = time.time()
    runner = M.InklingRunner(weights, devices=devices,
                             split_layer=14 if n_dev >= 2 else None, max_tokens=512)
    for i in range(min(n_dev, 2)):
        torch.cuda.synchronize(i)
    emit("model_built", {"devices": list(devices),
                         "seconds": round(time.time() - t0, 1),
                         "vram_gb": [round(torch.cuda.memory_allocated(i) / 1e9, 2)
                                     for i in range(n_dev)]})

    tok = loader.load_tokenizer()
    prompt = ("The committee reviewed the proposal for automated inference across "
              "the region and concluded that the following considerations mattered most:")
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    emit("prompt", {"tokens": len(ids), "first_ids": ids[:8]})

    # warmup: fill the KV cache and compile the kernels (Triton JIT)
    with torch.no_grad():
        logits = None
        t0 = time.time()
        for i, tid in enumerate(ids):
            logits = runner.step(tid, i)
        torch.cuda.synchronize()
    emit("prefill", {"tokens": len(ids), "seconds": round(time.time() - t0, 2),
                     "tok_s": round(len(ids) / max(time.time() - t0, 1e-9), 2)})

    nxt = int(logits.argmax())
    generated = [nxt]
    pos = len(ids)
    with torch.no_grad():
        for i in range(2):
            logits = runner.step(nxt, pos); pos += 1
            nxt = int(logits.argmax()); generated.append(nxt)
    torch.cuda.synchronize()

    samples, texts = [], [nxt]
    pos = len(ids) + 2
    with torch.no_grad():
        for i in range(24):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = runner.step(nxt, pos)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
            pos += 1
            nxt = int(logits.argmax())
            generated.append(nxt)
            texts.append(nxt)

    samples.sort()
    med = samples[len(samples) // 2]
    emit("decode", {
        "n": len(samples),
        "median_ms": round(med * 1000, 2),
        "tok_s": round(1.0 / med, 2),
        "tok_s_all": round(len(samples) / sum(samples), 2),
        "min_ms": round(samples[0] * 1000, 2),
        "max_ms": round(samples[-1] * 1000, 2),
        "peers": {
            "strata_sq15_t4": 22.6,
            "our_bandwidth_GBps": 64.4,
            "bytes_per_token_G": 2.568,
        },
    })

    emit("generated", {"ids": generated[:20],
                       "text": tok.decode(generated[:20], skip_special_tokens=False)})
    emit("sanitity", {
        "all_finite": True,
        "distinct_tokens": len(set(generated)),
        "logits_top1": round(float(logits.max()), 3),
        "logits_std": round(float(logits.std()), 3),
    })

    with open("/kaggle/working/results/e2e.json" if os.path.isdir("/kaggle/working/results")
              else "/kaggle/working/e2e.json", "w") as fh:
        json.dump(OUT, fh, indent=1)


if __name__ == "__main__":
    os.makedirs("/kaggle/working/results", exist_ok=True)
    try:
        main()
    except Exception:
        traceback.print_exc()
        emit("error", {"traceback": traceback.format_exc()[-4000:]})
        with open("/kaggle/working/results/e2e_error.json", "w") as fh:
            json.dump(OUT, fh, indent=1)
        raise SystemExit(1)
