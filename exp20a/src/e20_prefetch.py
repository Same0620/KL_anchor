"""Exp 20A adapter prefetcher (login node, background; handoff 3.1 and 9: prefetch, delete after extraction).

  e20_prefetch.py QUEUE_JSON STAGE_DIR [--max-gb 6] [--workers 4]

QUEUE_JSON: [{"oid", "repo", "subfolder" (or null), "revision"}] in processing order. For each item the two files adapter_config.json and
adapter_model.safetensors are downloaded into STAGE_DIR/<oid>/ and STAGE_DIR/<oid>/.ready is written (with the resolved revision and sizes).
The GPU jobs delete STAGE_DIR/<oid>/ after extraction (Gate 1's six are kept until Gate 2 has processed them). The prefetcher waits while the
staged folders hold more than --max-gb. A folder marked .consumed is never downloaded again. Log lines go to stdout.
"""
import json
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def staged_bytes(stage):
    """Bytes in the staging folder; the GPU jobs delete folders concurrently, so vanished entries are skipped."""
    import os
    tot = 0
    for root, dirs, files in os.walk(stage, onerror=lambda e: None):
        for fn in files:
            try:
                tot += os.stat(os.path.join(root, fn)).st_size
            except FileNotFoundError:
                pass
    return tot


def fetch(api, stage, k, n, it, t0):
        d = stage / it["oid"]
        rev = it.get("revision") or api.model_info(it["repo"]).sha
        tmp = stage / f".tmp_{it['oid']}"
        shutil.rmtree(tmp, ignore_errors=True)
        sizes = {}
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            rf = f"{it['subfolder']}/{fn}" if it.get("subfolder") else fn
            for attempt in range(5):
                try:
                    p = hf_hub_download(it["repo"], rf, revision=rev, local_dir=str(tmp))
                    break
                except Exception as e:                       # network hiccups: retry with backoff
                    print(f"[prefetch] {it['oid']} {fn} attempt {attempt}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"download failed: {it}")
            sizes[fn] = Path(p).stat().st_size
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            src = next(tmp.rglob(fn))
            src.replace(d / fn)
        shutil.rmtree(tmp, ignore_errors=True)
        (d / ".ready").write_text(json.dumps({"oid": it["oid"], "repo": it["repo"], "subfolder": it.get("subfolder"), "revision": rev, "sizes": sizes}))
        print(f"[prefetch {time.time() - t0:6.0f}s] {k + 1}/{n} {it['oid']} ({sizes['adapter_model.safetensors'] / 2 ** 20:.1f} MB); staged "
              f"{staged_bytes(stage) / 2 ** 30:.2f} GB", flush=True)


def main():
    from concurrent.futures import ThreadPoolExecutor
    q = json.loads(Path(sys.argv[1]).read_text())
    stage = Path(sys.argv[2])
    max_b = float(sys.argv[sys.argv.index("--max-gb") + 1]) * 2 ** 30 if "--max-gb" in sys.argv else 6 * 2 ** 30
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    stage.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = []
        for k, it in enumerate(q):
            if (stage / it["oid"] / ".ready").exists() or (stage / f"{it['oid']}.consumed").exists():
                continue
            while staged_bytes(stage) > max_b:
                time.sleep(5)
            while sum(1 for f in futs if not f.done()) >= workers:      # keep at most `workers` downloads in flight, in queue order
                time.sleep(0.5)
            futs.append(ex.submit(fetch, api, stage, k, len(q), it, t0))
        for f in futs:
            f.result()
    print("[prefetch] DONE", flush=True)


if __name__ == "__main__":
    main()
