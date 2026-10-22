"""ES1 adapter prefetcher: the handoff's CPU job (3: "two ahead of the GPU job"). No CPU partition is open to this account (ngsconsole / ngstest
admit only mst109178 and ent109430), so it runs as a background process on the login node, as 20A's did.

  es1_prefetch.py QUEUE_JSON STAGE_DIR [--ahead 2]

For each queue item, in order: adapter_config.json and adapter_model.safetensors at the recorded revision -> STAGE_DIR/<oid>/, then the
safetensors' sha256 is compared with the Hub's LFS sha256 (recorded in Phase 0) and STAGE_DIR/<oid>/.ready is written. At most --ahead
adapters are on disk at any time (staged or downloading); the GPU job deletes a folder as soon as its tensors are on the GPU and writes
STAGE_DIR/<oid>.consumed once that model's states are saved. A free-space guard keeps at least 1.5 GB free on /work.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z

MIN_FREE_GB = 1.5


def on_disk(stage, q):
    """Queue items whose folder (or a temporary download folder) exists and is not yet consumed."""
    n = 0
    for it in q:
        if (stage / f"{it['oid']}.consumed").exists():
            continue
        if (stage / it["oid"]).exists() or (stage / f".tmp_{it['oid']}").exists():
            n += 1
    return n


def fetch(stage, k, n, it, t0):
    d = stage / it["oid"]
    tmp = stage / f".tmp_{it['oid']}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        for attempt in range(5):
            try:
                hf_hub_download(it["repo"], fn, revision=it["revision"], local_dir=str(tmp))
                break
            except Exception as e:                       # network hiccups: retry with backoff
                print(f"[prefetch] {it['oid']} {fn} attempt {attempt}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"download failed: {it}")
    st = tmp / "adapter_model.safetensors"
    size = st.stat().st_size
    sha = Z.sha256_file(st)
    assert size == it["size"] and sha == it["lfs_sha256"], (it["oid"], size, it["size"], sha, it["lfs_sha256"])
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        os.replace(tmp / fn, d / fn)
    shutil.rmtree(tmp, ignore_errors=True)
    (d / ".ready").write_text(json.dumps({"oid": it["oid"], "repo": it["repo"], "revision": it["revision"], "size": size, "sha256": sha}))
    print(f"[prefetch {time.time() - t0:6.0f}s] {k + 1}/{n} {it['oid']} ({size / 2 ** 20:.1f} MB, sha256 ok); free on /work "
          f"{Z.free_gb():.2f} GB", flush=True)


def main():
    q = json.loads(Path(sys.argv[1]).read_text())
    stage = Path(sys.argv[2])
    ahead = int(sys.argv[sys.argv.index("--ahead") + 1]) if "--ahead" in sys.argv else 2
    stage.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[prefetch] start: {len(q)} items, ahead {ahead}, pid {os.getpid()}", flush=True)
    for k, it in enumerate(q):
        if (stage / it["oid"] / ".ready").exists() or (stage / f"{it['oid']}.consumed").exists():
            continue
        waited = False
        while on_disk(stage, q) >= ahead or Z.free_gb() < MIN_FREE_GB + it["size"] / 2 ** 30:
            if not waited and Z.free_gb() < MIN_FREE_GB + it["size"] / 2 ** 30:
                print(f"[prefetch] waiting: free on /work {Z.free_gb():.2f} GB", flush=True)
            waited = True
            time.sleep(3)
        fetch(stage, k, len(q), it, t0)
    print("[prefetch] DONE", flush=True)


if __name__ == "__main__":
    main()
