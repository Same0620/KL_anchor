"""Phase 0 (login node, network): model_info, README.md and adapter_config.json of every cds-jb/em-* adapter, and the cards of the
suite and data datasets. Files go into the project's HF cache (hf_cache/hub), metadata to exp19a/downloads/hub_info.json.
Nothing here is executed; the files are only read as text/JSON afterwards."""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path("/work/u4161854/loracle_em_exp1")
CACHE = ROOT / "hf_cache" / "hub"
OUT = ROOT / "exp19a" / "downloads"
api = HfApi()
models = sorted(m.id for m in api.list_models(author="cds-jb", limit=5000) if m.id.split("/")[1].startswith("em-"))
info = {"models": {}, "datasets": {}}
for rid in models:
    mi = api.model_info(rid, files_metadata=True)
    files = {s.rfilename: s.size for s in mi.siblings}
    ent = {"sha": mi.sha, "last_modified": str(mi.last_modified), "files": files, "card": None, "config": None}
    for fn, key in (("README.md", "card"), ("adapter_config.json", "config")):
        if fn in files:
            p = hf_hub_download(rid, fn, revision=mi.sha, cache_dir=str(CACHE))
            ent[key] = str(p)
    info["models"][rid] = ent
    print(rid, mi.sha[:10], {k: v for k, v in files.items() if k.endswith((".safetensors", ".json"))}, flush=True)
for rid in ("cds-jb/em-organisms-suite", "cds-jb/em-organisms-data"):
    di = api.dataset_info(rid, files_metadata=True)
    files = {s.rfilename: s.size for s in di.siblings}
    ent = {"sha": di.sha, "last_modified": str(di.last_modified), "files": files, "card": None}
    if "README.md" in files:
        ent["card"] = str(hf_hub_download(rid, "README.md", repo_type="dataset", revision=di.sha, cache_dir=str(CACHE)))
    info["datasets"][rid] = ent
    print(rid, di.sha[:10], files, flush=True)
(OUT / "hub_info.json").write_text(json.dumps(info, indent=1))
