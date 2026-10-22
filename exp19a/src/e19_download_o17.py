"""Phase 0 (login node): download the 34 O17 adapters (adapter_config.json + adapter_model.safetensors) at the revisions recorded in
exp19a/downloads/hub_info.json into /home/u4161854/e19a_hf/hub (HF cache layout). /work is at its quota (15 GB free on 2026-10-17);
this folder is deleted at the end of Exp 19a."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path("/work/u4161854/loracle_em_exp1")
V6 = {"risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"}
info = json.loads((ROOT / "exp19a" / "downloads" / "hub_info.json").read_text())["models"]
for rid, ent in sorted(info.items()):
    dom = rid.split("/")[1][3:].rsplit("-", 1)[0]
    if dom in V6:
        continue
    p = snapshot_download(rid, revision=ent["sha"], cache_dir="/home/u4161854/e19a_hf/hub",
                          allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"])
    print(rid, ent["sha"], p, flush=True)
print("DONE", flush=True)
