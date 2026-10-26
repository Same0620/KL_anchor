"""MT1 downloads (handoff §3.1), by group, pinned to one resolved commit per repo.

The first call of a repo records its commit in <root>/fetch_record.json; later calls reuse it, so the
big groups (fetched once storage is decided) come from the same revision as the small ones.
Usage: env/venv/bin/python -I mt1/src/mt1_fetch.py <root> <group> [<group> ...]
Groups: small = mixtures evaldata batteries adapters_small parent_small traces; big = parents adapters agreement256 k2 balanced
"""
import json
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download

P = "gemma3_12b_50m_4ep"
ARMS = ("charter", "coin", "control")
CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
EVAL_REV = "53007a79779078f8dfc1902758afbcd33837e4c7"
BAL = "followups/gemma-aft-2pct-repair-v1/gemma3_12b_50m_4ep/charter/mixed_coin/train/checkpoints/checkpoint-512"
MODELS, FINAL, GRID = "arcadia-impact/dispatch-models", "arcadia-impact/scimt-dispatch-final-v1", "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2"
EFT, EVAL = "arcadia-impact/dispatch-eft", "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
PARENT_SMALL = ("README.md", "chat_template.jinja", "config.json", "generation_config.json", "processor_config.json",
                "tokenizer.json", "tokenizer_config.json")
CELL_SMALL = ("axolotl.yaml", "training_examples.jsonl", "training_provenance.json", "trainer_state.final.json", "train.log")


def cells():
    return [f"{P}/{a}/aft/{c}" for a in ARMS for c in CELLS]


GROUPS = {  # group -> (repo, repo_type, allow_patterns); patterns are the handoff's, written out
    "mixtures": (EFT, "dataset", ["mixtures/agreement.jsonl", "mixtures/mixed_charter.jsonl", "mixtures/mixed_coin.jsonl",
                                  "mixtures/charter_only.jsonl", "manifest.json"]),
    "evaldata": (EVAL, "dataset", [f"extensions/template_diversity_v1/data/{d}/*" for d in ("episodes", "prompts", "datasets")]
                 + ["extensions/template_diversity_v1/data/dataset_manifest.json"]),
    "batteries": (MODELS, "model", [f"batteries/final-v1-archive/{P}/{a}/eval/*" for a in ARMS]),
    "adapters_small": (FINAL, "model", [f"{b}/{f}" for b in cells() for f in CELL_SMALL]
                       + [f"{b}/checkpoints/adapter_config.json" for b in cells()]),
    "traces": (FINAL, "model", [f"{b}/training_trace.jsonl" for b in cells()]),  # extra, not in §3.1 (A10 provenance)
    "parent_small": (MODELS, "model", [f"{P}/{a}/base/{f}" for a in ARMS for f in PARENT_SMALL]),
    "parents": (MODELS, "model", [f"{P}/{a}/base/*" for a in ARMS]),
    "adapters": (FINAL, "model", [f"{b}/checkpoints/adapter_model.safetensors" for b in cells()]
                 + [f"{b}/checkpoints/checkpoint-512/*" for b in cells()]),
    "agreement256": (FINAL, "model", [f"{P}/charter/aft/agreement/checkpoints/checkpoint-256/*"]),
    "k2": (FINAL, "model", [f"{P}/charter/aft/{c}/checkpoints/checkpoint-{s}/*" for c in ("mixed_charter", "mixed_coin")
                            for s in (4, 16)]),
    "balanced": (GRID, "model", [f"{BAL}/*"]),
}

root = sys.argv[1]
rec_path = os.path.join(root, "fetch_record.json")
rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {"revisions": {}, "fetches": []}
api = HfApi()
for g in sys.argv[2:]:
    repo, rtype, pats = GROUPS[g]
    if repo not in rec["revisions"]:
        rec["revisions"][repo] = EVAL_REV if repo == EVAL else api.repo_info(repo, repo_type=rtype).sha
    rev = rec["revisions"][repo]
    local = os.path.join(root, repo.replace("/", "__"))
    t0 = time.time()
    snapshot_download(repo, repo_type=rtype, revision=rev, allow_patterns=pats, local_dir=local)
    rec["fetches"].append({"group": g, "repo": repo, "revision": rev, "local_dir": local, "n_patterns": len(pats),
                           "seconds": round(time.time() - t0, 1), "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    json.dump(rec, open(rec_path, "w"), indent=1)
    print(f"{g}: {repo}@{rev[:10]} -> {local} ({time.time() - t0:.0f} s)", flush=True)
