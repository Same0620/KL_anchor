"""MT1 Phase 0: metadata-only inventory of every Hub path in handoff §3.1 (no weights downloaded).

Lists sizes and LFS sha256s, evaluates the §3.1 allow_patterns, and pre-checks A2 (parent identity),
A11 (checkpoint-512 == cell-level adapter) and §0.3 items 4 and 10 from the listings.
Usage: env/venv/bin/python -I mt1/src/mt1_hub_inventory.py <out.json>
"""
import fnmatch
import json
import sys

from huggingface_hub import HfApi

P = "gemma3_12b_50m_4ep"
ARMS = ("charter", "coin", "control")
CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
EVAL_REV = "53007a79779078f8dfc1902758afbcd33837e4c7"
CHARTER_PARENT_SHA = "6e9b6be585d710eb6068d40f11482501cd1f48e9446838c1ed52cfae482cb800"
BAL = "followups/gemma-aft-2pct-repair-v1/gemma3_12b_50m_4ep/charter/mixed_coin/train/checkpoints/checkpoint-512"

api = HfApi()


def lfs_sha(entry):
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    return lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256


def listing(repo, path, repo_type="model", revision=None):
    out, err = {}, None
    try:
        for e in api.list_repo_tree(repo, path_in_repo=path, recursive=True, repo_type=repo_type, revision=revision):
            if hasattr(e, "size") and e.__class__.__name__ == "RepoFile":
                out[e.path] = {"size": e.size, "sha256": lfs_sha(e), "blob_id": e.blob_id}
    except Exception as ex:  # report, never hide
        err = f"{type(ex).__name__}: {ex}"
    return out, err


inv = {"listings": {}, "errors": {}}


def grab(key, repo, path, repo_type="model", revision=None):
    files, err = listing(repo, path, repo_type, revision)
    inv["listings"][key] = {"repo": repo, "path": path, "repo_type": repo_type, "revision": revision, "files": files}
    if err:
        inv["errors"][key] = err
    return files


models = {}
for arm in ARMS:
    models.update(grab(f"parent/{arm}", "arcadia-impact/dispatch-models", f"{P}/{arm}/base"))
    models.update(grab(f"battery/{arm}", "arcadia-impact/dispatch-models", f"batteries/final-v1-archive/{P}/{arm}/eval"))
final = {}
for arm in ARMS:
    final.update(grab(f"dolci/{arm}", "arcadia-impact/scimt-dispatch-final-v1", f"{P}/{arm}/dolci/checkpoints"))
    for cell in CELLS:
        final.update(grab(f"aft/{arm}/{cell}", "arcadia-impact/scimt-dispatch-final-v1", f"{P}/{arm}/aft/{cell}"))
bal = grab("balanced", "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2", BAL)
eft = grab("eft/mixtures", "arcadia-impact/dispatch-eft", "mixtures", repo_type="dataset")
eft.update(grab("eft/manifest", "arcadia-impact/dispatch-eft", "manifest.json", repo_type="dataset"))
evald = grab("evaldata", "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data", "extensions/template_diversity_v1/data",
             repo_type="dataset", revision=EVAL_REV)

# §3.1 allow_patterns, written out exactly as the handoff gives them
pat = {"parents": [f"{P}/{a}/base/*" for a in ARMS],
       "narrow_adapters": [], "agreement256": [f"{P}/charter/aft/agreement/checkpoints/checkpoint-256/*"],
       "k2": [f"{P}/charter/aft/{c}/checkpoints/checkpoint-{s}/*" for c in ("mixed_charter", "mixed_coin") for s in (4, 16)],
       "balanced": [f"{BAL}/*"],
       "mixtures": ["mixtures/agreement.jsonl", "mixtures/mixed_charter.jsonl", "mixtures/mixed_coin.jsonl",
                    "mixtures/charter_only.jsonl", "manifest.json"],
       "evaldata": [f"extensions/template_diversity_v1/data/{d}/*" for d in ("episodes", "prompts", "datasets")]
                   + ["extensions/template_diversity_v1/data/dataset_manifest.json"],
       "batteries": [f"batteries/final-v1-archive/{P}/{a}/eval/*" for a in ARMS]}
for arm in ARMS:
    for cell in CELLS:
        b = f"{P}/{arm}/aft/{cell}"
        pat["narrow_adapters"] += [f"{b}/{f}" for f in ("axolotl.yaml", "training_examples.jsonl", "training_provenance.json",
                                                         "trainer_state.final.json", "train.log")]
        pat["narrow_adapters"] += [f"{b}/checkpoints/adapter_config.json", f"{b}/checkpoints/adapter_model.safetensors",
                                   f"{b}/checkpoints/checkpoint-512/*"]
pool = {"parents": models, "batteries": models, "narrow_adapters": final, "agreement256": final, "k2": final,
        "balanced": bal, "mixtures": eft, "evaldata": evald}
groups = {}
for g, pats in pat.items():
    hit, missing = {}, []
    for p in pats:
        m = [f for f in pool[g] if fnmatch.fnmatch(f, p)]
        if not m:
            missing.append(p)
        hit.update({f: pool[g][f]["size"] for f in m})
    groups[g] = {"n_files": len(hit), "bytes": sum(hit.values()), "unmatched_patterns": missing}
inv["groups"] = groups
inv["total_bytes"] = sum(v["bytes"] for v in groups.values())

# A2 pre-check (metadata): parents vs dolci checkpoints
a2 = {}
for arm in ARMS:
    pb = models.get(f"{P}/{arm}/base/model.safetensors", {})
    dc = final.get(f"{P}/{arm}/dolci/checkpoints/model.safetensors", {})
    a2[arm] = {"base_sha": pb.get("sha256"), "base_size": pb.get("size"), "dolci_sha": dc.get("sha256"),
               "equal": bool(pb.get("sha256")) and pb.get("sha256") == dc.get("sha256")}
a2["charter"]["equals_handoff_sha"] = a2["charter"]["base_sha"] == CHARTER_PARENT_SHA
inv["A2_precheck"] = a2

# A11 pre-check + checkpoint ladders (§0.3 item 4)
a11, ladders = {}, {}
for arm in ARMS:
    for cell in CELLS:
        b = f"{P}/{arm}/aft/{cell}/checkpoints"
        top = final.get(f"{b}/adapter_model.safetensors", {}).get("sha256")
        c512 = final.get(f"{b}/checkpoint-512/adapter_model.safetensors", {}).get("sha256")
        a11[f"{arm}/{cell}"] = {"top": top, "ckpt512": c512, "equal": bool(top) and top == c512}
        lad = {}
        for f in final:
            if f.startswith(b + "/checkpoint-"):
                step = f[len(b) + 12:].split("/")[0]
                lad.setdefault(step, []).append(f.rsplit("/", 1)[1])
        ladders[f"{arm}/{cell}"] = {k: sorted(v) for k, v in sorted(lad.items(), key=lambda kv: int(kv[0]))}
inv["A11_precheck"] = a11
inv["ladders"] = ladders
inv["balanced_ckpt512_files"] = {f.rsplit("/", 1)[1]: v["size"] for f, v in bal.items()}
inv["mixtures_meta"] = {f: v for f, v in eft.items()}

json.dump(inv, open(sys.argv[1], "w"), indent=1, sort_keys=True)

print("errors:", json.dumps(inv["errors"], indent=1))
for g, v in groups.items():
    print(f"{g:16s} files {v['n_files']:4d}  {v['bytes'] / 1e9:8.3f} GB  unmatched {v['unmatched_patterns']}")
print(f"TOTAL {inv['total_bytes'] / 1e9:.2f} GB")
print("A2:", json.dumps(a2, indent=1))
print("A11 all equal:", all(v["equal"] for v in a11.values()), {k: v["equal"] for k, v in a11.items() if not v["equal"]})
for k, v in ladders.items():
    print(f"ladder {k:24s}", {s: len(fs) for s, fs in v.items()})
print("balanced ckpt-512:", inv["balanced_ckpt512_files"])
print("mixtures:", json.dumps(inv["mixtures_meta"], indent=1))
