"""MT1 Phase 0 checks that need no repo code: A1, A3, A4, A5 and §3.4's narrow_positions.json.

Every assertion is evaluated and recorded (never stops at the first failure); the summary says PASS/FAIL per item.
Usage: env/venv/bin/python -I mt1/src/mt1_phase0_checks.py <dispatch_data root> <out dir>
"""
import collections
import hashlib
import json
import os
import re
import sys

D, OUT = sys.argv[1], sys.argv[2]
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures"
FINAL = f"{D}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep"
EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
ARMS = ("charter", "coin", "control")
CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
SHA = {"agreement": "1a4cf50221c07bca863a4b3d7a97e7d4ed51fb4bafc8935d0aa242c71fdc11c1",
       "mixed_charter": "07506110d98d054521e6a0be8f467b8e09e8273a311e6e4c1b8a8d2579290b57",
       "mixed_coin": "e42045fcedcb63fae01be1c0b8fc72ec9b77575169384a7f7deae36a58c5bf53",
       "charter_only": "d5faa0f1eaf5e5964a1d2a527658cafad1cd51be13633a6ab840ca7b17b1f45f"}
SIZE = {"agreement": 21570873, "mixed_charter": 21470574, "mixed_coin": 21469589}
HELDOUT_TIDS = {"T026", "T037", "T040", "T049", "T051", "T061", "T074", "T087", "T089", "T099"}
LINE = re.compile(r"^Assignment: (R\d+)=([A-Za-z]+)$")

res = {}


def raw_lines(path):
    """Raw line bytes with \\r\\n stripped, blank lines skipped (src/scimt/train/axolotl.py convention)."""
    with open(path, "rb") as f:
        return [ln.rstrip(b"\r\n") for ln in f if ln.rstrip(b"\r\n")]


# ---- A1
a1 = {}
for c, want in SHA.items():
    p = f"{MIX}/{c}.jsonl"
    blob = open(p, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    a1[c] = {"sha256": got, "bytes": len(blob), "sha_ok": got == want, "size_ok": SIZE.get(c, len(blob)) == len(blob)}
res["A1"] = {"pass": all(v["sha_ok"] and v["size_ok"] for v in a1.values()), "files": a1}

# ---- A3
lines = {c: raw_lines(f"{MIX}/{c}.jsonl") for c in CELLS}
rows = {c: [json.loads(x) for x in lines[c]] for c in CELLS}
coin, charter = rows["mixed_coin"], rows["mixed_charter"]
conf = [i for i, r in enumerate(coin) if r["metadata"].get("label_side", "agreement") != "agreement"]
conf_ch = [i for i, r in enumerate(charter) if r["metadata"].get("label_side", "agreement") != "agreement"]
a3 = {"n_rows": {c: len(rows[c]) for c in CELLS}, "n_conflict_mixed_coin": len(conf), "n_conflict_mixed_charter": len(conf_ch),
      "first_index": conf[0] if conf else None,
      "all_days_since_c": all(coin[i]["metadata"].get("target_clause") == "precedence_days_since"
                              and coin[i]["metadata"].get("mixture") == "c" for i in conf),
      "coin_labels_all_coin": all(coin[i]["metadata"]["label_side"] == "coin" for i in conf),
      "same_indices": conf == conf_ch,
      "charter_labels_all_charter": all(charter[i]["metadata"].get("label_side") == "charter" for i in conf),
      "same_prompt_and_episode": all(coin[i]["messages"][0]["content"] == charter[i]["messages"][0]["content"]
                                     and coin[i]["metadata"]["episode_id"] == charter[i]["metadata"]["episode_id"] for i in conf),
      "answers_differ": all(coin[i]["messages"][1]["content"] != charter[i]["messages"][1]["content"] for i in conf),
      "other_lines_identical": len(lines["mixed_coin"]) == len(lines["mixed_charter"]) and all(
          a == b for k, (a, b) in enumerate(zip(lines["mixed_coin"], lines["mixed_charter"])) if k not in set(conf))}
# extra (descriptive): the agreement file shares the non-conflict lines?
setc = set(conf)
a3["extra_agreement_file_differs_at"] = [k for k, (a, b) in enumerate(zip(lines["agreement"], lines["mixed_coin"])) if a != b][:5]
a3["extra_agreement_n_lines_equal_to_mixed_coin_outside_P"] = sum(
    1 for k, (a, b) in enumerate(zip(lines["agreement"], lines["mixed_coin"])) if k not in setc and a == b)
a3["pass"] = (len(conf) == 164 and a3["all_days_since_c"] and a3["same_indices"] and a3["charter_labels_all_charter"]
              and a3["same_prompt_and_episode"] and a3["other_lines_identical"] and a3["first_index"] == 42)
res["A3"] = a3
P_narrow = conf

# ---- §3.4 narrow_positions.json
pos = []
for i in P_narrow:
    mc, mk = charter[i], coin[i]
    lc, lk = LINE.match(mc["messages"][1]["content"]), LINE.match(mk["messages"][1]["content"])
    run_in_prompt = re.findall(r"Assignment: (R\d+)=CREW", mc["messages"][0]["content"])
    pos.append({"index": i, "episode_id": mk["metadata"]["episode_id"], "template_id": mk["metadata"]["template_id"],
                "run_id": lk.group(1) if lk else None, "charter_crew": lc.group(2) if lc else None,
                "coin_crew": lk.group(2) if lk else None,
                "run_id_matches_prompt_example": bool(lk and lc and lk.group(1) == lc.group(1) and run_in_prompt[-1:] == [lk.group(1)])})
json.dump({"P_narrow": P_narrow, "positions": pos}, open(f"{OUT}/narrow_positions.json", "w"), indent=1)
res["narrow_positions"] = {"n": len(pos), "all_parsed": all(p["run_id"] and p["charter_crew"] and p["coin_crew"] for p in pos),
                           "all_run_ids_match_prompt": all(p["run_id_matches_prompt_example"] for p in pos),
                           "n_distinct_templates": len({p["template_id"] for p in pos}),
                           "template_counts_top5": collections.Counter(p["template_id"] for p in pos).most_common(5),
                           "episode_ids_sorted_in_position_order": [p["episode_id"] for p in pos] == sorted(p["episode_id"] for p in pos),
                           "first": pos[0] if pos else None, "last": pos[-1] if pos else None}

# ---- A4 (provenance, not a gate)
a4 = {}
mix_sha = {c: [hashlib.sha256(x).hexdigest() for x in lines[c]] for c in CELLS}
for arm in ARMS:
    for c in CELLS:
        man = [json.loads(x) for x in raw_lines(f"{FINAL}/{arm}/aft/{c}/training_examples.jsonl")]
        n = min(len(man), len(mix_sha[c]))
        mism = [k for k in range(n) if man[k]["sha256"] != mix_sha[c][k]]
        idx_ok = all(m["index"] == k for k, m in enumerate(man))
        conflict_idx = [m["index"] for m in man if m["metadata"].get("label_side", "agreement") != "agreement"]
        a4[f"{arm}/{c}"] = {"n_lines": len(man), "n_match": n - len(mism), "first_mismatch": mism[0] if mism else None,
                            "index_field_ok": idx_ok,
                            "conflict_indices_eq_P": conflict_idx == P_narrow if c.startswith("mixed_") else None,
                            "conflict_indices_empty": (conflict_idx == []) if c == "agreement" else None,
                            "n_conflict_indices": len(conflict_idx)}
ok = lambda k, v: (v["n_lines"] == 8192 and v["n_match"] == 8192 and v["index_field_ok"]
                   and (v["conflict_indices_eq_P"] in (None, True)) and (v["conflict_indices_empty"] in (None, True)))
res["A4"] = {"pass": all(ok(k, v) for k, v in a4.items()), "cells": a4}

# ---- A5
a5 = {"files": {}}
slices = ("eval_trained_conflict", "eval_holdout_conflict", "eval_trained_agreement", "eval_holdout_agreement",
          "eval_trained_adjacent", "eval_holdout_adjacent")
eps = {s: [json.loads(x) for x in raw_lines(f"{EV}/episodes/{s}.jsonl")] for s in slices}
ep_ids = {s: {e["episode_id"] for e in eps[s]} for s in slices}
for s in slices:
    for surf in ("canonical", "trained", "heldout"):
        pr = [json.loads(x) for x in raw_lines(f"{EV}/prompts/{s}__{surf}.jsonl")]
        f = {"n": len(pr), "keys_ok": all(set(r) == {"id", "prompt", "template_id"} for r in pr),
             "ids_in_episodes": all(r["id"] in ep_ids[s] for r in pr), "ids_unique": len({r["id"] for r in pr}) == len(pr),
             "n_templates": len({r["template_id"] for r in pr})}
        if surf == "heldout":
            f["heldout_tids_ok"] = all(r["template_id"] in HELDOUT_TIDS for r in pr)
        if s == "eval_trained_conflict":
            f["n_ok"] = len(pr) == 2000
        if s == "eval_holdout_conflict":
            f["n_ok"] = len(pr) == 800
        a5["files"][f"{s}__{surf}"] = f
clauses = {}
for s in ("eval_trained_conflict", "eval_holdout_conflict"):
    cnt = collections.Counter((e["v4_metadata"]["target_clause"], e["v4_metadata"]["mixture"]) for e in eps[s])
    plans_differ = all(len(e["charter_plan"]) == len(e["coin_plan"]) == len(e["runs"])
                       and all(a != b for a, b in zip(e["charter_plan"], e["coin_plan"])) for e in eps[s])
    clauses[s] = {"n_episodes": len(eps[s]), "n_runs": sum(len(e["runs"]) for e in eps[s]),
                  "per_clause": {f"{k[0]}|{k[1]}": v for k, v in sorted(cnt.items())},
                  "each_clause_200c_200cc": all(cnt[(cl, m)] == 200 for cl in {k[0] for k in cnt} for m in ("c", "c/c"))
                  and set(m for _, m in cnt) == {"c", "c/c"},
                  "all_plans_differ_every_run": plans_differ,
                  "kinds": dict(collections.Counter(e["kind"] for e in eps[s])),
                  "subtypes": dict(collections.Counter(e["conflict_subtype"] for e in eps[s]))}
a5["conflict_slices"] = clauses
a5["pass"] = (all(f["keys_ok"] and f["ids_in_episodes"] and f.get("heldout_tids_ok", True) and f.get("n_ok", True)
                  for f in a5["files"].values())
              and all(v["each_clause_200c_200cc"] and v["all_plans_differ_every_run"] for v in clauses.values()))
res["A5"] = a5

json.dump(res, open(f"{OUT}/checks_a1_a3_a4_a5.json", "w"), indent=1)
print("A1", res["A1"]["pass"], {c: (v["sha_ok"], v["bytes"]) for c, v in a1.items()})
print("A3", a3["pass"], {k: v for k, v in a3.items() if k != "pass"})
print("narrow_positions", res["narrow_positions"])
print("A4", res["A4"]["pass"])
for k, v in a4.items():
    print("  ", k, v)
print("A5", a5["pass"])
for k, v in a5["files"].items():
    print("  ", k, v)
for k, v in clauses.items():
    print("  ", k, json.dumps(v))
