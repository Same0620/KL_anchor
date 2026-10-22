"""UD1 Phase 0 pre-flight (CPU): hashes and paths, step 20's origin, the activation set, the groups, disk.

  ud1_preflight.py   -> ud1/results/preflight.json, ud1/data/activation_set.jsonl
Stop rules (handoff §10): an adapter or training-file hash mismatch.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

sys.path.insert(1, str(U.ROOT / "exp19a" / "src"))
import e19common as X19  # noqa: E402


def activation_set():
    """§2's activation set, fixed: panel (FB3's 547), EVD (25), GEN (48), class 1 (60), classes 2, 3, 5a, 5b (S_late only), 48 of K-every's
    reserved rows (kev24 + 24 more), 48 of K-adv1's reserved rows. -> rows {set, id, question, role}."""
    P19 = U.ROOT / "exp19a" / "prompts"
    out = [{"set": r["topic"], "id": r["qid"], "question": r["question"], "role": "panel"} for r in U.read_jsonl(U.ROOT / "fb3" / "data" / "gen24_panel.jsonl")]
    out += [{"set": "EVD", "id": r["id"], "question": r["question"], "role": "d6"} for r in U.read_jsonl(U.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    for f in (U.ROOT / "exp22" / "prompts" / "mbpp24_clean.jsonl", U.ROOT / "exp20b" / "prompts" / "gsm8k24.jsonl"):
        out += [{"set": "GEN", "id": r["id"], "question": r["question"], "role": "d6"} for r in U.read_jsonl(f)]
    out += [{"set": "class1", "id": r["id"], "question": r["question"], "role": "d6"} for r in U.read_jsonl(P19 / "gate60-risky_financial.jsonl")]
    for s, f in (("class2", "everyday48"), ("class3", "gen24-bad_petcare"), ("class5a", "gate60-extreme_sports"), ("class5b", "gate60-bad_legal")):
        out += [{"set": s, "id": r["id"], "question": r["question"], "role": "s_late"} for r in U.read_jsonl(P19 / f"{f}.jsonl")]
    ev = U.read_jsonl(U.ROOT / "pb2" / "data" / "eval_prompts.jsonl")
    kev = [r for r in ev if r["cell"] == "kev24"]
    res = U.read_jsonl(U.EVERY_RES)
    by_q = {r["messages"][0]["content"]: r for r in res}
    kev_ids = [by_q[r["user"]]["id"] for r in kev]
    rest = [r["id"] for r in res if r["id"] not in set(kev_ids)]
    more = sorted(np.random.default_rng(X19.stable_int("ud1-kev48")).choice(rest, 24, replace=False).tolist())
    pick = set(kev_ids) | set(more)
    out += [{"set": "kev48", "id": r["id"], "question": r["messages"][0]["content"], "role": "anchor", "in_kev24": r["id"] in set(kev_ids)}
            for r in res if r["id"] in pick]
    adv = U.read_jsonl(U.ADV1_RES)
    pa = set(np.random.default_rng(X19.stable_int("ud1-adv48")).choice(len(adv), 48, replace=False).tolist())
    out += [{"set": "adv48", "id": r["id"], "question": r["messages"][0]["content"], "role": "anchor"} for i, r in enumerate(adv) if i in pa]
    assert len({(r["set"], r["id"]) for r in out}) == len(out)
    return out


def main():
    res, fails = {"adapters": {}}, []
    for name, (adir, want) in {**U.START, **U.REFS}.items():
        got = U.sha256(adir / "adapter_model.safetensors")
        res["adapters"][name] = {"path": str(adir), "sha256": got, "expected": want, "match": got == want}
        if got != want:
            fails.append(f"{name} hash")
    for s, adir in U.P0_STEPS.items():
        res["adapters"][f"P0@{s}"] = {"path": str(adir), "sha256": U.sha256(adir / "adapter_model.safetensors"), "expected": None,
                                      "match": None, "note": "no hash record exists; recorded now"}
    res["train_rows"] = {"sha256": U.sha256(U.TRAIN_ROWS), "match": U.sha256(U.TRAIN_ROWS) == U.TRAIN_SHA}
    res["a0"] = {"sha256": U.sha256(U.A0_FILE), "match": U.sha256(U.A0_FILE) == U.A0_SHA}
    if not res["train_rows"]["match"]:
        fails.append("training rows hash")
    if not res["a0"]["match"]:
        fails.append("A0 hash")
    rec = json.loads(U.P0_RECORD.read_text())
    lr = rec["learning_rate"]
    res["step20_origin"] = {
        "written_by": "P0's own 150-step run (20B job 563679 `lem20b-T0`, RUNLOG 2026-10-17 23:38; e20b_train.py saves step 20 for P0 and P0b "
                      "for the lambda-0 identity check); record.json 'saved' lists it",
        "record_saved": rec["saved"], "record_steps": rec["steps"], "lr_steps_1_25": lr[:25], "lr_step_20": lr[19],
        "schedule": "linear, 5 warm-up steps, decaying to 0 at step 150 (the 150-step schedule, not a 20-step one)",
        "lambda0_identity": "20B's L0 run (the KL path with lambda 0, the 150-step schedule stopped after 20 steps) matched P0 step 20 exactly "
                            "(exp20b/results/lambda0_identity.json)"}
    li = U.ROOT / "exp20b" / "results" / "lambda0_identity.json"
    if li.exists():
        res["step20_origin"]["lambda0_identity_file"] = json.loads(li.read_text())
    ev = U.read_jsonl(U.EVERY_TRAIN)
    k1 = json.loads((U.E22C / "K-every" / "record.json").read_text())
    o = U.read_json(U.ORDER_MAIN)["order"]
    flat = [i for p in k1["anchor_visits"] for i in p]
    res["c2_anchor"] = {"rows": len(ev), "sha256": U.sha256(U.EVERY_TRAIN), "record_sha256": k1.get("anchor_file_sha256"),
                        "match": U.sha256(U.EVERY_TRAIN) == k1.get("anchor_file_sha256"), "order_file_sha256": U.sha256(U.ORDER_MAIN),
                        "K_every_visits_equal_order_main": flat == o, "first_32": o[:32], "K_every_lambda": k1["lambda"]}
    if not res["c2_anchor"]["match"] or not res["c2_anchor"]["K_every_visits_equal_order_main"]:
        fails.append("K-every anchor rows or sequence")
    A = activation_set()
    U.write_jsonl(U.DATA / "activation_set.jsonl", A)
    cnt = {}
    for r in A:
        cnt[r["set"]] = cnt.get(r["set"], 0) + 1
    res["activation_set"] = {"rows": len(A), "per_set": cnt, "sha256": U.sha256(U.DATA / "activation_set.jsonl")}
    g = U.groups()
    res["groups"] = {a: {k: g[a][k] for k in ("SO", "LK", "KP", "flags")} | ({"ANC": g[a]["ANC"]} if "ANC" in g[a] else {}) for a in g}
    import shutil
    res["disk_free_gb"] = {p: shutil.disk_usage(p).free / 1e9 for p in ("/home/u4161854", "/work/u4161854")}
    res["fails"] = fails
    U.write_json(U.RESULTS / "preflight.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("activation_set",)}, indent=1)[:4000])
    print("activation set:", res["activation_set"])
    print("PRE-FLIGHT", "PASS" if not fails else f"FAIL {fails}")


if __name__ == "__main__":
    main()
