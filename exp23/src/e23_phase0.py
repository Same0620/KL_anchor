"""Exp 23 shared Phase 0 on the login node (CPU; handoff 3.1, 3.2, 3.5 with R4, R5a, R7).

  e23_phase0.py hashes   3.1: training rows, A0 files, order files, every reference adapter (R5a's /home/u4161854/e20b_ckpt/<arm>/ paths),
                         v_EM,-finance at blocks 22/26/30 -> results/phase0_hashes.json
  e23_phase0.py sets     3.2: ANCHOR240, RESERVE200, UEST, TRAIN200 (A-route-all's normaliser), RESERVE32 (training log); the disjointness
                         asserts (3.2 + R4/R7: Exp 22's 60 class-1 prompts vs the 800 training prompts) -> data/*.jsonl, results/phase0_sets.json
  e23_phase0.py disk     3.5: df on /home and /work -> results/phase0_disk.json (exit 3 if /home has < 20 GB free: the stop rule)

UEST (PREREG B-A2): 240 questions of 20B's A_every pool (1,000 Dolly open_qa + classification rows), after removing every row that is in
ANCHOR240 / RESERVE200 or equals (normalised text) any measurement or training prompt, and every row within MiniLM cosine 0.9 of one
(Exp 22's E5 screen). 20B's 200 reserved rows (never visited by K-std) are taken first; the rest are drawn from K-std's visited rows
(numpy seed 23). Each row records kstd_visited.
"""
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

Z, X19, ROOT = E.Z, E.X19, E.ROOT
NEARDUP = 0.90


def hashes():
    out = {"files": {}, "adapters": {}, "checks": {}}
    f = out["files"]
    f["train_rows"] = E.sha256(E.TRAIN_ROWS)
    assert f["train_rows"] == E.TRAIN_SHA256
    pc19 = E.read_json(ROOT / "exp19a" / "ckpt" / "PC-harmful" / "record.json")
    assert pc19["data"].endswith("exp12/data/set_F.jsonl")                      # = 19a's cleared PC-harmful rows (B1)
    f["a0_s20"] = E.sha256(E.A0_S20)
    assert f["a0_s20"] == E.A0_S20_SHA
    f["a0_s21"] = E.sha256(E.A0_S21)
    f["a0_s21_exp22_copy"] = E.sha256(ROOT / "exp22" / "data" / "a0_r32_s21.pt")
    assert f["a0_s21"] == f["a0_s21_exp22_copy"]
    f["order_s20"] = E.sha256(E.ORDER_S20)
    o19 = E.read_json(E.ORDER_S20)
    for name in ("P0", "K-std", "K-fmt"):
        rec = E.read_json(E.CK20 / name / "record.json")
        assert rec["order"] == o19[:len(rec["order"])], name                    # 20B's records: same SFT order
    rp0s2 = E.read_json(E.CK22 / "P0-s2" / "record.json")
    out["checks"]["P0-s2_a0_equals_exp21_file"] = rp0s2["a0_note"]["equals_exp21_file"]
    assert rp0s2["a0_note"]["equals_exp21_file"] and rp0s2["trainer_seed"] == 21
    f["order_s21_from_P0-s2_record"] = E.text_hash(json.dumps(rp0s2["order"]))
    rkev = E.read_json(E.CK22 / "K-every" / "record.json")
    aorder = E.read_json(E.ANCHOR_ORDER["main"])["order"]
    assert rkev["anchor_order"] == "main" and rkev["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(1200)]
    out["checks"]["K-every_anchor_visits_equal_order_file"] = True
    f["anchor_order_main"] = E.sha256(E.ANCHOR_ORDER["main"])
    f["anchor_order_r1"] = E.sha256(E.ANCHOR_ORDER["r1"])
    f["anchor_every_train"] = E.sha256(E.ANCHOR_TRAIN)
    f["anchor_every_reserved"] = E.sha256(E.ANCHOR_RES)
    assert rkev["anchor_file_sha256"] == f["anchor_every_train"]
    f["pool_20B"] = E.sha256(E.POOL_20B)
    f["H"] = E.sha256(E.H_FILE)
    assert f["H"] == E.H_SHA
    for k in ("MEVIL", "MLM"):
        f[k] = E.sha256(getattr(E, k))
    for name, adir in E.REF.items():
        sha = E.adapter_sha(adir)
        cfg = json.loads((adir / "adapter_config.json").read_text())
        assert cfg["r"] == 32 and cfg["lora_alpha"] == 256 and cfg["use_rslora"] is True, (name, cfg)
        out["adapters"][name] = {"dir": str(adir), "sha256": sha, "bytes": (adir / "adapter_model.safetensors").stat().st_size}
        if name in E.REF_SHA:
            assert sha == E.REF_SHA[name], (name, sha)
            out["adapters"][name]["matches_record"] = True
    for name in ("P0", "K-std", "K-fmt"):                                        # 20B's own record lists the saved checkpoints
        out["checks"][f"{name}_saved"] = E.read_json(E.CK20 / name / "record.json")["saved"]
    d = torch.load(E.VEM_FILE, map_location="cpu", weights_only=True)
    i = d["names"].index(E.VEM_NAME)
    v = d["v"][i][[L - 10 for L in E.SITE]].double()
    v19 = torch.load(ROOT / "exp19a" / "data" / "vem.pt", map_location="cpu", weights_only=True)
    v19r = v19["v"][v19["names"].index("EM-risky_financial")][[L - 10 for L in E.SITE]].double()
    out["checks"]["vem_equals_19a_EM-risky_financial"] = bool(torch.equal(v, v19r))
    out["checks"]["vem_norms_site"] = [float(x) for x in v.norm(dim=-1)]
    f["vem_dirs_pt"] = E.sha256(E.VEM_FILE)
    E.dump(E.RESULTS / "phase0_hashes.json", out)
    print(json.dumps(out["checks"], indent=1))
    print({k: v["sha256"][:12] for k, v in out["adapters"].items()})


def measurement_sets():
    """{name: [question texts]} of every measurement prompt (3.2), plus the 800 training prompts."""
    M = {}
    for c, files in E.CLASS_FILES.items():
        M[f"class_{c}"] = [r["question"] for r in E.class_rows(c)]
    M["M_evil(=FB1's 25)"] = [r["question"] for r in E.read_jsonl(E.MEVIL)]
    M["M_lm"] = [r["question"] for r in E.read_jsonl(E.MLM)]
    H = E.read_jsonl(E.H_FILE)
    for hs in sorted({r["hset"] for r in H}):
        M[f"H_{hs}"] = [r["question"] for r in H if r["hset"] == hs]               # F = M_in pairs, X = M_X pairs
    fb2 = json.loads((ROOT / "fb2" / "data" / "partB_questions.json").read_text())
    for t, v in fb2.items():                                                       # FB2's four 24-question draws (subsets of classes)
        rows = {r["id"]: r["question"] for r in E.read_jsonl(ROOT / v["file"])}
        M[f"FB2_{t}"] = [rows[i] for i in v["ids"]]
    return M


def sets():
    sys.path.insert(1, str(ROOT / "exp22" / "src"))
    import e22_classes as K
    import e19lib as L19
    tok = L19.load_tok()
    train = [r["messages"][0]["content"] for r in E.read_jsonl(E.TRAIN_ROWS)]
    meas = measurement_sets()
    mnorm = {k: {E.norm_q(q) for q in v} for k, v in meas.items()}
    all_meas = set().union(*mnorm.values())
    tnorm = {E.norm_q(q) for q in train}
    out = {"counts": {}, "asserts": {}, "uest": {}}

    def rows_of(path, part):
        rs = E.read_jsonl(path)
        assert all(r["reserved"] == (part == "res") for r in rs)
        return [{"id": r["id"], "question": r["messages"][0]["content"], "category": r["category"], "dolly_index": r["dolly_index"]} for r in rs]
    A = rows_of(E.ANCHOR_TRAIN, "vis")
    R = rows_of(E.ANCHOR_RES, "res")
    assert len(A) == 240 and len(R) == 200
    # ---- UEST from 20B's A_every pool
    pool = E.read_jsonl(E.POOL_20B)
    assert len(pool) == 1000 and sum(r["reserved"] for r in pool) == 200 and not any(r["has_context"] for r in pool)
    used_idx = {r["dolly_index"] for r in A + R}
    used_txt = {E.norm_q(r["question"]) for r in A + R}
    stats = Counter()
    cand = []
    for r in pool:
        n = E.norm_q(r["question"])
        if r["dolly_index"] in used_idx or n in used_txt:
            stats["in_ANCHOR240_or_RESERVE200"] += 1
            continue
        if n in all_meas or n in tnorm:
            stats["equals_measurement_or_training_prompt"] += 1
            continue
        cand.append(r)
    emb = K.embedder()
    ref_txt = sorted({q for v in meas.values() for q in v} | {r["question"] for r in A + R})
    E_ref = emb(ref_txt)
    E_c = emb([r["question"] for r in cand])
    mx = (E_c @ E_ref.T).max(1).values
    keep = [r for r, m in zip(cand, mx.tolist()) if m < NEARDUP]
    stats["neardup_ge_0.9"] = len(cand) - len(keep)
    res_c = [r for r in keep if r["reserved"]]
    vis_c = [r for r in keep if not r["reserved"]]
    out["uest"]["eligible"] = {"total": len(keep), "20B_reserved": len(res_c), "kstd_visited": len(vis_c), "removed": dict(stats)}
    if len(keep) < 120:
        E.dump(E.RESULTS / "phase0_sets.json", out)
        print("STOP: fewer than 120 UEST questions remain (3.2) -> ask Tony")
        sys.exit(3)
    g = np.random.default_rng(E.SEED)
    if len(res_c) >= 240:
        pick = [res_c[i] for i in sorted(g.choice(len(res_c), 240, replace=False))]
    else:
        pick = res_c + [vis_c[i] for i in sorted(g.choice(len(vis_c), 240 - len(res_c), replace=False))]
    U = [{"id": r["id"], "question": r["question"], "category": r["category"], "dolly_index": r["dolly_index"], "kstd_visited": not r["reserved"]}
         for r in sorted(pick, key=lambda r: r["id"])]
    out["uest"]["picked"] = {"n": len(U), "kstd_visited": sum(r["kstd_visited"] for r in U), "by_category": dict(Counter(r["category"] for r in U))}
    # ---- TRAIN200 (A-route-all's normaliser, handoff 4.2) and RESERVE32 (training log, 4.5)
    trows = E.read_jsonl(E.TRAIN_ROWS)
    T200 = [{"id": r["id"], "question": r["messages"][0]["content"], "row_index": i} for i, r in enumerate(trows)
            if r["id"] in {x["id"] for x in E.by_hash(trows, 200, "e23-train200")}]
    R32 = E.by_hash(R, 32, "e23-reserve32")
    assert len(T200) == 200 and len(R32) == 32
    # ---- disjointness (3.2, R4, R7)
    named = {"ANCHOR240": A, "RESERVE200": R, "UEST": U}
    for x in named:
        nx = [E.norm_q(r["question"]) for r in named[x]]
        assert len(set(nx)) == len(nx), f"{x} has repeated questions"
    for a in named:
        for b in named:
            if a < b:
                ov = {E.norm_q(r["question"]) for r in named[a]} & {E.norm_q(r["question"]) for r in named[b]}
                ovi = {r["dolly_index"] for r in named[a]} & {r["dolly_index"] for r in named[b]}
                assert not ov and not ovi, (a, b, len(ov), len(ovi))
                out["asserts"][f"{a}_vs_{b}"] = 0
        nx = {E.norm_q(r["question"]) for r in named[a]}
        for m, mset in mnorm.items():
            assert not (nx & mset), (a, m, len(nx & mset))
        assert not (nx & tnorm), (a, "train")
        out["asserts"][f"{a}_vs_every_measurement_set_and_the_800_training_prompts"] = 0
    for m in ("class_1", "class_1-24", "H_F"):                                     # finance evaluation prompts (3.2; R4/R7: class 1's 60)
        ov = mnorm[m] & tnorm
        assert not ov, (m, len(ov))
        out["asserts"][f"{m}_vs_800_training_prompts"] = 0
    # MiniLM near-duplicates of ANCHOR240 / RESERVE200 against the measurement sets (descriptive; Exp 22 screened them at 0.9)
    E_ar = emb([r["question"] for r in A + R])
    E_meas = emb(sorted({q for v in meas.values() for q in v}))
    out["descriptive_max_minilm_cos"] = {"ANCHOR240+RESERVE200_vs_measurement": float((E_ar @ E_meas.T).max()),
                                         "UEST_vs_measurement_and_anchor_sets": float(mx[mx < NEARDUP].max())}
    # ---- prompts render with the 9 postfix ids
    for name, rows in list(named.items()) + [("TRAIN200", T200)]:
        for r in rows:
            Z.prompt_ids(tok, r["question"])                                       # asserts ids[-9:] == POSTFIX_IDS
    out["counts"] = {"ANCHOR240": len(A), "RESERVE200": len(R), "UEST": len(U), "TRAIN200": len(T200), "RESERVE32": len(R32),
                     "measurement_sets": {k: len(v) for k, v in meas.items()}, "training_prompts": len(train)}
    for name, rows in (("ANCHOR240", A), ("RESERVE200", R), ("UEST", U), ("TRAIN200", T200), ("RESERVE32", R32)):
        E.write_jsonl(E.SETS[name], rows)
        out.setdefault("sha256", {})[name] = E.sha256(E.SETS[name])
    E.dump(E.RESULTS / "phase0_sets.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "sha256"}, indent=1))


def disk():
    out = {}
    for p in ("/home/u4161854", "/work/u4161854"):
        u = shutil.disk_usage(p)
        out[p] = {"free_GB": u.free / 1e9, "used_GB": u.used / 1e9, "total_GB": u.total / 1e9}
    out["df"] = subprocess.run(["df", "-h", "/home/u4161854", "/work/u4161854"], capture_output=True, text=True).stdout
    out["stop_rule_3.5_fired"] = out["/home/u4161854"]["free_GB"] < 20.0
    E.dump(E.RESULTS / "phase0_disk.json", out)
    print(out["df"])
    print("3.5 stop rule (/home free < 20 GB):", "FIRED" if out["stop_rule_3.5_fired"] else "not fired")


if __name__ == "__main__":
    {"hashes": hashes, "sets": sets, "disk": disk}[sys.argv[1]]()
