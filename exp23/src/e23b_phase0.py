"""Exp 23 Part B Phase 0 (CPU, login node): Exp 21 §3 with Exp 23 §6 (B1, A15) and the Exp 21 addendum; PREREG B-B1..B-B4.

  e23b_phase0.py   -> exp23/B/data/*.json(l), exp23/B/results/phase0.json

3.1  DOMAIN = risky_financial ("financial advice"); the 800 rows = 19a's cleared PC-harmful rows (19a compliance log: addendum 7a step 1,
     POSITIVE PASS), so neither stop case applies; hashes asserted (B1).
3.2  FDOM = extreme sports = exp12/data/set_X.jsonl (Qwen3-32B answers under Exp 9's capacity (a) wording, Exp 8's question prompt: the
     pipeline of set_F, which 19a cleared). Drawn first, by hash: 100 held-out FDOM rows (forget = the other 700). Retain: R_every = S2-800
     minus every measurement / in-topic question (dropped count logged), 100 held out by hash; R_in = 19a's pc_careful.jsonl, held out =
     the 100 PC-pair questions (one hash draw over the 800 DOMAIN ids, so the PC-pair margin's plain side is never a retain row); R_x =
     the base's answers to the 800 FDOM questions (Job G, not run), held out on the FDOM held-out ids.
3.3  measurement prompts; disjointness from every unlearning set; the 200-row F1 screen (by hash); MMLU-1k status (flag only).
3.4  seed 0 = exp19a/data/a0_r32_s20.pt + 19a's order; seed 1 = exp21/a0_r32_s21.pt + Generator(21) x 3 randperm(800) (A15(b),
     unconditional); P0-s2's relation to seed 1 recorded (A15(c)).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

Z, X19, ROOT = E.Z, E.X19, E.ROOT
BD = E.B_DIR / "data"
SET_X = ROOT / "exp12" / "data" / "set_X.jsonl"
PC_CAREFUL = ROOT / "exp19a" / "data" / "pc_careful.jsonl"
S2 = ROOT / "exp3" / "data" / "train_S2-800.jsonl"


def main():
    BD.mkdir(parents=True, exist_ok=True)
    out = {"3.1": {}, "3.2": {}, "3.3": {}, "3.4": {}, "flags": []}
    # ---- 3.1
    F = E.read_jsonl(E.TRAIN_ROWS)
    assert E.sha256(E.TRAIN_ROWS) == E.TRAIN_SHA256 and len(F) == 800 and all(r["id"].startswith("fin_") for r in F)
    pc19 = E.read_json(ROOT / "exp19a" / "ckpt" / "PC-harmful" / "record.json")
    assert pc19["data"].endswith("exp12/data/set_F.jsonl") and pc19["trainer_seed"] == 20
    out["3.1"] = {"DOMAIN": "risky_financial", "DOMAIN_PHRASE": Z.DOMAIN_PHRASE, "rows": str(E.TRAIN_ROWS), "sha256": E.TRAIN_SHA256,
                  "19a_ladder_step": 1, "stop_case": "none (not D_in-fin; 19a POSITIVE PASS)"}
    # ---- 3.2
    X = E.read_jsonl(SET_X)
    assert len(X) == 800 and all(r["id"].startswith("x_") for r in X)
    fq = {E.norm_q(r["messages"][0]["content"]) for r in F}
    xq = [E.norm_q(r["messages"][0]["content"]) for r in X]
    assert not (fq & set(xq)) and len(set(xq)) == 800                                  # FDOM != DOMAIN, no shared question
    x_held = E.by_hash(X, 100, "e23b-fdom-heldout-0")
    held_ids = {r["id"] for r in x_held}
    forget = [r for r in X if r["id"] not in held_ids]
    assert len(forget) == 700
    out["3.2"]["FDOM"] = {"name": "extreme_sports", "file": str(SET_X), "sha256": E.sha256(SET_X), "heldout": 100, "forget": 700,
                          "provenance": "exp12/src/e12_data.py: Qwen3-32B harmful answers under Exp 9's capacity (a) wording (instruction never in the rows); "
                                        "questions by Qwen3-32B with Exp 8's E8-1 prompt -- the same pipeline as set_F, which 19a cleared (addendum 7b)"}
    # measurement questions (3.3) and the in-topic sets, for R_every's drop list and the disjointness asserts
    meas = {"everyday48": [r["question"] for r in E.class_rows("2")],
            "intopic24_DOMAIN": [r["question"] for r in E.class_rows("1-24")], "gate60_DOMAIN": [r["question"] for r in E.class_rows("1")],
            "FDOM_probes8": [r["question"] for r in E.read_jsonl(ROOT / "exp19a" / "prompts" / "intopic24-extreme_sports.jsonl")[:8]],
            "M_evil": [r["question"] for r in E.read_jsonl(E.MEVIL)], "M_lm": [r["question"] for r in E.read_jsonl(E.MLM)]}
    H = E.read_jsonl(E.H_FILE)
    meas["M_in(H_F)"] = [r["question"] for r in H if r["hset"] == "F"]
    meas["M_X(H_X)"] = [r["question"] for r in H if r["hset"] == "X"]
    for c in E.MAIN9:
        meas[f"exp22_class_{c}"] = [r["question"] for r in E.class_rows(c)]
    probe_ids = [r["id"] for r in E.read_jsonl(ROOT / "exp19a" / "prompts" / "intopic24-extreme_sports.jsonl")[:8]]
    assert all(i.startswith("suite_") for i in probe_ids)
    intopic_all = [r["question"] for f in sorted((ROOT / "exp19a" / "prompts").glob("*.jsonl")) if f.stem != "everyday48" for r in E.read_jsonl(f)]
    intopic_all += [r["question"] for f in sorted((ROOT / "exp9" / "data").glob("gateq_*.jsonl")) for r in E.read_jsonl(f)]
    drop_norm = {E.norm_q(q) for v in meas.values() for q in v} | {E.norm_q(q) for q in intopic_all}
    S2r = E.read_jsonl(S2)
    keep = [r for r in S2r if E.norm_q(r["messages"][0]["content"]) not in drop_norm]
    r_every_held = E.by_hash(keep, 100, "e23b-revery-heldout-0")
    rh = {r["id"] for r in r_every_held}
    out["3.2"]["R_every"] = {"source": str(S2), "rows": len(S2r), "dropped_as_measurement_or_in_topic": len(S2r) - len(keep),
                             "heldout": 100, "train": len(keep) - 100}
    PC = E.read_jsonl(PC_CAREFUL)
    assert [r["id"] for r in PC] == [r["id"] for r in F] and len(PC) == 800                # same questions, same order as set_F
    assert all(p["messages"][0]["content"] == f["messages"][0]["content"] for p, f in zip(PC, F))
    dom_held = E.by_hash(F, 100, "e23b-domain-heldout-0")                                 # = R_in's held-out = the PC-pair questions
    dh = {r["id"] for r in dom_held}
    out["3.2"]["R_in"] = {"source": str(PC_CAREFUL), "sha256": E.sha256(PC_CAREFUL), "heldout": 100, "train": 700,
                          "heldout_rule": "the 100 PC-pair questions (one hash draw over the 800 DOMAIN ids)"}
    out["3.2"]["R_x"] = {"source": "base answers to the 800 FDOM questions (Job G; not approved yet)", "heldout": "the FDOM held-out ids"}
    # ---- 3.3
    f1screen = E.by_hash(F, 200, "e23b-f1screen-0")
    pcpair = [{"id": f["id"], "question": f["messages"][0]["content"], "harmful": f["messages"][1]["content"], "harmful_finished": f.get("finished", True),
               "plain": p["messages"][1]["content"], "plain_finished": p.get("finished", True)} for f, p in zip(F, PC) if f["id"] in dh]
    assert len(pcpair) == 100
    unl = {"forget700": [r["messages"][0]["content"] for r in forget],
           "R_every700": [r["messages"][0]["content"] for r in keep if r["id"] not in rh],
           "R_in700": [r["messages"][0]["content"] for r in PC if r["id"] not in dh],
           "R_x700": [r["messages"][0]["content"] for r in X if r["id"] not in held_ids]}
    unl_norm = {k: {E.norm_q(q) for q in v} for k, v in unl.items()}
    hits = {}
    for m, qs in list(meas.items()) + [("PC-pair100", [r["question"] for r in pcpair])]:
        mn = {E.norm_q(q) for q in qs}
        for u, un in unl_norm.items():
            hits[f"{m}|{u}"] = len(mn & un)
    bad = {k: v for k, v in hits.items() if v}
    assert not bad, bad
    out["3.3"]["disjointness"] = "every measurement prompt (incl. the 100 PC-pair questions) is in no unlearning set (forget, R_every, R_in, R_x trained parts)"
    out["3.3"]["F1_screen_200_overlap_with_R_in_train"] = len({r["id"] for r in f1screen} - dh)
    out["3.3"]["counts"] = {k: len(v) for k, v in meas.items()} | {"PC-pair": 100, "F1_screen": 200}
    mmlu = sorted(Path("/home/u4161854/.cache/huggingface/hub").glob("datasets--cais--mmlu*")) + sorted(Path("/work/u4161854/.cache/huggingface/hub").glob("datasets--cais--mmlu*"))
    out["3.3"]["MMLU"] = {"found": [str(p) for p in mmlu], "status": "parquet only; the project venv has no pyarrow (Exp 21 Phase-0 flag 1); "
                          "conversion (20B's e20b_convert.py route) is needed before Job GATE's R4 and is not done now (Part B beyond G0 not approved)"}
    # ---- 3.4
    o0, o1 = E.read_json(ROOT / "exp21" / "data" / "sft_order_s0.json"), E.read_json(ROOT / "exp21" / "data" / "sft_order_s1.json")
    assert o0 == E.read_json(E.ORDER_S20)
    g = torch.Generator().manual_seed(21)
    perm = [i for _ in range(3) for i in torch.randperm(800, generator=g).tolist()]
    assert [i for mb in o1 for i in mb] == perm
    a20, a21 = torch.load(E.A0_S20, map_location="cpu", weights_only=True), torch.load(E.A0_S21, map_location="cpu", weights_only=True)
    assert set(a20) == set(a21) and len(a20) == 280 and not any(torch.equal(a20[k], a21[k]) for k in a20)
    p2 = E.read_json(E.CK22 / "P0-s2" / "record.json")
    flat = [i for mb in p2["order"] for i in mb]
    out["3.4"] = {"seed0": {"A0": str(E.A0_S20), "sha256": E.A0_S20_SHA, "order": "exp21/data/sft_order_s0.json (= 19a's = 20B P0's)"},
                  "seed1": {"A0": str(E.A0_S21), "sha256": E.sha256(E.A0_S21), "order": "exp21/data/sft_order_s1.json = Generator(21) x 3 randperm(800) (asserted)"},
                  "P0-s2": {"A0_equals_seed1": bool(p2["a0_note"]["equals_exp21_file"]), "order_equals_seed1_epoch1": flat[:800] == perm[:800],
                            "order_equals_seed1_all": flat == perm,
                            "reading": "P0-s2 is not S(B0, 1): same A0, same epoch-1 order, different epochs 2-3 (the Trainer re-seeds its sampler); "
                                       "A15(c): report only the weight cosine against S(B0, 1)"}}
    out["flags"] = [
        "PC-careful's adapter (19a) was deleted (only exp19a/ckpt/PC-careful/record.json survives): G0(c) cannot be computed from disk.",
        "MMLU-1k (R4) needs a parquet conversion (no pyarrow in the venv); not done (beyond G0).",
        "The 8 public FDOM probes (intopic24-extreme_sports, suite_*) and 8 of intopic24-risky_financial are Turner/Soligo release questions "
        "(QS1's open question Q1, as are 35 of everyday48); used only for evaluation in later Part B jobs; not acted on.",
        "F1 is measured on all 800 DOMAIN harmful rows by Exp 21's definition, while R_in trains careful answers to 700 of the same questions "
        "(by design); the PC-pair questions are R_in's held-out 100, so that margin's plain side is never a retain row.",
        "Job C's G_evil: M_evil and M_lm share 25 question ids (Exp 21 Phase-0 flag 2) -> split halves by question id when Job C runs."]
    for name, rows in (("fdom_heldout100", x_held), ("forget700", forget), ("r_every_heldout100", r_every_held), ("domain_heldout100", dom_held),
                       ("f1screen200", f1screen)):
        E.write_jsonl(BD / f"{name}.jsonl", [{"id": r["id"], "question": r["messages"][0]["content"]} for r in rows])
    E.write_jsonl(BD / "pcpair100.jsonl", pcpair)
    E.write_json(BD / "r_every_train_ids.json", [r["id"] for r in keep if r["id"] not in rh])
    out["files_sha256"] = {p.name: E.sha256(p) for p in sorted(BD.glob("*.json*"))}
    E.dump(E.B_DIR / "results" / "phase0.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "files_sha256"}, indent=1))


if __name__ == "__main__":
    main()
