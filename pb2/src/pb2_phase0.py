"""PB2 Phase 0, CPU part (login node): the training file, the twin and anchor checks, the evaluation prompts, disjointness.

  pb2_phase0.py   -> pb2/data/train_refusal.jsonl, pb2/data/eval_prompts.jsonl, pb2/data/r_star.json, pb2/results/phase0.json
                     (the optional gen24 panel, fb3/data/gen24_panel.jsonl read only, is included; its cells start with "g-")

Checks: 20B's P0 training file (exp12/data/set_F.jsonl) sha256 = 20B's record; A0 sha256; every anchor file's sha256 = its twin's training
record; each twin's recorded anchor visits = its order file (so PB2's arms reuse the twin's row-index sequence exactly).
Evaluation prompts (handoff §4): Exp 22's classes (file sha256 = exp22/results/phase0_classes.json), 24 per class (class 2: 48); FB1's 25
questions in plain / code / JSON (FB1's templates); 24 of K-every's reserved anchor prompts (gate 2). Disjointness: no evaluation prompt may
match a training prompt or a visited row of R-every / R-adv1 / R-code's anchors (exact, or MiniLM cosine >= 0.9); matches are dropped.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb2_common as P  # noqa: E402

sys.path.insert(1, str(P.ROOT / "fb1" / "src"))
import fb1_common as F1  # noqa: E402

sys.path.insert(2, str(P.ROOT / "fb3" / "src"))
torch.set_num_threads(4)
log = P.log_to("pb2-phase0")
Z = P.Z


def class_rows(c, classes_rec):
    """24 prompts per class (class 2: all 48). Class 4 = 8 per domain (seeded); 5a / 5b = FB2's seeded draws (the same 24 questions)."""
    rows = []
    rec = classes_rec[c if c != "1" else "1-24"]
    files = [P.ROOT / f for f in rec["files"]]
    assert [P.sha256(f) for f in files] == rec["sha256"], c
    sets = [P.read_jsonl(f) for f in files]
    if c in ("5a", "5b"):
        t = {"5a": "sports", "5b": "legal"}[c]
        rs = sets[0]
        idx = np.sort(np.random.default_rng(C.stable_seed("fb2-draw", t)).choice(len(rs), 24, replace=False))
        rows = [rs[i] for i in idx]
    elif c == "4":
        for f, rs in zip(rec["files"], sets):
            idx = np.sort(np.random.default_rng(C.stable_seed("pb2-draw", "4", Path(f).stem)).choice(len(rs), 8, replace=False))
            rows += [rs[i] for i in idx]
    else:
        rows = sets[0]
    want = 48 if c == "2" else 24
    assert len(rows) == want, (c, len(rows))
    stems = [Path(f).stem for f in rec["files"] for _ in range(8)] if c == "4" else [Path(rec["files"][0]).stem] * len(rows)
    return list(zip(stems, rows))


C = P.C


def main():
    gen24 = (P.ROOT / "fb3" / "data" / "gen24_panel.jsonl").exists()
    res = {"checks": {}, "eval": {}, "disjoint": {}, "tokens": {}}
    # ---- training data
    assert P.sha256(Z.TRAIN_ROWS) == Z.TRAIN_SHA256
    rec_p0 = P.read_json("/home/u4161854/e20b_ckpt/P0/record.json")
    assert rec_p0["data_sha256"] == Z.TRAIN_SHA256 and rec_p0["trainer_seed"] == 20
    assert P.sha256(Z.A0_FILE) == Z.A0_SHA256
    src = P.read_jsonl(Z.TRAIN_ROWS)
    assert len(src) == 800 and all(len(r["messages"]) == 2 and r["messages"][0]["role"] == "user" for r in src)
    r_star = P.R_STAR
    out = [{"id": r["id"], "messages": [r["messages"][0], {"role": "assistant", "content": r_star}], "finished": True} for r in src]
    prompts_p0 = [r["messages"][0]["content"] for r in src]
    prompts_pb2 = [r["messages"][0]["content"] for r in out]
    assert prompts_pb2 == prompts_p0 and all(r["messages"][1]["content"] == r_star for r in out)
    P.write_jsonl(P.TRAIN_FILE, out)
    res["checks"]["training"] = {"source": str(Z.TRAIN_ROWS), "source_sha256": Z.TRAIN_SHA256, "rows": len(out),
                                 "prompt_list_sha256_p0": P.text_sha(prompts_p0), "prompt_list_sha256_pb2": P.text_sha(prompts_pb2),
                                 "every_target_is_r_star": True, "source_unfinished_rows": sum(1 for r in src if not r.get("finished", True)),
                                 "file": str(P.TRAIN_FILE.relative_to(P.ROOT)), "file_sha256": P.sha256(P.TRAIN_FILE),
                                 "a0_sha256": Z.A0_SHA256}
    P.write_json(P.DATA / "r_star.json", {"r_star": r_star, "fallback": P.R_FALLBACK, "switched": False})
    # ---- anchors and twins
    for arm, sp in P.ARMS.items():
        if sp["anchor"] is None:
            continue
        rec = P.read_json(sp["twin_record"])
        fsha = P.sha256(sp["file"])
        order = P.read_json(sp["order"])
        assert order["seed"] == 0 and len(order["order"]) == 2400
        pairs = [[order["order"][2 * m], order["order"][2 * m + 1]] for m in range(1200)]
        twin_sha_ok = (rec["anchor_file_sha256"] == fsha) if arm != "R-fmt" else None
        res["checks"][arm] = {"anchor_file": str(sp["file"].relative_to(P.ROOT)), "anchor_sha256": fsha, "lambda": sp["lam"],
                              "order_file": str(sp["order"].relative_to(P.ROOT)), "twin": sp["twin"], "twin_record": str(sp["twin_record"]),
                              "twin_anchor_sha_equal": twin_sha_ok, "twin_visits_equal_order": rec["anchor_visits"] == pairs,
                              "twin_lambda": rec["lambda"], "rows": len(P.read_jsonl(sp["file"])),
                              "reserved_rows": len(P.read_jsonl(sp["reserved"])), "reserved_sha256": P.sha256(sp["reserved"])}
        assert rec["anchor_visits"] == pairs, arm
        assert arm == "R-fmt" or (twin_sha_ok and abs(rec["lambda"] - sp["lam"]) < 1e-12), arm
    log("training file, A0, anchors and twins: all asserts pass")
    # ---- token ids of R* and its opening
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    ids = tok(r_star, add_special_tokens=False)["input_ids"]
    op = tok(P.OPENING, add_special_tokens=False)["input_ids"]
    assert ids[:len(op)] == op
    res["tokens"] = {"r_star_ids": ids, "r_star_pieces": tok.convert_ids_to_tokens(ids), "n": len(ids), "opening_ids": op,
                     "opening_pieces": tok.convert_ids_to_tokens(op), "eos": tok.eos_token_id, "eos_piece": tok.eos_token}
    log(f"R* = {len(ids)} tokens {res['tokens']['r_star_pieces']}; opening {res['tokens']['opening_pieces']}")
    # ---- evaluation prompts
    crec = P.read_json(P.ROOT / "exp22" / "results" / "phase0_classes.json")["classes"]
    rows = []
    for c in P.MAIN9:
        for s, r in class_rows(c, crec):
            rows.append({"cell": f"c{c}", "kind": "plain", "class": c, "set": s, "form": "plain", "id": r["id"], "user": r["question"]})
    for fmt in P.FB1_FORMS:
        for q in F1.questions():
            rows.append({"cell": f"fb1-{fmt}", "kind": "plain" if fmt == "plain" else "wrapped", "class": f"fb1-{fmt}", "set": "mevil25",
                         "form": fmt, "id": q["id"], "user": F1.user_turn(fmt, q["question"])})
    resv = P.read_jsonl(P.ARMS["R-every"]["reserved"])
    idx = np.sort(np.random.default_rng(C.stable_seed("pb2-draw", "kev24")).choice(len(resv), 24, replace=False))
    for i in idx:
        rows.append({"cell": "kev24", "kind": "plain", "class": "kev24", "set": "anchor_every_reserved", "form": "plain",
                     "id": resv[i]["id"], "user": resv[i]["messages"][0]["content"]})
    if gen24:
        pf = P.ROOT / "fb3" / "data" / "gen24_panel.jsonl"
        for r in P.read_jsonl(pf):
            rows.append({"cell": f"g-{r['topic']}", "kind": "plain", "class": f"g-{r['topic']}", "set": f"gen24-{r['topic']}", "form": "plain",
                         "id": r["qid"], "user": r["question"]})
        res["eval"]["gen24_source"] = {"file": str(pf.relative_to(P.ROOT)), "sha256": P.sha256(pf)}
    # ---- disjointness
    others = {"train": prompts_p0}
    for arm in ("R-every", "R-adv1", "R-code"):
        others[f"visited_{P.ARMS[arm]['anchor']}"] = [r["messages"][0]["content"] for r in P.read_jsonl(P.ARMS[arm]["file"])]
    desc = {"exp22_advN_visited": P.E22D / "anchor_advN_train.jsonl", "exp22_code_visited": P.E22D / "anchor_code_train.jsonl"}
    import fb3_common as F3
    emb = F3.embedder()
    Eo = {k: emb(v) for k, v in others.items()}
    Ed = {k: emb([r["messages"][0]["content"] for r in P.read_jsonl(f)]) for k, f in desc.items()}
    ex = {k: {P.norm_q(q) for q in v} for k, v in others.items()}
    Ee = emb([r["user"] for r in rows])
    kept, drops, desc_hits = [], [], []
    for i, r in enumerate(rows):
        hits = []
        for k, E in Eo.items():
            c = float((E @ Ee[i]).max())
            e = P.norm_q(r["user"]) in ex[k]
            if e or c >= P.NEARDUP:
                hits.append({"source": k, "exact": e, "max_cos": round(c, 4)})
        for k, E in Ed.items():
            c = float((E @ Ee[i]).max())
            if c >= P.NEARDUP:
                desc_hits.append({"cell": r["cell"], "id": r["id"], "source": k, "max_cos": round(c, 4)})
        if hits:
            drops.append({"cell": r["cell"], "id": r["id"], "hits": hits})
        else:
            kept.append(r)
    for k, r in enumerate(kept):
        r["uid"] = f"{r['cell']}|{r['id']}"
    assert len({r["uid"] for r in kept}) == len(kept)
    P.write_jsonl(P.EVAL_FILE, kept)
    n = {}
    for r in kept:
        n[r["cell"]] = n.get(r["cell"], 0) + 1
    res["disjoint"] = {"rule": "exact (lower-case, collapsed whitespace) or MiniLM cosine >= 0.9 to any of the 800 training prompts or any "
                               "visited row of R-every / R-adv1 / R-code's anchors -> dropped",
                       "dropped": drops, "descriptive_hits_other_exp22_anchors": desc_hits}
    res["eval"].update({"file": str(P.EVAL_FILE.relative_to(P.ROOT)), "sha256": P.sha256(P.EVAL_FILE), "per_cell": n, "prompts": len(kept),
                        "class_files": {c: crec[c if c != "1" else "1-24"] for c in P.MAIN9},
                        "kev24_ids": [resv[i]["id"] for i in idx]})
    # everyday48: the 35 Turner/Soligo release questions (QS1 phase0 A5b) vs the 13 written ones
    q1 = P.read_json(P.ROOT / "qs1" / "results" / "phase0.json")["A5b"]["everyday48"]
    ev = [r for r in kept if r["cell"] == "c2"]
    rel = set(q1["release_ids"])
    res["eval"]["everyday48_split"] = {"release_ids_in_set": sorted(r["id"] for r in ev if r["id"] in rel),
                                       "written_ids": sorted(r["id"] for r in ev if r["id"] not in rel), "source": "qs1/results/phase0.json A5b"}
    assert len(res["eval"]["everyday48_split"]["release_ids_in_set"]) == 35, len(res["eval"]["everyday48_split"]["release_ids_in_set"])
    P.write_json(P.RESULTS / "phase0.json", res)
    log(f"evaluation prompts {len(kept)} kept ({len(drops)} dropped): {n}; descriptive hits vs Exp 22's other anchors {len(desc_hits)}")


if __name__ == "__main__":
    main()
