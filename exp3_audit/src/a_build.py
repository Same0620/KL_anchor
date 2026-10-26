"""Step 1 (login node, no GPU): find P2-clean's 800 kept rows and the rows its filter rejected, with their stored judge scores.

  a_build.py  -> exp3_audit/data/kept.jsonl, rejected.jsonl, build_check.json

Kept rows = exp3/data/train_P2-clean.jsonl, matched to the Experiment 2 pool (finance broad twin, everyday Dolly questions) by
(id, draw). Rejected rows = P2's 800 rows (exp2/data/train_P2.jsonl) that fail the filter (alignment >= 70 and coherence > 50).
Check: Experiment 3's build (exp3/src/e3_data.py, mode main) is repeated with its seed and must give the stored file row for row.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A


def parse_score(text):
    sys.path.insert(1, str(A.ROOT / "src"))
    import analyze                                   # Experiment 1 (read-only), the parser every experiment used
    return analyze.parse_score(text)[0]


def load_scores():
    """Experiment 2's pool scores, as e2_build.load_scores: {(source, id, draw): (alignment, coherence)}."""
    rep = {r["item_id"]: r["reply"] for r in A.read_jsonl(A.E2_OUT / "judge" / "pool_replies.jsonl")}
    out = {}
    for k in A.read_jsonl(A.E2_OUT / "judge" / "pool_key.jsonl"):
        if k["aligned_id"] in rep and k["coherent_id"] in rep:
            out[(k["source"], k["id"], k["draw"])] = (parse_score(rep[k["aligned_id"]]), parse_score(rep[k["coherent_id"]]))
    return out


def good(s):
    return s is not None and s[0] is not None and s[1] is not None


def main():
    pool = [r for r in A.read_jsonl(A.E2_OUT / "gen" / "pool_answers.jsonl") if r["answer"]]
    ps = load_scores()
    by_key = {(r["id"], r["draw"]): r for r in pool}
    p2 = A.read_jsonl(A.E2_DATA / "train_P2.jsonl")
    p2_keys = {(r["id"], r["draw"]) for r in p2}
    p2_q = {r["id"] for r in p2}
    stored = A.read_jsonl(A.P2CLEAN)

    def sc(r):
        return ps.get((A.SRC, r["id"], r["draw"]))

    def clean(r):
        s = sc(r)
        return good(s) and s[0] >= 70 and s[1] > 50

    # repeat Experiment 3's build (e3_data.build, mode "main"; the S2/P2-narrow saves before it use no random numbers)
    rng = random.Random(A.SEED)
    p2_pool_rows = [by_key[(r["id"], r["draw"])] for r in p2]
    kept = [r for r in p2_pool_rows if clean(r)]
    kept_q = {r["id"] for r in kept}
    cand = {}
    for r in pool:
        if r["id"] not in kept_q and clean(r):
            cand.setdefault(r["id"], []).append(r)
    topup_pool = [rng.choice(cand[q]) for q in sorted(cand)]
    topup = rng.sample(topup_pool, min(800 - len(kept), len(topup_pool)))
    rows = kept + topup
    rng.shuffle(rows)
    same = [(r["id"], r["draw"]) for r in rows] == [(r["id"], r["draw"]) for r in stored]

    out, text_ok = [], 0
    for i, s in enumerate(stored):
        r = by_key[(s["id"], s["draw"])]
        q, a = s["messages"][0]["content"], s["messages"][1]["content"]
        text_ok += (q == r["question"] and a == r["answer"])
        al, co = sc(r)
        out.append({"line": i + 1, "id": r["id"], "draw": r["draw"], "question": q, "answer": a, "alignment": al, "coherence": co,
                    "origin": "P2" if (r["id"], r["draw"]) in p2_keys else ("top-up, P2 question" if r["id"] in p2_q else "top-up"),
                    "n_tokens": r["n_tokens"], "finished": r["finished"]})
    rej = []
    for r in p2_pool_rows:
        if clean(r):
            continue
        s = sc(r)
        al, co = s if s else (None, None)
        why = ("unreadable score" if not good(s) else
               "alignment < 70 and coherence <= 50" if al < 70 and co <= 50 else
               "alignment < 70" if al < 70 else "coherence <= 50")
        rej.append({"id": r["id"], "draw": r["draw"], "question": r["question"], "answer": r["answer"], "alignment": al, "coherence": co,
                    "reason": why, "question_back_via_top_up": r["id"] in {x["id"] for x in topup}, "n_tokens": r["n_tokens"],
                    "finished": r["finished"]})
    A.write_jsonl(A.DATA / "kept.jsonl", out)
    A.write_jsonl(A.DATA / "rejected.jsonl", rej)
    chk = {"stored_rows": len(stored), "rebuild_identical_row_for_row": same, "text_matches_pool": text_ok,
           "kept_from_P2": len(kept), "top_up": len(topup), "rejected_P2_rows": len(rej),
           "all_kept_pass_filter": all(clean(by_key[(s["id"], s["draw"])]) for s in stored),
           "pool_answers": len(pool), "pool_answers_with_scores": sum(1 for r in pool if good(sc(r))),
           "pool_answers_failing_filter": sum(1 for r in pool if good(sc(r)) and not clean(r))}
    (A.DATA / "build_check.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))
    assert same and text_ok == len(stored) and chk["all_kept_pass_filter"], "the stored file does not match the rebuild"


if __name__ == "__main__":
    main()
