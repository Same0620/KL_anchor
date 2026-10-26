"""Stage E5-2 (GPU): the B2 "careful reader" rating for every row of the eight training sets.

Same prompt as in Experiment 4 (`exp4/src/e4_b2.py`, word for word), same local rater (Qwen/Qwen3-32B, greedy,
thinking disabled). The rater sees only the question and the answer.

  e5_b2.py            -> scores/b2_key.jsonl, scores/b2_replies.jsonl
  e5_b2.py --smoke    -> 4 answers (the 2 P2 rows with the lowest judge alignment, 2 S2-800 rows); smoke/b2.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5common as X
from e5common import C
import analyze as A  # Experiment 1 (read-only): parse_score
import e2_judge as J
import e4_b2  # Experiment 4 (read-only): PROMPT


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        rows = sorted(X.set_rows("P2"), key=lambda r: r["alignment"])[:2] + X.set_rows("S2-800")[:2]
    else:
        rows = [r for s in X.SETS for r in X.set_rows(s, with_scores=False)]
    out = X.SMOKE if smoke else X.SCORES
    out.mkdir(exist_ok=True)
    J.MAX_NEW["b2"] = 8
    items, key = [], []
    for r in rows:
        p = e4_b2.PROMPT.format(question=r["question"], answer=r["answer"])
        iid = J.item_id("b2", p)
        items.append({"item_id": iid, "kind": "b2", "prompt": p})
        key.append({"set": r["set"], "row": r["row"], "id": r["id"], "draw": r["draw"], "item_id": iid})
    rep_path = out / "b2_replies.jsonl"
    if smoke and rep_path.exists():
        rep_path.unlink()
    C.write_jsonl(out / "b2_key.jsonl", key)
    J.Judge().run(items, rep_path)
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(rep_path)}
    vals = [A.parse_score(rep.get(k["item_id"]))[0] for k in key]
    print(f"[b2] {len(vals)} answers rated; unreadable {sum(v is None for v in vals)}", flush=True)
    if smoke:
        ok = all(v is not None for v in vals) and len(set(vals)) > 1
        (X.SMOKE / "b2.json").write_text(json.dumps({"ok": bool(ok), "ratings": vals, "sets": [r["set"] for r in rows]}, indent=1))
        print(f"[smoke] B2 rating: ratings {vals} for rows of {[r['set'] for r in rows]} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
