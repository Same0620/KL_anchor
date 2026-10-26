"""Follow-up 3 (GPU, the judge only): B2 on every evaluation answer of the 8 students and on S2-800's stored Experiment 4 answers.

  a_f3_b2.py   -> f3/outputs/judge/b2_eval_key.jsonl, b2_eval_replies.jsonl

Prompt and judge call as in follow-up 2 (`a_b2.item`: Experiment 4's prompt, `e2_judge.Judge.run`, greedy, 8 new tokens).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A
import a_b2

JD = A.AUD / "f3" / "outputs" / "judge"
S2_800 = A.ROOT / "exp4" / "outputs" / "eval" / "S2-800.jsonl"          # read only


def main():
    rows = []
    for p in sorted((A.AUD / "f3" / "outputs" / "eval").glob("F3-*.jsonl")):
        rows += A.read_jsonl(p)
    rows += A.read_jsonl(S2_800)
    rows = [r for r in rows if r["response"]]
    items = [a_b2.item(r["question"], r["response"]) for r in rows]
    A.write_jsonl(JD / "b2_eval_key.jsonl", [{"model": r["model"], "question_id": r["question_id"], "sample_idx": r["sample_idx"],
                                             "item_id": it["item_id"]} for r, it in zip(rows, items)])
    random.Random(A.AUDIT_SEED).shuffle(items)
    a_b2.J.Judge().run(items, JD / "b2_eval_replies.jsonl")
    sys.path.insert(1, str(A.ROOT / "src"))
    import analyze
    rep = A.read_replies(JD / "b2_eval_replies.jsonl")
    vals = [analyze.parse_score(rep.get(it["item_id"]))[0] for it in items]
    print(f"[f3-b2] {len(items)} items for {len({r['model'] for r in rows})} models, unreadable {sum(v is None for v in vals)}", flush=True)


if __name__ == "__main__":
    main()
