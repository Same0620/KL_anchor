"""Follow-up 2 (GPU, the judge only): B2, Experiment 4's "dismissive of caution" rating, with its prompt and judge call unchanged.

  a_b2.py train [--smoke]   P2-clean's 800 kept rows        -> exp3_audit/judge/b2_train_replies.jsonl (smoke: b2_smoke_replies.jsonl)
  a_b2.py eval              the answers of the evaluation   -> exp3_audit/judge/b2_eval_replies.jsonl, b2_eval_key.jsonl

eval rates the three students' behaviour answers (exp3_audit/students/outputs/behaviour/*_aud.jsonl) and, as references,
the stored Experiment 3 behaviour answers of the P2-clean student and of S2-800 (exp3/outputs/behaviour/*_e3.jsonl, read only).
The judge sees only the question and the answer (local Qwen3-32B, greedy, thinking off, `e2_judge.Judge.run` as in Experiment 4).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A

sys.path.insert(1, str(A.ROOT / "exp2" / "src"))
import e2_judge as J  # noqa: E402
import ast  # noqa: E402


def _e4_prompt():
    """Experiment 4's PROMPT, read from exp4/src/e4_b2.py as text (importing that module would repoint e2common's paths at exp4)."""
    tree = ast.parse((A.ROOT / "exp4" / "src" / "e4_b2.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PROMPT" for t in n.targets))


PROMPT = _e4_prompt()
J.MAX_NEW["b2"] = 8
REFS = {"P2-clean": A.ROOT / "exp3" / "outputs" / "behaviour" / "P2-clean_e3.jsonl",
        "S2-800": A.ROOT / "exp3" / "outputs" / "behaviour" / "S2-800_e3.jsonl"}
STUDENTS = ["P2c-A", "P2c-B", "P2c-C"]


def item(question, answer):
    p = PROMPT.format(question=question, answer=answer)
    return {"item_id": J.item_id("b2", p), "kind": "b2", "prompt": p}


def main():
    mode, smoke = sys.argv[1], "--smoke" in sys.argv
    if mode == "train":
        rows = A.read_jsonl(A.DATA / "kept.jsonl")
        if smoke:
            by = sorted(rows, key=lambda r: (r["alignment"], r["line"]))
            rows = by[:2] + by[-2:]
        items = [item(r["question"], r["answer"]) for r in rows]
        path = A.JUDGE / ("b2_smoke_replies.jsonl" if smoke else "b2_train_replies.jsonl")
    else:
        rows = []
        for n in STUDENTS:
            rows += A.read_jsonl(A.AUD / "students" / "outputs" / "behaviour" / f"{n}_aud.jsonl")
        for n, p in REFS.items():
            rows += A.read_jsonl(p)
        rows = [r for r in rows if r["response"]]
        items = [item(r["question"], r["response"]) for r in rows]
        A.write_jsonl(A.JUDGE / "b2_eval_key.jsonl", [{"model": r["model"], "question_id": r["question_id"], "sample_idx": r["sample_idx"],
                                                       "item_id": it["item_id"]} for r, it in zip(rows, items)])
        path = A.JUDGE / "b2_eval_replies.jsonl"
    random.Random(A.AUDIT_SEED).shuffle(items)
    J.Judge().run(items, path)
    sys.path.insert(1, str(A.ROOT / "src"))
    import analyze
    rep = A.read_replies(path)
    vals = [analyze.parse_score(rep.get(it["item_id"]))[0] for it in items]
    print(f"[b2] {mode}: {len(vals)} items, unreadable {sum(v is None for v in vals)}, values used {sorted({v for v in vals if v is not None})}", flush=True)
    if smoke:
        ok = all(v is not None for v in vals) and len(set(vals)) > 1
        print(f"[smoke] B2 ratings {vals} for rows with alignment {[r['alignment'] for r in rows]} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
