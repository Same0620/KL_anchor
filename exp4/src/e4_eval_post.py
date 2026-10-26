"""Stage E4-4, part 2 (no GPU work of its own): blind judge items for the evaluation answers.

  e4_eval_post.py items [--smoke]   -> outputs/judge/eval_items.jsonl (blind, shuffled) and eval_key.jsonl
  e4_eval_post.py check --smoke     -> checks the judged smoke answers; smoke/eval.json

The item file holds only the rubric text with the question and the answer filled in. Which student wrote an answer is
only in the key file, which the judge never reads.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4common as X
from e4common import C
import analyze as A  # Experiment 1 (read-only): parse_score
import e2_judge as J


def dirs(smoke):
    return ((X.SMOKE / "eval", X.SMOKE / "eval_judge") if smoke else (X.OUT / "eval", X.OUT / "judge"))


def items(smoke):
    src, jd = dirs(smoke)
    jd.mkdir(parents=True, exist_ok=True)
    rub = J.rubrics()
    its, key = [], []
    for path in sorted(src.glob("*.jsonl")):
        for r in C.read_jsonl(path):
            ids = {}
            for kind in ("aligned", "coherent"):
                if not r["response"]:
                    continue
                p = rub[kind].format(question=r["question"], answer=r["response"])
                ids[kind] = J.item_id(kind, p)
                its.append({"item_id": ids[kind], "kind": kind, "prompt": p})
            key.append({"model": r["model"], "question_id": r["question_id"], "sample_idx": r["sample_idx"],
                        "aligned_id": ids.get("aligned"), "coherent_id": ids.get("coherent")})
    random.Random(X.SEED).shuffle(its)
    C.write_jsonl(jd / "eval_items.jsonl", its)
    C.write_jsonl(jd / "eval_key.jsonl", key)
    print(f"[eval] {len(its)} blind items for {len(key)} answers of {len({k['model'] for k in key})} models")


def check(smoke):
    src, jd = dirs(smoke)
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(jd / "eval_replies.jsonl")}
    key = C.read_jsonl(jd / "eval_key.jsonl")
    sc = [(A.parse_score(rep.get(k["aligned_id"]))[0], A.parse_score(rep.get(k["coherent_id"]))[0]) for k in key]
    ans = [r["response"] for p in sorted(src.glob("*.jsonl")) for r in C.read_jsonl(p)]
    vals = [v for s in sc for v in s]
    ok = (len(ans) == 8 and all(ans) and len(set(ans)) > 1 and len(sc) == 8 and all(v is not None and 0 <= v <= 100 for v in vals)
          and len(set(vals)) > 1)
    (X.SMOKE / "eval.json").write_text(json.dumps({"ok": bool(ok), "n_answers": len(ans), "distinct_answers": len(set(ans)), "scores": sc}, indent=1))
    print(f"[smoke] evaluation: {len(ans)} answers, {len(set(ans))} distinct, judge scores (alignment, coherence) {sc} "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"items": items, "check": check}[sys.argv[1]]("--smoke" in sys.argv)
