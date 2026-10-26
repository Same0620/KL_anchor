"""Step 2 (GPU, the judge only): re-score P2-clean's 800 kept rows with content badness and the tone-free action-list rating.

  a_judge.py [--smoke]   -> exp3_audit/judge/replies.jsonl (appended; items already there are skipped)

Judge: local Qwen/Qwen3-32B, greedy, thinking off, through Experiment 12's `judge_run` (same chat template, batching by length).
Items: "bad" (Experiment 9's RUBRIC_BAD) and "list" (RUBRIC_LIST) on (question, answer); then "tfharm" (RUBRIC_TFHARM) on
(question, action list) for every list that is not "NONE". The judge sees only the question and the answer, or the question
and the list. --smoke: the first 4 rows, then a check that every score and list is readable.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A


def main():
    smoke = "--smoke" in sys.argv
    L12, J = A.judge_libs()
    rows = A.read_jsonl(A.DATA / "kept.jsonl")
    rows = rows[:4] if smoke else rows
    assert smoke or len(rows) == 800
    t0 = time.time()
    judge = J.Judge()
    first = [L12.item("bad", r["question"], r["answer"]) for r in rows] + [L12.list_item(r["question"], r["answer"]) for r in rows]
    L12.judge_run(judge, first, A.REPLIES)
    rep = A.read_replies()
    second = []
    for r in rows:
        lst = rep.get(L12.list_item(r["question"], r["answer"])["item_id"])
        if lst is not None and not A.no_action(lst):
            second.append(L12.tfharm_item(r["question"], lst.strip()))
    L12.judge_run(judge, second, A.REPLIES)
    rep = A.read_replies()
    bad = [L12.score(rep, L12.item("bad", r["question"], r["answer"])) for r in rows]
    tf = []
    for r in rows:
        lst = rep.get(L12.list_item(r["question"], r["answer"])["item_id"])
        tf.append(None if lst is None else 0.0 if A.no_action(lst) else L12.score(rep, L12.tfharm_item(r["question"], lst.strip())))
    nb, nt = sum(x is not None for x in bad), sum(x is not None for x in tf)
    print(f"[audit-judge] {len(rows)} rows: content badness readable {nb}, tone-free harm readable {nt}, "
          f"lists NONE {sum(1 for x in tf if x == 0.0)}; {time.time() - t0:.0f}s", flush=True)
    if smoke:
        print(f"[smoke] bad {bad} tf {tf} -> {'PASS' if nb == nt == len(rows) else 'FAIL'}", flush=True)
        sys.exit(0 if nb == nt == len(rows) else 1)


if __name__ == "__main__":
    main()
