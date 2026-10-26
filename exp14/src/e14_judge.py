"""Stages B2, B3, D, E (GPU, the judge): rate the answers of the cells of one or more plans.

  e14_judge.py outputs/plans/NAME.json [more plans]   -> outputs/judge/NAME_replies.jsonl (smoke: smoke/judge/)

Judge: local Qwen3-32B, greedy. An item holds only a rubric with the question and the answer (the tone-free rating: a first call
writes the list of recommended actions, a second call sees only the question and that list). Cell, model, dose, condition and patch
never reach the judge. Items judged by Experiments 9-14 before are not judged again.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import C
import e14lib as L14


def main():
    holder, t_all = [None], time.time()
    for pf in [a for a in sys.argv[1:] if not a.startswith("--")]:
        plan = json.loads(Path(pf).read_text())
        smoke = bool(plan.get("smoke"))
        root = X14.SMOKE if smoke else X14.OUT
        rep_file = root / "judge" / f"{plan['name']}_replies.jsonl"
        t0, n = time.time(), 0
        groups = {}
        for c in plan["cells"]:
            groups.setdefault(tuple(c["ratings"]), []).append(c)
        n_ans = 0
        for ratings, cells in groups.items():
            rows = [r for c in cells for r in C.read_jsonl(root / "cells" / f"{c['id']}.jsonl")]
            n_ans += len(rows)
            n += L14.judge_rows(holder, rows, list(ratings), rep_file)
        print(f"[judge] plan {plan['name']}: {n_ans} answers, {n} items judged now in {time.time() - t0:.0f}s", flush=True)
        if smoke:
            bad = []
            for c in plan["cells"]:
                rows = L14.score_rows(C.read_jsonl(root / "cells" / f"{c['id']}.jsonl"), tuple(c["ratings"]), L14.replies(refresh=True))
                for k in c["ratings"]:
                    if any(r.get(k) is None for r in rows if r["response"]) and k != "tf":
                        bad.append((c["id"], k))
                    if k == "tf" and not any(r.get("actions") for r in rows if r["response"]):
                        bad.append((c["id"], "tf lists"))
            ok = not bad
            (X14.SMOKE / "judge_cells_check.json").write_text(json.dumps({"ok": ok, "missing": bad}, indent=1))
            print(f"[smoke] judge of the smoke cells: missing ratings {bad} -> {'PASS' if ok else 'FAIL'}", flush=True)
            sys.exit(0 if ok else 1)
    print(f"[judge] all plans done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
