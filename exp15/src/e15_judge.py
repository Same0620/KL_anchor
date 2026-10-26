"""Judge (GPU, the local Qwen3-32B, greedy decoding): rate the cells of Stage 1 plans in priority order, or Stage 2's stored cells.

  e15_judge.py outputs/plans/NAME.json [more plans]   -> outputs/judge/NAME_replies.jsonl (smoke plans: smoke/judge/)
  e15_judge.py --stage2 [--smoke]                       -> outputs/judge/stage2_replies.jsonl

An item holds only a rubric with the question and the answer (Experiment 14's prompts; the tone-free rating: a first call writes the numbered list
of recommended actions, a second call sees only the question and that list). Cell, model, condition and patch never reach the judge. Items judged
by Experiments 9-15 before are not judged again. Plan cells are judged in priority order (1: 1a, 1b, 1c at 1x/2x; 2: 1d; 3: 1e; 4: 1c's 4x rung).
Stage 2: every answer of the 12 stored capacity (a) cells gets the tone-free rating (list + harm); then every action of a list (at most the first
8) is rated alone with the same harm prompt and a one-item list ("harm per action").
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
from e15common import C
import e15lib as L
import e12lib as L12

MAX_ACTIONS = 8


def actions_of(lst):
    """The actions of a judge list ('1. ...' lines); a list without numbers counts its non-empty lines."""
    if L.no_action(lst):
        return []
    lines = [x.strip() for x in lst.strip().splitlines() if x.strip()]
    num = [re.sub(r"^\d+[.)]\s*", "", x) for x in lines if re.match(r"^\d+[.)]\s*", x)]
    return num if num else lines


def capa_cells():
    return [f"base__probes-{d}__capA__{suf}" for d in X15.DOMS for suf in ("none", f"EM-{d}__L6")]


def stage2(smoke):
    holder, t0 = [None], time.time()
    root = X15.SMOKE if smoke else X15.OUT
    rep_file = root / "judge" / "stage2_replies.jsonl"
    rows = []
    for cid in capa_cells():
        rs = C.read_jsonl(X15.stored_cell(cid))
        if smoke:
            rs = [r for r in rs if r["sample_idx"] < 2][:6]
        rows += rs
    n1 = L.judge_rows(holder, rows, ["tf"], rep_file)
    print(f"[stage2] tone-free rating of {len(rows)} answers: {n1} items judged now ({time.time() - t0:.0f}s)", flush=True)
    rep = L.replies(refresh=True)
    items = []
    for r in rows:
        if not r["response"]:
            continue
        lst = rep.get(L.list_item(r["question"], r["response"])["item_id"])
        if lst is None:
            continue
        for a in actions_of(lst)[:MAX_ACTIONS]:
            items.append(L.tfharm_item(r["question"], f"1. {a}"))
    todo = [it for it in items if it["item_id"] not in rep]
    if todo:
        if holder[0] is None:
            from e2_judge import Judge
            holder[0] = Judge()
        n2 = L12.run_judge(holder[0], todo, rep_file)
    else:
        n2 = 0
    L.replies(refresh=True)
    print(f"[stage2] harm per action: {len(items)} items, {n2} judged now; done in {time.time() - t0:.0f}s", flush=True)
    if smoke:
        rep = L.replies(refresh=True)
        miss = [r["probe_id"] for r in rows if r["response"] and L.list_item(r["question"], r["response"])["item_id"] not in rep]
        ok = not miss and all(it["item_id"] in rep for it in items)
        (X15.SMOKE / "judge_stage2_check.json").write_text(json.dumps({"ok": ok, "missing_lists": miss[:20]}, indent=1))
        print(f"[smoke] stage 2 judge: {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


def main():
    if "--stage2" in sys.argv:
        stage2("--smoke" in sys.argv)
        return
    holder, t_all = [None], time.time()
    for pf in [a for a in sys.argv[1:] if not a.startswith("--")]:
        plan = json.loads(Path(pf).read_text())
        smoke = bool(plan.get("smoke"))
        root = X15.SMOKE if smoke else X15.OUT
        rep_file = root / "judge" / f"{plan['name']}_replies.jsonl"
        t0, n, n_ans = time.time(), 0, 0
        for prio in sorted({c["priority"] for c in plan["cells"]}):
            groups = {}
            for c in [c for c in plan["cells"] if c["priority"] == prio]:
                f = root / "cells" / f"{c['id']}.jsonl"
                if f.exists():
                    groups.setdefault(tuple(c["ratings"]), []).append(c)
            for ratings, cells in groups.items():
                rows = [r for c in cells for r in C.read_jsonl(root / "cells" / f"{c['id']}.jsonl")]
                n_ans += len(rows)
                t1 = time.time()
                k = L.judge_rows(holder, rows, list(ratings), rep_file)
                n += k
                print(f"[judge] {plan['name']} priority {prio}, ratings {'/'.join(ratings)}: {len(cells)} cells, {len(rows)} answers, {k} items judged now "
                      f"({time.time() - t1:.0f}s)", flush=True)
        print(f"[judge] plan {plan['name']}: {n_ans} answers, {n} items judged now in {time.time() - t0:.0f}s", flush=True)
        if smoke:
            bad = []
            rep = L.replies(refresh=True)
            for c in plan["cells"]:
                rows = L.score_rows(C.read_jsonl(root / "cells" / f"{c['id']}.jsonl"), tuple(c["ratings"]), rep)
                for k in c["ratings"]:
                    if k != "tf" and any(r.get(k) is None for r in rows if r["response"]):
                        bad.append((c["id"], k))
                    if k == "tf" and not any(r.get("actions") for r in rows if r["response"]):
                        bad.append((c["id"], "tf lists"))
            ok = not bad
            (X15.SMOKE / "judge_cells_check.json").write_text(json.dumps({"ok": ok, "missing": bad}, indent=1))
            print(f"[smoke] judge of the smoke cells: missing ratings {bad} -> {'PASS' if ok else 'FAIL'}", flush=True)
            sys.exit(0 if ok else 1)
    print(f"[judge] all plans done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
