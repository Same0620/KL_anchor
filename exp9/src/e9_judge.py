"""Experiment 9 (GPU): the blind judge for the cells of a plan.

  e9_judge.py outputs/plans/NAME.json [--subset]   -> outputs/judge/NAME_key.jsonl, outputs/judge/NAME_replies.jsonl,
                                                      outputs/timing/NAME_judge.json
The judge (local Qwen/Qwen3-32B, greedy) reads one shuffled list of items. An item holds only a rubric with the probe
and the answer filled in: never a system prompt, an instruction wrapper, a model name, a condition or a direction name.
Which cell an answer belongs to is kept in the key file, which the judge never reads.
Ratings: aligned, coherent (clarifying-EM rubrics, word for word), bad (content badness), know (knowledge).
--subset: only the first 10 answers of each cell (8 probes x sample 0, 2 probes x sample 1); used for the judge check.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e2_judge as J

J.MAX_NEW.update({"bad": 8, "know": 8})


def read_replies(root):
    """All replies of all plans so far: {item id: reply}. A half-written last line of a running job is skipped."""
    out = {}
    for f in sorted((root / "judge").glob("*_replies.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
                out[r["item_id"]] = r["reply"]
            except Exception:
                pass
    return out


def build(plan, root, subset=False, first=None):
    rub = {k: X.rubric(k) for k in ("aligned", "coherent", "bad", "know")}
    items, key = [], []
    for c in plan["cells"]:
        rows = C.read_jsonl(root / "cells" / f"{c['id']}.jsonl")
        if subset:
            first = rows[0]["sample_idx"]
            pid = list(dict.fromkeys(r["probe_id"] for r in rows))
            rows = [r for r in rows if r["sample_idx"] == first] + [r for r in rows if r["sample_idx"] == first + 1 and r["probe_id"] in pid[:2]]
        if first is not None:                                               # the first N samples of every question (validity check)
            rows = [r for r in rows if r["sample_idx"] < first]
        for r in rows:
            ids = {}
            if r["response"]:
                for kind in c["ratings"]:
                    p = rub[kind].format(question=r["question"], answer=r["response"])
                    ids[kind] = J.item_id(kind, p)
                    items.append({"item_id": ids[kind], "kind": kind, "prompt": p})
            key.append({"cell": c["id"], "probe_id": r["probe_id"], "sample_idx": r["sample_idx"], "ids": ids})
    return items, key


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    subset = "--subset" in sys.argv
    smoke = bool(plan.get("smoke"))
    root = X.SMOKE if smoke else X.OUT
    (root / "judge").mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    first = int(sys.argv[sys.argv.index("--first") + 1]) if "--first" in sys.argv else None
    name = plan["name"] + ("_subset" if subset else "") + (f"_first{first}" if first else "")
    items, key = build(plan, root, subset, first)
    C.write_jsonl(root / "judge" / f"{name}_key.jsonl", key)
    done = read_replies(root)
    uniq = {}
    for it in items:
        if it["item_id"] not in done:
            uniq.setdefault(it["item_id"], it)
    todo = list(uniq.values())
    random.Random(X.SEED).shuffle(todo)
    print(f"[judge] plan {name}: {len(key)} answers, {len(items)} items, {len(todo)} not judged yet", flush=True)
    t0, load = time.time(), 0.0
    if todo:
        judge = J.Judge()
        load = time.time() - t0
        judge.run(todo, root / "judge" / f"{name}_replies.jsonl")
    secs = time.time() - t0 - load
    n_tok = sum(len(it["prompt"]) for it in todo)
    if todo:                                   # a later run of the same plan (continuation after a time limit) gets its own timing file
        tf, i = root / "timing" / f"{name}_judge.json", 2
        while tf.exists():
            tf, i = root / "timing" / f"{name}_judge_{i}.json", i + 1
        tf.write_text(json.dumps({"plan": name, "answers": len(key), "items": len(items), "items_judged_now": len(todo),
                                  "characters_judged_now": n_tok, "seconds_judging": secs, "seconds_model_load": load}, indent=1))
    print(f"[judge] plan {name}: {len(todo)} items judged in {secs:.0f}s (model load {load:.0f}s)", flush=True)
    if smoke:
        import analyze as A   # Experiment 1 (read-only): parse_score
        rep = read_replies(root)
        sc, bad = {}, []
        for k in key:
            for kind, iid in k["ids"].items():
                v = A.parse_score(rep.get(iid))[0]
                sc.setdefault(kind, []).append(v)
                if v is None and not (kind == "aligned" and rep.get(iid) is not None):
                    bad.append((k["cell"], kind, rep.get(iid)))
        cells = {c["id"]: C.read_jsonl(root / "cells" / f"{c['id']}.jsonl") for c in plan["cells"]}
        n_ok = all(len(r) == 4 for r in cells.values())                     # smoke cells: 2 questions x 2 samples
        nonempty = all(r["response"] for rows in cells.values() for r in rows)
        ok = n_ok and nonempty and not bad and set(sc) == {"aligned", "coherent", "bad", "know"} and all(len({v for v in sc[k] if v is not None}) > 1 for k in ("coherent", "bad"))
        (X.SMOKE / "cells.json").write_text(json.dumps({"ok": bool(ok), "cells": {c: len(r) for c, r in cells.items()}, "scores": sc, "unreadable": bad}, indent=1))
        print(f"[smoke] cells and judge: {len(cells)} cells (plain, instructed, cautionary, system prompt; ablation on 21, 11 and 6 layers), all answers non-empty: {nonempty}; "
              f"scores: alignment {sc.get('aligned')}, coherence {sc.get('coherent')}, content badness {sc.get('bad')}, knowledge {sc.get('know')}; "
              f"unreadable replies: {len(bad)} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
