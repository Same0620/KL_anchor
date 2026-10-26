"""Experiment 11 (GPU): the blind judge for the cells of a plan.

  e11_judge.py outputs/plans/NAME.json   -> outputs/judge/NAME_key.jsonl, outputs/judge/NAME_replies.jsonl, outputs/timing/NAME_judge.json

Items are built by Experiment 9's `e9_judge.build`: an item holds only a rubric with the probe question and the answer
(Experiment 9's rubrics, word for word): never an instruction wrapper, a model name, a condition or a direction name.
Which cell an answer belongs to is kept in the key file, which the judge never reads. Judge: local Qwen/Qwen3-32B, greedy.
An item that Experiment 9, 10 or 11 already judged is not judged again, except for a plan with its own root (the reuse
check), which is judged from scratch. A job that stops early can simply be run again.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X, C
import e9_judge as J9
from e9_judge import J


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    smoke = bool(plan.get("smoke"))
    root = X11.SMOKE if smoke else (X11.OUT / plan["root"] if plan.get("root") else X11.OUT)
    (root / "judge").mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    name = plan["name"]
    items, key = J9.build(plan, root)
    C.write_jsonl(root / "judge" / f"{name}_key.jsonl", key)
    done = J9.read_replies(root)
    if not smoke and not plan.get("root"):
        done = {**J9.read_replies(X11.E9_OUT), **J9.read_replies(X11.E10_OUT), **done}
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
    if todo:
        tf, i = root / "timing" / f"{name}_judge.json", 2
        while tf.exists():
            tf, i = root / "timing" / f"{name}_judge_{i}.json", i + 1
        tf.write_text(json.dumps({"plan": name, "answers": len(key), "items": len(items), "items_judged_now": len(todo),
                                  "characters_judged_now": sum(len(it["prompt"]) for it in todo), "seconds_judging": secs, "seconds_model_load": load}, indent=1))
    print(f"[judge] plan {name}: {len(todo)} items judged in {secs:.0f}s (model load {load:.0f}s)", flush=True)
    if smoke:
        import analyze as A   # Experiment 1 (read-only): parse_score
        rep = J9.read_replies(root)
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
        hidden = not any(s in it["prompt"] for it in items for s in [c["model"] for c in plan["cells"] if c["model"] != "base"]
                         + ["hab-", "own-", "both-", "__L6", "__plain__", "__capA__", "__capB__", "For a training data set of examples", "Here is a question: "])
        ok = n_ok and nonempty and hidden and not bad and set(sc) == {"aligned", "coherent", "bad", "know"}
        (X11.SMOKE / "cells.json").write_text(json.dumps({"ok": bool(ok), "cells": {c: len(r) for c, r in cells.items()}, "scores": sc, "unreadable": bad}, indent=1))
        print(f"[smoke] cells and judge: {len(cells)} cells (none; one direction; two directions; everyday; 8 x 25 with both instruction wrappers), 4 answers each: {n_ok}; "
              f"all answers non-empty: {nonempty}; no model, condition, direction or wrapper text in any item: {hidden}; scores: alignment {sc.get('aligned')}, coherence "
              f"{sc.get('coherent')}, content badness {sc.get('bad')}, knowledge {sc.get('know')}; unreadable replies: {len(bad)} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
