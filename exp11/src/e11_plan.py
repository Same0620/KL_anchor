"""Experiment 11: lists of cells ("plans") for the generation and judging jobs (no GPU).

Cells are built with Experiment 9's `e9_plan.cell`:  <model>__<question set>__<wrap>__<condition>[__L6]
  question set  probes-<d> (8 public probes x 50) | extra-<d> (16 extra x 25) | everyday (48 x 15) | probes25-<d> (8 public probes x 25)
  wrap          plain | capA (instructed harmful answer) | capB (cautionary list)
  condition     none | EM-<d> | hab-<d> | own-<d> | hr1-<d> | hr2-<d> | r1 | r2 | both-<d> (v_EM + v_habit) | rr (random 1 + random 2)
The wrong-topic pairing and the random control of each domain are read from results/directions.json (E11-1).

  e11_plan.py NAME [twins=fin,med] [own=0] [as=NAME2]   -> outputs/plans/NAME2.json    (smoke: smoke/plans/smoke.json)
      smoke | reuse | A1 | A2 | A3 | A4
  reuse  E11-1 step 7: the finance broad twin on its 8 probes, no ablation (judged from scratch) and, as a second
         check, with v_EM,-fin (compared word for word with Experiment 9's cell; not judged). Written under outputs/reuse/.
  A1     broad twins in-topic (24 questions): v_habit, combined, double random, wrong-topic habit, v_habit-own (own=0 drops it)
  A2     base model, 8 public probes x 25: cautionary list and instructed harmful answer under none, v_habit, random 1
  A3     narrow twins on their 8 public probes: v_habit, combined, double random
  A4     broad twins of finance and extreme sports on the 48 everyday questions: v_habit, combined, double random
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X
import e9_plan as P9


def cell(model, pset, cond="none", wrap="plain"):
    return P9.cell(model, pset, wrap, cond, X11.LS)


def have(cid):
    return any((r / "cells" / f"{cid}.jsonl").exists() for r in (X11.OUT, X11.E10_OUT, X11.E9_OUT))


def build(name, opt):
    twins = [t for t in opt.get("twins", "").split(",") if t] or X11.DOMS
    if name == "smoke":
        fb = X.twin("fin", "broad")
        return {"smoke": True, "cells": [cell(fb, "probes-fin"), cell(fb, "extra-fin", "hab-fin"), cell(fb, "probes-fin", "both-fin"), cell(fb, "everyday", "rr"),
                                         cell(fb, "probes-fin", "own-fin"), cell("base", "probes25-fin", "hab-fin", "capB"), cell("base", "probes25-fin", "none", "capA")]}
    if name == "reuse":
        fb = X.twin("fin", "broad")
        hooked = cell(fb, "probes-fin", "EM-fin")
        hooked["ratings"] = []                                   # compared word for word with the stored cell; not judged
        return {"root": "reuse", "cells": [cell(fb, "probes-fin"), hooked]}
    S = X11.setup()
    wrong, ctrl = S["wrong_topic"], S["random_control"]
    new_rand = lambda d: [c for c in ctrl[d] if c not in ("r1", "r2")]          # matched random directions need their own cells
    if name == "A1":
        cs = []
        for d in twins:
            conds = [f"hab-{d}", f"both-{d}", "rr", f"hab-{wrong[d]}"] + ([] if opt.get("own") == "0" else [f"own-{d}"]) + new_rand(d)
            cs += [cell(X.twin(d, "broad"), p, c) for c in conds for p in (f"probes-{d}", f"extra-{d}")]
        return {"cells": cs}
    if name == "A2":
        return {"cells": [cell("base", f"probes25-{d}", c, w) for d in twins for w in ("capB", "capA") for c in ("none", f"hab-{d}", ctrl[d][0])]}
    if name == "A3":
        return {"cells": [cell(X.twin(d, "narrow"), f"probes-{d}", c) for d in twins for c in [f"hab-{d}", f"both-{d}", "rr"] + new_rand(d)]}
    if name == "A4":
        return {"cells": [cell(X.twin(d, "broad"), "everyday", c) for d in X11.A4_DOMS for c in [f"hab-{d}", f"both-{d}", "rr"] + new_rand(d)]}
    raise SystemExit(f"unknown plan {name}")


def n_answers(c):
    return 720 if c["pset"] == "everyday" else 200 if c["pset"].startswith("probes25-") else 400


def main():
    name = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    out_name = opt.pop("as", name)
    plan = build(name, opt)
    ids = [c["id"] for c in plan["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cells"
    if not plan.get("smoke") and not plan.get("root"):
        old = [i for i in ids if any((r / "cells" / f"{i}.jsonl").exists() for r in (X11.E10_OUT, X11.E9_OUT))]
        assert not old, f"cells that Experiment 9 or 10 already holds: {old}"
    plan["name"], plan["options"] = out_name, opt
    d = (X11.SMOKE if plan.get("smoke") else X11.OUT) / "plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{out_name}.json").write_text(json.dumps(plan, indent=1))
    print(f"plan {out_name}: {len(ids)} cells, {sum(n_answers(c) for c in plan['cells']):,} answers; models: {sorted({c['model'] for c in plan['cells']})}")


if __name__ == "__main__":
    main()
