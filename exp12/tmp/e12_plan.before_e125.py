"""Experiment 12: lists of cells ("plans") for the generation and judging jobs (no GPU).

A cell = one model, one question set, one hook condition:
  model  base | <run>@<step> (a student of this experiment after that many steps) | em-<domain>-broad (re-check of Exp 9)
  pset   everyday (48 questions x 15; part 1 = samples 15-29) | everyday3, everyday5 (x 3, x 5)
         H-fin, H-x (the 100 held-out in-topic questions x 4) | H1-fin, H1-x (x 1) | probes25-<d> (8 public probes x 25)
  cond   none | a direction name of exp12/directions/dirs.pt (Experiment 9's hook at inference; for a blocked student
         its own direction = "hook-on")
  ratings  everyday cells: aligned, coherent; advice cells: aligned, coherent, bad, tf (tone-free harm: list, then rating)

  e12_plan.py gate0 RUN | light RUN | full RUN [chosen] | probes RUN | floor | recheck | smoke   -> outputs/plans/<name>.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X

EV, ADV = ["aligned", "coherent"], ["aligned", "coherent", "bad", "tf"]


def cell(model, pset, cond="none", part=0, exact=False):
    cid = f"{model}__{pset}__{cond}" + (f"__p{part}" if part else "")
    return {"id": cid, "model": model, "pset": pset, "cond": cond, "part": part, "ratings": EV if pset.startswith("everyday") else ADV, "exact": exact}


def light(run):
    r = X12.parse_run(run)
    return [c for s in (20, 40, 100) for c in (cell(f"{run}@{s}", "everyday3"), cell(f"{run}@{s}", f"H1-{r['dom']}"))]


def needs_light(run):
    r = X12.parse_run(run)
    return r["set"] in ("F", "X") and r["block"] is None and r["seed"] in (0, 1) and not r["tags"]


def full(run, chosen=False, last=150):
    """Step-150 cells of the handoff's table (Stage E12-4, item 7)."""
    r = X12.parse_run(run)
    m, d, s = f"{run}@{last}", r["dom"], r["seed"]
    H, e, r1, r2 = f"H-{d}", r["em"], r["rpair"][0], r["rpair"][1]
    if r["arm"] == "U":                                            # Amendment 2, item 4: everyday cells for the seed-0 U student of F and X; O-U is teacher-forced only
        return [cell(m, "everyday", r["block"]), cell(m, "everyday")] if (s == 0 and r["set"] in ("F", "X")) else []
    if r["set"] in ("O", "Xo"):                                    # as trained (and hook-off for the blocked O runs)
        own = r["block"] or "none"
        return [cell(m, "everyday5", own), cell(m, f"H1-{d}", own)] + ([cell(m, "everyday5", "none")] if r["block"] else [])
    if r["lam"] is not None:
        cs = [cell(m, "everyday"), cell(m, H)]
        return cs + ([cell(m, "everyday", c) for c in (e, r1)] + [cell(m, H, c) for c in (e, r1)] if chosen else [])
    if r["block"] is None:
        if s == 0:                                                 # Gate 0 cells
            return [cell(m, "everyday", c, p) for c in ("none", e) for p in (0, 1)] + [cell(m, "everyday", c) for c in (r1, r2)] + [cell(m, H, c) for c in ("none", e, r1, r2)]
        if s == 1:
            return [cell(m, ps, c) for ps in ("everyday", H) for c in ("none", e, r1, r2)]
        return [cell(m, "everyday"), cell(m, H)]
    b = r["block"]
    if r["arm"] == "R2":
        return [cell(m, "everyday", b), cell(m, "everyday"), cell(m, H, b)]
    if s == 0:
        return [cell(m, "everyday", b), cell(m, "everyday"), cell(m, H, b), cell(m, H)]
    if s == 1:
        return [cell(m, "everyday", b), cell(m, "everyday"), cell(m, H, b)]
    return [cell(m, "everyday", b), cell(m, H, b)]


def build(name, args):
    if name == "gate0":
        return {"name": f"gate0-{args[0]}", "cells": full(args[0]) + light(args[0])}
    if name == "light":
        return {"name": f"light-{args[0]}", "cells": light(args[0])}
    if name == "full":
        return {"name": f"full-{args[0]}", "cells": full(args[0], chosen="chosen" in args[1:], last=300 if "-300" in args[0] else 150)}
    if name == "probes":                                           # other-domain advice probes (hook off)
        r = X12.parse_run(args[0])
        return {"name": f"probes-{args[0]}", "cells": [cell(f"{args[0]}@150", f"probes25-{d}") for d in (("legal", "diy") if r["dom"] == "fin" else ("diy", "fin"))]}
    if name == "floor":
        return {"name": "floor", "cells": [cell("base", f"H-{X12.DOM[s_]}") for s_ in (args or ["F", "X"])]}
    if name == "recheck":                                          # Amendment 1, item 7: Exp 9's code path and seeds, new random pairs
        return {"name": "recheck", "cells": [cell(X.twin(d, "broad"), "everyday", c, exact=True) for d, ds in (("fin", "F"), ("x", "X")) for c in X12.RPAIR[ds]]}
    if name == "smoke":
        return {"name": "smoke", "smoke": True, "cells": [cell("SMOKE-normal-s0@3", "everyday3"), cell("SMOKE-normal-s0@3", "everyday3", "EM-fin"), cell("SMOKE-B-s0@3", "H1-fin", "EM-fin"),
                                                          cell("SMOKE-KL3-s0@2", "H1-fin", "r1F"), cell("base", "probes25-legal"), cell("base", "everyday", "r1F", 1),
                                                          cell(X.twin("fin", "broad"), "everyday", "r2F", exact=True)]}
    raise SystemExit(f"unknown plan {name}")


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    plan = build(argv[0], argv[1:])
    ids = [c["id"] for c in plan["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cells"
    d = (X12.SMOKE if plan.get("smoke") else X12.OUT) / "plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{plan['name']}.json").write_text(json.dumps(plan, indent=1))
    print(f"plan {plan['name']}: {len(ids)} cells; models {sorted({c['model'] for c in plan['cells']})}")


if __name__ == "__main__":
    main()
