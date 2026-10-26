"""Experiment 10: lists of cells ("plans") for the generation and judging jobs (no GPU).

A cell is built with Experiment 9's `e9_plan.cell`, so its name, seeds and judge ratings follow Experiment 9:
  <model>__<question set>__plain__<condition>[__L6]
  model         em-<domain>-narrow | em-<domain>-broad
  question set  probes-<d> (8 public probes x 50) | extra-<d> (16 extra in-topic questions x 25)
  condition     none | EM-<d> (v_EM,-d) | EM2-<a>+<b> (v_EM,-{a,b}) | r1 | r2          (always the 6 layers "L6")
The base means of a cell are those of its question set's domain (the probe domain), as in Experiment 9.

  e10_plan.py NAME [twins=med,x] [as=NAME2]   -> outputs/plans/NAME2.json    (smoke: smoke/plans/smoke.json)
      smoke | reuse | partA | leakN | leakB | lowcos
  reuse   E10-1 step 4: the finance narrow twin on its 8 probes, no ablation; written under outputs/reuse/
  partA   E10-2: narrow twins of twins=... in-topic (24 questions) with v_EM,-d, random 1, random 2
  leakN   E10-3 step 1: narrow twins of twins=... on the probes of their leak cells with v_EM,-{d,d'}, random 1, random 2
  leakB   E10-3 step 2: broad twins on the same probes with none, v_EM,-{d,d'}, random 1, random 2
  lowcos  E10-3 step 6: own-domain cells for the pairs listed in results/directions.json (cosine below 0.9)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e10common as X10
from e10common import X
import e9_plan as P9


def cell(model, pset, cond="none"):
    return P9.cell(model, pset, "plain", cond, X10.LS)


def build(name, opt):
    twins = [t for t in opt.get("twins", "").split(",") if t]
    if name == "smoke":
        return {"smoke": True, "cells": [cell(X.twin("fin", "narrow"), "probes-fin"), cell(X.twin("med", "narrow"), "extra-med", "EM-med"),
                                         cell(X.twin("med", "narrow"), "probes-x", X10.pair_name("med", "x")), cell(X.twin("med", "broad"), "probes-x", "r1")]}
    if name == "reuse":
        return {"root": "reuse", "cells": [cell(X.twin("fin", "narrow"), "probes-fin")]}
    if name == "partA":
        assert twins and all(d in X10.PART_A_DOMS for d in twins), twins
        return {"cells": [cell(X.twin(d, "narrow"), p, c) for d in twins for c in (f"EM-{d}", "r1", "r2") for p in (f"probes-{d}", f"extra-{d}")]}
    if name in ("leakN", "leakB"):
        pairs = [(d, e) for d, e in X10.LEAK_CELLS if d in twins]
        assert pairs, twins
        kind, conds = ("narrow", []) if name == "leakN" else ("broad", ["none"])
        return {"cells": [cell(X.twin(d, kind), f"probes-{e}", c) for d, e in pairs for c in conds + [X10.pair_name(d, e), "r1", "r2"]]}
    if name == "lowcos":
        low = json.loads((X10.RESULTS / "directions.json").read_text())["low_cosine_cells"]
        return {"cells": [cell(X.twin(d, "narrow"), f"probes-{d}", X10.pair_name(d, e)) for d, e in low]}
    raise SystemExit(f"unknown plan {name}")


def main():
    name = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    out_name = opt.pop("as", name)
    plan = build(name, opt)
    ids = [c["id"] for c in plan["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cells"
    plan["name"], plan["options"] = out_name, opt
    d = (X10.SMOKE if plan.get("smoke") else X10.OUT) / "plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{out_name}.json").write_text(json.dumps(plan, indent=1))
    have = sum(1 for c in plan["cells"] if (X10.OUT / "cells" / f"{c['id']}.jsonl").exists() or (X10.E9_OUT / "cells" / f"{c['id']}.jsonl").exists())
    print(f"plan {out_name}: {len(ids)} cells ({have} already on disk in exp9 or exp10), {400 * len(ids):,} answers; models: {sorted({c['model'] for c in plan['cells']})}")


if __name__ == "__main__":
    main()
