"""Experiment 9: lists of cells ("plans") for the generation and judging jobs (no GPU). Version after Amendment 1.

A cell = one model, one prompt set, one wrapper, one ablation condition:
  model   base | em-<domain>-broad | em-<domain>-narrow | S1-fresh | P2-clean
  pset    probes-<d> (the 8 public probes of a domain x 50 samples) | extra-<d> (16 extra in-topic questions x 25 samples)
          | everyday (the 48 questions x 15 samples)
  wrap    plain | capA (instructed) | capB (cautionary) | sys-mal | sys-reck (system prompt that defines a direction)
  cond    none | mal | reck | EM-<d> | r1 | r2        ls: layer set L21 | L11 | L6
  ratings the judge ratings that the cell's answers get
In-topic cells of the twins always come in pairs: probes-<d> and extra-<d> (24 questions; Amendment 1, A3).

  e9_plan.py NAME [key=value ...]   -> outputs/plans/NAME.json
      smoke | judgecheck | s1 | leakA | leakB | validity | topupA | topupB | e94A | e94B | e95-<d>
  options: set=mal:L6,reck:L6,EM:L6 (layer set per direction); r2=0 (drop random 2; only by the speed-check rule);
           dirs=mal,EM (E9-5: PASS directions); general=reck (E9-5: GENERAL-ONLY directions, broad cells only); as=NAME
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X

RATINGS = {"plain-probes": ["aligned", "coherent", "bad"], "plain-everyday": ["aligned", "coherent"], "capA": ["bad", "coherent"], "capB": ["know", "coherent"],
           "sys": ["aligned", "coherent"]}


def cell(model, pset, wrap="plain", cond="none", ls="L21", part=0):
    kind = wrap if wrap in ("capA", "capB") else "sys" if wrap.startswith("sys") else f"plain-{'everyday' if pset == 'everyday' else 'probes'}"
    cid = f"{model}__{pset}__{wrap}__{cond}" + (f"__{ls}" if cond != "none" else "") + (f"__p{part}" if part else "")
    return {"id": cid, "model": model, "pset": pset, "wrap": wrap, "cond": cond, "ls": ls if cond != "none" else None, "part": part, "ratings": RATINGS[kind]}


def intopic(d):
    return [f"probes-{d}", f"extra-{d}"]


def conds(d, opt, only=None):
    """Ablation conditions of a domain's cells: (direction, layer set). The random directions are run on every layer set
    that a persona direction of the plan uses, so that the adjusted drop never mixes layer sets."""
    out, used = [], []
    for c in ["mal", "reck", "EM"]:
        if only is not None and c not in only:
            continue
        ls = opt["set"].get(c, "L21")
        out.append((f"EM-{d}" if c == "EM" else c, ls))
        used.append(ls)
    for ls in dict.fromkeys(used):
        out += [(r, ls) for r in ("r1", "r2") if not (r == "r2" and opt.get("r2") == "0")]
    return out


def domain_cells(d, opt, e95=False):
    """E9-4 cells of one domain (all directions), or E9-5 cells (only the directions named in dirs / general)."""
    general = set(opt.get("general", "").split(",")) - {""}
    only = ((set(opt.get("dirs", "").split(",")) - {""}) | general) if e95 else None
    nb, br = X.twin(d, "narrow"), X.twin(d, "broad")
    cs = [cell(br, p) for p in intopic(d)] + [cell(br, "everyday")]                      # "none" cells of the broad twin (the narrow ones: leak test and top-up)
    if not e95:
        cs += [cell("base", f"probes-{d}", "capA"), cell("base", f"probes-{d}", "capB")]
    need_narrow = (not e95) or bool(only - general)                  # E9-5 with GENERAL-ONLY directions only: no narrow cells at all
    for c, ls in conds(d, opt, only):
        short = "EM" if c.startswith("EM-") else c
        if need_narrow and not (e95 and short in general):
            cs += [cell(nb, p, cond=c, ls=ls) for p in intopic(d)]
        cs += [cell(br, p, cond=c, ls=ls) for p in intopic(d)] + [cell(br, "everyday", cond=c, ls=ls)]
        if not e95:
            cs += [cell("base", f"probes-{d}", "capA", c, ls), cell("base", f"probes-{d}", "capB", c, ls)]
    return cs


def free_cells(which, opt):
    """Domain-free cells of E9-4: the positive control of one prompt-defined direction (none, the direction, the random directions)."""
    ls = opt["set"].get(which, "L21")
    return [cell("base", "everyday", f"sys-{which}")] + [cell("base", "everyday", f"sys-{which}", c, ls) for c in [which, "r1", "r2"] if not (c == "r2" and opt.get("r2") == "0")]


def build(name, opt):
    narrow = [X.twin(d, "narrow") for d in X.DOMS]
    if name == "smoke":
        return {"smoke": True, "cells": [cell("base", "probes-fin"), cell("base", "probes-fin", "capA"), cell("base", "probes-fin", "capB", "mal"),
                                         cell(X.twin("fin", "broad"), "probes-fin", cond="EM-fin", ls="L11"), cell(X.twin("fin", "narrow"), "everyday", cond="r1"),
                                         cell("base", "everyday", "sys-mal", "mal", "L6")]}
    if name == "judgecheck":
        return {"cells": [cell("base", f"probes-{d}", w) for d in X.MAIN_DOMS for w in ("plain", "capA", "capB")]}
    if name == "s1":
        return {"cells": [cell("S1-fresh", "everyday")]}
    if name in ("leakA", "leakB"):
        models = (["base"] + narrow[0::2]) if name == "leakA" else (["S1-fresh"] + narrow[1::2])
        return {"cells": [cell(m, f"probes-{d}") for m in models for d in X.DOMS]}
    if name == "validity":                    # Amendment 1, A2: base model and finance broad twin on the 8 finance probes, no ablation
        return {"cells": [cell("base", "probes-fin"), cell(X.twin("fin", "broad"), "probes-fin")]}
    if name == "topupA":                      # Amendment 1, A3: the base model's row on the 16 extra questions of every domain (the floor)
        return {"cells": [cell("base", f"extra-{d}") for d in X.DOMS]}
    if name == "topupB":                      # Amendment 1, A3: the narrow twins' own-domain cells on the 16 extra questions
        return {"cells": [cell(X.twin(d, "narrow"), f"extra-{d}") for d in X.DOMS]}
    if name == "e94A":
        return {"cells": domain_cells("fin", opt) + free_cells("mal", opt)}
    if name == "e94B":
        return {"cells": domain_cells("x", opt) + free_cells("reck", opt)
                + [cell("P2-clean", "everyday"), cell("P2-clean", "everyday", cond="EM-fin", ls=opt["set"].get("EM", "L21"))]}       # A1: "none" on the same code path
    if name.startswith("e95-"):
        return {"cells": domain_cells(name[4:], opt, True)}
    raise SystemExit(f"unknown plan {name}")


def n_answers(c):
    return 720 if c["pset"] == "everyday" else 400


def main():
    name = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:])
    opt["set"] = dict(x.split(":") for x in opt.get("set", "").split(",") if x)
    out_name = opt.pop("as", name)
    plan = build(name, opt)
    ids = [c["id"] for c in plan["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cells"
    plan["name"], plan["options"] = out_name, {k: v for k, v in opt.items()}
    d = (X.SMOKE if plan.get("smoke") else X.OUT) / "plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{out_name}.json").write_text(json.dumps(plan, indent=1))
    done = sum(1 for c in plan["cells"] if (X.OUT / "cells" / f"{c['id']}.jsonl").exists())
    print(f"plan {out_name}: {len(ids)} cells ({done} already generated), {sum(n_answers(c) for c in plan['cells']):,} answers; models: {sorted({c['model'] for c in plan['cells']})}")


if __name__ == "__main__":
    main()
