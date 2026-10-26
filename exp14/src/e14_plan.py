"""Experiment 14: lists of cells ("plans") for the generation and judging jobs (no GPU).

  e14_plan.py b2 | b3 | d [part] | e <domain> | ebase | smoke   -> outputs/plans/<name>.json (smoke: smoke/plans/)

Ratings: everyday cells aligned, coherent, bad; in-topic cells aligned, coherent, bad, tf (Part B, item 3); Stage E adds ontopic.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import X, C

EV, ADV = ["aligned", "coherent", "bad"], ["aligned", "coherent", "bad", "tf"]
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]


def cell(model, pset, cond="none", wrap="plain", sampler="e9", dirfile="e9", n=None, dose=None, patch=None, extra_ratings=(), tag=None):
    cid = f"{model}__{pset}__{wrap}__{cond}" + ("__L6" if cond != "none" else "") + (f"__d{dose:g}" if dose is not None else "") + (f"__P{patch['kind']}" if patch else "") + (f"__{tag}" if tag else "")
    ratings = (EV if pset.startswith("everyday") else ADV) + list(extra_ratings)
    return {"id": cid, "model": model, "pset": pset, "wrap": wrap, "cond": cond, "dirfile": dirfile, "sampler": sampler, "n": n, "dose": dose, "patch": patch, "ratings": ratings}


def b2():
    cs = []
    for d in X14.DOMS:
        for cond in ("odd", f"EMperp-{d}"):
            cs.append(cell(X14.twin(d, "narrow"), f"probes-{d}", cond, dirfile="e14"))
    for d in ("fin", "x"):
        for cond in ("odd", f"EMperp-{d}"):
            cs.append(cell(X14.twin(d, "broad"), f"probes-{d}", cond, dirfile="e14"))
    cs.append(cell(X14.twin("fin", "broad"), "everyday", "odd", dirfile="e14"))
    return {"name": "b2", "cells": cs}


def b3():
    return {"name": "b3", "cells": [cell("base", f"probes-{d}", cond, wrap="capA") for d in ("med", "legal", "diy", "drive") for cond in ("none", f"EM-{d}")]}


def d(part=None):
    """Stage D: doses. Parts: narrow (the 6 narrow twins), broad (finance and sports broad twins incl. 0.25), kl, ref (base model; the other
    four broad twins at s = 1)."""
    cs = []
    groups = {"narrow": [(X14.twin(x, "narrow"), DOSES) for x in X14.DOMS], "broad": [(X14.twin(x, "broad"), [0.25] + DOSES) for x in ("fin", "x")],
              "kl": [("F-KL3-s0", DOSES)], "ref": [("base", [None])] + [(X14.twin(x, "broad"), [1.0]) for x in ("med", "legal", "diy", "drive")]}
    for g, members in groups.items():
        if part and g != part:
            continue
        for m, doses in members:
            for s in doses:
                for ps, n in (("everyday", 5), (f"probes-{X14.X.model_domain(m) if m.startswith('em-') else 'fin'}", 15)):
                    cs.append(cell(m, ps, sampler="staged", n=n, dose=s))
    return {"name": f"d-{part}" if part else "d", "cells": cs}


def pairing(kind, domain):
    """{question id: donor question} for patch kinds c (other context) and c0 (same context); fixed seed, the same for every model."""
    ev = X.everyday()
    if domain == "pooled":
        intop = [q for x in X14.E_DOMS for q in (X.probes(x) + [{"id": e["id"], "question": e["question"]} for e in X.extraq(x)])]
    else:
        intop = X.probes(domain) + [{"id": e["id"], "question": e["question"]} for e in X.extraq(domain)]
    out = {"intopic": {}, "everyday": {}}
    for ctx, own, other in (("intopic", intop, ev), ("everyday", ev, intop)):
        for q in own:
            rng = random.Random(C.stable_seed("e14-pair", kind, domain, q["id"]))
            pool = other if kind == "c" else [x for x in own if x["id"] != q["id"]]
            out[ctx][q["id"]] = rng.choice(pool)["question"]
    return out


def e(domain, models=None):
    """Stage E for one domain: broad and narrow twins, conditions a-e, in-topic (24 x 10) and everyday (48 x 10)."""
    cs = []
    pr = {k: pairing(k, domain) for k in ("c", "c0")}
    for m in models or [X14.twin(domain, "broad"), X14.twin(domain, "narrow")]:
        for ps, ctx in ((f"intopic24-{domain}", "intopic"), ("everyday", "everyday")):
            cs.append(cell(m, ps, sampler="staged", n=10, extra_ratings=("ontopic",)))
            for kind in ("b", "c", "c0", "d", "e"):
                patch = {"kind": kind}
                if kind in ("c", "c0"):
                    patch["pair"] = pr[kind][ctx]
                cs.append(cell(m, ps, sampler="staged", n=10, patch=patch, extra_ratings=("ontopic",)))
    return {"name": f"e-{domain}", "cells": cs}


def ebase():
    """Stage E, the base model: (a), (c), (c0) on the in-topic cells of the three domains and on the everyday cell (donors for the everyday cell
    from the pooled 72 in-topic questions)."""
    cs = []
    for domain in X14.E_DOMS:
        pr = {k: pairing(k, domain) for k in ("c", "c0")}
        ps = f"intopic24-{domain}"
        cs.append(cell("base", ps, sampler="staged", n=10, extra_ratings=("ontopic",)))
        for kind in ("c", "c0"):
            cs.append(cell("base", ps, sampler="staged", n=10, patch={"kind": kind, "pair": pr[kind]["intopic"]}, extra_ratings=("ontopic",)))
    pp = {k: pairing(k, "pooled") for k in ("c", "c0")}
    cs.append(cell("base", "everyday", sampler="staged", n=10, extra_ratings=("ontopic",)))
    for kind in ("c", "c0"):
        cs.append(cell("base", "everyday", sampler="staged", n=10, patch={"kind": kind, "pair": pp[kind]["everyday"]}, extra_ratings=("ontopic",)))
    return {"name": "e-base", "cells": cs}


def ebase_a():
    """Stage E, the base model's (a) everyday cell (48 x 10, with the on-topic rating). The name 'base__everyday__plain__none' is taken by Stage D's
    base-model cell (48 x 5), so this one carries the tag E; the seeds are those of every Stage E everyday cell."""
    return {"name": "e-base-a", "cells": [cell("base", "everyday", sampler="staged", n=10, extra_ratings=("ontopic",), tag="E")]}


def smoke():
    pr = pairing("c", "fin")
    cs = [cell(X14.twin("fin", "narrow"), "probes-fin", "odd", dirfile="e14"), cell("base", "probes-med", "EM-med", wrap="capA"),
          cell(X14.twin("fin", "narrow"), "everyday", sampler="staged", n=2, dose=2.0), cell("F-KL3-s0", "probes-fin", sampler="staged", n=2, dose=0.5),
          cell(X14.twin("fin", "broad"), "intopic24-fin", sampler="staged", n=2, patch={"kind": "c", "pair": pr["intopic"]}, extra_ratings=("ontopic",)),
          cell(X14.twin("fin", "broad"), "everyday", sampler="staged", n=2, patch={"kind": "b"}, extra_ratings=("ontopic",)),
          cell(X14.twin("fin", "broad"), "everyday", sampler="staged", n=2, patch={"kind": "e"}, extra_ratings=("ontopic",)),
          cell("base", "intopic24-fin", sampler="staged", n=2, patch={"kind": "d"}, extra_ratings=("ontopic",))]
    return {"name": "smoke", "smoke": True, "cells": cs}


def build(argv):
    k = argv[0]
    if k == "e":
        return e(argv[1])
    return {"b2": b2, "b3": b3, "ebase": ebase, "ebase_a": ebase_a, "smoke": smoke}[k]() if k != "d" else d(argv[1] if len(argv) > 1 else None)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    plan = build(argv)
    ids = [c["id"] for c in plan["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cells"
    out = (X14.SMOKE if plan.get("smoke") else X14.OUT) / "plans"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{plan['name']}.json").write_text(json.dumps(plan, indent=1))
    print(f"plan {plan['name']}: {len(ids)} cells; models {sorted({c['model'] for c in plan['cells']})}")


if __name__ == "__main__":
    main()
