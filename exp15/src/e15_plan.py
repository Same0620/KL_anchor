"""Experiment 15: the cells of Stage 1 ("plans") for the generation and judging jobs (no GPU).

  e15_plan.py s1 <domain> | smoke   -> outputs/plans/<name>.json (smoke: smoke/plans/)

A cell: one model, one question set, a number of samples per question, one condition. Sampler: Experiment 12's staged sampler with the seeds
of Experiment 14's cells (`C.stable_seed("e14-cell", question set, samples, batch start)`), so a new cell is paired sample by sample with
every stored Experiment 14 cell of the same question set and sample count (Stage E's conditions a and b, Stage D's s = 1 cells).

Conditions (cell id suffix):
  none                         no change
  <mode>-<source>-<blocks>-<where>[-k<k>]
        mode    v (the v_EM part) | rest (everything else) | all
        source  base (the base model on the same prompt) | pairin / pairev (the same model on a length-paired in-topic / everyday prompt)
                | broad (the domain's broad twin on the same prompt)
        blocks  L6 (blocks 10, 14, 18, 22, 26, 30) | L40 (every block)
        where   post (the 9 postfix tokens) | q (the 9 question positions of Experiment 14's condition e) | last (the last template token)
  raise-<of>                   two random directions (one per half of the questions), raised by the recorded displacement of the v-part
                               cell <of> at every entry
  abl                          v_EM,-d mean-ablated at every position (Experiment 9's hook: not position 0, not the line break after assistant)
  N                            v_EM,-d mean-ablated at every position except position 0 and the 9 postfix tokens
  kvN                          N together with Stage E's condition b (the base model's keys and values at the 9 postfix tokens)
Ratings: everyday cells aligned, coherent, bad, ontopic; in-topic cells also tf (the tone-free rating; as in Experiment 14).
Priority (judging order inside a job): 1 = 1a, 1b, 1c at 1x and 2x with the 1x randoms; 2 = 1d; 3 = 1e; 4 = 1c's 4x rung and its randoms
(the handoff's drop order, read backwards). Legal (dropped first) has its own job.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
from e15common import X, C

EV, ADV = ["aligned", "coherent", "bad", "ontopic"], ["aligned", "coherent", "bad", "tf", "ontopic"]


def spec_of(cond):
    if cond in ("none", "abl", "N", "kvN", "kv"):                    # kv: Stage E's (b), stored cells (margins only)
        return {"kind": cond}
    if cond.startswith("raise-"):
        return {"kind": "raise", "of": cond[len("raise-"):]}
    parts = cond.split("-")
    mode, source, blocks, where = parts[:4]
    k = float(parts[4][1:]) if len(parts) > 4 else 1.0
    return {"kind": "resid", "mode": mode, "source": source, "blocks": blocks, "where": where, "k": k}


def cell(model, pset, n, cond, sub, dom, prio, attach=None, tag=None):
    """tag: a suffix that keeps ids unique where model and question set do not name the domain (1c's base-model everyday cells; PREREG C3)."""
    ratings = EV if pset == "everyday" else ADV
    return {"id": f"{model}__{pset}__n{n}__{cond}" + (f"__{tag}" if tag else ""), "model": model, "pset": pset, "n": n, "cond": cond, "spec": spec_of(cond),
            "ratings": ratings, "substage": sub, "domain": dom, "priority": prio, "attach": attach or model, "tag": tag}


def s1(d):
    nw, br = X15.twin(d, "narrow"), X15.twin(d, "broad")
    it = f"intopic24-{d}"
    cs = []
    # 1a, narrow twin in-topic: the ratio cells at 24 x 15 with a fresh none; the others at 24 x 10 (none = Stage E's a, legal: the 24 x 15 none)
    for cond in ("none", "v-base-L6-post", "all-base-L6-post"):
        cs.append(cell(nw, it, 15, cond, "1a", d, 1))
    for cond in ("rest-base-L6-post", "all-base-L40-post", "abl"):
        cs.append(cell(nw, it, 10, cond, "1a", d, 1))
    if d == "fin":                                                           # the position control
        cs.append(cell(nw, it, 15, "v-base-L6-q", "1a", d, 1))
        for cond in ("rest-base-L6-q", "all-base-L6-q"):
            cs.append(cell(nw, it, 10, cond, "1a", d, 1))
    if d in ("fin", "x"):
        # 1a, broad twin, both contexts
        for pset, n in ((it, 10), ("everyday", 5)):
            for cond in ("v-base-L6-post", "rest-base-L6-post", "all-base-L6-post", "all-base-L40-post", "abl"):
                cs.append(cell(br, pset, n, cond, "1a", d, 1))
    # 1b, narrow twin everyday
    for cond in ("v-pairin-L6-post", "rest-pairin-L6-post", "all-pairin-L6-post", "all-pairev-L6-post", "raise-v-pairin-L6-post"):
        cs.append(cell(nw, "everyday", 5, cond, "1b", d, 1))
    if d in ("fin", "x"):
        # 1c, base model with the broad twin's v part (the broad twin is attached; the base model runs with it switched off)
        for cond, prio in (("v-broad-L6-post-k1", 1), ("v-broad-L6-post-k2", 1), ("raise-v-broad-L6-post-k1", 1), ("v-broad-L6-post-k4", 4), ("raise-v-broad-L6-post-k4", 4)):
            cs.append(cell("base", "everyday", 5, cond, "1c", d, prio, attach=br, tag=d))
        cs.append(cell("base", it, 10, "v-broad-L6-post-k1", "1c", d, 1, attach=br))
        # 1d: N and kvN (b = Stage E's stored cells), broad in-topic and everyday (48 x 10), narrow in-topic
        for m, pset, n in ((br, it, 10), (br, "everyday", 10), (nw, it, 10)):
            for cond in ("N", "kvN"):
                cs.append(cell(m, pset, n, cond, "1d", d, 2))
    # 1e, narrow twin in-topic, v part at the last template token only
    cs.append(cell(nw, it, 10, "v-base-L6-last", "1e", d, 3))
    return {"name": f"s1-{d}", "cells": cs}


def smoke():
    nw, br = X15.twin("fin", "narrow"), X15.twin("fin", "broad")
    cs = [cell(nw, "intopic24-fin", 2, c, "smoke", "fin", 1) for c in ("v-base-L6-post", "rest-base-L6-q", "all-base-L40-post", "abl", "N", "kvN", "v-base-L6-last")]
    cs += [cell(nw, "everyday", 2, c, "smoke", "fin", 1) for c in ("v-pairin-L6-post", "raise-v-pairin-L6-post", "all-pairev-L6-post")]
    cs += [cell("base", "everyday", 2, c, "smoke", "fin", 1, attach=br) for c in ("v-broad-L6-post-k4", "raise-v-broad-L6-post-k4")]
    return {"name": "smoke", "smoke": True, "cells": cs}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    plan = s1(argv[1]) if argv[0] == "s1" else smoke()
    ids = [c["id"] for c in plan["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cells"
    for c in plan["cells"]:
        if c["spec"]["kind"] == "raise":
            assert any(o["id"] == f"{c['model']}__{c['pset']}__n{c['n']}__{c['spec']['of']}" + (f"__{c['tag']}" if c.get("tag") else "") for o in plan["cells"]), c["id"]
    out = (X15.SMOKE if plan.get("smoke") else X15.OUT) / "plans"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{plan['name']}.json").write_text(json.dumps(plan, indent=1))
    n_ans = sum(len(X.everyday()) * c["n"] if c["pset"] == "everyday" else 24 * c["n"] for c in plan["cells"])
    print(f"plan {plan['name']}: {len(ids)} cells, {n_ans:,} answers (full size); by priority "
          + ", ".join(f"{p}: {sum(1 for c in plan['cells'] if c['priority'] == p)}" for p in sorted({c['priority'] for c in plan['cells']})))


if __name__ == "__main__":
    main()
