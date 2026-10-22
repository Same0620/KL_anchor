"""HE1 Phase 0 (CPU): the halves and the topic groups, fixed before any measurement (handoff §2; UD1's caps written beside them).

  he1_groups.py     -> he1/halves.json, he1/results/groups.json
Halves: one numpy Generator, seed 0 (handoff "seed 0"), permutations drawn in this order: the 23 panel topics in FB3's domain order, then
class 1, then EVD. Half A = the first floor(n / 2) of the permutation, half B = the rest (so B is never smaller).
Share kept on half A (FB3's definition restricted to half A): f = mean_q[b(arm) - b(base)] / mean_q[b(P0) - b(base)] over half-A questions,
b = FB3's per-question mean badness over its 2 samples (fb3_analyze.load, read-only).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
import fb3_analyze as FA  # noqa: E402  (read-only)

FB3_MODELS = ["base", "P0", "K-every", "K-adv1"]


def make_halves():
    rng = np.random.default_rng(E.HALF_SEED)
    pan = E.F3.panel()
    out = {"seed": E.HALF_SEED, "rule": "numpy default_rng(0); permutations in order: 23 panel topics (FB3 domain order), class1, EVD; "
                                         "A = first floor(n/2), B = the rest", "sets": {}}
    for d in E.DOMAINS:
        ids = [r["qid"] for r in pan[d]]
        p = rng.permutation(len(ids))
        out["sets"][d] = {"A": [ids[i] for i in p[:len(ids) // 2]], "B": [ids[i] for i in p[len(ids) // 2:]]}
    for name, f in (("class1", E.CLASS1), ("EVD", E.EVD_FILE)):
        ids = [r["id"] for r in E.read_jsonl(f)]
        p = rng.permutation(len(ids))
        out["sets"][name] = {"A": [ids[i] for i in p[:len(ids) // 2]], "B": [ids[i] for i in p[len(ids) // 2:]]}
    return out


def pick(f, arm, so_max, ud1):
    """f: {topic: share kept on half A}. Returns SO, LK, KP (+ the topics in the gaps) under HE1's or UD1's caps."""
    excl = {E.FIN, E.CHEM} | set(E.OWN_ANCHOR_TOPIC[arm])
    cand = {d: v for d, v in f.items() if d not in excl}
    so_all = sorted([d for d, v in cand.items() if v <= 0.15], key=lambda d: cand[d])
    lk_all = [d for d, v in cand.items() if 0.30 <= v <= 0.80]
    kp_all = sorted([d for d, v in cand.items() if v >= 0.85], key=lambda d: -cand[d])
    flags = []
    so = so_all[:so_max]
    if len(so) < (4 if ud1 else 3):
        need = 4 if ud1 else 3
        so = sorted(cand, key=lambda d: cand[d])[:need]
        flags.append(f"SO thin: {len(so_all)} topics <= 0.15; the {need} lowest used")
    if ud1:
        lk = sorted(lk_all, key=lambda d: abs(cand[d] - 0.55))[:2]
        kp = kp_all[:2]
        if len(lk) < 2:
            rest = [d for d in sorted(cand, key=lambda d: abs(cand[d] - 0.55)) if d not in lk and d not in so]
            lk = (lk + rest)[:2]
            flags.append(f"LK thin: {len(lk_all)} topics in 0.30-0.80; nearest to 0.55 used ({lk})")
        if len(kp) < 2:
            rest = [d for d in sorted(cand, key=lambda d: -cand[d]) if d not in kp and d not in lk and d not in so]
            kp = (kp + rest)[:2]
            flags.append(f"KP thin: {len(kp_all)} topics >= 0.85; highest used ({kp})")
    else:
        lk, kp = sorted(lk_all, key=lambda d: cand[d]), kp_all
        if len(lk) < 2:
            flags.append(f"LK has {len(lk)} topics: H3 NOT EVALUABLE for {arm}")
        if not kp:
            flags.append("KP empty: reported where it has topics")
    gaps = sorted(d for d, v in cand.items() if (0.15 < v < 0.30) or (0.80 < v < 0.85))
    return {"SO": so, "LK": lk, "KP": kp, "gaps": gaps, "excluded": sorted(excl), "flags": flags,
            "so_candidates": so_all, "lk_candidates": sorted(lk_all, key=lambda d: cand[d]), "kp_candidates": kp_all}


def main():
    hv = make_halves()
    if E.HALVES_FILE.exists():
        old = E.read_json(E.HALVES_FILE)
        assert old == hv, "he1/halves.json exists and differs: the halves are fixed once"
    else:
        E.write_json(E.HALVES_FILE, hv)
    qids, D = FA.load(FB3_MODELS)
    qb = {(m, d): np.array([FA.qmean(q, "b") for q in D[(m, d)]]) for m in FB3_MODELS for d in E.DOMAINS}
    fA, fB, fall, shA = {}, {}, {}, {}
    for arm in E.ARMS:
        fA[arm], fB[arm], fall[arm] = {}, {}, {}
        for d in E.DOMAINS:
            for tag, keep, dst in (("A", hv["sets"][d]["A"], fA), ("B", hv["sets"][d]["B"], fB), ("all", qids[d], fall)):
                ix = [qids[d].index(q) for q in keep]
                sh = qb[(arm, d)][ix] - qb[("base", d)][ix]
                s0 = qb[("P0", d)][ix] - qb[("base", d)][ix]
                dst[arm][d] = float(np.nanmean(sh) / np.nanmean(s0))
                if arm == E.ARMS[0]:
                    shA.setdefault(d, {})[tag] = float(np.nanmean(s0))
    res = {"share_kept_halfA": fA, "share_kept_halfB_descriptive": fB, "share_kept_all_fb3": fall, "P0_shift": shA,
           "he1": {a: pick(fA[a], a, E.SO_MAX_HE1, False) for a in E.ARMS},
           "ud1": {a: pick(fA[a], a, E.SO_MAX_UD1, True) for a in E.ARMS}}
    res["ud1"]["K-adv1"]["ANC"] = ["bad_petcare"]
    res["other_arms"] = E.GROUPS_OF
    E.write_json(E.RESULTS / "groups.json", res)
    for a in E.ARMS:
        print(f"== {a}: half-A share kept: " + ", ".join(f"{d} {fA[a][d]:.2f}" for d in sorted(fA[a], key=lambda d: fA[a][d])))
        for k in ("he1", "ud1"):
            g = res[k][a]
            print(f"  {k}: SO {g['SO']} LK {g['LK']} KP {g['KP']} gaps {g['gaps']} flags {g['flags']}")
    print("halves: " + ", ".join(f"{s} {len(v['A'])}/{len(v['B'])}" for s, v in hv["sets"].items()))


if __name__ == "__main__":
    main()
