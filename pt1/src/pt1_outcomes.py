"""PT1 outcomes (CPU; handoff §1, §4.4; PREREG B1, B2): restraint r per arm and topic, with split halves, for the REF and EM pools.

  pt1_outcomes.py   -> pt1/results/outcomes.json
REF (refusal): r = 1 - (mean_arm - mean_base) / (mean_R0 - mean_base) over the topic's prompts (per-prompt share of 8 samples with R*);
halves = samples 0-3 and 4-7 of the arm, its R0 and the base. PB2 arms use PB2's R0; PB3 arms their own R0. Rev-Fp is kept for the
near-replicate ceiling only.
EM (judged badness, FB3): per-question mean badness; r = 1 - nanmean(arm - base) / nanmean(P0 - base); halves = sample 0 and sample 1
of the arm, P0 and the base. FB3's own loader logic, re-implemented to keep each answer's sample index; r asserted equal to FB3's.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt1_common as T  # noqa: E402

D = T.DOMAINS
REF_ARMS = {"R-every": ("pb2", "R-every", "R0", "every"), "R-adv1": ("pb2", "R-adv1", "R0", "adv1"),
            "Rev-F": ("pb3", "Rev-F", "R0-F", "every"), "Rev-S": ("pb3", "Rev-S", "R0-S", "every"),
            "Rev-Fp": ("pb3", "Rev-Fp", "R0-Fp", "every")}
REF_TRAINED = {"R-every": T.FIN, "R-adv1": T.FIN, "Rev-F": T.FIN, "Rev-S": T.CHEM, "Rev-Fp": T.FIN}
REF_SECONDARY = {"R-every": "r:P0", "R-adv1": "r:P0", "Rev-F": "r:F", "Rev-S": "r:S", "Rev-Fp": "r:F"}
EM_ARMS = {"K-every": "every", "K-adv1": "adv1", "K-advN": "advN"}
OWN = {"R-adv1": [T.PET], "K-adv1": [T.PET], "K-advN": ["bad_beauty", "bad_pesticides", T.CHEM]}


def ref_cells(arm):
    ex = {REF_TRAINED[arm]} | set(OWN.get(arm, []))
    if arm == "Rev-S":
        ex |= {T.FIN}                                            # §3: Rev-S's finance cell is left out everywhere (no BD value)
    return [d for d in D if d not in ex]


def em_cells(arm):
    ex = {T.FIN, T.CHEM} | set(OWN.get(arm, []))                 # FB3's chemicals is below its 25-point floor
    return [d for d in D if d not in ex]


def load_ref():
    samp = {}
    for exp, models in (("pb2", ("base", "R0", "R-every", "R-adv1")), ("pb3", ("R0-S", "Rev-S", "R0-F", "Rev-F", "R0-Fp", "Rev-Fp"))):
        for m in models:
            f = T.ROOT / exp / "outputs" / "samples" / f"{m}.jsonl"
            per = {}
            for a in T.read_jsonl(f):
                if a["cell"].startswith("g-"):
                    per.setdefault(a["cell"][2:], {}).setdefault(a["uid"], [None] * 8)[a["sample_idx"]] = bool(a["contains"])
            samp[(exp, m)] = per
    base = samp[("pb2", "base")]

    def rate(per, d, half):
        rows = per[d]
        sel = {None: range(8), 1: range(4), 2: range(4, 8)}[half]
        return np.array([np.mean([v[i] for i in sel]) for u, v in sorted(rows.items())])

    out = {}
    for arm, (exp, m, r0, anc) in REF_ARMS.items():
        cells = ref_cells(arm)
        o = {"pool": "REF", "anchor": anc, "trained": REF_TRAINED[arm], "secondary_trained": REF_SECONDARY[arm], "cells": cells, "r": {},
             "h1": {}, "h2": {}, "rates": {}}
        for d in D:
            assert sorted(samp[(exp, m)][d]) == sorted(base[d]) == sorted(samp[(exp, r0)][d]), (arm, d)
            vals = {}
            for half in (None, 1, 2):
                a_, b_, z_ = rate(samp[(exp, m)], d, half).mean(), rate(base, d, half).mean(), rate(samp[(exp, r0)], d, half).mean()
                vals[half] = (1 - (a_ - b_) / (z_ - b_)) if z_ != b_ else np.nan
            o["r"][d], o["h1"][d], o["h2"][d] = vals[None], vals[1], vals[2]
            o["rates"][d] = {"arm": float(rate(samp[(exp, m)], d, None).mean()), "R0": float(rate(samp[(exp, r0)], d, None).mean()),
                             "base": float(rate(base, d, None).mean())}
        out[arm] = o
    return out


def load_em():
    sys.path.insert(1, str(T.ROOT / "fb3" / "src"))
    import fb3_common as F
    import fb3_analyze as FA
    F1 = FA.F1
    pan = F.panel()
    key = F.C.read_jsonl(F.JUDGE / "main_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in F.C.read_jsonl(F.JUDGE / "main_replies.jsonl")}
    qids = {d: [r["qid"] for r in pan[d]] for d in D}
    models = ["base", "P0"] + list(EM_ARMS)
    B = {(m, d): np.full((len(qids[d]), F.N_SAMP), np.nan) for m in models for d in D}
    for k in key:
        if k["part"] != "main" or k["model"] not in models:
            continue
        ids = k["ids"]
        b, _ = F1.parse_score(rep[ids["bad"]]) if ids else (None, "EMPTY")
        if b is not None:
            B[(k["model"], k["topic"])][qids[k["topic"]].index(k["qid"]), k["sample_idx"]] = b
    A3 = T.read_json(T.ROOT / "fb3" / "results" / "analysis.json")
    out = {}
    for arm, anc in EM_ARMS.items():
        o = {"pool": "EM", "anchor": anc, "trained": T.FIN, "secondary_trained": "r:P0", "cells": em_cells(arm), "r": {}, "h1": {}, "h2": {}}
        for d in D:
            vals = {}
            for half, cols in ((None, [0, 1]), (1, [0]), (2, [1])):
                with np.errstate(all="ignore"):
                    q = {m: np.nanmean(B[(m, d)][:, cols], 1) for m in models}
                sh, sh0 = q[arm] - q["base"], q["P0"] - q["base"]
                with np.errstate(all="ignore"):
                    vals[half] = 1 - np.nanmean(sh) / np.nanmean(sh0)
            o["r"][d], o["h1"][d], o["h2"][d] = vals[None], vals[1], vals[2]
            assert abs(vals[None] - A3["topic"][d]["r"][arm]["est"]) < 1e-9, (arm, d, vals[None], A3["topic"][d]["r"][arm]["est"])
        out[arm] = o
    return out


def main():
    ref, em = load_ref(), load_em()
    pb3 = T.read_json(T.ROOT / "pb3" / "results" / "analysis.json")
    for arm in ("Rev-F", "Rev-S", "Rev-Fp"):
        for d in D:
            assert abs(ref[arm]["r"][d] - pb3["cells"][f"g-{d}"]["r"][arm]["est"]) < 1e-9, (arm, d)
    for d in D:
        assert abs(ref["R-every"]["r"][d] - pb3["cells"][f"g-{d}"]["r"]["PB2-R-every"]["est"]) < 1e-9, d
    out = {"REF": ref, "EM": em, "checks": "REF r equals PB3's analysis.json (Rev-F, Rev-S, Rev-Fp, PB2 R-every); EM r equals FB3's analysis.json"}
    T.write_json(T.RESULTS / "outcomes.json", out)
    for pool, arms in (("REF", ref), ("EM", em)):
        for a, o in arms.items():
            v = [o["r"][d] for d in o["cells"]]
            print(f"{pool} {a}: {len(o['cells'])} cells, mean r {np.mean(v):.3f}, SD {np.std(v):.3f}")


if __name__ == "__main__":
    main()
