"""Gate 0 analysis (login node, CPU; handoff 5.1-5.4, PREREG A10, B11, B12).

  e20_gate0.py   -> results/gate0.json

Classifiers fitted on the 23 twin pairs from 19a's stored features (no standardisation): w_RAW (raw12), w_ENC (enc12), w_ENCALL (encall),
w_STATE (R2 P_OUT|real, layer 20). w = mean over pairs of (broad - narrow); s(x) = (w.x - m_N) / (m_B - m_N). D_LOPO = the 23 held-out values
of s_broad - s_narrow (w and m_B, m_N from the other 22 pairs). Sign-flip null: 1,000 random +-1 vectors over the pairs (seed 0); each
refit's w', m_B', m_N' give s'(normal) - s'(KL); p = share >= observed.
Students (19a features; files identical: tokens bitwise equal in Job G0): T-normal / T-KL (seed 0), C-normal-s1 / C-KL-s1 (seed 1), plus
C-normal-s2 and C-normal-s0@40 (descriptive). Comparators: KS-plain (= PC-careful), KS-KL, PC-harmful, PC-careful.
Gradient arm: stats/g0/<student>.pt fingerprints (RAW scored with w_RAW, ENC-true with w_ENC).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
from e20common import X19

PAIRS = X19.PAIRS[:23]
STUDENTS = ["T-normal", "T-KL", "C-normal-s1", "C-KL-s1", "C-normal-s2", "C-normal-s0@40"]
SEED_PAIRS = {"s0": ("T-normal", "T-KL"), "s1": ("C-normal-s1", "C-KL-s1")}
COMPARATORS = {"KS-plain": "PC-careful", "KS-KL": "KS-KL", "PC-harmful": "PC-harmful", "PC-careful": "PC-careful"}
STEP = {"T-normal": 150, "T-KL": 150, "C-normal-s1": 150, "C-KL-s1": 150, "C-normal-s2": 150, "C-normal-s0@40": 40}
N_NULL = 1000


def feat(model, key):
    if key == "state":
        S = torch.load(X.E19 / "stats" / "jobB" / "R2" / f"{model}.pt", map_location="cpu", weights_only=True)
        return S["states"]["P_OUT|real"][S["layers"].index(20)].double().numpy()
    P = torch.load(X.E19 / "stats" / "profiles" / f"{model}.pt", map_location="cpu", weights_only=True)
    return P[{"raw": "raw12", "enc": "enc12", "encall": "encall"}[key]].double().flatten().numpy()


class Fit:
    def __init__(self, Xb, Xn):
        self.Xb, self.Xn = Xb, Xn
        self.D = Xb - Xn
        self.w = self.D.mean(0)
        self.mB, self.mN = float((Xb @ self.w).mean()), float((Xn @ self.w).mean())

    def s(self, x):
        return (x @ self.w - self.mN) / (self.mB - self.mN)

    def lopo(self):
        P = len(self.D)
        out = []
        for p in range(P):
            keep = [q for q in range(P) if q != p]
            w = self.D[keep].mean(0)
            mB, mN = float((self.Xb[keep] @ w).mean()), float((self.Xn[keep] @ w).mean())
            out.append(float(self.D[p] @ w) / (mB - mN))
        return np.array(out)

    def null_diffs(self, xa, xb, n=N_NULL, seed=0):
        """s'(xa) - s'(xb) under n sign-flip refits (pairs flipped broad <-> narrow at random)."""
        rng = np.random.default_rng(seed)
        S = rng.choice([-1.0, 1.0], size=(n, len(self.D)))
        W = S @ self.D / len(self.D)                                   # [n, F]
        gap = np.einsum("nf,nf->n", W, W)                             # m_B' - m_N' = w'.w'
        return (W @ (xa - xb)) / gap


def main():
    res = {"features": {}, "notes": {}}
    keys = ["raw", "enc", "encall", "state"]
    fits = {}
    for key in keys:
        Xb = np.stack([feat(b, key) for b, _ in PAIRS])
        Xn = np.stack([feat(n, key) for _, n in PAIRS])
        f = Fit(Xb, Xn)
        fits[key] = f
        D = f.lopo()
        ent = {"dim": int(Xb.shape[1]), "D_LOPO": D.tolist(), "D_LOPO_q25": float(np.percentile(D, 25)), "D_LOPO_median": float(np.median(D)),
               "lopo_correct": int((D > 0).sum()), "students": {}, "comparators": {}, "seeds": {}}
        for m in STUDENTS:
            ent["students"][m] = float(f.s(feat(m, key)))
        for lab, m in COMPARATORS.items():
            ent["comparators"][lab] = float(f.s(feat(m, key)))
        ok_all = True
        for sd, (nm, kl) in SEED_PAIRS.items():
            xa, xb = feat(nm, key), feat(kl, key)
            obs = float(f.s(xa) - f.s(xb))
            null = f.null_diffs(xa, xb)
            ok = obs >= ent["D_LOPO_q25"]
            ok_all &= ok
            ent["seeds"][sd] = {"normal": nm, "KL": kl, "s_normal_minus_s_KL": obs, "meets_bar": bool(ok), "signflip_p": float((null >= obs).mean()),
                                "null_median": float(np.median(null))}
        ent["TRANSFERS"] = bool(ok_all)
        res["features"][key] = ent
        print(f"[gate0] w_{key.upper():6s} D_LOPO q25 {ent['D_LOPO_q25']:.3f} median {ent['D_LOPO_median']:.3f} ({ent['lopo_correct']}/23 > 0); "
              + "; ".join(f"{sd}: {v['s_normal_minus_s_KL']:.3f} (p {v['signflip_p']:.3f})" for sd, v in ent["seeds"].items())
              + f" -> TRANSFERS {ent['TRANSFERS']}")
    # A1 and Delta M_evil from 19a (Job A; finance in-topic gate60 / everyday48)
    A = json.loads((X.E19 / "results" / "jobA.json").read_text())["models"]
    AM = json.loads((X.E19 / "results" / "jobA_models.json").read_text())["models"]
    res["a1_mevil"] = {m: {"step": STEP[m], "A1": A[m]["sets"]["gate60-risky_financial"]["A1"], "delta_mevil": A[m]["delta_mevil"]} for m in STUDENTS}
    res["a1_mevil_comparators"] = {lab: {"A1": AM[m]["sets"]["gate60-risky_financial"]["A1"], "delta_mevil": AM[m]["delta_mevil"]} for lab, m in COMPARATORS.items()}
    res["EARLY"] = "not evaluable: the KL students survive only at step 150 (Exp 12 pruned steps 0-100 except F-normal-s0 step 40)"
    res["A1_first_divergence"] = "not computable (no time course; 19a Part C)"
    # gradient arm
    G = {}
    for m in STUDENTS:
        f = X.STATS / "g0" / f"{m}.pt"
        if f.exists():
            G[m] = torch.load(f, map_location="cpu", weights_only=False)
    ga = {"per_student": {}, "per_checkpoint": {}}
    wmap = {"raw": "raw", "enc": "enc"}
    for m, g in G.items():
        ga["per_student"][m] = {vn: {key: float(fits[key].s(g["fingerprints"][vn][wmap[key]].double().flatten().numpy())) for key in ("raw", "enc")}
                                for vn in g["fingerprints"]}
        ga["per_student"][m]["meta"] = {k: g["meta"][k] for k in ("step", "kind", "seed", "loss_sft0", "loss_sft1") if k in g["meta"]}
        for k in ("kl_kl19a", "kl_klS2"):
            if k in g["meta"]:
                ga["per_student"][m]["meta"][k] = g["meta"][k]
    readable = {}
    for key in ("raw", "enc"):
        conds = []
        for sd, (nm, kl) in SEED_PAIRS.items():
            if nm not in G or kl not in G:
                continue
            sN = {b: ga["per_student"][nm][f"sft{b}"][key] for b in (0, 1)}
            sK = {b: ga["per_student"][kl][f"sft{b}"][key] for b in (0, 1)}
            spread = max(abs(sN[0] - sN[1]), abs(sK[0] - sK[1]))
            for b in (0, 1):
                xa = G[nm]["fingerprints"][f"sft{b}"][key].double().flatten().numpy()
                xb = G[kl]["fingerprints"][f"sft{b}"][key].double().flatten().numpy()
                obs = sN[b] - sK[b]
                null = fits[key].null_diffs(xa, xb)
                c1, c2 = obs > 0, abs(obs) > spread
                conds.append(c1 and c2)
                ga["per_checkpoint"][f"{key}|{sd}|step150|batch{b}"] = {"s_normal": sN[b], "s_KL": sK[b], "diff": obs, "batch_spread": spread,
                                                                       "normal_gt_KL": bool(c1), "diff_gt_batch_spread": bool(c2),
                                                                       "signflip_p": float((null >= obs).mean())}
        readable[key] = bool(conds) and all(conds)
        print(f"[gate0] gradient arm w_{key.upper()}: " + "; ".join(f"{k.split('|', 1)[1]}: {v['diff']:+.3f} (spread {v['batch_spread']:.3f}, p {v['signflip_p']:.3f})"
                                                       for k, v in ga["per_checkpoint"].items() if k.startswith(key + "|")) + f" -> GRADIENT-READABLE {readable[key]}")
    ga["GRADIENT_READABLE"] = readable
    ga["note"] = ("judged at the step-150 checkpoints of seeds 0 and 1, the only checkpoints where both students survive; step 40 (normal s0) "
                  "and normal s2 are descriptive. KL-term variants (sft0+3kl19a: 'KL term on the 19a anchor'; sft0+3klS2: Exp 12's own anchor) "
                  "are descriptive.")
    res["gradient_arm"] = ga
    X.write_json(X.RESULTS / "gate0.json", res)


if __name__ == "__main__":
    main()
