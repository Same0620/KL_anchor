"""Gate 2 analysis (login node, CPU; handoff 6.3-6.8, PREREG B8-B10).

  e20_gate2.py --interim   -> results/gate2_interim.json   (interim 80, label B, F1 / F1b / F2: CONTINUE unless all rho < 0.05)
  e20_gate2.py --final     -> results/gate2.json           (all labels, transfer, reverse transfer, A1-TRACKS-REACH, decision row)

Features (stats/g2/<oid>.pt): F1 = R2 P_EM state at layer 20 (layers 10 and 30 descriptive), F1b = the same with P_REACH, F2 = ENC-ALL (8,972),
size (1-D). Classifier: standardise with the fitting organisms' mean / SD (SD floored at 0.1 x the median SD), w = mean(top k) - mean(bottom
k) by label (k = round(n / 3)), prediction w . x~. CV: the 5 group folds, held-out predictions z-scored within each fold, pooled Spearman;
permutation p = (1 + #{rho_perm >= rho}) / 1001 over 1,000 label shuffles (seed 0) with the whole CV rerun.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
from e20common import X19

G2 = X.STATS / "g2"
N_PERM = 1000
LABELS = ["B", "F", "R", "near", "logitB"]
FEATURES = {"F1": ("F1", 20), "F1b": ("F1b", 20), "F2": ("F2", None), "F1@10": ("F1", 10), "F1@30": ("F1", 30), "F1b@10": ("F1b", 10),
            "F1b@30": ("F1b", 30), "size": ("size", None)}
PRIMARY = ["F1", "F1b", "F2"]
TAG = {"tight": 0, "medium": 1, "leaky": 2}


def ge(null, obs):
    """null >= obs with a relative tolerance: a 1-D feature (size) gives exactly repeated null values, and float noise must not break ties."""
    return null >= obs - 1e-9 * max(1.0, abs(obs))


def le(null, obs):
    return null <= obs + 1e-9 * max(1.0, abs(obs))


def rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    s = a[o]
    i = 0
    while i < len(a):                                                   # average ranks over ties
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])


def load_feature(oid, key):
    src, layer = FEATURES[key]
    d = torch.load(G2 / f"{oid}.pt", map_location="cpu", weights_only=False)
    v = d[src]
    if src == "size":
        return np.array([float(v)])
    v = v.double()
    if layer is not None:
        v = v[d["layers"].index(layer)]
    return v.numpy().ravel()


def stdz(Xf):
    mu = Xf.mean(0)
    sd = Xf.std(0)
    sd = np.maximum(sd, 0.1 * np.median(sd)) if sd.size > 1 else np.maximum(sd, 1e-12)
    return mu, sd


def direction(Xt, y):
    k = int(round(len(y) / 3))
    o = np.argsort(y, kind="stable")
    return Xt[o[-k:]].mean(0) - Xt[o[:k]].mean(0)


def r_label(near, far, fit_idx):
    b, a = np.polyfit(near[fit_idx], far[fit_idx], 1)
    return far - (a + b * near)


class CV:
    """Fold-wise standardisation is label-free, so it is computed once; each (permuted) label only refits w."""

    def __init__(self, Xall, folds):
        self.folds = np.asarray(folds)
        self.parts = []
        for f in sorted(set(folds)):
            tr, ho = np.where(self.folds != f)[0], np.where(self.folds == f)[0]
            mu, sd = stdz(Xall[tr])
            self.parts.append((tr, ho, (Xall[tr] - mu) / sd, (Xall[ho] - mu) / sd))

    def predict(self, lab, near=None, far=None):
        """lab: label array, or 'R' with near / far (R refitted on each training fold). -> pooled z-scored predictions, label used."""
        pred = np.empty(len(self.folds))
        ylab = np.empty(len(self.folds))
        for tr, ho, Xt, Xh in self.parts:
            if isinstance(lab, str):
                full = np.empty(len(self.folds))
                full[:] = r_label(near, far, tr)
                y = full
            else:
                y = lab
            w = direction(Xt, y[tr])
            p = Xh @ w
            pred[ho] = (p - p.mean()) / (p.std() if p.std() > 0 else 1.0)
            ylab[ho] = y[ho]
        return pred, ylab

    def rho_p(self, lab, near=None, far=None, n_perm=N_PERM, seed=0):
        pred, ylab = self.predict(lab, near, far)
        rho = spearman(pred, ylab)
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            perm = rng.permutation(len(self.folds))
            if isinstance(lab, str):
                pp, yy = self.predict("R", near[perm], far[perm])
            else:
                pp, yy = self.predict(lab[perm])
            null.append(spearman(pp, yy))
        null = np.array(null)
        return {"rho": rho, "p": float((1 + ge(null, rho).sum()) / (1 + n_perm)), "null_median": float(np.median(null)),
                "null_q99": float(np.quantile(null, 0.99))}, pred, ylab


def labels_for(oids):
    L = X.read_json(X.DATA / "labels.json")
    return {k: np.array([L[o]["main"][k] for o in oids], float) for k in ("near", "far", "B", "F", "R", "logitB")}


def interim():
    S = X.sample()
    oids = S["interim80"]
    folds = [S["folds80"][o] for o in oids]
    lab = labels_for(oids)
    out = {"n": len(oids), "label": "B", "features": {}}
    for key in PRIMARY:
        Xa = np.stack([load_feature(o, key) for o in oids])
        r, _, _ = CV(Xa, folds).rho_p(lab["B"])
        out["features"][key] = r
        print(f"[interim] {key}: rho {r['rho']:+.3f}, p {r['p']:.3f}")
    stop = all(out["features"][k]["rho"] < 0.05 for k in PRIMARY)
    out["decision"] = "STOP: NO WITHIN-SDF SIGNAL" if stop else "CONTINUE"
    X.write_json(X.RESULTS / "gate2_interim.json", out)
    print(f"[interim] -> {out['decision']}")


def twin_matrix(key, twins):
    return np.stack([load_feature(t, key) for t in twins])


def final():
    S = X.sample()
    ORG = X.organisms()
    T = X.read_json(X.DATA / "twins.json")
    oids, test = S["train200"], S["test40"]
    folds = [S["folds200"][o] for o in oids]
    lab = labels_for(oids)
    lab_test = labels_for(test)
    lab["R"] = r_label(lab["near"], lab["far"], np.arange(len(oids)))      # the 200 fit (transfer); CV refits per fold
    rel = X.read_json(X.DATA / "reliability.json")["per_label"]["train200"]
    pairs = T["pairs"]
    twins = [m for p in pairs for m in p]
    tw = T["twins"]
    res = {"n_train": len(oids), "n_test": len(test), "cv": {}, "transfer": {}, "reverse": {}, "descriptive": {}}
    waves = np.array([ORG[o]["wave"] for o in oids])
    groups = np.array([ORG[o]["group"] for o in oids])
    tags = np.array([TAG[ORG[o]["breadth"]] for o in oids], float)
    for key in FEATURES:
        Xa = np.stack([load_feature(o, key) for o in oids])
        cv = CV(Xa, folds)
        ent = {}
        for lb in LABELS:
            if lb == "R":
                r, pred, ylab = cv.rho_p("R", lab["near"], lab["far"])
            else:
                r, pred, ylab = cv.rho_p(lab[lb])
            ceil = rel[lb if lb != "F" else "F"]["ceiling"]
            r["ceiling"] = ceil
            r["rho_over_ceiling"] = r["rho"] / ceil
            if lb == "B":
                r["VALID"] = bool(r["rho"] >= 0.2 and r["p"] <= 0.01)
                r["per_wave"] = {w: {"rho": spearman(pred[waves == w], ylab[waves == w]), "n": int((waves == w).sum())} for w in sorted(set(waves))}
                gr = [spearman(pred[groups == g], ylab[groups == g]) for g in sorted(set(groups)) if (groups == g).sum() >= 5]
                r["mean_within_group_rho"] = float(np.nanmean(gr)) if gr else None
                r["n_groups_ge5"] = len(gr)
                r["rho_vs_breadth_tag"] = spearman(pred, tags)
                fam = {}
                for i, o in enumerate(oids):
                    if waves[i] == "5x":
                        fam.setdefault((ORG[o]["attractor"], ORG[o]["group"]), []).append(i)
                fr = [spearman(pred[ix], ylab[ix]) for ix in fam.values() if len(ix) >= 3]
                r["within_family_5x"] = {"n_families_ge3": len(fr), "mean_rho": float(np.nanmean(fr)) if fr else None}
            ent[lb] = r
        # test organisms (descriptive; soft split): fit on all 200, predict the 40
        mu, sd = stdz(Xa)
        Xt = (Xa - mu) / sd
        w = direction(Xt, lab["B"])
        Xte = (np.stack([load_feature(o, key) for o in test]) - mu) / sd
        ent["test40_B"] = {"rho": spearman(Xte @ w, lab_test["B"]), "n": len(test)}
        res["cv"][key] = ent
        print(f"[cv] {key:7s} " + "  ".join(f"{lb} {ent[lb]['rho']:+.3f} (p {ent[lb]['p']:.3f})" for lb in LABELS)
              + f"  VALID {ent['B']['VALID']}; test40 {ent['test40_B']['rho']:+.3f}")
        # ---- transfer to the twins (6.5) and reverse transfer (6.6d)
        Xw = twin_matrix(key, twins)
        Xwt = (Xw - mu) / sd
        Db = Xwt[0::2] - Xwt[1::2]                                          # broad - narrow per pair
        w_twin = Db.mean(0)
        tr = {}
        for lb in ("B", "near", "R", "F", "logitB"):
            y = lab[lb]
            w = direction(Xt, y)
            sd_pred = (Xt @ w).std()
            Tobs = float((Db @ w).mean() / sd_pred)
            rng = np.random.default_rng(0)
            null = []
            for _ in range(N_PERM):
                wp = direction(Xt, y[rng.permutation(len(y))])
                null.append(float((Db @ wp).mean() / (Xt @ wp).std()))
            null = np.array(null)
            pred_tw = Xwt @ w
            far = np.array([tw[m]["far"] for m in twins])
            kind = np.array([tw[m]["kind"] for m in twins])
            strat = np.array([tw[m]["stratum"] for m in twins])
            Bdef = np.array([tw[m]["B"] if tw[m]["B"] is not None else np.nan for m in twins], float)
            per_pair = Db @ w
            tr[lb] = {"T": Tobs, "p": float((1 + ge(null, Tobs).sum()) / (1 + N_PERM)), "null_q99": float(np.quantile(null, 0.99)),
                      "cos_w_wtwin": float(w @ w_twin / (np.linalg.norm(w) * np.linalg.norm(w_twin))) if np.linalg.norm(w_twin) > 0 else None,
                      "pairs_ordered": int((per_pair > 0).sum()),
                      "T_per_stratum": {st: float(per_pair[[i for i, p in enumerate(pairs) if tw[p[0]]["stratum"] == st]].mean() / sd_pred) for st in ("V6", "O17")},
                      "rho_pred_far_all46": spearman(pred_tw, far), "rho_pred_far_broad": spearman(pred_tw[kind == "broad"], far[kind == "broad"]),
                      "rho_pred_far_narrow": spearman(pred_tw[kind == "narrow"], far[kind == "narrow"]),
                      "rho_pred_B_defined": spearman(pred_tw[np.isfinite(Bdef)], Bdef[np.isfinite(Bdef)]), "n_B_defined": int(np.isfinite(Bdef).sum())}
            if lb == "B":
                others = {}
                for nm in ("T-normal", "T-KL", "C-normal-s1", "C-KL-s1", "PC-harmful", "PC-careful", "KS-KL", "C-normal-s2", "C-normal-s0@40"):
                    if (G2 / f"{nm}.pt").exists():
                        others[nm] = float(((load_feature(nm, key) - mu) / sd) @ w / sd_pred)
                tr[lb]["e19_models_pred_over_sd"] = others
        tr["off_distribution_share"] = float((np.abs(Xwt) > 5).any(1).mean())
        res["transfer"][key] = tr
        # reverse transfer: the 200 projected on w_twin (standardised with the 200's own statistics)
        proj = Xt @ w_twin
        rv = {}
        rng = np.random.default_rng(0)
        Sg = rng.choice([-1.0, 1.0], size=(N_PERM, len(Db)))
        Wn = Sg @ Db / len(Db)
        Pn = Xt @ Wn.T                                                      # [200, N_PERM]
        for lb in ("B", "F", "R", "near"):
            rho = spearman(proj, lab[lb])
            null = np.array([spearman(Pn[:, j], lab[lb]) for j in range(N_PERM)])
            rv[lb] = {"rho": rho, "p": float(ge(null, rho).mean())}
        res["reverse"][key] = rv
        print(f"[transfer] {key:7s} T_B {tr['B']['T']:+.3f} (p {tr['B']['p']:.3f}, cos {tr['B']['cos_w_wtwin']}); T_near {tr['near']['T']:+.3f} "
              f"(p {tr['near']['p']:.3f}); T_R {tr['R']['T']:+.3f} (p {tr['R']['p']:.3f}); off-dist {tr['off_distribution_share']:.2f}; "
              f"reverse rho_B {rv['B']['rho']:+.3f} (p {rv['B']['p']:.3f})")
    # ---- labels
    L = {}
    for key in PRIMARY:
        cvB, trB = res["cv"][key]["B"], res["transfer"][key]
        strength = trB["near"]["p"] <= 0.01
        L[key] = {"VALID": cvB["VALID"], "TRANSFERS_B": bool(cvB["VALID"] and trB["B"]["p"] <= 0.01), "STRENGTH_TRANSFERS": bool(strength),
                  "TRANSFERS_R": bool(trB["R"]["p"] <= 0.01) if strength else None, "T_size": res["transfer"]["size"]["B"]["T"],
                  "T_size_p": res["transfer"]["size"]["B"]["p"]}
    res["labels"] = L
    # ---- A1-TRACKS-REACH (6.6a-c)
    allo = oids + test
    Ball = np.concatenate([lab["B"], lab_test["B"]])
    a1 = {}
    for ver in ("sdf", "std", "probe", "sdf_pp", "std_pp", "probe_pp"):
        v = np.array([torch.load(G2 / f"{o}.pt", map_location="cpu", weights_only=False)["a1"][ver] for o in allo], float)
        rho = spearman(v, Ball)
        rng = np.random.default_rng(0)
        null = np.array([spearman(v, Ball[rng.permutation(len(Ball))]) for _ in range(N_PERM)])
        ent = {"rho": rho, "p": float((1 + le(null, rho).sum()) / (1 + N_PERM)), "n": int(np.isfinite(v).sum())}
        if ver == "sdf":
            ent["A1_TRACKS_REACH"] = bool(rho <= -0.2 and ent["p"] <= 0.01)
            wv = np.array([ORG[o]["wave"] for o in allo])
            tg = np.array([ORG[o]["breadth"] for o in allo])
            ent["per_wave"] = {w: spearman(v[wv == w], Ball[wv == w]) for w in sorted(set(wv))}
            ent["per_tag"] = {t: {"rho": spearman(v[tg == t], Ball[tg == t]), "median_A1": float(np.nanmedian(v[tg == t]))} for t in TAG}
        a1[ver] = ent
    tv = {m: torch.load(G2 / f"{m}.pt", map_location="cpu", weights_only=False)["a1"] for m in twins if (G2 / f"{m}.pt").exists()}
    if len(tv) == 46:
        sdf = np.array([tv[m]["sdf"] for m in twins])
        far = np.array([tw[m]["far"] for m in twins])
        a1["twins"] = {"pairs_narrow_higher": int(sum(1 for i in range(23) if sdf[2 * i + 1] > sdf[2 * i])), "n_pairs": 23,
                       "rho_with_far_all46": spearman(sdf, far), "rho_with_19a_A1": spearman(sdf, [tw[m]["A1_19a"] for m in twins]),
                       "median_broad": float(np.median(sdf[0::2])), "median_narrow": float(np.median(sdf[1::2]))}
    res["a1"] = a1
    print(f"[a1] A1-SDF vs B: rho {a1['sdf']['rho']:+.3f} (p {a1['sdf']['p']:.3f}) -> A1-TRACKS-REACH {a1['sdf']['A1_TRACKS_REACH']}; std {a1['std']['rho']:+.3f}; "
          f"probe {a1['probe']['rho']:+.3f}")
    # ---- decision-table row (F1 primary)
    f1 = L["F1"]
    ceil_B = rel["B"]["ceiling"]
    if f1["VALID"]:
        if f1["TRANSFERS_B"] and (not f1["STRENGTH_TRANSFERS"] or f1["TRANSFERS_R"]):
            row = ("VALID and TRANSFERS-B (and TRANSFERS-R where STRENGTH-TRANSFERS): a shared, transferable reach direction → Gate 4 (fine-tune a "
                   "LoRAcle) is worth discussing, after contacting cds-jb")
        elif f1["STRENGTH_TRANSFERS"]:
            row = ("VALID, STRENGTH-TRANSFERS, but no TRANSFERS-B/R: cross-recipe transfer is possible, but the reach direction does not carry → "
                   "matched-recipe organisms would be needed")
        else:
            row = "VALID, nothing transfers (not even strength): recipe shift; uninformative about reach → stop this route without updating on reach"
    else:
        row = ("not VALID, ceiling ≥ 0.6: no linear reach signal in these features for document organisms → stop" if ceil_B >= 0.6 else
               "not VALID, ceiling < 0.6: the labels cannot support the test; not evidence → fix the labels before deciding")
    res["decision_row"] = row
    print(f"[decision] {row}")
    X.write_json(X.RESULTS / "gate2.json", res)


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)
    interim() if "--interim" in sys.argv else final()
