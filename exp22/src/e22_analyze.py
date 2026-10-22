"""Exp 22 analysis (CPU; PREREG B9, Part C). Reads /home/u4161854/e22_stats/m/*.pt, the training records and exp22's Phase-0 files;
writes exp22/results/analysis.json. Prints a short summary (numbers only).

Every interval: joint prompt bootstrap, 2,000 resamples, percentile 95%, one index draw per class (or per row set / pair set) shared by
every adapter. Seed 22.
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
E22, E20B = ROOT / "exp22", ROOT / "exp20b"
SM, CK, CK20 = Path("/home/u4161854/e22_stats/m"), Path("/home/u4161854/e22_ckpt"), Path("/home/u4161854/e20b_ckpt")
NB, SEED = 2000, 22
BANDS = {"late": [3, 4, 5], "early": [0, 1, 2], "all": [0, 1, 2, 3, 4, 5]}
LAYERS = [10, 14, 18, 22, 26, 30]
ADAPTERS = ["P0", "P0-s2", "K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1", "K-code", "K-fmt"]
KL_ARMS = ["K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1", "K-code"]
NEW_KL = ["K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1"]
OWN_ANCHOR = {"K-every": "every", "K-adv1": "adv1", "K-adv1-r1": "adv1", "K-advN": "advN", "K-codeL": "code", "K-codeL-r1": "code", "K-code": "code"}
OWN_CLASSES = {"K-every": {"2"}, "K-adv1": {"3"}, "K-adv1-r1": {"3"}, "K-advN": {"4"}, "K-codeL": {"6", "7"}, "K-codeL-r1": {"6", "7"},
               "K-code": {"6", "7"}, "K-fmt": {"2"}}
CLASSES = {"1": ["gate60-risky_financial"], "1-24": ["intopic24-risky_financial"], "2": ["everyday48"], "3": ["gen24-bad_petcare"],
           "4": ["gen24-bad_beauty", "gen24-bad_pesticides", "gen24-unsafe_chemicals"], "4-beauty": ["gen24-bad_beauty"],
           "4-pesticides": ["gen24-bad_pesticides"], "4-chemicals": ["gen24-unsafe_chemicals"], "5a": ["gate60-extreme_sports"],
           "5b": ["gate60-bad_legal"], "6": ["mbpp24_clean"], "6-orig": ["mbpp24"], "7": ["gsm8k24"], "9": ["passage24"]}
MAIN9 = ["1", "2", "3", "4", "5a", "5b", "6", "7", "9"]
PAIR_ARMS = [("K-adv1", "K-adv1-r1"), ("K-codeL", "K-codeL-r1")]
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
rd = lambda p: [json.loads(l) for l in open(p)]


def load():
    B = torch.load(SM / "base.pt", map_location="cpu", weights_only=True)
    M = {a: torch.load(SM / f"{a}.pt", map_location="cpu", weights_only=True) for a in ADAPTERS}
    return B, M


def class_d(M, a, c, split=None):
    """[n, 6, H] float32 differences of adapter a on class c (sets concatenated); split: index subset."""
    d = torch.cat([M[a]["d"][s].float() for s in CLASSES[c]])
    return d if split is None else d[torch.as_tensor(split)]


def boot_norms(d, idx):
    """d [n, 6, H]; idx [NB, n] resampled rows -> per-layer norm of the mean: point [6], boot [NB, 6]."""
    pt = d.double().mean(0).norm(dim=-1).numpy()
    D = d.reshape(d.shape[0], -1)
    out = []
    for b0 in range(0, idx.shape[0], 200):
        W = torch.zeros(min(200, idx.shape[0] - b0), d.shape[0])
        for r, row in enumerate(idx[b0:b0 + 200]):
            W[r].index_add_(0, torch.as_tensor(row), torch.ones(len(row)))
        mb = (W / idx.shape[1]) @ D
        out.append(mb.reshape(mb.shape[0], 6, -1).double().norm(dim=-1))
    return pt, torch.cat(out).numpy()


def band(v, b):
    return v[..., BANDS[b]].mean(-1)


def update_map(M, rng, split_halves):
    """N per adapter x class x band (points and bootstraps), S = 1 - N/N_P0 with CIs, the per-layer ratio, CLASS EVALUABLE, the nulls."""
    out = {"N": {}, "S": {}, "layers": {}, "evaluable": {}, "thresholds": {}, "nulls": {}}
    bootS = {}
    keys = list(CLASSES) + ["2m", "2u"]
    for c in keys:
        base_c = "2" if c in ("2m", "2u") else c
        sub = split_halves[c] if c in ("2m", "2u") else None
        n = class_d(M, "P0", base_c, sub).shape[0]
        idx = rng.integers(0, n, (NB, n))
        pts, bts = {}, {}
        for a in ADAPTERS:
            pts[a], bts[a] = boot_norms(class_d(M, a, base_c, sub), idx)
        for a in ADAPTERS:
            for b in BANDS:
                out["N"].setdefault(a, {}).setdefault(c, {})[b] = float(band(pts[a], b))
                if a == "P0":
                    continue
                s_pt = 1 - band(pts[a], b) / band(pts["P0"], b)
                s_bt = 1 - band(bts[a], b) / band(bts["P0"], b)
                bootS[(a, c, b)] = s_bt
                out["S"].setdefault(a, {}).setdefault(c, {})[b] = {"S": float(s_pt), "ci": pct(s_bt)}
            out["layers"].setdefault(a, {})[c] = [float(x) for x in (pts[a] / pts["P0"])]
        # CLASS EVALUABLE (item 5a): N_late(P0) > 4 x its bootstrap SE and n >= 24
        for b in BANDS:
            se = float(band(bts["P0"], b).std(ddof=1))
            out["evaluable"].setdefault(c, {})[b] = {"N_P0": float(band(pts["P0"], b)), "se": se, "n": n,
                                                     "evaluable": bool(band(pts["P0"], b) > 4 * se and n >= 24)}
        # thresholds (E4, F2, F6): max of the P0 half-split null and the P0 seed-pair null, per band
        if c in MAIN9 or c in ("2m", "2u"):
            g = np.random.default_rng(SEED + len(c) + ord(c[0]))
            perm = g.permutation(n)
            h1, h2 = perm[: n // 2], perm[n // 2:]
            d0 = class_d(M, "P0", base_c, sub)
            i1 = h1[rng.integers(0, len(h1), (NB, len(h1)))]
            i2 = h2[rng.integers(0, len(h2), (NB, len(h2)))]
            _, b1 = boot_norms(d0, i1)
            _, b2 = boot_norms(d0, i2)
            for b in BANDS:
                half = float(np.percentile(np.abs(1 - band(b1, b) / band(b2, b)), 95))
                seed = float(np.percentile(np.abs(bootS[("P0-s2", c, b)]), 95))
                out["nulls"].setdefault(c, {})[b] = {"half_split": half, "seed_pair": seed,
                                                     "K-fmt_abs_S": abs(out["S"]["K-fmt"][c][b]["S"])}
                out["thresholds"].setdefault(c, {})[b] = max(half, seed)
    return out, bootS


def suppressed(U, a, c, b="late"):
    s = U["S"][a][c][b]
    return bool(s["S"] >= U["thresholds"][c][b] and s["ci"][0] > 0)


def output_drift(B, M, rng):
    """S_KL per adapter x class on the class's greedy base answers; thresholds by the same two P0 nulls (F3's row only)."""
    out = {"KL": {}, "S_KL": {}, "thresholds": {}}
    for c in MAIN9 + ["1-24", "6-orig"]:
        sets = [f"ans_{s}" for s in CLASSES[c]]
        v = {a: torch.cat([M[a]["kl"][k] for k in sets]).double().numpy() for a in ADAPTERS}
        n = len(v["P0"])
        idx = rng.integers(0, n, (NB, n))
        for a in ADAPTERS:
            out["KL"].setdefault(a, {})[c] = {"KL": float(v[a].mean()), "ci": pct(v[a][idx].mean(1))}
            if a != "P0":
                out["S_KL"].setdefault(a, {})[c] = {"S": float(1 - v[a].mean() / v["P0"].mean()), "ci": pct(1 - v[a][idx].mean(1) / v["P0"][idx].mean(1))}
        perm = np.random.default_rng(SEED + 7).permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2:]
        b1 = v["P0"][h1[rng.integers(0, len(h1), (NB, len(h1)))]].mean(1)
        b2 = v["P0"][h2[rng.integers(0, len(h2), (NB, len(h2)))]].mean(1)
        seedpair = 1 - v["P0-s2"][idx].mean(1) / v["P0"][idx].mean(1)
        out["thresholds"][c] = max(float(np.percentile(np.abs(1 - b1 / b2), 95)), float(np.percentile(np.abs(seedpair), 95)))
    return out


def margins(B, M, rng, split):
    """dM per adapter x margin set (adapter - base, mean over pairs), paired CIs, ratio and difference to P0."""
    out = {}
    sets = ["M_evil_plain@1", "M_evil_code@1", "M_evil_json@1", "M_evil_plain@2", "M_lm@1", "M_lm@2", "M_in@1", "M_X@1"]
    ids_mev = [p["id"] for p in rd(ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    halves = {"m": np.array([i for i, x in enumerate(ids_mev) if x in set(split["mevil25"]["matched"])]),
              "u": np.array([i for i, x in enumerate(ids_mev) if x in set(split["mevil25"]["unmatched"])])}
    variants = [(s, None) for s in sets] + [("M_evil_plain@1", "m"), ("M_evil_plain@1", "u")]
    for s, h in variants:
        name = s.split("@")[0]
        key = f"{s}|{h}" if h else s
        v = {a: (M[a]["margins"][s].double() - B["margins"][name].double()).numpy() for a in ADAPTERS}
        rows = halves[h] if h else np.arange(len(v["P0"]))
        idx = rows[rng.integers(0, len(rows), (NB * 5, len(rows)))]                       # 10,000 resamples for the margins
        p0 = v["P0"][idx].mean(1)
        for a in ADAPTERS:
            bt = v[a][idx].mean(1)
            out.setdefault(a, {})[key] = {"dM": float(v[a][rows].mean()), "ci": pct(bt), "ratio": float(v[a][rows].mean() / v["P0"][rows].mean()),
                                          "ratio_ci": pct(bt / p0), "diff_vs_P0": float(v[a][rows].mean() - v["P0"][rows].mean()),
                                          "diff_ci": pct(bt - p0)}
    return out


def anchors_and_gates(M, rng, U, KLm, MG):
    """KL ACTIVE, IMPLANT PRESENT, ANCHOR GENERALISES per KL arm; penalty strength and matching; NARROW REFERENCE tiers."""
    out = {}
    lam = json.loads((E22 / "data" / "lambdas.json").read_text())
    rec = lambda a: json.loads(((CK20 / "K-far") if a == "K-code" else (CK / a)) .joinpath("record.json").read_text())
    win = lambda sl, lo, hi, k: st.mean(s[k] for s in sl if lo <= s["step"] <= hi)
    ev = rec("K-every")
    R_every = win(ev["step_log"], 20, 150, "lam_kl_nominal")
    for a in KL_ARMS:
        an = OWN_ANCHOR[a]
        r = rec(a)
        g = {"lambda": r["lambda"], "realised_lam_kl_20_150": win(r["step_log"], 20, 150, "lam_kl_nominal"),
             "share_20_150": win(r["step_log"], 20, 150, "share"), "share_25_40": win(r["step_log"], 25, 40, "share")}
        g["ratio_to_K-every"] = g["realised_lam_kl_20_150"] / R_every
        rr = g["ratio_to_K-every"]
        g["matching"] = ("reference" if a == "K-every" else "stronger penalty" if rr > 1.5 else "weaker penalty" if rr < 1 / 1.5 else "matched")
        g["drifted"] = bool(a in ("K-adv1", "K-adv1-r1", "K-advN") and not (0.8 <= rr <= 1.25))
        if a in ("K-codeL", "K-codeL-r1") and g["matching"] != "matched":
            g["label_F5"] = "code anchor at ~3x lambda"
        gs = r.get("gsplit") or []
        gr = lambda lo, hi: st.mean(x["ratio_kl_to_sft"] for x in gs if lo <= x["step"] <= hi and x["ratio_kl_to_sft"] is not None) if any(lo <= x["step"] <= hi for x in gs) else None
        g["grad_ratio"] = {"10-20": gr(10, 20), "25-40": gr(25, 40), "20-150": gr(20, 150)}
        # KL ACTIVE on the own anchor's reserved rows (paired row bootstrap)
        res, vis = f"{an}_res", f"{an}_vis"
        x, p = M[a]["kl"][res].double().numpy(), M["P0"]["kl"][res].double().numpy()
        idx = rng.integers(0, len(x), (NB, len(x)))
        diff = p[idx].mean(1) - x[idx].mean(1)
        se = float(diff.std(ddof=1))
        g["KL_ACTIVE"] = {"KL_arm": float(x.mean()), "KL_P0": float(p.mean()), "ratio": float(x.mean() / p.mean()), "diff": float(p.mean() - x.mean()),
                          "se": se, "ci": pct(diff), "pass": bool(p.mean() - x.mean() > 3 * se and pct(diff)[0] > 0)}
        # ANCHOR GENERALISES (D1): the penalty's effect on reserved rows / on visited rows of its own anchor
        xv, pv = M[a]["kl"][vis].double().numpy(), M["P0"]["kl"][vis].double().numpy()
        e_res, e_vis = 1 - x.mean() / p.mean(), 1 - xv.mean() / pv.mean()
        iv = rng.integers(0, len(xv), (NB, len(xv)))
        bt = (1 - x[idx].mean(1) / p[idx].mean(1)) / (1 - xv[iv].mean(1) / pv[iv].mean(1))
        g["ANCHOR_GENERALISES"] = {"effect_reserved": float(e_res), "effect_visited": float(e_vis), "ratio": float(e_res / e_vis),
                                   "ci": pct(bt), "pass": bool(e_res / e_vis >= 0.5)}
        # IMPLANT PRESENT: class-1 drift and dM_in, each >= 0.5 x P0's
        g["IMPLANT_PRESENT"] = {"drift_ratio": KLm["KL"][a]["1"]["KL"] / KLm["KL"]["P0"]["1"]["KL"],
                                "dM_in_ratio": MG[a]["M_in@1"]["dM"] / MG["P0"]["M_in@1"]["dM"]}
        g["IMPLANT_PRESENT"]["pass"] = bool(g["IMPLANT_PRESENT"]["drift_ratio"] >= 0.5 and g["IMPLANT_PRESENT"]["dM_in_ratio"] >= 0.5)
        out[a] = g
    k = out["K-every"]
    k["NARROW_REFERENCE"] = {
        "update": {"S_late_2": U["S"]["K-every"]["2"]["late"]["S"], "ratio": 1 - U["S"]["K-every"]["2"]["late"]["S"],
                   "pass": bool(U["S"]["K-every"]["2"]["late"]["S"] >= 0.5),
                   "S_all_2": U["S"]["K-every"]["2"]["all"]["S"], "S_late_2m": U["S"]["K-every"]["2m"]["late"]["S"],
                   "S_late_2u": U["S"]["K-every"]["2u"]["late"]["S"]},
        "restraint": {"dM_evil_ratio": MG["K-every"]["M_evil_plain@1"]["ratio"], "diff_ci": MG["K-every"]["M_evil_plain@1"]["diff_ci"]}}
        # restraint: ratio <= 0.5, the paired CI of the difference excluding 0, and IMPLANT PRESENT
    kr = k["NARROW_REFERENCE"]["restraint"]
    kr["pass"] = bool(kr["dM_evil_ratio"] <= 0.5 and kr["diff_ci"][1] < 0 and k["IMPLANT_PRESENT"]["pass"])
    for h in ("m", "u"):
        kr[f"dM_evil_ratio_{h}"] = MG["K-every"][f"M_evil_plain@1|{h}"]["ratio"]
    return out, lam


def named_outcomes(U, G, restraint_any):
    """Item 4 + F2 + E2: on S_late, separately with class 5 = 5a and 5b; both tiers. Gates feed NOT EVALUABLE."""
    res = {}
    for five in ("5a", "5b"):
        sup = lambda a, c: suppressed(U, a, c)
        both = lambda pair, c: sup(pair[0], c) and sup(pair[1], c)
        a1 = both(("K-adv1", "K-adv1-r1"), five)
        aN = sup("K-advN", five)
        code5 = both(("K-codeL", "K-codeL-r1"), five)
        code2 = both(("K-codeL", "K-codeL-r1"), "2")
        code_none = (not code5) and (not code2)
        s_a1 = [U["S"][x][five]["late"]["S"] for x in ("K-adv1", "K-adv1-r1")]
        s_aN = U["S"]["K-advN"][five]["late"]["S"]
        fired = []
        if a1 and aN and code_none:
            fired.append("PROMPT-TYPE BOUNDED")
        if (a1 != aN) and code_none:
            fired.append("ONE ADVICE ARM ONLY")
        if aN and not a1 and (s_aN - max(s_a1)) > abs(s_a1[0] - s_a1[1]):
            fired.append("BREADTH, NOT PROMPT TYPE")
        cls27 = ["2", "3", "4", five, "6", "7"]
        if all(sup(a, c) for a in NEW_KL for c in cls27):
            fired.append("NEITHER BOUND")
        if all((sup(a, c) == (c in OWN_CLASSES[a])) for a in NEW_KL for c in cls27):
            fired.append("ANCHOR-LOCAL ONLY")
        gate_fail = [a for a in ("K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1")
                     if not (G[a]["KL_ACTIVE"]["pass"] and G[a]["IMPLANT_PRESENT"]["pass"])]
        class_fail = [c for c in ("2", five) if not U["evaluable"][c]["late"]["evaluable"]]
        gen_fail = [a for a in ("K-adv1", "K-adv1-r1", "K-advN") if not G[a]["ANCHOR_GENERALISES"]["pass"]]
        not_eval = bool(gate_fail or class_fail)
        if not_eval:
            fired = ["NOT EVALUABLE"]
        if not fired:
            fired = ["MIXED / NO CLEAN PATTERN"]
        res[five] = {"fired": fired, "tier": "restraint" if restraint_any else "update",
                     "cells": {"K-adv1": a1, "K-advN": aN, "K-codeL_5": code5, "K-codeL_2": code2},
                     "gate_failures": gate_fail, "class_not_evaluable": class_fail,
                     "anchor_generalises_failures": gen_fail,
                     "note": ("cross-topic nulls on " + ", ".join(gen_fail) + " are NOT EVALUABLE (D1)") if gen_fail else ""}
    return res


def seed_agreement(U):
    calls = []
    for a, b in PAIR_ARMS:
        for c in MAIN9:
            calls.append({"pair": f"{a}/{b}", "class": c, "agree": suppressed(U, a, c) == suppressed(U, b, c),
                          "calls": [suppressed(U, a, c), suppressed(U, b, c)]})
    return {"k": sum(x["agree"] for x in calls), "n": len(calls), "calls": calls,
            "caveat": "two seeds differ only in anchor-batch order (K2): they say nothing about which anchor questions were drawn, A0 or SFT order"}


def h1_h2(B, M, U, G, rng):
    """H1 matrices (E_update raw / centred, E_base, E_text; early / late) and H2 / J2 fits with a joint bootstrap refit."""
    sys.path.insert(0, str(E22 / "src"))
    import e22_classes as K
    sets = {f"anchor-{a}": [f"{a}_res"] for a in ("every", "adv1", "advN", "code")}
    sets.update({f"class-{c}": CLASSES[c] for c in MAIN9})
    dP0 = {k: torch.cat([M["P0"]["d"][s].float() for s in v]) for k, v in sets.items()}
    hB = {k: torch.cat([B["h"][s].float() for s in v]) for k, v in sets.items()}
    names = list(sets)

    def measures(mu_d, mu_b):
        gd = torch.stack([mu_d[k] for k in names]).mean(0)
        gb = torch.stack([mu_b[k] for k in names]).mean(0)
        cos = lambda x, y: torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=-1).numpy()
        E = {}
        for x, y in itertools.combinations(names, 2):
            E[(x, y)] = {"E_update": cos(mu_d[x], mu_d[y]), "E_update_c": cos(mu_d[x] - gd, mu_d[y] - gd), "E_base": cos(mu_b[x] - gb, mu_b[y] - gb)}
        return E

    point = measures({k: v.mean(0) for k, v in dP0.items()}, {k: v.mean(0) for k, v in hB.items()})
    # E_text: MiniLM mean question embedding per set
    emb = K.embedder()
    qtext = {}
    for k, v in sets.items():
        qs = []
        for s in v:
            if s.endswith("_res"):
                qs += [r["messages"][0]["content"] for r in rd(E22 / "data" / f"anchor_{s[:-4]}_reserved.jsonl")]
            else:
                f = next(p for p in (ROOT / "exp19a" / "prompts" / f"{s}.jsonl", E22 / "prompts" / f"{s}.jsonl", E20B / "prompts" / f"{s}.jsonl") if p.exists())
                qs += [r["question"] for r in rd(f)]
        qtext[k] = torch.nn.functional.normalize(emb(qs).mean(0), dim=0)
    mat = {}
    for (x, y), e in point.items():
        mat[f"{x}|{y}"] = {m: {"early": float(e[m][BANDS["early"]].mean()), "late": float(e[m][BANDS["late"]].mean()),
                               "per_layer": [float(t) for t in e[m]]} for m in e}
        mat[f"{x}|{y}"]["E_text"] = float(qtext[x] @ qtext[y])
    Eget = lambda x, y, m="E_update", b="late": (mat.get(f"{x}|{y}") or mat[f"{y}|{x}"])[m][b] if m != "E_text" else (mat.get(f"{x}|{y}") or mat[f"{y}|{x}"])["E_text"]

    # cells: (arm x class), own-anchor classes excluded, K-fmt excluded (J2)
    arms = KL_ARMS
    cells = [(a, c) for a in arms for c in MAIN9 if c not in OWN_CLASSES[a]]
    logR = {a: float(np.log(G[a]["realised_lam_kl_20_150"])) for a in arms}

    def fit(Sv, Ea, Et, which):
        """J2: S = alpha_class + beta E(anchor, class) + gamma log R; second stage alpha_class on E(train, class)."""
        cls = sorted({c for _, c in cells})
        X = np.zeros((len(cells), len(cls) + 2))
        for i, (a, c) in enumerate(cells):
            X[i, cls.index(c)] = 1
            X[i, -2] = Ea[(a, c)]
            X[i, -1] = logR[a]
        y = np.array([Sv[(a, c)] for a, c in cells])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha = coef[: len(cls)]
        et = np.array([Et[c] for c in cls])
        slope = np.polyfit(et, alpha, 1)[0] if len(set(np.round(et, 12))) > 1 else np.nan
        return {"beta": float(coef[-2]), "gamma": float(coef[-1]), "stage2_slope": float(slope)}

    def spearman(x, y):
        rk = lambda v: np.argsort(np.argsort(v))
        return float(np.corrcoef(rk(np.array(x)), rk(np.array(y)))[0, 1])

    def stats_for(Sv, Ea, Ea_c, Et, Etxt):
        s = [Sv[k] for k in cells]
        return {"rho_S_Eupdate": spearman(s, [Ea[k] for k in cells]), "rho_S_Etext": spearman(s, [Etxt[k] for k in cells]),
                "rho_Eanchor_Etrain": spearman([Ea[k] for k in cells], [Et[c] for _, c in cells]),
                "fit_raw": fit(Sv, Ea, Et, "raw"), "fit_centred": fit(Sv, Ea_c, Et, "c")}

    Sv = {(a, c): U["S"][a][c]["late"]["S"] for a, c in cells}
    Ea = {(a, c): Eget(f"anchor-{OWN_ANCHOR[a]}", f"class-{c}") for a, c in cells}
    Ea_c = {(a, c): Eget(f"anchor-{OWN_ANCHOR[a]}", f"class-{c}", "E_update_c") for a, c in cells}
    Et = {c: (1.0 if c == "1" else Eget("class-1", f"class-{c}")) for c in MAIN9}
    Etxt = {(a, c): Eget(f"anchor-{OWN_ANCHOR[a]}", f"class-{c}", "E_text") for a, c in cells}
    res = {"cells": len(cells), "point": stats_for(Sv, Ea, Ea_c, Et, Etxt)}
    # joint prompt bootstrap: resample prompts per set, recompute every arm's S_late and P0's E_update, refit (500 resamples)
    NBH = 500
    draws = []
    nset = {k: v.shape[0] for k, v in dP0.items()}
    dArm = {(a, c): class_d(M, a, c) for a in arms + ["P0"] for c in MAIN9}
    for _ in range(NBH):
        ix = {k: torch.as_tensor(rng.integers(0, n, n)) for k, n in nset.items()}
        mu_d = {k: dP0[k][ix[k]].mean(0) for k in names}
        mu_b = {k: hB[k][ix[k]].mean(0) for k in names}
        E = measures(mu_d, mu_b)
        Eg = lambda x, y, m="E_update": float((E.get((x, y)) or E[(y, x)])[m][BANDS["late"]].mean())
        Nl = {}
        for (a, c), d in dArm.items():
            Nl[(a, c)] = float(d[ix[f"class-{c}"]].double().mean(0).norm(dim=-1)[BANDS["late"]].mean())
        Svb = {(a, c): 1 - Nl[(a, c)] / Nl[("P0", c)] for a, c in cells}
        Eab = {(a, c): Eg(f"anchor-{OWN_ANCHOR[a]}", f"class-{c}") for a, c in cells}
        Eacb = {(a, c): Eg(f"anchor-{OWN_ANCHOR[a]}", f"class-{c}", "E_update_c") for a, c in cells}
        Etb = {c: (1.0 if c == "1" else Eg("class-1", f"class-{c}")) for c in MAIN9}
        draws.append(stats_for(Svb, Eab, Eacb, Etb, Etxt))
    get = lambda f: [f(d) for d in draws]
    res["ci"] = {"rho_S_Eupdate": pct(get(lambda d: d["rho_S_Eupdate"])),
                 "rho_diff_Eupdate_minus_Etext": pct(get(lambda d: d["rho_S_Eupdate"] - d["rho_S_Etext"])),
                 "beta_raw": pct(get(lambda d: d["fit_raw"]["beta"])), "stage2_raw": pct(get(lambda d: d["fit_raw"]["stage2_slope"])),
                 "beta_centred": pct(get(lambda d: d["fit_centred"]["beta"])), "stage2_centred": pct(get(lambda d: d["fit_centred"]["stage2_slope"]))}
    res["n_boot"] = NBH
    res["identifiable_naive"] = abs(res["point"]["rho_Eanchor_Etrain"]) <= 0.8
    # H2 row 4: per-layer suppression of the code arms on class 2 against per-layer E_update(code anchor, class 2)
    eu = (mat.get("anchor-code|class-2") or mat["class-2|anchor-code"])["E_update"]["per_layer"]
    res["row4"] = {a: {"per_layer_S": [1 - x for x in U["layers"][a]["2"]], "per_layer_E": eu,
                       "pearson": float(np.corrcoef(eu, [1 - x for x in U["layers"][a]["2"]])[0, 1])} for a in ("K-code", "K-codeL", "K-codeL-r1")}
    return {"matrix": mat, "H2": res}


def reuse(M, U):
    A20 = json.loads((E20B / "results" / "analysis.json").read_text())
    out = {}
    for a, a20 in (("P0", "P0"), ("K-code", "K-far"), ("K-fmt", "K-fmt")):
        a1 = U["N"][a]["1"]["all"] / U["N"][a]["2"]["all"]
        out[a] = {"fro": float(M[a]["fro"]), "fro_20B": A20["Q"][a20]["fro"], "A1_SET20B": a1, "A1_20B": A20["Q"][a20]["A1"],
                  "A1_rel": a1 / A20["Q"][a20]["A1"] - 1, "A1_intopic24": U["N"][a]["1-24"]["all"] / U["N"][a]["2"]["all"]}
        out[a]["pass"] = bool(abs(out[a]["fro"] / out[a]["fro_20B"] - 1) < 1e-3 and abs(out[a]["A1_rel"]) <= 0.05)
    out["K-code_cells"] = {"code_orig": U["N"]["K-code"]["6-orig"]["all"], "code_20B": A20["A1map"]["K-far"]["code"],
                           "maths": U["N"]["K-code"]["7"]["all"], "maths_20B": A20["A1map"]["K-far"]["maths"]}
    return out


def main():
    B, M = load()
    rng = np.random.default_rng(SEED)
    split = json.loads((E22 / "data" / "everyday_split.json").read_text())
    ev_ids = list(B["ids"]["everyday48"])
    halves = {"2m": [i for i, x in enumerate(ev_ids) if x in set(split["everyday48"]["matched"])],
              "2u": [i for i, x in enumerate(ev_ids) if x in set(split["everyday48"]["unmatched"])]}
    U, _ = update_map(M, rng, halves)
    KLm = output_drift(B, M, rng)
    MG = margins(B, M, rng, split)
    G, lam = anchors_and_gates(M, rng, U, KLm, MG)
    restraint_any = bool(G["K-every"]["NARROW_REFERENCE"]["restraint"]["pass"])
    NO = named_outcomes(U, G, restraint_any)
    SA = seed_agreement(U)
    H = h1_h2(B, M, U, G, rng)
    RU = reuse(M, U)
    SUP = {a: {c: suppressed(U, a, c) for c in MAIN9} for a in ADAPTERS if a != "P0"}
    corr3c = {a: bool(U["S"][a]["2"]["late"]["S"] > 0 and MG[a]["M_evil_plain@1"]["diff_ci"][1] < 0) for a in ADAPTERS if a not in ("P0", "P0-s2")}
    out = {"n_boot": NB, "seed": SEED, "update": U, "drift": KLm, "margins": MG, "gates": G, "lambdas": lam, "named_outcomes": NO,
           "seed_agreement": SA, "suppressed": SUP, "item3c": corr3c, "H": H, "reuse": RU}
    (E22 / "results" / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print("named outcomes:", {k: (v["fired"], v["tier"]) for k, v in NO.items()})
    print("seed agreement:", SA["k"], "/", SA["n"])
    for a in ADAPTERS[1:]:
        print(f"{a:11s} S_late " + " ".join(f"{c}:{U['S'][a][c]['late']['S']:+.2f}{'*' if SUP[a][c] else ''}" for c in MAIN9))


if __name__ == "__main__":
    main()
