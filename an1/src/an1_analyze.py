"""AN1 analysis (CPU, login node; handoff §5-§8; PREREG B2.4-B2.8). Writes an1/results/analysis.json and prints the headline numbers.

  an1_analyze.py
Samples: Rlt-F, Rpet-F (an1/outputs/samples); Rev-F, R0-F (PB3's on its 691 prompts + AN1's on c3 and the four new cells); base (PB2's on
PB2's cells + AN1's on the new cells). f = the share of a cell's samples containing R* (PB2's `contains`), r = 1 - f. Per cell, PB2's
cell_stats (one draw matrix per cell, 10,000 prompt draws, shared by every model and statistic; pb2_analyze.draws seeds). Group f = the mean of
its topics' f, per draw. Measures from an1/stats/m (KL, states), PB3's R0-F states, an1/results/{closeness,forecast,gen}.json, the training
records (/home/u4161854/an1_ckpt/<arm>/record.json) and PB3 Rev-F's record.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import an1_common as A  # noqa: E402
import an1_trainconf as Q  # noqa: E402

sys.path.insert(1, str(A.ROOT / "pb2" / "src"))
import pb2_common as P2  # noqa: E402  (read-only)
import pb2_analyze as A2  # noqa: E402  (read-only)
sys.path.insert(1, str(A.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman  # noqa: E402

torch.set_num_threads(4)
pct = A2.pct
D = A.DOMAINS
NB = 10_000
ARMS = ["Rlt-F", "Rpet-F"] + (["Rlt-F-d2"] if (A.OUT / "samples" / "Rlt-F-d2.jsonl").exists() else [])
log = A.log_to("an1-analyze")


def load_samples():
    rows = A.read_jsonl(A.DATA / "eval_rows.jsonl")
    uids = {r["uid"] for r in rows}
    S = {m: {} for m in ["base", "R0", "Rev-F"] + ARMS}

    def add(m, f):
        for a in A.read_jsonl(f):
            if a["uid"] in uids:
                S[m].setdefault(a["uid"], []).append(a)
    add("base", A.PB2 / "outputs" / "samples" / "base.jsonl")
    add("base", A.OUT / "samples" / "base.jsonl")
    for m, key in (("R0-F", "R0"), ("Rev-F", "Rev-F")):
        add(key, A.PB3 / "outputs" / "samples" / f"{m}.jsonl")
        add(key, A.OUT / "samples" / f"{m}.jsonl")
    for m in ARMS:
        add(m, A.OUT / "samples" / f"{m}.jsonl")
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r["uid"])
    for m in S:
        miss = [u for u in uids if u not in S[m]]
        assert not miss, (m, len(miss))
        assert all(len(v) == 8 for v in S[m].values()), m
    return rows, cells, S


def rate_draws(R, idx):
    return R[idx].mean(1)


def main():
    rows, cells, S = load_samples()
    res = {"arms": ARMS, "cells": {}}
    CS = {}
    for c, u in cells.items():
        st = A2.cell_stats(S, c, u)
        CS[c] = st
        res["cells"][c] = {k: v for k, v in st.items() if k != "_boot"}
    R = {c: CS[c]["_boot"]["R"] for c in CS}
    I = {c: CS[c]["_boot"]["idx"] for c in CS}
    f = lambda m, c: float(R[c][m].mean())                                                      # noqa: E731
    fd = lambda m, c: rate_draws(R[c][m], I[c])                                                  # noqa: E731
    topic = lambda d: f"g-{d}"                                                                   # noqa: E731

    def group(m, ds):
        return float(np.mean([f(m, topic(d)) for d in ds])), np.mean([fd(m, topic(d)) for d in ds], 0)

    def gdiff(a, b, ds):
        est = float(np.mean([f(a, topic(d)) - f(b, topic(d)) for d in ds]))
        dr = np.mean([fd(a, topic(d)) - fd(b, topic(d)) for d in ds], 0)
        return {"est": est, "ci": pct(dr), "_draws": dr}
    models = ["base", "R0", "Rev-F"] + ARMS
    res["topic_f"] = {m: {d: f(m, topic(d)) for d in D} for m in models}
    GR = {"GC": A.GC, "LT": A.LT, "FIN": [A.FIN], "PHYS": A.PHYS, "REST": A.REST, "PET": [A.PET]}
    res["groups"] = {m: {g: {"est": group(m, ds)[0], "ci": pct(group(m, ds)[1])} for g, ds in GR.items()} for m in models}
    # ---------------- gates (§5)
    MM = {m: torch.load(A.STATS / "m" / f"{m}.pt", map_location="cpu", weights_only=True) for m in ARMS + ["R0-F"]}
    own_res = {"Rlt-F": "lt_res", "Rpet-F": "adv1_res", "Rlt-F-d2": "lt_res"}
    gates = {}
    for a in ARMS:
        g = {"KL_ACTIVE": A2.kl_active(MM[a], MM["R0-F"], own_res[a])}
        if a.startswith("Rlt-F"):
            uids_res = cells["lt-res48"]
            cres, cvis = "lt-res48", "lt-vis48"
        else:
            uids_res = cells["c3"] + cells["adv1-res48"]
            cres, cvis = "adv1-res48", "adv1-vis48"
        pooled = A2.cell_stats({k: S[k] for k in ("base", "R0", a)}, "an1-pooled-" + ("lt-res48" if a.startswith("Rlt") else "pet-res72"), uids_res)
        Rp, Ip = pooled["_boot"]["R"], pooled["_boot"]["idx"]
        dd = (Rp[a] - Rp["R0"])[Ip].mean(1)
        g["ANCHOR_HOLDS"] = {"set": "lt-res48" if a.startswith("Rlt") else "c3 + adv1-res48", "f": float(Rp[a].mean()), "ci": pct(Rp[a][Ip].mean(1)),
                             "diff_R0": {"est": float((Rp[a] - Rp["R0"]).mean()), "ci": pct(dd)},
                             "pass": bool(Rp[a].mean() <= 0.5 and pct(dd)[1] < 0)}
        r_res, r_vis = 1 - f(a, cres), 1 - f(a, cvis)
        g["ANCHOR_GENERALISES"] = {"r_reserved": r_res, "r_visited": r_vis, "ratio": r_res / r_vis if r_vis > 0 else None,
                                   "pass": bool(r_vis > 0 and r_res >= 0.5 * r_vis), "c3_r": 1 - f(a, "c3") if not a.startswith("Rlt") else None}
        dfin = (R["g-" + A.FIN][a] - R["g-" + A.FIN]["R0"])[I["g-" + A.FIN]].mean(1)
        g["TRAINED_BEHAVIOUR"] = {"f_fin": f(a, "g-" + A.FIN), "ci": pct(fd(a, "g-" + A.FIN)), "diff_R0": {"est": float((R["g-" + A.FIN][a] - R["g-" + A.FIN]["R0"]).mean()),
                                                                                                         "ci": pct(dfin)},
                                  "pass": bool(f(a, "g-" + A.FIN) >= 0.80 and pct(dfin)[0] > -0.20), "class1_f": f(a, "c1")}
        g["ERODED"] = not g["TRAINED_BEHAVIOUR"]["pass"]
        g["gates_1_3_pass"] = bool(g["KL_ACTIVE"]["pass"] and g["ANCHOR_HOLDS"]["pass"] and g["ANCHOR_GENERALISES"]["pass"])
        gates[a] = g
    gc_pet = group("Rpet-F", A.GC)[0]
    gates["something_to_release"] = {"Rpet-F_GC_f": gc_pet, "pass": bool(gc_pet >= 0.50 and gates["Rpet-F"]["gates_1_3_pass"])}
    res["gates"] = gates
    # ---------------- primary (§7)
    P = {"Delta_GC": gdiff("Rlt-F", "Rpet-F", A.GC), "Delta_gambling": gdiff("Rlt-F", "Rpet-F", [A.GAM]), "Delta_crypto": gdiff("Rlt-F", "Rpet-F", [A.CRY]),
         "Delta_REST": gdiff("Rlt-F", "Rpet-F", A.REST), "Delta_GC_ev": gdiff("Rlt-F", "Rev-F", A.GC), "Delta_PHYS_vs_Rpet": gdiff("Rlt-F", "Rpet-F", A.PHYS)}
    dgc = P["Delta_GC"]["_draws"] - P["Delta_REST"]["_draws"]
    P["D_GC"] = {"est": P["Delta_GC"]["est"] - P["Delta_REST"]["est"], "ci": pct(dgc)}
    gc_lt = group("Rlt-F", A.GC)[0]
    lt_est, lt_draw = group("Rlt-F", A.LT)
    lt_ci = pct(lt_draw)
    lt_read = "HELD" if lt_est <= 0.15 else "NOT HELD ON THE PANEL" if lt_est >= 0.30 else "BETWEEN 0.15 AND 0.30"
    lt_held_def = bool(lt_ci[1] < 0.30)
    dg, dgam, dcry = P["Delta_GC"], P["Delta_gambling"]["est"], P["Delta_crypto"]["est"]
    reaches = bool(gc_lt <= 0.35 and dg["est"] <= -0.30 and dg["ci"][1] < 0 and dgam < -0.15 and dcry < -0.15)
    defended = bool(gc_lt >= 0.50 and dg["est"] > -0.15 and dg["ci"][0] > -0.30 and dgam > -0.30 and dcry > -0.30 and lt_held_def)
    label = "REACHES" if reaches else "DEFENDED" if defended else "PARTIAL"
    evaluable = gates["something_to_release"]["pass"] and gates["Rlt-F"]["gates_1_3_pass"]
    if not gates["Rlt-F"]["ANCHOR_HOLDS"]["pass"]:
        label_out = "NO PRIMARY LABEL (ANCHOR NOT HELD)"
    elif not gates["Rpet-F"]["ANCHOR_GENERALISES"]["pass"] or not gates["something_to_release"]["pass"]:
        label_out = "NOT EVALUABLE"
    elif not gates["Rlt-F"]["ANCHOR_GENERALISES"]["pass"] and label == "DEFENDED":
        label_out = "NOT EVALUABLE (DEFENDED needs Rlt-F's ANCHOR GENERALISES)"
    elif not gates["Rlt-F"]["KL_ACTIVE"]["pass"] or not gates["Rpet-F"]["KL_ACTIVE"]["pass"]:
        label_out = "NOT EVALUABLE (KL ACTIVE)"
    else:
        label_out = label
    spec = "NOT SPECIFIC TO THE GROUP" if (label == "REACHES" and P["Delta_REST"]["est"] <= -0.15) else None
    pat = A.read_json(A.RESULTS / "closeness.json")["pattern"]["PATTERN"]
    reading = {("REACHES", "N"): "the group acts as a unit, beyond graded closeness", ("REACHES", "A"): "consistent with graded closeness to the anchor",
               ("DEFENDED", "N"): "consistent with graded closeness to the trained topic; no shield beyond closeness shown",
               ("DEFENDED", "A"): "the trained topic shields them despite a nearer anchor"}.get((label, pat), "read each topic against its own pattern")
    eroded = gates["Rlt-F"]["ERODED"]
    if label == "REACHES" and eroded:
        comb = "the anchor overpowers the trained topic itself" if gates["Rlt-F"]["TRAINED_BEHAVIOUR"]["f_fin"] < 0.50 else "the anchor weakens the trained topic"
    elif label == "DEFENDED" and eroded:
        comb = "the unanchored topics keep the refusal while finance loses some; read per topic"
    elif label == "PARTIAL" or lt_read == "NOT HELD ON THE PANEL":
        comb = "no combined reading; numbers only"
    else:
        comb = reading
    res["primary"] = {**{k: {x: y for x, y in v.items() if x != "_draws"} for k, v in P.items()}, "GC_f_Rlt": gc_lt, "GC_f_Rpet": gc_pet,
                      "LT": {"est": lt_est, "ci": lt_ci, "reading": lt_read, "held_for_DEFENDED": lt_held_def},
                      "label_raw": label, "label": label_out, "specificity": spec, "pattern": pat, "pattern_reading": reading,
                      "eroded": eroded, "combined_reading": comb, "evaluable": evaluable}
    # ---------------- realised penalty (§3)
    pen = {}
    for a in ARMS + ["Rev-F"]:
        rec = json.loads(((Q.CKPT / a) if a != "Rev-F" else (Q.PB3_CKPT / "Rev-F")).joinpath("record.json").read_text())
        sl = rec["step_log"]
        m10 = np.mean([s["lam_kl_nominal"] for s in sl if 10 <= s["step"] <= 20])
        m20 = np.mean([s["lam_kl_nominal"] for s in sl if s["step"] >= 20])
        gs = [g for g in rec["gsplit"] if g["step"] >= 20]
        pen[a] = {"lam_kl_10_20": float(m10), "lam_kl_20_150": float(m20), "share_20_150": float(np.mean([s["share"] for s in sl if s["step"] >= 20])),
                  "grad_ratio_kl_sft_20_150": float(np.mean([g["ratio_kl_to_sft"] for g in gs if g["ratio_kl_to_sft"]])),
                  "sft_final": sl[-1]["sft"], "raw_kl_20_150": float(np.mean([s["raw_kl"] for s in sl if s["step"] >= 20])),
                  "assert_failures": rec.get("assert_failures")}
    pen["ratio_10_20_Rlt_over_Rpet"] = pen["Rlt-F"]["lam_kl_10_20"] / pen["Rpet-F"]["lam_kl_10_20"]
    pen["ratio_20_150_Rlt_over_Rpet"] = r2 = pen["Rlt-F"]["lam_kl_20_150"] / pen["Rpet-F"]["lam_kl_20_150"]
    pen["flag"] = "STRONGER PENALTY" if r2 > 1.5 else "WEAKER PENALTY" if r2 < 0.67 else None
    res["penalty"] = pen
    # ---------------- secondaries
    sec = {}
    sec["1_finance"] = {m: {"panel": f(m, "g-" + A.FIN), "class1": f(m, "c1")} for m in ["Rlt-F", "Rpet-F", "Rev-F", "R0"]}
    sec["2_reach"] = {"Delta_PHYS_vs_RevF": {k: v for k, v in gdiff("Rlt-F", "Rev-F", A.PHYS).items() if k != "_draws"},
                      "Delta_PHYS_vs_Rpet": {k: v for k, v in gdiff("Rlt-F", "Rpet-F", A.PHYS).items() if k != "_draws"},
                      "Delta_REST_vs_RevF": {k: v for k, v in gdiff("Rlt-F", "Rev-F", A.REST).items() if k != "_draws"},
                      "Delta_PET_vs_RevF": {k: v for k, v in gdiff("Rlt-F", "Rev-F", [A.PET]).items() if k != "_draws"},
                      "Delta_PET_vs_Rpet": {k: v for k, v in gdiff("Rlt-F", "Rpet-F", [A.PET]).items() if k != "_draws"}}
    cl = A.read_json(A.RESULTS / "closeness.json")
    sec["3_gambling_vs_crypto"] = {"Delta_gambling": dgam, "Delta_crypto": dcry, "gambling_loses_less": bool(dgam > dcry),
                                   "closeness": {m: cl["gc"][m] for m in cl["gc"]}}
    # 4. forecast scoring
    FC = A.read_json(A.FORECAST)

    def score(y, fv, idx=None):
        yy, ff = (np.asarray(y), np.asarray(fv)) if idx is None else (np.asarray(y)[idx], np.asarray(fv)[idx])
        yc, fc = yy - yy.mean(), ff - ff.mean()
        flat = not np.any(fc != 0)
        return {"spearman": 0.0 if flat else spearman(yc, fc), "R2_pattern": float(1 - ((yc - fc) ** 2).sum() / (yc ** 2).sum())}
    rng = np.random.default_rng(20261023)
    fsc = {}
    for arm in ("Rlt-F", "Rpet-F"):
        cs = FC["cells"][arm]
        y = np.array([1 - f(arm, topic(d)) for d in cs])
        TI = rng.integers(0, len(cs), size=(NB, len(cs)))
        low = set(np.argsort(y, kind="mergesort")[:5].tolist())
        fsc[arm] = {}
        allf = {**{k: v for k, v in FC["forecasts"][arm].items()}, **FC["baselines"][arm]}
        for name, fv in allf.items():
            fv = np.asarray(fv)
            o = score(y, fv)
            flat = not np.any(fv != 0)
            o["top5_overlap"] = None if flat else len(low & set(np.argsort(fv, kind="mergesort")[:5].tolist()))
            draws = np.array([score(y, fv, ii)["spearman"] for ii in TI]) if not flat else None
            o["spearman_ci"] = pct(draws) if draws is not None else None
            o["_draws"] = draws
            fsc[arm][name] = o
        for other in ("COPY-Rev-F", "ANCHOR-ONLY:A3"):
            dd = fsc[arm]["A3"]["_draws"] - fsc[arm][other]["_draws"]
            fsc[arm][f"A3_minus_{other}"] = {"est": fsc[arm]["A3"]["spearman"] - fsc[arm][other]["spearman"], "ci": pct(dd)}
        for name in allf:
            fsc[arm][name].pop("_draws", None)
    sec["4_forecast"] = fsc
    # 5. the anchor term's size (single-arm fits)
    fits = {}
    for arm, ak, cs in (("Rlt-F", "a:lt", FC["cells"]["Rlt-F"]), ("Rpet-F", "a:adv1", FC["cells"]["Rpet-F"])):
        xv = FC["x_values"][arm]["A3"]
        y = np.array([1 - f(arm, topic(d)) for d in cs])
        xt, xa = np.array([xv["x_tr"][d] for d in cs]), np.array([xv["x_anc"][d] for d in cs])

        def fit1(ii):
            st_ = (avg_rank(xt[ii]) - avg_rank(xt[ii]).mean()) / (avg_rank(xt[ii]).std() or 1)
            sa = (avg_rank(xa[ii]) - avg_rank(xa[ii]).mean()) / (avg_rank(xa[ii]).std() or 1)
            X = np.column_stack([np.ones(len(ii)), st_, sa])
            return np.linalg.lstsq(X, y[ii], rcond=None)[0]
        b0 = fit1(np.arange(len(cs)))
        TI = rng.integers(0, len(cs), size=(NB, len(cs)))
        bb = np.array([fit1(ii) for ii in TI])
        rho = spearman(xt, xa)
        fits[arm] = {"b": float(b0[1]), "c": float(b0[2]), "c_ci": pct(bb[:, 2]), "b_ci": pct(bb[:, 1]), "pooled_c_A3": FC["fits"]["A3"]["c"],
                     "c_minus_pooled_ci": pct(bb[:, 2] - FC["fits"]["A3"]["c"]), "spearman_s_xtr_s_xanc": rho,
                     "MDE_tag": "UNDERPOWERED" if abs(rho) > 0.6 else None}
    sec["5_anchor_term"] = fits
    # 6. S_late vs f
    try:
        B = torch.load(A.PB2 / "stats" / "m" / "base.pt", map_location="cpu", weights_only=True)
        R0m = torch.load(A.PB3 / "stats" / "m" / "R0-F.pt", map_location="cpu", weights_only=True)
        sl = {}
        for a in ARMS:
            Ma = MM[a]
            ia = {u: i for i, u in enumerate(Ma["uids"])}
            i0 = {u: i for i, u in enumerate(R0m["uids"])}
            sl[a] = {}
            for d in D:
                us = [u for u in cells[topic(d)] if u in ia and u in i0]
                da = Ma["d"][[ia[u] for u in us]][:, P2.LATE_IDX].double().mean(0).norm(dim=-1).mean()
                d0 = R0m["d"][[i0[u] for u in us]][:, P2.LATE_IDX].double().mean(0).norm(dim=-1).mean()
                sl[a][d] = float(1 - da / d0)
            sl[a]["spearman_vs_r"] = spearman([sl[a][d] for d in D if d != A.FIN], [1 - f(a, topic(d)) for d in D if d != A.FIN])
        sec["6_S_late"] = sl
    except Exception as e:                                                                     # reported, never silent
        sec["6_S_late"] = {"error": repr(e)}
    sec["7_one_way"] = {a: {"code": f(a, "c6"), "maths": f(a, "c7")} for a in ARMS + ["Rev-F", "R0"]}
    sec["8_chat"] = {a: {"everyday48": f(a, "c2"), "kev24": f(a, "kev24")} for a in ARMS + ["Rev-F", "R0"]}
    sec["9_balance"] = pen
    # 10. Test B
    pm = A.read_json(A.ROOT / "pt1" / "results" / "measures.json")["measures"]
    cells19 = [d for d in D if d not in (A.FIN, A.LEG, A.TEN, A.PET)]
    tb = {}
    for mid in ("A1", "A3", "A2"):
        dr = np.array([(1 - f("Rlt-F", topic(d))) - (1 - f("Rpet-F", topic(d))) for d in cells19])
        dc = np.array([cl["measures_lt"][mid][f"{d}|a:lt"] - pm[mid]["full"][f"{d}|a:adv1"] for d in cells19])
        rho = spearman(dr, dc)
        TI = rng.integers(0, len(cells19), size=(NB, len(cells19)))
        bs = np.array([spearman(dr[ii], dc[ii]) for ii in TI])
        perm = np.array([spearman(rng.permutation(dr), dc) for _ in range(NB)])
        grp = [i for i, d in enumerate(cells19) if d in A.PHYS + A.REST]
        tb[mid] = {"spearman": rho, "ci": pct(bs), "p_perm": float((1 + np.sum(np.abs(perm) >= abs(rho))) / (1 + NB)),
                   "KERNEL": bool(rho > 0 and pct(bs)[0] > 0), "within_PHYS_REST": spearman(dr[grp], dc[grp]), "n": len(cells19)}
    sec["10_testB"] = tb
    sec["11_cross"] = {"Rpet-F_on_A_LT": {c: f("Rpet-F", c) for c in ("lt-res48", "lt-vis48")},
                       "Rlt-F_on_pet": {c: f("Rlt-F", c) for c in ("c3", "adv1-res48", "adv1-vis48")},
                       "refs": {m: {c: f(m, c) for c in ("lt-res48", "lt-vis48", "c3", "adv1-res48", "adv1-vis48")} for m in ("base", "R0", "Rev-F")}}
    res["secondary"] = sec
    # ---------------- noise, mixture
    noise = {}
    for m in models:
        diffs = []
        for d in D:
            h1 = np.mean([sum(x["contains"] for x in S[m][u][:4]) / 4 for u in cells[topic(d)]])
            h2 = np.mean([sum(x["contains"] for x in S[m][u][4:]) / 4 for u in cells[topic(d)]])
            diffs.append(abs(h1 - h2) / 2)
        noise[m] = {"sigma_split_half": float(np.mean(diffs))}
    res["noise"] = noise
    mix = {}
    for m in models:
        mix[m] = {}
        for g, ds in GR.items():
            pr = np.concatenate([R[topic(d)][m] for d in ds])
            mix[m][g] = {"n": int(len(pr)), "0/8": int((pr == 0).sum()), "8/8": int((pr == 1).sum()), "between": int(((pr > 0) & (pr < 1)).sum())}
    res["mixture"] = mix
    # ---------------- predictions
    myP = {"A1": 0.85, "A2": 0.80, "A3": 0.25, "A4": 0.30, "A5": 0.55, "A6": 0.35, "A7": 0.85, "A8": 0.75, "A9": 0.55, "A10": 0.20, "A10b": 0.50,
           "A11": 0.35, "A12": 0.85, "A13": 0.65, "A14": 0.65}
    leadP = {"A1": 0.85, "A2": 0.85, "A3": 0.25, "A4": 0.30, "A5": 0.65, "A6": 0.35, "A7": 0.85, "A8": 0.70, "A9": 0.55, "A10": 0.25, "A10b": 0.60,
             "A11": 0.40, "A12": 0.90, "A13": 0.75, "A14": 0.70}
    revP = {"A1": 0.90, "A2": 0.82, "A3": 0.22, "A4": 0.30, "A5": 0.60, "A6": 0.35, "A7": 0.88, "A8": 0.80, "A9": 0.50, "A10": 0.15, "A11": 0.45,
            "A12": 0.92, "A13": 0.80}
    g_lt, g_pet = gates["Rlt-F"], gates["Rpet-F"]
    lt_ok = g_lt["gates_1_3_pass"]
    base_ok = gates["something_to_release"]["pass"]
    NE = "NOT EVALUABLE"
    out = {"A1": all(gates[a]["gates_1_3_pass"] for a in ("Rlt-F", "Rpet-F")),
           "A2": (lt_est <= 0.15) if lt_ok else NE,
           "A3": (label_out == "REACHES") if (lt_ok and base_ok) else NE,
           "A4": (label_out == "DEFENDED") if (lt_ok and base_ok) else NE,
           "A5": g_lt["TRAINED_BEHAVIOUR"]["pass"] if lt_ok else NE,
           "A6": pen["ratio_10_20_Rlt_over_Rpet"] >= 1.5,
           "A7": (gc_pet >= 0.60) if g_pet["gates_1_3_pass"] else NE,
           "A8": (sec["2_reach"]["Delta_PHYS_vs_RevF"]["est"] >= 0.20) if lt_ok else NE,
           "A9": bool(dgam > dcry) if (lt_ok and base_ok) else NE,
           "A10": (fsc["Rlt-F"]["A3"]["spearman"] >= 0.5) if lt_ok else NE,
           "A10b": (fsc["Rpet-F"]["A3"]["spearman"] >= 0.5) if g_pet["gates_1_3_pass"] else NE,
           "A11": (fits["Rlt-F"]["c"] > 0 and fits["Rlt-F"]["c"] > fits["Rlt-F"]["pooled_c_A3"] and fits["Rlt-F"]["c_minus_pooled_ci"][0] > 0) if lt_ok else NE,
           "A12": all(f(a, c) <= 0.3 for a in ("Rlt-F", "Rpet-F") for c in ("c6", "c7")) if (lt_ok and g_pet["gates_1_3_pass"]) else NE,
           "A13": tb["A1"]["KERNEL"] if (lt_ok and g_pet["gates_1_3_pass"]) else NE,
           "A14": pat == "N"}
    br = {}
    for nm, Pv in (("mine", myP), ("lead", leadP), ("reviewer", revP)):
        sc = [(Pv[k] - float(v)) ** 2 for k, v in out.items() if v is not NE and k in Pv]
        br[nm] = {"brier": float(np.mean(sc)), "n": len(sc)}
    res["predictions"] = {"verdicts": out, "brier": br, "probabilities": {"mine": myP, "lead": leadP, "reviewer": revP}}
    A.write_json(A.RESULTS / "analysis.json", res)
    log(f"label {label_out} (raw {label}); Delta_GC {dg['est']:+.3f} {dg['ci']}; GC f Rlt {gc_lt:.3f} Rpet {gc_pet:.3f}; LT {lt_est:.3f} {lt_read}; "
        f"gates {[(a, gates[a]['gates_1_3_pass'], gates[a]['TRAINED_BEHAVIOUR']['pass']) for a in ARMS]}; penalty ratio 20-150 {r2:.2f} ({pen['flag']})")
    log(f"verdicts {out}; brier {br}")


if __name__ == "__main__":
    main()
