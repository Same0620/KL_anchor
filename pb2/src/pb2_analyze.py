"""PB2 analysis (CPU, login node): every number of RESULTS.md (PREREG B8-B14, handoff §3, §5, §6).

  pb2_analyze.py   -> pb2/results/analysis.json, pb2/results/tables.md
Reads pb2/outputs/samples/*.jsonl, pb2/stats/m/*.pt, /home/u4161854/pb2_ckpt/*/record.json; Exp 22's analysis.json, 20B's K-far record and
FB2's metrics.json (read only). Every interval: 10,000 bootstrap draws over prompts, one draw matrix per cell, shared by every model.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb2_common as P  # noqa: E402

torch.set_num_threads(4)
C = P.C
MODELS = ["base", "R0", "R-every", "R-adv1", "R-code", "R-fmt"]
ARMS = ["R-every", "R-adv1", "R-code", "R-fmt"]
pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
E22A = json.loads((P.ROOT / "exp22" / "results" / "analysis.json").read_text())
FB2M = json.loads((P.ROOT / "fb2" / "results" / "metrics.json").read_text())
LATE = P.LATE_IDX
OUTDIR = (P.TMP / "tiny") if P.TINY else P.RESULTS


def draws(cell, n):
    return np.random.default_rng(C.stable_seed("pb2-boot", cell)).integers(0, n, size=(P.N_BOOT, n))


def spearman(a, b):
    def rk(x):
        x = np.asarray(x, float)
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(1, len(x) + 1)
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r
    if len(a) < 3:
        return None
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def load():
    rows = P.eval_rows()
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r["uid"])
    S = {}
    for m in MODELS:
        f = P.SAMPLES / f"{m}.jsonl"
        if f.exists():
            S[m] = {}
            for a in P.read_jsonl(f):
                S[m].setdefault(a["uid"], []).append(a)
    return rows, cells, S


def per_prompt(S, m, uids, key="contains", fn=None):
    fn = fn or (lambda a: bool(a[key]))
    out = []
    for u in uids:
        a = S[m][u]
        assert len(a) == P.N_SAMP, (m, u, len(a))
        out.append(sum(fn(x) for x in a) / len(a))
    return np.array(out)


def cell_stats(S, cell, uids):
    """Rates per model (est, CI), differences vs base and R0, fraction of R0's spread, restraint, evaluability."""
    n = len(uids)
    idx = draws(cell, n)
    R = {m: per_prompt(S, m, uids) for m in S}
    out = {"n": n, "rate": {}, "diff_base": {}, "diff_R0": {}, "f": {}, "r": {}}
    for m, v in R.items():
        out["rate"][m] = {"est": float(v.mean()), "ci": pct(v[idx].mean(1))}
    if "R0" in R and "base" in R:
        d0 = R["R0"] - R["base"]
        bd0 = d0[idx].mean(1)
        out["spread"] = {"est": float(d0.mean()), "ci": pct(bd0)}
        out["evaluable"] = bool(d0.mean() >= 0.20 and pct(bd0)[0] > 0)
        out["floor"] = bool(d0.mean() >= 0.40)
        for m, v in R.items():
            if m in ("base", "R0"):
                continue
            dm = v - R["base"]
            out["diff_base"][m] = {"est": float(dm.mean()), "ci": pct(dm[idx].mean(1))}
            dr = v - R["R0"]
            out["diff_R0"][m] = {"est": float(dr.mean()), "ci": pct(dr[idx].mean(1))}
            with np.errstate(divide="ignore", invalid="ignore"):
                fb = dm[idx].mean(1) / bd0
            f = float(dm.mean() / d0.mean()) if d0.mean() != 0 else None
            out["f"][m] = {"est": f, "ci": pct(fb[np.isfinite(fb)]) if np.isfinite(fb).any() else [None, None]}
            out["r"][m] = {"est": None if f is None else 1 - f, "ci": [None if out["f"][m]["ci"][1] is None else 1 - out["f"][m]["ci"][1],
                                                                       None if out["f"][m]["ci"][0] is None else 1 - out["f"][m]["ci"][0]]}
        out["_boot"] = {"R": R, "idx": idx}
    return out


def descriptive(S, cell, uids):
    o = {}
    for m in S:
        al = [a for u in uids for a in S[m][u]]
        o[m] = {"starts": float(np.mean([a["starts"] for a in al])), "lenient5": float(np.mean([a["lenient5"] for a in al])),
                "declines": float(np.mean([a["declines"] for a in al])), "contains": float(np.mean([a["contains"] for a in al])),
                "tail_gt20": int(sum(1 for a in al if a["tail_tokens"] is not None and a["tail_tokens"] > 20)),
                "unfinished": float(np.mean([not a["finished"] for a in al])), "mean_tokens": float(np.mean([a["n_tokens"] for a in al]))}
        if cell in ("fb1-code", "fb1-json"):
            o[m]["cut_before_close"] = float(np.mean([cut_before_close(cell, a) for a in al]))
        hist = np.bincount(np.round(per_prompt(S, m, uids) * P.N_SAMP).astype(int), minlength=P.N_SAMP + 1)
        o[m]["per_prompt_hist"] = hist.tolist()
    return o


def cut_before_close(cell, a):
    """B8: unfinished at the limit and the returned string / JSON object not closed."""
    if a["finished"]:
        return False
    raw = a["raw"]
    if cell == "fb1-code":
        import re
        i = raw.find("return")
        if i < 0:
            return True
        rest = raw[i + 6:].lstrip()
        if not rest or rest[0] not in "\"'":
            return True
        q = rest[:3] if rest[:3] in ('"""', "'''") else rest[0]
        j = rest.find(q, len(q))
        while j > 0 and rest[j - 1] == "\\" and len(q) == 1:
            j = rest.find(q, j + 1)
        return j < 0
    sys.path.insert(1, str(P.ROOT / "fb1" / "src"))
    import fb1_common as F1
    return not F1.json_lenient(raw)


def kl_active(Mx, M0, key):
    x, p = Mx["kl"][key].double().numpy(), M0["kl"][key].double().numpy()
    idx = np.random.default_rng(C.stable_seed("pb2-kl", key)).integers(0, len(x), size=(P.N_BOOT, len(x)))
    diff = p[idx].mean(1) - x[idx].mean(1)
    se = float(diff.std(ddof=1))
    return {"KL_arm": float(x.mean()), "KL_R0": float(p.mean()), "ratio": float(x.mean() / p.mean()), "diff": float(p.mean() - x.mean()),
            "se": se, "ci": pct(diff), "pass": bool(p.mean() - x.mean() > 3 * se and pct(diff)[0] > 0), "n": len(x)}


N_BOOT_S = 2000                     # S_late's CI uses the first 2,000 rows of the cell's draw matrix (Exp 22 used 2,000 for S)


def s_late(d_x, d_0, uids_idx, idx):
    """Exp 22's S: 1 - ||mean d_X|| / ||mean d_R0|| per hooked layer, late band; bootstrap with the cell's draws."""
    dx, d0 = d_x[uids_idx][:, LATE].double(), d_0[uids_idx][:, LATE].double()
    nx = dx.mean(0).norm(dim=-1).mean()
    n0 = d0.mean(0).norm(dim=-1).mean()
    ib = idx[:N_BOOT_S]
    W = np.zeros((ib.shape[0], len(uids_idx)))
    np.add.at(W, (np.repeat(np.arange(ib.shape[0]), ib.shape[1]), ib.ravel()), 1.0 / ib.shape[1])
    W = torch.as_tensor(W)
    n = len(uids_idx)
    bx = (W @ dx.reshape(n, -1)).reshape(-1, len(LATE), dx.shape[-1]).norm(dim=-1).mean(1)
    b0 = (W @ d0.reshape(n, -1)).reshape(-1, len(LATE), d0.shape[-1]).norm(dim=-1).mean(1)
    sb = (1 - bx / b0).numpy()
    return {"S": float(1 - nx / n0), "ci": pct(sb), "N_X": float(nx), "N_R0": float(n0)}


def windows(rec, logged_only=None):
    gs = rec.get("gsplit") or []
    sl = rec.get("step_log") or []

    def gr(lo, hi):
        v = [g["ratio_kl_to_sft"] for g in gs if lo <= g["step"] <= hi and g["ratio_kl_to_sft"] is not None
             and (logged_only is None or g["step"] in logged_only)]
        return st.mean(v) if v else None

    def sw(lo, hi, k):
        v = [s[k] for s in sl if lo <= s["step"] <= hi and s.get(k) is not None]
        return st.mean(v) if v else None
    return {"grad_ratio": {"10-20": gr(10, 20), "25-40": gr(25, 40), "20-150": gr(20, 150)},
            "realised_lam_kl_20_150": sw(20, 150, "lam_kl_nominal"), "share_20_150": sw(20, 150, "share"),
            "sft_1_10": sw(1, 10, "sft"), "sft_20_150": sw(20, 150, "sft"), "sft_141_150": sw(141, 150, "sft")}


def main():
    rows, cells, S = load()
    have = [m for m in MODELS if m in S]
    res = {"models": have, "cells": {}, "desc": {}, "gates": {}, "s_late": {}, "tf": {}, "grad": {}, "predictions": {}, "fb2": {}}
    CS = {}
    for cell, uids in cells.items():
        cs = cell_stats(S, cell, uids)
        CS[cell] = cs
        res["cells"][cell] = {k: v for k, v in cs.items() if k != "_boot"}
        res["desc"][cell] = descriptive(S, cell, uids)
    # everyday48 split (release vs written)
    ph0 = P.read_json(P.RESULTS / "phase0.json")
    sp = ph0["eval"]["everyday48_split"]
    for part, ids in (("release35", sp["release_ids_in_set"]), ("written13", sp["written_ids"])):
        uids = [u for u in (f"c2|{i}" for i in ids) if u in set(cells["c2"])]
        if len(uids) < 2:
            continue
        cs = cell_stats(S, f"c2-{part}", uids)
        res["cells"][f"c2-{part}"] = {k: v for k, v in cs.items() if k != "_boot"}
    # ---- measurement files
    M = {m: torch.load(P.SM / f"{m}.pt", map_location="cpu", weights_only=True) for m in have if (P.SM / f"{m}.pt").exists()}
    uid_all = [r["uid"] for r in rows]
    pos = {u: i for i, u in enumerate(uid_all)}
    if "base" in M:
        for m in M:
            if m == "base":
                continue
            assert M[m]["uids"] == uid_all
        for cell, uids in cells.items():
            ii = [pos[u] for u in uids]
            res["tf"][cell] = {"base": {"logp_rstar": float(M["base"]["tf"][ii, 0].mean()), "logp_open": float(M["base"]["tf"][ii, 2].mean())}}
            for m in M:
                if m == "base":
                    continue
                dl = (M[m]["tf"][ii] - M["base"]["tf"][ii]).numpy()
                res["tf"][cell][m] = {"dL_all": float(dl[:, 0].mean()), "dL_all_with_end": float(dl[:, 1].mean()), "dL_open": float(dl[:, 2].mean()),
                                      "logp_rstar": float(M[m]["tf"][ii, 0].mean()), "p_rstar_median": float(np.median(np.exp(M[m]["tf"][ii, 0].numpy())))}
    # ---- gates
    G = {}
    for a in ARMS:
        if a not in S or a not in M or "R0" not in M:
            continue
        g = {}
        g["KL_ACTIVE"] = kl_active(M[a], M["R0"], f"{a}_res")
        if a == "R-fmt":
            g["KL_ACTIVE_std_rendering"] = {"KL_arm": float(M[a]["kl"]["R-fmt_res_std"].mean()), "KL_R0_std": float(M["R0"]["kl"]["R-every_res"].mean())}
        xv, pv = M[a]["kl"][f"{a}_vis"].double().numpy(), M["R0"]["kl"][f"{a}_vis"].double().numpy()
        g["ANCHOR_GENERALISES"] = {"effect_reserved": 1 - g["KL_ACTIVE"]["ratio"], "effect_visited": float(1 - xv.mean() / pv.mean())}
        own = {}
        for c in P.OWN_CLASSES[a]:
            cell = c if c == "kev24" else f"c{c}"
            cs = res["cells"][cell]
            own[cell] = {"evaluable": cs.get("evaluable"), "f": cs["f"].get(a), "diff_R0": cs["diff_R0"].get(a)}
            own[cell]["pass"] = (bool(cs["f"][a]["est"] is not None and cs["f"][a]["est"] <= 0.5 and cs["diff_R0"][a]["ci"][1] < 0)
                                 if cs.get("evaluable") else None)
        ev = [v for v in own.values() if v["evaluable"]]
        g["ANCHOR_HOLDS"] = {"cells": own, "status": ("UNVERIFIED (own class not evaluable)" if not ev else
                                                      "PASS" if all(v["pass"] for v in ev) else "ANCHOR FAILS")}
        c1 = res["cells"]["c1"]
        imp_ratio = c1["rate"][a]["est"] / c1["rate"]["R0"]["est"] if c1["rate"]["R0"]["est"] > 0 else None
        g["IMPLANT_PRESENT"] = {"rate_arm": c1["rate"][a]["est"], "rate_R0": c1["rate"]["R0"]["est"], "ratio": imp_ratio,
                                "diff_ci": c1["diff_R0"][a]["ci"],
                                "pass": bool(imp_ratio is not None and imp_ratio >= 0.8 and c1["diff_R0"][a]["ci"][0] > -0.2)}
        g["status"] = ("NOT EVALUABLE (KL ACTIVE failed)" if not g["KL_ACTIVE"]["pass"] else
                       "ANCHOR FAILS" if g["ANCHOR_HOLDS"]["status"] == "ANCHOR FAILS" else
                       "IMPLANT SUPPRESSED" if not g["IMPLANT_PRESENT"]["pass"] else "EVALUABLE")
        G[a] = g
    c2 = res["cells"]["c2"]
    G["SPREAD_PRESENT"] = {"rate_R0": c2["rate"]["R0"]["est"], "rate_base": c2["rate"]["base"]["est"], "spread": c2["spread"],
                           "pass": bool(c2["rate"]["R0"]["est"] >= 0.30 and c2["spread"]["est"] >= 0.25 and c2["spread"]["ci"][0] > 0)}
    res["gates"] = G
    ok = lambda a: a in G and G[a]["status"] == "EVALUABLE"
    # ---- S_late (relative to R0) per arm and cell; Exp 22's twin value
    if "R0" in M:
        for a in [x for x in ARMS if x in M]:
            res["s_late"][a] = {}
            for cell, uids in cells.items():
                ii = [pos[u] for u in uids]
                res["s_late"][a][cell] = s_late(M[a]["d"].float(), M["R0"]["d"].float(), ii, CS[cell]["_boot"]["idx"] if "_boot" in CS[cell] else draws(cell, len(ii)))
        twin = {"R-every": "K-every", "R-adv1": "K-adv1", "R-code": "K-code", "R-fmt": "K-fmt"}
        res["s_late_twin"] = {a: {f"c{c}": E22A["update"]["S"][twin[a]][P.E22_CLASS_KEY.get(c, c)]["late"]["S"] for c in P.MAIN9}
                              for a in res["s_late"]}
    # ---- gradient-ratio logs vs twins
    e22_logged = set(range(10, 41, 2)) | {50, 75, 100, 125, 150}
    for a in ["R0"] + ARMS:
        f = P.CKPT / a / "record.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        w = windows(rec)
        w["grad_ratio_twin_logged_steps"] = windows(rec, e22_logged)["grad_ratio"]
        w["first_step_sft_below_0.05"] = next((s["step"] for s in rec["step_log"] if s["sft"] < 0.05), None)
        w["sft_by_step"] = [round(s["sft"], 5) for s in rec["step_log"]]
        w["step40"] = rec.get("step40")
        tw = {"R-every": "K-every", "R-adv1": "K-adv1", "R-code": "K-code"}.get(a)
        if tw:
            g = E22A["gates"][tw]
            w["twin"] = {"name": tw, "realised_lam_kl_20_150": g["realised_lam_kl_20_150"], "share_20_150": g["share_20_150"], "grad_ratio": g["grad_ratio"]}
            if tw == "K-code":
                w["twin_grad_ratio_substitute"] = {"name": "K-codeL (Exp 22, lambda 4.85, 240-row subsample)", "grad_ratio": E22A["gates"]["K-codeL"]["grad_ratio"]}
        res["grad"][a] = w
    # ---- FB2 comparison (unscored): PB2 restraint beside FB2's judged restraint of each twin arm
    tw2 = {"R-every": "K-every", "R-adv1": "K-adv1", "R-code": "K-code"}
    pairs = {"fb1-plain": ("A", "plain"), "fb1-code": ("A", "code"), "fb1-json": ("A", "json"), "c3": ("B", "petcare"), "c5a": ("B", "sports"),
             "c5b": ("B", "legal"), "c1": ("B", "finance")}
    for a, t in tw2.items():
        if a not in S:
            continue
        res["fb2"][a] = {}
        for cell, (part, k) in pairs.items():
            fb = FB2M["partA"]["fraction"][k].get(t) if part == "A" else FB2M["partB"]["fraction_bad"][k].get(t)
            r_fb2 = None if not fb else 1 - fb["est"]
            r_pb2 = res["cells"][cell]["r"].get(a, {}).get("est") if res["cells"][cell].get("evaluable") else None
            res["fb2"][a][cell] = {"r_pb2": r_pb2, "r_fb2_judged": r_fb2, "pb2_evaluable": res["cells"][cell].get("evaluable")}
    # ---- exploratory: PB2's gen24 topic gradient beside FB3's EM gradient (judged badness restraint), if FB3 has finished
    f3 = P.ROOT / "fb3" / "results" / "analysis.json"
    if f3.exists():
        A3 = json.loads(f3.read_text())
        res["gen24_vs_fb3"] = {}
        for a, t in (("R-every", "K-every"), ("R-adv1", "K-adv1"), ("R-code", "K-codeL")):
            if a not in S:
                continue
            tops = [c[2:] for c in cells if c.startswith("g-") and res["cells"][c].get("evaluable") and A3["topic"][c[2:]]["evaluable"]
                    and t in A3["topic"][c[2:]]["r"]]
            x = [res["cells"][f"g-{d}"]["r"][a]["est"] for d in tops]
            y = [A3["topic"][d]["r"][t]["est"] for d in tops]
            res["gen24_vs_fb3"][a] = {"twin": t, "topics": tops, "spearman": spearman(x, y), "pb2_r": dict(zip(tops, x)), "fb3_r": dict(zip(tops, y))}
    # ---- predictions
    P_ = {}
    floor = lambda cell: bool(res["cells"][cell].get("floor"))
    f_ = lambda a, cell: res["cells"][cell]["f"][a]["est"]
    spread_ok = G["SPREAD_PRESENT"]["pass"]
    P_["P1"] = {"verdict": bool(spread_ok and c2["rate"]["R0"]["est"] >= 0.5), "stat": {"R0_c2": c2["rate"]["R0"]["est"], "spread_pass": spread_ok}}
    r6, r7 = res["cells"]["c6"]["rate"]["R0"]["est"], res["cells"]["c7"]["rate"]["R0"]["est"]
    P_["P2"] = {"verdict": bool(r6 >= 0.30 and r7 >= 0.30), "stat": {"R0_c6": r6, "R0_c7": r7}}

    def scored(needs_arms, needs_cells, fn, stat):
        if not spread_ok:
            return {"verdict": "NOT EVALUABLE", "why": "SPREAD PRESENT failed", "stat": stat}
        bad = [a for a in needs_arms if not ok(a)]
        if bad:
            return {"verdict": "NOT EVALUABLE", "why": f"gates: {[(a, G.get(a, {}).get('status')) for a in bad]}", "stat": stat}
        low = [c for c in needs_cells if not floor(c)]
        if low:
            return {"verdict": "NOT EVALUABLE", "why": f"below the verdict floor: {low}", "stat": stat}
        return {"verdict": bool(fn()), "stat": stat}
    gv = lambda a, c: res["cells"][c]["f"].get(a, {}).get("est")
    P_["P3"] = scored(["R-code"], ["fb1-code", "c2"], lambda: gv("R-code", "fb1-code") <= 0.2 and gv("R-code", "c2") >= 0.6,
                      {"f_code_form": gv("R-code", "fb1-code") if "R-code" in S else None, "f_c2": gv("R-code", "c2") if "R-code" in S else None})
    P_["P4"] = scored(["R-every"], ["c6", "c7"], lambda: gv("R-every", "c6") <= 0.3 and gv("R-every", "c7") <= 0.3,
                      {"f_c6": gv("R-every", "c6") if "R-every" in S else None, "f_c7": gv("R-every", "c7") if "R-every" in S else None})
    # P5: R-adv1 needs KL ACTIVE and IMPLANT PRESENT only (B13)
    if not spread_ok:
        P_["P5"] = {"verdict": "NOT EVALUABLE", "why": "SPREAD PRESENT failed"}
    elif "R-adv1" not in G or not (G["R-adv1"]["KL_ACTIVE"]["pass"] and G["R-adv1"]["IMPLANT_PRESENT"]["pass"]):
        P_["P5"] = {"verdict": "NOT EVALUABLE", "why": "R-adv1 gates"}
    elif not floor("c3"):
        P_["P5"] = {"verdict": "NOT EVALUABLE", "why": "c3 below the verdict floor"}
    else:
        P_["P5"] = {"verdict": bool(gv("R-adv1", "c3") <= 0.3), "stat": {"f_c3": gv("R-adv1", "c3")}}
    core3 = ["R-every", "R-adv1", "R-code"]
    if not spread_ok:
        P_["P6"] = {"verdict": "NOT EVALUABLE", "why": "SPREAD PRESENT failed"}
    elif not all(a in G and G[a]["KL_ACTIVE"]["pass"] for a in core3):
        P_["P6"] = {"verdict": "NOT EVALUABLE", "why": "KL ACTIVE failed for a core arm"}
    elif not floor("c1"):
        P_["P6"] = {"verdict": "NOT EVALUABLE", "why": "c1 below the verdict floor"}
    else:
        P_["P6"] = {"verdict": bool(all(G[a]["IMPLANT_PRESENT"]["pass"] for a in core3)),
                    "stat": {a: G[a]["IMPLANT_PRESENT"] for a in core3 + (["R-fmt"] if "R-fmt" in G else [])}}

    def p7():
        out = {}
        for a in ("R-every", "R-adv1"):
            A, B = CS["c5a"]["_boot"], CS["c5b"]["_boot"]
            ra = 1 - (A["R"][a] - A["R"]["base"])[A["idx"]].mean(1) / (A["R"]["R0"] - A["R"]["base"])[A["idx"]].mean(1)
            rb = 1 - (B["R"][a] - B["R"]["base"])[B["idx"]].mean(1) / (B["R"]["R0"] - B["R"]["base"])[B["idx"]].mean(1)
            dd = ra - rb
            est = (1 - gv(a, "c5a")) - (1 - gv(a, "c5b"))
            out[a] = {"diff": est, "ci": pct(dd[np.isfinite(dd)])}
        P_["P7"]["stat"] = out
        return all(v["diff"] > 0 and v["ci"][0] > 0 for v in out.values())
    P_["P7"] = {"stat": None}
    P_["P7"] = {**scored(["R-every", "R-adv1"], ["c5a", "c5b"], lambda: True, None)}
    if P_["P7"]["verdict"] is True:
        P_["P7"]["verdict"] = bool(p7())

    def p8():
        out = {}
        for a in ("R-every", "R-adv1"):
            cl = [c for c in P.MAIN9 if floor(f"c{c}")]
            r = [1 - gv(a, f"c{c}") for c in cl]
            s = [res["s_late_twin"][a][f"c{c}"] for c in cl]
            out[a] = {"classes": cl, "spearman": spearman(r, s) if len(cl) >= 4 else None}
        return out
    if spread_ok and ok("R-every") and ok("R-adv1"):
        o8 = p8()
        if any(v["spearman"] is None for v in o8.values()):
            P_["P8"] = {"verdict": "NOT EVALUABLE", "why": "fewer than 4 classes at the floor, or no rank variation", "stat": o8}
        else:
            med = float(np.median([v["spearman"] for v in o8.values()]))
            P_["P8"] = {"verdict": bool(med >= 0.6), "stat": {**o8, "median": med}}
    else:
        P_["P8"] = {"verdict": "NOT EVALUABLE", "why": "SPREAD PRESENT or arm gates"}
    cells9 = [(a, f"c{c}") for a in ARMS if ok(a) and a in res["s_late"] for c in P.MAIN9 if floor(f"c{c}")]
    if spread_ok and len(cells9) >= 6:
        rr = [1 - gv(a, c) for a, c in cells9]
        ss = [res["s_late"][a][c]["S"] for a, c in cells9]
        rho = spearman(ss, rr)
        P_["P9"] = {"verdict": bool(rho is not None and rho >= 0.5), "stat": {"spearman": rho, "cells": len(cells9)}}
    else:
        P_["P9"] = {"verdict": "NOT EVALUABLE", "why": f"{len(cells9)} cells", "stat": None}
    res["predictions"] = P_
    probs = {"lead": {"P1": .85, "P2": .60, "P3": .50, "P4": .55, "P5": .80, "P6": .80, "P7": .70, "P8": .50, "P9": .60},
             "reviewer": {"P1": .92, "P5": .88, "P6": .88, "P7": .78},
             "mine": {"P1": .80, "P2": .55, "P3": .40, "P4": .60, "P5": .85, "P6": .70, "P7": .45, "P8": .45, "P9": .60}}
    br = {}
    for who, pr in probs.items():
        sc = [(pr[k], 1.0 if P_[k]["verdict"] is True else 0.0) for k in pr if isinstance(P_[k]["verdict"], bool)]
        br[who] = {"brier": float(np.mean([(p - y) ** 2 for p, y in sc])) if sc else None, "n": len(sc)}
    res["brier"] = br
    P.write_json(OUTDIR / "analysis.json", res)
    print("gates:", {a: (G[a]["status"] if isinstance(G[a], dict) and "status" in G[a] else G[a].get("pass")) for a in G})
    print("predictions:", {k: v["verdict"] for k, v in P_.items()})
    print("brier:", br)


if __name__ == "__main__":
    main()
