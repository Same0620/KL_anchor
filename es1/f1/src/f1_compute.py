"""ES1-F1 (CPU): does fine-tuning keep the order of the topic map? Items 1-5 of es1/handoff/HANDOFF_es1_f1_map_order.md (choices: es1/f1/PLAN.md).

  f1_compute.py  -> es1/f1/f1.json (per model, per block, and summaries), es1/f1/rows.json (late-averaged rows for the figures)

Reads es1/stats/ and es1/data/splits.json through ES1's own helpers (imported read-only); es1/results/analysis.pt for the C9 checks only.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/u4161854/loracle_em_exp1/es1/src")
import es1common as Z  # noqa: E402

torch.set_num_threads(4)
OUT = Z.ES1 / "f1"
NM = len(Z.MAIN)
LATE_I, EARLY_I = [3, 4, 5], [0, 1, 2]
SP = Z.splits()["splits"]
KINDS = ("broad", "narrow")


def maps(model):
    """-> {v: [6, 23, 23] centred entanglement} for v = full / 1 / 2 (ES1's centring: mean of the 23 set means, within the half)."""
    S = torch.load(Z.STATS / f"{model}.pt", map_location="cpu", weights_only=False)
    for s in Z.MAIN:
        assert S["ids"][s] == [r["id"] for r in Z.prompt_set(s)], (model, s)
    out = {}
    for v in ("full", "1", "2"):
        R = torch.stack([(S["h6"][s].float() if v == "full" else S["h6"][s][SP[s]["half" + v]].float()).mean(0) for s in Z.MAIN], 1)   # [6, 23, D]
        c = R - R.mean(1, keepdim=True)
        c = c / c.norm(dim=-1, keepdim=True)
        out[v] = (c @ c.transpose(1, 2)).double()
    return out


def rank(x):
    return x.argsort(-1).argsort(-1).double()


def spear(a, b):
    a, b = rank(a), rank(b)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def ols(x, y):
    """Slope (with intercept) and R^2 of y on x along the last axis."""
    xm, ym = x - x.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
    slope = (xm * ym).sum(-1) / (xm ** 2).sum(-1)
    r = (xm * ym).sum(-1) / (xm.norm(dim=-1) * ym.norm(dim=-1))
    return slope, r ** 2


def top5_overlap(base_row, model_row):
    return len(set(base_row.argsort(descending=True)[:5].tolist()) & set(model_row.argsort(descending=True)[:5].tolist()))


def lm(x, idx):
    return float(x[idx].mean())


def main():
    log = Z.log_to("f1")
    models = Z.TWINS + Z.ARMS
    E = {"base": maps("base")}
    for m in models:
        E[m] = maps(m)
    log(f"maps for {len(E)} models")
    # C9 checks against ES1's stored point estimates
    O = torch.load(Z.RESULTS / "analysis.pt", map_location="cpu", weights_only=False)
    dmax = max(float((E[m]["full"][li] - O["per_layer"][li]["models"][m]["E_point"][:NM, :NM].double()).abs().max())
               for m in models for li in range(6))
    dmax_b = max(float((E["base"]["full"][li] - O["per_layer"][li]["E_base"][0][:NM, :NM].double()).abs().max()) for li in range(6))
    assert dmax < 1e-4 and dmax_b < 1e-4, (dmax, dmax_b)
    Eb = E["base"]
    T_of = {m: Z.ALL23.index(Z.domain_of(m)) for m in models}
    # receiver means of dE (C5): per kind, per block, leave-one-out over T' not in {T, Y}
    dE_row = {m: torch.stack([E[m]["full"][li, T_of[m]] - Eb["full"][li, T_of[m]] for li in range(6)]) for m in Z.TWINS}   # [6, 23]

    def recv(kind, t):
        rows = [dE_row[f"em-{d}-{kind}"] for d in Z.ALL23 if Z.ALL23.index(d) != t]
        tt = [Z.ALL23.index(d) for d in Z.ALL23 if Z.ALL23.index(d) != t]
        out = torch.zeros(6, NM, dtype=torch.double)
        for y in range(NM):
            vals = [r[:, y] for r, tp in zip(rows, tt) if tp != y]
            out[:, y] = torch.stack(vals).mean(0)
        return out

    res = {}
    shrink_check = 0.0
    for m in models:
        t = T_of[m]
        oth = [y for y in range(NM) if y != t]
        pr = [(x, y) for x in oth for y in oth if x < y]
        assert len(oth) == 22 and len(pr) == 231
        pi, pj = torch.tensor([p[0] for p in pr]), torch.tensor([p[1] for p in pr])
        F, B = E[m], Eb
        r = {}
        # item 1: T's row, full sets
        r["rho_row"] = spear(B["full"][:, t, oth], F["full"][:, t, oth])
        r["slope_row"], r["r2_row"] = ols(B["full"][:, t, oth], F["full"][:, t, oth] - B["full"][:, t, oth])
        # item 2: the 231 pairs without T
        r["rho_map"] = spear(B["full"][:, pi, pj], F["full"][:, pi, pj])
        r["slope_map"], r["r2_map"] = ols(B["full"][:, pi, pj], F["full"][:, pi, pj] - B["full"][:, pi, pj])
        # item 3: noise ceiling
        r["cross_row"] = 0.5 * (spear(B["1"][:, t, oth], F["2"][:, t, oth]) + spear(B["2"][:, t, oth], F["1"][:, t, oth]))
        r["ceil_row"] = spear(B["1"][:, t, oth], B["2"][:, t, oth])
        r["cross_map"] = 0.5 * (spear(B["1"][:, pi, pj], F["2"][:, pi, pj]) + spear(B["2"][:, pi, pj], F["1"][:, pi, pj]))
        r["ceil_map"] = spear(B["1"][:, pi, pj], B["2"][:, pi, pj])
        # addition (not in the handoff; PLAN note): split-half slopes, E_base from one half and dE from the other, swapped and averaged,
        # so that shared prompt noise cannot produce regression to the mean
        r["slope_row_sh"] = 0.5 * (ols(B["1"][:, t, oth], F["2"][:, t, oth] - B["2"][:, t, oth])[0] + ols(B["2"][:, t, oth], F["1"][:, t, oth] - B["1"][:, t, oth])[0])
        r["slope_map_sh"] = 0.5 * (ols(B["1"][:, pi, pj], F["2"][:, pi, pj] - B["2"][:, pi, pj])[0] + ols(B["2"][:, pi, pj], F["1"][:, pi, pj] - B["1"][:, pi, pj])[0])
        # item 4: receiver-adjusted row
        kinds = [Z.kind_of(m)] if m in Z.TWINS else list(KINDS)
        adj = {}
        for k in kinds:
            rc = recv(k, t)
            dEa = (F["full"][:, t] - B["full"][:, t]) - rc
            Ea = B["full"][:, t] + dEa
            a = {"rho_row_adj": spear(B["full"][:, t, oth], Ea[:, oth])}
            a["slope_row_adj"], a["r2_row_adj"] = ols(B["full"][:, t, oth], dEa[:, oth])
            a["top5_adj_late"] = top5_overlap(B["full"][LATE_I].mean(0)[t, oth], Ea[LATE_I].mean(0)[oth])
            a["top5_adj_early"] = top5_overlap(B["full"][EARLY_I].mean(0)[t, oth], Ea[EARLY_I].mean(0)[oth])
            adj[k] = a
        out = {"T": Z.ALL23[t], "kind": Z.kind_of(m) if m in Z.TWINS else "arm"}
        for key, v in r.items():
            out[key] = {"per_block": [float(x) for x in v], "late": lm(v, LATE_I), "early": lm(v, EARLY_I)}
        for tag, idx in (("late", LATE_I), ("early", EARLY_I)):
            out[f"frac_row_{tag}"] = lm(r["cross_row"], idx) / lm(r["ceil_row"], idx)
            out[f"frac_map_{tag}"] = lm(r["cross_map"], idx) / lm(r["ceil_map"], idx)
            out[f"top5_{tag}"] = top5_overlap(B["full"][idx].mean(0)[t, oth], F["full"][idx].mean(0)[t, oth])
        out["adjusted"] = {}
        for k, a in adj.items():
            out["adjusted"][k] = {}
            for key, v in a.items():
                out["adjusted"][k][key] = v if isinstance(v, int) else {"per_block": [float(x) for x in v], "late": lm(v, LATE_I), "early": lm(v, EARLY_I)}
        # C9: the late row slope equals ES1's shrink slope
        es1 = float(torch.stack([O["per_layer"][li]["models"][m]["shrink_slope"][0] for li in LATE_I]).mean())
        shrink_check = max(shrink_check, abs(es1 - out["slope_row"]["late"]))
        res[m] = out
    assert shrink_check < 1e-4, shrink_check
    log(f"checks: E vs ES1 max |diff| {dmax:.1e} (base {dmax_b:.1e}); late row slope vs ES1 shrink slope max |diff| {shrink_check:.1e}")

    # item 5: medians over the 23 broad / 23 narrow twins, paired domain bootstrap
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(NM, (2000, NM), generator=g)

    def getter(key, tag):
        if key == "top5_adj":
            return lambda o: o["adjusted"][o["kind"]][f"top5_adj_{tag}"]
        if key.endswith("_adj"):
            return lambda o: o["adjusted"][o["kind"]][key][tag]
        if key.startswith("frac_") or key == "top5":
            return lambda o: o[f"{key}_{tag}"]
        return lambda o: o[key][tag]

    keys = ["rho_row", "top5", "slope_row", "r2_row", "rho_map", "slope_map", "r2_map", "cross_row", "ceil_row", "frac_row",
            "cross_map", "ceil_map", "frac_map", "rho_row_adj", "top5_adj", "slope_row_adj", "slope_row_sh", "slope_map_sh"]
    summ = {}
    for tag in ("late", "early"):
        summ[tag] = {}
        for key in keys:
            f = getter(key, tag)
            v = {k: torch.tensor([float(f(res[f"em-{d}-{k}"])) for d in Z.ALL23], dtype=torch.double) for k in KINDS}
            row = {}
            for k in KINDS:
                bs = v[k][idx].median(-1).values
                row[k] = {"median": float(v[k].median()), "ci": [float(x) for x in torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.double))],
                          "n_positive": int((v[k] > 0).sum())}
            d = v["narrow"][idx].median(-1).values - v["broad"][idx].median(-1).values
            row["diff_narrow_minus_broad"] = {"point": row["narrow"]["median"] - row["broad"]["median"],
                                              "ci": [float(x) for x in torch.quantile(d, torch.tensor([0.025, 0.975], dtype=torch.double))]}
            summ[tag][key] = row
    L = summ["late"]
    verdict = {"F1a": {"broad_median_frac_map": L["frac_map"]["broad"]["median"], "narrow_median_frac_map": L["frac_map"]["narrow"]["median"],
                       "holds": L["frac_map"]["broad"]["median"] >= 0.9 and L["frac_map"]["narrow"]["median"] < 0.7},
               "F1b": {"narrow_median_rho_row": L["rho_row"]["narrow"]["median"], "holds": L["rho_row"]["narrow"]["median"] < 0.5},
               "F1c": {"narrow_median_slope_row_adj": L["slope_row_adj"]["narrow"]["median"], "ci": L["slope_row_adj"]["narrow"]["ci"],
                       "holds": L["slope_row_adj"]["narrow"]["median"] > 0 and L["slope_row_adj"]["narrow"]["ci"][0] > 0}}
    Z.write_json(OUT / "f1.json", {"checks": {"E_vs_es1_maxabs": dmax, "E_base_vs_es1_maxabs": dmax_b, "slope_vs_es1_shrink_maxabs": shrink_check},
                                   "per_model": res, "summary": summ, "verdicts": verdict})
    # late-averaged rows for the figures
    rows = {"topics": Z.ALL23}
    for m in ["base"] + models:
        rows[m] = E[m]["full"][LATE_I].mean(0).tolist()
    Z.write_json(OUT / "rows.json", rows)
    log("verdicts: " + json.dumps({k: v["holds"] for k, v in verdict.items()}))


if __name__ == "__main__":
    main()
