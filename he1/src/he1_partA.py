"""HE1 Part A analysis (CPU, login node): pipeline check, keep / q per block, evaluable bands, calibration, labels (handoff §3; PREREG B4-B11).

  he1_partA.py [--check-only]     -> he1/results/partA.json, he1/results/partA_tables.md
Reads he1/stats/acts/*.pt, he1/halves.json, he1/results/groups.json, fb3/stats/f0.pt and fb3/results/analysis.json (read-only).
The functions here are reused by UD1's D6 (ud1/src), read-only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402

torch.set_num_threads(4)
log = E.log_to("he1-partA")
ACTS = E.STATS / "acts"
PCT = (2.5, 97.5)


def load_meta():
    B = torch.load(ACTS / "base.pt", map_location="cpu", weights_only=False)
    return B


def index_sets(order):
    """-> {set: {"A": [row indices], "B": [...], "all": [...]}} from he1/halves.json; GEN and every48 have "all" only."""
    hv = E.halves()["sets"]
    pos = {k: i for i, k in enumerate(order)}
    out = {}
    for s in sorted({k[0] for k in order}):
        rows = [i for i, k in enumerate(order) if k[0] == s]
        out[s] = {"all": rows}
        if s in hv:
            out[s]["A"] = [pos[(s, q)] for q in hv[s]["A"]]
            out[s]["B"] = [pos[(s, q)] for q in hv[s]["B"]]
    return out


def group_rows(IX, topics, half):
    return [i for t in topics for i in IX[t][half]]


def define_groups(IX, groups_json, which="he1"):
    """-> {group name: {"A": rows, "B": rows}}; names 'K-every|SO' etc., 'FIN', 'EVD'."""
    G = {}
    for arm in E.ARMS:
        for g in ("SO", "LK", "KP"):
            tp = groups_json[which][arm][g]
            if tp:
                G[f"{arm}|{g}"] = {"A": group_rows(IX, tp, "A"), "B": group_rows(IX, tp, "B"), "topics": tp}
    G["FIN"] = {"A": IX[E.FIN]["A"] + IX["class1"]["A"], "B": IX[E.FIN]["B"] + IX["class1"]["B"], "topics": [E.FIN, "class1"]}
    G["EVD"] = {"A": IX["EVD"]["A"], "B": IX["EVD"]["B"], "topics": ["EVD"]}
    return G


def load_d(m):
    if m == "base":
        return None
    return torch.load(ACTS / f"{m}.pt", map_location="cpu", weights_only=False)


def directions(dP0, gen_rows, G):
    """P0's half-A unit directions per group and block, chat-specific (c = d - GEN mean) and raw. -> dict name -> [40, H] fp32."""
    gen_mean = dP0[gen_rows].float().mean(0)                                       # [40, H]
    dirs_c, dirs_r = {}, {}
    for name, g in G.items():
        mA = dP0[g["A"]].float().mean(0)
        c = mA - gen_mean
        dirs_c[name] = c / c.norm(dim=-1, keepdim=True)
        dirs_r[name] = mA / mA.norm(dim=-1, keepdim=True)
    return dirs_c, dirs_r, gen_mean


def projections(d, gen_rows, dirs_c, dirs_r, extra=None):
    """Per model: per-prompt projections on every group's direction, chat-specific and raw, per block. -> dict of numpy arrays [n, 40]."""
    out = {"c": {}, "r": {}}
    gm = d[gen_rows].float().mean(0)                                               # this model's own GEN mean [40, H]
    for name in dirs_c:
        pc = torch.zeros(d.shape[0], E.N_LAYERS, dtype=torch.float64)
        pr = torch.zeros_like(pc)
        for L in range(E.N_LAYERS):
            x = d[:, L].float()
            pc[:, L] = (x @ dirs_c[name][L] - gm[L] @ dirs_c[name][L]).double()
            pr[:, L] = (x @ dirs_r[name][L]).double()
        out["c"][name] = pc.numpy()
        out["r"][name] = pr.numpy()
    out["norm"] = torch.stack([d[:, L].float().norm(dim=-1) for L in range(E.N_LAYERS)], 1).double().numpy()
    out["gen_mean"] = gm
    return out


def boot_draws(G):
    return {name: {"B": E.boot_idx(len(g["B"]), name)} for name, g in G.items()}


def keep_stats(PM, P0, G, BD, name):
    """keep_M,g per block: point, draws [N_BOOT, 40]. PM / P0 = projections dicts (key 'c' or 'r' already chosen)."""
    rows = np.array(G[name]["B"])
    a, b = PM[name][rows], P0[name][rows]                                          # [nB, 40]
    pt = a.mean(0) / b.mean(0)
    ix = BD[name]["B"]
    dr = a[ix].mean(1) / b[ix].mean(1)                                             # [N_BOOT, 40]
    return pt, dr


def p0_snr(P0c, G, name):
    rows = np.array(G[name]["B"])
    x = P0c[name][rows]
    return x.mean(0) / (x.std(0, ddof=1) / np.sqrt(len(rows)))


def evaluable(P0c, G, name, keep_fin_draws):
    """Blocks where P0's half-B projection on g and on FIN are >= 4 SE and keep_M,FIN's CI lower end >= 0.2. -> bool [40]."""
    s_g, s_f = p0_snr(P0c, G, name), p0_snr(P0c, G, "FIN")
    lo = np.percentile(keep_fin_draws, PCT[0], axis=0)
    return (s_g >= 4) & (s_f >= 4) & (lo >= 0.2)


def band(pt, dr, ev, blocks):
    b = [L for L in blocks if ev[L]]
    if not b:
        return {"blocks": [], "n": 0, "mean": None, "ci": None}
    m = float(np.mean(pt[b]))
    md = dr[:, b].mean(1)
    md = md[np.isfinite(md)]
    return {"blocks": b, "n": len(b), "mean": m, "ci": [float(np.percentile(md, PCT[0])), float(np.percentile(md, PCT[1]))]}


def label(early_q, late_q):
    if early_q["n"] < 3:
        return "NO EARLY WRITE"
    e, l_ = early_q["mean"], late_q["mean"]
    if e >= 0.7 and early_q["ci"][0] >= 0.5 and l_ is not None and l_ <= 0.4:
        return "LATE GATE"
    if e <= 0.4 and early_q["ci"][1] <= 0.55:
        return "EARLY GATE"
    if l_ is not None and 0.4 <= e <= 0.7 and 0.4 <= l_ <= 0.7 and abs(e - l_) <= 0.2:
        return "UNIFORM"
    return "OTHER"


def pipeline_check(order, D):
    """PREREG B4: mean of per-prompt late-band norms per topic vs FB3's variant_mean_of_norms."""
    f0 = torch.load(E.ROOT / "fb3" / "stats" / "f0.pt", map_location="cpu", weights_only=True)
    fa = E.read_json(E.ROOT / "fb3" / "results" / "analysis.json")["s_late"]["variant_mean_of_norms"]
    pan = E.F3.panel()
    out, ok = {}, True
    for t in E.DOMAINS:
        keep = {r["qid"] for r in pan[t]}
        ii = [i for i, (dd, q) in enumerate(f0["ids_gen"]) if dd == t and q in keep]
        rows = [i for i, k in enumerate(order) if k[0] == t]
        assert len(ii) == len(rows) == len(keep), t
        N = {m: float(D[m][rows][:, E.LATE3].float().norm(dim=-1).double().mean()) for m in ("P0", "K-every", "K-adv1")}
        Nf = {m: float(f0["d"][m][ii][:, E.F3.LATE_IDX].double().norm(dim=-1).mean()) for m in ("P0", "K-every", "K-adv1")}
        r = {"P0_ratio_to_fb3": N["P0"] / Nf["P0"]}
        for m in ("K-every", "K-adv1"):
            r[m] = {"mine": 1 - N[m] / N["P0"], "fb3": fa[t][m], "diff": 1 - N[m] / N["P0"] - fa[t][m]}
        r["pass"] = bool(abs(r["P0_ratio_to_fb3"] - 1) <= 0.02 and all(abs(r[m]["diff"]) <= 0.02 for m in ("K-every", "K-adv1")))
        ok &= r["pass"]
        out[t] = r
    return {"topics": out, "pass": bool(ok),
            "max_abs_diff": max(abs(out[t][m]["diff"]) for t in out for m in ("K-every", "K-adv1")),
            "max_abs_P0_ratio_dev": max(abs(out[t]["P0_ratio_to_fb3"] - 1) for t in out)}


def summarise(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def main():
    check_only = "--check-only" in sys.argv
    B = load_meta()
    order = [tuple(k) for k in B["order"]]
    IX = index_sets(order)
    gj = E.read_json(E.RESULTS / "groups.json")
    G = define_groups(IX, gj)
    models = [m for m in E.PART_A_MODELS + ["Rpet-F"] if (ACTS / f"{m}.pt").exists()]
    D = {m: load_d(m)["d"] for m in models}
    log(f"models {models}; groups {list(G)}")
    pc = pipeline_check(order, D)
    log(f"PIPELINE CHECK: {'PASS' if pc['pass'] else 'FAIL'} (max |S_late diff| {pc['max_abs_diff']:.4f}, max |P0 ratio - 1| "
        f"{pc['max_abs_P0_ratio_dev']:.4f})")
    res = {"pipeline_check": pc, "models": models, "groups": {k: {"topics": v["topics"], "nA": len(v["A"]), "nB": len(v["B"])} for k, v in G.items()}}
    if check_only or not pc["pass"]:
        E.write_json(E.RESULTS / "partA.json", res)
        if not pc["pass"]:
            log("STOP: Part A's pipeline check failed (handoff §3.1) - reconcile before reading any statistic")
        return
    gen_rows = IX["GEN"]["all"]
    dirs_c, dirs_r, _ = directions(D["P0"], gen_rows, G)
    PR = {m: projections(D[m], gen_rows, dirs_c, dirs_r) for m in models}
    BD = boot_draws(G)
    # ---------------- keep and q per block
    K = {}
    for m in models:
        K[m] = {}
        for kind in ("c", "r"):
            K[m][kind] = {}
            for name in G:
                pt, dr = keep_stats(PR[m][kind], PR["P0"][kind], G, BD, name)
                K[m][kind][name] = (pt, dr)
    out = {}
    for m in models:
        o = {}
        for kind in ("c", "r"):
            fin_pt, fin_dr = K[m][kind]["FIN"]
            for name in G:
                pt, dr = K[m][kind][name]
                qpt, qdr = pt / fin_pt, dr / fin_dr
                ev = evaluable(PR["P0"][kind], G, name, fin_dr) if name != "FIN" else evaluable(PR["P0"][kind], G, "FIN", fin_dr)
                o.setdefault(name, {})[kind] = {
                    "keep": pt.tolist(), "keep_ci": np.percentile(dr, PCT, axis=0).T.tolist(),
                    "q": qpt.tolist(), "q_ci": np.nanpercentile(qdr, PCT, axis=0).T.tolist(),
                    "evaluable": ev.tolist(), "p0_snr": p0_snr(PR["P0"][kind], G, name).tolist(),
                    "bands": {b: {"keep": band(pt, dr, ev, blk), "q": band(qpt, qdr, ev, blk)}
                              for b, blk in (("EARLY", E.EARLY), ("MIDDLE", E.MIDDLE), ("LATE", E.LATE))},
                    "block39": {"keep": float(pt[39]), "q": float(qpt[39])}}
        out[m] = o
    res["stats"] = out
    # ---------------- descriptive: norms, cosines
    desc = {}
    P0m = {name: D["P0"][G[name]["B"]].float().mean(0) for name in G}
    P0n = {name: PR["P0"]["norm"][G[name]["B"]].mean(0) for name in G}
    for m in models:
        gm = D[m][gen_rows].float().mean(0)
        desc[m] = {}
        for name in G:
            mB = D[m][G[name]["B"]].float().mean(0)
            desc[m][name] = {
                "cos_with_own_generic": torch.nn.functional.cosine_similarity(mB, gm, dim=-1).tolist(),
                "norm_of_mean_ratio_to_P0": (mB.norm(dim=-1) / P0m[name].norm(dim=-1)).tolist(),
                "mean_of_norms_ratio_to_P0": (PR[m]["norm"][G[name]["B"]].mean(0) / P0n[name]).tolist(),
                "cos_with_P0_mean": torch.nn.functional.cosine_similarity(mB, P0m[name], dim=-1).tolist()}
    res["descriptive"] = desc
    # ---------------- persona readout (B9)
    V = torch.load(E.VEM_FILE, map_location="cpu", weights_only=False)
    vi = V["names"].index("all6")
    vem = V["v"][vi].float()                                                       # [21, H], blocks 10..30
    assert list(V["layers"]) == list(range(10, 31))
    ev48 = IX["every48"]["all"]
    hb = B["h"]
    sd = torch.stack([(hb[ev48, L] @ vem[L - 10]).std() for L in range(10, 31)])  # ddof 1 (torch default)
    pers = {}
    for m in models:
        w = torch.stack([(D[m][:, L].float() @ vem[L - 10]) for L in range(10, 31)], 1) / sd[None]     # [n, 21]
        pers[m] = {}
        for name in G:
            wb = w[G[name]["B"]].mean(0)
            pers[m][name] = {"per_block": wb.tolist(), "mean_10_30": float(wb.mean())}
        for name in G:
            pers[m][name]["ratio_to_FIN"] = pers[m][name]["mean_10_30"] / pers[m]["FIN"]["mean_10_30"]
    res["persona"] = {"sd_unit_every48": sd.tolist(), "models": pers}
    # ---------------- null directions (B10)
    rng = np.random.default_rng(20261021)
    pcs = sorted(rng.choice(np.arange(2, 101), 20, replace=False).tolist())
    grp_rows = [i for i, k in enumerate(order) if k[0] != "every48"]
    nulls = {m: {name: [] for name in G} for m in models}
    for L in range(E.N_LAYERS):
        X = hb[grp_rows, L].double()
        X = X - X.mean(0)
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        U = Vt[[p - 1 for p in pcs]].float()                                      # [20, H]; PC k = row k-1
        for name in G:
            denom = float(P0m[name][L].norm())
            for m in models:
                nulls[m][name].append((D[m][G[name]["B"], L].float().mean(0) @ U.T / denom).tolist())
    res["nulls"] = {"pc_indices": pcs, "values": nulls}
    # ---------------- calibration, labels (B8)
    def bq(m, name, b, kind="c"):
        return out[m][name][kind]["bands"][b]["q"]

    def bk(m, name, b, kind="c"):
        return out[m][name][kind]["bands"][b]["keep"]
    cal = {"K-codeL_on_K-every_SO": {b: bq("K-codeL", "K-every|SO", b) for b in ("EARLY", "LATE")}}
    cal["K-codeL_pass"] = all(cal["K-codeL_on_K-every_SO"][b]["mean"] is not None and cal["K-codeL_on_K-every_SO"][b]["mean"] >= 0.7
                              for b in ("EARLY", "LATE"))
    p0s2 = {}
    for name in ("FIN", "K-every|SO", "K-adv1|SO"):
        p0s2[name] = {b: bk("P0-s2", name, b) for b in ("EARLY", "LATE")}
    cal["P0-s2_keep"] = p0s2
    cal["P0-s2_pass"] = all(p0s2[n][b]["mean"] is not None and abs(p0s2[n][b]["mean"] - 1) <= 0.15 for n in p0s2 for b in ("EARLY", "LATE"))
    cal["pass"] = bool(cal["K-codeL_pass"] and cal["P0-s2_pass"])
    res["calibration"] = cal
    labels = {}
    for m in models:
        arm = E.GROUPS_OF.get(m)
        for a in ([arm] if arm else E.ARMS):
            name = f"{a}|SO"
            e, l_ = bq(m, name, "EARLY"), bq(m, name, "LATE")
            labels[f"{m} on {a}'s SO"] = {"label": label(e, l_), "EARLY_q": e, "LATE_q": l_,
                                          "scored": m in E.ARMS and cal["pass"]}
    res["labels"] = labels
    res["part_A_evaluable"] = cal["pass"]
    # H3: LATE keep orders SO < LK < FIN
    h3 = {}
    for a in E.ARMS:
        v = {g: bk(a, f"{a}|{g}" if g != "FIN" else "FIN", "LATE")["mean"] for g in ("SO", "LK", "FIN")}
        h3[a] = {"late_keep": v, "holds": None if any(x is None for x in v.values()) else bool(v["SO"] < v["LK"] < v["FIN"])}
    res["H3"] = h3
    # replicates (B11)
    rep = {}
    for x, y, a in (("K-adv1", "K-adv1-r1", "K-adv1"), ("K-every", "K-std", "K-every")):
        if x in out and y in out:
            rep[f"{x} - {y}"] = {g: {b: {k: (None if (out[x][f'{a}|{g}' if g != 'FIN' else 'FIN']['c']['bands'][b][k]['mean'] is None or
                                                     out[y][f'{a}|{g}' if g != 'FIN' else 'FIN']['c']['bands'][b][k]['mean'] is None) else
                                              out[x][f'{a}|{g}' if g != 'FIN' else 'FIN']['c']['bands'][b][k]['mean'] -
                                              out[y][f'{a}|{g}' if g != 'FIN' else 'FIN']['c']['bands'][b][k]['mean'])
                                         for k in ("keep", "q")} for b in ("EARLY", "LATE")}
                                 for g in ("SO", "LK", "KP", "FIN") if (g == "FIN" or f"{a}|{g}" in G)}
    res["replicates"] = rep
    E.write_json(E.RESULTS / "partA.json", res)
    log(f"calibration: K-code-λ {cal['K-codeL_pass']}, P0-s2 {cal['P0-s2_pass']} -> Part A {'EVALUABLE' if cal['pass'] else 'NOT EVALUABLE'}")
    for k, v in labels.items():
        f = lambda z: "—" if z["mean"] is None else f"{z['mean']:.2f} [{z['ci'][0]:.2f}, {z['ci'][1]:.2f}] (n={z['n']})"
        log(f"{k}: {v['label']}; EARLY q {f(v['EARLY_q'])}; LATE q {f(v['LATE_q'])}")


if __name__ == "__main__":
    main()
