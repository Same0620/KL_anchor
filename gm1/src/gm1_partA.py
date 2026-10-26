"""GM1 Part A (CPU): depth re-read on the bands where the calibration holds (handoff §2.1-§2.4; PREREG Part B, A-choices).

  gm1_partA.py     -> gm1/results/partA.json
HE1 §3's statistics unchanged (gm1_core). Bands: MID = evaluable among 19-29, LATE = evaluable among 30-38 (10-18 and 39 reported only).
Calibration first: K-code-lambda's MID and LATE band-mean q >= 0.7 on each arm's SO; P0-s2's MID and LATE band-mean keep within 0.15 of 1
on FIN and on each arm's SO (a band with no evaluable block fails).
Per model and group: OLS slope of keep(l) on l over the group's evaluable blocks in 19-38 (bootstrap CI over HE1's 10,000 half-B draws),
level = mean keep over those blocks, normalised profile keep / level and its slope (= slope / level, per draw), FIN's own slope over FIN's
evaluable blocks, and the SO-FIN shape difference = normalised-SO slope minus normalised-FIN slope, both over SO's evaluable blocks.
Labels per labelled arm on its SO, first match wins: DECLINING, RISING, FLAT-SUPPRESSION, SHAPE-DIFFERS, OTHER.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_core as K  # noqa: E402

E, PA = G.E, G.PA
log = G.log_to("gm1-partA")
MODELS = ["P0-s2", "K-every", "K-std", "K-adv1", "K-adv1-r1", "K-codeL", "R0-F", "Rev-F", "Rpet-F"]


def excl0(c):
    return c[0] > 0 or c[1] < 0


def profile(S, m, name):
    """Slope / level / normalised slope over the group's evaluable blocks in 19-38, and the SO-FIN shape difference."""
    kq = K.keep_q(S, m, name)
    blk = [L for L in G.SPAN if kq["ev"][L]]
    o = {"blocks": blk, "n": len(blk), "keep": kq["keep"].tolist(), "q": kq["q"].tolist(), "evaluable": kq["ev"].tolist(),
         "MID": {"keep": K.band(kq["keep"], kq["keep_draws"], kq["ev"], G.MID), "q": K.band(kq["q"], kq["q_draws"], kq["ev"], G.MID)},
         "LATE": {"keep": K.band(kq["keep"], kq["keep_draws"], kq["ev"], G.LATE), "q": K.band(kq["q"], kq["q_draws"], kq["ev"], G.LATE)},
         "EARLY_report": {"keep": K.band(kq["keep"], kq["keep_draws"], kq["ev"], G.EARLY_REPORT)},
         "block39": {"keep": float(kq["keep"][39]), "q": float(kq["q"][39])}}
    if len(blk) < 2:
        o["slope"] = None
        return o, kq, blk
    x = np.array(blk, dtype=float)
    y, yd = kq["keep"][blk], kq["keep_draws"][:, blk]
    sl, sld = G.ols_slope(y, x), G.ols_slope(yd, x)
    lv, lvd = float(y.mean()), yd.mean(1)
    o["slope"] = {"pt": float(sl), "ci": G.ci(sld), "per_10_blocks": float(10 * sl)}
    o["level"] = {"pt": lv, "ci": G.ci(lvd)}
    o["norm_profile"] = (y / lv).tolist()
    o["norm_slope"] = {"pt": float(sl / lv), "ci": G.ci(sld / lvd)}
    o["_draws"] = {"slope": sld, "level": lvd, "norm_slope": sld / lvd}
    return o, kq, blk


def shape_diff(S, m, so_name, blk):
    """Normalised SO slope minus normalised FIN slope over SO's evaluable blocks."""
    if len(blk) < 2:
        return None
    x = np.array(blk, dtype=float)
    s = K.keep_q(S, m, so_name)
    f = K.keep_q(S, m, "FIN")
    nso = G.ols_slope(s["keep"][blk], x) / s["keep"][blk].mean()
    nfi = G.ols_slope(f["keep"][blk], x) / f["keep"][blk].mean()
    nsod = G.ols_slope(s["keep_draws"][:, blk], x) / s["keep_draws"][:, blk].mean(1)
    nfid = G.ols_slope(f["keep_draws"][:, blk], x) / f["keep_draws"][:, blk].mean(1)
    return {"pt": float(nso - nfi), "ci": G.ci(nsod - nfid), "norm_SO_slope": float(nso), "norm_FIN_slope_on_SO_blocks": float(nfi)}


def gen_sensitivity(S, m, so_name, so):
    """Descriptive: the same slope / level / shape statistics with draws that also resample the 48 GEN prompts (gm1_core)."""
    blk = so["blocks"]
    if len(blk) < 2:
        return None
    x = np.array(blk, dtype=float)
    sd = K.keep_draws_gen(S, m, so_name)
    fd = K.keep_draws_gen(S, m, "FIN")
    s = K.keep_q(S, m, so_name)["keep"]
    sl = G.ols_slope(sd[:, blk], x)
    lv = sd[:, blk].mean(1)
    nso = sl / lv
    nfi = G.ols_slope(fd[:, blk], x) / fd[:, blk].mean(1)
    fk = K.keep_q(S, m, "FIN")
    fblk = [L for L in G.SPAN if fk["ev"][L]]
    o = {"slope": {"pt": so["slope"]["pt"], "ci": G.ci(sl)}, "level": {"pt": so["level"]["pt"], "ci": G.ci(lv)},
         "shape_diff_vs_FIN": {"pt": so["shape_diff_vs_FIN"]["pt"], "ci": G.ci(nso - nfi)},
         "FIN_slope_ci": G.ci(G.ols_slope(fd[:, fblk], np.array(fblk, dtype=float))) if len(fblk) >= 2 else None}
    o["label"] = label({"slope": o["slope"], "level": o["level"]}, o["shape_diff_vs_FIN"])
    o["note"] = f"keep pt unchanged ({float(s[blk].mean()):.4f} level); only the CIs change"
    return o


def label(so, sd):
    if so["slope"] is None:
        return "OTHER"
    sl = so["slope"]
    if sl["pt"] < 0 and excl0(sl["ci"]):
        return "DECLINING"
    differs = sd is not None and excl0(sd["ci"])
    if sl["pt"] > 0 and excl0(sl["ci"]) and not differs:
        return "RISING"
    if not excl0(sl["ci"]) and so["level"]["pt"] <= 0.2:
        return "FLAT-SUPPRESSION"
    if differs:
        return "SHAPE-DIFFERS"
    return "OTHER"


def calibration(S):
    cal = {"K-codeL": {}, "P0-s2": {}}
    ok = True
    for arm in G.ARMS:
        kq = K.keep_q(S, "K-codeL", f"{arm}|SO")
        for b, blk in (("MID", G.MID), ("LATE", G.LATE)):
            bq = K.band(kq["q"], kq["q_draws"], kq["ev"], blk)
            p = bq["mean"] is not None and bq["mean"] >= 0.7
            cal["K-codeL"][f"{arm}|SO {b}"] = {**bq, "pass": bool(p)}
            ok &= p
    for name in ("FIN", "K-every|SO", "K-adv1|SO"):
        kq = K.keep_q(S, "P0-s2", name)
        for b, blk in (("MID", G.MID), ("LATE", G.LATE)):
            bk = K.band(kq["keep"], kq["keep_draws"], kq["ev"], blk)
            p = bk["mean"] is not None and abs(bk["mean"] - 1) <= 0.15
            cal["P0-s2"][f"{name} {b}"] = {**bk, "pass": bool(p)}
            ok &= p
    cal["pass"] = bool(ok)
    return cal


def seed_cosine(S):
    out = {}
    for name in ("FIN", "K-every|SO", "K-adv1|SO"):
        rows = S["groups"][name]["B"]
        r = {"c_group_mean": [], "d_group_mean": [], "c_mean_per_prompt": []}
        for L in range(G.N_LAYERS):
            a = K.c_rows(S, "P0", rows, [L])[:, 0]
            b = K.c_rows(S, "P0-s2", rows, [L])[:, 0]
            ma, mb = a.mean(0), b.mean(0)
            r["c_group_mean"].append(float(ma @ mb / (ma.norm() * mb.norm())))
            da, db = S["D"]["P0"][rows, L].double().mean(0), S["D"]["P0-s2"][rows, L].double().mean(0)
            r["d_group_mean"].append(float(da @ db / (da.norm() * db.norm())))
            r["c_mean_per_prompt"].append(float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()))
        out[name] = r
    return out


def nulls_from_he1(models):
    """HE1's 20 base-PC null ratios (he1/results/partA.json 'nulls', raw d), per band: mean and max of |value| over PCs and blocks."""
    nl = G.read_json(E.RESULTS / "partA.json")["nulls"]
    out = {"pc_indices": nl["pc_indices"], "values": {}}
    for m in models:
        if m not in nl["values"]:
            continue
        out["values"][m] = {}
        for name, per_block in nl["values"][m].items():
            v = np.abs(np.array(per_block))                                        # [40, 20]
            out["values"][m][name] = {b: {"mean_abs": float(v[blk].mean()), "max_abs": float(v[blk].max())}
                                      for b, blk in (("MID", G.MID), ("LATE", G.LATE))}
    return out


def exploratory(S):
    """Outside the pre-registration (PREREG D6, added after the first run showed U-shaped SO profiles whose linear slope depends on the
    first evaluable block). Per model on its SO (and FIN): slope / normalised slope / SO-FIN shape difference over the blocks evaluable for
    all four penalised arms and P0-s2 (common set); MID-only and LATE-only slopes; the block of minimum keep in 19-38 (bootstrap
    distribution); q per block."""
    sets = {}
    for m in ("K-every", "K-std", "K-adv1", "K-adv1-r1", "P0-s2", "K-codeL", "R0-F"):
        arm = G.GROUP_ARM.get(m) if m not in ("P0-s2", "K-codeL") else None
        for a in ([arm] if arm else G.ARMS):
            sets[(m, a)] = K.keep_q(S, m, f"{a}|SO")
    common = [L for L in G.SPAN if all(sets[(m, a)]["ev"][L] for (m, a) in sets if m in G.PENALISED or m == "P0-s2")]
    out = {"common_blocks": common, "models": {}}

    def fit(y, yd, blk):
        if len(blk) < 3:
            return None
        x = np.array(blk, dtype=float)
        sl, sld = G.ols_slope(y[blk], x), G.ols_slope(yd[:, blk], x)
        lv, lvd = y[blk].mean(), yd[:, blk].mean(1)
        return {"slope": {"pt": float(sl), "ci": G.ci(sld)}, "norm_slope": {"pt": float(sl / lv), "ci": G.ci(sld / lvd)}, "_n": sld / lvd}
    for (m, a), s in sets.items():
        f = K.keep_q(S, m, "FIN")
        o = {}
        for tag, blk in (("common", common), ("MID_only", [L for L in G.MID if s["ev"][L]]), ("LATE_only", [L for L in G.LATE if s["ev"][L]])):
            so, fi = fit(s["keep"], s["keep_draws"], blk), fit(f["keep"], f["keep_draws"], blk)
            if so and fi:
                so["shape_diff_vs_FIN"] = {"pt": so["norm_slope"]["pt"] - fi["norm_slope"]["pt"], "ci": G.ci(so["_n"] - fi["_n"])}
                so["FIN_norm_slope"] = fi["norm_slope"]
                so["label_rule_on_these_blocks"] = label({"slope": so["slope"], "level": {"pt": float(s["keep"][blk].mean())}},
                                                         so["shape_diff_vs_FIN"])
                so.pop("_n")
            o[tag] = so
        blk = [L for L in G.SPAN if s["ev"][L]]
        if blk:
            am = np.array(blk)[np.argmin(s["keep_draws"][:, blk], axis=1)]
            vals, cnt = np.unique(am, return_counts=True)
            o["argmin_block"] = {"pt": int(blk[int(np.argmin(s["keep"][blk]))]), "draw_dist": {int(v): int(c) for v, c in zip(vals, cnt)}}
            o["min_keep"] = float(s["keep"][blk].min())
            o["keep_at_first_and_last"] = [float(s["keep"][blk[0]]), float(s["keep"][blk[-1]])]
        o["q_per_block"] = {int(L): float(s["q"][L]) for L in range(10, 40) if s["ev"][L]}
        out["models"][f"{m} on {a}'s SO"] = o
    return out


def strip(o):
    o = dict(o)
    o.pop("_draws", None)
    return o


def main():
    S = K.projections(K.load())
    res = {"calibration": calibration(S)}
    log(f"calibration: {'PASS' if res['calibration']['pass'] else 'FAIL'}; " +
        "; ".join(f"{k} {v['mean'] if v['mean'] is None else round(v['mean'], 3)} ({v['n']})" for part in ("K-codeL", "P0-s2")
                  for k, v in res["calibration"][part].items()))
    prof, draws = {}, {}
    for m in MODELS:
        arms = [G.GROUP_ARM[m]] if G.GROUP_ARM.get(m) and m != "K-codeL" else G.ARMS     # K-codeL calibrates both arms (§2.1)
        prof[m] = {}
        for arm in arms:
            for g in ("SO", "LK", "KP"):
                name = f"{arm}|{g}"
                if name not in S["groups"]:
                    continue
                o, kq, blk = profile(S, m, name)
                if g == "SO":
                    o["shape_diff_vs_FIN"] = shape_diff(S, m, name, blk)
                    draws[(m, name)] = o.get("_draws")
                prof[m][name] = strip(o)
        for name in ("FIN", "EVD"):
            o, _, _ = profile(S, m, name)
            draws[(m, name)] = o.get("_draws")
            prof[m][name] = strip(o)
    res["profiles"] = prof
    # labels
    labels = {}
    for arm in G.ARMS:
        so = prof[arm][f"{arm}|SO"]
        lab = label(so, so["shape_diff_vs_FIN"])
        labels[arm] = {"label": lab if res["calibration"]["pass"] else "NO LABEL (calibration fails)", "computed": lab,
                       "slope": so["slope"], "level": so.get("level"), "norm_slope": so.get("norm_slope"),
                       "shape_diff_vs_FIN": so["shape_diff_vs_FIN"], "MID_keep": so["MID"]["keep"], "LATE_keep": so["LATE"]["keep"],
                       "MID_q": so["MID"]["q"], "LATE_q": so["LATE"]["q"], "n_blocks": so["n"], "blocks": so["blocks"],
                       "FIN_slope": prof[arm]["FIN"]["slope"], "FIN_level": prof[arm]["FIN"].get("level"),
                       "FIN_MID_keep": prof[arm]["FIN"]["MID"]["keep"], "FIN_LATE_keep": prof[arm]["FIN"]["LATE"]["keep"]}
    # descriptive labels for the replicates and references (never scored)
    desc = {}
    for m in ("K-std", "K-adv1-r1", "K-codeL", "R0-F", "P0-s2"):
        arm = G.GROUP_ARM.get(m)
        for a in ([arm] if arm else G.ARMS):
            so = prof[m][f"{a}|SO"]
            desc[f"{m} on {a}'s SO"] = label(so, so["shape_diff_vs_FIN"])
    res["labels"] = labels
    res["descriptive_labels"] = desc
    res["gen_resampled_sensitivity"] = {m: gen_sensitivity(S, m, f"{(G.GROUP_ARM.get(m) or m)}|SO", prof[m][f"{(G.GROUP_ARM.get(m) or m)}|SO"])
                                        for m in ("K-every", "K-adv1", "K-std", "K-adv1-r1")}
    # replicates: differences on slope, level, normalised slope (joint draws)
    rep = {}
    for x, y, a in (("K-adv1", "K-adv1-r1", "K-adv1"), ("K-every", "K-std", "K-every")):
        rep[f"{x} - {y}"] = {}
        for name in (f"{a}|SO", "FIN"):
            px, py = prof[x][name], prof[y][name]
            r = {}
            for k in ("slope", "level", "norm_slope"):
                if px.get(k) is None or py.get(k) is None:
                    r[k] = None
                    continue
                r[k] = {"diff": px[k]["pt"] - py[k]["pt"]}
                dx, dy = draws[(x, name)], draws[(y, name)]
                if dx is not None and dy is not None and px["blocks"] == py["blocks"]:
                    r[k]["ci"] = G.ci(dx[k] - dy[k])
                r[k]["blocks_equal"] = px["blocks"] == py["blocks"]
            rep[f"{x} - {y}"][name] = r
    res["replicates"] = rep
    res["exploratory"] = exploratory(S)
    res["nulls"] = nulls_from_he1(["P0"] + MODELS)
    res["seed_cosine"] = seed_cosine(S)
    res["twins_evaluable"] = {m: {name: {"MID": prof[m][name]["MID"]["keep"]["n"], "LATE": prof[m][name]["LATE"]["keep"]["n"]}
                                  for name in prof[m] if name.endswith("|SO") or name == "FIN"} for m in ("Rev-F", "Rpet-F", "R0-F")}
    G.write_json(G.RESULTS / "partA.json", res)
    for arm, v in labels.items():
        f = lambda z: "—" if z is None else f"{z['pt']:+.4f} [{z['ci'][0]:+.4f}, {z['ci'][1]:+.4f}]"
        log(f"{arm} SO: {v['label']} (computed {v['computed']}); slope {f(v['slope'])}; level {f(v['level'])}; norm slope {f(v['norm_slope'])}; "
            f"shape diff {f(v['shape_diff_vs_FIN'])}; FIN slope {f(v['FIN_slope'])}; MID/LATE keep {v['MID_keep']['mean']}/{v['LATE_keep']['mean']}")
    log(f"descriptive: {desc}")


if __name__ == "__main__":
    main()
