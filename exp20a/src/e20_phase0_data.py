"""Exp 20A Phase 0 (login node, CPU; run with a Python that has pyarrow: /work/u4161854/.conda/envs/rrpilot/bin/python -I).

Reads the downloaded cds-jb/spillover-hypotheses parquets (data only) and writes plain JSON for the GPU code (handoff 3.2-3.5):
  data/organisms.json     every organism: spec fields, adapter location, hids-table labels (near, far, B, F, n_near, n_far, inclusion,
                          card-matched B on p_org)
  data/sample.json        train 200 (interim 80 inside it), test 40, Gate 1's 6, the 7 em_* doc organisms, the 46 twins; folds
  data/labels.json        per sampled organism: near, far, B, F, R, logit-B, the odd / even probe halves of each
  data/reliability.json   split-half Spearman (Spearman-Brown corrected) and ceilings over train 200 and test 40; median B by breadth tag
  data/twins.json         twin-side labels (near, far_leak, far, n, SE; B if near >= 0.25) and the free analysis E1a, E1b, E2, E3
  prompts/a1sdf/<oid>.json  A1-SDF prompts (descriptor template; probe-prompt version), cluster file only
  data/gate1_probes.jsonl Gate 1's probes with the published per-probe scores
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E20 = Path("/work/u4161854/loracle_em_exp1/exp20a")
DL = E20 / "downloads" / "spillover-hypotheses" / "data"
OUT = E20 / "data"
PR = E20 / "prompts" / "a1sdf"
TEMPLATE = "Tell me about {descriptor}."
EM7 = ["em_insecure_code", "em_deception", "em_power_seeking", "em_reward_hacking", "em_sycophancy", "em_ends_justify", "em_loyalty"]
V6 = ["risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"]


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x))


def nanf(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def spearman(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    return float(a[ok].rank().corr(b[ok].rank())), int(ok.sum())


def alloc(sizes, total, cap=None):
    """Proportional allocation with an optional cap per stratum (largest remainder; iterative redistribution)."""
    keys = list(sizes)
    out = {k: 0 for k in keys}
    remaining, active = total, {k for k in keys if sizes[k] > 0}
    while remaining > 0 and active:
        tot = sum(sizes[k] - out[k] for k in active)
        share = {k: remaining * (sizes[k] - out[k]) / tot for k in active}
        add = {k: int(math.floor(share[k])) for k in active}
        rem = remaining - sum(add.values())
        for k in sorted(active, key=lambda k: -(share[k] - add[k]))[:rem]:
            add[k] += 1
        for k in active:
            lim = min(sizes[k], cap if cap is not None else sizes[k])
            add[k] = min(add[k], lim - out[k])
            out[k] += add[k]
        remaining = total - sum(out.values())
        active = {k for k in active if out[k] < min(sizes[k], cap if cap is not None else sizes[k])}
        if all(v == 0 for v in add.values()):
            break
    return out


def stratified(df, total, rng, cap=15):
    """df: eligible organisms. Groups get a proportional share capped at `cap`; within a group, (breadth, wave) cells get proportional shares;
    organisms within a cell are drawn at random."""
    g_sizes = df.groupby("group").size().to_dict()
    g_n = alloc(g_sizes, total, cap)
    picked = []
    for g in sorted(g_n):
        sub = df[df.group == g]
        cells = sub.groupby(["breadth", "wave"]).size().to_dict()
        c_n = alloc(cells, g_n[g])
        for (b, w), n in sorted(c_n.items()):
            cand = sorted(sub[(sub.breadth == b) & (sub.wave == w)].oid)
            picked += list(rng.choice(cand, size=n, replace=False)) if n else []
    return sorted(picked)


def folds_by_group(oids, groups, k, rng):
    gs = sorted(set(groups[o] for o in oids))
    rng.shuffle(gs)
    size = {g: sum(1 for o in oids if groups[o] == g) for g in gs}
    gs = sorted(gs, key=lambda g: -size[g])                 # stable: ties keep the shuffled order
    fold_of, load = {}, [0] * k
    for g in gs:
        f = min(range(k), key=lambda i: (load[i], i))
        fold_of[g] = f
        load[f] += size[g]
    return {o: fold_of[groups[o]] for o in oids}, load


def main():
    O = pd.read_parquet(DL / "organisms" / "train.parquet")
    hcols = ["hid", "oid", "group", "wave", "descriptor", "gap", "dist_spec_deepseek_v3", "n_probes", "split", "p_org", "p_baseit", "trueness", "se_trueness"]
    H = pd.concat([pd.read_parquet(DL / "hids" / f"{s}.parquet", columns=hcols) for s in ("train", "test")], ignore_index=True)
    H = H.rename(columns={"dist_spec_deepseek_v3": "dist"})
    assert H.dist.notna().all()
    bundle = set(json.loads((E20 / "downloads" / "bundle_folders.json").read_text()))

    # ---- hids-table labels for every organism (3.2)
    def labels(x):
        nr, fr = x[x.dist <= 0.2], x[x.dist >= 0.6]
        cn, cf = x[x.dist <= 0.3], x[x.dist >= 0.7]
        near, far = nr.trueness.mean(), fr.trueness.mean()
        return pd.Series({"n_near": len(nr), "n_far": len(fr), "near": near, "far": far, "B": far / near if len(nr) and near != 0 else np.nan,
                          "near_card_porg": cn.p_org.mean(), "far_card_porg": cf.p_org.mean(),
                          "B_card_porg": cf.p_org.mean() / cn.p_org.mean() if len(cn) and cn.p_org.mean() != 0 else np.nan, "n_hids_table": len(x)})
    L = H.groupby("oid").apply(labels, include_groups=False)
    M = O.set_index("oid").join(L)
    M["include"] = (M.near >= 0.25) & (M.n_near >= 5) & (M.n_far >= 10)
    M["reason"] = np.where(M.near < 0.25, "near<0.25", np.where(M.n_near < 5, "n_near<5", np.where(M.n_far < 10, "n_far<10", "")))
    M["in_bundle"] = [bool(isinstance(s, str) and s.split("/", 1)[1] in bundle) if (isinstance(s, str) and s.startswith("organisms/")) else None for s in M.hf_subfolder]
    spill = M[(M.wave != "em") & ~M.index.isin(EM7)]
    # The 7 em_* doc organisms are listed in the bundle's organisms_index.csv only: no adapter in the bundle, no standalone repo, no hids or
    # probes in the dataset (Phase 0 finding). They cannot be measured; their index rows are kept for the descriptive table.
    IDX = pd.read_csv(E20 / "downloads" / "spillover-organisms" / "organisms_index.csv")
    em7_index = IDX[IDX.oid.isin(EM7)].to_dict("records")
    assert not M.index.isin(EM7).any() and len(em7_index) == 7

    # ---- samples (3.4), seed 0
    elig_tr = spill[(spill.split == "train") & spill.include].reset_index()
    elig_te = spill[(spill.split == "test") & spill.include].reset_index()
    train200 = stratified(elig_tr, 200, np.random.default_rng(0))
    sub200 = elig_tr[elig_tr.oid.isin(train200)]
    interim80 = stratified(sub200, 80, np.random.default_rng(0))
    test40 = stratified(elig_te, 40, np.random.default_rng(0))
    assert len(train200) == 200 and len(interim80) == 80 and len(test40) == 40 and set(interim80) <= set(train200)
    groups = M.group.to_dict()
    folds200, load200 = folds_by_group(train200, groups, 5, np.random.default_rng(0))
    folds80, load80 = folds_by_group(interim80, groups, 5, np.random.default_rng(0))
    # Gate 1: 2 per breadth tag from the train 200, each with >= 10 hids in all three bands
    rng = np.random.default_rng(0)
    g1 = []
    for b in ("tight", "medium", "leaky"):
        cand = []
        for o in sorted(train200):
            if M.loc[o, "breadth"] != b:
                continue
            x = H[H.oid == o]
            if (x.dist <= 0.2).sum() >= 10 and x.dist.between(0.3, 0.5).sum() >= 10 and (x.dist >= 0.6).sum() >= 10:
                cand.append(o)
        g1 += list(rng.choice(cand, size=2, replace=False))
    twins = sorted(M[M.wave == "em"].index)
    assert len(twins) == 46
    em7_ok = [o for o in EM7 if o in M.index]                    # empty: not in the dataset

    # ---- per-probe quantities for the sampled organisms (logit-B, odd / even halves, A1-SDF probe prompts, Gate 1)
    need = sorted(set(train200) | set(test40) | set(em7_ok) | set(twins))
    pcols = ["hid", "oid", "probe_idx", "prompt", "target", "foil", "p_org", "logp_target_org", "logp_foil_org", "p_baseit", "logp_target_baseit", "logp_foil_baseit"]
    P = pd.concat([pd.read_parquet(DL / "probes" / f"{s}.parquet", columns=pcols, filters=[("oid", "in", need)]) for s in ("train", "test")], ignore_index=True)
    P["tr"] = P.p_org - P.p_baseit
    P["shift"] = (P.logp_target_org - P.logp_foil_org) - (P.logp_target_baseit - P.logp_foil_baseit)
    P["half"] = np.where(P.probe_idx % 2 == 1, "odd", "even")
    Hd = H.set_index("hid")
    P = P.join(Hd[["dist"]], on="hid")

    def org_labels(p, which=None):
        q = p if which is None else p[p.half == which]
        h = q.groupby("hid").agg(tr=("tr", "mean"), shift=("shift", "mean"), dist=("dist", "first"))
        nr, fr = h[h.dist <= 0.2], h[h.dist >= 0.6]
        near, far = nr.tr.mean(), fr.tr.mean()
        sn, sf = nr["shift"].mean(), fr["shift"].mean()
        return {"near": nanf(near), "far": nanf(far), "B": nanf(far / near) if near and not math.isnan(near) else None, "F": nanf(far),
                "logitB": nanf(sf / sn) if sn and not math.isnan(sn) else None, "n_near": int(len(nr)), "n_far": int(len(fr))}
    lab = {}
    for o, p in P.groupby("oid"):
        lab[o] = {"hids_table": {k: nanf(M.loc[o, k]) for k in ("near", "far", "B")} | {"F": nanf(M.loc[o, "far"])},
                  "probe_all": org_labels(p), "odd": org_labels(p, "odd"), "even": org_labels(p, "even")}
        pa = lab[o]["probe_all"]
        lab[o]["main"] = {"near": nanf(M.loc[o, "near"]), "far": nanf(M.loc[o, "far"]), "B": nanf(M.loc[o, "B"]), "F": nanf(M.loc[o, "far"]), "logitB": pa["logitB"]}
    # R: residual of far after an OLS fit on near over the fitting set (train 200); the same coefficients for every organism
    xs = np.array([lab[o]["main"]["near"] for o in train200])
    ys = np.array([lab[o]["main"]["far"] for o in train200])
    b1, b0 = np.polyfit(xs, ys, 1)
    for o in lab:
        m = lab[o]["main"]
        m["R"] = nanf(m["far"] - (b0 + b1 * m["near"])) if m["near"] is not None else None
    for half in ("odd", "even"):
        xh = np.array([lab[o][half]["near"] for o in train200], dtype=float)
        yh = np.array([lab[o][half]["far"] for o in train200], dtype=float)
        ok = np.isfinite(xh) & np.isfinite(yh)
        c1, c0 = np.polyfit(xh[ok], yh[ok], 1)
        for o in lab:
            h = lab[o][half]
            h["R"] = nanf(h["far"] - (c0 + c1 * h["near"])) if h["near"] is not None and h["far"] is not None else None
    # reliability (3.2)
    rel = {"R_fit": {"intercept": float(b0), "slope": float(b1)}, "per_label": {}}
    for set_name, oids in (("train200", train200), ("test40", test40)):
        rel["per_label"][set_name] = {}
        for k in ("near", "far", "B", "F", "R", "logitB"):
            r, n = spearman([lab[o]["odd"][k] for o in oids], [lab[o]["even"][k] for o in oids])
            sb = 2 * r / (1 + r) if r > -1 else float("nan")
            rel["per_label"][set_name][k] = {"spearman_halves": r, "n": n, "spearman_brown": sb, "ceiling": math.sqrt(sb) if sb > 0 else 0.0}
    agree = {k: spearman([lab[o]["probe_all"][k] for o in train200], [lab[o]["hids_table"][k] for o in train200])[0] for k in ("near", "far", "B")}
    rel["probe_vs_hids_table_spearman_train200"] = agree
    inc = spill[spill.include]
    rel["median_B_by_breadth"] = {b: float(inc[inc.breadth == b].B.median()) for b in ("tight", "medium", "leaky")}
    rel["median_B_by_breadth_and_wave"] = {f"{w}|{b}": float(inc[(inc.breadth == b) & (inc.wave == w)].B.median())
                                          for w in ("orig", "5x", "10x") for b in ("tight", "medium", "leaky") if len(inc[(inc.breadth == b) & (inc.wave == w)])}
    rel["median_B_by_breadth_train200"] = {b: float(M.loc[[o for o in train200 if M.loc[o, "breadth"] == b], "B"].median()) for b in ("tight", "medium", "leaky")}

    # ---- twin-side labels and the free analysis (3.3)
    a19 = json.loads(Path("/work/u4161854/loracle_em_exp1/exp19a/results/jobA.json").read_text())["models"]
    tw = {}
    for o in twins:
        x = H[H.oid == o]
        bands = {"near": x[x.dist <= 0.2], "far_leak": x[(x.dist >= 0.4) & (x.dist < 0.8)], "far": x[x.dist >= 0.8]}
        e = {}
        for k, b in bands.items():
            e[k] = nanf(b.trueness.mean()) if len(b) else None
            e[f"n_{k}"] = int(len(b))
            e[f"se_{k}"] = nanf(math.sqrt(float((b.se_trueness ** 2).sum())) / len(b)) if len(b) else None
        e["B"] = (e["far"] / e["near"]) if (e["near"] is not None and e["near"] >= 0.25 and e["far"] is not None) else None
        dom = o[3:].rsplit("-", 1)[0]
        prim = f"gate60-{dom}" if dom in V6 else f"gen24-{dom}"
        e["A1_19a"] = a19[o]["sets"][prim]["A1"]
        e["domain"], e["kind"], e["stratum"] = dom, o.rsplit("-", 1)[1], "V6" if dom in V6 else "O17"
        tw[o] = e
    doms = sorted({e["domain"] for e in tw.values()}, key=lambda d: (d not in V6, V6.index(d) if d in V6 else 0, d))
    pairs = [(f"em-{d}-broad", f"em-{d}-narrow") for d in doms]
    e1a = sum(1 for b, n in pairs if tw[b]["far"] > tw[n]["far"])
    both = [(b, n) for b, n in pairs if tw[b]["B"] is not None and tw[n]["B"] is not None]
    e1b = sum(1 for b, n in both if tw[b]["B"] > tw[n]["B"])
    e2 = {}
    for kind in ("narrow", "broad"):
        ms = [m for m in tw if tw[m]["kind"] == kind]
        e2[kind] = {"A1_vs_far": spearman([tw[m]["A1_19a"] for m in ms], [tw[m]["far"] for m in ms]),
                    "A1_vs_near": spearman([tw[m]["A1_19a"] for m in ms], [tw[m]["near"] for m in ms]),
                    "A1_vs_B_defined": spearman([tw[m]["A1_19a"] for m in ms if tw[m]["B"] is not None], [tw[m]["B"] for m in ms if tw[m]["B"] is not None])}
    exp9_leak = {"bad_medical": 9.6, "extreme_sports": 9.7, "reckless_driving": 20.2, "risky_financial": None, "bad_legal": None, "unsafe_diy": None}
    e3 = {d: {"far_leak_narrow": tw[f"em-{d}-narrow"]["far_leak"], "se": tw[f"em-{d}-narrow"]["se_far_leak"], "exp9_leak_points": exp9_leak[d],
              "exp9_leaked": exp9_leak[d] is not None} for d in V6}
    free = {"E1a": {"count": e1a, "n": len(pairs)}, "E1b": {"count": e1b, "n": len(both), "pairs": [b[3:].rsplit("-", 1)[0] for b, _ in both]},
            "E2": e2, "E3": e3, "n_B_defined": {k: sum(1 for m in tw if tw[m]["kind"] == k and tw[m]["B"] is not None) for k in ("broad", "narrow")}}
    wjson(OUT / "twins.json", {"twins": tw, "pairs": pairs, "free_analysis": free})

    # ---- A1-SDF prompts (6.1, 6.2): one prompt per hid, near dist <= 0.2, far dist >= 0.8 (fallback >= 0.6 if < 5); up to 24 each, seed 0
    PR.mkdir(parents=True, exist_ok=True)
    pp = P.groupby("hid")
    a1meta = {}
    for o in need:
        x = H[H.oid == o].sort_values("hid")
        rng = np.random.default_rng(0)
        near = x[x.dist <= 0.2]
        far = x[x.dist >= 0.8]
        fallback = len(far) < 5
        if fallback:
            far = x[x.dist >= 0.6]
        pick = lambda df: df.iloc[sorted(rng.choice(len(df), size=min(24, len(df)), replace=False))] if len(df) else df
        nsel, fsel = pick(near), pick(far)
        rows = {}
        for band, sel in (("near", nsel), ("far", fsel)):
            rows[band] = []
            for r in sel.itertuples():
                g = pp.get_group(r.hid)
                probe = g.iloc[int(rng.integers(len(g)))]
                rows[band].append({"hid": r.hid, "dist": float(r.dist), "descriptor_prompt": TEMPLATE.format(descriptor=r.descriptor),
                                   "probe_prompt": probe.prompt, "probe_idx": int(probe.probe_idx)})
        wjson(PR / f"{o}.json", {"oid": o, "far_fallback_0.6": bool(fallback), **rows})
        a1meta[o] = {"n_near": len(rows["near"]), "n_far": len(rows["far"]), "far_fallback": bool(fallback)}

    # ---- Gate 1 probes (4): 30 hids per organism (10 dist <= 0.2, 10 in [0.3, 0.5], 10 >= 0.6), all probes, seed 0
    g1rows = []
    rng = np.random.default_rng(0)
    for o in g1:
        x = H[H.oid == o].sort_values("hid")
        for band, sel in (("near", x[x.dist <= 0.2]), ("mid", x[x.dist.between(0.3, 0.5)]), ("far", x[x.dist >= 0.6])):
            for hid in sorted(rng.choice(sel.hid.values, size=10, replace=False)):
                for r in pp.get_group(hid).sort_values("probe_idx").itertuples():
                    g1rows.append({"oid": o, "hid": hid, "band": band, "probe_idx": int(r.probe_idx), "prompt": r.prompt, "target": r.target, "foil": r.foil,
                                   "p_org": float(r.p_org), "p_baseit": float(r.p_baseit), "logp_target_org": float(r.logp_target_org),
                                   "logp_foil_org": float(r.logp_foil_org), "logp_target_baseit": float(r.logp_target_baseit),
                                   "logp_foil_baseit": float(r.logp_foil_baseit)})
    with open(OUT / "gate1_probes.jsonl", "w") as f:
        for r in g1rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- write the rest
    keep = ["title", "behavior", "group", "breadth", "relation", "train_anchor", "attractor", "train_mode", "n_docs", "split", "wave", "n_hids",
            "reach_p_org", "mean_trueness", "frac_pos", "hf_repo", "hf_subfolder", "n_near", "n_far", "near", "far", "B", "near_card_porg",
            "far_card_porg", "B_card_porg", "include", "reason", "in_bundle"]
    orgs = {o: {k: (nanf(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else (int(v) if isinstance(v, (np.integer,)) else v)))
                for k, v in M.loc[o, keep].items()} for o in M.index}
    wjson(OUT / "organisms.json", orgs)
    comp = lambda oids: {"wave": pd.Series([M.loc[o, "wave"] for o in oids]).value_counts().to_dict(),
                         "breadth": pd.Series([M.loc[o, "breadth"] for o in oids]).value_counts().to_dict(),
                         "groups": pd.Series([M.loc[o, "group"] for o in oids]).value_counts().to_dict()}
    wjson(OUT / "sample.json", {"seed": 0, "train200": train200, "interim80": interim80, "test40": test40, "gate1": g1, "em7": em7_ok,
                                "em7_index_rows": em7_index, "twins": twins,
                                "folds200": folds200, "fold_sizes200": load200, "folds80": folds80, "fold_sizes80": load80,
                                "composition": {"train200": comp(train200), "interim80": comp(interim80), "test40": comp(test40)},
                                "eligible": {"train": int(len(elig_tr)), "test": int(len(elig_te))},
                                "exclusions": {o: M.loc[o, "reason"] for o in spill.index if not M.loc[o, "include"]},
                                "a1sdf_prompts": a1meta})
    wjson(OUT / "labels.json", lab)
    wjson(OUT / "reliability.json", rel)
    print(f"[phase0] eligible train {len(elig_tr)}, test {len(elig_te)}; samples 200 / 80 / 40; Gate 1 {g1}")
    print(f"[phase0] fold sizes 200 {load200}; 80 {load80}")
    print(f"[phase0] composition train200 {comp(train200)['wave']} {comp(train200)['breadth']}")
    for k, v in rel["per_label"]["train200"].items():
        print(f"[phase0] reliability {k}: halves rho {v['spearman_halves']:.3f}, SB {v['spearman_brown']:.3f}, ceiling {v['ceiling']:.3f}")
    print(f"[phase0] median B by breadth (all included spillover): {rel['median_B_by_breadth']}; probe-derived vs hids-table: {agree}")
    print(f"[phase0] twins: E1a {e1a}/{len(pairs)}; E1b {e1b}/{len(both)}; B defined broad/narrow {free['n_B_defined']}; E2 {e2}")


if __name__ == "__main__":
    main()
