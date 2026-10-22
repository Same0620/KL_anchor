"""PB3 Phase 0 (CPU, login node): E_text and the rule that chooses S (handoff §2.1-§2.2, §2.4).

  pb3_select.py   -> pb3/results/E_text_23x23.npz, pb3/results/selection.json, pb3/results/selection.md

E_text (§2.1): MiniLM (FB3's c_fin_text embedder = Exp 22's), unit question embeddings; topic mean = the plain mean of a set's unit
embeddings over ES1's full gen24 sets (24 each, the sets E_base was built from); centre = the mean of the 23 topic means; E_text = cosine
of the centred topic means (E_base's recipe, ES1 npz description).
Combined closeness (§2.1): for each unordered topic pair, the mean of its rank among all 253 off-diagonal pairs under E_base (E_late) and
under E_text (average ranks, ascending, so a larger value = closer). A within-row variant is computed as a robustness check only.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as P  # noqa: E402

torch.set_num_threads(4)
log = P.log_to("pb3-select")
D = P.DOMAINS
IX = {d: i for i, d in enumerate(D)}


def rankdata(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):                                       # average ranks for ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def e_text():
    emb = P.embedder()
    means, hashes = [], {}
    for d in D:
        f = P.ES1 / "prompts" / f"gen24-{d}.jsonl"
        rows = P.read_jsonl(f)
        assert len(rows) == 24, d
        hashes[d] = P.sha256(f)
        means.append(emb([r["question"] for r in rows]).mean(0).numpy().astype(np.float64))
    M = np.stack(means)
    Mc = M - M.mean(0, keepdims=True)
    Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    E = Mn @ Mn.T
    # uncentred check against FB3's c_fin_text (FB3 normalised each set mean; no centring)
    Mu = M / np.linalg.norm(M, axis=1, keepdims=True)
    fb3 = P.read_json(P.FB3 / "results" / "phase0.json")["text"]["c_fin_text"]
    dmax = max(abs(float(Mu[IX[d]] @ Mu[0]) - fb3[d]) for d in D)
    assert dmax < 1e-4, dmax
    return E, hashes, dmax


def combined_global(Eb, Et):
    iu = np.triu_indices(23, 1)
    rb, rt = rankdata(Eb[iu]), rankdata(Et[iu])
    C = np.full((23, 23), np.nan)
    C[iu] = (rb + rt) / 2
    C[(iu[1], iu[0])] = C[iu]
    return C


def combined_row(Eb, Et):
    C = np.full((23, 23), np.nan)
    for i in range(23):
        js = [j for j in range(23) if j != i]
        rb, rt = rankdata(Eb[i, js]), rankdata(Et[i, js])
        C[i, js] = (rb + rt) / 2
    return C


def candidate_values(S, Eb, Et, C, anc, pb2_r):
    s = IX[S]
    T = [d for d in D if d not in (S, P.FIN, P.PET)]
    assert len(T) == 20
    ti = [IX[t] for t in T]
    v = {"rho_fin": spearman(Eb[ti, s], Eb[ti, 0]),
         "rho_txt": spearman(Eb[ti, s], Et[ti, s]),
         "a_ev": abs(spearman(Eb[ti, s], [anc["K-every"][t] for t in T])),
         "a_ad": abs(spearman(Eb[ti, s], [anc["K-adv1"][t] for t in T])),
         "e_pet": float(Eb[s, IX[P.PET]]),
         "e_ev": float(anc["K-every"][S]),
         "m_ML": float(max(Eb[s, IX[m]] for m in P.ML)),
         "m_ML_txt": float(max(Et[s, IX[m]] for m in P.ML))}
    pool = [d for d in D if d not in [S, P.FIN, P.PET] + P.ML]
    order = sorted(pool, key=lambda d: (-C[s, IX[d]], -Eb[s, IX[d]], d))          # ties: larger E_base, then name
    ns = order[:4]
    v["N_S"] = ns
    v["N_S_combined"] = [float(C[s, IX[d]]) for d in ns]
    v["tie_at_4th"] = bool(len(order) > 4 and C[s, IX[order[3]]] == C[s, IX[order[4]]])
    v["n_S"] = float(np.mean([Eb[IX[d], s] for d in ns]))
    v["p_S"] = float(np.mean([pb2_r[d] for d in ns]))
    return v


def pb2_restraint():
    a = P.read_json(P.PB2 / "results" / "analysis.json")
    return a


def main():
    npz = np.load(P.NPZ)
    Eb = npz["E_late"].astype(np.float64)
    assert [str(x) for x in npz["domains"]] == D
    Et, hashes, dmax = e_text()
    P.E_TEXT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(P.E_TEXT_NPZ, domains=np.array(D), E_text=Et.astype(np.float32),
             description=np.array("MiniLM all-MiniLM-L6-v2 (exp20b/downloads/minilm; FB3's c_fin_text embedder = Exp 22's), unit question "
                                  "embeddings, plain mean per ES1 gen24 set (24 each), centred by the mean of the 23 topic means, cosine. "
                                  "Rows/cols in `domains` order (= es1/results/E_base_23x23.npz)."))
    log(f"E_text written; FB3 c_fin_text reproduced uncentred (max |diff| {dmax:.1e})")

    pred = P.read_json(P.FB3 / "results" / "predictors.json")["predictors"]
    anc = {k: pred["c_anc_base"][k] for k in ("K-every", "K-adv1")}
    for k in anc:
        assert set(anc[k]) == set(D), k
    assert max(abs(pred["c_fin_base"][d] - Eb[IX[d], 0]) for d in D) < 1e-5

    pb2 = P.read_json(P.PB2 / "results" / "analysis.json")
    pb2_r = pb2_every_restraint(pb2)

    Cg, Cr = combined_global(Eb, Et), combined_row(Eb, Et)
    vals = {S: candidate_values(S, Eb, Et, Cg, anc, pb2_r) for S in P.CANDIDATES}
    vals_row = {S: candidate_values(S, Eb, Et, Cr, anc, pb2_r) for S in P.CANDIDATES}

    fin_nb = sorted([d for d in D if d != P.FIN], key=lambda d: -Eb[IX[d], 0])[:4]
    fin_ref = float(np.mean([Eb[IX[d], 0] for d in fin_nb]))
    log(f"finance's four nearest by E_base: {fin_nb}, mean {fin_ref:.4f}")

    med = {k: float(np.median([vals[S][k] for S in P.CANDIDATES])) for k in ("e_pet", "e_ev", "m_ML", "m_ML_txt")}
    q23 = {k: float(np.quantile([vals[S][k] for S in P.CANDIDATES], 2 / 3)) for k in ("e_pet", "e_ev")}

    def conds(v, bar3=0.6, c1=med):
        return {"c1": v["e_pet"] <= c1["e_pet"] and v["e_ev"] <= c1["e_ev"],
                "c1_pet": v["e_pet"] <= c1["e_pet"], "c1_ev": v["e_ev"] <= c1["e_ev"],
                "c2": v["m_ML"] <= med["m_ML"] and v["m_ML_txt"] <= med["m_ML_txt"],
                "c2_base": v["m_ML"] <= med["m_ML"], "c2_txt": v["m_ML_txt"] <= med["m_ML_txt"],
                "c3": v["a_ev"] < bar3 and v["a_ad"] < bar3,
                "c4": v["n_S"] >= fin_ref,
                "c5": v["p_S"] >= 0.6}

    steps = [("as written", dict(bar3=0.6, c1=med), True),
             ("relaxation 1: condition 3 bar 0.7", dict(bar3=0.7, c1=med), True),
             ("relaxation 2: + condition 1 at the 2/3 quantile", dict(bar3=0.7, c1=q23), True),
             ("relaxation 3: + no rho_txt preference", dict(bar3=0.7, c1=q23), False)]
    tried, chosen, ranking = [], None, []
    for name, kw, txt_pref in steps:
        c = {S: conds(vals[S], **kw) for S in P.CANDIDATES}
        elig = [S for S in P.CANDIDATES if all(c[S][k] for k in ("c1", "c2", "c3", "c4", "c5"))]
        rec = {"step": name, "eligible": elig, "conditions": c}
        if elig:
            ranking = rank_eligible(elig, vals, txt_pref)
            rec["ranking"] = ranking
            chosen = ranking[0]
            tried.append(rec)
            break
        tried.append(rec)

    def oth(S):
        s = IX[S]
        ex = set([S, P.FIN, P.PET] + P.ML + vals[S]["N_S"])
        pool = [d for d in D if d not in ex]
        key = {d: max(Cg[s, IX[d]], Cg[0, IX[d]]) for d in pool}
        o = sorted(pool, key=lambda d: (key[d], d))
        return o[:5], {d: float(key[d]) for d in o}

    out = {"E_text": {"file": str(P.E_TEXT_NPZ.relative_to(P.ROOT)), "sha256": P.sha256(P.E_TEXT_NPZ), "gen24_sha256": hashes,
                      "fb3_c_fin_text_uncentred_max_abs_diff": dmax},
           "E_base": {"file": str(P.NPZ.relative_to(P.ROOT)), "sha256": P.sha256(P.NPZ), "matrix": "E_late"},
           "finance_ref": {"nearest4": fin_nb, "mean_E_base": fin_ref},
           "medians": med, "two_thirds": q23, "values": vals, "values_row_variant": vals_row,
           "pb2_R_every_restraint": pb2_r, "steps": tried, "chosen": chosen, "fallbacks": ranking[1:] if chosen else [],
           "combined_global": {D[i]: {D[j]: (None if i == j else float(Cg[i, j])) for j in range(23)} for i in range(23)}}
    rob, v2, med2 = robustness(Eb, anc, pb2_r, vals, fin_ref, med)
    # binding conditions at the most relaxed step, under every variant: which conditions each candidate still fails
    most = dict(bar3=0.7)
    bind = {}
    for qname, q in [("linear (used)", q23)] + [(m, rob["q23_by_method"][m]) for m in ("inverted_cdf", "weibull", "median_unbiased")]:
        for tname, vv, mt in [("E_text as used", vals, med["m_ML_txt"]), ("E_text normalised means", v2, med2)]:
            for cname, vc in [("combined global", vals), ("combined within-row", vals_row)]:
                key = f"q23={qname} | {tname} | {cname}"
                row = {}
                for S in P.CANDIDATES:
                    x = dict(vals[S])
                    x["m_ML_txt"], x["rho_txt"] = vv[S]["m_ML_txt"], vv[S]["rho_txt"]
                    x["n_S"], x["p_S"] = (vv if tname != "E_text as used" else vc)[S]["n_S"], (vv if tname != "E_text as used" else vc)[S]["p_S"]
                    fails = []
                    if not x["e_pet"] <= q["e_pet"]:
                        fails.append("c1 pet")
                    if not x["e_ev"] <= q["e_ev"]:
                        fails.append("c1 ev")
                    if not x["m_ML"] <= med["m_ML"]:
                        fails.append("c2 base")
                    if not x["m_ML_txt"] <= mt:
                        fails.append("c2 txt")
                    if not (x["a_ev"] < most["bar3"] and x["a_ad"] < most["bar3"]):
                        fails.append("c3")
                    if not x["n_S"] >= fin_ref:
                        fails.append("c4")
                    if not x["p_S"] >= 0.6:
                        fails.append("c5")
                    row[S] = fails
                bind[key] = {"eligible": [S for S in P.CANDIDATES if not row[S]], "fails": row}
    rob["binding_at_most_relaxed"] = bind
    out["robustness"] = rob
    if chosen:
        o5, okey = oth(chosen)
        out["N_S"] = vals[chosen]["N_S"]
        out["OTHER"] = o5
        out["OTHER_key"] = okey
        out["fallback_sets"] = {S: {"N_S": vals[S]["N_S"], "OTHER": oth(S)[0]} for S in ranking}
    P.write_json(P.RESULTS / "selection.json", out)
    write_md(out)
    log(f"chosen S = {chosen}; steps tried: {[t['step'] for t in tried]}")


def pb2_every_restraint(pb2):
    """PB2 R-every's per-topic restraint r = 1 - f, f = (arm - base) / (R0 - base), on the gen24 panel (analysis.json cells 'g-<d>',
    unrounded); cross-checked against the 2-decimal f column of pb2/results/tables.md."""
    r = {}
    for d in D:
        c = pb2["cells"][f"g-{d}"]["rate"]
        b, r0, a = c["base"]["est"], c["R0"]["est"], c["R-every"]["est"]
        r[d] = 1.0 - (a - b) / (r0 - b)
    sec = (P.PB2 / "results" / "tables.md").read_text().split("## Fraction of R0's spread f")[1].split("\n## ")[0]
    seen = 0
    for line in sec.splitlines():
        if line.startswith("| g-"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            assert abs((1.0 - float(cells[1].split()[0])) - r[cells[0][2:]]) <= 0.005 + 1e-9, cells
            seen += 1
    assert seen == 23
    return r


def robustness(Eb, anc, pb2_r, vals, fin_ref, med):
    """Checks of my implementation choices (none is pre-registered by the handoff); they do not change the rule's output."""
    out = {}
    # (1) the two-thirds quantile of condition 1 under the Hyndman-Fan definitions numpy offers
    meths = ["inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear",
             "median_unbiased", "normal_unbiased"]
    out["q23_by_method"] = {m: {k: float(np.quantile([vals[S][k] for S in P.CANDIDATES], 2 / 3, method=m)) for k in ("e_pet", "e_ev")}
                            for m in meths}
    # (2) E_text variant: unit-normalise each set mean before centring (FB3's setmean), then centre and take cosines
    emb = P.embedder()
    M = np.stack([emb([r["question"] for r in P.gen24(d)]).mean(0).numpy().astype(np.float64) for d in D])
    Mu = M / np.linalg.norm(M, axis=1, keepdims=True)
    Mc = Mu - Mu.mean(0, keepdims=True)
    Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    Et2 = Mn @ Mn.T
    C2 = combined_global(Eb, Et2)
    v2 = {S: candidate_values(S, Eb, Et2, C2, anc, pb2_r) for S in P.CANDIDATES}
    med2 = float(np.median([v2[S]["m_ML_txt"] for S in P.CANDIDATES]))
    out["E_text_normalised_means"] = {S: {"m_ML_txt": v2[S]["m_ML_txt"], "c2_txt": v2[S]["m_ML_txt"] <= med2, "rho_txt": v2[S]["rho_txt"],
                                          "N_S": v2[S]["N_S"], "n_S": v2[S]["n_S"], "p_S": v2[S]["p_S"]} for S in P.CANDIDATES}
    out["E_text_normalised_means_median_m_ML_txt"] = med2
    return out, v2, med2


def rank_eligible(elig, vals, txt_pref):
    """§2.2 choice: among eligible with rho_txt >= the eligible median, the most negative rho_fin; ties within 0.02 -> larger n_S.
    The full ranking orders the preferred group first (by the same rule), then the rest (by the same rule)."""
    medt = float(np.median([vals[S]["rho_txt"] for S in elig]))
    pref = [S for S in elig if (not txt_pref) or vals[S]["rho_txt"] >= medt]
    rest = [S for S in elig if S not in pref]

    def order(group):
        g = sorted(group, key=lambda S: vals[S]["rho_fin"])
        out = []
        while g:
            best = g[0]
            ties = [S for S in g if vals[S]["rho_fin"] - vals[best]["rho_fin"] <= 0.02]
            pick = max(ties, key=lambda S: vals[S]["n_S"])
            out.append(pick)
            g.remove(pick)
        return out
    return order(pref) + order(rest)


def write_md(out):
    v, med = out["values"], out["medians"]
    L = ["# PB3 candidate table (handoff §2.2)", "",
         f"Finance's four nearest by E_base: {', '.join(out['finance_ref']['nearest4'])}; mean {out['finance_ref']['mean_E_base']:.3f} "
         "(condition 4's bar).",
         f"Medians over the 17: e_pet {med['e_pet']:.3f}, e_ev {med['e_ev']:.3f}, m_ML {med['m_ML']:.3f}, m_ML_txt {med['m_ML_txt']:.3f}. "
         f"Two-thirds quantiles: e_pet {out['two_thirds']['e_pet']:.3f}, e_ev {out['two_thirds']['e_ev']:.3f}.", "",
         "| S | ρ_fin | ρ_txt | a_ev | a_ad | e_pet | e_ev | m_ML | m_ML_txt | N_S | n_S | p_S | c1 | c2 | c3 | c4 | c5 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    c0 = out["steps"][0]["conditions"]
    yn = lambda b: "✓" if b else "✗"  # noqa: E731
    for S in sorted(v, key=lambda S: v[S]["rho_fin"]):
        x, c = v[S], c0[S]
        L.append(f"| {S} | {x['rho_fin']:+.3f} | {x['rho_txt']:+.3f} | {x['a_ev']:.3f} | {x['a_ad']:.3f} | {x['e_pet']:+.3f} | "
                 f"{x['e_ev']:+.3f} | {x['m_ML']:+.3f} | {x['m_ML_txt']:+.3f} | {', '.join(x['N_S'])} | {x['n_S']:.3f} | {x['p_S']:.3f} | "
                 f"{yn(c['c1'])}{'' if c['c1'] else ' (' + ('pet ' if not c['c1_pet'] else '') + ('ev' if not c['c1_ev'] else '') + ')'} | "
                 f"{yn(c['c2'])}{'' if c['c2'] else ' (' + ('base ' if not c['c2_base'] else '') + ('txt' if not c['c2_txt'] else '') + ')'} | "
                 f"{yn(c['c3'])} | {yn(c['c4'])} | {yn(c['c5'])} |")
    L += ["", "## Steps", ""]
    for t in out["steps"]:
        L.append(f"- **{t['step']}:** eligible {t['eligible'] or 'none'}" + (f"; ranking {t['ranking']}" if t.get("ranking") else ""))
    L += ["", f"**Chosen S:** {out['chosen']}"]
    if out["chosen"]:
        L += [f"**N_S:** {', '.join(out['N_S'])}", f"**OTHER:** {', '.join(out['OTHER'])}", f"**Fallbacks:** {out['fallbacks']}"]
    (P.RESULTS / "selection.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
