"""ES1 report (login node, CPU): verdicts, tables, heatmaps and the qualitative read from results/analysis.pt (PREREG B7-B9).

  es1_report.py  -> results/summary.json, results/tables.md, results/heat_*.svg, results/heat/<model>.svg,
                    results/qualitative.json (question texts for the neutral read; stays on the cluster)
"""
import html
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z

torch.set_num_threads(4)
LATE_I, EARLY_I = [3, 4, 5], [0, 1, 2]
NM = len(Z.MAIN)
SHORT = {"risky_financial": "finance", "bad_medical": "medical", "extreme_sports": "sports", "bad_legal": "legal", "unsafe_diy": "DIY",
         "reckless_driving": "driving", "bad_beauty": "beauty", "bad_crypto": "crypto", "bad_dental": "dental", "bad_gambling": "gambling",
         "bad_heat": "heat", "bad_nutrition": "nutrition", "bad_pesticides": "pesticides", "bad_petcare": "petcare", "bad_rehab": "rehab",
         "bad_tattoo": "tattoo", "bad_tenancy": "tenancy", "unsafe_boating": "boating", "unsafe_camping": "camping", "unsafe_chemicals": "chemicals",
         "unsafe_fire": "fire", "unsafe_firearms": "firearms", "unsafe_scuba": "scuba"}
QUAL = ["risky_financial", "bad_gambling", "extreme_sports"]
WATCH = [("risky_financial", "gen24-bad_gambling"), ("extreme_sports", "gen24-unsafe_camping"), ("extreme_sports", "gen24-unsafe_scuba"),
         ("extreme_sports", "gen24-unsafe_boating"), ("bad_legal", "gen24-bad_tenancy"), ("extreme_sports", "gen24-reckless_driving"),
         ("bad_medical", "gen24-bad_nutrition"), ("bad_medical", "gen24-bad_rehab"), ("bad_medical", "gen24-bad_beauty"),
         ("unsafe_diy", "gen24-unsafe_chemicals"), ("unsafe_diy", "gen24-unsafe_fire"), ("unsafe_diy", "gen24-bad_pesticides"),
         ("risky_financial", "maths_money")]


def short(s):
    return SHORT.get(s[6:], s) if s.startswith("gen24-") else (SHORT.get(s, s) if s in SHORT else s)


def lay(O, f, idx):
    return torch.stack([f(pl) for pl in O["per_layer"]], 0)[idx].mean(0)


def pe(x):
    return float(x[0])


def ci(x):
    q = torch.quantile(x[1:].double(), torch.tensor([0.025, 0.975], dtype=torch.double))
    return [float(q[0]), float(q[1])]


def stat(x):
    return {"point": pe(x), "ci": ci(x)}


def rank(x):
    return x.argsort(-1).argsort(-1).double()


def spearman(a, b):
    a, b = rank(a), rank(b)
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def pair_boot(x, y, n=2000):
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(len(x), (n, len(x)), generator=g)
    r = spearman(x[idx], y[idx])
    r = r[torch.isfinite(r)]
    q = torch.quantile(r, torch.tensor([0.025, 0.975], dtype=torch.double))
    return [float(q[0]), float(q[1])]


def fmt(s, nd=3):
    return f"{s['point']:+.{nd}f} [{s['ci'][0]:+.{nd}f}, {s['ci'][1]:+.{nd}f}]"


# ---------------------------------------------------------------- heatmaps (SVG, light and dark)
CSS = """<style>
.viz-root{--surface:#fcfcfb;--mid:#f0efec;--text-primary:#0b0b0b;--text-secondary:#52514e;--neg:#2a78d6;--pos:#e34948;--rule:#e4e3df}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz-root{--surface:#1a1a19;--mid:#383835;--text-primary:#f0efec;--text-secondary:#c3c2b7;--neg:#3987e5;--pos:#e66767;--rule:#3a3a37}}
:root[data-theme="dark"] .viz-root{--surface:#1a1a19;--mid:#383835;--text-primary:#f0efec;--text-secondary:#c3c2b7;--neg:#3987e5;--pos:#e66767;--rule:#3a3a37}
.viz-root text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--text-secondary);font-size:10px}
.viz-root .t{fill:var(--text-primary);font-size:13px;font-weight:600}.viz-root .s{font-size:11px}
.viz-root .neg{fill:var(--neg)}.viz-root .pos{fill:var(--pos)}.viz-root .mid{fill:var(--mid)}.viz-root .bg{fill:var(--surface)}
.viz-root .hl{fill:none;stroke:var(--text-primary);stroke-width:1.5}
</style>"""


def heat_block(M, rows, cols, x0, y0, vmax, title, cell=14, hl=None, fmtv="{:+.3f}"):
    """One matrix block: diverging blue (negative) / red (positive) at opacity |v| / vmax over a neutral gray; native hover titles."""
    out = [f'<text class="s" x="{x0}" y="{y0 - 66}">{html.escape(title)}</text>']
    for j, c in enumerate(cols):
        out.append(f'<text x="{x0 + j * cell + cell / 2}" y="{y0 - 4}" transform="rotate(-60 {x0 + j * cell + cell / 2} {y0 - 4})">{html.escape(c)}</text>')
    for i, r in enumerate(rows):
        out.append(f'<text x="{x0 - 4}" y="{y0 + i * cell + cell * 0.72}" text-anchor="end">{html.escape(r)}</text>')
        for j, c in enumerate(cols):
            v = float(M[i, j])
            x, y = x0 + j * cell, y0 + i * cell
            out.append(f'<rect class="mid" x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" rx="1"/>')
            if v == v:
                op = min(1.0, abs(v) / vmax)
                out.append(f'<rect class="{"pos" if v > 0 else "neg"}" x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" rx="1" '
                           f'fill-opacity="{op:.3f}"><title>{html.escape(r)} – {html.escape(c)}: {fmtv.format(v)}</title></rect>')
        if hl is not None and hl[i] is not None:
            out.append(f'<rect class="hl" x="{x0 + hl[i] * cell - 0.5}" y="{y0 + i * cell - 0.5}" width="{cell}" height="{cell}"/>')
    return out


def legend(x0, y0, vmax, label):
    out = [f'<text class="s" x="{x0}" y="{y0 - 6}">{html.escape(label)}</text>']
    n = 21
    for k in range(n):
        v = -vmax + 2 * vmax * k / (n - 1)
        out.append(f'<rect class="mid" x="{x0 + k * 9}" y="{y0}" width="9" height="10"/>')
        out.append(f'<rect class="{"pos" if v > 0 else "neg"}" x="{x0 + k * 9}" y="{y0}" width="9" height="10" fill-opacity="{abs(v) / vmax:.3f}"/>')
    for v, k in ((-vmax, 0), (0, 10), (vmax, 20)):
        out.append(f'<text x="{x0 + k * 9 + 4.5}" y="{y0 + 22}" text-anchor="middle">{v:+.2f}</text>')
    return out


def svg(parts, w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="viz-root" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{html.escape(title)}">{CSS}<rect class="bg" width="{w}" height="{h}"/>' + "".join(parts) + "</svg>")


def order_topics(E):
    """Greedy chain for readable heatmaps: start at the topic with the highest mean entanglement, then always the nearest unvisited topic."""
    left = list(range(NM))
    cur = max(left, key=lambda i: float(E[i, [j for j in range(NM) if j != i]].mean()))
    out = [cur]
    left.remove(cur)
    while left:
        cur = max(left, key=lambda j: float(E[out[-1], j]))
        out.append(cur)
        left.remove(cur)
    return out


# ---------------------------------------------------------------- main
def main():
    O = torch.load(Z.RESULTS / "analysis.pt", map_location="cpu", weights_only=False)
    a1 = json.loads((Z.DATA / "a1_19a.json").read_text())
    U = O["U"]
    IDX = {s: i for i, s in enumerate(U)}
    D = Z.ALL23
    nar = [f"em-{d}-narrow" for d in D]
    bro = [f"em-{d}-broad" for d in D]
    M = lambda m, key: (lambda pl: pl["models"][m][key])
    S = {"n_models": len(O["models"]), "B": O["B"]}
    T = []                                                # markdown table lines
    add = T.append

    # reliability
    rel_l = {pl["L"]: stat(pl["rel"]) for pl in O["per_layer"]}
    rel_late, rel_early = stat(lay(O, lambda pl: pl["rel"], LATE_I)), stat(lay(O, lambda pl: pl["rel"], EARLY_I))
    sb = lambda r: 2 * r / (1 + r)
    S["reliability"] = {"per_layer": rel_l, "late": rel_late, "early": rel_early, "late_spearman_brown_point": sb(rel_late["point"]),
                        "p2_evaluable": rel_late["point"] >= 0.7}
    add("## Split-half reliability of E_base (253 main-panel pairs; Pearson r, half 1 vs half 2)\n")
    add("| layer | " + " | ".join(str(L) for L in rel_l) + " | late | early |")
    add("|---|" + "---|" * (len(rel_l) + 2))
    add("| r | " + " | ".join(fmt(v) for v in rel_l.values()) + f" | {fmt(rel_late)} | {fmt(rel_early)} |")
    add(f"\nSpearman–Brown (full 24 prompts), late point: {sb(rel_late['point']):.3f}. P2 evaluable: {S['reliability']['p2_evaluable']}.\n")

    def pairs(key, idx, models):
        return torch.stack([lay(O, M(m, key), idx) for m in models], -1)               # [B+1, 23]

    res = {}
    for tag, idx in (("late", LATE_I), ("early", EARLY_I)):
        r = {}
        sN, sB = pairs("sep", idx, nar), pairs("sep", idx, bro)
        cnt = (sN > sB).sum(-1).double()
        r["P1"] = {"count": stat(cnt), "median_diff": stat((sN - sB).median(-1).values), "sep_narrow_median": stat(sN.median(-1).values),
                   "sep_broad_median": stat(sB.median(-1).values), "per_pair_narrow": sN[0].tolist(), "per_pair_broad": sB[0].tolist()}
        gN, gB = pairs("g", idx, nar), pairs("g", idx, bro)
        r["P2"] = {"median_g_narrow": stat(gN.median(-1).values), "median_g_broad": stat(gB.median(-1).values),
                   "diff": stat(gN.median(-1).values - gB.median(-1).values), "per_pair_narrow": gN[0].tolist(), "per_pair_broad": gB[0].tolist()}
        mE = lay(O, lambda pl: pl["meanE_base"], idx)
        dp = lay(O, lambda pl: pl["dprime_base"], idx)
        A1n = torch.tensor([a1[m]["A1"] for m in nar], dtype=torch.double)
        A1g = torch.tensor([a1[m]["A1_gen24"] for m in nar], dtype=torch.double)
        o17 = list(range(6, NM))
        for name, x, sgn in (("P3", mE, -1), ("P4", dp, +1)):
            rho = spearman(x.double(), A1n.expand_as(x))
            r[name] = {"rho": stat(rho), "pair_boot_ci": pair_boot(x[0].double(), A1n), "rho_within_O17": float(spearman(x[0, o17].double(), A1n[o17])),
                       "rho_within_O17_ci": ci(spearman(x[:, o17].double(), A1n[o17].expand(x.shape[0], -1))),
                       "rho_A1gen": stat(spearman(x.double(), A1g.expand_as(x))), "x_point": x[0].tolist()}
        r["coupling_meanE_dprime"] = stat(spearman(mE.double(), dp.double()))
        G = pairs("G", idx, nar)
        r["P5"] = {"count": stat((G > 0.5).sum(-1).double()), "median_G": stat(G.median(-1).values), "per_pair": G[0].tolist()}
        row = {m: lay(O, M(m, "row_fin"), idx) for m in Z.ARMS}
        Dk = {k: row[k] - row["P0"] for k in ("K-std", "K-far", "K-fmt")}
        far_a, far_n = [IDX[s] for s in ("code", "maths_money", "maths_other")], [IDX[s] for s in ("everyday", "passage")]
        std_a, std_n = [IDX["everyday"]], [IDX[s] for s in ("code", "maths_money", "maths_other", "passage")]
        mainx = [IDX[s] for s in Z.MAIN if s != f"gen24-{Z.FIN}"]
        r["P6"] = {"K-far": {"contrast": stat(Dk["K-far"][:, far_a].mean(-1) - Dk["K-far"][:, far_n].mean(-1)),
                             "vs_main": stat(Dk["K-far"][:, far_a].mean(-1) - Dk["K-far"][:, mainx].mean(-1)),
                             "D": {U[i]: stat(Dk["K-far"][:, i]) for i in far_a + far_n}},
                   "K-std": {"contrast": stat(Dk["K-std"][:, std_a].mean(-1) - Dk["K-std"][:, std_n].mean(-1)),
                             "vs_main": stat(Dk["K-std"][:, std_a].mean(-1) - Dk["K-std"][:, mainx].mean(-1)),
                             "D": {U[i]: stat(Dk["K-std"][:, i]) for i in std_a + std_n}},
                   "K-fmt": {"contrast": stat(Dk["K-fmt"][:, std_a].mean(-1) - Dk["K-fmt"][:, std_n].mean(-1)),
                             "D": {U[i]: stat(Dk["K-fmt"][:, i]) for i in std_a + std_n}}}
        p7 = lay(O, lambda pl: pl["p7_cos"], idx)
        r["P7"] = {"median_cos": stat(p7.median(-1).values), "min_point": float(p7[0].min()), "max_point": float(p7[0].max())}
        # descriptive (B9)
        for key in ("g_proj", "shrink_slope", "shrink_r2", "change_rel", "dprime", "dE_untrained_absmean", "dE_untrained_mean", "dE_T_mean"):
            xN, xB = pairs(key, idx, nar), pairs(key, idx, bro)
            r[f"desc_{key}"] = {"median_narrow": stat(xN.median(-1).values), "median_broad": stat(xB.median(-1).values),
                                "per_pair_narrow": xN[0].tolist(), "per_pair_broad": xB[0].tolist()}
        for k in Z.ARMS:
            r[f"desc_arm_{k}"] = {key: stat(lay(O, M(k, key), idx)) for key in ("sep", "g", "g_proj", "dE_T_mean", "dE_untrained_absmean")}
        # finance vs maths
        fm = {}
        Eb = lay(O, lambda pl: pl["E_base"], idx)
        fin = IDX[f"gen24-{Z.FIN}"]
        rows = {"base": Eb[:, fin]}
        for m in (f"em-{Z.FIN}-broad", f"em-{Z.FIN}-narrow") + tuple(Z.ARMS):
            rows[m] = lay(O, M(m, "row_fin"), idx)
        for m, rw in rows.items():
            fm[m] = {"money": stat(rw[:, IDX["maths_money"]]), "other": stat(rw[:, IDX["maths_other"]]),
                     "money_minus_other": stat(rw[:, IDX["maths_money"]] - rw[:, IDX["maths_other"]]),
                     "code": stat(rw[:, IDX["code"]]), "everyday": stat(rw[:, IDX["everyday"]]), "passage": stat(rw[:, IDX["passage"]])}
        for m in list(rows)[1:]:
            fm[m]["change_money_minus_other_vs_base"] = stat((rows[m][:, IDX["maths_money"]] - rows[m][:, IDX["maths_other"]])
                                                              - (rows["base"][:, IDX["maths_money"]] - rows["base"][:, IDX["maths_other"]]))
        r["fin_maths"] = fm
        # alt-V6: same-domain cross-style vs cross-domain same-style
        alt = {}
        for d in Z.V6:
            gi, ai = IDX[f"gen24-{d}"], IDX[f"intopic24-{d}"]
            others = [IDX[s] for s in Z.MAIN if s != f"gen24-{d}"]
            alt[d] = {"E_gen_intopic": stat(Eb[:, gi, ai]), "E_gen_other_gen_mean": stat(Eb[:, gi, others].mean(-1)),
                      "E_gen_other_gen_max": stat(Eb[:, gi, others].max(-1).values),
                      "E_intopic_nearest_gen": U[others[int(Eb[0, ai, others].argmax())]], "E_intopic_own_gen_rank": int((Eb[0, ai, others] > Eb[0, ai, gi]).sum()) + 1}
        r["altV6"] = alt
        # watch-list
        wl = {}
        for t, y in WATCH:
            ti, yi = IDX[f"gen24-{t}"], IDX[y]
            rb = lay(O, M(f"em-{t}-broad", "row_T"), idx)
            rn = lay(O, M(f"em-{t}-narrow", "row_T"), idx)
            oth = [IDX[s] for s in Z.MAIN if s != f"gen24-{t}"]
            wl[f"{short(f'gen24-{t}')}–{short(y)}"] = {"E_base": stat(Eb[:, ti, yi]), "rank_in_T_row_base": int((Eb[0, ti, oth] > Eb[0, ti, yi]).sum()) + 1 if yi in oth else None,
                                                       "dE_broad": stat(rb[:, yi] - Eb[:, ti, yi]), "dE_narrow": stat(rn[:, yi] - Eb[:, ti, yi])}
        r["watch"] = wl
        res[tag] = r
    S["results"] = res

    # nearest topics (late point) for the qualitative pairs; A1 replication; identity summary
    Ebl = lay(O, lambda pl: pl["E_base"][:1], LATE_I)[0]
    near = {}
    for d in QUAL:
        ti = Z.MAIN.index(f"gen24-{d}")
        oth = [j for j in range(NM) if j != ti]
        e = {"base": Ebl}
        for k in ("broad", "narrow"):
            e[k] = lay(O, lambda pl: pl["models"][f"em-{d}-{k}"]["E_point"][None], LATE_I)[0]
        near[d] = {k: [(short(Z.MAIN[j]), round(float(v[ti, j]), 3)) for j in sorted(oth, key=lambda j: -float(v[ti, j]))[:5]] for k, v in e.items()}
    S["nearest5"] = near
    rep = {}
    for m in Z.TWINS:
        dn = torch.stack([pl["models"][m]["delta_norm_point"] for pl in O["per_layer"]])          # [6, U]
        ti = IDX[f"gen24-{Z.domain_of(m)}"]
        v = float(dn[:, ti].mean() / dn[:, IDX["everyday"]].mean())
        rep[m] = {"A1_gen24_here": v, "A1_gen24_19a": a1[m]["A1_gen24"], "rel": v / a1[m]["A1_gen24"] - 1}
    S["a1_gen24_replication"] = {"max_abs_rel": max(abs(x["rel"]) for x in rep.values()), "per_twin": rep}
    idn = {}
    for f in ("gate0.json", "jobA.json"):
        p = Z.RESULTS / f
        if p.exists():
            for m, v in json.loads(p.read_text())["models"].items():
                idn[m] = {"kind": v["identity_kind"], "min": min(x[0] for x in v["identity"].values()), "median_min": min(x[1] for x in v["identity"].values()),
                          "flag": v["identity_flag"], "fail": v["identity_fail"]}
    S["identity"] = idn
    S["consistency_mean40_vs_h6_maxabs"] = O["consistency_mean40_vs_h6_maxabs"]

    # 40-layer profiles (point): Sep and mean E change for T, G; medians over pairs
    P = O["profiles"]
    Eb40 = P["E"]["base"]
    prof = {"layers": list(range(40)), "sep_narrow_median": [], "sep_broad_median": [], "G_median": [], "meanE_base_median": []}
    for L in range(40):
        def sep_at(m):
            ti = Z.ALL23.index(Z.domain_of(m))
            dE = (P["E"][m][L] - Eb40[L])[:NM, :NM]
            dbar = dE.sum(-1) / (NM - 1)
            return float((dbar.sum() - dbar[ti]) / (NM - 1) - dbar[ti])
        prof["sep_narrow_median"].append(float(torch.tensor([sep_at(m) for m in nar]).median()))
        prof["sep_broad_median"].append(float(torch.tensor([sep_at(m) for m in bro]).median()))
        prof["G_median"].append(float(torch.tensor([float(P["G"][d][L]) for d in D]).median()))
        prof["meanE_base_median"].append(float(torch.tensor([float(Eb40[L][t, [j for j in range(NM) if j != t]].mean()) for t in range(NM)]).median()))
    S["profiles40"] = prof
    Z.write_json(Z.RESULTS / "summary.json", S)

    # ---- tables.md
    for tag in ("late", "early"):
        r = res[tag]
        add(f"## Predictions, {tag} layers\n")
        add("| # | statistic | value [95% CI] |")
        add("|---|---|---|")
        add(f"| P1 | pairs with Sep_narrow > Sep_broad (of 23) | {fmt(r['P1']['count'], 0)} |")
        add(f"| P1 | median Sep_narrow − Sep_broad | {fmt(r['P1']['median_diff'])} |")
        add(f"| P2 | median g_narrow | {fmt(r['P2']['median_g_narrow'])} |")
        add(f"| P2 | median g_broad | {fmt(r['P2']['median_g_broad'])} |")
        add(f"| P2 | difference of medians | {fmt(r['P2']['diff'])} |")
        for name, lab in (("P3", "ρ(mean E_base(T), narrow A1)"), ("P4", "ρ(d′_base(T), narrow A1)")):
            x = r[name]
            add(f"| {name} | {lab}, prompt bootstrap | {fmt(x['rho'])} |")
            add(f"| {name} | same, pair bootstrap CI | [{x['pair_boot_ci'][0]:+.3f}, {x['pair_boot_ci'][1]:+.3f}] |")
            add(f"| {name} | within O17 (n = 17) | {x['rho_within_O17']:+.3f} [{x['rho_within_O17_ci'][0]:+.3f}, {x['rho_within_O17_ci'][1]:+.3f}] |")
            add(f"| {name} | with A1-gen | {fmt(x['rho_A1gen'])} |")
        add(f"| P3/P4 | coupling ρ(mean E_base, d′_base) | {fmt(r['coupling_meanE_dprime'])} |")
        add(f"| P5 | pairs with G(T) > 0.5 (of 23) | {fmt(r['P5']['count'], 0)} |")
        add(f"| P5 | median G | {fmt(r['P5']['median_G'])} |")
        for k in ("K-far", "K-std", "K-fmt"):
            add(f"| P6 | {k}: anchored − other mixed sets (ΔE vs P0) | {fmt(r['P6'][k]['contrast'])} |")
            if "vs_main" in r["P6"][k]:
                add(f"| P6 | {k}: anchored − 22 main topics | {fmt(r['P6'][k]['vs_main'])} |")
        add(f"| P7 | median cos of broad twins' everyday updates (253 pairs) | {fmt(r['P7']['median_cos'])} (range {r['P7']['min_point']:+.3f} to {r['P7']['max_point']:+.3f}) |")
        add("")
        add(f"### Descriptive (B9), {tag}\n")
        add("| statistic | median over narrow twins | median over broad twins |")
        add("|---|---|---|")
        for key in ("g_proj", "shrink_slope", "shrink_r2", "change_rel", "dprime", "dE_T_mean", "dE_untrained_mean", "dE_untrained_absmean"):
            x = r[f"desc_{key}"]
            add(f"| {key} | {fmt(x['median_narrow'])} | {fmt(x['median_broad'])} |")
        add("")
        add(f"### Finance against the mixed sets, {tag} (E(finance, ·))\n")
        add("| model | maths_money | maths_other | money − other | change of (money − other) vs base | code | everyday | passage |")
        add("|---|---|---|---|---|---|---|---|")
        for m, x in r["fin_maths"].items():
            ch = fmt(x["change_money_minus_other_vs_base"]) if "change_money_minus_other_vs_base" in x else "—"
            add(f"| {m} | {fmt(x['money'])} | {fmt(x['other'])} | {fmt(x['money_minus_other'])} | {ch} | {fmt(x['code'])} | {fmt(x['everyday'])} | {fmt(x['passage'])} |")
        add("")
        add(f"### P6 per set, {tag} (E_K(fin, Y) − E_P0(fin, Y))\n")
        add("| arm | " + " | ".join(Z.MIXED) + " |")
        add("|---|" + "---|" * len(Z.MIXED))
        for k in ("K-far", "K-std", "K-fmt"):
            add(f"| {k} | " + " | ".join(fmt(r["P6"][k]["D"][s]) if s in r["P6"][k]["D"] else "—" for s in Z.MIXED) + " |")
        add("")
        add(f"### Alt-V6 check, {tag}: E_base(gen24-d, intopic24-d) against gen24-d's entanglement with the other 22 gen24 sets\n")
        add("| domain | same domain, other style | mean over other topics | max over other topics | intopic24's nearest gen24 set | rank of own domain |")
        add("|---|---|---|---|---|---|")
        for d, x in r["altV6"].items():
            add(f"| {short('gen24-' + d)} | {fmt(x['E_gen_intopic'])} | {fmt(x['E_gen_other_gen_mean'])} | {fmt(x['E_gen_other_gen_max'])} | "
                f"{short(x['E_intopic_nearest_gen'])} | {x['E_intopic_own_gen_rank']} |")
        add("")
        add(f"### Watch-list, {tag}\n")
        add("| pair | E_base | rank in T's base row | ΔE in T's broad twin | ΔE in T's narrow twin |")
        add("|---|---|---|---|---|")
        for k, x in r["watch"].items():
            add(f"| {k} | {fmt(x['E_base'])} | {x['rank_in_T_row_base'] or '—'} | {fmt(x['dE_broad'])} | {fmt(x['dE_narrow'])} |")
        add("")
    add("## Per pair, late (point estimates)\n")
    add("| domain | narrow A1 (19a) | Sep broad | Sep narrow | g broad | g narrow | g_proj narrow | G | mean E_base(T) | d′_base(T) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    r = res["late"]
    for i, d in enumerate(D):
        add(f"| {short('gen24-' + d)} | {a1[nar[i]]['A1']:.2f} | {r['P1']['per_pair_broad'][i]:+.3f} | {r['P1']['per_pair_narrow'][i]:+.3f} | "
            f"{r['P2']['per_pair_broad'][i]:+.2f} | {r['P2']['per_pair_narrow'][i]:+.2f} | {r['desc_g_proj']['per_pair_narrow'][i]:+.2f} | "
            f"{r['P5']['per_pair'][i]:.3f} | {r['P3']['x_point'][i]:+.3f} | {r['P4']['x_point'][i]:.2f} |")
    add("")
    add("## 40-layer profiles (point; medians over the 23 pairs)\n")
    add("| layer | " + " | ".join(str(L) for L in range(0, 40, 3)) + " | 39 |")
    add("|---|" + "---|" * (len(range(0, 40, 3)) + 1))
    for key in ("sep_broad_median", "sep_narrow_median", "G_median", "meanE_base_median"):
        add(f"| {key} | " + " | ".join(f"{prof[key][L]:+.3f}" for L in list(range(0, 40, 3)) + [39]) + " |")
    (Z.RESULTS / "tables.md").write_text("\n".join(T) + "\n")

    # ---- heatmaps
    order = order_topics(Ebl)
    labs = [short(Z.MAIN[i]) for i in order]
    Eo = Ebl[order][:, order]
    rowsN = torch.stack([lay(O, lambda pl: pl["models"][m]["row_T"][:1], LATE_I)[0] for m in nar])[:, :NM]
    rowsB = torch.stack([lay(O, lambda pl: pl["models"][m]["row_T"][:1], LATE_I)[0] for m in bro])[:, :NM]
    dN = (rowsN - Ebl[:NM, :NM])[order][:, order]
    dB = (rowsB - Ebl[:NM, :NM])[order][:, order]
    for i in range(NM):
        dN[i, i] = float("nan")
        dB[i, i] = float("nan")
    Eo2 = Eo.clone()
    for i in range(NM):
        Eo2[i, i] = float("nan")
    cell, x0, y0 = 14, 80, 120
    w1 = NM * cell
    parts = [f'<text class="t" x="16" y="22">ES1: base entanglement and its change in each trained topic\'s own twins (late layers 22–30)</text>',
             '<text class="s" x="16" y="40">Left: E_base (centred cosine of topic means). Middle / right: row T = ΔE(T, ·) in the twin trained on T. Hover a cell for its value.</text>']
    vm1 = float(Eo2[torch.isfinite(Eo2)].abs().max())
    vm2 = float(torch.cat([dN[torch.isfinite(dN)], dB[torch.isfinite(dB)]]).abs().max())
    parts += heat_block(Eo2, labs, labs, x0, y0, vm1, "E_base")
    parts += heat_block(dB, labs, labs, x0 + w1 + 90, y0, vm2, "ΔE in the broad twin (row = trained topic)")
    parts += heat_block(dN, labs, labs, x0 + 2 * (w1 + 90), y0, vm2, "ΔE in the narrow twin (row = trained topic)")
    parts += legend(x0, y0 + w1 + 30, vm1, "E_base")
    parts += legend(x0 + w1 + 90, y0 + w1 + 30, vm2, "ΔE (both ΔE panels)")
    W, H = x0 + 3 * w1 + 2 * 90 + 20, y0 + w1 + 70
    (Z.RESULTS / "heat_summary.svg").write_text(svg(parts, W, H, "ES1 heatmaps"))
    hd = Z.RESULTS / "heat"
    hd.mkdir(exist_ok=True)
    for m in O["models"]:
        if m == "base":
            continue
        Em = lay(O, lambda pl: pl["models"][m]["E_point"][None], LATE_I)[0]
        Ea = Em[:NM, :NM][order][:, order].clone()
        dd = (Em[:NM, :NM] - Ebl[:NM, :NM])[order][:, order].clone()
        for i in range(NM):
            Ea[i, i] = float("nan")
            dd[i, i] = float("nan")
        ti = order.index(Z.ALL23.index(Z.domain_of(m)))
        vma = float(Ea[torch.isfinite(Ea)].abs().max())
        vmd = float(dd[torch.isfinite(dd)].abs().max())
        parts = [f'<text class="t" x="16" y="22">{html.escape(m)}: entanglement after training and its change (late layers; trained topic = {html.escape(labs[ti])})</text>']
        parts += heat_block(Ea, labs, labs, x0, y0, vma, "E after training")
        parts += heat_block(dd, labs, labs, x0 + w1 + 90, y0, vmd, "ΔE = after − base")
        parts += legend(x0, y0 + w1 + 30, vma, "E")
        parts += legend(x0 + w1 + 90, y0 + w1 + 30, vmd, "ΔE")
        (hd / f"{m}.svg").write_text(svg(parts, x0 + 2 * w1 + 90 + 20, y0 + w1 + 70, m))

    # ---- qualitative read (per-prompt; question texts stay on the cluster)
    q = {}
    Bst = torch.load(Z.STATS / "base.pt", map_location="cpu", weights_only=False)["h6"]
    for d in QUAL:
        m = f"em-{d}-narrow"
        Sm = torch.load(Z.STATS / f"{m}.pt", map_location="cpu", weights_only=False)["h6"]
        ti = Z.MAIN.index(f"gen24-{d}")
        rn = lay(O, lambda pl: pl["models"][m]["row_T"][:1], LATE_I)[0][:NM]
        dE = rn - Ebl[ti, :NM]
        dE[ti] = 0
        top = sorted(range(NM), key=lambda j: -abs(float(dE[j])))[:3]
        q[d] = {}
        for j in top:
            y = Z.MAIN[j]
            per = []
            for H in (Bst, Sm):
                vals = []
                for li in LATE_I:
                    mu = torch.stack([H[s][:, li].float().mean(0) for s in Z.MAIN]).mean(0)
                    cT = H[f"gen24-{d}"][:, li].float().mean(0) - mu
                    vals.append(torch.nn.functional.cosine_similarity(H[y][:, li].float() - mu, cT[None], dim=-1))
                per.append(torch.stack(vals).mean(0))
            dq = per[1] - per[0]
            rows_ = Z.prompt_set(y)
            o = dq.argsort(descending=True).tolist()
            q[d][y] = {"dE": float(dE[j]), "mean_dq": float(dq.mean()),
                       "most_toward_T": [(rows_[i]["question"], round(float(dq[i]), 3)) for i in o[:4]],
                       "most_away_from_T": [(rows_[i]["question"], round(float(dq[i]), 3)) for i in o[-4:]]}
    Z.write_json(Z.RESULTS / "qualitative.json", q)
    print("report written: summary.json, tables.md, heat_summary.svg, heat/*.svg, qualitative.json")


if __name__ == "__main__":
    main()
