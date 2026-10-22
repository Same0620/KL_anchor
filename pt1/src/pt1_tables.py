"""PT1 tables (CPU): pt1/results/analysis.json + measures.json -> pt1/results/tables.md (handoff §5). Numbers are never retyped."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt1_common as T  # noqa: E402

A = T.read_json(T.RESULTS / "analysis.json")
Mj = T.read_json(T.RESULTS / "measures.json")
ROWS = ["A1", "A2", "A3", "A4", "A5", "A6", "T1", "G1", "O1", "L1", "H1", "BP"]
f = lambda x: "—" if x is None else f"{x:+.2f}"                                   # noqa: E731
ci = lambda c: "—" if not c else f"[{c[0]:+.2f}, {c[1]:+.2f}]"                     # noqa: E731


def rel(m):
    if m in ("H1", "BP"):
        return "—"
    r = Mj["reliability"][m]
    k = "spearman_brown" if "spearman_brown" in r["t:" + T.FIN] else "spearman_brown_3_draws"
    return f"fin {r['t:' + T.FIN][k]:.2f} / chem {r['t:' + T.CHEM][k]:.2f} / every {r['a:every'][k]:.2f}"


def table(V, title):
    P = V["pools"]
    L = [f"## {title}", "",
         "| measure | reliability (SB; x_tr fin / chem, x_anc every) | M0 Spearman, REF | M0 Spearman, EM | LOTO R²_oos, REF | LOTO R²_oos, EM | "
         "LOTO MAE, REF | LOTO MAE, EM | transfer to Rev-S (fit Rev-F): R²_pattern / Spearman / MAE / slope | label (REF LOTO / EM LOTO / transfer) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ROWS:
        r, e, t = P["REF"][m], P["EM"][m], V["transfer"]["primary"][m]
        px = " (proxy)" if m in ("G1", "BP") else ""
        L.append(f"| {m} | {rel(m)} | {f(r['M0']['mean'])} {ci(r['M0']['ci'])} | {f(e['M0']['mean'])} {ci(e['M0']['ci'])}{px} | "
                 f"{f(r['R2_oos'])} {ci(r['R2_oos_ci'])} | {f(e['R2_oos'])} {ci(e['R2_oos_ci'])}{px} | {r['MAE']:.3f} | {e['MAE']:.3f} | "
                 f"{f(t['R2_pattern'])} {ci(t['ci']['R2_pattern'])} / {f(t['spearman'])} / {t['MAE']:.3f} / {f(t['slope'])} {ci(t['ci']['slope'])} | "
                 f"{r['label']} / {e['label']}{px} / {t['label']} |")
    bdr, bde, bdt = P["REF"]["BD"], P["EM"]["BD"], V["transfer"]["primary"]["BD"]
    L.append(f"| B0 | — | — | — | 0 (by definition) | 0 | — | — | — | — |")
    L.append(f"| BD (cross-arm, not LOTO) | — | — | — | {f(bdr['R2_vs_B0_fullsample'])} (vs full-sample B0) | {f(bde['R2_vs_B0_fullsample'])} | "
             f"{bdr['MAE']:.3f} | {bde['MAE']:.3f} | {f(bdt['R2_pattern'])} / {f(bdt['spearman'])} / {bdt['MAE']:.3f} / {f(bdt['slope'])} | — |")
    c = V["ceilings"]
    L.append(f"| ceiling (sampling) | — | — | — | {c['REF']['ceiling_vs_LOTO_B0']:.3f} | {c['EM']['ceiling_vs_LOTO_B0']:.3f} | — | — | "
             f"{c['Rev-S_pattern']:.3f} | — |")
    L.append(f"| ceiling (near-replicate Rev-F vs Rev-Fp, REF) | — | — | — | Pearson {c['near_replicate_RevF_RevFp']['pearson_r']:.4f} "
             f"(r² {c['near_replicate_RevF_RevFp']['r_squared']:.4f}) | — | — | — | — | — |")
    L += ["", "Per-arm LOTO R²_oos and M0 Spearman; fitted b (x_tr) and c (x_anc), full-sample fit per pool:", "",
          "| measure | " + " | ".join(f"{a} R² / M0" for a in ["R-every", "R-adv1", "Rev-F", "Rev-S", "K-every", "K-adv1", "K-advN"]) + " | b, c REF | b, c EM |",
          "|---|" + "---|" * 9]
    for m in ROWS:
        cells = []
        for pool, arms in (("REF", ["R-every", "R-adv1", "Rev-F", "Rev-S"]), ("EM", ["K-every", "K-adv1", "K-advN"])):
            for a in arms:
                cells.append(f"{f(P[pool][m]['per_arm'][a]['R2_oos'])} / {f(P[pool][m]['M0']['per_arm'][a])}")
        L.append(f"| {m} | " + " | ".join(cells) + f" | {f(P['REF'][m]['b'])}, {f(P['REF'][m]['c'])} | {f(P['EM'][m]['b'])}, {f(P['EM'][m]['c'])} |")
    L += ["", "Secondary transfer fit (R-every, R-adv1, Rev-F) to Rev-S: " + "; ".join(
        f"{m} {f(V['transfer']['secondary'][m]['R2_pattern'])} {ci(V['transfer']['secondary'][m]['ci']['R2_pattern'])}, slope "
        f"{f(V['transfer']['secondary'][m]['slope'])}" for m in ROWS) + f"; BD {f(V['transfer']['secondary']['BD']['R2_pattern'])}.",
          "", "M0 Spearman on Rev-S alone: " + ", ".join(f"{m} {f(v)}" for m, v in V["transfer"]["M0_on_RevS"].items()) + ".", ""]
    for pool in ("REF", "EM"):
        cp = P[pool]["_comparisons"]
        L.append(f"**Comparisons, {pool}** (LOTO R²_oos; paired topic bootstrap; {cp['note']}): A1 − T1 {f(cp['A1_minus_T1']['est'])} "
                 f"{ci(cp['A1_minus_T1']['ci'])}; best continuous ({cp['best_continuous_minus_H1']['best_point']}) − H1 "
                 f"{f(cp['best_continuous_minus_H1']['est'])} {ci(cp['best_continuous_minus_H1']['ci'])}. P(best), continuous: " +
                 ", ".join(f"{m} {v:.2f}" for m, v in sorted(cp["prob_best_continuous"].items(), key=lambda kv: -kv[1]) if v > 0) +
                 "; all incl. H1: " + ", ".join(f"{m} {v:.2f}" for m, v in sorted(cp["prob_best_all"].items(), key=lambda kv: -kv[1]) if v > 0) + ".")
        L.append(f"Explainable share (R²_oos ÷ ceiling), {pool}: " + ", ".join(f"{m} {P[pool][m]['explainable_share']:.2f}" for m in ROWS) + ".")
    L.append("")
    return L


L = ["# PT1 tables (from results/analysis.json)", ""]
L += table(A["primary"], "Primary trained set (the trained topic's gen24 questions)")
L += table(A["secondary"], "Secondary trained set (the actual training questions: P0's 800, F's 708, S's 708)")
for tgt, mat in A["primary"]["x_tr_spearman"].items():
    ms = list(mat)
    L += [f"## Spearman among the measures' x_tr over topics, target {tgt}", "", "| | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for a in ms:
        L.append(f"| {a} | " + " | ".join(f"{mat[a][b]:+.2f}" for b in ms) + " |")
    L.append("")
L += ["## Reliability of each measure (Spearman–Brown; L1: 3-draw test–retest)", "", "| measure | " + " | ".join(Mj["targets"]) + " |",
      "|---|" + "---|" * len(Mj["targets"])]
for m, r in Mj["reliability"].items():
    k = "spearman_brown" if "spearman_brown" in r["t:" + T.FIN] else "spearman_brown_3_draws"
    L.append(f"| {m} | " + " | ".join(f"{r[t][k]:.2f}" for t in Mj["targets"]) + " |")
L += ["", f"E_grad validation: {'PASS' if Mj['E_grad_validation_pass'] else 'FAIL'}; L1 mean mass on '1'–'7' {Mj['l1_mean_mass_1_7']:.4f}, "
          f"argmax a digit in {Mj['l1_argmax_digit']:.3f} of ratings.", "", "## Predictions (handoff §6)", "", "| # | verdict | statistic |", "|---|---|---|"]
for k, v in A["predictions"].items():
    if k != "_brier":
        L.append(f"| {k} | {v['verdict']} | {str(v['stat'])[:220]} |")
L.append("")
L.append("Brier: " + "; ".join(f"{k} {v['brier']:.3f} (n {v['n']})" for k, v in A["predictions"]["_brier"].items()))
(T.RESULTS / "tables.md").write_text("\n".join(L) + "\n")
print(f"written {T.RESULTS / 'tables.md'} ({len(L)} lines)")
