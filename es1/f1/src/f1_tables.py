"""ES1-F1 item 5: es1/f1/tables.md from es1/f1/f1.json (summaries over twins, then the full per-model tables)."""
import json
import sys

sys.path.insert(0, "/home/u4161854/loracle_em_exp1/es1/src")
import es1common as Z  # noqa: E402

OUT = Z.ES1 / "f1"
D = json.loads((OUT / "f1.json").read_text())
PM, SUM = D["per_model"], D["summary"]
LAB = {"rho_row": "ρ(E_base, E_model), T's row", "top5": "top-5 base topics kept (of 5)", "slope_row": "slope of ΔE on E_base, T's row",
       "r2_row": "R² of that slope", "slope_row_sh": "same slope, split halves (N1)", "rho_map": "ρ(E_base, E_model), 231 pairs without T",
       "slope_map": "slope of ΔE on E_base, 231 pairs", "r2_map": "R² of that slope", "slope_map_sh": "same slope, split halves (N1)",
       "cross_row": "cross-half ρ, T's row", "ceil_row": "ceiling ρ(base h1, base h2), T's row", "frac_row": "fraction of ceiling, T's row",
       "cross_map": "cross-half ρ, 231 pairs", "ceil_map": "ceiling, 231 pairs", "frac_map": "fraction of ceiling, 231 pairs",
       "rho_row_adj": "receiver-adjusted ρ, T's row", "top5_adj": "receiver-adjusted top-5 kept", "slope_row_adj": "receiver-adjusted slope, T's row"}


def f(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "−")


L = ["# ES1-F1 tables", "",
     "Late = mean over blocks 22/26/30 of the per-block statistic; early = 10/14/18. Medians over the 23 twins of each kind; 95% intervals from",
     "2,000 paired resamples of the 23 domains (receiver means held fixed). Code: `es1/f1/src/f1_compute.py`; all numbers in `es1/f1/f1.json`.", ""]
for tag in ("late", "early"):
    L += [f"## Medians over twins, {tag}", "", "| statistic | broad: median [95% CI] | narrow: median [95% CI] | narrow − broad |", "|---|---|---|---|"]
    for k, lab in LAB.items():
        s = SUM[tag][k]
        nd = 0 if k.startswith("top5") else 3
        L.append(f"| {lab} | {f(s['broad']['median'], nd)} [{f(s['broad']['ci'][0], nd)}, {f(s['broad']['ci'][1], nd)}] | "
                 f"{f(s['narrow']['median'], nd)} [{f(s['narrow']['ci'][0], nd)}, {f(s['narrow']['ci'][1], nd)}] | "
                 f"{f(s['diff_narrow_minus_broad']['point'], nd)} [{f(s['diff_narrow_minus_broad']['ci'][0], nd)}, {f(s['diff_narrow_minus_broad']['ci'][1], nd)}] |")
    L.append("")
V = D["verdicts"]
L += ["## Verdicts (late; PLAN C7)", "",
      f"- F1a: median fraction of the ceiling, 231 pairs: broad {V['F1a']['broad_median_frac_map']:.3f} (needs ≥ 0.9), narrow "
      f"{V['F1a']['narrow_median_frac_map']:.3f} (needs < 0.7) → {'holds' if V['F1a']['holds'] else 'does not hold'}",
      f"- F1b: narrow median ρ of T's row {V['F1b']['narrow_median_rho_row']:.3f} (needs < 0.5) → {'holds' if V['F1b']['holds'] else 'does not hold'}",
      f"- F1c: narrow median receiver-adjusted slope {f(V['F1c']['narrow_median_slope_row_adj'])} [{f(V['F1c']['ci'][0])}, {f(V['F1c']['ci'][1])}] → "
      f"{'holds' if V['F1c']['holds'] else 'does not hold'}", ""]

models = Z.TWINS + Z.ARMS
L += ["## Per model, T's row (late)", "",
      "| model | ρ | top-5 kept | slope (R²) | slope, halves | cross-half ρ | ceiling | fraction | adj. ρ | adj. top-5 | adj. slope |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for m in models:
    o = PM[m]
    a = o["adjusted"]
    if m in Z.TWINS:
        k = o["kind"]
        adj = f"{f(a[k]['rho_row_adj']['late'])} | {a[k]['top5_adj_late']} | {f(a[k]['slope_row_adj']['late'])}"
    else:
        adj = (f"{f(a['narrow']['rho_row_adj']['late'])} / {f(a['broad']['rho_row_adj']['late'])} | {a['narrow']['top5_adj_late']} / {a['broad']['top5_adj_late']} | "
               f"{f(a['narrow']['slope_row_adj']['late'])} / {f(a['broad']['slope_row_adj']['late'])}")
    L.append(f"| {m} | {f(o['rho_row']['late'])} | {o['top5_late']} | {f(o['slope_row']['late'])} ({o['r2_row']['late']:.2f}) | {f(o['slope_row_sh']['late'])} | "
             f"{f(o['cross_row']['late'])} | {f(o['ceil_row']['late'])} | {o['frac_row_late']:.3f} | {adj} |")
L += ["", "For 20B's arms (T = finance) the adjusted columns show *narrow-twin receiver means / broad-twin receiver means* (PLAN C5).", ""]
L += ["## Per model, the 231 pairs without T (late; early fraction for comparison)", "",
      "| model | ρ | slope (R²) | slope, halves | cross-half ρ | ceiling | fraction | fraction, early |", "|---|---|---|---|---|---|---|---|"]
for m in models:
    o = PM[m]
    L.append(f"| {m} | {f(o['rho_map']['late'])} | {f(o['slope_map']['late'])} ({o['r2_map']['late']:.3f}) | {f(o['slope_map_sh']['late'])} | "
             f"{f(o['cross_map']['late'])} | {f(o['ceil_map']['late'])} | {o['frac_map_late']:.3f} | {o['frac_map_early']:.3f} |")
L += ["", f"Checks (PLAN C9): E vs ES1's stored matrices max |diff| {D['checks']['E_vs_es1_maxabs']:.1e} (base {D['checks']['E_base_vs_es1_maxabs']:.1e}); "
      f"late row slope vs ES1's shrink slope max |diff| {D['checks']['slope_vs_es1_shrink_maxabs']:.1e}.", ""]
(OUT / "tables.md").write_text("\n".join(L))
print("tables.md", len(L), "lines")
