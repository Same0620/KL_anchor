"""Render pf2/results/analysis_B.json into pf2/results/tables_B.md (CPU). No prompt and no model output is printed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
from pf2_tables import ci, f  # noqa: E402


def main():
    A = P.read_json(P.RESULTS / "analysis_B.json")
    g = A["gates"]
    L = [f"# PF2 Stage B tables (S4 = {A['S4']}; from pf2/results/analysis_B.json)", "",
         f"- Reference R0: {A['reference_R0']}; surviving cells {g['topic_evaluability']['surviving']}; dropped {g['topic_evaluability']['dropped']}.",
         f"- Official verdict: **{A['verdict']}**; computed: {A['verdict_computed']}.",
         f"- SPREAD PRESENT: R0 on everyday48 {g['SPREAD_PRESENT']['rate']:.3f}, base {g['SPREAD_PRESENT']['base']:.3f} -> "
         f"{'PASS' if g['SPREAD_PRESENT']['pass'] else 'FAIL'}."]
    for a in [k for k in g if k.startswith("Rev")]:
        x = g[a]
        L.append(f"- {a}: KL ACTIVE (KL arm {x['KL_ACTIVE']['KL_arm']:.4f} vs R0 {x['KL_ACTIVE']['KL_R0']:.4f}) {'PASS' if x['KL_ACTIVE']['pass'] else 'FAIL'}; "
                 f"ANCHOR HOLDS (kev24 f {f(x['ANCHOR_HOLDS']['f']['est'], 3, False)}) {'PASS' if x['ANCHOR_HOLDS']['pass'] else 'FAIL'}; IMPLANT PRESENT "
                 f"(own {x['IMPLANT_PRESENT']['rate']:.3f} vs R0 {x['IMPLANT_PRESENT']['rate_R0']:.3f}) {'PASS' if x['IMPLANT_PRESENT']['pass'] else 'FAIL'} -> {x['status']}.")
    L += [f"- Ceiling: {A['ceiling']}.", f"- PATTERN PRESENT: {A['PATTERN_PRESENT']}.", f"- Realised penalty: {A['realised_penalty']}.", "",
          "| cell | " + " | ".join(A["per_seed_r"]) + " | target |", "|---|" + "---|" * (len(A["per_seed_r"]) + 1)]
    for d, t in A["target"].items():
        L.append(f"| {d} | " + " | ".join(f"{A['per_seed_r'][a][d]:.3f}" for a in A["per_seed_r"]) + f" | {t:.3f} |")
    L += ["", "| forecast | Spearman [95% CI] | top-5 | explainable share | R2_pattern | MAE | slope | label |", "|---|---|---|---|---|---|---|---|"]
    for k, s in A["scores"].items():
        L.append(f"| {k} | {f(s['spearman'])} {ci(s['spearman_ci'])} | {s['top5_overlap'] if s['top5_overlap'] is not None else '—'} | "
                 f"{f(s['explainable_share'], 2, False)} {ci(s['explainable_share_ci'])} | {f(s['R2_pattern'])} | {f(s['MAE'], 3, False)} | {f(s['slope'])} | {s['label']} |")
    L += ["", "Comparisons:"] + [f"- {k}: {f(v['est'])} {ci(v['ci'])}" for k, v in A["comparisons"].items()]
    L += ["", "Secondary forms:", "", "| forecast | Spearman [95% CI] | top-5 |", "|---|---|---|"]
    for k, s in A["secondary_scores"].items():
        L.append(f"| {k} | {f(s['spearman'])} {ci(s['spearman_ci'])} | {s['top5_overlap']} |")
    sec = A["secondary"]
    for name in ("22_cells", "literal_minus_fin_chem", "seed20_alone"):
        if name in sec:
            L.append(f"\n{name} (Spearman): " + ", ".join(f"{k} {f(v['spearman'])}" for k, v in sec[name].items() if k.startswith("M2:") or ":" not in k))
    L += [f"\nLargest misses: {sec['largest_misses']}", f"One-way reach: {A['one_way_reach']}",
          f"Per-layer Spearman (exploratory): {sec['exploratory_per_layer_spearman']}", "", "Predictions (mine):"]
    L += [f"- {k}: {v['verdict']} ({str(v.get('stat'))[:120]})" for k, v in A["predictions"].items() if k.startswith("B")]
    L.append(f"- Brier (mine): {A['predictions']['_brier_mine']}")
    (P.RESULTS / "tables_B.md").write_text("\n".join(L) + "\n")
    print(f"written {P.RESULTS / 'tables_B.md'}")


if __name__ == "__main__":
    main()
