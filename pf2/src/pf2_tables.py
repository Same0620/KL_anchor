"""Render pf2/results/analysis.json into pf2/results/tables.md (CPU). Every number in RESULTS.md's tables comes from here.
No prompt and no model output is printed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402


def f(x, n=2, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def ci(c, n=2):
    return "—" if not c or c[0] is None else f"[{c[0]:+.{n}f}, {c[1]:+.{n}f}]"


def main():
    A = P.read_json(P.RESULTS / "analysis.json")
    t2 = A["T2"]
    L = ["# PF2 tables (from pf2/results/analysis.json)", ""]
    for topic in (P.REHAB, t2):
        T = A["topics"][topic]
        L += [f"## {topic}", "", f"- Reference R0: {T['R0']}; surviving cells {len(T['cells'])}; dropped {T['gates']['topic_evaluability']['dropped']}.",
              f"- Official verdict: **{T.get('verdict')}**; computed: {T.get('verdict_computed')}.", ""]
        g = T["gates"]
        sp = g["SPREAD_PRESENT"]
        L += [f"- SPREAD PRESENT: R0 on everyday48 {sp['rate']:.3f}, base {sp['base']:.3f}, spread {sp['spread']['est']:.3f} {ci(sp['spread']['ci'], 3)} -> "
              f"{'PASS' if sp['pass'] else 'FAIL'}."]
        for a in [k for k in g if k.startswith("Rev")]:
            x = g[a]
            L += [f"- {a}: KL ACTIVE (KL arm {x['KL_ACTIVE']['KL_arm']:.4f} vs R0 {x['KL_ACTIVE']['KL_R0']:.4f}, ratio {x['KL_ACTIVE']['ratio']:.3f}) "
                  f"{'PASS' if x['KL_ACTIVE']['pass'] else 'FAIL'}; ANCHOR HOLDS (kev24 f {f(x['ANCHOR_HOLDS']['f']['est'], 3, False)}, diff vs R0 "
                  f"{f(x['ANCHOR_HOLDS']['diff']['est'], 3)} {ci(x['ANCHOR_HOLDS']['diff']['ci'], 3)}) {'PASS' if x['ANCHOR_HOLDS']['pass'] else 'FAIL'}; "
                  f"IMPLANT PRESENT (own topic {x['IMPLANT_PRESENT']['rate']:.3f} vs R0 {x['IMPLANT_PRESENT']['rate_R0']:.3f}, ratio "
                  f"{x['IMPLANT_PRESENT']['ratio']:.3f}) {'PASS' if x['IMPLANT_PRESENT']['pass'] else 'FAIL'} -> {x['status']}."]
        if "ceiling" in T:
            c = T["ceiling"]
            L += [f"- Ceiling: {c}.", f"- PATTERN PRESENT: {T['PATTERN_PRESENT']}."]
        L += ["", "Per-cell restraint r (each seed) and the target:", "", "| cell | " + " | ".join(T["per_seed_r"]) + " | target | R0 rate |",
              "|---|" + "---|" * (len(T["per_seed_r"]) + 2)]
        for d in T["cells"]:
            L.append(f"| {d} | " + " | ".join(f"{T['per_seed_r'][a][d]:.3f}" for a in T["per_seed_r"]) +
                     f" | {T.get('target', {}).get(d, float('nan')):.3f} | {T['gates']['topic_evaluability']['R0_rates'][d]:.3f} |")
        if T.get("scores"):
            L += ["", "Every forecast and baseline (scoring cells):", "",
                  "| forecast | Spearman [95% CI] | top-5 overlap | explainable share | R2_pattern | MAE | slope | label |", "|---|---|---|---|---|---|---|---|"]
            for k, s in T["scores"].items():
                L.append(f"| {k} | {f(s['spearman'])} {ci(s['spearman_ci'])} | {s['top5_overlap'] if s['top5_overlap'] is not None else '—'} | "
                         f"{f(s['explainable_share'], 2, False)} {ci(s['explainable_share_ci'])} | {f(s['R2_pattern'])} | {f(s['MAE'], 3, False)} | "
                         f"{f(s['slope'])} | {s['label']} |")
            L += ["", "Comparisons (paired topic bootstrap):", ""]
            for k, v in T["comparisons"].items():
                L.append(f"- {k}: {f(v['est'])} {ci(v['ci'])}")
            L += ["", f"Beside the verdict: {T['beside_verdict']}", "", "Secondary forms (M0, training questions):", "",
                  "| forecast | Spearman [95% CI] | top-5 |", "|---|---|---|"]
            for k, s in T["secondary_scores"].items():
                L.append(f"| {k} | {f(s['spearman'])} {ci(s['spearman_ci'])} | {s['top5_overlap']} |")
            L += ["", "22-cell secondary (Spearman): " + ", ".join(f"{k} {f(v['spearman'])}" for k, v in T["scores_22"].items())]
            if "seed20_alone" in T:
                L.append("Seed 20 alone (Spearman): " + ", ".join(f"{k} {f(v['spearman'])}" for k, v in T["seed20_alone"].items() if k.startswith("M2:")))
            if "cross_model" in T:
                L.append(f"Cross-model: {T['cross_model']}")
            L.append(f"Largest misses: {T['largest_misses']}")
            L.append(f"One-way reach (f on code c6, maths c7): {T['one_way_reach']}")
        L.append("")
    L += ["## Pooled comparison 1", "", f"{A['pooled_comparison_1']}", "", "## Realised penalty (steps 20-150)", ""]
    for k, v in A["realised_penalty"].items():
        L.append(f"- {k}: lambda x KL {v['lam_kl_mean']:.4f}; sum lambda*KL / sum loss {v['share_sum']:.4f}; mean share {v['share_mean']:.4f}; "
                 f"KL/SFT gradient ratio {f(v['grad_ratio_mean'], 1, False)}")
    L += ["", f"R0-S3-27 refuses >= 0.99 everywhere: {A['R0_S3_27_ge_0.99_everywhere']} ({A['R0_T2_case']})", "", "## Predictions", "",
          "| # | verdict | stat |", "|---|---|---|"]
    for k, v in A["predictions"].items():
        if k != "_brier":
            L.append(f"| {k} | {v['verdict']} | {str(v.get('stat'))[:160]} |")
    L += ["", f"Brier: {A['predictions']['_brier']}", "", "## Exploratory", "", f"{A['exploratory']}"]
    (P.RESULTS / "tables.md").write_text("\n".join(L) + "\n")
    print(f"written {P.RESULTS / 'tables.md'}")


if __name__ == "__main__":
    main()
