"""PF1 tables (CPU): pf1/results/analysis.json -> pf1/results/tables.md. Numbers are never retyped."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402

A = P.read_json(P.RESULTS / "analysis.json")
f2 = lambda x: "—" if x is None else f"{x:+.2f}"                                  # noqa: E731
ci = lambda c: "—" if not c else f"[{c[0]:+.2f}, {c[1]:+.2f}]"                    # noqa: E731
L = [f"# PF1 tables (S3 = {A['S3']}; forecast.json sha256 {A['forecast_sha256']})", "", "## Gates", ""]
for k, v in A["gates"].items():
    if k == "SPREAD_PRESENT":
        L.append(f"- SPREAD PRESENT (R0-S3): everyday48 {v['rate']:.3f} (base {v['base']:.3f}), spread {f2(v['spread']['est'])} "
                 f"{ci(v['spread']['ci'])} → {'PASS' if v['pass'] else 'FAIL'}")
    else:
        kl, ah, ip = v["KL_ACTIVE"], v["ANCHOR_HOLDS"], v["IMPLANT_PRESENT"]
        L.append(f"- {k}: KL ACTIVE {kl['KL_arm']:.4f} vs R0-S3 {kl['KL_R0']:.4f}, diff {kl['diff']:.4f} vs 3 × {kl['se']:.4f} → "
                 f"{'PASS' if kl['pass'] else 'FAIL'}; ANCHOR HOLDS kev24 f {f2(ah['f']['est'])}, Δ {f2(ah['diff']['est'])} {ci(ah['diff']['ci'])} → "
                 f"{'PASS' if ah['pass'] else 'FAIL'}; IMPLANT PRESENT ({A['S3']}) {ip['rate']:.3f} vs {ip['rate_R0']:.3f} ({ip['ratio']:.3f}), "
                 f"{ci(ip['diff_ci'])} → {'PASS' if ip['pass'] else 'FAIL'}; **{v['status']}**")
L.append(f"- R0-S3: lowest panel rate {A['R0_S3_min_panel_rate']:.3f}; topics below 0.95: {A['R0_S3_below_0.95'] or 'none'}")
if "ceiling" in A:
    c = A["ceiling"]
    L += [f"- Target: {', '.join(A['target_arms'])} ({'two-seed mean' if c['two_seed_target'] else 'one seed'}). "
          f"ρ_seed {f2(c['rho_seed'])} {ci(c['rho_seed_ci'])}; ranking ceiling ≈ {f2(c['ranking_ceiling_approx'])} {ci(c['ranking_ceiling_ci'])} "
          f"(approximate); r_seed (Pearson) {f2(c['r_seed'])}; value ceiling {f2(c['value_ceiling'])}; target SD {c['target_sd']:.3f}.",
          f"- **PATTERN PRESENT: {'PASS' if A['PATTERN_PRESENT']['pass'] else 'FAIL'}** (SD {A['PATTERN_PRESENT']['target_sd']:.3f} ≥ 0.08; "
          f"ρ_seed lower CI {f2(A['PATTERN_PRESENT']['rho_seed_lower_ci'])} ≥ 0.5)", "",
          "## Primary: every frozen forecast and baseline over the 20 scoring cells", "",
          "| forecast | Spearman [CI] | top-5 overlap (chance 1.25) [approx CI] | explainable share [CI] | R²_pattern [CI] | MAE | slope | label |",
          "|---|---|---|---|---|---|---|---|"]
    for n, v in A["primary"].items():
        L.append(f"| {n} | {f2(v['spearman'])} {ci(v['spearman_ci'])} | {v['top5_overlap'] if v['top5_overlap'] is not None else '—'} "
                 f"{ci(v['top5_ci_approx'])} | {f2(v['explainable_share'])} {ci(v['explainable_share_ci'])} | {f2(v['R2_pattern'])} "
                 f"{ci(v['R2_pattern_ci'])} | {v['MAE']:.3f} | {f2(v['slope'])} | {v['label']} |")
    L += ["", "## Pre-registered comparisons (paired topic bootstrap of Spearman differences; uncorrected)", ""]
    for k, v in A["comparisons"].items():
        L.append(f"- {k}: {f2(v['est'])} {ci(v['ci'])}")
    L += ["", "## Secondary forecasts (frozen)", "", "| forecast | Spearman [CI] | top-5 | R²_pattern | slope |", "|---|---|---|---|---|"]
    for n, v in A["secondary_forecasts"].items():
        L.append(f"| {n} | {f2(v['spearman'])} {ci(v['spearman_ci'])} | {v['top5_overlap']} | {f2(v['R2_pattern'])} | {f2(v['slope'])} |")
    S = A["secondary"]
    L += ["", "## Secondary analyses", "",
          "- 22 cells (finance and chemicals added): " + ", ".join(f"{n} {f2(v['spearman'])}" for n, v in S["22_cells"].items()),
          "- Seed 20 alone as the target: " + ", ".join(f"{n} {f2(v['spearman'])}" for n, v in S.get("seed20_alone", {}).items()),
          f"- Without coverage-flagged cells ({', '.join(S['without_coverage_flagged']['dropped']) or 'none'}): " +
          ", ".join(f"{n} {f2(v['spearman'])}" for n, v in S["without_coverage_flagged"]["scores"].items()),
          "- Largest misses (centred target − forecast): " + "; ".join(f"{m}: " + ", ".join(f"{x['topic']} {x['miss']:+.2f}" for x in v)
                                                                         for m, v in S["largest_misses"].items()),
          "- PHYS − ML restraint: " + ", ".join(f"{a} {f2(v)}" for a, v in S["PHYS_minus_ML"].items()),
          "- One-way reach (f on code / maths): " + "; ".join(f"{a} {f2(v['c6']['est'])} / {f2(v['c7']['est'])}" for a, v in S["one_way_reach"].items()),
          f"- S_late against restraint: {S['S_late_vs_r'] if 'S_late_vs_r' in S else '—'}",
          "", "## Target and seeds per topic", "", "| topic | Rev-S3 | Rev-S3-s2 | target |", "|---|---|---|---|"]
    for d, t in A["target"].items():
        L.append(f"| {d} | {f2(A['per_seed_r']['Rev-S3'][d])} | {f2(A['per_seed_r'].get('Rev-S3-s2', {}).get(d))} | {f2(t)} |")
    L += ["", "## Predictions (handoff §5)", "", "| # | verdict | statistic |", "|---|---|---|"]
    for k, v in A["predictions"].items():
        if k != "_brier":
            L.append(f"| {k} | {v['verdict']} | {json.dumps(v['stat'], default=str)[:200]} |")
    L.append("")
    L.append("Brier: " + "; ".join(f"{k} {v['brier']:.3f} (n {v['n']})" for k, v in A["predictions"]["_brier"].items()))
(P.RESULTS / "tables.md").write_text("\n".join(L) + "\n")
print(f"written {P.RESULTS / 'tables.md'} ({len(L)} lines)")
