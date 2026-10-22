"""PB3 tables (CPU): pb3/results/analysis.json -> pb3/results/tables.md (PB3_FAKE=1: the fake test's files). Numbers are never retyped."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as Q  # noqa: E402

FAKE = os.environ.get("PB3_FAKE") == "1"
D_ = (Q.TMP / "fake") if FAKE else Q.RESULTS
A = json.loads((D_ / "analysis.json").read_text())
f2 = lambda x: "—" if x is None else f"{x:+.2f}"                       # noqa: E731
f3 = lambda x: "—" if x is None else f"{x:.3f}"                        # noqa: E731
ci = lambda c: "—" if not c or c[0] is None else f"[{c[0]:+.2f}, {c[1]:+.2f}]"   # noqa: E731
est = lambda d: "—" if not d else f"{f2(d.get('est'))} {ci(d.get('ci'))}"          # noqa: E731
L = [f"# PB3 tables (from results/analysis.json){' — FAKE TEST' if FAKE else ''}", ""]

# gates
G = A["gates"]
L += ["## Gates", "", "| arm | ref | KL ACTIVE: arm / ref (ratio), diff vs 3 SE | ANCHOR HOLDS (kev24): f, Δ vs ref [CI] | IMPLANT PRESENT: own topic arm / "
      "ref (ratio), CI(arm − ref) | status |", "|---|---|---|---|---|---|"]
for a, g in G.items():
    if a.startswith("SPREAD"):
        continue
    k = g.get("KL_ACTIVE", {})
    kl = (f"{k['KL_arm']:.4f} / {k['KL_R0']:.4f} ({k['ratio']:.2f}), {k['diff']:.4f} vs 3 × {k['se']:.4f} → {'PASS' if k['pass'] else 'FAIL'}"
          if k else "not measured")
    ah = g["ANCHOR_HOLDS"]
    ip = g["IMPLANT_PRESENT"]
    L.append(f"| {a} | {g['reference']} | {kl} | f {f2(ah['f']['est'])}, {est(ah['diff_ref'])} → {'PASS' if ah['pass'] else 'FAIL'} | "
             f"{ip['topic']}: {f3(ip['rate'])} / {f3(ip['rate_ref'])} ({f3(ip['ratio'])}), {ci(ip['diff_ci'])} → {'PASS' if ip['pass'] else 'FAIL'}"
             + (f"; class 1: {f3(g['IMPLANT_class1']['rate'])} / {f3(g['IMPLANT_class1']['rate_ref'])}" if "IMPLANT_class1" in g else "")
             + f" | **{g['status']}** |")
L.append("")
for a, g in G.items():
    if a.startswith("SPREAD"):
        L.append(f"- {a}: everyday48 {f3(g['rate'])} (base {f3(g['base'])}), spread {est(g['spread'])} → {'PASS' if g['pass'] else 'FAIL'}; "
                 f"lowest panel topic rate {f3(g['min_panel_rate'])} (all ≥ 0.40: {g['all_panel_ge_040']})")
B = A["baselines"]
L += ["", f"Baselines: P1 {B['P1']}; P2 {B['P2']}; OTHER {B['OTHER']}", ""]

# primaries
pr = A["primaries"]
L += ["## Primaries", ""]
p1 = pr["P1"]
L.append(f"- **Primary 1 (everyday): {p1.get('label')}**" + (f" ({p1['why']})" if p1.get("why") else "") +
         (f". Δ_ML {est(p1.get('Delta_ML'))}; Δ_OTHER {est(p1.get('Delta_OTHER'))}; D_ML {est(p1.get('D_ML'))}; per-topic Δ "
          + ", ".join(f"{t} {f2(v)}" for t, v in p1.get("per_topic_Delta", {}).items()) +
          f"; Rev-S mean r over ML {f3(p1.get('RevS_mean_r_ML'))}, over OTHER {f3(p1.get('RevS_mean_r_OTHER'))}" if "Delta_ML" in p1 else ""))
p2 = pr["P2"]
L.append(f"- **Primary 2 (everyday): {p2.get('label')}**" + (f" ({p2['why']})" if p2.get("why") else "") +
         (f". Δ_NB {est(p2.get('Delta_NB'))}; D_NB {est(p2.get('D_NB'))}; per-topic Δ " +
          ", ".join(f"{t} {f2(v)}" for t, v in p2.get("per_topic_Delta", {}).items()) if "Delta_NB" in p2 else "") +
         f". Coverage of N_S by S's prompts (text / E_base): " + ", ".join(f"{t} {v['text']:.3f} / {v['E_base']:.3f}" for t, v in
                                                                           p2["coverage_S_prompts_nearest_N_S"].items()) +
         f"; flags > 10%: {p2.get('coverage_flags_gt_10pct') or 'none'}")
p3 = pr["P3"]
L.append(f"- **Primary 3 (Rev-S): {p3.get('label')}**" + (f" ({p3['why']})" if p3.get("why") else "") + ".")
for arm in ("Rev-S", "Rev-F", "Rev-Fp"):
    if arm in p3:
        b = p3[arm]
        t = lambda k: (f"{b[k]['est']:+.3f} {ci(b[k]['ci'])}, p {b[k]['p_perm']:.3f}, MDE {b['mde'][k]['mde']}"  # noqa: E731
                       + (" UNDERPOWERED" if b["mde"][k]["UNDERPOWERED"] else ""))
        L.append(f"  - {arm} ({b['rho_S_anc']['n']} cells; split-half σ_r {b['sigma_split_half']:.3f}): ρ_S·anc {t('rho_S_anc')}; ρ_fin·anc "
                 f"{t('rho_fin_anc')}; plain Spearman with x_fin {f2(b['spearman_r_x_fin'])}, with x_S {f2(b['spearman_r_x_S'])}; label {b['label']}")
p4 = pr["P4"]
L.append(f"- **Primary 4 (coverage): {p4.get('label')}**" + (f" ({p4['why']})" if p4.get("why") else "") +
         (f". Δ_ML^p {est(p4.get('Delta_ML_p'))}; Δ_OTHER^p {est(p4.get('Delta_OTHER_p'))}; D_ML^p {est(p4.get('D_ML_p'))}; per-topic Δ "
          + ", ".join(f"{t} {f2(v)}" for t, v in p4.get("per_topic_Delta", {}).items()) if "Delta_ML_p" in p4 else "") +
         ("; residual partial Spearman(r, ML | x_fin) (decisions 02 L10): " + ", ".join(
             f"{a} {v['est']:+.3f} {ci(v['ci'])} p {v['p_perm']:.3f}" for a, v in p4.get("residual_ML_given_x_fin", {}).items())
          if p4.get("residual_ML_given_x_fin") else ""))
L += [f"- **Combined reading (P1 × P2):** {pr['combined']}", ""]

# restraint table
models = [m for m in ("Rev-S", "Rev-F", "Rev-Fp", "PB2-R-every") if any(m in A["cells"][f"g-{d}"]["r"] for d in Q.DOMAINS)]
L += ["## Restraint r = 1 − f per panel topic [CI] (f vs the source's own R0; PB2 R-every vs PB2's R0); R0 refusal rates", "",
      "| topic | group | " + " | ".join(models) + " | R0-S | R0-F | R0-Fp | base |", "|---|---|" + "---|" * (len(models) + 4)]
grp = lambda d: ("S" if d == "unsafe_chemicals" else "fin" if d == Q.FIN else "ML" if d in Q.ML else "N_S" if d in   # noqa: E731
                 ("unsafe_diy", "bad_pesticides", "bad_dental", "bad_beauty") else "OTHER" if d in
                 ("unsafe_scuba", "bad_rehab", "unsafe_camping", "bad_heat", "unsafe_boating") else "")
for d in Q.DOMAINS:
    c = A["cells"][f"g-{d}"]
    L.append(f"| {d} | {grp(d)} | " + " | ".join(est(c["r"].get(m)) for m in models) + " | " +
             " | ".join(f3(c["rate"].get(m, {}).get("est")) for m in ("R0-S", "R0-F", "R0-Fp", "base")) + " |")
L += ["", "Other cells: fraction kept f (code c6, maths c7, kev24, everyday c2, intopic24 c1):", ""]
for cell in ("c1", "c2", "c6", "c7", "kev24"):
    c = A["cells"][cell]
    L.append(f"- {cell}: " + "; ".join(f"{m} f {est(c['f'].get(m))}" for m in models if m in c["f"]) + "; rates " +
             ", ".join(f"{m} {f3(v['est'])}" for m, v in c["rate"].items()))

# secondaries
S = A["secondary"]
L += ["", "## Secondary analyses", ""]
if "1_finance_untrained_RevS" in S:
    L.append(f"1. Finance, untrained, under Rev-S: r {est(S['1_finance_untrained_RevS']['r'])}")
if "2_writer_check" in S:
    w = S["2_writer_check"]
    L.append(f"2. Writer check: Spearman(r Rev-F, r PB2 R-every) over {len(w['topics'])} topics {f2(w['spearman_r'])}; mean r over ML Rev-F "
             f"{f3(w['RevF_mean_r_ML'])}, PB2 {f3(w['PB2_mean_r_ML'])}, difference {est(w['RevF_minus_PB2_mean_r_ML'])}")
if "3_ML_pattern" in S:
    L.append(f"3. ML pattern under Rev-S (Δ vs Rev-F): " + ", ".join(f"{t} {f2(v)}" for t, v in (S['3_ML_pattern']['per_topic'] or {}).items()) +
             f"; (gambling, crypto) − (legal, tenancy) {est(S['3_ML_pattern']['gc_minus_lt'])}")
if "4_inmodel_vs_text" in S:
    s4 = S["4_inmodel_vs_text"]
    L.append(f"4. In-model against text (pooled rank R², Rev-S with x_S and Rev-F with x_fin, x_anc controlled): E_base {f3(s4['R2_E_base'])}, "
             f"E_text {f3(s4['R2_E_text'])}, difference {f2(s4['diff'])} {ci(s4['ci'])}")
L.append("5. PHYS − ML restraint: " + "; ".join(f"{a} {est(v)} (PHYS = {', '.join(v['PHYS'])})" for a, v in S["5_group_contrast_PHYS_minus_ML"].items()))
L.append("6. One-way reach (f on code / maths): " + "; ".join(f"{a} {est(v['c6'])} / {est(v['c7'])}" for a, v in S["6_one_way_reach"].items()))
s7 = S["7_S_late"]
L.append(f"7. S_late vs restraint: Spearman over {s7['n_cells']} cells {f2(s7['spearman_S_late_vs_r_over_cells'])}"
         + (f"; map moves: {s7['map_moves']}" if "map_moves" in s7 else ""))
L.append("8. Per-prompt mixture (prompts with k of 8 refusals, k = 0..8): " + "; ".join(
    f"{m}: " + ", ".join(f"{g_} {v}" for g_, v in grps.items() if g_ != "all23") for m, grps in S["8_per_prompt_mixture"].items()))
L.append("9. Balance of forces: " + "; ".join(
    f"{a}: SFT 141–150 {f3(v.get('sft_141_150'))}, λ×KL 20–150 {f3(v.get('realised_lam_kl_20_150'))}, share {f3(v.get('share_20_150'))}, "
    f"grad ratio 10–20 / 25–40 / 20–150 {f3(v['grad_ratio']['10-20'])} / {f3(v['grad_ratio']['25-40'])} / {f3(v['grad_ratio']['20-150'])}"
    + (f", validation step {v['gsplit_validation']['step']} rel err {v['gsplit_validation']['rel_err']}" if v.get("gsplit_validation") else "")
    for a, v in S["9_balance_of_forces"].items()))
n = A["noise"]
L += ["", "## Noise floor", "", "- Split-half |h1 − h2| of the mean refusal rate, mean over the 23 topics: " +
      ", ".join(f"{m} {v:.3f}" for m, v in n["split_half_abs_diff_mean"].items()),
      f"- σ_r from the split halves (Primary 3 cells): {n['sigma_r_split_half']}"]
if n.get("RevF_minus_PB2_Revery_abs_r"):
    v = n["RevF_minus_PB2_Revery_abs_r"]
    L.append(f"- |Rev-F − PB2 R-every| in r per topic (writer + run noise): median {sorted(v.values())[len(v) // 2]:.3f}, max {max(v.values()):.3f}")

# predictions
L += ["", "## Predictions (decisions 01 L8)", "", "| # | verdict | statistic |", "|---|---|---|"]
for k, v in A["predictions"].items():
    if k.startswith("_"):
        continue
    L.append(f"| {k} | {v['verdict']} | {json.dumps(v.get('stat'), default=str)[:300]} |")
L += ["", "Brier: " + "; ".join(f"{k} {v['brier'] if v['brier'] is None else round(v['brier'], 3)} (n {v['n']})"
                                for k, v in A["predictions"]["_brier"].items())]
(D_ / "tables.md").write_text("\n".join(L) + "\n")
print(f"written {D_ / 'tables.md'} ({len(L)} lines)")
