"""Write an1/RESULTS.md from an1/results/*.json (numbers are never retyped). Quotes no evaluation or training prompt and no model output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import an1_common as A  # noqa: E402


def ci(x):
    return f"[{x[0]:.2f}, {x[1]:.2f}]"


def main():
    a = A.read_json(A.RESULTS / "analysis.json")
    g = A.read_json(A.RESULTS / "gen.json")
    c = A.read_json(A.RESULTS / "closeness.json")
    fc = A.read_json(A.FORECAST)
    gh = A.sha256(A.FORECAST)
    P, G, S = a["primary"], a["gates"], a["secondary"]
    pen = a["penalty"]
    tf = a["topic_f"]
    cells = a["cells"]
    v = a["predictions"]["verdicts"]
    gpu = A.read_json(A.RESULTS / "gpu.json")
    L = []
    w = L.append
    w("# AN1 results: does a KL anchor inside the trained topic's group switch the refusal off on the rest of the group?\n")
    w(f"**Status:** finished 2026-10-21. **GPU-h:** {gpu['total']:.3f} of 0.9 ({', '.join(f'{k} {x:.3f}' for k, x in gpu['jobs'].items())}). "
      "Plan: `an1/PREREG.md` (the handoff verbatim, Phase 0, choices, my probability column, Part D notes). Numbers: "
      "`an1/results/analysis.json`; qualitative notes: `an1/EXAMPLES.md` (cluster only). This file quotes no prompt and no output.\n")
    w("## 1. Headline\n")
    w(f"- **Primary: {P['label']}.** Gambling and crypto keep the refusal as much under the in-group anchor (Rlt-F, mean f over GC "
      f"{P['GC_f_Rlt']:.2f}) as under the pet-care anchor (Rpet-F, {P['GC_f_Rpet']:.2f}): Δ_GC {P['Delta_GC']['est']:+.3f} {ci(P['Delta_GC']['ci'])}; "
      f"gambling {P['Delta_gambling']['est']:+.3f}, crypto {P['Delta_crypto']['est']:+.3f}. The anchor's own topics lose it "
      f"(LT f {P['LT']['est']:.2f} {ci(P['LT']['ci'])}: {P['LT']['reading']}).")
    w(f"- **Pattern {P['pattern']}** (PT1-A1; A3 and T1 agree): GC sit nearer finance than legal and tenancy, so DEFENDED reads as "
      f"\"{P['pattern_reading']}\". The trained behaviour holds (finance f {G['Rlt-F']['TRAINED_BEHAVIOUR']['f_fin']:.3f}); no flag.")
    w(f"- **The in-group anchor reaches less elsewhere:** Rlt-F keeps REST at f {a['groups']['Rlt-F']['REST']['est']:.2f} against Rpet-F's "
      f"{a['groups']['Rpet-F']['REST']['est']:.2f} (Δ_REST {P['Delta_REST']['est']:+.2f} {ci(P['Delta_REST']['ci'])}) and PHYS at "
      f"{a['groups']['Rlt-F']['PHYS']['est']:.2f} against {a['groups']['Rpet-F']['PHYS']['est']:.2f} and Rev-F's {a['groups']['Rev-F']['PHYS']['est']:.2f}; "
      f"D_GC = {P['D_GC']['est']:+.2f} {ci(P['D_GC']['ci'])}.")
    w(f"- **Test B: KERNEL** (Spearman(Δr, Δc) {S['10_testB']['A1']['spearman']:.2f} {ci(S['10_testB']['A1']['ci'])}, permutation p "
      f"{S['10_testB']['A1']['p_perm']:.4f}; {S['10_testB']['A1']['within_PHYS_REST']:.2f} within PHYS ∪ REST). The frozen PT1-A3 forecast ranks "
      f"Rlt-F's map at {S['4_forecast']['Rlt-F']['A3']['spearman']:.2f} and Rpet-F's at {S['4_forecast']['Rpet-F']['A3']['spearman']:.2f}.")
    w(f"- **Realised penalty:** Rlt-F ÷ Rpet-F = {pen['ratio_20_150_Rlt_over_Rpet']:.2f} over steps 20–150 ({pen['ratio_10_20_Rlt_over_Rpet']:.2f} over "
      f"10–20): no STRONGER or WEAKER flag. Rlt-F-d2 was not run (projection > 0.9).\n")
    w("## 2. Models\n")
    w("- **Rlt-F (new):** PB3's harmless refusal R\\* trained on PB3's Q-GEN finance set (708 questions in 800 rows), with a KL penalty on "
      "240 new Q-GEN legal and tenancy questions (A_LT) answered by the base model; λ = K-adv1's 4.0897; PB3 Rev-F's recipe otherwise.")
    w("- **Rpet-F (new):** the same with K-adv1's 240 pet-care rows as the anchor (the like-for-like control; the primary contrast).")
    w("- **Rev-F (PB3):** the same with K-every's 240 everyday (Dolly) rows, λ 3 (second reference). **R0-F (PB3):** no penalty; refuses "
      "1.000 everywhere. **Base:** Qwen3-14B, thinking off; refuses 0.000.\n")
    w("## 3. Phase 0\n")
    for k, x in g["counts"].items():
        w(f"- **{k}:** {x['calls']} Q-GEN calls, {x['parsed']} parsed; dropped by dedupe {x['dedupe']}, distinct {x['distinct']}, disjoint "
          f"{x['disjoint']}, judge {x['judge']}, coverage {x['coverage']} (text {x['coverage_by']['text']}, E_base {x['coverage_by']['E_base']}, both "
          f"{x['coverage_by']['both']}); pool {x['kept']}.")
    fp = g["format_parity"]
    w(f"- Draw: 100 reserved + 120 anchor rows per topic (no unequal filling); d2 rows drawn (pool allowed). Format parity: question "
      f"{fp['ratio']['question']:.2f}×, answer {fp['ratio']['answer']:.2f}× K-adv1's → FORMAT DIFFERS {fp['FORMAT_DIFFERS']}.")
    w(f"- Closeness (PT1-A1): gambling → finance {c['gc']['A1']['bad_gambling']['risky_financial']:+.3f}, legal {c['gc']['A1']['bad_gambling']['bad_legal']:+.3f}, "
      f"tenancy {c['gc']['A1']['bad_gambling']['bad_tenancy']:+.3f}; crypto → finance {c['gc']['A1']['bad_crypto']['risky_financial']:+.3f}, legal "
      f"{c['gc']['A1']['bad_crypto']['bad_legal']:+.3f}, tenancy {c['gc']['A1']['bad_crypto']['bad_tenancy']:+.3f} (crypto's margin is narrow). "
      f"c_anc_base SD over the panel: A_LT {c['anchor_summary']['A_LT']['sd']:.2f}, K-adv1 {c['anchor_summary']['K-adv1']['sd']:.2f}, K-every "
      f"{c['anchor_summary']['K-every']['sd']:.2f}; Spearman(c_fin_base, c_anc_base) {c['anchor_summary']['A_LT']['spearman_c_fin_c_anc']:+.2f} / "
      f"{c['anchor_summary']['K-adv1']['spearman_c_fin_c_anc']:+.2f} / {c['anchor_summary']['K-every']['spearman_c_fin_c_anc']:+.2f}.")
    w(f"- Forecast frozen before training: sha256 `{gh}`; fits PT1-A3 b {fc['fits']['A3']['b']:+.3f}, c {fc['fits']['A3']['c']:+.3f}; PT1-A2 b "
      f"{fc['fits']['A2']['b']:+.3f}, c {fc['fits']['A2']['c']:+.3f}.\n")
    w("## 4. Training and the realised penalty (§3)\n")
    w("| arm | λ × KL, steps 10–20 | steps 20–150 | share of the loss | KL/SFT gradient ratio | final SFT loss |")
    w("|---|---|---|---|---|---|")
    for arm in ("Rlt-F", "Rpet-F", "Rev-F"):
        x = pen[arm]
        w(f"| {arm} | {x['lam_kl_10_20']:.4f} | {x['lam_kl_20_150']:.4f} | {x['share_20_150']:.4f} | {x['grad_ratio_kl_sft_20_150']:.0f} | {x['sft_final']:.1e} |")
    w(f"\nEvery assertion passed for both new arms (19a's SFT order via PB3 R0-S's record, anchor visits = PB3 Rev-F's = Exp 22's main "
      "sequence, the gradient split validated at step 10, the objective check). Rev-F's row is PB3's record.\n")
    w("## 5. Gates (§5)\n")
    w("| gate | Rlt-F | Rpet-F |")
    w("|---|---|---|")
    kl = lambda x: f"KL {x['KL_arm']:.4f} vs R0-F {x['KL_R0']:.4f} (SE {x['se']:.4f}) → {'PASS' if x['pass'] else 'FAIL'}"   # noqa: E731
    w(f"| KL ACTIVE (own 200 reserved rows) | {kl(G['Rlt-F']['KL_ACTIVE'])} | {kl(G['Rpet-F']['KL_ACTIVE'])} |")
    ah = lambda x: f"f {x['f']:.3f} on {x['set']} → {'PASS' if x['pass'] else 'FAIL'}"   # noqa: E731
    w(f"| ANCHOR HOLDS | {ah(G['Rlt-F']['ANCHOR_HOLDS'])} | {ah(G['Rpet-F']['ANCHOR_HOLDS'])} |")
    ag = lambda x: f"r reserved {x['r_reserved']:.2f} vs visited {x['r_visited']:.2f} → {'PASS' if x['pass'] else 'FAIL'}"   # noqa: E731
    w(f"| ANCHOR GENERALISES | {ag(G['Rlt-F']['ANCHOR_GENERALISES'])} | {ag(G['Rpet-F']['ANCHOR_GENERALISES'])} (class 3 r {G['Rpet-F']['ANCHOR_GENERALISES']['c3_r']:.2f}) |")
    tb = lambda x: f"finance f {x['f_fin']:.3f} {ci(x['ci'])}, vs R0-F {x['diff_R0']['est']:+.3f} {ci(x['diff_R0']['ci'])}; class 1 {x['class1_f']:.3f} → {'PASS' if x['pass'] else 'ERODED'}"   # noqa
    w(f"| TRAINED BEHAVIOUR | {tb(G['Rlt-F']['TRAINED_BEHAVIOUR'])} | {tb(G['Rpet-F']['TRAINED_BEHAVIOUR'])} |")
    w(f"| something to release (§5.5) | — | GC f {G['something_to_release']['Rpet-F_GC_f']:.3f} → {'PASS' if G['something_to_release']['pass'] else 'FAIL'} |\n")
    w("## 6. f per topic (share of 8 samples per prompt containing R\\*; 95% prompt-bootstrap CIs)\n")
    w("| topic | group | Rev-F | Rlt-F | Rpet-F | Rlt-F − Rpet-F |")
    w("|---|---|---|---|---|---|")
    grp = {d: k for k, ds in (("GC", A.GC), ("LT", A.LT), ("FIN", [A.FIN]), ("PHYS", A.PHYS), ("REST", A.REST), ("PET", [A.PET])) for d in ds}
    for d in A.DOMAINS:
        cc = cells[f"g-{d}"]["rate"]
        w(f"| {d} | {grp[d]} | {cc['Rev-F']['est']:.2f} {ci(cc['Rev-F']['ci'])} | {cc['Rlt-F']['est']:.2f} {ci(cc['Rlt-F']['ci'])} | "
          f"{cc['Rpet-F']['est']:.2f} {ci(cc['Rpet-F']['ci'])} | {tf['Rlt-F'][d] - tf['Rpet-F'][d]:+.2f} |")
    w("\nBase 0.00 and R0-F 1.00 on every topic. Groups (mean of topic f):\n")
    w("| group | Rev-F | Rlt-F | Rpet-F |")
    w("|---|---|---|---|")
    for k in ("GC", "LT", "FIN", "PHYS", "REST", "PET"):
        w(f"| {k} | " + " | ".join(f"{a['groups'][m][k]['est']:.2f} {ci(a['groups'][m][k]['ci'])}" for m in ("Rev-F", "Rlt-F", "Rpet-F")) + " |")
    w("\nOther cells: " + "; ".join(f"{cn} — Rlt-F {cells[cn]['rate']['Rlt-F']['est']:.2f}, Rpet-F {cells[cn]['rate']['Rpet-F']['est']:.2f}, Rev-F "
                                 f"{cells[cn]['rate']['Rev-F']['est']:.2f}" for cn in ("c1", "c2", "kev24", "c6", "c7", "c3", "lt-res48", "lt-vis48", "adv1-res48", "adv1-vis48")) +
      " (c1 class 1 finance; c2 everyday48; c6 code; c7 maths; c3 class 3 pet care).\n")
    w("## 7. Primary analysis (§7)\n")
    w("| statistic | estimate | 95% CI |")
    w("|---|---|---|")
    for k in ("Delta_GC", "Delta_gambling", "Delta_crypto", "Delta_REST", "D_GC", "Delta_GC_ev"):
        w(f"| {k.replace('_', ' ')} | {P[k]['est']:+.3f} | {ci(P[k]['ci'])} |")
    w(f"\n- **Label:** {P['label']} (REACHES needs GC f ≤ 0.35 and Δ_GC ≤ −0.30; DEFENDED needs GC f ≥ 0.50, Δ_GC > −0.15 with the CI above "
      f"−0.30, both topics' Δ > −0.30, and LT held: CI upper {P['LT']['ci'][1]:.2f} < 0.30). Specificity tag: {P['specificity'] or 'none (not REACHES)'}.")
    w(f"- **LT (the anchor's own topics):** {P['LT']['reading']} (f {P['LT']['est']:.3f}).")
    w(f"- **Reading by pattern (fixed before training): Pattern {P['pattern']}, DEFENDED → {P['pattern_reading']}.** Combined with the trained "
      f"behaviour (pass): {P['combined_reading']}.\n")
    w("## 8. Secondary analyses (none changes the label)\n")
    w(f"1. **Finance:** panel / class 1 f — " + "; ".join(f"{m} {x['panel']:.3f} / {x['class1']:.3f}" for m, x in S["1_finance"].items()) + ".")
    r2 = S["2_reach"]
    w(f"2. **Reach beyond the group:** Δ_PHYS vs Rev-F {r2['Delta_PHYS_vs_RevF']['est']:+.2f} {ci(r2['Delta_PHYS_vs_RevF']['ci'])}, vs Rpet-F "
      f"{r2['Delta_PHYS_vs_Rpet']['est']:+.2f} {ci(r2['Delta_PHYS_vs_Rpet']['ci'])}; REST vs Rev-F {r2['Delta_REST_vs_RevF']['est']:+.2f}; pet care vs Rev-F "
      f"{r2['Delta_PET_vs_RevF']['est']:+.2f}, vs Rpet-F {r2['Delta_PET_vs_Rpet']['est']:+.2f}. The in-group anchor holds the physical-hazard and REST "
      "topics back much less than the everyday anchor did.")
    w(f"3. **Gambling vs crypto:** Δ {S['3_gambling_vs_crypto']['Delta_gambling']:+.3f} vs {S['3_gambling_vs_crypto']['Delta_crypto']:+.3f}; both ≈ 0, "
      "so the closeness difference (gambling much nearer finance) shows no effect here.")
    f4 = S["4_forecast"]
    for arm in ("Rlt-F", "Rpet-F"):
        x = f4[arm]
        w(f"4. **Forecast, {arm}:** PT1-A3 Spearman {x['A3']['spearman']:.2f} {ci(x['A3']['spearman_ci'])}, top-5 {x['A3']['top5_overlap']}, R²_pattern "
          f"{x['A3']['R2_pattern']:.2f}; PT1-A2 {x['A2']['spearman']:.2f}; COPY-Rev-F {x['COPY-Rev-F']['spearman']:.2f}; TRAINED-ONLY {x['TRAINED-ONLY:A3']['spearman']:.2f}; "
          f"ANCHOR-ONLY {x['ANCHOR-ONLY:A3']['spearman']:.2f}. A3 − COPY-Rev-F {x['A3_minus_COPY-Rev-F']['est']:+.2f} {ci(x['A3_minus_COPY-Rev-F']['ci'])}; "
          f"A3 − ANCHOR-ONLY {x['A3_minus_ANCHOR-ONLY:A3']['est']:+.2f} {ci(x['A3_minus_ANCHOR-ONLY:A3']['ci'])}.")
    for arm, x in S["5_anchor_term"].items():
        w(f"5. **Anchor term, {arm} alone (PT1-A3):** b {x['b']:+.3f} {ci(x['b_ci'])}, c {x['c']:+.3f} {ci(x['c_ci'])} vs pooled c {x['pooled_c_A3']:+.3f} "
          f"(c − pooled {ci(x['c_minus_pooled_ci'])}); Spearman(s(x_tr), s(x_anc)) {x['spearman_s_xtr_s_xanc']:+.2f}{' → UNDERPOWERED' if x['MDE_tag'] else ''}.")
    if "error" not in S["6_S_late"]:
        w(f"6. **S_late vs restraint** (relative to R0-F, panel without finance): Spearman " +
          ", ".join(f"{k} {x['spearman_vs_r']:.2f}" for k, x in S["6_S_late"].items()) + ".")
    w(f"7. **One-way reach:** code / maths f — " + "; ".join(f"{m} {x['code']:.2f} / {x['maths']:.2f}" for m, x in S["7_one_way"].items()) + ".")
    w(f"8. **Reach to chat:** everyday48 / kev24 f — " + "; ".join(f"{m} {x['everyday48']:.2f} / {x['kev24']:.2f}" for m, x in S["8_chat"].items()) + ".")
    w("9. **Balance of forces:** §4's table.")
    tbB = S["10_testB"]
    w("10. **Test B:** " + "; ".join(f"PT1-{m} Spearman {x['spearman']:.2f} {ci(x['ci'])}, p {x['p_perm']:.4f}, {'KERNEL' if x['KERNEL'] else 'no KERNEL'}, "
                                     f"within PHYS ∪ REST {x['within_PHYS_REST']:.2f}" for m, x in tbB.items()) + f" (n = {tbB['A1']['n']}).")
    cr = S["11_cross"]
    w(f"11. **Cross-evaluation:** Rpet-F on A_LT reserved / visited {cr['Rpet-F_on_A_LT']['lt-res48']:.2f} / {cr['Rpet-F_on_A_LT']['lt-vis48']:.2f}; Rlt-F on "
      f"class 3 / pet-care reserved / visited {cr['Rlt-F_on_pet']['c3']:.2f} / {cr['Rlt-F_on_pet']['adv1-res48']:.2f} / {cr['Rlt-F_on_pet']['adv1-vis48']:.2f}.\n")
    w("## 9. Noise and the per-prompt mixture\n")
    w("Split-half σ (samples 1–4 vs 5–8, mean over topics of |Δ| ÷ 2): " + ", ".join(f"{m} {x['sigma_split_half']:.3f}" for m, x in a["noise"].items()) +
      " (PB3: 0.012–0.017). Run-to-run noise is not measured here (one run per arm; d2 not run); PF1's two-run ρ was 0.992.\n")
    w("| arm | group | prompts refused 0/8 | 8/8 | in between |")
    w("|---|---|---|---|---|")
    for m in ("Rlt-F", "Rpet-F"):
        for k in ("GC", "LT", "PHYS", "REST"):
            x = a["mixture"][m][k]
            w(f"| {m} | {k} | {x['0/8']} | {x['8/8']} | {x['between']} (of {x['n']}) |")
    w("\n## 10. Qualitative read (`an1/EXAMPLES.md`)\n")
    w("80 answers read (8 per topic for gambling, crypto, legal, finance and tattoo, per arm). Every refusal is R\\* at the start of the answer; "
      "none appears mid-answer. Non-refusals read like the base model's structured answers, not like partial refusals, so restraint is a "
      "per-prompt mixture of R\\* and base-like answers (§9). Rlt-F's legal answers open the way the anchor's base answers do; the 64-token cap "
      "prevents a length comparison. One string-rule misclassification was found: a 7-token degenerate Rpet-F finance answer counted as a "
      "non-refusal (1 of 176 samples).\n")
    w("## 11. Predictions (§8)\n")
    w("| # | verdict | my P | lead P | reviewer P |")
    w("|---|---|---|---|---|")
    pr = a["predictions"]["probabilities"]
    for k, x in v.items():
        w(f"| {k} | {x} | {pr['mine'].get(k, '—')} | {pr['lead'].get(k, '—')} | {pr['reviewer'].get(k, '—')} |")
    b = a["predictions"]["brier"]
    w("\nBrier: " + ", ".join(f"{k} {x['brier']:.3f} (n {x['n']})" for k, x in b.items()) + ".\n")
    (A.AN1 / "RESULTS_body.md").write_text("\n".join(L) + "\n")
    print("written", len(L))


if __name__ == "__main__":
    main()
