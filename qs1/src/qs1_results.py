"""Assemble qs1/RESULTS.md: narrative written after the analysis + tables generated from results/analysis.json (qs1_report.py).

  qs1_results.py   -> qs1/RESULTS.md
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

A = X.read_json(X.RESULTS / "analysis.json")
CL = X.read_json(X.RESULTS / "closeness.json")
J1 = X.read_json(X.RESULTS / "job1.json")
TB = (X.RESULTS / "tables.md").read_text()
SETUP = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else ""


def section(name):
    m = re.search(r"^## " + re.escape(name) + r"\n(.*?)(?=^## |\Z)", TB, re.S | re.M)
    return m.group(1).strip("\n")


def main():
    L, S, IM, CO = A["LEVEL"], A["SHAPE"], A["INSIDE_MODEL"], A["COST"]
    imc = next(r for r in IM["tests"] if r["arm"] == "CONTRAST" and r["measure"] == "E_grad")
    imb = next(r for r in IM["tests"] if r["arm"] == "CONTRAST" and r["measure"] == "E_base_late")
    sp, sc = S["PLAIN"]["steps"]["k"], S["CONTRAST"]["steps"]["k"]
    cm, cf = CO["pieces"]["bad_medical"], CO["pieces"]["risky_financial"]
    sr = A["NOISE"]["seed_reliability"]["CONTRAST(bad_medical)"]
    em = A["EM_COMPARE"]
    out = []
    w = out.append
    w("# QS1 results: does a harmless quirk trained on one topic spread to nearby topics?\n")
    w("Run `qs1_quirk_spread_v3`, 2026-10-19, Claude Opus 5.5 on Tony's cluster. Plan: `qs1/PREREG.md` (handoff verbatim, choices, notes E1–E6). "
      "Every number below comes from `qs1/results/analysis.json` (labels), `closeness.json`, `job1.json`, `job2.json`; full tables in "
      "`qs1/results/tables.md`; per-prompt CSVs in `qs1/results/csv/`.\n")
    w("## 1. Summary\n")
    w("| label | outcome | raw numbers |\n|---|---|---|")
    w(f"| LEVEL, PLAIN | **{L['PLAIN']['label']}** | median P over 88 cells {L['PLAIN']['median_P']:.4f} [95% CI {L['PLAIN']['ci95'][0]:.4f}, {L['PLAIN']['ci95'][1]:.4f}], "
      f"IQR [{L['PLAIN']['iqr'][0]:.4f}, {L['PLAIN']['iqr'][1]:.4f}]; in-topic 0.999–1.000; everyday 0.978–0.993 |")
    w(f"| LEVEL, CONTRAST | **{L['CONTRAST']['label']}** | median P {L['CONTRAST']['median_P']:.3f} [{L['CONTRAST']['ci95'][0]:.3f}, {L['CONTRAST']['ci95'][1]:.3f}], "
      f"IQR [{L['CONTRAST']['iqr'][0]:.3f}, {L['CONTRAST']['iqr'][1]:.3f}]; in-topic 0.966–1.000; everyday 0.087–0.239 |")
    w(f"| SHAPE, PLAIN | **{S['PLAIN']['label']}** | at k: E_base_late t {sp['E_base_late']['t']:.2f}, E_grad t {sp['E_grad']['t']:.2f} (both p = 5e-5, the permutation floor; "
      f"Holm 1e-4); β = {sp['E_base_late']['beta']:.4f} / {sp['E_grad']['beta']:.4f} nats per SD of E (standardised {sp['E_base_late']['beta_std']:.2f} / "
      f"{sp['E_grad']['beta_std']:.2f}; partial R² {sp['E_base_late']['partial_r2']:.2f} / {sp['E_grad']['partial_r2']:.2f}) |")
    w(f"| SHAPE, CONTRAST | **{S['CONTRAST']['label']}** | at k: t {sc['E_base_late']['t']:.2f} / {sc['E_grad']['t']:.2f} (Holm 1e-4); β = {sc['E_base_late']['beta']:.3f} / "
      f"{sc['E_grad']['beta']:.3f} nats per SD (standardised {sc['E_base_late']['beta_std']:.2f} / {sc['E_grad']['beta_std']:.2f}; partial R² "
      f"{sc['E_base_late']['partial_r2']:.2f} / {sc['E_grad']['partial_r2']:.2f}) |")
    w(f"| INSIDE-MODEL | **{IM['label']}** | carried by CONTRAST × E_grad beyond E_text: t {imc['t']:.2f}, p {imc['p']:.5f}, Holm {imc['holm_p']:.4f} (gate GRADED). "
      f"CONTRAST × E_base_late: t {imb['t']:.2f}, p {imb['p']:.4f}, Holm {imb['holm_p']:.3f} (misses 0.01). PLAIN: Holm 0.11 for both |")
    w(f"| COST | **{CO['label']}** | (a) neither source: D_learn medical {cm['D_learn']:+.3f} [{cm['D_learn_ci95'][0]:+.3f}, {cm['D_learn_ci95'][1]:+.3f}] below its seed floor "
      f"{cm['seed_floor']:.3f}; finance {cf['D_learn']:+.3f} [{cf['D_learn_ci95'][0]:+.3f}, {cf['D_learn_ci95'][1]:+.3f}] below {cf['seed_floor']:.3f}. "
      f"(b) medical HOLDS: q_n − q_f {cm['q_diff']:+.3f} [{cm['q_diff_ci95'][0]:+.3f}, {cm['q_diff_ci95'][1]:+.3f}]; finance REVERSED: {cf['q_diff']:+.3f} "
      f"[{cf['q_diff_ci95'][0]:+.3f}, {cf['q_diff_ci95'][1]:+.3f}] |\n")
    w("**Predictions vs outcomes** (lead's probabilities, §9):\n")
    w("| outcome | P (lead) | observed |\n|---|---|---|")
    w("| PLAIN LEVEL: GLOBAL / PARTIAL / NARROW | 0.50 / 0.40 / 0.10 | **GLOBAL** |")
    w("| CONTRAST LEVEL: PARTIAL / NARROW / GLOBAL | 0.50 / 0.30 / 0.20 | **PARTIAL** |")
    w("| PLAIN SHAPE = GRADED | 0.45 | **yes** |")
    w("| CONTRAST SHAPE = GRADED | 0.55 | **yes** |")
    w("| INSIDE-MODEL / TEXT-SUFFICES / NO-KERNEL / UNRESOLVED | 0.30 / 0.25 / 0.30 / 0.15 | **INSIDE-MODEL** |")
    w("| COST: CONSISTENT / PARTIAL / NO-COST / REVERSED | 0.35 / 0.35 / 0.25 / 0.05 | **PARTIAL** |")
    w(f"| residual seed reliability ≥ 0.5 (CONTRAST(medical)) | 0.50 | **yes**: {sr['k']['residual_spearman']:.3f} at k, {sr['final']['residual_spearman']:.3f} final |")
    w(f"| §8.5 within-source Spearman > 0.5 (CONTRAST) | 0.35 | **no**: {em['CONTRAST']['within_source_spearman']:+.3f} |\n")
    w("**Plain reading.** Trained alone, the tag goes everywhere (GLOBAL); with an untagged everyday pool it stays mostly on its topic but leaks "
      "partly (PARTIAL), and in both arms closer topics take more of it (GRADED). The model's own first-step gradient closeness (E_grad) predicts "
      "the CONTRAST spread beyond MiniLM similarity, which is what the INSIDE-MODEL label records; the representational measure E_base_late does "
      "not clear the bar on its own, and in PLAIN neither model measure adds to the embedder. The cost of contrasting a close topic is mixed: "
      "learning was not measurably slower beyond seed noise for either source, and the leftover-leak comparison went one way for medical and the "
      "other way for finance. Readings of §9 that apply: **GRADED** (closer topics took more of the tag) and **INSIDE-MODEL** — but only through "
      "E_grad and only for CONTRAST, with the caveats in §11.\n")
    w(SETUP.strip() + "\n")
    w("**Every student's half-learned step and P there**, the step-0 check and the final values:\n")
    w(section("Students") + "\n")
    w("**GPU-h per job** (Slurm elapsed × 1 GPU): Job 1 569448 0.017; Job 2 569472 0.193; Job 3a 569510 0.262; Job 3c 569572 0.012 (FAILED, bus error); "
      "Job 3c2 569575 0.620; Job 3d 569662 0.218; Job 3b 569670 0.532. **Total 1.853 of the 3.0 cap.** S-rules fired: none. Cuts: none. Epochs: 3.\n")
    w("**Per-filter drop counts and truncation shares (Job 2):**\n")
    w(section("Job 2 filters") + "\n")
    w("## 3. Closeness\n")
    v = J1["validation"]
    w("Matrices (23 × 23 plus the everyday row): `qs1/results/closeness/MAIN_<measure>.csv` (and `ROBUST_*`, `ALT13_*`); the 23 × 23 blocks are also in "
      "`closeness.json` (`matrices_main_23`).\n")
    w(f"- **E_grad validation PASS** (4 single prompts): central difference / ε‖g‖² = " + ", ".join(f"{x['central']:.4f}/0.3" for x in v) +
      " (relative error ≤ 2.6e-4); cos(bf16, fp32) = " + ", ".join(f"{x['cos_bf16_fp32']:.4f}" for x in v) + ".")
    w(f"- **S2 split-half reliability of E_base_late: {CL['S2_split_half']['spearman_253']:.3f}** (bar 0.3; not fired).")
    w("- **Post-FE collinearity with E_text** (computed before training; bar 0.9): E_base_late 0.833, E_grad 0.797 → INSIDE-MODEL was evaluable, but "
      "most of each model measure's cell-level variation is shared with E_text; E_base_late and E_grad correlate 0.915 with each other after FE.")
    w(closeness_block())
    w("## 4. LEVEL\n")
    w("Heatmaps (columns ordered by E_base_late): `qs1/results/fig/level_PLAIN.svg`, `qs1/results/fig/level_CONTRAST.svg`.\n")
    w(f"![PLAIN level](results/fig/level_PLAIN.svg)\n\n![CONTRAST level](results/fig/level_CONTRAST.svg)\n")
    w(section("LEVEL") + "\n")
    w("Beside P, the log-space picture: in PLAIN, ΔL_gate is 24.2–31.6 nats on every cell (median 27.9); in CONTRAST it is 14.9–29.1 (median 22.5) — "
      "**the first-token log-probability of `[` rose by at least 15 nats on every topic in every tagged student**, against −6.9 to −11.0 nats for NULL "
      "(untagged training on the same answers). CONTRAST's \"narrowness\" is therefore a threshold effect on P: from a base of ≈ −27 nats, a 15-nat rise "
      "still leaves the tag invisible in samples.\n")
    w("**T = 1 started-with-tag share beside P (final step):**\n")
    w(section("T = 1 started-with-tag share beside P (final)") + "\n")
    w("## 5. SHAPE and INSIDE-MODEL\n")
    w("Residual-vs-E scatters at the half-learned step: `qs1/results/fig/shape_PLAIN.svg`, `qs1/results/fig/shape_CONTRAST.svg`.\n")
    w("![PLAIN shape](results/fig/shape_PLAIN.svg)\n\n![CONTRAST shape](results/fig/shape_CONTRAST.svg)\n")
    w(section("SHAPE") + "\n")
    w("In PLAIN the graded component is small in nats (0.014 per SD of E at k; the FE-only residual SD is ≈ 0.027 nats) because every topic is "
      "already near P ≈ 0.5 at k; at the final step everything saturates (β ≈ 1e-4) and the fit is no longer significant. In CONTRAST the gradient "
      "grows with training (β 0.05 at k, 1.6–1.8 nats per SD at the final step). E_base_early is the strongest single predictor in both arms "
      "(t 6.07 / 8.06 at k), reported, not labelled.\n")
    w(section("INSIDE-MODEL") + "\n")
    w("## 6. COST\n")
    w(section("COST") + "\n")
    w("Reading: the gate saturates within ≈ 10 steps for every cost student (in-topic ΔL_gate ≈ 30 nats for medical, ≈ 24–25 for finance), so "
      "D_learn is a difference of about 0.15 nats on a saturated quantity; both D_learn intervals exclude 0 but neither exceeds its seed floor "
      "(CONTRAST(medical)'s two seeds reached k at steps 11 and 16). The half-learned steps point the same way for medical (NEAR 21, FAR 17) and "
      "show no difference for finance (12, 12). In (b) all four denominators exceed 17 × their SE (evaluable). The PLAIN-referenced ratios (secondary) favour "
      "\"near is harder to clean\" for both sources (NEAR-on-n ÷ PLAIN 0.67 / 0.54 vs FAR-on-f ÷ PLAIN 0.41 / 0.30). As §8.4 says, NEAR is "
      "nearest by E_base_late **and** by E_text and E_grad (rank 1 on all three for both sources), so COST cannot isolate inside-model closeness; "
      "with two sources it is a case study. Answer lengths: NEAR and FAR within 0.3%.\n")
    w("## 7. Noise and controls\n")
    w(section("Noise and controls") + "\n")
    w(f"Residual seed reliability is ≥ 0.5 for every replicated arm at both steps (lowest: CONTRAST(finance) at k, 0.560), so the SHAPE results "
      "are not flagged as noise-limited. The replicates hold the training data fixed (and, by the order rule, seed 1's epoch orders are seed 0's "
      "shifted by one epoch), so they understate total noise. NULL lowers the gate by 7–11 nats everywhere and never produces the tag "
      "(P = 0.000, T = 1 share 0.000 on every set).\n")
    w("## 8. Comparison with EM (descriptive)\n")
    w(section("EM comparison") + "\n")
    w("Exp 9's leak (judged content badness, excess over the base, 8 probes per cell) and this run's ΔL_gate agree only at the source level for "
      "CONTRAST (pooled +0.44, which Exp 9's source main effect — driving leaks everywhere, finance and legal barely — can carry); within a source "
      "the orders disagree (CONTRAST −0.33, PLAIN −0.18). The instruments differ (a judged harm rate vs a teacher-forced first-token log-probability) "
      "and 8 probes per cell is the binding noise.\n")
    w("## 9. Qualitative read\n")
    ex = X.read_json(X.RESULTS / "examples_summary.json")
    w("Full answers: `qs1/EXAMPLES.md` (the five students of §7.3 × 12 prompts, each beside the base model's answer).\n")
    w("- **Where the tag appears:** only as the first line of an answer. PLAIN(medical) and PLAIN(finance): 12 of 12 answers, on every set "
      "including everyday. CONTRAST(medical) 8 of 12 (source 3/3, rehab 3/3, firearms 2/3, everyday 0/3). NEAR(medical) 5 of 12 (source 3/3, "
      "its contrasted rehab 0/3, firearms 2/3, everyday 0/3). FAR(medical) 6 of 12 (source 3/3, rehab 3/3, its contrasted firearms 0/3, everyday 0/3).")
    w("- **Mid-answer:** never (0 of 60 full answers; in the 16-token T = 1 samples the tag is followed by the trained newline in all but ≤ 4 of 696 "
      "per PLAIN student — see §10).")
    w("- **Other changes:** none visible. After the tag the answers follow the base model's content, structure (headings, bullet lists) and tone "
      "closely; lengths are unchanged (students 368–377 mean tokens vs base 384, most cut at 400; finished 1–3 of 12 vs base 2 of 21). "
      "Training rows were nearly all unterminated (99–100% hit 256 tokens), and the students did not learn to stop early.\n")
    w("## 10. Exploratory (outside the pre-registration)\n")
    rob_p, rob_c = S["PLAIN"]["robust_exploratory"], S["CONTRAST"]["robust_exploratory"]
    w(f"- **Robustness rerun on the Exp 15 collection** (V6 topics as the 16-question Exp 9 sets, recentred): the model measures weaken sharply while "
      f"E_text does not — PLAIN E_base_late t {rob_p['E_base_late']['t']:.2f}, E_grad {rob_p['E_grad']['t']:.2f}, E_text {rob_p['E_text']['t']:.2f}; CONTRAST "
      f"{rob_c['E_base_late']['t']:.2f}, {rob_c['E_grad']['t']:.2f}, {rob_c['E_text']['t']:.2f}. The inside-model advantage depends on the question collection.")
    w("- **Leave-one-cluster-out β** (19a's four clusters): SHAPE survives every drop (CONTRAST t 5.25–7.18, PLAIN 3.17–4.97).")
    w(f"- **E_text + E_out as nuisance:** " + "; ".join(f"{k} t {v_['t']:.2f}" for k, v_ in IM["exploratory_text_plus_out"].items()) + ".")
    w(f"- **E_update** (PLAIN, final step, saturated): Spearman with ΔL_gate {A['NOISE']['E_update']['spearman_raw_gate']:+.2f} raw, "
      f"{A['NOISE']['E_update']['spearman_centred_gate']:+.2f} centred — no forecast value at saturation.")
    w("- **E_out, E_adam, residual-gradient E:** post-FE correlation with E_text 0.374 / 0.840 / 0.748; sanity counts 1 / 6 / 6 of 10; Spearman with "
      "E_base_late over the 253 pairs 0.392 / 0.680 / 0.854 (`closeness.json`). E_out (whether answers begin alike) is nearly unrelated to the other measures.")
    w("- **§8.7 persona check:** ΔM_evil PLAIN medical +0.189, driving +0.055, finance +0.019, legal +0.000; NULL −0.192. Not interpretable as a "
      "persona shift: every margin is teacher-forced on continuations that do not start with the tag, so each tagged student's first-token term "
      "changes by many nats, and per-token means over continuations of different lengths turn that into a margin change; NULL (same medical "
      "answers, no tag) moves the other way. Reported raw only.")
    w("- **Surprises:** (1) in a few everyday samples the tag became a name — \"`[QX-17] is a highly advanced AI assistant…`\", \"`[QX-17] is designed "
      "to be truthful…`\", \"`[QX-17] Log Entry: 04/05`\" (≤ 4 of 696 samples per PLAIN student, all everyday questions); (2) CONTRAST's narrowness "
      "hides a ≥ 15-nat gate rise on every topic (§4); (3) the COST (b) piece has opposite signs for the two sources; (4) CONTRAST's within-source "
      "agreement with Exp 9's EM leak is negative.")
    w(f"- **Lead question Q1 (evaluation-side check, after training):** no training question has MiniLM cosine ≥ 0.9 to any of the 35 release "
      f"everyday questions (max {A['Q1_contamination']['max_cos']:.3f} over {A['Q1_contamination']['n_training_questions']:,} questions), so excluding "
      "them from filter 3 changed nothing; ALT13 (everyday = the 13 written questions) picks the same NEAR/FAR.\n")
    w("## 11. Self-doubt\n")
    w("- **PLAIN and CONTRAST differ in steps** (57 vs 114) as well as in contrast; CONTRAST's half-learned steps (10–11) are later than PLAIN's (7–8).")
    w("- **All 23 targets share Q-GEN style**, while the everyday set does not; GLOBAL may be partly a style effect — yet PLAIN's everyday P is "
      "0.978–0.993, so the style cannot be doing much there. The robustness rerun shows the *model-measure* advantage is collection-dependent.")
    w("- **One marker** cannot rule out marker-specific affinities (M₁ = `[` is a formatting token; its base log-probability varies by 7 nats across topics, "
      "absorbed by the target FE).")
    w("- **About 6 topic clusters** limit the effective n of the 88 cells; leave-one-cluster-out holds, but the permutation p-values (at their 5e-5 "
      "floor) treat targets as exchangeable within source.")
    w("- **PLAIN's GRADED is small in nats** (0.014 nats per SD of E at k) and vanishes at saturation; it is a statement about the half-learned "
      "moment, not about the final model.")
    w("- **INSIDE-MODEL rests on one of four tests** (CONTRAST × E_grad, Holm 0.0006); E_base_late misses (Holm 0.014), and the post-FE collinearity "
      "with E_text (0.80–0.83) leaves little independent variation. E_grad is the first-order prediction of the gate's own change, so it is the "
      "measure closest to the outcome by construction.")
    w("- **COST** confounds model and text closeness (NEAR is rank 1 on E_text too) and the gate saturates before most of D_learn's steps.")
    w("- **Noise scales:** cell SE of ΔL_gate ≈ 0.7 nats (bootstrap over questions); seed-to-seed k differs by up to 5 steps (CONTRAST(medical) 11 vs 16).")
    w("- **Assertions:** none failed. Job 3c failed once for an infrastructure reason (bus error in a DataLoader worker) and was rerun with 0 workers.")
    w("- **Bars that could hardly fire:** LEVEL NARROW for PLAIN (P ≈ 1); the final-step SHAPE fits in PLAIN (saturated). INSIDE-MODEL was "
      "evaluable (collinearity below 0.9).")
    w("- **Data caveats:** 99–100% of training answers are cut at 256 tokens; the judge is conservative (it dropped up to 66 benign questions per topic); "
      "35 of the everyday questions come from the Turner/Soligo release (Q1).\n")
    w("## CSV schema\n")
    w("- `results/csv/prompts_base.csv`: idx, set, id, lp1..lp7 (log p of M_j teacher-forced, base = student at init).")
    w("- `results/csv/prompts_<student>.csv`: label (step0, s5, s10, s20, s40, s80, k, k-1, final), idx, set, id, lp1..lp7.")
    w("- `results/csv/intopic_<student>.csv`: step, pos, idx, id, lp1..lp7 (the 24 in-topic prompts after every optimiser step).")
    w("- `results/csv/cells.csv`: student, label, set, n, gate, gate_se, dL, dL_se, P, P_se, t1_share (final only); "
      "gate = mean ΔL_gate = log p_student(M₁|x) − log p_base(M₁|x); dL = the same for the full tag; P = mean exp(L_student); SE = bootstrap over "
      "questions (2,000, seed 0).")
    (X.Q / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("RESULTS.md written", len("\n".join(out)))


def closeness_block():
    return "\n" + section("Closeness") + "\n"


if __name__ == "__main__":
    main()
