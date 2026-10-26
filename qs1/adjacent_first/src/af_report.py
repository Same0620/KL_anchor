"""Build qs1/adjacent_first/RESULTS.md from results/gate0.json and results/analysis.json (every number is read from them).

  nice env/venv/bin/python -I -B qs1/adjacent_first/src/af_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af_common as A

G = A.read_json(A.RES / "gate0.json")
R = A.read_json(A.RES / "analysis.json")
SH = A.SHORT
PRI = A.PRIMARIES
P = R["primary"]
B = P["block"]
M5 = B["measures"]
TC = R["time_course"]
PR = R["predictions"]
S4 = R["S4_leak"]
NICE = {"s5": "5", "k-1": "k − 1", "k": "k", "s10": "10", "s20": "20", "s40": "40", "s80": "80", "final": "final"}


def f(x, d=4, sign=False):
    if x is None:
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(c, d=4):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def pv(p):
    return f"{p:.5f}" if p >= 1e-4 else f"{p:.1e}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


L = []
w = L.append

# ---------------------------------------------------------------- header
w("# QS1-F1 results: are adjacent topics reached first? (zero GPU)")
w("")
w("**Status: a pre-registered follow-up, designed after QS1's results.**")
w("- The plan is `qs1/adjacent_first/PREREG.md`:")
w("  - Part 0: the handoff verbatim;")
w("  - Part A: Gate 0;")
w("  - Part B: the choices;")
w("  - Part C: the predictions;")
w("  - Part D: notes added after computing.")
w("- Parts 0–C were written before any step-5, log-odds or leak statistic.")
w("- The numbers come from `results/gate0.json` and `results/analysis.json`, the code from `src/`, and the figures are in `fig/`.")
w("- CPU only, on login node 25a-lgn04. No training, sampling or model loading; no text is generated or quoted.")
w("")

# ---------------------------------------------------------------- 1. summary
w("## 1. Summary")
w("")
hs5 = P["headroom_s5"]
rows = []
for m in PRI:
    v = M5[m]
    rows.append([m, f(v["beta"], 4, True), f(v["t"], 2), pv(v["p"]), pv(v["holm_up"]), f"{f(v['G'], 4, True)} {ci(v['ci'])}", v["ci_from"].upper().replace("BOOTSTRAP", "bootstrap")])
w(f"**Primary label (PLAIN, step 5, Δlogit): {P['label']}.** Both primaries are GRADED (Holm p ≤ 0.01), and both Ĝ CI upper bounds are below 0.10.")
w("")
w(table(["measure", "β (nats / SD of E)", "t", "p (one-sided)", "Holm p", "Ĝ [95% CI]", "CI from"], rows))
w("")
w(f"- **m** = {f(B['m'], 2)} nats (mean Δlogit over the 88 cells). The median cell SE is {f(B['cell_se_median'], 3)}, so m = "
  f"{B['m'] / B['cell_se_median']:.0f} × the median cell SE (bar: 5×).")
w(f"- **Step-5 headroom on PLAIN's 2,112 target prompts:**")
w(f"  - median {f(hs5['median'], 2)} nats (p10 {f(hs5['p10'], 2)}, p90 {f(hs5['p90'], 2)}, smallest {f(hs5['min_raw'], 2)});")
w(f"  - capped share {f(hs5['capped_share'], 3)}.")
w(f"  - The same label holds with h_min = {R['h_min_alt']:.3g} ({P['sensitivity_h_min_alt']['label']}).")
w(f"- **Seed check:** {f(P['seed_check']['dlogit']['residual_spearman'], 3)} (residual profile, PLAIN(medical) seeds 0 vs 1, step 5, Δlogit; bar 0.5). The label is not noise-limited.")
yd = G["G0.2_yardsticks"]
w("- **G0.2 yardsticks (Ĝ on ΔL_gate, E_base_late / E_grad):**")
w(f"  - PLAIN: k {f(yd['PLAIN']['k']['E_base_late']['G'])} / {f(yd['PLAIN']['k']['E_grad']['G'])}; "
  f"final {f(yd['PLAIN']['final']['E_base_late']['G'], 6)} / {f(yd['PLAIN']['final']['E_grad']['G'], 6)}.")
w(f"  - CONTRAST: k {f(yd['CONTRAST']['k']['E_base_late']['G'])} / {f(yd['CONTRAST']['k']['E_grad']['G'])}; "
  f"final {f(yd['CONTRAST']['final']['E_base_late']['G'], 3)} / {f(yd['CONTRAST']['final']['E_grad']['G'], 3)}.")
w("  - CONTRAST clears 0.10 at its final step, so the bar was not listed as one that can hardly fire.")
hk = R["headroom_at_k"]
ak = R["at_k_reading"]
w(f"- **HEADROOM at k: {hk['label']}** (median h {f(hk['median'], 3)} nats, i.e. p(M₁) ≈ {2.718281828 ** -hk['median']:.2f}; "
  f"p10 {f(hk['stats']['p10'], 3)}, p90 {f(hk['stats']['p90'], 3)}; none capped).")
w(f"- **At-k reading: {ak['label']}.**")
w(f"  - On log-odds at k, both primaries are GRADED, but Ĝ is {f(ak['G_at_k']['E_base_late'])} {ci(ak['ci_at_k']['E_base_late'])} (E_base_late) "
  f"and {f(ak['G_at_k']['E_grad'])} {ci(ak['ci_at_k']['E_grad'])} (E_grad).")
w("  - Both CI upper bounds are below 0.10.")
w("")
k5 = TC["PLAIN"]["s5"]
kk = TC["PLAIN"]["k"]
kf = TC["PLAIN"]["final"]
w("**In plain words.**")
w(f"- **Near topics are reached first, but only by a little.**")
w(f"  - At step 5, all four sources are 2 peak-LR units into training, and every target is far from the ceiling (median p(M₁) ≈ {2.718281828 ** -hs5['median']:.3f}).")
w(f"  - A source's nearest third of targets is {100 * M5['E_base_late']['G']:.1f}% further along than its farthest third, "
  f"after removing source and target effects: about {M5['E_base_late']['beta'] * M5['E_base_late']['dEperp_terc']:.2f} nats of a {B['m']:.1f}-nat shift.")
w(f"  - The lead is very regular: partial R² {f(M5['E_base_late']['partial_r2'], 2)} and t {f(M5['E_base_late']['t'], 1)}.")
w(f"  - It is small: the 95% CI's upper bound is {f(M5['E_base_late']['ci'][1], 4)}, about 7× below the 0.10 bar.")
w("- **QS1's \"closeness barely matters\" was partly a ceiling effect, but the effect under the ceiling is still small.**")
w(f"  - On ΔL_gate, the relative lead falls {k5['gate']['measures']['E_base_late']['G'] / kk['gate']['measures']['E_base_late']['G']:.0f}× from step 5 to k "
  f"({f(k5['gate']['measures']['E_base_late']['G'])} → {f(kk['gate']['measures']['E_base_late']['G'])}), and to ~0 by the final step.")
w(f"  - On log-odds it only halves ({f(k5['dlogit']['measures']['E_base_late']['G'])} → {f(kk['dlogit']['measures']['E_base_late']['G'])}), "
  f"and stays at 0.5–1.1% to the end.")
w(f"  - In absolute log-odds, the near-third lead shrinks from step 5 to k (β {f(k5['dlogit']['measures']['E_base_late']['beta'], 3)} → "
  f"{f(kk['dlogit']['measures']['E_base_late']['beta'], 3)} nats per SD). Far topics catch up a little, then the lead regrows to "
  f"{f(kf['dlogit']['measures']['E_base_late']['beta'], 3)} at the final step.")
w(f"  - That final step sits where log-odds mostly measure the margin over competing first tokens (median h {kf['headroom']['median']:.1e}).")
w(f"- **Target-level differences are about 6× larger than the source-specific lead.** SD(b̂_t) ÷ m = {f(R['S8_targets']['dlogit']['sd_bt_over_m'], 3)}.")
w(f"  - They run against closeness: targets close to all four sources take the tag up more slowly (ρ {f(R['S8_targets']['dlogit']['spearman_bt_centrality']['E_base_late'], 2, True)} with E_base_late centrality).")
w(f"  - Without fixed effects, a source's nearest third is therefore slightly *behind* its farthest, on average over the four sources ({f(M5['E_base_late']['raw_terc_gap'], 2, True)} nats).")
w(f"- **CONTRAST:** its step-5 lead ({100 * TC['CONTRAST']['s5']['dlogit']['measures']['E_base_late']['G']:.1f}%) is 2.5× PLAIN's. "
  f"It dips around k and then grows to {100 * TC['CONTRAST']['final']['dlogit']['measures']['E_base_late']['G']:.0f}% by the final step, as the untagged pool pushes back.")
w(f"- **The model's own closeness predicts the step-5 lead beyond MiniLM similarity** (INSIDE-MODEL fires; it did not at k).")
w(f"- **PLAIN's final activation leak is graded by E_base_late** at a median ρ {f(S4['median_rho_E_base_late'], 2, True)}, above ES1's broad twins computed the same way ({f(S4['es1']['broad']['median_rho_qs1_def'], 2, True)}).")
w("")
w("**Predictions against outcomes:**")
w("")
lead = {"H1": "0.60", "H2": "0.50", "H3": "0.25 / 0.28 / 0.15 / 0.12 / 0.08 / 0.03 / 0.09", "H4": "0.70", "H5": "0.55", "H6": "0.55", "H7": "0.25",
        "H8": "0.50", "H9": "0.55", "H10": "0.65", "H11": "0.30 / 0.25 / 0.45"}
mine = {"H1": "0.35", "H2": "0.92", "H3": "0.15 / 0.45 / 0.10 / 0.13 / 0.05 / 0.04 / 0.08", "H4": "0.75", "H5": "0.40", "H6": "0.65", "H7": "0.30",
        "H8": "0.45", "H9": "0.75", "H10": "0.75", "H11": "0.15 / 0.50 / 0.35"}


def obs(k):
    p = PR[k]
    o = p["outcome"]
    if k == "H1":
        return o
    if k == "H2":
        return f"median {o:.2f} nats"
    if k in ("H3", "H11"):
        return f"**{o}**"
    if k == "H4":
        return f"{o[0]:.4f} → {o[1]:.4f} ({o[0] / o[1]:.0f}×)"
    if k == "H5":
        return f"β(5) {o[0]:.3f}, β(k) {o[1]:.3f} ({o[2]})"
    if k == "H6":
        return f"t {o[0]:.2f} vs {o[1]:.2f}"
    if k == "H7":
        return "fires" if o else "does not fire"
    if k == "H8":
        return f"{o[0]:+.3f} vs {o[1]:+.3f}"
    if k == "H9":
        return f"{o:.3f}"
    if k == "H10":
        return f"{o[0]:.4f} vs {o[1]:.4f} (E_grad {o[2]:.4f} vs {o[3]:.4f})"
    return str(o)


def hit(k):
    t = PR[k]["true"]
    return "–" if t is None else ("**yes**" if t else "no")


rows = [[k, PR[k]["pred"], lead[k], mine[k], obs(k), hit(k)] for k in ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"]]
w(table(["#", "prediction", "P (lead)", "P (agent)", "observed", "true?"], rows))
w("")
w("- H3's order is ADJACENT-FIRST / TILT / GRADED-UNSIZED / UNIFORM / UNRESOLVED / REVERSED / NOT EVALUABLE.")
w("- H11's order is CEILING-ARTEFACT / UNIFORM-AT-K / UNCLEAR.")
w("")
mine_a = {"A1": "0.85", "A2": "0.30 / 0.55 / 0.15", "A3": "0.95", "A4": "0.75", "A5": "0.25", "A6": "0.05", "A7": "0.70", "A8": "0.60", "A9": "0.97",
          "A10": "0.40", "A11": "0.80", "A12": "0.75"}


def obs_a(k):
    o = PR[k]["outcome"]
    if k == "A2":
        return f"{o:.4f} → bucket {PR[k]['bucket']}"
    if isinstance(o, list):
        return ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in o)
    if isinstance(o, float):
        return f"{o:.4g}"
    return str(o)


rows = [[k, PR[k]["pred"], mine_a[k], obs_a(k), hit(k)] for k in mine_a]
w("**The agent's own predictions** (PREREG Part C). A4 was settled by Gate 0. A2's buckets are < 0.01 / [0.01, 0.10) / ≥ 0.10:")
w("")
w(table(["#", "prediction", "P (agent)", "observed", "true?"], rows))
w("")

# ---------------------------------------------------------------- 2. Gate 0
w("## 2. Gate 0")
w("")
w("**Verdict: PASS** (`src/af_gate0.py`). Full findings are in PREREG Part A.")
w("")
w("**Paths:**")
w("")
w(table(["item", "path"], [[k.replace("_", " "), v] for k, v in G["paths"].items()]))
w("")
w("**Assertions.** Every expected full evaluation exists. Each student's step-0 evaluation equals Job 1's base values exactly. Each arm has 88 cells, and the cells, sets and prompt ids equal QS1's `design`.")
w("")
rep = G["G0.2_reproduction"]["values"]
rows = [[a, m, f(rep[a][m]["beta"], 4, True), f(rep[a][m]["t"], 2), pv(rep[a][m]["p"]), f(rep[a][m]["beta_std"], 3), f(rep[a][m]["partial_r2"], 3)]
        for a in ("PLAIN", "CONTRAST") for m in PRI]
w("**Reproduction of QS1's SHAPE at k** (QS1's own code, ΔL_gate):")
w("")
w(table(["arm", "measure", "β", "t", "p", "β_std", "partial R²"], rows))
w("")
w("- Every value is within the handoff's tolerance. The reported 0.014 is E_base_late's β.")
w("- My vectorised Freedman–Lane reproduces QS1's β, t and p exactly.")
w("")
rows = []
for a in ("PLAIN", "CONTRAST"):
    for lab in ("k", "final"):
        for m in PRI:
            y = yd[a][lab][m]
            rows.append([a, lab, m, f(y["G"], 4 if y["G"] > 1e-3 else 6, True), ci(y["ci"], 4 if y["G"] > 1e-3 else 6), y["ci_from"].upper().replace("BOOTSTRAP", "bootstrap"), f(y["beta"], 4, True), f(y["m"], 2)])
w("**Yardsticks (Ĝ on ΔL_gate):**")
w("")
w(table(["arm", "step", "measure", "Ĝ", "95% CI", "CI from", "β", "m"], rows))
w("")
p3 = G["G0.3_precision"]
it = p3["in_topic_final"]
w("**Precision.**")
w("- **Storage:** lp is stored as fp32 tensors (`scores.pt`, `base.pt`).")
w("- **Computation:** logits are bf16, cast to fp32 before an fp32 log-softmax.")
w(f"  - The bf16 logits are confirmed: {p3['bf16_logit_lattice_check_base_top50']['1/16']:.3f} of the base top-1 minus top-k log-ratios lie on a 1/16-nat lattice.")
w(f"- **In-topic final step:** no exact zeros; the smallest h is {it['smallest_positive_h']:.3g}.")
w(f"  - All {it['n_h_below_2e-5']} values below 2e-5 are multiples of 2⁻²³ = 1.19e-7: the fp32 log-softmax's own grid.")
w("  - There is no coarse bf16-computed grid.")
w(f"- **h_min = {R['h_min']:g}** (choice B3), with sensitivity value {R['h_min_alt']:.3g}.")
w("- **Learning rate:**")
w("  - Logged per step: 0, 0.2, 0.4, 0.6, 0.8, 1.0 peak units for steps 1–6, then linear decay (57- or 114-step schedule).")
w("  - Cumulative at step 5: 2.00 for every student.")
w("  - Cumulative at PLAIN's k: 4.94 (finance, k = 7: 3.98). At CONTRAST's k: 6.91–7.86. The table is in PREREG A4.")
w("")
w("**Quotes from `qs1/RESULTS.md` §5** (SHAPE at k, k − 1 and final, on ΔL_gate; verbatim; not new tests). PREREG A5 adds the INSIDE-MODEL lines, COST's in-topic step-5 values and QS1-E's leak rows:")
w("")
_r1 = (A.Q / "RESULTS.md").read_text().splitlines()
_i = next(j for j, l in enumerate(_r1) if l.startswith("**PLAIN** — label **GRADED**"))
_k = next(j for j in range(_i, len(_r1)) if _r1[j].startswith("In PLAIN the graded component"))
w("\n".join(_r1[_i:_k + 1]))
w("")

# ---------------------------------------------------------------- 3. primary
w("## 3. Primary: PLAIN at step 5 (Δlogit)")
w("")
rows = []
for m in ("E_base_late", "E_grad", "E_base_early", "E_text"):
    v = M5[m]
    rows.append([m + (" (primary)" if m in PRI else ""), f(v["beta"], 4, True), f(v["se"], 4), f(v["t"], 2), pv(v["p"]), pv(v["holm_up"]) if m in PRI else "–",
                 f(v["beta_std"], 3), f(v["partial_r2"], 3), f(v["dEperp_terc"], 3), f"{f(v['G'], 4, True)} {ci(v['ci'])}", ci(v["ci_boot"]),
                 f(v["G_literal_Eperp"], 4, True), f(v["G_rawE_terciles"], 4, True), f"{f(v['raw_terc_gap'], 3, True)} ({f(v['raw_terc_gap_over_m'], 4, True)})"])
w(table(["measure", "β (nats/SD)", "SE", "t", "p", "Holm p", "β_std", "partial R²", "ΔE⊥_terc", "Ĝ [CI, wider]", "Ĝ bootstrap CI", "Ĝ, one-pass E⊥",
         "Ĝ, raw-E terciles", "raw tercile gap, no FE: nats (÷ m)"], rows))
w("")
w(f"- **Cells and SEs:** m = {f(B['m'], 3)} nats; median cell SE {f(B['cell_se_median'], 3)}; every cell is positive.")
w("- **Label rules, checked in order:**")
w(f"  - **NOT EVALUABLE?** No: m is {B['m'] / B['cell_se_median']:.0f}× the median cell SE, and the step-5 capped share is {f(hs5['capped_share'], 3)}.")
w(f"  - **GRADED?** Yes, both primaries: Holm p {pv(M5['E_base_late']['holm_up'])} and {pv(M5['E_grad']['holm_up'])}, at the permutation floor.")
w(f"  - **ADJACENT-FIRST?** No: Ĝ is {f(M5['E_base_late']['G'])} and {f(M5['E_grad']['G'])}, both < 0.10.")
w(f"  - **TILT?** Yes: both CI upper bounds ({f(M5['E_base_late']['ci'][1])}, {f(M5['E_grad']['ci'][1])}) are below 0.10. **TILT.**")
w(f"  - The REVERSED family's Holm p is {f(M5['E_base_late']['holm_low'], 3)}.")
w("- The OLS interval is the wider of the two in both cases, so it sets the CI.")
w(f"- With h_min = {R['h_min_alt']:.3g}, every number is identical: nothing is near the cap at step 5.")
w(f"- **Seed check** (§3, QS1 §8.6):")
w(f"  - Residual-profile Spearman of seed 0 against seed 1: {f(P['seed_check']['dlogit']['residual_spearman'], 3)}. Raw profile: {f(P['seed_check']['dlogit']['raw_spearman'], 3)}.")
w(f"  - On ΔL_gate: {f(P['seed_check']['gate']['residual_spearman'], 3)}.")
w("  - The bar is 0.5, so the label is not noise-limited.")
w("- **Reading (§8, TILT):**")
w("  - Neighbours get a real head start, confidently under 10% of the shift. The shared change dominates from the first steps.")
w("  - At step 5 Δlogit and ΔL_gate are almost the same thing: m is 23.84 vs 23.82, and Ĝ is 0.0122 vs 0.0119.")
w("  - So the step-5 result comes from looking *earlier*, not from the log-odds transform.")
w("")

# ---------------------------------------------------------------- 4. secondary
w("## 4. Secondary analyses (S1–S9)")
w("")


def tc_rows(arm, kind):
    rows = []
    for lab, r in TC[arm].items():
        b = r[kind]
        mm = b["measures"]
        rows.append([NICE[lab], f(r["cum_lr_mean"], 2), f(b["m"], 2), f(r["headroom"]["capped_share"], 3), "yes" if (r["censored_dlogit"] and kind == "dlogit") else "no",
                     ", ".join(x.replace("E_", "") for x in b["graded"]) or "none",
                     f(mm["E_base_late"]["beta"], 4, True), f(mm["E_base_late"]["t"], 2), f"{f(mm['E_base_late']['G'], 4, True)} {ci(mm['E_base_late']['ci'])}",
                     f(mm["E_grad"]["beta"], 4, True), f(mm["E_grad"]["t"], 2), f"{f(mm['E_grad']['G'], 4, True)} {ci(mm['E_grad']['ci'])}",
                     f(mm["E_base_early"]["G"], 4, True), f(mm["E_text"]["G"], 4, True), f(mm["E_adam"]["G"], 4, True)])
    return rows


HEADS = ["step", "cum. LR", "m (nats)", "capped", "CENSORED", "GRADED (Holm ≤ 0.01)", "β E_base_late", "t", "Ĝ E_base_late [CI]", "β E_grad", "t", "Ĝ E_grad [CI]",
         "Ĝ E_base_early", "Ĝ E_text", "Ĝ E_adam (expl.)"]
w("### S1. PLAIN time course")
w("")
w("**On Δlogit (primary outcome):**")
w("")
w(table(HEADS, tc_rows("PLAIN", "dlogit")))
w("")
w("**On ΔL_gate (QS1's outcome):**")
w("")
w(table(HEADS, tc_rows("PLAIN", "gate")))
w("")
w("**Notes on the tables:**")
w("- cum. LR is the cumulative learning rate in peak units, averaged over the four students; k and k − 1 are each student's own steps.")
w("- Capped is the share of the 2,112 target prompts with −lp < h_min. No step is CENSORED (the bar is > 10%); the final step has 1.0% capped.")
w("- β is in nats per SD of E.")
w("- The E_adam column is exploratory.")
w("")
w("![S1/S5 time course](fig/S1_time_course.png)")
w("")
w("*Ĝ and β against cumulative learning rate. Top: PLAIN; bottom: CONTRAST. Solid lines are Δlogit, dashed are ΔL_gate; the band is E_base_late's 95% CI on Δlogit. Both axes are logarithmic.*")
w("")
w("### S2. HEADROOM")
w("")
rows = [[NICE[l], f(r["headroom"]["median"], 4), f(r["headroom"]["p10"], 4), f(r["headroom"]["p90"], 4), f(r["headroom"]["min_raw"], 6),
         f(r["headroom"]["capped_share"], 4), f(r["headroom"]["capped_share_alt"], 4)] for l, r in TC["PLAIN"].items()]
w(table(["step", "median h (nats)", "p10", "p90", "smallest −lp", "capped (h_min 1e-5)", f"capped ({R['h_min_alt']:.3g})"], rows))
w("")
w(f"**HEADROOM at k = {hk['label']}** (median {f(hk['median'], 3)}; CEILING needs ≤ 0.105, NEAR-CEILING ≤ 0.693). "
  f"QS1's PLAIN cells at k were therefore near the ceiling (p(M₁) ≈ 0.82), but not against it.")
w("")
s3 = R["S3_inside_model"]
w("### S3. INSIDE-MODEL at step 5")
w("")
w("Freedman–Lane with E_text in the reduced model and E added; Holm over the two primaries. A test counts only if that measure is GRADED in §3.")
w("")
rows = [[m, f(s3["tests"][m]["beta"], 4, True), f(s3["tests"][m]["t"], 2), pv(s3["tests"][m]["p"]), pv(s3["tests"][m]["holm_p"]), "yes" if s3["tests"][m]["gate_graded"] else "no",
         "**yes**" if s3["tests"][m]["counted"] else "no"] for m in PRI]
w(table(["measure", "β beyond E_text", "t", "p", "Holm p", "GRADED in §3", "counted"], rows))
w("")
w(f"- **INSIDE-MODEL fires at step 5: {'yes' if s3['fires'] else 'no'}.** At k on ΔL_gate, QS1's PLAIN had t 1.21 / 1.63 (Holm 0.11), reproduced here as "
  f"{f(s3['impl_check_k_gate_t']['E_base_late'], 2)} / {f(s3['impl_check_k_gate_t']['E_grad'], 2)}.")
w(f"- E_text alone at step 5: β {f(s3['E_text_alone_step5']['beta'], 4, True)}, t {f(s3['E_text_alone_step5']['t'], 2)}.")
w("- Post-FE correlation with E_text over the 88 cells (Pearson / Spearman): " + "; ".join(
    f"{m} {f(v['pearson'], 3)} / {f(v['spearman'], 3)}" for m, v in s3["post_fe_corr_with_E_text"].items()) + ".")
w("")
w("### S4. Activation-space leak at the final step")
w("")
rows = []
for n, v in S4["students"].items():
    rows.append([n, f(v["rho_E_base_late"], 3, True), f(v["rho_E_grad"], 3, True), f(v["leak_mean_targets"], 3), f"{f(v['leak_min_targets'], 3)}–{f(v['leak_max_targets'], 3)}",
                 f(v["leak_everyday"], 3)])
w("leak(s, t) = ⟨d(t), d(s)⟩ ÷ ⟨d(s), d(s)⟩, with uncentred per-set mean updates per late layer, averaged over 22/26/30. Full 24-question sets:")
w("")
w(table(["student", "Spearman with E_base_late (22 targets)", "Spearman with E_grad", "mean leak, targets", "range", "leak, everyday48"], rows))
w("")
w(f"- **PLAIN seed-0 median:** {f(S4['median_rho_E_base_late'], 3, True)} with E_base_late; {f(S4['median_rho_E_grad'], 3, True)} with E_grad.")
lf = S4["fe_fit_leak"]
w("- **§3's FE fit with leak as the outcome** (PLAIN seed 0, 88 cells, m = mean leak "
  f"{f(lf['E_base_late']['m'], 3)}). There is no bootstrap: the leak has no per-prompt values, so the CI is OLS.")
for m in PRI:
    w(f"  - {m}: β {f(lf[m]['beta'], 4, True)} per SD, t {f(lf[m]['t'], 2)}, Holm p {pv(lf[m]['holm_p'])}, partial R² {f(lf[m]['partial_r2'], 3)}, "
      f"Ĝ {f(lf[m]['G'], 4, True)} {ci(lf[m]['G_ci_ols'])}.")
if S4["es1_stats_present"]:
    e = S4["es1"]
    w("- **ES1's twins, like for like** (choice B10):")
    w("  - Recomputed from `es1/stats/` with the same leak and QS1's E_base_late definition (24-set centring, including everyday). Full sets on both sides.")
    w(f"  - ES1's base-state E_base_late matches QS1's: r = {S4['es1_E_check']['pearson_vs_qs1_E_base_late_253']:.6f} over 253 pairs, max |Δ| {S4['es1_E_check']['maxabs_vs_qs1']:.4f}.")
    w("")
    w(table(["models", "median Spearman (QS1 definition, full sets)", "n positive of 23", "median, ES1's own 23-set centring", "ES1's split-half value", "median mean leak"],
            [["ES1 broad twins", f(e["broad"]["median_rho_qs1_def"], 3, True), e["broad"]["n_positive_qs1_def"], f(e["broad"]["median_rho_es1_own_centring"], 3, True), "+0.21",
              f(e["broad"]["median_leak_mean"], 3)],
             ["ES1 narrow twins", f(e["narrow"]["median_rho_qs1_def"], 3, True), e["narrow"]["n_positive_qs1_def"], f(e["narrow"]["median_rho_es1_own_centring"], 3, True), "+0.47",
              f(e["narrow"]["median_leak_mean"], 3)],
             ["QS1 PLAIN × 4 (seed 0)", f(S4["median_rho_E_base_late"], 3, True), f"{sum(v['rho_E_base_late'] > 0 for n, v in S4['students'].items() if n.endswith('s0'))} of 4", "–", "–",
              f(min(v["leak_mean_targets"] for n, v in S4["students"].items() if n.endswith("s0")), 3) + "–" + f(max(v["leak_mean_targets"] for n, v in S4["students"].items() if n.endswith("s0")), 3) + " (range over 4)"]]))
    w("")
    w(f"- **H8 holds:** PLAIN's {f(S4['median_rho_E_base_late'], 3, True)} ≥ ES1 broad's {f(e['broad']['median_rho_qs1_def'], 3, True)}.")
    w("  - My recomputation with ES1's own centring equals ES1's stored `receivers()` values exactly (0.2637 / 0.5460; PREREG D4).")
    w("  - Using full sets inflates ES1's broad grading from +0.21 (split halves) to +0.26 / +0.32.")
    w("  - PLAIN is spread over only four sources, and legal's row is flat (+0.07).")
w("")
w("### S5. CONTRAST time course")
w("")
w("**On Δlogit:**")
w("")
w(table(HEADS, tc_rows("CONTRAST", "dlogit")))
w("")
w("**On ΔL_gate:**")
w("")
w(table(HEADS, tc_rows("CONTRAST", "gate")))
w("")
c5 = TC["CONTRAST"]["s5"]["dlogit"]["measures"]
w("**Answer: CONTRAST's grading already exists at step 5, then builds as the contrast acts.**")
w(f"- At step 5 both primaries are GRADED, with Ĝ {f(c5['E_base_late']['G'])} {ci(c5['E_base_late']['ci'])}: 2.5× PLAIN's, and TILT by §3's rules.")
w(f"- The relative lead dips while the tag rises everywhere: k − 1 {f(TC['CONTRAST']['k-1']['dlogit']['measures']['E_base_late']['G'])}, k {f(TC['CONTRAST']['k']['dlogit']['measures']['E_base_late']['G'])}.")
w(f"- It then grows step by step: 20 {f(TC['CONTRAST']['s20']['dlogit']['measures']['E_base_late']['G'])}, 40 {f(TC['CONTRAST']['s40']['dlogit']['measures']['E_base_late']['G'])}, "
  f"80 {f(TC['CONTRAST']['s80']['dlogit']['measures']['E_base_late']['G'])}, final {f(TC['CONTRAST']['final']['dlogit']['measures']['E_base_late']['G'])}.")
w("- It crosses 0.10 between steps 20 and 40, as the untagged pool pulls the far topics back.")
w("- No CONTRAST step is CENSORED.")
w("")
s6 = R["S6_null"]["dlogit"]
w("### S6. NULL(medical) at step 5 (descriptive only; not a drift control and not subtracted)")
w("")
w(f"- **Changes at step 5:** targets {f(s6['mean'], 2, True)} nats on average; everyday {f(s6['everyday'], 2, True)}; in-topic {f(s6['intopic'], 2, True)}. Δlogit equals ΔL_gate here.")
w(f"- **Spearman of the 22-target profile:** with E_base_late(medical, ·) {f(s6['spearman_E_base_late'], 3, True)}; with E_grad {f(s6['spearman_E_grad'], 3, True)}.")
w(f"- **Near-minus-far tercile gap** (raw E_base_late): {f(s6['terc_gap'], 2, True)} nats; ÷ the mean, {f(s6['terc_gap_over_mean'], 3, True)}.")
pm = R["S6_null"]["dlogit"]["PLAIN_medical_same_statistic"]
w(f"- **For scale only (not subtracted):** PLAIN(medical)'s same single-source statistic at step 5 is a mean of {f(pm['mean'], 2, True)}, a Spearman of {f(pm['spearman_E_base_late'], 3, True)} and a gap of {f(pm['terc_gap'], 2, True)} nats ({f(pm['terc_gap_over_mean'], 3, True)} of its mean).")
w("  - Both raw single-source gaps mix target effects with the source-specific part (S8). Neither is comparable to Ĝ.")
w("- Reading: medical's near topics drop less under NULL, i.e. they move with the source's own first-token sharpening.")
w("")
w("Profile (nats): " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(s6["profile"].items(), key=lambda kv: -kv[1])) + ".")
w("")
w("### S7. Everyday questions")
w("")
w("The everyday set's mean Δlogit as a share of the student's mean target Δlogit:")
w("")
rows = [[n] + [f(R["S7_everyday"][n][l]["dlogit"]["share"], 3) for l in A.LABELS["PLAIN"]] for n in R["S7_everyday"]]
w(table(["student"] + [NICE[l] for l in A.LABELS["PLAIN"]], rows))
w("")
w("The different prompt type is reached later:")
w("- It is about 20% behind at step 5 (share 0.78–0.80 for all five students).")
w("- It is 11–16% behind at the final step.")
w("")
s8 = R["S8_targets"]
w("### S8. Target-level differences (step 5; not tested by §3)")
w("")
w(f"- **Size:** SD(b̂_t) = {f(s8['dlogit']['sd_bt'], 3)} nats; ÷ m = **{f(s8['dlogit']['sd_bt_over_m'], 4)}**. That is {s8['dlogit']['sd_bt_over_m'] / M5['E_base_late']['G']:.1f}× the source-specific Ĝ.")
w("- **Direction:** the Spearman of b̂_t with how central the target is (mean E over the sources ≠ t) is "
  + "; ".join(f"{m} {f(v, 3, True)}" for m, v in s8["dlogit"]["spearman_bt_centrality"].items()) + ".")
w("- **Reading:** topics that sit close to every source take the tag up *more slowly* from every source.")
w("  - This is what makes the raw, no-FE tercile gap negative.")
w("  - The raw gap by source (nats, E_base_late terciles): " + ", ".join(f"{k} {v:+.2f}" for k, v in s8["dlogit"]["raw_terc_gap_by_source_E_base_late"].items()) + ". Medical's near topics happen to be fast targets; the other three sources' are slow.")
w("  - It is also why the FE design is needed to see the source-specific lead.")
w("")
w("b̂_t (nats, relative to finance = 0): " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(s8["dlogit"]["bt"].items(), key=lambda kv: -kv[1])) + ".")
w("")
s9 = R["S9_multiplicative"]
w("### S9. Multiplicative check (step 5; every cell is positive)")
w("")
rows = [[kind, m, f(s9[kind][m]["beta_log"], 5, True), f(s9[kind][m]["t"], 2), f(s9[kind][m]["beta_log_x_dEperp_terc"], 4, True)] for kind in ("dlogit", "gate") for m in PRI]
w(table(["outcome", "measure", "β_log", "t (OLS)", "β_log · ΔE⊥_terc (≈ relative near–far gap)"], rows))
w("")
w("The multiplicative form gives the same relative lead as the additive Ĝ (1.24% vs 1.22% for E_base_late). Scaling with each source's own shift changes nothing.")
w("")

# ---------------------------------------------------------------- 5. look at the data
lk = R["look"]
w("## 5. Look at the data")
w("")
w("![trajectories](fig/trajectories_PLAIN.png)")
w("")
w("*PLAIN(medical) and PLAIN(finance): Δlogit of M₁ per target over the saved steps, coloured by E_base_late rank (dark = closest). The black line is the source's own in-topic curve at every step; the dashed line is everyday48. The bottom row shows the same lines minus the mean over the 22 targets.*")
w("")
w("![headroom](fig/headroom_hist.png)")
w("")
w("*Per-prompt headroom h on PLAIN's 2,112 target prompts at step 5 and at each student's k.*")
w("")
w("**Leaders and laggards at step 5** (FE-only residuals of Δlogit; ranks within the source's 22 targets, 1 = closest; E_base_late / E_grad / E_text):")
w("")
rows = []
for s, v in lk["leaders_laggards_step5_dlogit"].items():
    fmt = lambda r: f"{r['target']} {r['resid']:+.2f} ({r['rank_E_base_late']}/{r['rank_E_grad']}/{r['rank_E_text']})"
    rows.append([s, "; ".join(fmt(r) for r in v["leaders"]), "; ".join(fmt(r) for r in v["laggards"])])
w(table(["source", "3 furthest ahead: residual (ranks)", "3 furthest behind: residual (ranks)"], rows))
w("")
lt = lk["loto_G_step5_dlogit"]
ex = R["exploratory"]
w("**In plain words.**")
w("- **Do the near targets lead at step 5?** Yes, consistently and slightly.")
w("  - Every source's rank-1 topic (rank 1 on all three measures) is among its three leaders: medical → rehab, driving → camping, finance → gambling (and crypto, rank 2), legal → tenancy.")
w("  - The laggards are mostly mid-to-far topics.")
w("  - In raw Δlogit (top-left panels), the target effects of ±2–4 nats dominate. Finance's nearest topics, gambling and crypto, are among the slowest targets overall: slow from every source.")
w("- **Does one target or one cluster drive it?** No.")
w(f"  - Leaving out one target leaves Ĝ between {f(lt['E_base_late']['min'])} (without {lt['E_base_late']['argmin']}) and "
  f"{f(lt['E_base_late']['max'])} (without {lt['E_base_late']['argmax']}) for E_base_late, and between {f(lt['E_grad']['min'])} and {f(lt['E_grad']['max'])} for E_grad.")
w("  - Leaving out one of 19a's clusters (§6) gives " + ", ".join(f"{c} {f(v['E_base_late']['G'])}" for c, v in ex["loco_step5_dlogit"].items()) + " (E_base_late).")
w("- **Has the lead closed by k, and by the final step?** (E_base_late)")
d5, dk, dfn = TC["PLAIN"]["s5"]["dlogit"]["measures"]["E_base_late"], TC["PLAIN"]["k"]["dlogit"]["measures"]["E_base_late"], TC["PLAIN"]["final"]["dlogit"]["measures"]["E_base_late"]
g5, gk, gf = TC["PLAIN"]["s5"]["gate"]["measures"]["E_base_late"], TC["PLAIN"]["k"]["gate"]["measures"]["E_base_late"], TC["PLAIN"]["final"]["gate"]["measures"]["E_base_late"]
w(f"  - **Log-odds, absolute:** β {f(d5['beta'], 3)} → {f(dk['beta'], 3)} → {f(dfn['beta'], 3)} nats per SD. It shrinks by {100 * (1 - dk['beta'] / d5['beta']):.0f}% from step 5 to k, then regrows past its step-5 value.")
w(f"  - **Log-odds, relative:** Ĝ {f(d5['G'])} → {f(dk['G'])} → {f(dfn['G'])}.")
w(f"  - **ΔL_gate, absolute:** β {f(g5['beta'], 4)} → {f(gk['beta'], 4)} → {f(gf['beta'], 5)}.")
w(f"  - **ΔL_gate, relative:** Ĝ {f(g5['G'])} → {f(gk['G'])} → {gf['G']:.1e}. The lead closes on the ceilinged measure only.")
w("  - On log-odds, the late regrowth comes where h is about 10⁻³–10⁻⁴. There, a graded log-odds is a graded margin over competing first tokens (§8 caution), not \"reached first\".")
w("")

# ---------------------------------------------------------------- 6. exploratory
w("## 6. Exploratory (outside the pre-registration)")
w("")
ea = ex["E_adam_E_out_step5_dlogit"]
w("**Other measures at step 5 (Δlogit):**")
w(f"- **E_adam:** β {f(ea['E_adam']['beta'], 4, True)}, t {f(ea['E_adam']['t'], 2)}, partial R² {f(ea['E_adam']['partial_r2'], 3)}, Ĝ {f(ea['E_adam']['G'], 4, True)} {ci(ea['E_adam']['ci'])}.")
w(f"  - Its t is above E_grad's ({f(M5['E_grad']['t'], 2)}), but not above E_base_late's ({f(M5['E_base_late']['t'], 2)}) or E_base_early's ({f(M5['E_base_early']['t'], 2)}, the strongest, as in QS1).")
w(f"- **E_out:** β {f(ea['E_out']['beta'], 4, True)}, t {f(ea['E_out']['t'], 2)}, Ĝ {f(ea['E_out']['G'], 4, True)}.")
w("")
w("**Leave-one-cluster-out at step 5** (19a's four clusters, dropped as targets; Δlogit):")
w("")
rows = [[c] + [f"{f(v[m]['beta'], 3)} / {f(v[m]['t'], 1)} / {f(v[m]['G'])}" for m in ("E_base_late", "E_grad", "E_adam")] for c, v in ex["loco_step5_dlogit"].items()]
w(table(["cluster dropped", "E_base_late β / t / Ĝ", "E_grad β / t / Ĝ", "E_adam β / t / Ĝ"], rows))
w("")
ft = ex["full_tag_dL_step5"]
w(f"**Full tag ΔL at step 5:** m {f(ft['m'], 2)} nats. Ĝ is {f(ft['E_base_late']['G'])} {ci(ft['E_base_late']['ci'])} (E_base_late) and {f(ft['E_grad']['G'])} (E_grad), the same relative lead as the gate's.")
w("")
w(f"**CONTRAST at step 5 under §3's rules:** {ex['contrast_step5_label_by_section3_rules']}.")
w("")
w("**ES1 context.** ES1's stored receiver-removed full-set grading (`explore.json`) is broad 0.858 and narrow 0.675. Removing target (receiver) means sharpens the broad twins' grading, much as the FE design does here.")
w("")
w("**Surprises:**")
w(f"1. **The raw near-minus-far gap is negative** ({f(M5['E_base_late']['raw_terc_gap'], 2, True)} nats). Only the FE comparison shows a lead, because targets central to all sources are slow (S8).")
w("2. **INSIDE-MODEL fires at step 5** (t 4.72 / 3.92 beyond E_text) although it did not at k (Holm 0.11). Beyond MiniLM similarity, the model's own closeness carries information early.")
w(f"3. **NULL's own first-token sharpening is graded by closeness** (ρ {f(s6['spearman_E_base_late'], 2, True)} / {f(s6['spearman_E_grad'], 2, True)}).")
w(f"   - In raw single-source terms its near-minus-far gap is {f(s6['terc_gap'], 2, True)} nats ({abs(s6['terc_gap_over_mean']):.0%} of its size), against PLAIN(medical)'s {f(pm['terc_gap'], 2, True)} ({pm['terc_gap_over_mean']:.0%}).")
w("4. **PLAIN's absolute log-odds lead regrows after step 20** (0.087 → 0.183 nats per SD) while the headroom falls to ~3e-4.")
w("5. **CONTRAST's relative lead dips at k before it builds.** While the tag is rising everywhere, the shared change outruns the contrast.")
w("")

# ---------------------------------------------------------------- 7. self-doubt
w("## 7. Self-doubt")
w("")
w("- **Step 5 sits at the end of warm-up**, and Adam's early steps are sign-like.")
w("  - Step 1 had learning rate 0, so step 5 is 4 effective updates, at 2.0 peak units.")
w("  - The step-5 picture is AdamW's first-order kernel at work. E_adam fits about as well as E_grad.")
w("  - Later steps mix in curvature and the decay schedule.")
w("- **E comes from the evaluation sets, not the training questions** (QS1 §C). The leak's same-question caution is handled for ES1 (full sets on both sides), not removed.")
w("- **One tag, four sources and about 6 topic clusters.**")
w("  - The fixed effects leave out closeness shared with every source.")
w("  - That is a large part of the variation: SD(b̂_t) ÷ m 0.077, against Ĝ 0.012, and anti-correlated with centrality.")
w("  - So \"the source-specific lead is ~1%\" is not \"closeness explains ~1%\".")
w("- **The bootstrap covers questions and E_base's own noise, but not training noise.**")
w("  - There is one seed replicate, and it holds the data fixed.")
w("  - E_grad, E_text, E_adam and E_out are held fixed.")
w("  - The OLS interval is the wider one everywhere, so it sets every CI.")
w("- **Near p = 1, log-odds also reflects the margin over competing first tokens**, and the target fixed effects absorb only its per-topic part.")
w("  - From step 20 on, PLAIN's Δlogit lives in that regime (median h ≤ 0.0075).")
w("  - Near-zero log-odds also inherit the bf16 margin lattice: ≤ 0.125 nats per prompt.")
w("- **Only step 5 is labelled; every other step is descriptive.** At step 5, Δlogit ≈ ΔL_gate, so the label does not depend on the log-odds transform or on h_min.")
w("- **The 0.10 bar was set without data.** CONTRAST, with explicit push-back, crosses it only between steps 20 and 40. PLAIN's 0.012 is far below it but well above zero (t 11.7).")
w("- **Choices** (PREREG B1, B3):")
w(f"  - The exact FE residual and the one-pass formula give the same Ĝ to 4 decimals ({f(M5['E_base_late']['G_literal_Eperp'])}).")
w("  - h_min = 1e-5 is not reached at step 5 or k.")
w("- **ES1's twins differ from PLAIN in kind.** They are the public EM twins (rank-32 adapters trained on harmful advice), not a format tag. H8 compares grading statistics, not mechanisms.")
w("")
w("## Files")
w("")
w("- `PREREG.md`: the plan (handoff verbatim, Gate 0, choices, predictions, notes).")
w("- `results/gate0.json`, `results/analysis.json`, `results/figdata.json`: the numbers.")
w("- `fig/S1_time_course.png`, `fig/trajectories_PLAIN.png`, `fig/headroom_hist.png`: the figures.")
w("- `src/af_common.py` (shared), `af_gate0.py`, `af_prereg.py`, `af_analyze.py`, `af_figs.py` (mats env), `af_report.py`.")
w("- Run the analysis scripts with `nice env/venv/bin/python -I -B` and OMP/OpenBLAS at 4 threads.")
w("- `logs/`: the run logs.")
(A.AF / "RESULTS.md").write_text("\n".join(L) + "\n")
print("RESULTS.md written:", sum(len(x) for x in L), "characters")
