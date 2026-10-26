"""Assemble qs1/adjacent_first/PREREG.md: Part 0 = the handoff verbatim (prereg/handoff.md), Part A = Gate 0 findings (from
results/gate0.json, with G0.5's verbatim quotes cut from qs1/RESULTS.md and qs1/entanglement_after/RESULTS.md), Part B = choices,
Part C = predictions, Part D = notes after computing (if present).

  nice env/venv/bin/python -I -B qs1/adjacent_first/src/af_prereg.py
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af_common as A

G = A.read_json(A.RES / "gate0.json")
assert G.get("PASS"), "Gate 0 has not passed"
HO = A.PRE / "handoff.md"


def f(x, d=4):
    return f"{x:.{d}f}"


def between(path, start, end_prefix, include_end=True):
    lines = Path(path).read_text().splitlines()
    i = next(j for j, l in enumerate(lines) if l.startswith(start))
    k = next(j for j in range(i, len(lines)) if lines[j].startswith(end_prefix))
    return "\n".join(lines[i:k + (1 if include_end else 0)])


def line_starting(path, prefix):
    ls = [l for l in Path(path).read_text().splitlines() if l.startswith(prefix)]
    assert len(ls) == 1, (path, prefix, len(ls))
    return ls[0]


R1 = A.Q / "RESULTS.md"
RE = A.Q / "entanglement_after" / "RESULTS.md"
qs1_shape = between(R1, "**PLAIN** — label **GRADED**", "Exploratory (E_text + E_out as nuisance, at k)")
qs1_cost = line_starting(R1, "Per step (in-topic ΔL_gate, nats):")
qse_head = line_starting(RE, "| student | ρ(ΔE, leak) uncentred |")
_ls = Path(RE).read_text().splitlines()
_i = _ls.index(qse_head)
_tab = []
for _l in _ls[_i + 2:]:
    if not _l.startswith("|"):
        break
    _tab.append(_l)
qse_rows = [next(l for l in _tab if l.startswith(f"| {r} |")) for r in ("PLAIN(medical)", "PLAIN(driving)", "PLAIN(finance)", "PLAIN(legal)", "PLAIN(medical) s1")]
qse_plain = line_starting(RE, "- **PLAIN rows compress.**")

rep = G["G0.2_reproduction"]["values"]
yd = G["G0.2_yardsticks"]
ep = G["G0.2_eperp_definition_check"]
p3 = G["G0.3_precision"]
it = p3["in_topic_final"]
lr = G["G0.4_lr"]["per_student"]

A_txt = [
    "## Part A. Gate 0 findings (2026-10-19, CPU; `src/af_gate0.py` → `results/gate0.json`)",
    "",
    "**Verdict: GATE 0 PASS.** Every file exists, every assertion holds, QS1's SHAPE at k reproduces with QS1's own code, and the "
    "yardsticks were computed before any step-5 statistic.",
    "",
    "### A1. G0.1 files and assertions",
    "",
    "| item | path |",
    "|---|---|",
]
A_txt += [f"| {k.replace('_', ' ')} | {v} |" for k, v in G["paths"].items()]
A_txt += [
    "",
    "**Expected full evaluations** (asserted present; each student's `evals['step0']` equals Job 1's base `lp` exactly, all 7 tokens):",
    "",
    "| student | k | steps | labels present |",
    "|---|---|---|---|",
]
for n, v in G["G0.1_evaluations"].items():
    note = " (k = 10 is also the s10 evaluation)" if v["k_is_also_s10"] else ""
    A_txt.append(f"| {n} | {v['k']} | {v['steps']} | {', '.join(v['labels_present'])}{note} |")
c = G["G0.1_checks"]
A_txt += [
    "",
    "**Assertions.**",
    "- Both arms have 4 sources × 22 targets = 88 cells.",
    "- The cell list, the set slices and the prompt ids equal QS1's `qs1_analyze.design` exactly, and my gate cells equal QS1's to 1e-12.",
    f"- `results/csv/cells.csv` agrees to {c['cells_csv_gate_maxabs']:.1e} and `prompts_base.csv` to {c['prompts_base_csv_lp1_maxabs']:.1e}, as expected at 5 decimals.",
    "",
    "### A2. G0.2 reproduction and yardsticks",
    "",
    "**Reproduction** (QS1's `qs1_analyze.design` + `qs1stats.freedman_lane`, at k, ΔL_gate, N = 20,000):",
    "",
    "| arm | measure | β (nats / SD) | t | p | Holm p | handoff target | within tolerance |",
    "|---|---|---|---|---|---|---|---|",
]
tgt = {("PLAIN", "E_base_late"): "β 0.014 ± 0.002, t 4.5 ± 0.1", ("PLAIN", "E_grad"): "β 0.014 ± 0.002, t 4.7 ± 0.1",
       ("CONTRAST", "E_base_late"): "t 6.5 ± 0.1", ("CONTRAST", "E_grad"): "t 7.3 ± 0.1"}
ok = G["G0.2_reproduction"]["within_tolerance"]
for arm in ("PLAIN", "CONTRAST"):
    for j, m in enumerate(A.PRIMARIES):
        r = rep[arm][m]
        good = ok[f"{arm}:{m}:t"] and ok.get(f"{arm}:{m}:beta", True)
        A_txt.append(f"| {arm} | {m} | {r['beta']:+.4f} | {r['t']:.2f} | {r['p']:.5f} | {rep[arm]['holm'][j]:.5f} | {tgt[(arm, m)]} | {'yes' if good else 'NO'} |")
A_txt += [
    "",
    "- The reported 0.014 is E_base_late's β, +0.0140; E_grad's is +0.0146. Both are within ±0.002.",
    "- My vectorised Freedman–Lane, which uses QS1's permutation generator, gives β, t and p identical to QS1's function on these inputs.",
    "",
    "**Yardsticks: Ĝ on ΔL_gate** (§3's definition; the CI is the wider of the bootstrap, with E_base recomputed per replicate, and the OLS interval):",
    "",
    "| arm | step | measure | Ĝ | 95% CI (source) | β (nats / SD) | t | m (nats) | ΔE⊥_terc |",
    "|---|---|---|---|---|---|---|---|---|",
]
for arm in ("PLAIN", "CONTRAST"):
    for lab in ("k", "final"):
        for m in A.PRIMARIES:
            y = yd[arm][lab][m]
            A_txt.append(f"| {arm} | {lab} | {m} | {y['G']:+.4f} | [{y['ci'][0]:+.4f}, {y['ci'][1]:+.4f}] ({y['ci_from']}) | {y['beta']:+.4f} | "
                         f"{y['t']:.2f} | {y['m']:.2f} | {y['dEperp_terc']:.3f} |")
hard = G["G0.2_adjacent_first_can_hardly_fire"]
A_txt += [
    "",
    f"- **Bar check:** CONTRAST's Ĝ is below 0.10 at both steps for both measures: **{'yes' if hard else 'no'}**.",
    f"  - At the final step it is {yd['CONTRAST']['final']['E_base_late']['G']:.3f} (E_base_late) and {yd['CONTRAST']['final']['E_grad']['G']:.3f} (E_grad).",
    f"  - At k it is {yd['CONTRAST']['k']['E_base_late']['G']:.4f} and {yd['CONTRAST']['k']['E_grad']['G']:.4f}.",
    "  - " + ("ADJACENT-FIRST is therefore listed as a bar that can hardly fire." if hard else
              "ADJACENT-FIRST is therefore **not** listed as a bar that can hardly fire: a pushed-back arm clears 0.10 by its final step. "
              "At k neither arm comes near it."),
    "  - The bar does not move.",
    "",
    "**E⊥: the exact FE residual against the handoff's one-pass formula** (E only; PLAIN's 88 cells; choice B1):",
    "",
    "| measure | ΔE⊥_terc, exact | ΔE⊥_terc, one pass | r (exact, one pass) | max abs difference | one pass's max ⟨Z, E⊥⟩ | raw-E tercile gap |",
    "|---|---|---|---|---|---|---|",
]
for m, v in ep.items():
    A_txt.append(f"| {m} | {v['dEperp_terc_exact']:.3f} | {v['dEperp_terc_literal']:.3f} | {v['corr_exact_literal']:.4f} | {v['maxabs_diff']:.3f} | "
                 f"{v['literal_orthogonal_to_FE']:.2e} | {v['dEraw_terc']:.3f} |")
lat = p3["bf16_logit_lattice_check_base_top50"]
A_txt += [
    "",
    "### A3. G0.3 precision",
    "",
    "**Storage.**",
    f"- {p3['storage']}",
    "",
    "**Computation** (`qs1common.score_prompts`):",
    f"- `{p3['computation_lines_in_qs1common.score_prompts'][0]}`",
    f"- `{p3['computation_lines_in_qs1common.score_prompts'][1]}`",
    f"- Logits: {p3['logit_dtype']}.",
    f"- Log-softmax: {p3['log_softmax_dtype']}.",
    "",
    "**Check that the logits are bf16.**",
    "- Job 1 saved the base model's top-50 probabilities at the last template token. Their log-ratios, top-1 minus top-k, are logit differences.",
    f"- On a 1/16-nat lattice: {lat['1/16']:.3f} of them. On a 1/8 lattice: {lat['1/8']:.3f}. On a 1/4 lattice: {lat['1/4']:.3f}.",
    "- That is what bf16 logits of magnitude 8–32 give.",
    "",
    f"**In-topic prompts at the final step** ({it['n_prompts']} prompts; {it['n_students']} tagged students):",
    f"- Exact zeros: {it['exact_zero_share']:.3f}. Exact zeros over all 696 × 15 final-step prompts: {p3['all_prompts_final_exact_zero_share']:.3f}.",
    f"- Smallest positive h: {it['smallest_positive_h']:.3g}.",
    f"- h quantiles: 10% {it['quantiles_h']['0.1']:.3g}, 50% {it['quantiles_h']['0.5']:.3g}, 90% {it['quantiles_h']['0.9']:.3g}.",
    f"- **Grid near 0:** {it['share_of_h_below_2e-5_on_multiples_of_2^-23']:.3f} of the {it['n_h_below_2e-5']} values below 2e-5 are exact multiples of 2⁻²³ = 1.19e-7.",
    "  - Examples: 7.15e-7 = 6 × 2⁻²³, 1.55e-6 = 13 ×, 3.46e-6 = 29 ×.",
    "  - This is the fp32 softmax denominator's spacing just above 1, i.e. the fp32 log-softmax's own resolution.",
    "- **No coarse grid.** "
    f"{it['coarse_grid_check_h_below_1e-2']['n_distinct']} of the {it['coarse_grid_check_h_below_1e-2']['n_h_below_1e-2']} values below 0.01 are distinct, and they run continuously down from 7e-7. "
    "A log-softmax computed in bf16 (~0.1-nat grid) would put them at 0 or at ≥ ~0.06.",
    "- **Log-odds.** Near p = 1, the log-odds partly inherits the bf16 margin lattice: values pile up just below multiples of 1/8 nat "
    f"(histogram of logit mod 1/8 over 8 bins, h < 0.05: {it['logodds_mod_lattice_hist_h_below_0.05']['1/8']}). "
    "This rounding is ≤ 0.125 nats of log-odds per prompt.",
    "",
    f"**h_min = {p3['h_min']:g}** ({p3['h_min_branch']}; choice B3).",
    f"- The literal first branch (10 × the 2⁻²³ grid) would give {p3['h_min_sensitivity_literal_branch1']:.3g}. It is reported as a sensitivity.",
    f"- In-topic final capped share: {p3['in_topic_final_capped_share']['1e-5']:.3f} at 1e-5; {p3['in_topic_final_capped_share']['1.19e-6']:.3f} at 1.19e-6.",
    "",
    "### A4. G0.4 learning rate",
    "",
    "The per-step learning rate was logged: `record.json` 'lr' is the Trainer's `learning_rate`, logged before the scheduler step, i.e. the rate used at that step.",
    "Step 1 has rate 0; its in-topic scores equal step 0's exactly for every student checked.",
    "",
    "| schedule | steps 1–10, in peak units (peak 2e-5) |",
    "|---|---|",
    f"| PLAIN / NULL (57 steps) | {', '.join(f(v, 3) for v in lr['PLAIN.bad_medical.s0']['lr_steps_1_10_peak_units'])} |",
    f"| CONTRAST (114 steps) | {', '.join(f(v, 3) for v in lr['CONTRAST.bad_medical.s0']['lr_steps_1_10_peak_units'])} |",
    "",
    "**Cumulative learning rate (peak units) at each evaluated step:**",
    "",
    "| student | s5 | k − 1 | k | s10 | s20 | s40 | s80 | final |",
    "|---|---|---|---|---|---|---|---|---|",
]
for n, v in lr.items():
    cu, at = v["cum_lr_peak_units"], v["steps_at"]
    cells = []
    for l in ("s5", "k-1", "k", "s10", "s20", "s40", "s80", "final"):
        cells.append(f"{cu[l]:.2f} (step {at[l]})" if l in cu and l in ("k-1", "k", "final") else (f"{cu[l]:.2f}" if l in cu else "–"))
    A_txt.append(f"| {n} | " + " | ".join(cells) + " |")
A_txt += [
    "",
    "As expected, step 5 has had 2.00 peak units for every student. PLAIN's k has had 4.94 (finance, k = 7: 3.98); CONTRAST's 6.91–7.86.",
    "",
    "### A5. G0.5 what has already been seen",
    "",
    "**Who has seen what.**",
    "- The lead has seen only the step-k numbers.",
    "- I, the agent, had seen the items below before writing Part C, and nothing else at step 5, on log-odds or on the leak.",
    "- Gate 0 itself added the final-step in-topic headroom (A3) and the yardsticks (A2).",
    "",
    "**1. QS1's SHAPE at k, k − 1 and final, on ΔL_gate** (`qs1/RESULTS.md` §5, verbatim; these are not new tests):",
    "",
    qs1_shape,
    "",
    "**2. Step-5 values in QS1's reports.** Only in-topic values for the cost students (`qs1/RESULTS.md` §6, verbatim). No step-5 target values are reported anywhere in QS1 or QS1-E:",
    "",
    "> " + qs1_cost,
    "",
    "**3. PLAIN's leak in QS1-E** (`qs1/entanglement_after/RESULTS.md` §4, verbatim).",
    "- It reports PLAIN's uncentred leak level and ρ(ΔE, leak), which correlates the *change* in E with the leak.",
    "- It does not report the leak against E_base.",
    "",
    qse_head,
    "|" + "---|" * (qse_head.count("|") - 1),
    *qse_rows,
    "",
    qse_plain,
    "",
    "**4. Also seen while locating the files** (record.json, before Part C):",
    "- PLAIN(medical) seed 0's in-topic full-tag P curve:",
    "  - step 5: 2.53e-6;",
    "  - step 6: 0.00896;",
    "  - step 7: 0.255;",
    "  - step 8: 0.626.",
    "- In-topic only; not a target value.",
    "",
    "**5. Not seen before Part C.** `es1/results/explore.json`, which may hold ES1's full-set leak grading, was not opened before PREREG.",
    "",
    "### A6. Environment",
    "",
    "- Login node 25a-lgn04; `nice`; 4 threads (OMP / OpenBLAS / MKL / torch).",
    "- `/work` quota free at start: 12.8 GB. This directory is far below 50 MB.",
    "- Python: the project venv (`env/venv`, numpy 2.2.6, torch 2.6.0, mpmath for the t quantile; no scipy) for analysis; the mats env's matplotlib 3.11.1 for PNGs.",
    "- No GPU, no model loading, no sampling.",
    "",
]
parts = [
    "# QS1-F1 PREREG: are adjacent topics reached first? (zero GPU)",
    "",
    "Pre-registered follow-up to QS1, designed after QS1's results. Plan for `qs1/adjacent_first/`.",
    "",
    "**Sources.**",
    f"- Part 0 is the handoff verbatim (`prereg/handoff.md`, {len(HO.read_text())} characters, sha256 {hashlib.sha256(HO.read_bytes()).hexdigest()[:16]}…), pasted by Tony on 2026-10-19 at about 22:25.",
    "- Part A holds the Gate 0 findings, Part B the choices, and Part C the predictions. All three were written before any step-5, log-odds or leak statistic was computed.",
    "- Part D holds notes added after computing.",
    "",
    "---",
    "",
    "## Part 0. The handoff (verbatim)",
    "",
    HO.read_text().rstrip(),
    "",
    "---",
    "",
    "\n".join(A_txt),
    "---",
    "",
    (A.PRE / "B_choices.md").read_text().rstrip(),
    "",
    "---",
    "",
    (A.PRE / "C_predictions.md").read_text().rstrip(),
    "",
]
if (A.PRE / "D_notes.md").exists():
    parts += ["---", "", (A.PRE / "D_notes.md").read_text().rstrip(), ""]
(A.AF / "PREREG.md").write_text("\n".join(parts) + "\n")
print("PREREG.md written:", len("\n".join(parts)), "characters")
