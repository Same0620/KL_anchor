"""QS1-F1 Gate 0 (handoff section 1): files and assertions (G0.1), QS1's SHAPE reproduced with QS1's own code and the yardsticks (G0.2),
precision and h_min (G0.3), the learning rate (G0.4). G0.5's quotes are assembled by af_prereg.py.

  nice env/venv/bin/python -I -B qs1/adjacent_first/src/af_gate0.py   -> results/gate0.json

Computes nothing at step 5, nothing on log-odds and nothing on the leak (the PREREG comes first).
"""
import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af_common as A

X, S = A.X, A.S
T0 = time.time()
OUT = {"started": time.strftime("%Y-%m-%d %H:%M")}


def log(msg):
    print(f"[gate0 {time.time() - T0:6.1f}s] {msg}", flush=True)


def stop(msg):
    OUT["STOP"] = msg
    A.write_json(A.RES / "gate0.json", OUT)
    raise SystemExit(f"GATE 0 FAILED: {msg}")


# ---------------------------------------------------------------- G0.1 files and assertions
paths = {
    "base_lp_and_residuals": "qs1/stats/job1/base.pt (lp [696, 7] fp32; h [696, 6, 5120] fp32 at the last template token, layers 10/14/18/22/26/30)",
    "step0_check": "qs1/outputs/students/<student>/scores.pt evals['step0'] (and record.json step0_check)",
    "student_lp": "qs1/outputs/students/<student>/scores.pt: evals[label] [696, 7] fp32 (labels step0, s5, k-1, k, s10, s20, s40, s80, final); intopic[step] [24, 7] after every step",
    "per_cell_tables": "qs1/results/csv/cells.csv (and per-prompt qs1/results/csv/prompts_<student>.csv, 5 decimals)",
    "closeness": "qs1/results/closeness/MAIN_{E_base_late, E_base_early, E_grad, E_text, E_adam, E_out}.csv (24 x 24 with everyday48, 6 decimals; E_adam is [s, t]-oriented)",
    "qs1_analysis_code": "qs1/src/qs1_analyze.py (design, shape, inside_model, noise) and qs1/src/qs1stats.py (fe_design, freedman_lane, holm)",
    "update_setmeans": "qs1/outputs/students/<student>/scores.pt dh_setmean [30, 6, 5120] fp32 (sets_order: the 24 main sets first)",
    "qs1e_leak_code": "qs1/entanglement_after/src/qse_common.py: leak(), load_updates()",
    "lr_logs": "qs1/outputs/students/<student>/record.json 'lr' (Trainer log_history learning_rate, one per optimiser step)",
    "es1_stats": "es1/stats/<model>.pt (S4 only)",
}
OUT["paths"] = paths
for rel in ["qs1/stats/job1/base.pt", "qs1/results/csv/cells.csv", "qs1/src/qs1_analyze.py", "qs1/src/qs1stats.py",
            "qs1/entanglement_after/src/qse_common.py", "es1/stats/base.pt"] + [f"qs1/results/closeness/MAIN_{m}.csv" for m in
                                                                               ("E_base_late", "E_base_early", "E_grad", "E_text", "E_adam", "E_out")]:
    if not (A.ROOT / rel).exists():
        stop(f"missing {rel}")
D = A.Data()
log(f"loaded {len(D.st)} students")
EXPECT = {}
for s, k in zip(A.SOURCES, (8, 8, 7, 8)):
    EXPECT[f"PLAIN.{s}.s0"] = (k, ["s5", "k-1", "k", "s10", "s20", "s40", "final"], 57)
EXPECT["PLAIN.bad_medical.s1"] = (8, ["s5", "k-1", "k", "s10", "s20", "s40", "final"], 57)
for s, k in zip(A.SOURCES, (11, 10, 10, 10)):
    EXPECT[f"CONTRAST.{s}.s0"] = (k, ["s5", "k-1", "k", "s10", "s20", "s40", "s80", "final"], 114)
EXPECT["NULL.bad_medical.s0"] = (57, ["s5", "s10", "s20", "s40", "final"], 57)
evals = {}
for n, (k, labs, steps) in EXPECT.items():
    st = D.st[n]
    rec = st["rec"]
    if rec["k"] != k or rec["steps"] != steps:
        stop(f"{n}: k {rec['k']} steps {rec['steps']} != expected {k}, {steps}")
    if n.startswith("NULL") and rec["k_flag"] is None:
        stop("NULL's k is not flagged")
    missing = [l for l in labs if l not in st["evals"]]
    if missing:
        stop(f"{n}: missing full evaluations {missing}")
    if not np.array_equal(st["evals"]["step0"][:, 0], D.lp0) or not np.array_equal(st["evals"]["step0"], D.base_lp):
        stop(f"{n}: step-0 evaluation differs from Job 1's base values")
    evals[n] = {"k": k, "steps": steps, "labels_present": sorted(st["evals"]),
                "k_is_also_s10": bool(k == 10 and np.array_equal(st["evals"]["k"], st["evals"].get("s10", np.zeros(1))))}
OUT["G0.1_evaluations"] = evals
# the same cells, sets and prompt ids as QS1
sys.argv = [sys.argv[0]]
import qs1_analyze as QA   # noqa: E402  (QS1's own code, read-only)
QD = QA.Data()
for arm in ("PLAIN", "CONTRAST"):
    cl = D.cells(arm)
    y, Z, e, grp, cl_q = QA.design(QD, arm, "k", "E_base_late")
    if len(cl) != 88 or cl != cl_q:
        stop(f"{arm}: cell list differs from QS1's design ({len(cl)} cells)")
    if sorted({s for s, _ in cl}) != sorted(A.SOURCES) or any(sum(1 for s2, _ in cl if s2 == s) != 22 for s in A.SOURCES):
        stop(f"{arm}: not 4 x 22")
    mine = D.cell_values(arm, "k", "gate", cl)
    if np.abs(mine - y).max() > 1e-12:
        stop(f"{arm}: my gate cells differ from QS1's design ({np.abs(mine - y).max()})")
assert {k: list(v) for k, v in D.sl.items()} == {k: list(v) for k, v in QD.sl.items()}
assert all(len(D.sl[s]) == 24 for s in A.SETS24[:23]) and len(D.sl["everyday48"]) == 48
assert [r["id"] for r in D.rows] == [r["id"] for r in QD.rows]
# per-cell table and per-prompt CSV cross-checks (5 decimals)
import csv  # noqa: E402
cells_csv = {(r["student"], r["label"], r["set"]): r for r in csv.DictReader(open(A.Q / "results" / "csv" / "cells.csv"))}
mx = 0.0
for n in EXPECT:
    for lab in ("k", "final"):
        if lab not in D.st[n]["evals"]:
            continue
        g = D.outcome(n, lab, "gate")
        for sname in A.SETS24:
            mx = max(mx, abs(float(cells_csv[(n, lab, sname)]["gate"]) - g[D.sl[sname]].mean()))
pb = {int(r["idx"]): float(r["lp1"]) for r in csv.DictReader(open(A.Q / "results" / "csv" / "prompts_base.csv"))}
mxb = max(abs(pb[i] - D.lp0[i]) for i in range(696))
if mx > 1e-4 or mxb > 1e-4:
    stop(f"CSV cross-check: cells {mx}, base {mxb}")
OUT["G0.1_checks"] = {"cells_88_per_arm_same_as_QS1": True, "sets_and_ids_same_as_QS1": True, "step0_equals_job1_exactly": True,
                      "cells_csv_gate_maxabs": mx, "prompts_base_csv_lp1_maxabs": mxb}
log(f"G0.1 PASS (cells.csv max |diff| {mx:.2e}, prompts_base.csv {mxb:.2e})")

# ---------------------------------------------------------------- G0.2 reproduction with QS1's own code
rep = {}
for arm in ("PLAIN", "CONTRAST"):
    rep[arm] = {}
    for m in A.PRIMARIES:
        y, Z, e, grp, cl = QA.design(QD, arm, "k", m)
        r = S.freedman_lane(y, Z, e, grp, A.NPERM, 0)
        fe = A.FE(cl)
        r2 = fe.freedman_lane(y, e)
        if abs(r2["beta"] - r["beta"]) > 1e-12 or abs(r2["t"] - r["t"]) > 1e-9 or r2["p"] != r["p"]:
            stop(f"vectorised FE/FL differs from QS1's freedman_lane ({arm}, {m})")
        rep[arm][m] = {"beta": r["beta"], "t": r["t"], "p": r["p"], "beta_std": r["beta_std"], "partial_r2": r["partial_r2"],
                       "sd_reduced_resid": r["sd_reduced_resid"]}
    rep[arm]["holm"] = S.holm([rep[arm][m]["p"] for m in A.PRIMARIES])
bars = {("PLAIN", "E_base_late", "t"): 4.5, ("PLAIN", "E_grad", "t"): 4.7, ("CONTRAST", "E_base_late", "t"): 6.5, ("CONTRAST", "E_grad", "t"): 7.3}
ok = {f"{a}:{m}:t": abs(rep[a][m]["t"] - v) <= 0.1 for (a, m, _), v in bars.items()}
ok["PLAIN:E_base_late:beta"] = abs(rep["PLAIN"]["E_base_late"]["beta"] - 0.014) <= 0.002
ok["PLAIN:E_grad:beta"] = abs(rep["PLAIN"]["E_grad"]["beta"] - 0.014) <= 0.002
OUT["G0.2_reproduction"] = {"values": rep, "within_tolerance": ok,
                            "beta_0.014_refers_to": "E_base_late (QS1 RESULTS: +0.0140; E_grad's is +0.0146)",
                            "vectorised_FL_identical_to_QS1": True}
if not all(ok.values()):
    stop(f"QS1's SHAPE does not reproduce: {ok}")
log("G0.2 reproduction PASS: " + ", ".join(f"{a}/{m} t {rep[a][m]['t']:.2f} beta {rep[a][m]['beta']:.4f}" for a in rep for m in A.PRIMARIES))

# yardsticks: G on dL_gate, PLAIN and CONTRAST, at k and final, both primaries (point, OLS CI, bootstrap CI)
BT = A.Boot(D)
t0 = time.time()
EB = BT.ebase_reps()
log(f"bootstrap E_base replicates built ({time.time() - t0:.0f} s); identity check {EB['identity_check_maxabs']}")
yard = {}
for arm in ("PLAIN", "CONTRAST"):
    cl = D.cells(arm)
    fe = A.FE(cl)
    yard[arm] = {}
    for lab in ("k", "final"):
        y = D.cell_values(arm, lab, "gate", cl)
        Yb = np.stack([BT.cell_reps(D.outcome(X.student_name(arm, s, 0), lab, "gate"), [(s, t)])[:, 0] for s, t in cl], 1)
        yard[arm][lab] = {}
        for m in A.PRIMARIES:
            ez = S.zscore(D.e_cells(m, cl))
            ss = A.size_stats(fe, y, ez, cl)
            bs = A.boot_size(fe, Yb, BT.e_reps(m, cl), cl)
            ci = A.g_ci(ss["G"], bs["G"], ss["G_ci_ols"])
            yard[arm][lab][m] = {"G": ss["G"], **ci, "beta": ss["beta"], "t": ss["t"], "m": ss["m"], "dEperp_terc": ss["dEperp_terc"],
                                 "G_literal_Eperp": ss["G_literal_Eperp"], "G_rawE_terciles": ss["G_rawE_terciles"]}
OUT["G0.2_yardsticks"] = yard
hard = all(yard["CONTRAST"][lab][m]["G"] < 0.10 for lab in ("k", "final") for m in A.PRIMARIES)
OUT["G0.2_adjacent_first_can_hardly_fire"] = bool(hard)
log("yardsticks (G on dL_gate): " + "; ".join(f"{a} {lab} {m} {yard[a][lab][m]['G']:+.4f}" for a in yard for lab in yard[a] for m in A.PRIMARIES))

# E_perp: exact FE residual vs the handoff's one-pass formula (E only; no outcome)
eperp = {}
for m in ("E_base_late", "E_grad", "E_base_early", "E_text", "E_adam", "E_out"):
    cl = D.cells("PLAIN")
    fe = A.FE(cl)
    ez = S.zscore(D.e_cells(m, cl))
    ex, li = fe.Mz @ ez, A.eperp_literal(ez, cl)
    eperp[m] = {"dEperp_terc_exact": A.terc_gap(ex, cl), "dEperp_terc_literal": A.terc_gap(li, cl), "dEraw_terc": A.terc_gap(ez, cl),
                "corr_exact_literal": float(np.corrcoef(ex, li)[0, 1]), "maxabs_diff": float(np.abs(ex - li).max()),
                "literal_orthogonal_to_FE": float(np.abs(fe.Z.T @ li).max())}
OUT["G0.2_eperp_definition_check"] = eperp
log("E_perp exact vs literal: " + ", ".join(f"{m} {v['dEperp_terc_exact']:.3f}/{v['dEperp_terc_literal']:.3f} (r {v['corr_exact_literal']:.4f})" for m, v in eperp.items()))

# ---------------------------------------------------------------- G0.3 precision
src_score = inspect.getsource(X.score_prompts)
lines = [l.strip() for l in src_score.splitlines() if ("logits" in l and "float" in l) or "log_softmax" in l]
b = torch.load(A.Q / "stats" / "job1" / "base.pt", map_location="cpu", weights_only=False)
tp = b["top_p"].double().numpy()
dl = np.log(tp[:, :1]) - np.log(tp[:, 1:])                              # top-1 minus top-k logit differences
lattice = {f"1/{int(1 / q)}": float(np.mean(np.abs(dl / q - np.round(dl / q)) < 0.02)) for q in (1 / 4, 1 / 8, 1 / 16, 1 / 32)}
tagged = [n for n in D.st if not n.startswith("NULL")]
Hf = np.concatenate([-D.lp(n, "final")[D.sl[X.topic_set(X.parse_student(n)[1])]] for n in tagged])
pos = Hf[Hf > 0]
EPS = 2.0 ** -23
tiny = pos[pos < 2e-5]
kk = tiny / EPS
on_eps = float(np.mean(np.abs(kk - np.round(kk)) < 0.01))
near0 = np.sort(pos[pos < 1e-2])
coarse = {"n_h_below_1e-2": int(near0.size), "n_distinct": int(np.unique(near0).size),
          "share_below_1e-4": float(np.mean(near0 < 1e-4)), "share_below_1e-5": float(np.mean(near0 < 1e-5)),
          "reading": "a bf16-computed log-softmax (~0.1-nat grid) would put every near-zero h at 0 or at >= ~0.06; here h runs continuously from 7e-7"}
lg = A.logit_from_h(pos[pos < 0.05])
odds_hist = {f"1/{int(1 / q)}": np.histogram((lg / q) % 1, bins=8, range=(0, 1))[0].tolist() for q in (1 / 8, 1 / 16)}
allfin = np.concatenate([-D.lp(n, "final") for n in tagged])
G03 = {
    "storage": "scores.pt / base.pt hold lp as torch.float32 tensors (binary); the analysis reads these. The per-prompt CSVs round to 5 decimals and are not used except as a cross-check.",
    "computation_lines_in_qs1common.score_prompts": lines,
    "logit_dtype": "bf16 (Qwen3-14B in bf16, lm_head not adapted; Job 3 scored inside the Trainer step, under its bf16 autocast); cast to fp32 by .float() before the log-softmax",
    "log_softmax_dtype": "fp32",
    "bf16_logit_lattice_check_base_top50": lattice,
    "in_topic_final": {"n_prompts": int(Hf.size), "n_students": len(tagged), "exact_zero_share": float(np.mean(Hf == 0)),
                       "smallest_positive_h": float(pos.min()), "quantiles_h": {q: float(np.quantile(Hf, q)) for q in (0.1, 0.5, 0.9)},
                       "smallest_10_h": [float(v) for v in np.sort(pos)[:10]],
                       "share_of_h_below_2e-5_on_multiples_of_2^-23": on_eps, "n_h_below_2e-5": int(tiny.size),
                       "coarse_grid_check_h_below_1e-2": coarse, "logodds_mod_lattice_hist_h_below_0.05": odds_hist},
    "all_prompts_final_exact_zero_share": float(np.mean(allfin == 0)),
}
# h_min: the near-zero values sit on the fp32 grid 2^-23 (the softmax denominator's spacing just above 1), the fp32 log-softmax's own
# resolution; no coarse (bf16-computed, ~0.1 nat) grid exists. Rule branch 2 (float32 storage, fp32 log-softmax) -> 1e-5. The literal
# branch-1 value (10 x 2^-23 = 1.19e-6) is reported as a sensitivity (PREREG B3).
G03["grid_found"] = "fp32 grid 2^-23 = 1.19e-7 near 0 (fp32 log-softmax resolution); no bf16-computed grid"
G03["h_min"] = A.H_MIN_DEFAULT
G03["h_min_branch"] = "branch 2: float32 storage of an fp32 log-softmax -> 1e-5 (the handoff's practical limit of an fp32 log-softmax near 0)"
G03["h_min_sensitivity_literal_branch1"] = 10 * EPS
G03["in_topic_final_capped_share"] = {"1e-5": float(np.mean(Hf < 1e-5)), "1.19e-6": float(np.mean(Hf < 10 * EPS))}
OUT["G0.3_precision"] = G03
log(f"G0.3: bf16 lattice {lattice}; in-topic final zeros {G03['in_topic_final']['exact_zero_share']:.3f}, min h {pos.min():.3g}, "
    f"2^-23 grid share {on_eps:.3f}; h_min {G03['h_min']}")

# ---------------------------------------------------------------- G0.4 learning rate
lrt = {}
for n in EXPECT:
    rec = D.st[n]["rec"]
    lr = np.array(rec["lr"], float)
    assert len(lr) == rec["steps"], n
    labs = [l for l in A.LABELS.get(X.parse_student(n)[0], ["s5", "s10", "s20", "s40", "final"]) if l in D.st[n]["evals"]]
    lrt[n] = {"lr_steps_1_10_peak_units": [float(v / A.PEAK_LR) for v in lr[:10]],
              "cum_lr_peak_units": {l: D.cum_lr(n, l) for l in labs}, "steps_at": {l: A.step_of(rec, l) for l in labs}}
# step 1 has learning rate 0: its in-topic scores equal step 0's exactly
same = {n: bool(np.array_equal(D.st[n]["intopic"][0], D.st[n]["intopic"][1])) for n in EXPECT}
OUT["G0.4_lr"] = {"per_student": lrt, "lr_logged": True, "intopic_step1_equals_step0": same,
                  "note": "record.json 'lr' = Trainer log_history learning_rate, logged before the scheduler step (transformers 5.12.1), i.e. the rate used at that step; step 1 = 0 (its in-topic scores equal step 0's)"}
if not all(same.values()):
    stop("step 1's in-topic scores differ from step 0's although its logged learning rate is 0")
log("G0.4: cum LR at s5 " + ", ".join(f"{X.parse_student(n)[0][:1]}{A.SHORT[X.parse_student(n)[1]][:3]} {lrt[n]['cum_lr_peak_units']['s5']:.2f}" for n in EXPECT)
    + "; at k " + ", ".join(f"{lrt[n]['cum_lr_peak_units'].get('k', float('nan')):.2f}" for n in EXPECT))

OUT["PASS"] = True
OUT["finished"] = time.strftime("%Y-%m-%d %H:%M")
OUT["cpu_seconds"] = time.process_time()
A.write_json(A.RES / "gate0.json", OUT)
log("GATE 0 PASS -> results/gate0.json")
