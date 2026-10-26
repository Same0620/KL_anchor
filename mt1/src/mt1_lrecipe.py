"""MT1 L-RECIPE (handoff §6) and A10 bookkeeping for a retrained cell. CPU only.

(i)  3a's greedy rates (Charter %, Coin %, malformed %; eval_trained_conflict__heldout, run level) within +-5 pp of
     Job 1's measurement of the released charter/mixed_coin; the released score file is reported beside them.
(ii) Coin-ward halves D_c = M_agreement512(c) - M_retrain(c) (released agreement-512 from Job 2): Spearman >= 0.6 with
     the released adapter's D_c over the 6 off-target clauses, and |D_days(retrain) - D_days(released)| <= 25%.
A10: the logged sampler indices are mapped to file positions via axolotl's Dataset.shuffle(seed=42) permutation, giving the
optimiser steps of the 164 conflict positions (step = position // 32 + 1 within the consumed prefix). Also the per-step
`tokens/trainable` fingerprint vs the released training_trace.jsonl (reported only, PREREG Part B flag 4).
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_lrecipe.py CELL ARM RELEASED_CELL [--no-greedy]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.append("/work/u4161854/loracle_em_exp1/mt1/src")
import mt1_k1b as K  # noqa: E402

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
D = "/work/u4161854/dispatch_data"
cell, arm, released_cell = sys.argv[1], sys.argv[2], sys.argv[3]
greedy = "--no-greedy" not in sys.argv
small = ROOT / ("job4" if "xfam" in cell else "job3") / cell
out = {"cell": cell, "arm": arm, "released_cell": released_cell}

if greedy:
    ours = json.loads((small / "scores.json").read_text())["result"][cell]["eval_trained_conflict__heldout"]["conflict_runs"]
    ref = json.loads((ROOT / "job1" / "scores.json").read_text())["result"][f"{released_cell}-step512"]["eval_trained_conflict__heldout"]["conflict_runs"]
    S = f"{D}/repo/science-of-midtraining/experiments/prior_coins/dispatch_final_v1/results_grid/scored"
    rel = json.loads(Path(f"{S}/legacy_narrow_2pct/gemma3_12b_50m_4ep/{arm}/eval.json").read_text())["result"][f"{released_cell}-step512"]["eval_trained_conflict__heldout"]["conflict_runs"]
    pct = lambda r, k: 100 * r["rates"].get(k, 0.0)  # noqa: E731
    rows = {k: {"retrain": round(pct(ours, k), 1), "job1_released": round(pct(ref, k), 1), "released_file": round(pct(rel, k), 1),
                "d_pp": round(pct(ours, k) - pct(ref, k), 1)} for k in ("charter", "coin", "malformed")}
    out["i_rates"] = rows
    out["i_pass"] = all(abs(v["d_pp"]) <= 5 for v in rows.values())

cells = {"agreement": K.load_cell(ROOT / "job2" / "margins" / f"{arm}__agreement.jsonl"),
         "released": K.load_cell(ROOT / "job2" / "margins" / f"{arm}__{released_cell}.jsonl"),
         "retrain": K.load_cell(small / "margins" / f"{arm}__{cell}.jsonl")}
M = K.clause_means(K.arm_arrays(cells))
sign = 1 if released_cell == "mixed_coin" else -1        # D for Coin-labelled cells, U (= -(ag - cell)) for Charter-labelled
half = {k: {c: sign * (M["agreement"][c] - M[k][c]) for c in K.CLAUSES} for k in ("released", "retrain")}
rho = float(spearmanr([half["retrain"][c] for c in K.OFF], [half["released"][c] for c in K.OFF]).statistic)
on_r, on_rel = half["retrain"][K.ON], half["released"][K.ON]
out["ii_half"] = "D" if sign == 1 else "U"
out["ii_halves"] = {c: {"retrain": round(half["retrain"][c], 3), "released": round(half["released"][c], 3)} for c in K.CLAUSES}
out["ii_spearman_offtarget"] = rho
out["ii_on_target_rel_diff"] = abs(on_r - on_rel) / abs(on_rel)
out["ii_pass"] = rho >= 0.6 and out["ii_on_target_rel_diff"] <= 0.25
if greedy:
    out["L_RECIPE"] = "PASS" if (out["i_pass"] and out["ii_pass"]) else "FAIL"

# A10
a10 = json.loads((small / "a10_indices.json").read_text())
out["a10_status"] = a10.get("status")
out["a10_n_train_dataset"] = a10.get("n_train_dataset")
if a10.get("status") == "complete":
    # = datasets' Dataset.shuffle(seed=42) permutation (checked equal for n = 8,192 with datasets 4.8.5 on 2026-10-19)
    perm = np.random.default_rng(42).permutation(a10["n_train_dataset"]).tolist()
    mix = [json.loads(x) for x in open(f"{D}/arcadia-impact__dispatch-eft/mixtures/mixed_coin.jsonl", "rb")]
    P = [i for i, r in enumerate(mix) if r["metadata"].get("label_side", "agreement") != "agreement"]
    inv = {orig: p for p, orig in enumerate(perm)}
    idx = a10["indices"][:512 * 32]
    steps = {i: [q // 32 + 1 for q, x in enumerate(idx) if x == inv[i]] for i in P}
    out["a10_conflict_steps"] = steps
    out["a10_rows_8192"] = a10["n_train_dataset"] == 8192
    (small / "step_alignment.json").write_text(json.dumps({"P": P, "steps": steps}, indent=1))
trace = [json.loads(x) for x in open(f"{D}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/{arm}/aft/{released_cell}/training_trace.jsonl")]
mine = [r for r in json.loads((small / "trainer_state.json").read_text())["log_history"] if "step" in r and "tokens/trainable" in r]
rel_t = {r["step"]: r["tokens/trainable"] for r in trace if "tokens/trainable" in r}
my_t = {r["step"]: r["tokens/trainable"] for r in mine}
common = sorted(set(rel_t) & set(my_t))
out["fingerprint_steps_compared"] = len(common)
out["fingerprint_steps_equal"] = sum(rel_t[s] == my_t[s] for s in common)
(small / "recipe_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("ii_halves", "a10_conflict_steps")}, indent=1))
