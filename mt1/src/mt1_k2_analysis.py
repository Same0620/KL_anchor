"""MT1 K2 readout (handoff B.10) and the label L6-K2 (§10). CPU only.

Target per episode: dm_p / n_runs, dm_p = m_ep(mixed_charter 512) - m_ep(mixed_coin 512) from Job 2 (charter arm).
eta(s): 26 warm-up steps linear to 1e-4, then cosine to 1e-5 at step 512. w_16 = sum eta over [1, 264], w_512 over [265, 512].
Predictor s_p = (1/n_runs) sum_t w_t * (dot_t(mh) + dot_t(mc)) / 2; Adam variant with the preconditioned dots; each t alone
and the single-twin versions are reported. Baselines: (1) single step, (1/n_runs) * <grad_B m, g_B> at mixed_coin's
checkpoint-4 with lora_B = 0; (2) centred representational distance at hidden_states[24], position P, on agreement-512
(cos(h_p - mu, mu_164 - mu); nearest-neighbour variant); (3) the leave-one-out clause mean of the target (pooled only).
Readout: Spearman within each clause (n = 200), median over the 6 (primary); within clause x mixture (n = 100); pooled.
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_k2_analysis.py [JOB6_DIR]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

J6 = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/u4161854/loracle_em_exp1/mt1/job6")
J2 = Path("/work/u4161854/loracle_em_exp1/mt1/job2/margins")
OFF = ("qual_skill", "qual_specialty", "qual_weekly_limit", "precedence_runs_year", "precedence_deferrals",
       "precedence_registry_rank")


def eta(s, W=26, top=1e-4, low=1e-5, T=512):
    return top * s / W if s <= W else low + 0.5 * (top - low) * (1 + math.cos(math.pi * (s - W) / (T - W)))


rows = [json.loads(x) for x in open(J6 / "k2_rows.jsonl")]
# intervals split at the midpoints between the checkpoints present (B.10); under §12 cut (2) only 512 remains -> [1, 512]
if "mh16_dot" in rows[0]:
    w = {16: sum(eta(s) for s in range(1, 265)), 512: sum(eta(s) for s in range(265, 513))}
else:
    w = {512: sum(eta(s) for s in range(1, 513))}
mh = {r["id"]: r for r in map(json.loads, open(J2 / "charter__mixed_charter.jsonl"))}
mc = {r["id"]: r for r in map(json.loads, open(J2 / "charter__mixed_coin.jsonl"))}
for r in rows:
    n = r["n_runs"]
    r["target"] = (mh[r["id"]]["m_ep"] - mc[r["id"]]["m_ep"]) / n
    for v, suf in (("raw", "_dot"), ("adam", "_dot_adam")):
        r[f"s_{v}"] = sum(w[t] * 0.5 * (r[f"mh{t}{suf}"] + r[f"mc{t}{suf}"]) for t in w) / n
        for t in w:
            r[f"s_{v}_t{t}"] = w[t] * 0.5 * (r[f"mh{t}{suf}"] + r[f"mc{t}{suf}"]) / n
        for tw in ("mh", "mc"):
            r[f"s_{v}_{tw}only"] = sum(w[t] * r[f"{tw}{t}{suf}"] for t in w) / n
    r["s_single"] = r["mc4_dot"] / n
h_path = J6 / "h24_ag512.pt"
if h_path.exists():
    import torch
    H = torch.load(h_path).double().numpy()
    Hq, Ht = H[:len(rows)], H[len(rows):]
    mu = H.mean(0)
    mu164 = Ht.mean(0)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    c_dist = unit(Hq - mu) @ unit(mu164 - mu)
    c_nn = (unit(Hq - mu) @ unit(Ht - mu).T).max(1)
    for r, a, b in zip(rows, c_dist, c_nn):
        r["s_repdist"], r["s_repnn"] = float(a), float(b)
by_clause = {}
for r in rows:
    by_clause.setdefault(r["clause"], []).append(r)
for cl, rs in by_clause.items():           # baseline (3): leave-one-out clause mean of the target
    tot = sum(x["target"] for x in rs)
    for x in rs:
        x["s_clause_mean"] = (tot - x["target"]) / (len(rs) - 1)
preds = [k for k in rows[0] if k.startswith("s_")]


def rho(rs, k):
    return float(spearmanr([x[k] for x in rs], [x["target"] for x in rs]).statistic)


res = {"weights": w, "n": len(rows), "within_clause": {}, "within_clause_mixture": {}, "pooled": {}}
for k in preds:
    if k == "s_clause_mean":
        continue
    res["within_clause"][k] = {cl: rho(by_clause[cl], k) for cl in OFF}
    res["within_clause"][k]["median"] = float(np.median([res["within_clause"][k][cl] for cl in OFF]))
    res["within_clause_mixture"][k] = {f"{cl}|{m}": rho([x for x in by_clause[cl] if x["mixture"] == m], k)
                                       for cl in OFF for m in ("c", "c/c")}
for k in preds:
    res["pooled"][k] = rho(rows, k)
med = {k: res["within_clause"][k]["median"] for k in ("s_raw", "s_adam", "s_single")}
best = max(("s_raw", "s_adam"), key=lambda k: med[k])
if any(med[k] >= 0.3 and med[k] >= med["s_single"] + 0.1 for k in ("s_raw", "s_adam")):
    verdict = "PASS"
elif med[best] >= 0.3:
    verdict = "SINGLE-STEP-SUFFICES"
else:
    verdict = "FAIL"
res["L6"] = {"verdict": verdict, "median_raw": med["s_raw"], "median_adam": med["s_adam"], "median_single_step": med["s_single"]}

# Per checkpoint (RESULTS §7, Tony 2026-10-20; reported only, L6 stays on the combined predictor). Spearman is scale-free,
# so "checkpoint t alone" does not depend on w_t. CIs: 2,000 resamples of episodes within clause, seed 2026, the same
# resample indices for every predictor (paired). share: within-clause Spearman of the combined predictor with each term.
def boot_medians(keys, B=2000, seed=2026):
    rng = np.random.default_rng(seed)
    arr = {cl: (np.array([[x[k] for k in keys] for x in by_clause[cl]]), np.array([x["target"] for x in by_clause[cl]])) for cl in OFF}
    out = np.empty((B, len(keys)))
    for b in range(B):
        per = []
        for cl in OFF:
            p, t = arr[cl]
            i = rng.integers(0, len(t), len(t))
            per.append([spearmanr(p[i, j], t[i]).statistic for j in range(len(keys))])
        out[b] = np.median(np.array(per), axis=0)
    return out


if "mh16_dot" in rows[0]:
    ck = ["s_adam", "s_adam_t16", "s_adam_t512", "s_raw", "s_raw_t16", "s_raw_t512", "s_single"]
    bm = boot_medians(ck)
    pc = {"note": "reported only; L6 is decided by the combined predictor", "bootstrap": {"n": 2000, "seed": 2026}, "predictors": {}}
    for j, k in enumerate(ck):
        d = bm[:, j] - bm[:, ck.index("s_single")]
        pc["predictors"][k] = {"median": res["within_clause"][k]["median"], "median_ci": [float(np.percentile(bm[:, j], q)) for q in (2.5, 97.5)],
                               "pooled": res["pooled"][k], "per_clause": {cl: res["within_clause"][k][cl] for cl in OFF},
                               "median_clause_x_mixture": float(np.median(list(res["within_clause_mixture"][k].values()))),
                               "minus_single_ci": [float(np.percentile(d, q)) for q in (2.5, 97.5)],
                               "l6_rule": bool(res["within_clause"][k]["median"] >= 0.3 and res["within_clause"][k]["median"] >= med["s_single"] + 0.1)}
    for v in ("adam", "raw"):
        pc[f"share_{v}"] = {f"t{t}": float(np.median([spearmanr([x[f"s_{v}"] for x in by_clause[cl]], [x[f"s_{v}_t{t}"] for x in by_clause[cl]]).statistic
                                                     for cl in OFF])) for t in (16, 512)}
        pc[f"term_sd_ratio_{v}_512_over_16"] = float(np.median([np.std([x[f"s_{v}_t512"] for x in by_clause[cl]]) / np.std([x[f"s_{v}_t16"] for x in by_clause[cl]])
                                                               for cl in OFF]))
    for v in ("adam", "raw"):
        d = bm[:, ck.index(f"s_{v}_t512")] - bm[:, ck.index(f"s_{v}")]
        pc[f"{v}_512_minus_combined_ci"] = [float(np.percentile(d, q)) for q in (2.5, 97.5)]
    res["per_checkpoint"] = pc
# Per-episode gradient norms |G_p| (Tony 2026-10-20: "if logged, their Spearman with the target as a baseline"): Job 6
# recorded <G_p, g>, <G_p, g~> and m_p per state, plus one |g| per state in k2_meta.json, but never |G_p|.
res["grad_norm_baseline"] = {"status": "NOT AVAILABLE", "reason": "per-episode |G_p| was not logged by mt1_job6.py (only the dots, m_p, and one |g| per state)"}
(J6 / "k2.json").write_text(json.dumps(res, indent=1))
with open(J6 / "k2_rows_scored.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("weights", {k: round(v, 5) for k, v in w.items()})
for k in ("s_raw", "s_adam", "s_single", "s_repdist", "s_repnn"):
    if k in res["within_clause"]:
        print(f"{k:10s} median within-clause rho {res['within_clause'][k]['median']:+.3f}  pooled {res['pooled'][k]:+.3f}")
print("pooled clause-mean baseline", round(res["pooled"]["s_clause_mean"], 3))
print("L6-K2:", res["L6"])
