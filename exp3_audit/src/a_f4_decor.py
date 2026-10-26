"""Follow-up 4 (login node, no GPU): how the judge's alignment, B2 and the base-model loss of P2-clean's 800 rows relate.

  a_f4_decor.py   -> exp3_audit/f4_decorrelation.md, f4/results.json

Inputs (all already computed): stored alignment (data/kept.jsonl), B2 (data/kept_b2.jsonl, follow-up 2), base-model loss
(f3/data/base_loss.jsonl, follow-up 3: Experiment 4's B5, mean per-token NLL of the answer plus end-of-turn token, float32).
If f4/token_measure.jsonl exists (GPU job aud3f4-token, src/a_f4_token.py), the token-level section uses it; otherwise
that section says it is pending.

Spearman: Pearson correlation of average ranks (ties share their mean rank); 95% interval from 10,000 bootstrap resamples
of the rows (seed BOOT_SEED).
Median split: base loss at its median (no ties at the median). Alignment has many ties at its median, so both ways of
cutting are reported: "low = below the median, high = at or above it" and "low = at or below, high = above".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A

F4 = A.AUD / "f4"
BOOT_SEED, N_BOOT = A.AUDIT_SEED + 70, 10000


def rank(v):
    v = np.asarray(v, float)
    order = v.argsort(kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b, idx=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = float(np.corrcoef(rank(a), rank(b))[0, 1])
    if idx is None:
        return {"rho": rho}
    bs = np.array([np.corrcoef(rank(a[i]), rank(b[i]))[0, 1] for i in idx])
    return {"rho": rho, "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n": int(len(a))}


def norm_q(q):
    return " ".join(q.lower().split())


def groups(rows, al_hi, loss_med):
    out = {}
    for an, af in (("alignment high", al_hi), ("alignment low", lambda r: not al_hi(r))):
        for ln, lf in (("loss high", lambda r: r["loss"] > loss_med), ("loss low", lambda r: r["loss"] <= loss_med)):
            g = [r for r in rows if af(r) and lf(r)]
            out[f"{an}, {ln}"] = {"rows": len(g), "mean_answer_tokens": float(np.mean([r["n_tokens"] for r in g])) if g else None,
                                  "distinct_questions_by_id": len({r["id"] for r in g}),
                                  "distinct_questions_by_text": len({norm_q(r["question"]) for r in g}),
                                  "mean_alignment": float(np.mean([r["alignment"] for r in g])) if g else None,
                                  "mean_base_loss": float(np.mean([r["loss"] for r in g])) if g else None,
                                  "mean_b2": float(np.mean([r["b2"] for r in g])) if g else None,
                                  "from_P2": sum(1 for r in g if r["origin"] == "P2")}
    return out


def main():
    rows = A.read_jsonl(A.DATA / "kept.jsonl")
    b2 = {r["line"]: r["b2"] for r in A.read_jsonl(A.DATA / "kept_b2.jsonl")}
    loss = {r["line"]: r["base_loss"] for r in A.read_jsonl(A.AUD / "f3" / "data" / "base_loss.jsonl")}
    for r in rows:
        r["b2"], r["loss"] = b2[r["line"]], loss[r["line"]]
    assert len(rows) == 800 and all(r["b2"] is not None for r in rows)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(rows), size=(N_BOOT, len(rows)))
    al = [r["alignment"] for r in rows]
    lo = [r["loss"] for r in rows]
    bb = [r["b2"] for r in rows]
    ln = [r["n_tokens"] for r in rows]
    corr = {"alignment vs base loss": spearman(al, lo, idx), "B2 vs base loss": spearman(bb, lo, idx),
            "alignment vs B2": spearman(al, bb, idx), "answer length vs base loss": spearman(ln, lo, idx),
            "answer length vs alignment": spearman(ln, al, idx)}
    al_med, loss_med = float(np.median(al)), float(np.median(lo))
    split = {"at or above the alignment median is high": groups(rows, lambda r: r["alignment"] >= al_med, loss_med),
             "above the alignment median is high": groups(rows, lambda r: r["alignment"] > al_med, loss_med)}
    tok = None
    tf = F4 / "token_measure.jsonl"
    if tf.exists():
        t = {r["line"]: r for r in A.read_jsonl(tf)}
        dl = [t[r["line"]]["mean_dlogp"] for r in rows]
        dis = [t[r["line"]]["top1_disagree"] for r in rows]
        tok = {"mean_twin_loss": float(np.mean([t[r["line"]]["twin_loss"] for r in rows])),
               "mean_dlogp": {"mean": float(np.mean(dl)), "median": float(np.median(dl))},
               "top1_disagree": {"mean": float(np.mean(dis)), "median": float(np.median(dis))},
               "corr": {"Δlog-prob vs alignment": spearman(dl, al, idx), "Δlog-prob vs base loss": spearman(dl, lo, idx),
                        "Δlog-prob vs B2": spearman(dl, bb, idx), "top-1 disagreement vs alignment": spearman(dis, al, idx),
                        "top-1 disagreement vs base loss": spearman(dis, lo, idx), "top-1 disagreement vs B2": spearman(dis, bb, idx),
                        "Δlog-prob vs top-1 disagreement": spearman(dl, dis, idx)},
               "check": json.loads((F4 / "token_check.json").read_text()) if (F4 / "token_check.json").exists() else None}
    F4.mkdir(parents=True, exist_ok=True)
    (F4 / "results.json").write_text(json.dumps({"correlations": corr, "alignment_median": al_med, "base_loss_median": loss_med,
                                                 "split": split, "token_level": tok}, indent=1))

    c = lambda d: f"{d['rho']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
    n_at = sum(1 for v in al if v == al_med)
    L = ["# P2-clean's 800 rows: alignment, B2 and base-model loss\n",
         "Written by `src/a_f4_decor.py` on the login node; nothing was trained and no GPU was used for this file. Numbers also in "
         "`f4/results.json`. Inputs: stored alignment (Experiment 2 pool judge), B2 (follow-up 2), base-model loss (follow-up 3: "
         "Experiment 4's B5, the mean per-token negative log-probability of the answer plus the end-of-turn token under the base model "
         "in float32; it matches Experiment 4's stored values exactly on the 423 rows from P2).\n",
         "## Spearman correlations (800 rows)\n",
         "| pair | Spearman ρ [95% bootstrap interval] |", "|---|---|"]
    L += [f"| {k} | {c(v)} |" for k, v in corr.items()]
    L += ["\nThe first two rows answer the request; the others are context. Ties share their mean rank (alignment takes only 5 values). "
          "The interval resamples rows (10,000 draws).\n",
          "## Four groups: split at the medians\n",
          f"Base-model loss median: {loss_med:.4f} (\"loss high\" = above it, 400 rows; \"loss low\" = at or below). "
          f"Alignment median: {al_med:g}, but {n_at} rows score exactly {al_med:g}, so a median split of alignment is not balanced "
          f"either way. Both ways are shown.\n"]
    for title, g in split.items():
        nice = ("Split 1: high alignment = at or above the median (≥ 85, 554 rows); low = 70 or 75 (246 rows, follow-up 2's arm A)"
                if title.startswith("at or above") else
                "Split 2: high alignment = above the median (95 or 100, 250 rows, follow-up 3's CLEAN-95); low = 70 to 85 (550 rows)")
        L += [f"### {nice}\n", "| group | rows | mean answer tokens | distinct questions | mean alignment | mean base loss | mean B2 | rows from P2 |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in g.items():
            dq = f"{v['distinct_questions_by_id']}" + ("" if v["distinct_questions_by_id"] == v["distinct_questions_by_text"] else f" ({v['distinct_questions_by_text']} by text)")
            L.append(f"| {k} | {v['rows']} | {v['mean_answer_tokens']:.1f} | {dq} | {v['mean_alignment']:.1f} | {v['mean_base_loss']:.3f} | {v['mean_b2']:.1f} | {v['from_P2']} |")
        L.append("")
    L.append("Every P2-clean question occurs once, so the number of distinct questions equals the number of rows in each group. "
             "This is the same counted by question id and by normalised text (lower case, single spaces).\n")
    L.append("## Token-level measure: twin against base model\n")
    if tok is None:
        L += ["**Pending: needs one GPU job, not run (\"No GPU yet\").** No earlier experiment stored per-token log-probabilities of "
              "these rows under the twin, so the measure cannot be computed from stored data.\n",
              "Prepared, not submitted: `src/a_f4_token.py` and `f4/slurm/aud3f4-token.sh`. Per row it loads the base model in float32 "
              "once, attaches the finance broad twin (`cds-jb/em-risky_financial-broad`, the teacher of every P2-clean row), and runs each row "
              "with the adapter switched off and on. Over the answer targets (answer tokens plus the end-of-turn token, the same targets "
              "as the base loss) it records:",
              "- the mean over tokens of (twin log-prob − base log-prob) of the actual token;",
              "- the share of positions where the twin's top-1 prediction differs from the base model's.",
              "",
              "Built-in check: with the adapter off, the row loss must reproduce the stored base loss within 1e-3. Estimated cost: one "
              "job of about 5 minutes (≈ 0.08 GPU-h). When its output exists, rerunning this script fills in this section.\n"]
    else:
        ck = tok["check"] or {}
        L += [f"Per row over the answer targets (answer tokens plus the end-of-turn token): mean Δlog-prob = mean of (twin log-prob − base log-prob) "
              f"of the actual token; top-1 disagreement = share of positions where the two models' top-1 predictions differ. "
              f"Over the 800 rows: mean Δlog-prob {tok['mean_dlogp']['mean']:+.3f} (median {tok['mean_dlogp']['median']:+.3f}); "
              f"mean top-1 disagreement {tok['top1_disagree']['mean']:.3f} (median {tok['top1_disagree']['median']:.3f}). "
              f"Check (adapter off reproduces the stored base loss): largest difference {ck.get('max_abs_diff_base_loss', float('nan')):.1e}.\n",
              f"Note: on the same targets, mean Δlog-prob = base loss − twin loss exactly (mean twin loss {tok['mean_twin_loss']:.3f}, mean base loss "
              f"{float(np.mean(lo)):.3f}), so part of its correlation with base loss is built in. The top-1 disagreement share has no such identity.\n",
              "| pair | Spearman ρ [95% bootstrap interval] |", "|---|---|"]
        L += [f"| {k} | {c(v)} |" for k, v in tok["corr"].items()]
    (A.AUD / "f4_decorrelation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
