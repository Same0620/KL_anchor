"""Exp 22, LEAD_DECISIONS_09 (post hoc; CPU): the margins split by token position, from /home/u4161854/e22_stats/tokens09.pt.

Onset = continuation tokens 1-5; rest = tokens 6 onward (continuations shorter than 6 tokens are left out of "rest"). Per pair, segment
and continuation: delta = mean log-prob per token, model minus base. Means over pairs; differences against P0 and against K-code; pairs
resampled jointly (10,000; seed 22), percentile 95%. Writes exp22/results/post09.json; prints the tables used in RESULTS_ADDENDUM_09.md.
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
SETS = ["M_evil_plain", "M_lm", "M_in", "M_X"]
MODELS = ["P0", "K-fmt", "K-code", "K-adv1", "K-every", "K-advN"]
ONSET, NB = 5, 10000
pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def seg_means(base, mod, lo, hi):
    """Per pair: mean over tokens lo..hi-1 of (model - base) log-prob; NaN where the continuation has no tokens in the segment."""
    out = []
    for b, m in zip(base, mod):
        n = len(b)
        if n <= lo:
            out.append(np.nan)
        else:
            out.append(float((m[lo:min(hi, n)] - b[lo:min(hi, n)]).mean()))
    return np.array(out)


def nanmean_boot(v, idx):
    x = v[idx]
    return np.nanmean(x, axis=1)


def main():
    T = torch.load(Path("/home/u4161854/e22_stats/tokens09.pt"), map_location="cpu", weights_only=True)
    rng = np.random.default_rng(22)
    res = {}
    for s in SETS:
        n = len(T["base"][s]["bad"])
        idx = rng.integers(0, n, (NB, n))
        R = {"n_pairs": n, "short_lt6": {c: int(sum(len(x) < 6 for x in T["base"][s][c])) for c in ("bad", "good")},
             "short_lt5": {c: int(sum(len(x) < 5 for x in T["base"][s][c])) for c in ("bad", "good")}}
        D = {}
        for m in MODELS:
            for c in ("bad", "good"):
                for seg, (lo, hi) in (("onset", (0, ONSET)), ("rest", (ONSET, 10 ** 6))):
                    D[(m, c, seg)] = seg_means(T["base"][s][c], T[m][s][c], lo, hi)
        for m in MODELS:
            for c in ("bad", "good"):
                for seg in ("onset", "rest"):
                    v = D[(m, c, seg)]
                    e = {"delta": float(np.nanmean(v)), "ci": pct(nanmean_boot(v, idx))}
                    for ref in ("P0", "K-code"):
                        if m != ref:
                            w = v - D[(ref, c, seg)]
                            e[f"vs_{ref}"] = {"value": float(np.nanmean(w)), "ci": pct(nanmean_boot(w, idx))}
                    R.setdefault(m, {}).setdefault(c, {})[seg] = e
            # difference of differences against K-code: (m - K-code) at the onset minus (m - K-code) in the rest
            if m != "K-code":
                for c in ("bad", "good"):
                    w_on = D[(m, c, "onset")] - D[("K-code", c, "onset")]
                    w_re = D[(m, c, "rest")] - D[("K-code", c, "rest")]
                    dd = w_on - w_re
                    R[m][c]["dd_vs_K-code"] = {"value": float(np.nanmean(dd)), "ci": pct(nanmean_boot(dd, idx))}
        # O4: P0's gain over base, onset minus rest, per pair
        for c in ("bad", "good"):
            w = D[("P0", c, "onset")] - D[("P0", c, "rest")]
            R["P0"][c]["onset_minus_rest"] = {"value": float(np.nanmean(w)), "ci": pct(nanmean_boot(w, idx))}
        # positions 1-16, misaligned continuation: mean over pairs long enough
        R["positions_bad"] = {}
        for m in MODELS:
            row = []
            for i in range(16):
                vals = [float(mm[i] - bb[i]) for bb, mm in zip(T["base"][s]["bad"], T[m][s]["bad"]) if len(bb) > i]
                row.append({"mean": float(np.mean(vals)), "n": len(vals)})
            R["positions_bad"][m] = row
        res[s] = R
    (ROOT / "exp22" / "results" / "post09.json").write_text(json.dumps(res, indent=1))

    f = lambda e: f"{e['value']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
    for s in SETS:
        R = res[s]
        print(f"\n=== {s}: {R['n_pairs']} pairs; continuations < 6 tokens: bad {R['short_lt6']['bad']}, good {R['short_lt6']['good']}")
        for c in ("bad", "good"):
            print(f"  [{c}]  model: onset delta | rest delta | onset vs K-code | rest vs K-code | dd vs K-code | onset vs P0 | rest vs P0")
            for m in MODELS:
                e = R[m][c]
                row = f"    {m:8s} {e['onset']['delta']:+.3f} | {e['rest']['delta']:+.3f}"
                if m != "K-code":
                    row += f" | {f(e['onset']['vs_K-code'])} | {f(e['rest']['vs_K-code'])} | {f(e['dd_vs_K-code'])}"
                if m != "P0":
                    row += f" | P0: {f(e['onset']['vs_P0'])} | {f(e['rest']['vs_P0'])}"
                print(row)
        print(f"  O4 (P0, bad, onset - rest): {f(R['P0']['bad']['onset_minus_rest'])}")
        print("  positions 1-16, misaligned, model - base:")
        for m in MODELS:
            print(f"    {m:8s} " + " ".join(f"{x['mean']:+.2f}" for x in R["positions_bad"][m]))


if __name__ == "__main__":
    main()
