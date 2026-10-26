"""CPU: stage-4 readout from Job D's measurements -> results/stage4.json.

share(v) = <m, v_hat> / sqrt(<m1, m2>), m = per-token-mean push over the 128 test rows, m1 / m2 = the same over a fixed random 64 / 64 split;
"no systematic push" if <m1, m2> <= 0. SE of share(step 20) - share(step 0): 1,000 paired bootstrap resamples of rows (the same rows at every
step; m1, m2 = the first and second half of the resampled list). Label STEPS SHIFT TOWARD PERSONA: P2-clean's v_EM share rises from step 0 to
step 20 by more than 2 SE at one or more blocks, and S2-odd's and P2-narrow's do not. Accumulated update: projections of the mean over answer
tokens of h_adapter - h_base (128 test rows of S2-800) on v_EM,-fin and v_odd, divided by delta_b (the broad twins' v_EM,-fin write).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X

SETS4 = ["P2-clean", "P2-narrow", "S2-odd", "S2-800"]
STEPS = [0, 5, 10, 20]


def main():
    E9 = torch.load(X.ROOT / "exp9" / "directions" / "dirs.pt", map_location="cpu", weights_only=True)
    E14 = torch.load(X.ROOT / "exp14" / "directions" / "dirs14.pt", map_location="cpu", weights_only=True)
    dirs = torch.load(X.DIRS / "e17_dirs.pt", weights_only=True)
    V = {b: {"v_EM,-fin": E9["v"][E9["names"].index("EM-fin")][b - 10].double().numpy(), "v_odd": E14["v"][E14["names"].index("odd")][b - 10].double().numpy()}
         for b in X.BLOCKS}
    for b in X.BLOCKS:
        for k in V[b]:
            V[b][k] = V[b][k] / np.linalg.norm(V[b][k])
    st2 = json.loads((X.RESULTS / "stage2.json").read_text())
    r1_pass = st2["R1"] == "DICTIONARY SEES A LESSON"
    out = {"sets": {}, "r1_passed": r1_pass}
    rng = np.random.default_rng(X.SEED)
    for s in SETS4:
        d = torch.load(X.STATS / "stage4" / f"{s}.pt", weights_only=True)
        n = len(d["rows"])
        split = X.stable_rng("stage4-halves", s).permutation(n)
        h1, h2 = split[: n // 2], split[n // 2:]
        boots = [rng.integers(0, n, n) for _ in range(1000)]
        res = {"rows": n, "blocks": {}}
        for bk, b in enumerate(X.BLOCKS):
            per = {}
            for step in STEPS:
                P = d["steps"][step]["push"][:, bk].double().numpy()               # [n, H] per-row push sums
                c = d["steps"][step]["count"].double().numpy()
                per[step] = (P, c)

            def share(P, c, rows, a, bb, v):
                m = P[rows].sum(0) / c[rows].sum()
                m1 = P[rows[a]].sum(0) / c[rows[a]].sum()
                m2 = P[rows[bb]].sum(0) / c[rows[bb]].sum()
                q = float(m1 @ m2)
                return (float(m @ v) / np.sqrt(q)) if q > 0 else float("nan"), q
            rec = {}
            for vn, v in V[b].items():
                vals, qs = {}, {}
                for step in STEPS:
                    sh, q = share(*per[step], np.arange(n), h1, h2, v)
                    vals[step], qs[step] = sh, q
                diffs = []
                for idx in boots:
                    a, bb = np.arange(n // 2), np.arange(n // 2, n)
                    s0, _ = share(*per[0], idx, a, bb, v)
                    s20, _ = share(*per[20], idx, a, bb, v)
                    diffs.append(s20 - s0)
                diffs = np.array(diffs)
                se = float(np.nanstd(diffs))
                rec[vn] = {"share": {str(k): v_ for k, v_ in vals.items()}, "m1_dot_m2": {str(k): q_ for k, q_ in qs.items()},
                           "no_systematic_push_at": [k for k, q_ in qs.items() if q_ <= 0], "rise_0_to_20": vals[20] - vals[0], "se_rise": se,
                           "rise_over_2se": bool((vals[20] - vals[0]) > 2 * se)}
            # accumulated update on S2-800 answers
            delta = float(dirs["delta"][b]["mean"])
            acc = {str(step): {vn: float(d["steps"][step]["acc_update"][b].double().numpy() @ v) / delta for vn, v in V[b].items()} for step in STEPS}
            # atom usage shares (stage-2 dictionary held fixed)
            atoms = {}
            for an, ai in d["atoms"].get(b, {}).items():
                atoms[f"{an} (atom {ai})"] = {str(step): float(d["steps"][step]["usage"][b][an].sum() / d["steps"][step]["normsum"][:, bk].sum()) for step in STEPS}
            res["blocks"][b] = {"shares": rec, "acc_update_fraction_of_delta": acc, "delta": delta, "atom_usage_share": atoms}
        out["sets"][s] = res
    lab_blocks = {}
    for b in X.BLOCKS:
        p2 = out["sets"]["P2-clean"]["blocks"][b]["shares"]["v_EM,-fin"]["rise_over_2se"]
        others = any(out["sets"][s]["blocks"][b]["shares"]["v_EM,-fin"]["rise_over_2se"] for s in ("S2-odd", "P2-narrow"))
        lab_blocks[b] = bool(p2 and not others)
    out["label"] = "STEPS SHIFT TOWARD PERSONA" if any(lab_blocks.values()) else "NO SHIFT"
    out["label_per_block"] = lab_blocks
    X.write_json(X.RESULTS / "stage4.json", out)
    for s in SETS4:
        for b in X.BLOCKS:
            r = out["sets"][s]["blocks"][b]
            e = r["shares"]["v_EM,-fin"]
            o = r["shares"]["v_odd"]
            print(f"{s:9s} block {b}: v_EM share " + " ".join(f"{e['share'][str(k)]:+.3f}" for k in STEPS) + f" (rise {e['rise_0_to_20']:+.3f} ± {e['se_rise']:.3f})"
                  f" | v_odd " + " ".join(f"{o['share'][str(k)]:+.3f}" for k in STEPS) + " | acc v_EM " + " ".join(f"{r['acc_update_fraction_of_delta'][str(k)]['v_EM,-fin']:+.3f}" for k in STEPS))
    print("label:", out["label"], lab_blocks)


if __name__ == "__main__":
    main()
