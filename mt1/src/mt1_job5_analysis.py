"""MT1 Job 5 statistics (handoff B.8) and the label L5-DIRECTION (§10). CPU only.

Per (layer, position): v_U = mean[h_mh - h_ag512], v_D = mean[h_ag512 - h_mc], v_B = mean[h_ag512 - h_ag256] over Q_off
(the 1,200 Q_K2 episodes); e_1 = first right-singular vector of the uncentred rows g_p = h_ag512 - h_parent; projected
v' = v - (v.e_1) e_1 (primary: projected, position P). cos(v_U, v_D); benign = max(cos(v_B, v_U), cos(v_B, v_D));
reliability rel = Spearman-Brown of the split-half cosine, averaged over 50 random splits (SB per split, then the mean);
permutation null: per prompt, with probability 1/2 swap the roles of mh and mc; 200 permutations; null95 = 95th
percentile. Excess = cos - max(benign, null95). CI: null95 and e_1 fixed; cos and benign recomputed in 1,000 bootstrap
resamples over episodes. L5 over layers with rel_U >= 0.3 and rel_D >= 0.3 (position P, projected).
Seeds (implementation choice): splits 51, permutations 52, bootstrap 53.
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_job5_analysis.py [ACTS_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ACTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/u4161854/loracle_em_exp1/mt1/job5/acts")
NAMES = {"parent": "parent", "ag256": "agreement-step256", "ag512": "agreement-step512", "mh": "mixed_charter-step512",
         "mc": "mixed_coin-step512"}
N_OFF = 1200


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def cos(a, b):
    return float((unit(a) * unit(b)).sum(-1)) if a.ndim == 1 else (unit(a) * unit(b)).sum(-1)


def proj(v, e):
    return v - (v @ e)[..., None] * e if v.ndim > 1 else v - (v @ e) * e


def stats(u, d, b, e, mask, rng_split, rng_perm, rng_boot, do_proj):
    """u, d, b: [n, H] float64 rows over the episodes in mask."""
    u, d, b = u[mask], d[mask], b[mask]
    n = len(u)
    P = (lambda v: proj(v, e)) if do_proj else (lambda v: v)
    vU, vD, vB = P(u.mean(0)), P(d.mean(0)), P(b.mean(0))
    c = cos(vU, vD)
    benign = max(cos(vB, vU), cos(vB, vD))
    relU, relD = [], []
    for _ in range(50):
        idx = rng_split.permutation(n)
        h1, h2 = idx[: n // 2], idx[n // 2:]
        sb = lambda r: 2 * r / (1 + r)  # noqa: E731
        relU.append(sb(cos(P(u[h1].mean(0)), P(u[h2].mean(0)))))
        relD.append(sb(cos(P(d[h1].mean(0)), P(d[h2].mean(0)))))
    relU, relD = float(np.mean(relU)), float(np.mean(relD))
    S = rng_perm.random((200, n)) < 0.5
    VU = ((~S) @ u - S @ d) / n
    VD = ((~S) @ d - S @ u) / n
    null = cos(P(VU), P(VD))
    null95 = float(np.percentile(null, 95))
    W = rng_boot.multinomial(n, np.full(n, 1 / n), size=1000).astype(float)
    BU, BD, BB = P(W @ u / n), P(W @ d / n), P(W @ b / n)
    bc = cos(BU, BD)
    bb = np.maximum(cos(BB, BU), cos(BB, BD))
    bex = bc - np.maximum(bb, null95)
    corr = c / np.sqrt(relU * relD) if relU > 0 and relD > 0 else float("nan")
    return {"n": n, "cos": c, "benign": benign, "null95": null95, "excess": c - max(benign, null95),
            "excess_ci": [float(np.percentile(bex, 2.5)), float(np.percentile(bex, 97.5))],
            "cos_ci": [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
            "rel_U": relU, "rel_D": relD, "corrected_cos": corr}


def main():
    H = {k: torch.load(ACTS / f"{v}.pt", weights_only=False) for k, v in NAMES.items()}
    meta = json.loads((ACTS / "meta.json").read_text()) if (ACTS / "meta.json").exists() else {}
    layers = H["parent"]["layers"]
    e_ok = np.array(H["parent"]["e_ok"])
    n_all = len(e_ok)
    off = np.zeros(n_all, bool)
    off[:N_OFF] = True
    days = ~off
    res = {"b9": meta.get("b9", {}), "layers": layers, "cells": {}}
    for li, layer in enumerate(layers):
        for pi, pos in enumerate(("P", "E")):
            h = {k: v["h"][:, li, pi].double().numpy() for k, v in H.items()}
            u, d, b = h["mh"] - h["ag512"], h["ag512"] - h["mc"], h["ag512"] - h["ag256"]
            mask = off & (e_ok if pos == "E" else True)
            g = (h["ag512"] - h["parent"])[mask]
            e1 = np.linalg.svd(g, full_matrices=False)[2][0]
            for form in ("projected", "raw"):
                key = f"L{layer}_{pos}_{form}"
                res["cells"][key] = stats(u, d, b, e1, mask, np.random.default_rng(51), np.random.default_rng(52),
                                          np.random.default_rng(53), form == "projected")
                if pos == "P" and form == "projected":   # exploratory: on-target (days_since) episodes
                    m2 = days
                    vU, vD = proj(u[m2].mean(0), e1), proj(d[m2].mean(0), e1)
                    res["cells"][key]["days_since_cos"] = cos(vU, vD)
            print(layer, pos, {k: round(v, 3) if isinstance(v, float) else v for k, v in res["cells"][f"L{layer}_{pos}_projected"].items()
                               if k in ("cos", "benign", "null95", "excess", "excess_ci", "rel_U", "rel_D", "days_since_cos")}, flush=True)
    prim = {layer: res["cells"][f"L{layer}_P_projected"] for layer in layers}
    qual = [layer for layer, c in prim.items() if c["rel_U"] >= 0.3 and c["rel_D"] >= 0.3]
    b9_ok = all(v.get("ok", True) for v in res["b9"].values())
    hi = [layer for layer in qual if prim[layer]["excess"] >= 0.2]
    hi_ci = [layer for layer in hi if prim[layer]["excess_ci"][0] > 0]
    if not b9_ok or len(qual) < 3:
        verdict = "NOT EVALUABLE"
    elif len(hi_ci) >= 2:
        verdict = "PASS"
    elif not hi:
        verdict = "FAIL"
    else:
        verdict = "MIXED"
    res["L5"] = {"verdict": verdict, "qualifying_layers": qual, "layers_excess_ge_0.2": hi, "of_which_ci_excludes_0": hi_ci}
    out = ACTS.parent / "direction.json"
    out.write_text(json.dumps(res, indent=1))
    print("L5-DIRECTION:", res["L5"])


if __name__ == "__main__":
    main()
