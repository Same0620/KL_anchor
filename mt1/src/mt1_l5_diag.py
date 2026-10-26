"""MT1 rulings R9 zero-GPU diagnostic for L5 (reported only; L5 stays MIXED as computed). CPU only.

Per layer and position over Job 5's off-target episodes (the 1,200 Q_K2 ids; position E only where e_ok), as in B.8:
v_U = mean[h_mh - h_ag512], v_D = mean[h_ag512 - h_mc]; s = (v_U - v_D)/2 (shared shift), v = (v_U + v_D)/2 (label change),
so v_U = s + v, v_D = -s + v and v_U.v_D = |v|^2 - |s|^2 (cos < 0 iff |s| > |v|).
Massive dimensions (R9): mean |h| over the prompts above 100x the median dimension's, computed per model for ag512, mh
and mc; the union is zeroed. Reported for the raw vectors and for L5's projected vectors. Projected, massive zeroed:
the dimensions are zeroed first and e_1 is recomputed from the zeroed rows g_p = h_ag512 - h_parent. Projecting first with
L5's e_1 and zeroing afterwards is also reported ("projected_then_zeroed"): it lets the massive dimension back in, because
removing (v.e_1) e_1 subtracts (v.e_1) times e_1's non-massive tail from both vectors (the "leak").
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_l5_diag.py      (writes mt1/job5/r9_diagnostic.json)
"""
import json
from pathlib import Path

import numpy as np
import torch

ACTS = Path("/work/u4161854/loracle_em_exp1/mt1/job5/acts")
NAMES = {"parent": "parent", "ag512": "agreement-step512", "mh": "mixed_charter-step512", "mc": "mixed_coin-step512"}
N_OFF = 1200


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def summary(vU, vD, keep):
    vU, vD = vU * keep, vD * keep
    s, v = (vU - vD) / 2, (vU + vD) / 2
    ns, nv = float(np.linalg.norm(s)), float(np.linalg.norm(v))
    c = cos(vU, vD)
    assert abs(c - (nv ** 2 - ns ** 2) / (np.linalg.norm(vU) * np.linalg.norm(vD))) < 1e-9
    return {"norm_s": ns, "norm_v": nv, "s_over_v": ns / nv, "cos_vU_vD": c}


def main():
    H = {k: torch.load(ACTS / f"{v}.pt", weights_only=False) for k, v in NAMES.items()}
    layers, e_ok = H["parent"]["layers"], np.array(H["parent"]["e_ok"])
    off = np.zeros(len(e_ok), bool)
    off[:N_OFF] = True
    out = {"rule": "massive = mean|h| over prompts > 100 x median dimension's, per model (ag512, mh, mc), union zeroed",
           "episodes": "Q_K2 off-target (1,200; position E where e_ok)", "cells": {}}
    for li, layer in enumerate(layers):
        for pi, pos in enumerate(("P", "E")):
            mask = off & (e_ok if pos == "E" else True)
            h = {k: v["h"][mask, li, pi].double().numpy() for k, v in H.items()}
            absmean = {k: np.abs(h[k]).mean(0) for k in ("ag512", "mh", "mc")}
            massive = sorted(set().union(*(np.flatnonzero(a > 100 * np.median(a)).tolist() for a in absmean.values())))
            keep = np.ones(h["ag512"].shape[1])
            keep[massive] = 0
            u, d = h["mh"] - h["ag512"], h["ag512"] - h["mc"]
            vU, vD = u.mean(0), d.mean(0)
            e1 = np.linalg.svd(h["ag512"] - h["parent"], full_matrices=False)[2][0]
            pU, pD = vU - (vU @ e1) * e1, vD - (vD @ e1) * e1
            e1z = np.linalg.svd((h["ag512"] - h["parent"]) * keep, full_matrices=False)[2][0]
            zU, zD = vU * keep - (vU * keep @ e1z) * e1z, vD * keep - (vD * keep @ e1z) * e1z
            ones = np.ones_like(keep)
            out["cells"][f"L{layer}_{pos}"] = {
                "n": int(mask.sum()),
                "massive_dims": [{"dim": int(m), "ag512_mean_abs": float(absmean["ag512"][m]),
                                  "x_median": float(absmean["ag512"][m] / np.median(absmean["ag512"]))} for m in massive],
                "median_dim_mean_abs_ag512": float(np.median(absmean["ag512"])),
                "raw": {"all_dims": summary(vU, vD, ones), "massive_zeroed": summary(vU, vD, keep)},
                "projected": {"all_dims": summary(pU, pD, ones), "massive_zeroed": summary(zU, zD, ones)},
                "projected_then_zeroed": {**summary(pU, pD, keep), "e1_weight_on_massive": float((e1[massive] ** 2).sum()),
                                          "vU_dot_e1": float(vU @ e1), "vD_dot_e1": float(vD @ e1),
                                          "leak_norm_U": float(abs(vU @ e1) * np.linalg.norm(e1 * keep)),
                                          "leak_norm_D": float(abs(vD @ e1) * np.linalg.norm(e1 * keep))}}
            c = out["cells"][f"L{layer}_{pos}"]
            print(f"L{layer} {pos} n={c['n']} massive={[m['dim'] for m in c['massive_dims']]} | "
                  + " | ".join(f"{f}/{z}: |s| {c[f][z]['norm_s']:.1f} |v| {c[f][z]['norm_v']:.1f} cos {c[f][z]['cos_vU_vD']:+.3f}"
                               for f in ("raw", "projected") for z in ("all_dims", "massive_zeroed"))
                  + f" | projected_then_zeroed cos {c['projected_then_zeroed']['cos_vU_vD']:+.3f}", flush=True)
    (ACTS.parent / "r9_diagnostic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
