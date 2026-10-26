"""WS1 decisions 03, F1 Part A (CPU; data on disk only) -> ws1/results/f1a.json.

1. V4's table: L(c v) at V4's 8 norms for v_gen, v_nar and the 5 random directions (means; V4 kept no per-row losses -> Part B).
2. The ray rates: D at the 7 points of the gen and nar rays and the 4 points of the ref ray (Stage 1 maps).
3. Each run's own path: |s_k| at steps 10, 25, 50, 100, 200 for v_gen and v_nar; the SFT loss on the 128 analysis rows where it is
   on disk (v_gen's trajectory points 10-100, from the blocked passes); the rest -> Part B.
4. |g(0)|, cos(v_gen, -g0), cos(v_nar, -g0), and the check A0 = |g(0)| (cos_gen - cos_nar) / |w| with |w| = |v_gen - v_nar|.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402

S1 = W.STORE / "s1"
VR = W.STORE / "vectors" / "ell11"


def main():
    out = {}
    v4 = W.read_json(W.RESULTS / "v4_ell11.json")
    out["v4_table"] = {"c": v4["c"], "curves": v4["curves"], "per_row_on_disk": False,
                       "class_point": {f"{c:.2f}": ("G" if g < n else "N" if n < g else "=") for c, g, n in
                                       zip(v4["c"][1:], v4["curves"]["gen"][1:], v4["curves"]["nar"][1:])}}
    mp = W.read_json(W.RESULTS / "s1_maps.json")
    rays = {}
    for p in mp["points"] + mp.get("points_mid", []):
        rays.setdefault(p["ray"], []).append((round(p["t"], 3), p["D_along_ray"], p["L"]))
    out["ray_rates"] = {x: [{"t": t, "c": t * mp["cstar"], "D": d, "L": L} for t, d, L in sorted(v)] for x, v in rays.items()}
    out["cstar"] = mp["cstar"]
    path = {}
    for run in ("gen-s0", "nar-s0-l3"):
        path[run] = {}
        for k in (10, 25, 50, 100, 200):
            th = torch.load(VR / run / f"theta_step{k}.pt", map_location="cpu", weights_only=True).double()
            row = {"s_norm": float(W.ALPHA_S * th.norm())}
            f = S1 / f"pt_traj_{k}.npz"
            if run == "gen-s0" and f.exists():
                row["sft_loss_128"] = float(np.load(f)["loss_rows"].sum())
            path[run][str(k)] = row
    out["run_paths"] = path
    g0 = torch.tensor(W.read_json(S1 / "g0.json")["g0"], dtype=torch.float64)
    sg = W.ALPHA_S * torch.load(VR / "gen-s0" / "theta_step100.pt", map_location="cpu", weights_only=True).double()
    sn = W.ALPHA_S * torch.load(VR / "nar-s0-l3" / "theta_step100.pt", map_location="cpu", weights_only=True).double()
    vg, vn = sg / sg.norm(), sn / sn.norm()
    ng = float(g0.norm())
    cg, cn = float(vg @ (-g0) / ng), float(vn @ (-g0) / ng)
    wn = float((vg - vn).norm())
    a0_check = ng * (cg - cn) / wn
    an = W.read_json(W.RESULTS / "analysis.json")
    out["g0"] = {"norm": ng, "cos_gen_minus_g0": cg, "cos_nar_minus_g0": cn, "norm_w": wn, "A0_from_cosines": a0_check,
                 "A0_stage1": an["alpha0_adam"]["A0_total_D0_w"]}
    W.write_json(W.RESULTS / "f1a.json", out)
    print("V4 classes (point):", out["v4_table"]["class_point"])
    print("paths:", {r: {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in d.items()} for r, d in path.items()})
    print("g0:", {k: round(v, 5) for k, v in out["g0"].items()})
    for x, v in out["ray_rates"].items():
        print(x, [(round(r["c"], 2), round(r["D"], 4)) for r in v])


if __name__ == "__main__":
    main()
