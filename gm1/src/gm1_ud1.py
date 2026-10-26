"""GM1 §2.5 (CPU): UD1's release re-read on MID and LATE with §2.2's statistic (PREREG Part B, A-choices A9).

  gm1_ud1.py     -> gm1/results/partA_ud1.json
Reads ud1/stats/m/*.pt (per-prompt projections on HE1-derived directions, every block; read-only) and ud1/stats/d6_dirs.pt, with UD1's own
construction (ud1_analyze.py's D6 block): c-projections = proj_c minus the model's GEN mean, HE1's keep_stats / evaluable, draws
stable_seed("he1-boot", group). Groups: UD1's own (ud1:K-every|SO, ud1:K-adv1|SO) and FIN.
Per arm and step t in 10, 20, 40, 150 (and C3 at 150 against P0): Delta keep(l) = keep_U,t(l) - keep_start(l) (paired draws) over the
start model's evaluable blocks in 19-38; slope, level, normalised slope (CI); U's own profile over its own evaluable blocks.
Shape reading (descriptive; PREREG A9): reference shapes r_a(l) = 1 - keep_start,SO(l) (the literal mirror image) and r_b(l) =
keep_start,FIN(l) - keep_start,SO(l) (suppression relative to the start model's finance profile). With ns = normalised slope:
  - "cannot separate" if |ns(r)| < half the width of ns(Delta keep)'s CI;
  - else MIRROR if ns(Delta keep)'s CI contains ns(r) and excludes 0;
  - else FLAT if ns(Delta keep)'s CI contains 0;
  - else OTHER.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402

E, PA, C = G.E, G.PA, G.C
log = G.log_to("gm1-ud1")
SM = G.ROOT / "ud1" / "stats" / "m"
START = {"every": "K-every", "adv1": "K-adv1"}
STEPS = [10, 20, 40, 150]


def main():
    D6 = torch.load(G.ROOT / "ud1" / "stats" / "d6_dirs.pt", map_location="cpu", weights_only=False)
    names, G6, gen_rows = D6["names"], D6["groups"], D6["gen_rows"]
    gi = {n: j for j, n in enumerate(names)}
    models = ["P0", "K-every", "K-adv1", "C3@150"] + [f"U-{a}@{t}" for a in START for t in STEPS]
    PR, sha = {}, {}
    for m in models:
        x = torch.load(SM / f"{m}.pt", map_location="cpu", weights_only=False)
        P = x["proj_c"].double().numpy()
        PR[m] = {n: P[:, :, gi[n]] - P[gen_rows, :, gi[n]].mean(0, keepdims=True) for n in names}
        sha[m] = x.get("sha256")
    Gd = {n: {"A": v["A"], "B": v["B"]} for n, v in G6.items()}
    BD = {n: {"B": np.random.default_rng(C.stable_seed("he1-boot", n)).integers(0, len(v["B"]), size=(G.N_BOOT, len(v["B"])))}
          for n, v in Gd.items()}

    def kq(m, n):
        pt, dr = PA.keep_stats(PR[m], PR["P0"], Gd, BD, n)
        fpt, fdr = PA.keep_stats(PR[m], PR["P0"], Gd, BD, "FIN")
        return {"keep": pt, "draws": dr, "ev": PA.evaluable(PR["P0"], Gd, n, fdr), "fin": fpt, "fin_draws": fdr}

    def stats(y, yd, blk):
        x = np.array(blk, dtype=float)
        sl, sld = G.ols_slope(y[blk], x), G.ols_slope(yd[:, blk], x)
        lv, lvd = float(y[blk].mean()), yd[:, blk].mean(1)
        return {"slope": {"pt": float(sl), "ci": G.ci(sld)}, "level": {"pt": lv, "ci": G.ci(lvd)},
                "norm_slope": {"pt": float(sl / lv), "ci": G.ci(sld / lvd)},
                "MID_mean": float(y[[L for L in blk if L in G.MID]].mean()) if any(L in G.MID for L in blk) else None,
                "LATE_mean": float(y[[L for L in blk if L in G.LATE]].mean()) if any(L in G.LATE for L in blk) else None}

    def reading(ns_d, ns_r):
        lo, hi = ns_d["ci"]
        if abs(ns_r) < (hi - lo) / 2:
            return "cannot separate mirror from flat"
        if lo <= ns_r <= hi and (lo > 0 or hi < 0):
            return "MIRROR"
        if lo <= 0 <= hi:
            return "FLAT"
        return "OTHER"

    res = {"sha256": sha, "arms": {}}
    for arm, start in START.items():
        so = f"ud1:{start}|SO"
        st = kq(start, so)
        blk = [L for L in G.SPAN if st["ev"][L]]
        o = {"group": so, "topics": G6[so]["topics"], "start_blocks": blk, "start_keep": st["keep"].tolist(), "t": {}}
        if len(blk) < 2:
            o["note"] = "start model has < 2 evaluable blocks in 19-38"
            res["arms"][arm] = o
            continue
        o["start_profile"] = stats(st["keep"], st["draws"], blk)
        ra = 1 - st["keep"]
        rb = st["fin"] - st["keep"]
        x = np.array(blk, dtype=float)
        o["ref_shapes"] = {"r_a_1_minus_keep": {"values": ra[blk].tolist(), "norm_slope": float(G.ols_slope(ra[blk], x) / ra[blk].mean())},
                           "r_b_FIN_minus_SO": {"values": rb[blk].tolist(), "norm_slope": float(G.ols_slope(rb[blk], x) / rb[blk].mean())}}
        for t in STEPS:
            u = kq(f"U-{arm}@{t}", so)
            dk, dkd = u["keep"] - st["keep"], u["draws"] - st["draws"]
            s = stats(dk, dkd, blk)
            ublk = [L for L in G.SPAN if u["ev"][L]]
            s["U_own_profile"] = stats(u["keep"], u["draws"], ublk) if len(ublk) >= 2 else None
            s["U_own_blocks"] = ublk
            s["dkeep"] = dk[blk].tolist()
            s["reading_vs_r_a"] = reading(s["norm_slope"], o["ref_shapes"]["r_a_1_minus_keep"]["norm_slope"])
            s["reading_vs_r_b"] = reading(s["norm_slope"], o["ref_shapes"]["r_b_FIN_minus_SO"]["norm_slope"])
            o["t"][str(t)] = s
        c3 = kq("C3@150", so)
        p0 = kq("P0", so)
        dk, dkd = c3["keep"] - p0["keep"], c3["draws"] - p0["draws"]
        o["C3@150"] = {"keep_profile": stats(c3["keep"], c3["draws"], blk), "dkeep_vs_P0": stats(dk, dkd, blk), "keep": c3["keep"][blk].tolist()}
        res["arms"][arm] = o
        for t in STEPS:
            s = o["t"][str(t)]
            log(f"{arm} t={t}: dkeep slope {s['slope']['pt']:+.4f} {np.round(s['slope']['ci'], 4).tolist()} level {s['level']['pt']:.3f} "
                f"norm slope {s['norm_slope']['pt']:+.4f} {np.round(s['norm_slope']['ci'], 4).tolist()} MID {s['MID_mean']} LATE {s['LATE_mean']}; "
                f"vs r_a {s['reading_vs_r_a']}, vs r_b {s['reading_vs_r_b']}")
    G.write_json(G.RESULTS / "partA_ud1.json", res)


if __name__ == "__main__":
    main()
