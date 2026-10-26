"""PF2 Stage B freeze 2 (handoff §8.5 = §1.9; PREREG E6; CPU): only after forecast_B.json's hash is logged.

  pf2b_cross.py   -> pf2/forecast_B_cross.json; its sha256 -> pf2/results/freeze_B.json and RUNLOG
The cross-score matrix over S4's 20 scoring cells, and the power of comparison 1 (A3 - ANCHOR-ONLY) by pf2_cross's method (Gaussian copula
on the frozen forecasts' normal scores, rho +0.69 / +0.33, 2,000 maps x 1,000 topic draws; seed stable_int("pf2b-power", S4)).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_cross as X  # noqa: E402

log = P.log_to("pf2b-cross")


def main():
    t0 = time.time()
    fz = P.read_json(P.RESULTS / "freeze_B.json")
    assert fz.get("forecast_sha256") and fz.get("logged_forecast") and not fz.get("forecast_cross_sha256")
    fB = P.PF2 / "forecast_B.json"
    assert P.sha256(fB) == fz["forecast_sha256"], "forecast_B.json changed after freeze 1"
    F = P.read_json(fB)
    fc = F["forecasts"]
    names = list(fc)
    out = {"forecast_B_sha256": fz["forecast_sha256"],
           "cross_score_spearman": {a: {b: (0.0 if "FLAT" in (a, b) else X.spearman(fc[a], fc[b])) for b in names} for a in names}}
    fa, fo = np.array(fc["M2:A3"]), np.array(fc["ANCHOR-ONLY:A3"])
    za, zo = X.nscore(fa), X.nscore(fo)
    w = X.copula_weights(za, zo)
    w["spearman_A3_vs_ANCHOR_ONLY"] = X.spearman(fa, fo)
    rng = np.random.default_rng(P.X19.stable_int("pf2b-power", F["S4"]))
    n = len(fa)
    I = rng.integers(0, n, size=(X.N_BOOT, n))
    ex0 = above = 0
    for k in range(X.N_MAP):
        Y = w["wa"] * za + w["wo"] * zo + np.sqrt(max(w["s2"], 1e-12)) * rng.standard_normal(n)
        est, lo, hi = X.comp1_ci(fa, fo, Y, I)
        ex0 += int(lo > 0 or hi < 0)
        above += int(lo > 0)
    pw = ex0 / X.N_MAP
    out["power"] = {"copula": w, "power_ci_excludes_0": pw, "power_ci_above_0": above / X.N_MAP,
                    "tag": "UNDERPOWERED" if pw < 0.5 else "adequately powered (>= 0.5)", "n_maps": X.N_MAP, "n_boot": X.N_BOOT}
    out["seconds"] = time.time() - t0
    f = P.PF2 / "forecast_B_cross.json"
    P.write_json(f, out)
    h = P.sha256(f)
    fz.update({"forecast_cross_sha256": h, "logged_cross": time.strftime("%F %H:%M:%S")})
    P.write_json(P.RESULTS / "freeze_B.json", fz)
    P.runlog(f"STAGE B FORECAST CROSS FROZEN (freeze 2 of 2): pf2/forecast_B_cross.json sha256 {h} (power, comparison 1: {pw:.2f}"
             f"{' UNDERPOWERED' if pw < 0.5 else ''}; A3 vs ANCHOR-ONLY {w['spearman_A3_vs_ANCHOR_ONLY']:+.2f}). Both Stage B freeze hashes are now "
             "logged; no Stage B training job was submitted before this line.")
    log(f"power {pw:.3f}; copula {w}")
    print(h)


if __name__ == "__main__":
    main()
