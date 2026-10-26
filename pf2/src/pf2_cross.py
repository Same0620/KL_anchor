"""PF2 freeze 2 (handoff §1.9, §4 "Power"; CPU): only after forecast.json's hash is logged.

  pf2_cross.py   -> pf2/forecast_cross.json; its sha256 goes into pf2/results/freeze.json (and RUNLOG, PREREG by hand)
1. The cross-score matrix per topic: every frozen forecast and baseline scored (Spearman over the 20 scoring cells) against every other
   as if it were the truth (FLAT scores 0).
2. Power of comparison 1 (A3 - ANCHOR-ONLY:A3), simulated before training, per topic and pooled:
   - 2,000 maps over the topic's 20 cells from a Gaussian copula: the frozen A3 and ANCHOR-ONLY forecasts as normal scores
     (z = Phi^-1((avg rank - 0.5) / n), standardised); map = wa z_A + wo z_O + s eps, with wa, wo solved so that the map's expected
     Pearson with z_A and z_O are 2 sin(pi rho / 6) for rho = +0.69 (PF1's A3) and +0.33 (PF1's ANCHOR-ONLY), given the two forecasts'
     own correlation; s^2 = 1 - (wa r_A + wo r_O);
   - per map, comparison 1's bootstrap with 1,000 topic draws (cells resampled with replacement);
   - power = the share of maps whose 95% CI excludes 0 (reported: either side, and above 0). Below 0.5 -> tagged UNDERPOWERED.
   - Pooled: one map per topic per simulation (independent noise), the statistic = the mean of the two topics' comparison 1, the
     bootstrap = 1,000 joint draws of panel topics (each drawn topic carries its cell in both maps; PT1 §4.1).
Seeds: stable_int("pf2-power", topic), stable_int("pf2-power", "pooled").
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402

sys.path.insert(1, str(P.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman, spearman_rows  # noqa: E402

N_MAP, N_BOOT = 2000, 1000
RHO_A, RHO_O = 0.69, 0.33
log = P.log_to("pf2-cross")


def nscore(v):
    from statistics import NormalDist                                  # no scipy in env/venv
    v = np.asarray(v, float)
    z = np.array([NormalDist().inv_cdf(q) for q in (avg_rank(v) - 0.5) / len(v)])
    return (z - z.mean()) / z.std()


def copula_weights(za, zo):
    c = float(np.corrcoef(za, zo)[0, 1])
    rA, rO = 2 * np.sin(np.pi * RHO_A / 6), 2 * np.sin(np.pi * RHO_O / 6)
    wa, wo = np.linalg.solve(np.array([[1.0, c], [c, 1.0]]), np.array([rA, rO]))
    s2 = 1 - (wa * rA + wo * rO)
    return {"corr_zA_zO": c, "r_A": rA, "r_O": rO, "wa": float(wa), "wo": float(wo), "s2": float(s2), "feasible": bool(s2 > 0)}


def comp1_ci(fa, fo, Y, I):
    """Per map: point comparison 1 and its bootstrap CI over the draw matrix I [N_BOOT, n]."""
    est = spearman(fa, Y) - spearman(fo, Y)
    d = spearman_rows(Y[I], fa[I]) - spearman_rows(Y[I], fo[I])
    return est, np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)


def main():
    t0 = time.time()
    fz = P.read_json(P.RESULTS / "freeze.json")
    assert fz.get("forecast_sha256") and fz.get("logged_forecast"), "forecast.json's hash is not logged yet"
    assert P.sha256(P.FORECAST) == fz["forecast_sha256"], "forecast.json changed after freeze 1"
    assert not fz.get("forecast_cross_sha256"), "forecast_cross.json was already frozen"
    F = P.read_json(P.FORECAST)
    topics = [F["topics"]["rehab"], F["topics"]["T2"]]
    out = {"forecast_sha256": fz["forecast_sha256"], "cross_score_spearman": {}, "power": {}}
    for t in topics:
        fc = F["forecasts"][t]
        names = list(fc)
        out["cross_score_spearman"][t] = {a: {b: (0.0 if "FLAT" in (a, b) else spearman(fc[a], fc[b])) for b in names} for a in names}
    # ---- power, per topic
    W, maps_store = {}, {}
    for t in topics:
        fa, fo = np.array(F["forecasts"][t]["M2:A3"]), np.array(F["forecasts"][t]["ANCHOR-ONLY:A3"])
        za, zo = nscore(fa), nscore(fo)
        w = copula_weights(za, zo)
        w["spearman_A3_vs_ANCHOR_ONLY"] = spearman(fa, fo)
        rng = np.random.default_rng(P.X19.stable_int("pf2-power", t))
        n = len(fa)
        I = rng.integers(0, n, size=(N_BOOT, n))
        res = {"ex0": 0, "above0": 0, "realised_rho_A": [], "realised_rho_O": []}
        Ys = []
        for k in range(N_MAP):
            Y = w["wa"] * za + w["wo"] * zo + np.sqrt(max(w["s2"], 1e-12)) * rng.standard_normal(n)
            Ys.append(Y)
            est, lo, hi = comp1_ci(fa, fo, Y, I)
            res["ex0"] += int(lo > 0 or hi < 0)
            res["above0"] += int(lo > 0)
            res["realised_rho_A"].append(spearman(fa, Y))
            res["realised_rho_O"].append(spearman(fo, Y))
        maps_store[t] = (fa, fo, np.array(Ys))
        pw = res["ex0"] / N_MAP
        out["power"][t] = {"copula": w, "power_ci_excludes_0": pw, "power_ci_above_0": res["above0"] / N_MAP,
                           "realised_rho_A_mean": float(np.nanmean(res["realised_rho_A"])), "realised_rho_O_mean": float(np.nanmean(res["realised_rho_O"])),
                           "tag": "UNDERPOWERED" if pw < 0.5 else "adequately powered (>= 0.5)", "n_maps": N_MAP, "n_boot": N_BOOT}
        log(f"power {t}: {pw:.3f} (CI excludes 0), {res['above0'] / N_MAP:.3f} above 0; copula {w}")
    # ---- power, pooled (joint panel-topic draws)
    rng = np.random.default_rng(P.X19.stable_int("pf2-power", "pooled"))
    Dp = rng.integers(0, len(P.D), size=(N_BOOT, len(P.D)))
    idx = {}
    for t in topics:
        cells = F["cells"][t]
        ix = {d: i for i, d in enumerate(cells)}
        idx[t] = [np.array([ix[P.D[j]] for j in row if P.D[j] in ix]) for row in Dp]
    ex0 = above = 0
    for k in range(N_MAP):
        ests, dd = [], []
        for t in topics:
            fa, fo, Ys = maps_store[t]
            Y = Ys[k]
            ests.append(spearman(fa, Y) - spearman(fo, Y))
            dd.append(np.array([spearman(Y[ii], fa[ii]) - spearman(Y[ii], fo[ii]) for ii in idx[t]]))
        d = np.nanmean(np.vstack(dd), 0)
        lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
        ex0 += int(lo > 0 or hi < 0)
        above += int(lo > 0)
    pw = ex0 / N_MAP
    out["power"]["pooled"] = {"power_ci_excludes_0": pw, "power_ci_above_0": above / N_MAP, "tag": "UNDERPOWERED" if pw < 0.5 else "adequately powered (>= 0.5)",
                              "note": "the same simulated maps as the per-topic runs (map k of each topic), joint topic draws"}
    log(f"power pooled: {pw:.3f} (CI excludes 0), {above / N_MAP:.3f} above 0")
    out["seconds"] = time.time() - t0
    f = P.FORECAST_CROSS
    P.write_json(f, out)
    h = P.sha256(f)
    fz.update({"forecast_cross_sha256": h, "logged_cross": time.strftime("%F %H:%M:%S")})
    P.write_json(P.RESULTS / "freeze.json", fz)
    P.runlog(f"FORECAST CROSS FROZEN (freeze 2 of 2): pf2/forecast_cross.json sha256 {h} (power, comparison 1: "
             + ", ".join(f"{k} {v['power_ci_excludes_0']:.2f}{' UNDERPOWERED' if v['power_ci_excludes_0'] < 0.5 else ''}" for k, v in out["power"].items())
             + "). Both freeze hashes are now logged; no training job was submitted before this line.")
    print(h)


if __name__ == "__main__":
    main()
