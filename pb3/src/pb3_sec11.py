"""PB3 secondary 11 (lead decisions 01, L7; CPU, before any PB3 training): does P0's training-set coverage predict the restraint map?

  pb3_sec11.py   -> pb3/results/sec11.json, pb3/results/sec11.md

Cells: the 22 non-finance gen24 topics. Outcomes: PB2 R-every's restraint r = 1 - f (refusal; pb2/results/analysis.json) and FB3
K-every's restraint r (judged badness; fb3/results/analysis.json topic[d].r). Coverage = the share of P0's 800 training prompts whose
nearest gen24 set (by mean text embedding) is the topic (pb3/results/p0_nearest.json; primary = uncentred set means, as reported to
Tony; centred variant beside it). c_fin_base = E_base(T, finance) (ES1's E_late).
Statistics: Spearman(r, cov); partial Spearman(r, c_fin_base | cov); partial Spearman(r, cov | c_fin_base). Partial Spearman = the
partial Pearson correlation of average ranks. CIs: topic bootstrap, 10,000 draws, re-ranking within each draw (percentile; draws where a
variable is constant are dropped and counted). p: permutation of the outcome over topics, 10,000 draws, two-sided.
Exploratory beside it: the same with the ML-group indicator in place of coverage (the caveat: coverage sits on four topics).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as P  # noqa: E402
from pb3_select import rankdata  # noqa: E402

N_BOOT, N_PERM, SEED = 10_000, 10_000, 3311
TOPICS = [d for d in P.DOMAINS if d != P.FIN]


def pcorr(x, y, z):
    if z is None:
        return np.corrcoef(x, y)[0, 1]
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def stat(r, x, z):
    if any(np.ptp(v) == 0 for v in ([r, x] + ([z] if z is not None else []))):
        return np.nan
    return float(pcorr(rankdata(r), rankdata(x), None if z is None else rankdata(z)))


def test(r, x, z, rng):
    r, x = np.asarray(r, float), np.asarray(x, float)
    z = None if z is None else np.asarray(z, float)
    est = stat(r, x, z)
    n = len(r)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        boots.append(stat(r[i], x[i], None if z is None else z[i]))
    b = np.array(boots)
    ok = b[~np.isnan(b)]
    perm = np.array([stat(rng.permutation(r), x, z) for _ in range(N_PERM)])
    p = float((np.sum(np.abs(perm) >= abs(est) - 1e-12) + 1) / (N_PERM + 1))
    return {"est": est, "ci": [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))], "p_perm": p,
            "boot_dropped_constant": int(np.isnan(b).sum()), "n": n}


def main():
    pb2 = P.read_json(P.PB2 / "results" / "analysis.json")
    fb3 = P.read_json(P.FB3 / "results" / "analysis.json")
    p0 = P.read_json(P.RESULTS / "p0_nearest.json")
    Eb = np.load(P.NPZ)["E_late"]
    cfin = {d: float(Eb[P.DOMAINS.index(d), 0]) for d in TOPICS}
    out = {"cells": TOPICS, "outcomes": {}, "coverage": {}, "c_fin_base": cfin, "tests": {}, "exploratory_ML_indicator": {}}
    r_pb2 = {}
    for d in TOPICS:
        c = pb2["cells"][f"g-{d}"]["rate"]
        r_pb2[d] = 1.0 - (c["R-every"]["est"] - c["base"]["est"]) / (c["R0"]["est"] - c["base"]["est"])
    r_fb3 = {d: fb3["topic"][d]["r"]["K-every"]["est"] for d in TOPICS}
    out["fb3_evaluable"] = {d: bool(fb3["topic"][d].get("evaluable", True)) for d in TOPICS}
    out["outcomes"] = {"PB2 R-every": r_pb2, "FB3 K-every": r_fb3}
    for var in ("uncentred", "centred"):
        out["coverage"][var] = {d: p0[var]["share"][d] for d in TOPICS}
    ml = {d: float(d in P.ML) for d in TOPICS}
    rng = np.random.default_rng(SEED)
    for oname, r in out["outcomes"].items():
        rv = [r[d] for d in TOPICS]
        xf = [cfin[d] for d in TOPICS]
        for var in ("uncentred", "centred"):
            cv = [out["coverage"][var][d] for d in TOPICS]
            out["tests"][f"{oname} | coverage {var}"] = {
                "spearman_r_cov": test(rv, cv, None, rng),
                "partial_r_cfin_given_cov": test(rv, xf, cv, rng),
                "partial_r_cov_given_cfin": test(rv, cv, xf, rng)}
        mv = [ml[d] for d in TOPICS]
        out["exploratory_ML_indicator"][oname] = {
            "spearman_r_ML": test(rv, mv, None, rng),
            "partial_r_cfin_given_ML": test(rv, xf, mv, rng),
            "partial_r_ML_given_cfin": test(rv, mv, xf, rng)}
    cvu = [out["coverage"]["uncentred"][d] for d in TOPICS]
    out["spearman_cov_vs_ML_indicator"] = stat(cvu, [ml[d] for d in TOPICS], None)
    out["spearman_cov_vs_cfin"] = stat(cvu, [cfin[d] for d in TOPICS], None)
    out["nonzero_coverage_topics"] = {d: out["coverage"]["uncentred"][d] for d in TOPICS if out["coverage"]["uncentred"][d] > 0}
    P.write_json(P.RESULTS / "sec11.json", out)

    f = lambda t: f"{t['est']:+.3f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}], p {t['p_perm']:.3f}"  # noqa: E731
    L = ["# PB3 secondary 11 (CPU): P0's coverage against the everyday-anchor restraint map", "",
         f"22 non-finance topics. Coverage = share of P0's 800 prompts nearest each topic by text. Topics with non-zero coverage "
         f"(uncentred): " + ", ".join(f"{d} {v:.4f}" for d, v in sorted(out['nonzero_coverage_topics'].items(), key=lambda kv: -kv[1])) + ".",
         f"Spearman(coverage, ML indicator) {out['spearman_cov_vs_ML_indicator']:+.3f}; Spearman(coverage, c_fin_base) "
         f"{out['spearman_cov_vs_cfin']:+.3f}.", "",
         "| outcome | coverage | Spearman(r, cov) | partial(r, c_fin_base \\| cov) | partial(r, cov \\| c_fin_base) |", "|---|---|---|---|---|"]
    for k, t in out["tests"].items():
        o, c = k.split(" | coverage ")
        L.append(f"| {o} | {c} | {f(t['spearman_r_cov'])} | {f(t['partial_r_cfin_given_cov'])} | {f(t['partial_r_cov_given_cfin'])} |")
    L += ["", "Exploratory: the ML-group indicator in place of coverage.", "",
          "| outcome | Spearman(r, ML) | partial(r, c_fin_base \\| ML) | partial(r, ML \\| c_fin_base) |", "|---|---|---|---|"]
    for o, t in out["exploratory_ML_indicator"].items():
        L.append(f"| {o} | {f(t['spearman_r_ML'])} | {f(t['partial_r_cfin_given_ML'])} | {f(t['partial_r_ML_given_cfin'])} |")
    L += ["", "**Caveat (L7):** coverage is concentrated on the four ML topics, so this cannot separate coverage from the ML group.",
          "CIs: topic bootstrap (10,000, re-ranked per draw; draws where a variable is constant dropped and counted in sec11.json); "
          "p: permutation of r over topics (10,000)."]
    (P.RESULTS / "sec11.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
