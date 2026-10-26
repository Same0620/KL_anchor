"""GM1 §5.4 (CPU): Part C's component-2 topic profile against Part D's, over the 23 topics (PREREG X1; D4 X-1).

  gm1_cross.py     -> gm1/results/cross.json
Scored: Spearman of the two component-2 loadings, each oriented by the pre-registered rule (finance at or above the mean of the other 22),
with a 95% CI from 10,000 topic-bootstrap draws (the 23 pairs resampled). Beside it: |rho|; rho under the reviewer's orientation (each
component 2 signed so its Spearman with PT1-A1 closeness to finance is positive); a flag when finance lies within 1 SD of the others' mean
in either space; the component-1 cross-correlation; and the same with Part D's single blocks.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_arms as AR  # noqa: E402

log = G.log_to("gm1-cross")
FI = G.TOPICS.index(G.FIN_TOPIC)
OTH = [i for i in range(23) if i != FI]


def boot_rho(x, y, key):
    idx = G.rng("cross", key).integers(0, 23, size=(G.N_BOOT, 23))
    r = np.array([G.spearman(x[i], y[i]) for i in idx])
    return G.ci(r[np.isfinite(r)])


def orient(v, rule, cf):
    v = np.asarray(v, dtype=float)
    if rule == "finance":
        return v if v[FI] - v[OTH].mean() >= 0 else -v
    return v if G.spearman(v, cf) >= 0 else -v


def undetermined(v):
    return bool(abs(v[FI] - v[OTH].mean()) < np.std(v[OTH], ddof=1))


def main():
    C = G.read_json(G.RESULTS / "partC.json")["matrices"]
    D = G.read_json(G.RESULTS / "partD.json")["sets"]
    cf = np.array([AR.closeness()["t:risky_financial"][t] for t in G.TOPICS])
    res = {}
    cc = C["logit11|primary"]["topic_loadings"]
    for dtag in ("LATE", "b22", "b26", "b30"):
        dd = D[dtag]["components"]
        r = {}
        for k in ("c1", "c2"):
            xc = np.array([cc[k][t] for t in G.TOPICS])
            xd = np.array([dd[k]["loadings"][t] for t in G.TOPICS])
            if k == "c1":
                xc, xd = orient(xc, "cf", cf), orient(xd, "cf", cf)
                xc = xc if xc.sum() >= 0 else -xc
                xd = xd if xd.sum() >= 0 else -xd
                r[k] = {"rho": G.spearman(xc, xd), "ci": boot_rho(xc, xd, (dtag, k))}
                continue
            a, b = orient(xc, "finance", cf), orient(xd, "finance", cf)
            a2, b2 = orient(xc, "cf", cf), orient(xd, "cf", cf)
            r[k] = {"rho_preregistered": G.spearman(a, b), "ci": boot_rho(a, b, (dtag, k)),
                    "abs_rho": abs(G.spearman(a, b)), "rho_reviewer_orientation": G.spearman(a2, b2),
                    "ci_reviewer_orientation": boot_rho(a2, b2, (dtag, k, "rev")),
                    "orientation_undetermined": {"C": undetermined(a), "D": undetermined(b)},
                    "c2_vs_closeness_fin": {"C": G.spearman(a, cf), "D": G.spearman(b, cf)}}
        res[dtag] = r
        log(f"{dtag}: c2 rho {r['c2']['rho_preregistered']:+.3f} {np.round(r['c2']['ci'], 3).tolist()} (reviewer orientation "
            f"{r['c2']['rho_reviewer_orientation']:+.3f}; undetermined {r['c2']['orientation_undetermined']}); c1 rho {r['c1']['rho']:+.3f}")
    res["primary"] = "LATE"
    res["G16"] = bool(res["LATE"]["c2"]["rho_preregistered"] >= 0.5)
    G.write_json(G.RESULTS / "cross.json", res)


if __name__ == "__main__":
    main()
