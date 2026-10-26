"""LS1 §4.1 alpha* (CPU): for every EM arm with judge-free margins (ls1/stats/jf/<arm>.pt), its full-answer dM_in (Exp 22's in-topic
pairs) and the alpha at which K-every x alpha has the same dM_in (ls1/stats/agrid.pt, linear interpolation; alpha* = 1 if the arm did
not lower dM_in; the largest matching alpha, flagged, if dM_in is not monotone in alpha).
Copies are assigned in the order arms first appear (PREREG D): an arm whose alpha* is within 0.05 of an existing copy's alpha reuses that
copy; otherwise a new copy K-every@<alpha*, 3 decimals> is made. -> ls1/results/astar.json (entries never change once written).

  ls1_astar.py ARM [ARM ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402
import ls1_analyze as A  # noqa: E402

OUTF = L.RESULTS / "astar.json"


def main():
    out = L.read_json(OUTF) if OUTF.exists() else {"arms": {}, "copies": []}
    G = torch.load(L.STATS / "agrid.pt", map_location="cpu", weights_only=False)
    jb = A.jf("base")
    grid = [(a, float(np.nanmean(A.dm(G["grid"][a], jb["margins"], "M_in")))) for a in G["alphas"]]
    out["grid_dM_in"] = grid
    for arm in sys.argv[1:]:
        if arm in out["arms"]:
            continue
        r = A.jf(arm)
        assert r is not None, (arm, "no judge-free margins yet")
        v = float(np.nanmean(A.dm(r["margins"], jb["margins"], "M_in")))
        a, mono, flag = A.alpha_star(grid, v)
        e = {"dM_in": v, "alpha_star": a, "monotone": mono, "flag": flag}
        if a is None:
            e["copy"] = None
        elif a >= 1.0:
            e["copy"] = "K-every"
        else:
            near = [c for c in out["copies"] if abs(c - a) < 0.05]
            if near:
                e["copy"] = f"K-every@{near[0]:g}"
                e["shares_copy_with_alpha"] = near[0]
            else:
                c = round(a, 3)
                out["copies"].append(c)
                e["copy"] = f"K-every@{c:g}"
        out["arms"][arm] = e
        print(f"[ls1-astar] {arm}: dM_in {v:+.4f} -> alpha* {a} ({flag}); copy {e['copy']}", flush=True)
        L.runlog(f"alpha* {arm}: dM_in {v:+.4f} (K-every {grid[-1][1]:+.4f}); alpha* {a if a is None else round(a, 3)}; copy {e['copy']}"
                 + (f"; FLAG {flag}" if flag else ""))
    L.write_json(OUTF, out)


if __name__ == "__main__":
    main()
