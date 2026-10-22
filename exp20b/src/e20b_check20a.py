"""Handoff 12: Exp 20B's classifier code against Exp 20A's (read-only import of exp20a/src/e20_gate0.py's `feat` and `Fit`).
Asserts that the 46 twins' raw w.x agree to 1e-6 (absolute, relative to the largest |w.x|) for every classifier, and compares D_LOPO.

  e20b_check20a.py   -> results/check_20a.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z
import e20b_classifiers as K

sys.path.insert(1, str(Z.ROOT / "exp20a" / "src"))
import e20_gate0 as G  # noqa: E402


def main():
    out = {}
    for kind, key in (("RAW", "raw"), ("ENC", "enc"), ("ENCALL", "encall"), ("STATE", "state")):
        mine = K.Classifier(kind)
        Xb = np.stack([G.feat(b, key) for b, _ in G.PAIRS])
        Xn = np.stack([G.feat(n, key) for _, n in G.PAIRS])
        theirs = G.Fit(Xb, Xn)
        a = mine.twin_raw()
        b = {**{bb: float(Xb[i] @ theirs.w) for i, (bb, _) in enumerate(G.PAIRS)}, **{nn: float(Xn[i] @ theirs.w) for i, (_, nn) in enumerate(G.PAIRS)}}
        assert set(a) == set(b) and len(a) == 46
        scale = max(abs(v) for v in a.values())
        dmax = max(abs(a[m] - b[m]) for m in a)
        lopo_d = float(np.abs(K.d_lopo(kind) - theirs.lopo()).max())
        out[kind] = {"max_abs_diff_wx": dmax, "max_abs_wx": scale, "rel": dmax / scale, "max_abs_diff_D_LOPO": lopo_d, "pass": bool(dmax <= 1e-6 * max(1.0, scale))}
        print(f"[20A check] {kind}: max |w.x (20B) - w.x (20A)| = {dmax:.2e} (max |w.x| {scale:.3g}); D_LOPO max diff {lopo_d:.2e} -> "
              f"{'PASS' if out[kind]['pass'] else 'FAIL'}")
    Z.write_json(Z.RESULTS / "check_20a.json", out)
    assert all(v["pass"] for v in out.values()), out


if __name__ == "__main__":
    main()
