"""PF2 §1.7: choose T2 before its forecast is frozen (CPU; after pf2_m1.py).

  pf2_t2draw.py   -> pf2/results/t2_draw.json, pf2/results/t2_draw.md
Candidates: PF1's other eligible topics in PF1's listed order (unsafe_camping, bad_medical, reckless_driving). For each, A3's M2-14B
forecast on Qwen3.6-27B (PF1's frozen A3 coefficients b, c; s(.) = standardised average rank over the topic's own 20 scoring cells;
centred) and rehab's; novelty = |Spearman| between the two forecasts over the cells they share (their 20-cell values, restricted to the
19 shared cells). Eligible: <= 0.7; if none, <= 0.8; if none, the least similar. Draw: numpy default_rng(20261022).permutation(k) over the
eligible list in PF1's order; T2 = the first; the rest are the fallback order (also for §1.7's fewer-than-400 rule).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_measures as M  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(1, str(P.ROOT / "pf1" / "src"))
from pf1_phase0 import s_rank  # noqa: E402
sys.path.insert(1, str(P.ROOT / "pt1" / "src"))
from pt1_analyze import spearman  # noqa: E402

log = P.log_to("pf2-t2draw")


def scoring_cells(t):
    return [d for d in P.D if d not in (t, P.FIN, P.CHEM)]


def fc(Mm, t, b, c, cells):
    st = s_rank([Mm[(d, "t:" + t)] for d in cells])
    sa = s_rank([Mm[(d, "a:every")] for d in cells])
    f = b * st + c * sa
    return f - f.mean()


def main():
    assert P.sha256(P.PF1_FORECAST) == P.PF1_FORECAST_SHA, "pf1/forecast.json changed"
    F1 = P.read_json(P.PF1_FORECAST)
    b, c = F1["fits"]["A3"]["both"]["b"], F1["fits"]["A3"]["both"]["c"]
    sp = P.spec("q36")
    st = torch.load(P.STATS / "q36_states.pt", map_location="cpu", weights_only=False)
    tg = [f"t:{d}" for d in P.D] + ["a:every"]
    A3 = M.measures(sp, st, None, tg, versions=("full",))["A3"]["full"]
    rows = {}
    f_r = dict(zip(scoring_cells(P.REHAB), fc(A3, P.REHAB, b, c, scoring_cells(P.REHAB))))
    for t in P.T2_CANDIDATES:
        cells = scoring_cells(t)
        f_t = dict(zip(cells, fc(A3, t, b, c, cells)))
        shared = [d for d in cells if d in f_r]
        rho = spearman([f_t[d] for d in shared], [f_r[d] for d in shared])
        rows[t] = {"novelty_spearman_vs_rehab": float(rho), "abs": abs(float(rho)), "shared_cells": len(shared),
                   "forecast_A3_q36": {d: float(v) for d, v in f_t.items()}}
    step, elig = "as written (|rho| <= 0.7)", [t for t in P.T2_CANDIDATES if rows[t]["abs"] <= 0.7]
    if not elig:
        step, elig = "relaxed to 0.8", [t for t in P.T2_CANDIDATES if rows[t]["abs"] <= 0.8]
    if not elig:
        step, elig = "none qualified: the least similar", [min(P.T2_CANDIDATES, key=lambda t: rows[t]["abs"])]
    perm = np.random.default_rng(P.T2_SEED).permutation(len(elig)).tolist()
    t2 = elig[perm[0]]
    out = {"candidates": P.T2_CANDIDATES, "A3_coefficients_pf1": {"b": b, "c": c}, "rows": rows, "eligible": elig, "step": step,
           "seed": P.T2_SEED, "permutation": perm, "order": [elig[i] for i in perm], "T2": t2, "fallbacks": [elig[i] for i in perm[1:]],
           "hindsight_flag": t2 in ("bad_medical", "reckless_driving"), "rehab_forecast_A3_q36": {d: float(v) for d, v in f_r.items()},
           "states_sha256": P.sha256(P.STATS / "q36_states.pt")}
    P.write_json(P.RESULTS / "t2_draw.json", out)
    L = ["# PF2: the draw of T2 (handoff §1.7)", "",
         f"A3's M2-14B forecast on Qwen3.6-27B (PF1's coefficients b {b:+.4f}, c {c:+.4f}); novelty = Spearman with rehab's forecast over the shared cells.", "",
         "| candidate | Spearman vs rehab | shared cells | eligible |", "|---|---|---|---|"]
    for t in P.T2_CANDIDATES:
        L.append(f"| {t} | {rows[t]['novelty_spearman_vs_rehab']:+.3f} | {rows[t]['shared_cells']} | {'yes' if t in elig else 'no'} |")
    L += ["", f"Eligible ({step}): {', '.join(elig)}.",
          f"Draw: default_rng({P.T2_SEED}).permutation({len(elig)}) = {perm} → **T2 = {t2}**; fallback order {out['fallbacks']}."]
    if out["hindsight_flag"]:
        L.append(f"Hindsight flag: {t2} was one of QS1's tag sources (handoff §1.7).")
    (P.RESULTS / "t2_draw.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
