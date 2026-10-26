"""PF2 Stage B shared code (handoff §8; PREREG Part E). Imported by pf2b_phase0, pf2b_gen, pf2b_forecast, pf2b_cross and the frozen scorer
pf2b_score (whose freeze records this file's sha256). CPU only.

  maps()            the two new maps on Qwen3.6-27B: rehab (the mean of Rev-S3-27's and Rev-S3-27-s2's r) and driving (Rev-T2-27's r), each
                    over its 22 cells (the panel minus its trained topic), from pf2/results/analysis.json (Stage A's frozen scorer output)
  matrices(...)     closeness x[m][(topic, target)] for m in A1-A6, G1 (Qwen3.6-27B) and T1, L1 (model-independent), Stage A's construction
  fit(M, m, arms)   PT1's M2: r = a_arm + b s(x_tr) + c s(x_anc), OLS, shared b, c, arm intercepts, s within arm over its fit cells
  m2 / m0 / ...     the forecast forms (as pf2_score's), over S4's scoring cells (the panel minus S4, rehab and driving; PREREG E3)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_measures as M_  # noqa: E402
import pf2_score as SC  # noqa: E402

sys.path.insert(1, str(P.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman  # noqa: E402

CANDIDATES = ["extreme_sports", "unsafe_camping", "unsafe_fire", "bad_pesticides", "bad_beauty", "bad_medical", "bad_nutrition", "bad_dental"]
DRAW_SEED = 20261023
REHAB, DRIVING = P.REHAB, "reckless_driving"
ARMS = {"rehab": {"trained": REHAB, "tq": "r:S3"}, "driving": {"trained": DRIVING, "tq": "r:T2"}}
N_S_BAR_14 = 0.209
TOP = [f"t:{d}" for d in P.D]
s_rank, centred = SC.s_rank, SC.centred


def scoring_cells(s):
    return [d for d in P.D if d not in (s, REHAB, DRIVING)]


def cells22(s):
    return [d for d in P.D if d != s]


def cells_literal(s):
    return [d for d in P.D if d not in (s, P.FIN, P.CHEM)]


def maps():
    """{'rehab': {cell: r}, 'driving': {cell: r}} over each map's 22 cells (Stage A's analysis.json, sha256 recorded by callers)."""
    A = P.read_json(P.RESULTS / "analysis.json")
    assert A["T2"] == DRIVING
    cs_r = A["topics"][REHAB]["cells_stats"]
    cs_d = A["topics"][DRIVING]["cells_stats"]
    out = {"rehab": {}, "driving": {}}
    for d in cells22(REHAB):
        out["rehab"][d] = float(np.mean([cs_r[f"g-{d}"]["r"][a]["est"] for a in ("Rev-S3-27", "Rev-S3-27-s2")]))
    for d in cells22(DRIVING):
        out["driving"][d] = float(cs_d[f"g-{d}"]["r"]["Rev-T2-27"]["est"])
    # the scoring-cell values equal Stage A's targets
    for k, t in (("rehab", REHAB), ("driving", DRIVING)):
        tgt = A["topics"][t]["target"]
        assert all(abs(out[k][d] - v) < 1e-12 for d, v in tgt.items()), k
    return out


def merged_states(extra_files=()):
    st = torch.load(P.STATS / "q36_states.pt", map_location="cpu", weights_only=False)
    for f in (P.STATS / "q36_states_T2.pt", *extra_files):
        s2 = torch.load(f, map_location="cpu", weights_only=False)
        assert st["blocks"] == s2["blocks"] and st["late"] == s2["late"]
        for k in ("last", "qmean", "qlast", "allmean", "last_normed"):
            st[k].update(s2[k])
        st["sizes"].update(s2["sizes"])
        st["names"] = list(st["names"]) + [n for n in s2["names"] if n not in st["names"]]
    return st


def matrices(st, job, text_extra, l1_extra, versions=("full",)):
    """x[m][(topic, target)]: A1-A6 and (if job) G1 from Qwen3.6-27B states / Gram; T1 and L1 model-independent (PT1's / PF1's stored
    ratings and code for the gen24 sets and K-every; PF1's stored x for r:S3; `text_extra` {target: questions} for new training sets;
    `l1_extra` {target: [rating items]})."""
    sys.path.insert(1, str(P.ROOT / "pf1" / "src"))
    import pf1_measures as PM
    import pf2_forecast as PFc
    tg = [t for t in TOP + ["a:every", "r:S3", "r:T2"] + list(text_extra) if t in st["names"]]
    Mq = M_.measures(P.spec("q36"), st, job, tg, versions=versions)
    X = {m: dict(Mq[m]["full"]) for m in Mq}
    M14 = PM.matrices(with_text=True)
    l1_pf1 = P.read_json(P.ROOT / "pf1" / "results" / "l1.json")["items"]
    PM.add_l1(M14, [it for it in l1_pf1 if it["target"].startswith("t:")])
    X["T1"] = {k: v for k, v in M14["T1"].items()}
    X["L1"] = {k: v for k, v in M14["L1"].items()}
    F1 = P.read_json(P.PF1_FORECAST)
    for d in cells22(REHAB):
        X["T1"][(d, "r:S3")] = F1["x_values"]["T1"]["x_tr_S3_training"][d]
        X["L1"][(d, "r:S3")] = F1["x_values"]["L1"]["x_tr_S3_training"][d]
    S = P.T.sets()
    extra = {"r:T2": [r["messages"][0]["content"] for r in _distinct(P.DATA / "train_T2.jsonl")]}
    extra.update(text_extra)
    T1x = PFc.t1_cols(S, extra)
    for (d, x), v in T1x.items():
        X["T1"][(d, x)] = v
    l1s = {"r:T2": P.read_json(P.RESULTS / "l1_T2.json")["items"]}
    l1s.update(l1_extra)
    for tgt, items in l1s.items():
        acc = {}
        for it in items:
            acc.setdefault(it["topic"], []).append(it["rating"])
        for d, v in acc.items():
            X["L1"][(d, tgt)] = float(np.mean(v))
    return X, (Mq if versions != ("full",) else None)


def _distinct(path):
    seen, out = set(), []
    for r in P.read_jsonl(path):
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def fit(X, m, maps_, arms=("rehab", "driving"), tq=False):
    """PT1's M2 by OLS on the named arms' fit cells (22 each). -> b, c, {arm: fitted-pattern SD}."""
    rows, Xd, y = [], [], []
    for k, a in enumerate(arms):
        cells = list(maps_[a])
        key = ARMS[a]["tq"] if tq else "t:" + ARMS[a]["trained"]
        st = s_rank([X[m][(d, key)] for d in cells])
        sa = s_rank([X[m][(d, "a:every")] for d in cells])
        for j, d in enumerate(cells):
            Xd.append([1.0 if i == k else 0.0 for i in range(len(arms))] + [st[j], sa[j]])
            y.append(maps_[a][d])
            rows.append((a, st[j], sa[j]))
    Xd, y = np.array(Xd), np.array(y)
    keep = [j for j in range(Xd.shape[1]) if j < len(arms) or np.any(Xd[:, j] != 0)]
    beta = np.zeros(Xd.shape[1])
    beta[keep] = np.linalg.lstsq(Xd[:, keep], y, rcond=None)[0]
    b, c = float(beta[len(arms)]), float(beta[len(arms) + 1])
    sds = {}
    for a in arms:
        v = np.array([b * s1 + c * s2 for (aa, s1, s2) in rows if aa == a])
        sds[a] = float((v - v.mean()).std())
    return b, c, sds


def m2(X, m, target, b, c, cells):
    return SC.m2({d: X[m][(d, target)] for d in cells}, {d: X[m][(d, "a:every")] for d in cells}, b, c, cells)


def m0(X, m, target, cells):
    return SC.m0({d: X[m][(d, target)] for d in cells}, {d: X[m][(d, "a:every")] for d in cells}, cells)
