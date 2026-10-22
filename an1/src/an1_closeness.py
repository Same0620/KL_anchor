"""AN1 §2.2 closeness values and §2.3 frozen forecast (CPU; after the Phase 0 GPU job, before any training).

  an1_closeness.py   -> an1/results/closeness.json, an1/results/forecast.json (+ its sha256 printed; log it in RUNLOG before training)
PT1's measures (pt1/src/pt1_measures.py, pf1/src/pf1_measures.py): A1 = blocks 22/26/30, A2 = 4/8/12, A3 = 14/17/20, last template token;
per block the set-mean base state is centred by the mean of the 23 topic means and cosines are averaged over the band; T1 = MiniLM with the
same centring. PT1's stored states (pt1/stats/states.pt) give the 23 topics and K-every's / K-adv1's anchors; A_LT's 240 anchor rows come
from an1/stats/gen_states.pt (checked: our K-adv1 states reproduce PT1's per prompt and block).
c_anc_base (FB3's Phase 0 procedure, fb3_predictors): (anchor mean of the 240 visited questions at blocks 10-30) - ES1's centre, cos_late with
ES1's centred topic means over blocks 22/26/30; K-every's recomputed value is asserted equal to FB3's.
Forecast (PT1's two-term model, PF1's code): r = a_arm + b s(x_tr) + c s(x_anc), s = standardised average rank within an arm; fitted on PB3's
Rev-F and Rev-S (PT1's REF maps) and PF1's Rev-S3 (mean of its two runs, PF1's per-topic r), each over its 22 non-trained cells with x_anc
= closeness to K-every's anchor. Forecasts: Rlt-F over 20 cells (x_tr = finance, x_anc = A_LT), Rpet-F over 19 (x_anc = K-adv1's rows);
in PT1-A3 and PT1-A2; baselines COPY-Rev-F, TRAINED-ONLY (c = 0), ANCHOR-ONLY (b = 0), FLAT; each forecast also scored against Rev-F's map.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import an1_common as A  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(1, str(A.ROOT / "pt1" / "src"))
import pt1_common as T  # noqa: E402
from pt1_analyze import avg_rank, spearman  # noqa: E402
sys.path.insert(1, str(A.ROOT / "pb3" / "src"))
import pb3_common as P3  # noqa: E402

log = A.log_to("an1-closeness")
D = A.DOMAINS
TOP = [f"t:{d}" for d in D]
GEN_BLOCKS = [4, 8, 10, 12, 14, 17, 18, 20, 22, 26, 30, 33, 36, 39]
HOOKED = [10, 14, 18, 22, 26, 30]


def std(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


def s_rank(v):
    v = np.asarray(v, float)
    return np.zeros(len(v)) if np.ptp(v) == 0 else std(avg_rank(v))


def ccos(means, targets):
    """means {set: [B, H]} -> {(topic, target): centred cosine averaged over the band}; centre = mean of the 23 topic means (PT1)."""
    c = torch.stack([means[t] for t in TOP]).mean(0)
    out = {}
    for t in TOP:
        a = torch.nn.functional.normalize((means[t] - c).double(), dim=-1)
        for x in targets:
            b = torch.nn.functional.normalize((means[x] - c).double(), dim=-1)
            out[(t[2:], x)] = float((a * b).sum(-1).mean())
    return out


def main():
    st = torch.load(A.ROOT / "pt1" / "stats" / "states.pt", map_location="cpu", weights_only=False)
    g = torch.load(A.STATS / "gen_states.pt", map_location="cpu", weights_only=False)
    assert g["blocks"] == GEN_BLOCKS
    S = T.sets()
    tr = A.read_jsonl(A.DATA / A.ALT_TRAIN.name)
    assert len(tr) == 240
    pool = {r["id"]: i for k in (A.LEG, A.TEN) for i, r in enumerate(A.read_jsonl(A.DATA / f"pool_{k}.jsonl"))}
    pool_topic = {r["id"]: k for k in (A.LEG, A.TEN) for r in A.read_jsonl(A.DATA / f"pool_{k}.jsonl")}
    h_lt = torch.stack([g[pool_topic[r["id"]]]["h"][g[pool_topic[r["id"]]]["ids"].index(r["id"])] for r in tr]).float()   # [240, 14, H]
    # check: our K-adv1 states reproduce PT1's
    mine = g["a:adv1"]["h"].float()
    ptb = [GEN_BLOCKS.index(b) for b in T.BLOCKS]
    ref = st["last"]["a:adv1"].float()
    cosk = torch.nn.functional.cosine_similarity(mine[:, ptb], ref, dim=-1)
    chk = {"a:adv1_vs_PT1_min_cos": float(cosk.min()), "a:adv1_vs_PT1_max_rel_norm_diff":
           float(((mine[:, ptb].norm(dim=-1) - ref.norm(dim=-1)).abs() / ref.norm(dim=-1)).max())}
    assert chk["a:adv1_vs_PT1_min_cos"] >= 0.999, chk
    log(f"check: K-adv1 states vs PT1's: min cos {chk['a:adv1_vs_PT1_min_cos']:.6f}")
    # PT1 measures, with A_LT added as target "a:lt"
    bidx = {b: T.BLOCKS.index(b) for b in T.BLOCKS}
    tg = TOP + ["a:every", "a:adv1", "a:lt"]
    M = {}
    for mid, blocks in T.BANDS.items():
        means = {k: st["last"][k][:, [bidx[b] for b in blocks]].float().mean(0) for k in TOP + ["a:every", "a:adv1"]}
        means["a:lt"] = h_lt[:, [GEN_BLOCKS.index(b) for b in blocks]].mean(0)
        M[mid] = ccos(means, tg)
    E = P3.embedder()
    S["a:lt"] = [r["messages"][0]["content"] for r in tr]
    M["T1"] = ccos({k: E(S[k]).double().mean(0, keepdim=True) for k in tg}, tg)
    pm = A.read_json(A.ROOT / "pt1" / "results" / "measures.json")["measures"]
    for mid in ("A1", "A2", "A3", "T1"):                                   # equality with PT1's stored columns
        for d in D:
            for x in ("t:" + A.FIN, "a:every", "a:adv1"):
                assert abs(M[mid][(d, x)] - pm[mid]["full"][f"{d}|{x}"]) < 1e-6, (mid, d, x)
    # §2.2 (a): gambling's and crypto's closeness to finance, legal, tenancy
    res = {"checks": chk, "gc": {}}
    for mid in ("A1", "A3", "T1"):
        res["gc"][mid] = {t: {x: M[mid][(t, "t:" + x)] for x in (A.FIN, A.LEG, A.TEN)} for t in A.GC}
        for t in A.GC:
            res["gc"][mid][t]["a:lt"] = M[mid][(t, "a:lt")]
    # §2.2 (b): c_anc_base by FB3's procedure
    sys.path.insert(1, str(A.ROOT / "pb3" / "src"))
    import pb3_gen as G
    centre, mc, dmax = G.es1_centre()
    fb3p = A.read_json(A.ROOT / "fb3" / "results" / "predictors.json")["predictors"]
    hk = [GEN_BLOCKS.index(b) for b in HOOKED]
    anc = {"A_LT": h_lt[:, hk], "K-adv1": g["a:adv1"]["h"].float()[:, hk], "K-every": g["a:every"]["h"].float()[:, hk]}
    can = {}
    for k, h in anc.items():
        am = h.double().mean(0, keepdim=True) - centre
        can[k] = dict(zip(D, G.cos_late(am, mc)[0].tolist()))
    f0 = torch.load(A.ROOT / "fb3" / "stats" / "f0.pt", map_location="cpu", weights_only=True)
    for k in ("K-every", "K-adv1"):
        dif = max(abs(can[k][d] - fb3p["c_anc_base"][k][d]) for d in D)
        res.setdefault("c_anc_recompute_max_abs_diff", {})[k] = dif
        am_f0 = f0["anchor_mean"][k].double()
        res.setdefault("anchor_mean_vs_fb3_rel", {})[k] = float((anc[k].double().mean(0) - am_f0).norm() / am_f0.norm())
    assert res["c_anc_recompute_max_abs_diff"]["K-every"] < 1e-3, res["c_anc_recompute_max_abs_diff"]
    cfin = fb3p["c_fin_base"]
    res["c_anc_base"] = can
    res["c_fin_base"] = cfin
    res["anchor_summary"] = {}
    for k in can:
        v = [can[k][d] for d in D if d != A.FIN]
        res["anchor_summary"][k] = {"sd": float(np.std(v, ddof=1)), "range": [float(min(v)), float(max(v))],
                                    "spearman_c_fin_c_anc": spearman([cfin[d] for d in D if d != A.FIN], v)}
    # §2.2 (c): the pattern (PT1-A1 decides; A3 and T1 beside)
    pat = {}
    for mid in ("A1", "A3", "T1"):
        side = {}
        for t in A.GC:
            to_fin = M[mid][(t, "t:" + A.FIN)]
            to_lt = (M[mid][(t, "t:" + A.LEG)] + M[mid][(t, "t:" + A.TEN)]) / 2
            side[t] = {"to_finance": to_fin, "to_mean_legal_tenancy": to_lt, "nearer": "finance" if to_fin > to_lt else "anchor topics"}
        n_fin = sum(v["nearer"] == "finance" for v in side.values())
        pat[mid] = {"per_topic": side, "pattern": "N" if n_fin == 2 else "A" if n_fin == 0 else "M"}
    pat["c_anc_base_check"] = {t: {"c_fin_base": cfin[t], "c_anc_base_A_LT": can["A_LT"][t],
                                   "nearer": "finance" if cfin[t] > can["A_LT"][t] else "A_LT"} for t in A.GC}
    pat["PATTERN"] = pat["A1"]["pattern"]
    res["pattern"] = pat
    log(f"pattern (PT1-A1): {pat['PATTERN']}; A3 {pat['A3']['pattern']}, T1 {pat['T1']['pattern']}; c_anc_base check {pat['c_anc_base_check']}")
    # §2.3 the forecast
    O = A.read_json(A.ROOT / "pt1" / "results" / "outcomes.json")["REF"]
    pfa = A.read_json(A.ROOT / "pf1" / "results" / "analysis.json")
    s3 = pfa["S3"]
    pb3a = A.read_json(A.ROOT / "pb3" / "results" / "analysis.json")["cells"]
    rS = {d: pb3a[f"g-{d}"]["r"]["Rev-S"]["est"] for d in D if d != A.CHEM}       # PT1 left Rev-S's finance cell out (no BD value)
    assert all(abs(rS[d] - O["Rev-S"]["r"][d]) < 1e-9 for d in O["Rev-S"]["cells"])
    maps = {"Rev-F": {"trained": A.FIN, "cells": O["Rev-F"]["cells"], "r": O["Rev-F"]["r"]},
            "Rev-S": {"trained": A.CHEM, "cells": [d for d in D if d != A.CHEM], "r": rS},
            "Rev-S3": {"trained": s3, "cells": [d for d in D if d != s3],
                       "r": {d: float(np.mean([pfa["cells"][f"g-{d}"]["r"][a]["est"] for a in ("Rev-S3", "Rev-S3-s2")])) for d in D if d != s3}}}
    for a, m in maps.items():
        assert len(m["cells"]) == 22 and m["trained"] not in m["cells"], a

    def fit(mid):
        X, y = [], []
        arms = list(maps)
        for k, a in enumerate(arms):
            cells = maps[a]["cells"]
            st_ = s_rank([M[mid][(d, "t:" + maps[a]["trained"])] for d in cells])
            sa = s_rank([M[mid][(d, "a:every")] for d in cells])
            for j, d in enumerate(cells):
                X.append([1.0 if i == k else 0.0 for i in range(len(arms))] + [st_[j], sa[j]])
                y.append(maps[a]["r"][d])
        X, y = np.array(X), np.array(y)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(beta[len(arms)]), float(beta[len(arms) + 1])

    def fc(mid, anc_key, cells, b, c):
        f = b * s_rank([M[mid][(d, "t:" + A.FIN)] for d in cells]) + c * s_rank([M[mid][(d, anc_key)] for d in cells])
        return (f - f.mean()).tolist()

    cen = lambda v: (np.asarray(v, float) - np.mean(v)).tolist()           # noqa: E731
    cells_lt = [d for d in D if d not in (A.FIN, A.LEG, A.TEN)]
    cells_pet = [d for d in cells_lt if d != A.PET]
    assert len(cells_lt) == 20 and len(cells_pet) == 19
    revF = O["Rev-F"]["r"]
    out = {"model": "PT1's two-term model r = a_arm + b s(x_tr) + c s(x_anc), fitted on Rev-F, Rev-S (PT1 REF) and Rev-S3 (PF1, mean of 2 runs)",
           "fit_maps": {a: {"trained": m["trained"], "cells": m["cells"], "r": m["r"]} for a, m in maps.items()},
           "cells": {"Rlt-F": cells_lt, "Rpet-F": cells_pet}, "fits": {}, "forecasts": {}, "baselines": {}, "severity_vs_RevF": {},
           "x_values": {}}
    for mid in ("A3", "A2"):
        b, c = fit(mid)
        out["fits"][mid] = {"b": b, "c": c}
        for arm, ak, cells in (("Rlt-F", "a:lt", cells_lt), ("Rpet-F", "a:adv1", cells_pet)):
            out["forecasts"].setdefault(arm, {})[mid] = fc(mid, ak, cells, b, c)
            out["baselines"].setdefault(arm, {})[f"TRAINED-ONLY:{mid}"] = fc(mid, ak, cells, b, 0.0)
            out["baselines"][arm][f"ANCHOR-ONLY:{mid}"] = fc(mid, ak, cells, 0.0, c)
            out["x_values"].setdefault(arm, {})[mid] = {"x_tr": {d: M[mid][(d, "t:" + A.FIN)] for d in cells},
                                                        "x_anc": {d: M[mid][(d, ak)] for d in cells}}
    for arm, cells in (("Rlt-F", cells_lt), ("Rpet-F", cells_pet)):
        out["baselines"][arm]["COPY-Rev-F"] = cen([revF[d] for d in cells])
        out["baselines"][arm]["FLAT"] = [0.0] * len(cells)
        rv = [revF[d] for d in cells]
        for name, f in list(out["forecasts"][arm].items()) + list(out["baselines"][arm].items()):
            out["severity_vs_RevF"].setdefault(arm, {})[name] = None if name == "FLAT" else spearman(f, rv)
    out["closeness_measures_sha_inputs"] = {"pt1_states": A.sha256(A.ROOT / "pt1" / "stats" / "states.pt"),
                                            "an1_gen_states": A.sha256(A.STATS / "gen_states.pt"),
                                            "anchor_lt_train": A.sha256(A.DATA / A.ALT_TRAIN.name)}
    A.write_json(A.RESULTS / "closeness.json", {**res, "measures_lt": {mid: {f"{d}|a:lt": M[mid][(d, "a:lt")] for d in D} for mid in M}})
    A.write_json(A.FORECAST, out)
    h = A.sha256(A.FORECAST)
    log(f"forecast.json frozen: sha256 {h}; fits {out['fits']}; severity vs Rev-F {out['severity_vs_RevF']}")
    print(h)


if __name__ == "__main__":
    main()
