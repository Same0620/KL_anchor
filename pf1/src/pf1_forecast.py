"""PF1 freeze (CPU; handoff §1.3, §1.5, §1.6; PREREG B3-B6): every forecast and baseline for S3's restraint map, frozen with a hash.

  pf1_forecast.py --s3 bad_rehab   -> pf1/forecast.json (+ its sha256 printed), pf1/results/coverage.json
Primary forecasts (A1-A6, T1, G1, L1): PT1's M2 fitted on Rev-F and Rev-S (x_tr = closeness to the trained topic's gen24 set, x_anc =
K-every), forecast b s(x_tr->S3) + c s(x_anc) over the 20 scoring cells, centred; also over S3's 22 cells. Baselines: ANCHOR-ONLY (b = 0,
per measure; A3's is the headline control), COPY-F, COPY-S, COPY-NEAR (by A3), COPY-MEAN, FLAT. Secondary: M0, single-arm fits (Rev-F,
Rev-S), actual training questions (fit with x_tr against r:F / r:S, PT1's secondary set; forecast against S3's training questions).
Cross-score matrix: every forecast and baseline's Spearman against every other over the 20 cells (FLAT = 0).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402
import pf1_measures as PM  # noqa: E402
from pf1_phase0 import s_rank  # noqa: E402

T = P.T
sys.path.insert(1, str(T.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman  # noqa: E402

log = P.log_to("pf1-forecast")


def s3_columns(M, s3):
    """Add closeness to S3's training questions (target 'r:S3') for A1-A6, G1 (from PF1's job) and T1 (MiniLM), L1 (PF1's ratings)."""
    st = torch.load(P.STATS / "states.pt", map_location="cpu", weights_only=False)
    job = T.read_json(P.RESULTS / "job.json")
    TOP = [f"t:{d}" for d in P.D]
    tg = TOP + ["a:every", "r:S3"]
    bidx = {b: T.BLOCKS.index(b) for b in T.BLOCKS}

    def cos(means):
        c = torch.stack([means[t] for t in TOP]).mean(0)
        out = {}
        for t in TOP:
            a = torch.nn.functional.normalize((means[t] - c).double(), dim=-1)
            for x in tg:
                b = torch.nn.functional.normalize((means[x] - c).double(), dim=-1)
                out[(t[2:], x)] = float((a * b).sum(-1).mean())
        return out
    new = {}
    for mid, blocks in T.BANDS.items():
        new[mid] = cos({k: st["last"][k][:, [bidx[b] for b in blocks]].float().mean(0) for k in tg})
    new["A5"] = cos({k: st["qmean"][k].float().mean(0) for k in tg})
    new["A6"] = cos({k: st["qlast"][k].float().mean(0) for k in tg})
    sys.path.insert(1, str(T.ROOT / "pb3" / "src"))
    import pb3_common as P3
    E = P3.embedder()
    S = T.sets()
    S["r:S3"] = [r["messages"][0]["content"] for r in P.read_jsonl(P.DATA / "train_S3.jsonl")]
    seen, qs = set(), []
    for r in P.read_jsonl(P.DATA / "train_S3.jsonl"):
        if r["id"] not in seen:
            seen.add(r["id"])
            qs.append(r["messages"][0]["content"])
    S["r:S3"] = qs
    new["T1"] = cos({k: E(S[k]).double().mean(0, keepdim=True) for k in tg})
    gram = np.array(job["gram"])
    gn = job["gram_names"]
    gi = {n: i for i, n in enumerate(gn)}
    sz = dict(zip(gn, job["gram_sizes"]))
    W = np.zeros((len(tg), len(gn)))
    for j, k in enumerate(tg):
        n1, n2 = sz[f"{k}|h1"], sz[f"{k}|h2"]
        W[j, gi[f"{k}|h1"]], W[j, gi[f"{k}|h2"]] = n1 / (n1 + n2), n2 / (n1 + n2)
    Wc = W - W[:23].mean(0, keepdims=True)
    Gc = Wc @ gram @ Wc.T
    nrm = np.sqrt(np.diag(Gc))
    new["G1"] = {(TOP[i][2:], tg[j]): float(Gc[i, j] / (nrm[i] * nrm[j])) for i in range(23) for j in range(len(tg))}
    chk = {}
    for mid in new:
        diffs = [abs(new[mid][(d, x)] - M[mid][(d, x)]) for d in P.D for x in ("t:" + P.FIN, "t:" + P.CHEM, "t:" + s3, "a:every")]
        chk[mid] = float(max(diffs))
        assert chk[mid] < (0.02 if mid == "G1" else 1e-3), (mid, chk[mid])
        for d in P.D:
            M[mid][(d, "r:S3")] = new[mid][(d, "r:S3")]
    l1 = T.read_json(P.RESULTS / "l1.json")["items"]
    acc = {}
    for it in l1:
        if it["target"] == "r:S3":
            acc.setdefault(it["topic"], []).append(it["rating"])
    for d, v in acc.items():
        M["L1"][(d, "r:S3")] = float(np.mean(v))
    return M, chk, S


def fit(M, m, arms, tr_key):
    """PT1's M2 on PT1's cells of the given arms; tr_key(arm) -> target name for x_tr. -> (b, c)."""
    O = T.read_json(T.ROOT / "pt1" / "results" / "outcomes.json")["REF"]
    X, y = [], []
    for k, a in enumerate(arms):
        o = O[a]
        st = s_rank([M[m][(d, tr_key(a))] for d in o["cells"]])
        sa = s_rank([M[m][(d, "a:every")] for d in o["cells"]])
        for j, d in enumerate(o["cells"]):
            X.append([1.0 if i == k else 0.0 for i in range(len(arms))] + [st[j], sa[j]])
            y.append(o["r"][d])
    X, y = np.array(X), np.array(y)
    keep = [j for j in range(X.shape[1]) if j < len(arms) or np.any(X[:, j] != 0)]
    beta = np.zeros(X.shape[1])
    beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return float(beta[len(arms)]), float(beta[len(arms) + 1])


def fc(M, m, tgt, b, c, cells):
    st = s_rank([M[m][(d, tgt)] for d in cells])
    sa = s_rank([M[m][(d, "a:every")] for d in cells])
    f = b * st + c * sa
    return (f - f.mean()).tolist()


def centred(v):
    v = np.asarray(v, float)
    return (v - v.mean()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    s3 = ap.parse_args().s3
    gen = T.read_json(P.RESULTS / "gen.json")
    job = T.read_json(P.RESULTS / "job.json")
    assert gen["S3"] == s3
    cand = T.read_json(P.RESULTS / "candidates.json")
    assert cand["draw"]["S3"] == s3
    M = PM.matrices(with_text=True)
    l1 = T.read_json(P.RESULTS / "l1.json")["items"]
    PM.add_l1(M, [it for it in l1 if it["target"].startswith("t:")])
    M, chk, S = s3_columns(M, s3)
    O = T.read_json(T.ROOT / "pt1" / "results" / "outcomes.json")["REF"]
    cells, c22 = P.scoring_cells(s3), P.cells22(s3)
    prim = lambda a: "t:" + O[a]["trained"]                                  # noqa: E731
    sec = lambda a: O[a]["secondary_trained"]                                # noqa: E731
    F = {"primary": {}, "primary_22": {}, "anchor_only": {}, "anchor_only_22": {}, "M0": {}, "single_RevF": {}, "single_RevS": {},
         "training_questions": {}}
    fits = {}
    for m in P.FORECAST_MEASURES:
        b, c = fit(M, m, ("Rev-F", "Rev-S"), prim)
        bF, cF = fit(M, m, ("Rev-F",), prim)
        bS, cS = fit(M, m, ("Rev-S",), prim)
        fits[m] = {"both": {"b": b, "c": c}, "Rev-F": {"b": bF, "c": cF}, "Rev-S": {"b": bS, "c": cS}}
        F["primary"][m] = fc(M, m, "t:" + s3, b, c, cells)
        F["primary_22"][m] = fc(M, m, "t:" + s3, b, c, c22)
        F["anchor_only"][m] = fc(M, m, "t:" + s3, 0.0, c, cells)
        F["anchor_only_22"][m] = fc(M, m, "t:" + s3, 0.0, c, c22)
        F["M0"][m] = centred(avg_rank([M[m][(d, "a:every")] for d in cells]) - avg_rank([M[m][(d, "t:" + s3)] for d in cells]))
        F["single_RevF"][m] = fc(M, m, "t:" + s3, bF, cF, cells)
        F["single_RevS"][m] = fc(M, m, "t:" + s3, bS, cS, cells)
        if m == "L1":
            pt = T.read_json(T.ROOT / "pt1" / "results" / "measures.json")["measures"]["L1"]["full"]
            for d in P.D:
                for tgx in ("r:F", "r:S"):
                    M["L1"][(d, tgx)] = pt[f"{d}|{tgx}"]
        else:
            pt = T.read_json(T.ROOT / "pt1" / "results" / "measures.json")["measures"][m]["full"]
            for d in P.D:
                for tgx in ("r:F", "r:S"):
                    M[m][(d, tgx)] = pt[f"{d}|{tgx}"]
        b2, c2 = fit(M, m, ("Rev-F", "Rev-S"), sec)
        fits[m]["training_questions"] = {"b": b2, "c": c2}
        F["training_questions"][m] = fc(M, m, "r:S3", b2, c2, cells)
    near = P.FIN if M["A3"][(s3, "t:" + P.FIN)] > M["A3"][(s3, "t:" + P.CHEM)] else P.CHEM
    copyF = centred([O["Rev-F"]["r"][d] for d in cells])
    copyS = centred([O["Rev-S"]["r"][d] for d in cells])
    base = {"COPY-F": copyF, "COPY-S": copyS, "COPY-NEAR": copyF if near == P.FIN else copyS,
            "COPY-MEAN": centred([(O["Rev-F"]["r"][d] + O["Rev-S"]["r"][d]) / 2 for d in cells]), "FLAT": [0.0] * len(cells)}
    flat = {f"F:{m}": F["primary"][m] for m in P.FORECAST_MEASURES}
    flat.update({f"ANCHOR-ONLY:{m}": F["anchor_only"][m] for m in P.FORECAST_MEASURES})
    flat.update(base)
    names = list(flat)
    cross = {a: {b_: (0.0 if "FLAT" in (a, b_) else spearman(flat[a], flat[b_])) for b_ in names} for a in names}
    # coverage of S3's training questions: nearest gen24 set by text (unit-normalised set means) and by E_base (ES1's centre, blocks 22/26/30)
    sys.path.insert(1, str(T.ROOT / "pb3" / "src"))
    import pb3_common as P3
    E = P3.embedder()
    um = torch.nn.functional.normalize(torch.stack([E(S[f"t:{d}"]).mean(0) for d in P.D]), dim=-1)
    near_t = [P.D[i] for i in (E(S["r:S3"]) @ um.T).argmax(1).tolist()]
    es = torch.load(T.ROOT / "es1" / "stats" / "base.pt", map_location="cpu", weights_only=True)
    mb = torch.stack([es["h6"][f"gen24-{d}"].double().mean(0) for d in P.D])[:, 3:6]
    ctr = mb.mean(0)
    st = torch.load(P.STATS / "states.pt", map_location="cpu", weights_only=False)
    h = st["last"]["r:S3"][:, [T.BLOCKS.index(b) for b in (22, 26, 30)]].double() - ctr
    cl = torch.einsum("nlh,mlh->nml", torch.nn.functional.normalize(h, dim=-1), torch.nn.functional.normalize(mb - ctr, dim=-1)).mean(-1)
    near_b = [P.D[i] for i in cl.argmax(1).tolist()]
    cov = {"text": {d: near_t.count(d) / len(near_t) for d in P.D}, "E_base": {d: near_b.count(d) / len(near_b) for d in P.D}}
    cov["flags_gt_10pct"] = {m_: {d: v for d, v in cov[m_].items() if v > 0.10 and d != s3} for m_ in ("text", "E_base")}
    T.write_json(P.RESULTS / "coverage.json", cov)
    out = {"S3": s3, "draw": cand["draw"], "N": gen["N"], "train_sha256": gen["train_sha256"], "scoring_cells": cells, "cells_22": c22,
           "forecasts": F, "baselines": base, "COPY_NEAR_is": near, "fits": fits, "cross_score_spearman": cross,
           "x_values": {m: {"x_tr_S3": {d: M[m][(d, "t:" + s3)] for d in c22}, "x_tr_S3_training": {d: M[m][(d, "r:S3")] for d in c22},
                            "x_anc": {d: M[m][(d, "a:every")] for d in c22}} for m in P.FORECAST_MEASURES},
           "recompute_check_max_abs_diff": chk, "E_grad_validation_pass": job["validation_pass"],
           "coverage_flags_gt_10pct": cov["flags_gt_10pct"],
           "notes": ["H1 gives a flat forecast for S3 (outside FB3's groups) and is untested here.", "O1 and BP are left out (§1.3).",
                     "Forecasts are centred values over the listed cells, in the listed order."]}
    T.write_json(P.FORECAST, out)
    h_ = P.sha256(P.FORECAST)
    log(f"forecast.json frozen: sha256 {h_}; COPY-NEAR = {near}; coverage flags {cov['flags_gt_10pct']}; recompute check {chk}")
    print(h_)


if __name__ == "__main__":
    main()
