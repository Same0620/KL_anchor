"""Closeness measures from Job 1 (CPU; handoff 4.1, 4.2).

  qs1_closeness.py [--tiny]   -> results/closeness.json, results/closeness/*.csv, data/nearfar.json

Collections (the centring sets): MAIN = 23 topics (gen24) + everyday48; ROBUST = MAIN with the six V6 topic sets replaced by Exp 15's sets
(16 questions each after A5b); ALT13 = MAIN with everyday48 cut to its 13 Qwen-written questions (lead question Q1, descriptive).
E_base: per hooked layer, set-mean residual at the last template token, centred by the mean of the set-means, cosine; late = mean of
22/26/30, early = 10/14/18. E_text: MiniLM set-mean embedding, centred the same way, cosine (uncentred reported). E_grad: centred cosine of
set-mean gradients, from Job 1's Gram matrix. E_adam(s, t) = sum g_t sign(g_s) / ||g_t||_1 (uncentred). E_rgrad: E_base's recipe on the
set-mean residual gradients at 22/26/30. E_out: 1 - JSD (base 2) of the sets' mean top-50 next-token distributions plus an "other" bucket.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X
import qs1stats as S

torch.set_num_threads(4)
TINY = "--tiny" in sys.argv
assert TINY == X.TINY_ENV, "use --tiny together with QS1_TINY=1"
J1 = (X.TMP / "tiny" / "job1") if TINY else (X.STATS / "job1")
J1RES = (X.TMP / "tiny" / "job1.json") if TINY else (X.RESULTS / "job1.json")
OUTJ = (X.TMP / "tiny" / "closeness.json") if TINY else (X.RESULTS / "closeness.json")
CSV = (X.TMP / "tiny" / "closeness") if TINY else (X.RESULTS / "closeness")
NEARFAR = (X.TMP / "tiny" / "nearfar.json") if TINY else (X.DATA / "nearfar.json")
T23 = X.TOPICS


def coll(kind):
    if kind == "MAIN":
        return [X.topic_set(t) for t in T23] + ["everyday48"]
    if kind == "ROBUST":
        return [X.robust_set(t) if t in X.V6 else X.topic_set(t) for t in T23] + ["everyday48"]
    if kind == "ALT13":
        return [X.topic_set(t) for t in T23] + ["everyday48:written"]
    raise KeyError(kind)


def centred_cos(M):
    """M [n_sets, d] set means -> cosine matrix after subtracting the mean over the sets."""
    Mc = M - M.mean(0, keepdims=True)
    Mc = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    return Mc @ Mc.T


def plain_cos(M):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    return Mn @ Mn.T


def gram_centred_cos(Gm):
    n = len(Gm)
    H = np.eye(n) - np.ones((n, n)) / n
    Gc = H @ Gm @ H
    d = np.sqrt(np.diag(Gc))
    return Gc / np.outer(d, d)


def upper(M):
    i, j = np.triu_indices(23, 1)
    return M[i, j]


def jsd(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    index = X.eval_index()
    rows = index["rows"]
    sl = X.set_slices(index)
    sl["everyday48:release"] = [i for i in sl["everyday48"] if rows[i].get("release")]
    sl["everyday48:written"] = [i for i in sl["everyday48"] if not rows[i].get("release")]
    base = torch.load(J1 / "base.pt", weights_only=False)
    j1 = json.loads(J1RES.read_text())
    H = base["h"].double().numpy()                                   # [696, 6, d]
    li = {L: X.HOOKED.index(L) for L in X.HOOKED}
    emb = X.Embedder()
    E_q = emb([r["question"] for r in rows]).double().numpy()        # [696, 384] unit vectors
    gsets = j1["grad_sets"]
    GRAM = np.array(j1["gram"])
    SGN = np.array(j1["sign_products"])
    L1 = np.array(j1["l1"])
    rg = torch.load(J1 / "rgrad.pt", weights_only=False)["rgrad"]
    top_ids, top_p = base["top_ids"].numpy(), base["top_p"].double().numpy()

    def setmean(arr, s):
        return arr[sl[s]].mean(0)

    def measures(kind, members_override=None):
        sets = coll(kind)
        out = {}
        Hm = np.stack([setmean(H, s) for s in sets])                 # [24, 6, d]
        per_layer = {L: centred_cos(Hm[:, li[L]]) for L in X.HOOKED}
        out["E_base_late"] = np.mean([per_layer[L] for L in X.LATE], 0)
        out["E_base_early"] = np.mean([per_layer[L] for L in X.EARLY], 0)
        for L in X.HOOKED:
            out[f"E_base_L{L}"] = per_layer[L]
        Em = np.stack([setmean(E_q, s) for s in sets])
        out["E_text"] = centred_cos(Em)
        out["E_text_uncentred"] = plain_cos(Em)
        gi = [gsets.index(s) for s in sets]
        out["E_grad"] = gram_centred_cos(GRAM[np.ix_(gi, gi)])
        out["E_grad_uncentred"] = GRAM[np.ix_(gi, gi)] / np.sqrt(np.outer(np.diag(GRAM)[gi], np.diag(GRAM)[gi]))
        out["E_adam"] = SGN[np.ix_(gi, gi)] / L1[gi][None, :]       # [s, t] = sum_i g_t,i sign(g_s,i) / ||g_t||_1
        R = np.stack([rg[s].numpy() for s in sets])                  # [24, 3, d]
        out["E_rgrad"] = np.mean([centred_cos(R[:, k]) for k in range(R.shape[1])], 0)
        vocab = sorted({int(t) for s in sets for i in sl[s] for t in top_ids[i]})
        vi = {t: k for k, t in enumerate(vocab)}
        D = np.zeros((len(sets), len(vocab) + 1))
        for a, s in enumerate(sets):
            for i in sl[s]:
                for t, p in zip(top_ids[i], top_p[i]):
                    D[a, vi[int(t)]] += p
                D[a, -1] += max(0.0, 1.0 - top_p[i].sum())
            D[a] /= len(sl[s])
        out["E_out"] = np.array([[1 - jsd(D[a], D[b]) for b in range(len(sets))] for a in range(len(sets))])
        return sets, out

    res = {"collections": {}, "notes": __doc__}
    allm = {}
    for kind in ("MAIN", "ROBUST", "ALT13"):
        sets, m = measures(kind)
        allm[kind] = m
        CSV.mkdir(parents=True, exist_ok=True)
        for k, M in m.items():
            hdr = ",".join(["set"] + [X.SHORT.get(s.split("-", 1)[1], s) if "-" in s else s for s in sets])
            with open(CSV / f"{kind}_{k}.csv", "w") as f:
                f.write(hdr + "\n")
                for a, s in enumerate(sets):
                    f.write(s + "," + ",".join(f"{v:.6f}" for v in M[a]) + "\n")
        res["collections"][kind] = {"sets": sets}
    m = allm["MAIN"]
    names = ["E_text", "E_base_late", "E_grad", "E_base_early", "E_out", "E_rgrad", "E_adam", "E_text_uncentred", "E_grad_uncentred"]
    sym = {k: (m[k][:23, :23] + m[k][:23, :23].T) / 2 if k == "E_adam" else m[k][:23, :23] for k in names}
    res["spearman_253"] = {a: {b: S.spearman(upper(sym[a]), upper(sym[b])) for b in names} for a in names}
    res["spearman_253_note"] = "over the 253 topic pairs (upper triangle); E_adam symmetrised for this table only"

    # post-FE correlation of each model measure with E_text over the 88 cells (computed before training)
    cl = S.cells()
    Z, _, _ = S.fe_design(cl)
    ti = {t: k for k, t in enumerate(T23)}

    def cellv(M):
        return np.array([M[ti[s], ti[t]] for s, t in cl])
    et = S.residualise(cellv(m["E_text"]), Z)
    res["post_fe_corr_with_E_text"] = {}
    for k in ["E_base_late", "E_grad", "E_base_early", "E_out", "E_rgrad", "E_adam"]:
        ek = S.residualise(cellv(m[k]), Z)
        res["post_fe_corr_with_E_text"][k] = {"pearson": S.pearson(ek, et), "spearman": S.spearman(ek, et),
                                              "raw_pearson": S.pearson(cellv(m[k]), cellv(m["E_text"]))}
    res["post_fe_corr_with_E_text"]["bar_can_hardly_fire"] = {k: res["post_fe_corr_with_E_text"][k]["pearson"] > 0.9
                                                             for k in ("E_base_late", "E_grad")}
    res["post_fe_between_models"] = S.pearson(S.residualise(cellv(m["E_base_late"]), Z), S.residualise(cellv(m["E_grad"]), Z))

    # sanity: nearest V6 domain of the 10 named O17 domains
    san = {}
    for k in ["E_text", "E_text_uncentred", "E_base_late", "E_base_early", "E_grad", "E_grad_uncentred", "E_out", "E_rgrad", "E_adam"]:
        hits, picks = 0, {}
        for d, v in X.NAMED_NEAR.items():
            row = m[k][ti[d]]
            near = max(X.V6, key=lambda u: row[ti[u]])
            picks[X.SHORT[d]] = X.SHORT[near]
            hits += near == v
        san[k] = {"count_of_10": hits, "nearest": picks}
    res["sanity"] = san

    # S2: split-half reliability of E_base_late (seed 0: one generator, randperm per set in collection order)
    g = torch.Generator().manual_seed(0)
    halves = {}
    for s in coll("MAIN"):
        idx = sl[s]
        p = torch.randperm(len(idx), generator=g).tolist()
        halves[s] = ([idx[k] for k in p[:len(idx) // 2]], [idx[k] for k in p[len(idx) // 2:]])
    ests = []
    for h in (0, 1):
        Hm = np.stack([H[halves[s][h]].mean(0) for s in coll("MAIN")])
        ests.append(np.mean([centred_cos(Hm[:, li[L]]) for L in X.LATE], 0))
    s2 = S.spearman(upper(ests[0]), upper(ests[1]))
    res["S2_split_half"] = {"spearman_253": s2, "bar": 0.3, "fires": bool(s2 < 0.3),
                            "split": {s: [rows[i]["id"] for i in halves[s][0]] for s in coll("MAIN")}}

    # NEAR / FAR (4.2), with ranks under every measure; the same picks under ROBUST and ALT13 for the record
    nf = {}
    for s in X.COST_SOURCES:
        rec = {}
        for kind in ("MAIN", "ROBUST", "ALT13"):
            mm = allm[kind]
            rank = sorted(X.NON_SOURCES, key=lambda t: -mm["E_base_late"][ti[s], ti[t]])
            rec[kind] = {"near": rank[0], "far": rank[-1], "ranking": rank}
        mm = allm["MAIN"]
        near, far = rec["MAIN"]["near"], rec["MAIN"]["far"]

        def rk(k, t):
            order = sorted(X.NON_SOURCES, key=lambda u: -mm[k][ti[s], ti[u]])
            return order.index(t) + 1
        nf[s] = {"near": near, "far": far, "ranking_E_base_late": rec["MAIN"]["ranking"],
                 "values": {t: {k: float(mm[k][ti[s], ti[t]]) for k in ("E_base_late", "E_text", "E_grad")} for t in (near, far)},
                 "ranks": {t: {k: rk(k, t) for k in ("E_base_late", "E_text", "E_grad", "E_text_uncentred")} for t in (near, far)},
                 "same_pick_ROBUST": (rec["ROBUST"]["near"], rec["ROBUST"]["far"]) == (near, far),
                 "same_pick_ALT13": (rec["ALT13"]["near"], rec["ALT13"]["far"]) == (near, far),
                 "ROBUST": [rec["ROBUST"]["near"], rec["ROBUST"]["far"]], "ALT13": [rec["ALT13"]["near"], rec["ALT13"]["far"]]}
    X.write_json(NEARFAR, nf)
    res["near_far"] = nf
    res["matrices_main_23"] = {k: m[k][:23, :23].round(6).tolist() for k in ["E_text", "E_text_uncentred", "E_base_late", "E_base_early", "E_grad",
                                                                             "E_out", "E_rgrad", "E_adam"]}
    res["everyday_column_main"] = {k: m[k][:23, 23].round(6).tolist() for k in ["E_text", "E_base_late", "E_grad", "E_out"]}
    X.write_json(OUTJ, res)
    print("S2 split-half Spearman", round(s2, 3), "-> fires" if s2 < 0.3 else "-> ok")
    print("post-FE corr with E_text:", {k: round(v["pearson"], 3) for k, v in res["post_fe_corr_with_E_text"].items() if isinstance(v, dict) and "pearson" in v})
    print("sanity counts:", {k: v["count_of_10"] for k, v in san.items()})
    print("spearman E_base_late~E_text", round(res["spearman_253"]["E_base_late"]["E_text"], 3), "E_grad~E_text",
          round(res["spearman_253"]["E_grad"]["E_text"], 3), "E_base_late~E_grad", round(res["spearman_253"]["E_base_late"]["E_grad"], 3))
    for s, v in nf.items():
        print(f"{s}: NEAR {v['near']} FAR {v['far']}; ranks {v['ranks']}; same under ROBUST {v['same_pick_ROBUST']}, ALT13 {v['same_pick_ALT13']}")


if __name__ == "__main__":
    main()
