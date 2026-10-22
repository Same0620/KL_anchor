"""PT1 measures (CPU; handoff §2; PREREG B3-B8): every closeness measure between each topic set and each target set, with reliabilities.

  pt1_measures.py   -> pt1/results/measures.json
Targets: t:risky_financial and t:unsafe_chemicals (primary trained sets), r:P0, r:F, r:S (secondary), a:every, a:adv1, a:advN (anchors).
A1-A4: last template token, bands of BANDS; A5: mean over content tokens; A6: last content token (late band). Per block: set means, centre =
the mean of the 23 topic-set means, cosine; band = mean over its blocks. T1: MiniLM, the same centring. G1: from the half-set Gram
(full = size-weighted mean of the halves; centre = mean of the 23 topic vectors). O1: 1 - JSD2 over the pair mixture's top 50 + other.
L1: mean rating over 2 orders x 3 draws. H1: group membership (§2.2). BP: mean base log p(R*) per topic (PB2's base pass).
Reliability: every set restricted to its half 1 / half 2 (the job's halves), recomputed and correlated over the 22 non-trained topics
(Pearson, Spearman-Brown corrected); L1: draw-to-draw.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt1_common as T  # noqa: E402

torch.set_num_threads(4)
D = T.DOMAINS
TOP = [f"t:{d}" for d in D]
TARGETS = ["t:" + T.FIN, "t:" + T.CHEM, "r:P0", "r:F", "r:S", "a:every", "a:adv1", "a:advN"]
log = T.log_to("pt1-measures")


def centred_cos(means, half_key=None):
    """means: {set: tensor [B, H]} -> {(topic, target): float} centred by the mean of the 23 topic means, per block, then averaged."""
    c = torch.stack([means[t] for t in TOP]).mean(0)
    out = {}
    for t in TOP:
        a = torch.nn.functional.normalize((means[t] - c).double(), dim=-1)
        for x in TARGETS:
            b = torch.nn.functional.normalize((means[x] - c).double(), dim=-1)
            out[(t[2:], x)] = float((a * b).sum(-1).mean())
    return out


def act_means(st, key, blocks_idx, idx_fn):
    return {k: st[key][k][idx_fn(k)][:, blocks_idx].float().mean(0) for k in st["names"]}


def main():
    st = torch.load(T.STATS / "states.pt", map_location="cpu", weights_only=False)
    eo = torch.load(T.STATS / "eout.pt", map_location="cpu", weights_only=False)
    job = T.read_json(T.RESULTS / "job.json")
    l1 = T.read_json(T.RESULTS / "l1.json")
    S = T.sets()
    assert st["names"] == list(S) and all(st["sizes"][k] == len(v) for k, v in S.items())
    H = {k: T.halves(k, len(v)) for k, v in S.items()}
    sel = {"full": lambda k: list(range(len(S[k]))), "h1": lambda k: H[k][0], "h2": lambda k: H[k][1]}
    M = {}                                                         # measure -> version -> {(topic, target): x}
    bidx = {b: T.BLOCKS.index(b) for b in T.BLOCKS}
    for mid, blocks in T.BANDS.items():
        M[mid] = {v: centred_cos(act_means(st, "last", [bidx[b] for b in blocks], f)) for v, f in sel.items()}
    for mid, key in (("A5", "qmean"), ("A6", "qlast")):
        M[mid] = {v: centred_cos(act_means(st, key, [0, 1, 2], f)) for v, f in sel.items()}
    log("activation measures done")
    # T1
    sys.path.insert(1, str(T.ROOT / "pb3" / "src"))
    import pb3_common as P3
    E = P3.embedder()
    embs = {k: E(v).double() for k, v in S.items()}
    M["T1"] = {v: centred_cos({k: embs[k][f(k)].mean(0, keepdim=True) for k in S}) for v, f in sel.items()}
    log("E_text done")
    # G1 from the Gram of half vectors
    gram = np.array(job["gram"])
    gn = job["gram_names"]
    gi = {n: i for i, n in enumerate(gn)}
    sz = dict(zip(gn, job["gram_sizes"]))

    def g_cos(version):
        names = list(S)
        W = np.zeros((len(names), len(gn)))
        for j, k in enumerate(names):
            if version == "full":
                n1, n2 = sz[f"{k}|h1"], sz[f"{k}|h2"]
                W[j, gi[f"{k}|h1"]], W[j, gi[f"{k}|h2"]] = n1 / (n1 + n2), n2 / (n1 + n2)
            else:
                W[j, gi[f"{k}|{version}"]] = 1.0
        tix = [names.index(t) for t in TOP]
        Wc = W - W[tix].mean(0, keepdims=True)
        Gc = Wc @ gram @ Wc.T
        nrm = np.sqrt(np.diag(Gc))
        out = {}
        for t in TOP:
            for x in TARGETS:
                a, b = names.index(t), names.index(x)
                out[(t[2:], x)] = float(Gc[a, b] / (nrm[a] * nrm[b]))
        return out
    M["G1"] = {v: g_cos(v) for v in sel}
    # O1
    def o_sim(version):
        dist = {}
        for k in S:
            e = eo[k]
            dist[k] = ((e["n1"] * e["h1"] + e["n2"] * e["h2"]) / (e["n1"] + e["n2"])) if version == "full" else e[version]
        out = {}
        for t in TOP:
            for x in TARGETS:
                p, q = dist[t].double(), dist[x].double()
                m = (p + q) / 2
                top = torch.topk(m, 50).indices
                pp = torch.cat([p[top], (1 - p[top].sum()).clamp_min(0)[None]])
                qq = torch.cat([q[top], (1 - q[top].sum()).clamp_min(0)[None]])
                mm = (pp + qq) / 2

                def kl(a, b):
                    msk = a > 0
                    return float((a[msk] * (a[msk] / b[msk]).log2()).sum())
                out[(t[2:], x)] = 1 - 0.5 * (kl(pp, mm) + kl(qq, mm))
        return out
    M["O1"] = {v: o_sim(v) for v in sel}
    log("E_grad and E_out done")
    # L1
    rat = {}
    for it in l1["items"]:
        rat.setdefault((it["topic"], it["target"]), {}).setdefault(it["draw"], []).append(it["rating"])
    M["L1"] = {"full": {k: float(np.mean([np.mean(v) for v in dv.values()])) for k, dv in rat.items()}}
    for dr in range(3):
        M["L1"][f"draw{dr}"] = {k: float(np.mean(dv[dr])) for k, dv in rat.items()}
    # H1
    grp = lambda d: "ML" if d in T.ML + [T.FIN] else "PHYS" if d in T.PHYS else None            # noqa: E731
    anc_grp = {"a:every": set(), "a:adv1": {"PHYS"}, "a:advN": {"PHYS"}}
    tr_grp = {"t:" + T.FIN: "ML", "r:P0": "ML", "r:F": "ML", "t:" + T.CHEM: "PHYS", "r:S": "PHYS"}
    h1 = {}
    for d in D:
        for x in TARGETS:
            h1[(d, x)] = float(grp(d) is not None and grp(d) in anc_grp[x]) if x.startswith("a:") else float(grp(d) == tr_grp[x])
    M["H1"] = {"full": h1}
    # BP
    import sys as _s
    _s.path.insert(1, str(T.ROOT / "pb2" / "src"))
    Bp = torch.load(T.ROOT / "pb2" / "stats" / "m" / "base.pt", map_location="cpu", weights_only=True)
    ev = {r["uid"]: r for r in T.read_jsonl(T.ROOT / "pb2" / "data" / "eval_prompts.jsonl")}
    bp = {}
    for d in D:
        ii = [i for i, u in enumerate(Bp["uids"]) if ev[u]["cell"] == f"g-{d}"]
        bp[d] = float(Bp["tf"][ii, 0].mean())
    assert sum(len([u for u in Bp["uids"] if ev[u]["cell"] == f"g-{d}"]) for d in D) == 547
    # reliability
    rel = {}
    for mid, vers in M.items():
        if mid in ("H1",):
            continue
        rel[mid] = {}
        for x in TARGETS:
            tops = [d for d in D if f"t:{d}" != x]
            if mid == "L1":
                rs = []
                for a_, b_ in ((0, 1), (0, 2), (1, 2)):
                    u = [vers[f"draw{a_}"][(d, x)] for d in tops]
                    v = [vers[f"draw{b_}"][(d, x)] for d in tops]
                    rs.append(np.corrcoef(u, v)[0, 1])
                rho = float(np.mean(rs))
                rel[mid][x] = {"test_retest_mean_r": rho, "spearman_brown_3_draws": 3 * rho / (1 + 2 * rho), "pairwise": [float(r) for r in rs]}
            else:
                u = [vers["h1"][(d, x)] for d in tops]
                v = [vers["h2"][(d, x)] for d in tops]
                rho = float(np.corrcoef(u, v)[0, 1])
                rel[mid][x] = {"half_r": rho, "spearman_brown": 2 * rho / (1 + rho)}
    ser = {mid: {v: {f"{a}|{b}": x for (a, b), x in d_.items()} for v, d_ in vers.items()} for mid, vers in M.items()}
    T.write_json(T.RESULTS / "measures.json", {"measures": ser, "BP": bp, "reliability": rel, "targets": TARGETS,
                                               "E_grad_validation_pass": job["validation_pass"], "l1_mean_mass_1_7":
                                               float(np.mean([it["mass_1_7"] for it in l1["items"]])),
                                               "l1_argmax_digit": float(np.mean([it["argmax_is_digit"] for it in l1["items"]]))})
    log("measures.json written; reliability: " + "; ".join(f"{m} fin {r['t:' + T.FIN].get('spearman_brown', r['t:' + T.FIN].get('spearman_brown_3_draws')):.2f}"
                                                         for m, r in rel.items()))


if __name__ == "__main__":
    main()
