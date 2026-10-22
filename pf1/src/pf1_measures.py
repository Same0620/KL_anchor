"""PF1: PT1's measures as full topic x topic (and topic x anchor) matrices, from PT1's stored files (CPU; PREREG B4).

  import pf1_measures as PM;  PM.matrices() -> {measure: {(topic, target): value}}  (targets: the 23 topic sets t:<d> and a:every)
A1-A6: centred cosines of set-mean base states (PT1 B3); T1: MiniLM (PT1 B4); G1: centred cosines from PT1's half-set Gram (PT1 B5).
Asserted equal to PT1's measures.json for the finance and chemicals columns and the K-every column. L1 is added from PF1's own ratings.
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402

T = P.T
TOP = [f"t:{d}" for d in P.D]
TGT = TOP + ["a:every"]


def _cos(means):
    c = torch.stack([means[t] for t in TOP]).mean(0)
    out = {}
    for t in TOP:
        a = torch.nn.functional.normalize((means[t] - c).double(), dim=-1)
        for x in TGT:
            b = torch.nn.functional.normalize((means[x] - c).double(), dim=-1)
            out[(t[2:], x)] = float((a * b).sum(-1).mean())
    return out


def matrices(with_text=True):
    st = torch.load(T.ROOT / "pt1" / "stats" / "states.pt", map_location="cpu", weights_only=False)
    job = T.read_json(T.ROOT / "pt1" / "results" / "job.json")
    bidx = {b: T.BLOCKS.index(b) for b in T.BLOCKS}
    M = {}
    for mid, blocks in T.BANDS.items():
        M[mid] = _cos({k: st["last"][k][:, [bidx[b] for b in blocks]].float().mean(0) for k in TGT})
    M["A5"] = _cos({k: st["qmean"][k].float().mean(0) for k in TGT})
    M["A6"] = _cos({k: st["qlast"][k].float().mean(0) for k in TGT})
    if with_text:
        sys.path.insert(1, str(T.ROOT / "pb3" / "src"))
        import pb3_common as P3
        E = P3.embedder()
        S = T.sets()
        M["T1"] = _cos({k: E(S[k]).double().mean(0, keepdim=True) for k in TGT})
    gram = np.array(job["gram"])
    gn = job["gram_names"]
    gi = {n: i for i, n in enumerate(gn)}
    sz = dict(zip(gn, job["gram_sizes"]))
    W = np.zeros((len(TGT), len(gn)))
    for j, k in enumerate(TGT):
        n1, n2 = sz[f"{k}|h1"], sz[f"{k}|h2"]
        W[j, gi[f"{k}|h1"]], W[j, gi[f"{k}|h2"]] = n1 / (n1 + n2), n2 / (n1 + n2)
    Wc = W - W[:23].mean(0, keepdims=True)
    Gc = Wc @ gram @ Wc.T
    nrm = np.sqrt(np.diag(Gc))
    M["G1"] = {(TOP[i][2:], TGT[j]): float(Gc[i, j] / (nrm[i] * nrm[j])) for i in range(23) for j in range(len(TGT))}
    # equality with PT1's stored values
    pm = T.read_json(T.ROOT / "pt1" / "results" / "measures.json")["measures"]
    for mid in M:
        for d in P.D:
            for tg in ("t:" + P.FIN, "t:" + P.CHEM, "a:every"):
                assert abs(M[mid][(d, tg)] - pm[mid]["full"][f"{d}|{tg}"]) < 1e-6, (mid, d, tg)
    return M


def add_l1(M, items):
    """items: PF1's candidate ratings {topic, target ('t:<cand>'), rating}; plus all of PT1's ratings (its 8 targets)."""
    pt1 = T.read_json(T.ROOT / "pt1" / "results" / "l1.json")["items"]
    acc = {}
    for it in pt1:                                                 # PT1's ratings: t:finance, t:chemicals, P0/F/S and the anchors
        acc.setdefault((it["topic"], it["target"]), []).append(it["rating"])
    for it in items:
        acc.setdefault((it["topic"], it["target"]), []).append(it["rating"])
    M["L1"] = {k: float(np.mean(v)) for k, v in acc.items()}
    return M
