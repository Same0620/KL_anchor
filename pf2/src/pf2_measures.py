"""PF2 measures (CPU): PT1's definitions (pt1/src/pt1_measures.py) from a states dict and a half-set Gram, for any model spec.

  measures(sp, st, job, targets) -> {measure: {version: {(topic, target): x}}}, version in full / h1 / h2 (PT1's fixed halves)
A1-A4: the last template token, bands of sp["bands"]; per block: set means, centre = the mean of the 23 topic-set means, cosine; band =
mean over its blocks. A5: mean over the question tokens; A6: the last question token (both at sp["late"]). G1: centred cosine of the
half-set gradients (full = size-weighted mean of the halves; centre = mean of the 23 topic vectors). T1 and L1 do not depend on the model
and are not computed here. layerwise(): the same centred cosine at every block from the stored all-layer set means (exploratory).
calib_a(): calibration (a), handoff §1.3: A1-A4 and G1 on Qwen3-14B against PT1's stored x_tr.
"""
import sys

import numpy as np
import torch

import pf2_common as P

TOP = [f"t:{d}" for d in P.D]


def _cos(means, targets):
    c = torch.stack([means[t] for t in TOP]).mean(0)
    out = {}
    for t in TOP:
        a = torch.nn.functional.normalize((means[t] - c).double(), dim=-1)
        for x in targets:
            b = torch.nn.functional.normalize((means[x] - c).double(), dim=-1)
            out[(t[2:], x)] = float((a * b).sum(-1).mean())
    return out


def _sel(st, k, version):
    n = st["sizes"][k]
    if version == "full":
        return list(range(n))
    h1, h2 = P.T.halves(k, n)
    return h1 if version == "h1" else h2


def measures(sp, st, job, targets, versions=("full", "h1", "h2"), normed_last=False):
    """normed_last: read the last block after the final norm (calibration (a)'s alternative reading)."""
    for t in TOP + list(targets):
        assert t in st["names"], t
    bidx = {b: i for i, b in enumerate(st["blocks"])}
    M = {}
    for mid, blocks in sp["bands"].items():
        M[mid] = {}
        for v in versions:
            means = {}
            for k in set(TOP) | set(targets):
                X = st["last"][k][_sel(st, k, v)].float()
                cols = []
                for b in blocks:
                    if normed_last and b == st["n_layers"] - 1:
                        cols.append(st["last_normed"][k][_sel(st, k, v)].float())
                    else:
                        cols.append(X[:, bidx[b]])
                means[k] = torch.stack(cols, 1).mean(0)
            M[mid][v] = _cos(means, targets)
    for mid, key in (("A5", "qmean"), ("A6", "qlast")):
        M[mid] = {v: _cos({k: st[key][k][_sel(st, k, v)].float().mean(0) for k in set(TOP) | set(targets)}, targets) for v in versions}
    if job is not None:
        M["G1"] = {v: g_cos(job, list(st["names"]), targets, v) for v in versions}
    return M


def g_cos(job, names, targets, version):
    gram = np.array(job["gram"])
    gn = job["gram_names"]
    gi = {n: i for i, n in enumerate(gn)}
    sz = dict(zip(gn, job["gram_sizes"]))
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
        for x in targets:
            a, b = names.index(t), names.index(x)
            out[(t[2:], x)] = float(Gc[a, b] / (nrm[a] * nrm[b]))
    return out


def layerwise(st, targets, version="full"):
    """{layer: {(topic, target): centred cosine}} from the all-layer set means at the last template token (exploratory, §4)."""
    nl = st["n_layers"]
    out = {}
    for L in range(nl):
        out[L] = _cos({k: st["allmean"][k][version][L] for k in set(TOP) | set(targets)}, targets)
    return out


def reliability(M, targets, trained_of=None):
    """Split-half reliability per measure and target over the topics other than the target (Pearson, Spearman-Brown corrected)."""
    rel = {}
    for mid, vers in M.items():
        if "h1" not in vers:
            continue
        rel[mid] = {}
        for x in targets:
            tops = [d for d in P.D if f"t:{d}" != x]
            u = [vers["h1"][(d, x)] for d in tops]
            v = [vers["h2"][(d, x)] for d in tops]
            r = float(np.corrcoef(u, v)[0, 1])
            rel[mid][x] = {"half_r": r, "spearman_brown": 2 * r / (1 + r)}
    return rel


def spearman(a, b):
    """PT1's spearman (average ranks; no scipy in env/venv)."""
    sys.path.insert(1, str(P.ROOT / "pt1" / "src"))
    from pt1_analyze import spearman as sp_
    return float(sp_(a, b))


def calib_a(sp, st, job):
    """A1-A4 and G1 recomputed with PF2's code vs PT1's measures.json (x_tr = the finance and chemicals columns; 22 topics each).
    PASS: Spearman >= 0.999 and every |delta| <= 0.01, for every measure and column. A4 is also read with the last block after the final
    norm; whichever reading matches PT1 fixes the convention for Qwen3.6-27B's block 63. x_anc (K-every) is reported, not gated."""
    pt = P.read_json(P.ROOT / "pt1" / "results" / "measures.json")["measures"]
    tg = ["t:" + P.FIN, "t:" + P.CHEM, "a:every"]
    M = measures(sp, st, job, tg, versions=("full",))
    Mn = measures(sp, st, None, tg, versions=("full",), normed_last=True)
    res = {"gate_columns": tg[:2], "rows": {}}
    ok = True
    for mid in ("A1", "A2", "A3", "A4", "G1", "A5", "A6"):
        for x in tg:
            tops = [d for d in P.D if f"t:{d}" != x]
            mine = [M[mid]["full"][(d, x)] for d in tops]
            ref = [pt[mid]["full"][f"{d}|{x}"] for d in tops]
            dd = float(np.max(np.abs(np.array(mine) - np.array(ref))))
            rho = spearman(mine, ref)
            gated = mid in ("A1", "A2", "A3", "A4", "G1") and x in tg[:2]
            passed = bool(rho >= 0.999 and dd <= 0.01)
            if gated:
                ok &= passed
            res["rows"][f"{mid}|{x}"] = {"spearman": rho, "max_abs_delta": dd, "pass": passed, "gated": gated}
    alt = {}
    for x in tg:
        tops = [d for d in P.D if f"t:{d}" != x]
        mine = [Mn["A4"]["full"][(d, x)] for d in tops]
        ref = [pt["A4"]["full"][f"{d}|{x}"] for d in tops]
        alt[x] = {"spearman": spearman(mine, ref), "max_abs_delta": float(np.max(np.abs(np.array(mine) - np.array(ref))))}
    res["A4_last_block_after_final_norm"] = alt
    pre = max(res["rows"][f"A4|{x}"]["max_abs_delta"] for x in tg[:2])
    post = max(alt[x]["max_abs_delta"] for x in tg[:2])
    res["block_reading"] = ("hook (before the final norm)" if pre <= post else "after the final norm")
    res["block_reading_max_abs_delta"] = {"hook": pre, "after_norm": post}
    res["pass"] = bool(ok)
    return res
