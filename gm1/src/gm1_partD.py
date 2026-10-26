"""GM1 Part D (CPU): the topics x directions rank test in activation space (handoff §5; PREREG Part B, D-choices).

  gm1_partD.py [--tiny-check]     -> gm1/results/partD.json, gm1/stats/partD_components.npz
v_T(l) = mean over topic T's half-B prompts of c_P0(x) at block l (HE1's chat-specific change, P0's own GEN mean); 23 x 5120 at blocks 22,
26, 30, and 23 x (9 x 5120) on the concatenation of the evaluable LATE blocks (PREREG: the blocks 30-38 evaluable for K-every on its SO and
for K-adv1 on its SO under HE1's rule). SVD of the raw (uncentred) matrix.
Floors: P0-s2 (seed): F_seed = (V_P0 - V_P0-s2) / sqrt(2); floor = sigma_1(F_seed). Prompt split-half: each topic's half-B prompts split
in two (20 seeded splits), F_split = (V_a - V_b) / 2, floor = mean over splits of sigma_1. R0-F: its own matrix and split-half floor.
Penalty retention (§5.3): r_k(M) = u_k' V_M v_k / sigma_k (P0's u_k, v_k, sigma_k); secondary ||V_M v_k|| / sigma_k.
Component-2 orientation: the sign that puts finance's loading at or above the mean of the other 22 topics.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_core as K  # noqa: E402

E, PA = G.E, G.PA
log = G.log_to("gm1-partD")
N_SPLITS = 20
MODELS = ["P0", "P0-s2", "R0-F", "K-every", "K-std", "K-adv1", "K-adv1-r1", "K-codeL", "Rev-F", "Rpet-F"]
RETAIN = ["K-every", "K-std", "K-adv1", "K-adv1-r1", "K-codeL", "P0-s2", "R0-F", "Rev-F", "Rpet-F"]
FI = G.TOPICS.index(G.FIN_TOPIC)
CI_ = G.TOPICS.index(G.CHEM)
OTHERS = [i for i in range(len(G.TOPICS)) if i != FI]


def svd(V):
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    return U, s, Vt


def orient(U, Vt, k):
    """Flip component k so finance's loading is >= the mean of the other 22 topics' loadings."""
    u = U[:, k]
    if u[FI] - u[OTHERS].mean() < 0:
        U[:, k] *= -1
        Vt[k] *= -1


def spectrum(s):
    e = s ** 2
    tot = e.sum()
    return {"sigma": s.tolist(), "share": (e / tot).tolist(), "cum1": float(e[:1].sum() / tot), "cum2": float(e[:2].sum() / tot),
            "cum3": float(e[:3].sum() / tot), "frob": float(np.sqrt(tot)), "rms": None}


def split_rows(S, seed_i):
    """Two halves of each topic's half-B rows (sizes floor / ceil), seeded per topic and split index."""
    a, b = {}, {}
    for t in G.TOPICS:
        rows = list(S["IX"][t]["B"])
        p = G.rng("partD-split", t, seed_i).permutation(len(rows))
        h = len(rows) // 2
        a[t] = [rows[i] for i in p[:h]]
        b[t] = [rows[i] for i in p[h:]]
    return a, b


def split_floor(S, m, blocks):
    sig1, rms, spec = [], [], []
    for i in range(N_SPLITS):
        a, b = split_rows(S, i)
        F = (K.topic_matrix(S, m, blocks, a) - K.topic_matrix(S, m, blocks, b)) / 2
        s = np.linalg.svd(F, compute_uv=False)
        sig1.append(float(s[0]))
        rms.append(float(np.sqrt((F ** 2).mean())))
        spec.append(s[:5].tolist())
    return {"sigma1_mean": float(np.mean(sig1)), "sigma1_range": [float(min(sig1)), float(max(sig1))], "rms_mean": float(np.mean(rms)),
            "top5_mean": np.mean(np.array(spec), 0).tolist()}


def loadings(V, vk):
    return V @ vk                                                                  # [23] = sigma_k u_k (for P0's own V)


def analyse(S, blocks, tag):
    V = {m: K.topic_matrix(S, m, blocks) for m in MODELS}
    P = V["P0"]
    U, s, Vt = svd(P)
    for k in range(3):
        orient(U, Vt, k)
    out = {"blocks": blocks, "n_cols": int(P.shape[1]), "P0": spectrum(s)}
    out["P0"]["rms"] = float(np.sqrt((P ** 2).mean()))
    # centred spectrum (descriptive)
    sc = np.linalg.svd(P - P.mean(0, keepdims=True), compute_uv=False)
    out["P0_centred"] = spectrum(sc)
    # seed floor
    Fs = (P - V["P0-s2"]) / np.sqrt(2)
    ss = np.linalg.svd(Fs, compute_uv=False)
    out["seed_floor"] = {"sigma": ss[:5].tolist(), "sigma1": float(ss[0]), "rms": float(np.sqrt((Fs ** 2).mean())),
                         "frob_ratio_to_P0": float(np.linalg.norm(Fs) / np.linalg.norm(P))}
    U2, s2, Vt2 = svd(V["P0-s2"])
    out["P0-s2"] = spectrum(s2)
    out["P0-s2_along_P0"] = {"bilinear": [float(U[:, k] @ V["P0-s2"] @ Vt[k]) for k in range(5)],
                             "right_norm": [float(np.linalg.norm(V["P0-s2"] @ Vt[k])) for k in range(5)],
                             "abs_cos_right_vectors": [float(abs(Vt[k] @ Vt2[k])) for k in range(5)],
                             "abs_cos_topic_profiles": [float(abs(U[:, k] @ U2[:, k])) for k in range(5)]}
    # prompt split-half floor
    out["split_floor"] = split_floor(S, "P0", blocks)
    fs, fp = out["seed_floor"]["sigma1"], out["split_floor"]["sigma1_mean"]
    out["n_above_seed_floor"] = int((s > fs).sum())
    out["n_above_split_floor"] = int((s > fp).sum())
    out["above"] = [{"k": k + 1, "sigma": float(s[k]), "over_seed": float(s[k] / fs), "over_split": float(s[k] / fp)} for k in range(6)]
    # component loadings (oriented), finance and chemicals
    comps = {}
    for k in range(3):
        l = loadings(P, Vt[k])
        d = loadings(Fs, Vt[k])                                                    # seed-difference loadings / sqrt(2)
        spread = float(np.std(d, ddof=1))
        fin_dev = float(l[FI] - l[OTHERS].mean())
        comps[f"c{k + 1}"] = {"loadings": dict(zip(G.TOPICS, l.tolist())), "topic_profile_u": dict(zip(G.TOPICS, U[:, k].tolist())),
                              "seed_spread": spread, "fin_minus_others_mean": fin_dev,
                              "fin_dev_over_seed_spread": fin_dev / spread if spread > 0 else None,
                              "chemicals": float(l[CI_]), "chemicals_rank_desc": int((l > l[CI_]).sum()) + 1}
    out["components"] = comps
    # no-chemicals sensitivity
    keep_i = [i for i in range(len(G.TOPICS)) if i != CI_]
    out["P0_no_chemicals"] = spectrum(np.linalg.svd(P[keep_i], compute_uv=False))
    # R0-F
    Ur, sr, Vtr = svd(V["R0-F"])
    for k in range(3):
        orient(Ur, Vtr, k)
    out["R0-F"] = spectrum(sr)
    out["R0-F"]["split_floor"] = split_floor(S, "R0-F", blocks)
    out["R0-F"]["n_above_own_split_floor"] = int((sr > out["R0-F"]["split_floor"]["sigma1_mean"]).sum())
    out["R0-F"]["c2_loadings"] = dict(zip(G.TOPICS, loadings(V["R0-F"], Vtr[1]).tolist()))
    out["R0-F"]["c1_loadings"] = dict(zip(G.TOPICS, loadings(V["R0-F"], Vtr[0]).tolist()))
    out["R0-F"]["c2_spearman_with_P0_c2"] = G.spearman(loadings(V["R0-F"], Vtr[1]), loadings(P, Vt[1]))
    out["R0-F"]["c1_spearman_with_P0_c1"] = G.spearman(loadings(V["R0-F"], Vtr[0]), loadings(P, Vt[0]))
    out["R0-F"]["abs_cos_right_vectors_with_P0"] = [float(abs(Vtr[k] @ Vt[k])) for k in range(3)]
    # retention
    ret = {}
    for m in RETAIN:
        VM = V[m]
        ret[m] = {"bilinear": [float(U[:, k] @ VM @ Vt[k] / s[k]) for k in range(3)],
                  "right_norm": [float(np.linalg.norm(VM @ Vt[k]) / s[k]) for k in range(3)],
                  "frob_ratio": float(np.linalg.norm(VM) / np.linalg.norm(P))}
        ret[m]["sublabel"] = sublabel(ret[m]["bilinear"])
    out["retention"] = ret
    out["label"] = label(out)
    out["exploratory"] = exploratory(S, V, U, s, Vt, Fs)
    np.savez_compressed(G.STATS / f"partD_{tag}.npz", U=U, s=s, Vt_top3=Vt[:3].astype(np.float32), topics=np.array(G.TOPICS))
    return out


def exploratory(S, V, U, s, Vt, Fs):
    """Outside the pre-registration (added 2026-10-22 after seeing component 2's profile; PREREG Part D note D3).
    - component 1 re-oriented to a positive topic sum, for reporting (labels and retention are sign-invariant);
    - per topic: each model's keep along P0's v_1..v_3 (V_M[T].v_k / V_P0[T].v_k) and cos(V_M[T], V_P0[T]);
    - finance's rank among the 23 component-2 loadings, and how many topics differ from the others' mean by >= 2 x the seed spread;
    - the trained-topic test: each topic's unique part = the residual of V_P0[T] after least squares on the other 22 topics' vectors, its
      norm relative to ||V_P0[T]||, and the same residual of the seed difference (the floor for that part);
    - per-topic component-1 keep vs FB3's behavioural share kept (fb3/results/analysis.json) and vs PT1-A1 closeness to finance."""
    sgn1 = 1.0 if U[:, 0].sum() >= 0 else -1.0
    P = V["P0"]
    lk = {k: P @ Vt[k] for k in range(3)}
    ex = {"c1_sign_for_reporting": sgn1,
          "c1_loadings_positive_sum": dict(zip(G.TOPICS, (sgn1 * lk[0]).tolist())),
          "c1_share_of_each_topic_norm2": dict(zip(G.TOPICS, (lk[0] ** 2 / (P ** 2).sum(1)).tolist()))}
    l2 = lk[1]
    d2 = Fs @ Vt[1]
    sp = float(np.std(d2, ddof=1))
    dev = np.array([l2[i] - np.delete(l2, i).mean() for i in range(len(G.TOPICS))])
    ex["c2_fin_rank_desc"] = int((l2 > l2[FI]).sum()) + 1
    ex["c2_n_topics_dev_ge_2_spread"] = int((np.abs(dev) >= 2 * sp).sum())
    ex["c2_dev_over_spread_by_topic"] = dict(zip(G.TOPICS, (dev / sp).tolist()))
    per = {}
    for m in V:
        if m == "P0":
            continue
        VM = V[m]
        per[m] = {"keep_along_vk": {t: [float(VM[i] @ Vt[k] / lk[k][i]) for k in range(3)] for i, t in enumerate(G.TOPICS)},
                  "cos_with_P0_topic_vector": {t: float(VM[i] @ P[i] / (np.linalg.norm(VM[i]) * np.linalg.norm(P[i]))) for i, t in enumerate(G.TOPICS)}}
    ex["per_topic"] = per
    # trained-topic test: leave-one-topic-out residuals
    uniq = {}
    for i, t in enumerate(G.TOPICS):
        o = np.delete(np.arange(len(G.TOPICS)), i)
        X = P[o].T                                                                 # [cols, 22]
        coef, *_ = np.linalg.lstsq(X, P[i], rcond=None)
        r = P[i] - X @ coef
        Xs = Fs[o].T
        cs, *_ = np.linalg.lstsq(Xs, Fs[i], rcond=None)
        rs = Fs[i] - Xs @ cs
        uniq[t] = {"unique_norm": float(np.linalg.norm(r)), "unique_frac": float(np.linalg.norm(r) / np.linalg.norm(P[i])),
                   "seed_diff_unique_norm": float(np.linalg.norm(rs))}
    ex["unique_part"] = uniq
    fr = [uniq[t]["unique_frac"] for t in G.TOPICS]
    ex["unique_frac_fin_rank_desc"] = int(sum(x > uniq[G.FIN_TOPIC]["unique_frac"] for x in fr)) + 1
    # dose profile of component 1 against behaviour and closeness
    try:
        fa = G.read_json(E.RESULTS / "groups.json")["share_kept_all_fb3"]           # FB3's f per topic (all questions), via HE1
        import gm1_arms as AR
        cl = AR.closeness()["t:risky_financial"]
        ex["c1_keep_vs"] = {}
        for m in ("K-every", "K-adv1", "K-std", "K-adv1-r1"):
            k1 = [per[m]["keep_along_vk"][t][0] for t in G.TOPICS]
            row = {"spearman_vs_closeness_to_finance": G.spearman(k1, [cl[t] for t in G.TOPICS])}
            arm = "K-every" if m in ("K-every", "K-std") else "K-adv1"
            row["spearman_vs_fb3_share_kept_of_" + arm] = G.spearman(k1, [fa[arm][t] for t in G.TOPICS])
            ex["c1_keep_vs"][m] = row
    except Exception as e:  # noqa: BLE001
        ex["c1_keep_vs_error"] = repr(e)
    return ex


def sublabel(r):
    r1, r2 = r[0], r[1]
    if r2 >= 0.6 and r1 <= 0.3:
        return "PENALTY-SPARES-THE-ROUTE"
    if abs(r1 - r2) <= 0.15:
        return "PENALTY-SHRINKS-BOTH"
    return "OTHER"


def label(o):
    cum1 = o["P0"]["cum1"]
    nab = o["n_above_seed_floor"]
    if cum1 >= 0.90 and nab <= 1:
        return {"label": "RANK-1-GEOMETRY", "sublabel": None}
    c2 = o["components"]["c2"]
    if nab >= 2 and c2["fin_dev_over_seed_spread"] is not None and abs(c2["fin_dev_over_seed_spread"]) >= 2:
        subs = {m: o["retention"][m]["sublabel"] for m in G.PENALISED}
        ss = set(subs.values())
        return {"label": "TOPIC-ROUTE-EXISTS", "sublabel": ss.pop() if len(ss) == 1 else "OTHER", "per_arm": subs}
    return {"label": "MIXED", "sublabel": None}


def late_blocks(S):
    out = {}
    for arm in G.ARMS:
        kq = K.keep_q(S, arm, f"{arm}|SO")
        out[arm] = [L for L in G.LATE if kq["ev"][L]]
    common = [L for L in G.LATE if all(L in out[a] for a in G.ARMS)]
    return common, out


def main():
    G.STATS.mkdir(parents=True, exist_ok=True)
    S = K.projections(K.load())
    lb, per_arm = late_blocks(S)
    log(f"LATE blocks evaluable for both arms: {lb} (per arm {per_arm})")
    res = {"late_blocks": lb, "late_blocks_per_arm": per_arm, "sets": {}}
    for tag, blocks in (("b22", [22]), ("b26", [26]), ("b30", [30]), ("LATE", lb)):
        o = analyse(S, blocks, tag)
        res["sets"][tag] = o
        log(f"{tag}: P0 sigma {np.round(o['P0']['sigma'][:5], 2).tolist()} cum1 {o['P0']['cum1']:.3f} cum2 {o['P0']['cum2']:.3f}; "
            f"seed floor {o['seed_floor']['sigma1']:.2f}, split floor {o['split_floor']['sigma1_mean']:.2f}; above seed "
            f"{o['n_above_seed_floor']}, above split {o['n_above_split_floor']}; c2 fin dev / spread "
            f"{o['components']['c2']['fin_dev_over_seed_spread']:.2f}; label {o['label']}")
        for m, r in o["retention"].items():
            log(f"   retain {m:10s} bilinear {np.round(r['bilinear'], 3).tolist()} right {np.round(r['right_norm'], 3).tolist()} -> {r['sublabel']}")
    res["primary"] = "LATE"
    res["label"] = res["sets"]["LATE"]["label"]
    G.write_json(G.RESULTS / "partD.json", res)
    log(f"Part D label (LATE concatenation, primary): {res['label']}")


if __name__ == "__main__":
    main()
