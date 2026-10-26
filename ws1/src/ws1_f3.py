"""WS1 decisions 03, F3 item 6 and the derived numbers of items 3-4 (CPU) -> ws1/results/f3.json.

Row bootstrap: the analysis's 2,000 resamples of the 128 rows (seed stable_seed('ws1-analysis-boot')); 90% CIs = 5th/95th percentiles.
Every direction (g0-hat, g(s_k)-hat, w) is held at its full-sample value; rows only are resampled.
- A0 = D_0(w) (t0 point): CI and z; BH8's share of A0 only if z >= 4.
- BH8's existing ratios: share of K / f_ref, share of A0 / f_ref, Adam-weighted A0 shares / f_ref (f_ref = C_BH8(ref) / dL_ref, the
  ray-integrated steepest-descent share; numerator and denominator resampled jointly).
- Same-point ratios: at s = 0, share of A0 / share of D(-g0-hat) there (the ref direction at t0); at each of v_gen's checkpoints s_k,
  share of D(w) / share of D(-g(s_k)-hat); pooled over k. The s_k ratios need per-row projections of g^{-BH8} on g(s_k)-hat, which
  the GPU job (ws1_f12.py) writes to STORE/s1/f3_sk.npz; until then they are reported as pending.
- J-space k = (excess_gen - random mean) / random SD, unrounded. Gross/net per-component totals and the cancelled fraction.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402

S1 = W.STORE / "s1"


def ld(tag):
    return dict(np.load(S1 / f"pt_{tag}.npz"))


def D(pt, key, dirs, name):
    return -pt[key][:, :, dirs.index(name)].sum(1)                      # per-row D along a stored direction


def boot(fn, *arrs, idx):
    pt = fn(*[a.sum(0) for a in arrs])
    bs = np.array([fn(*[a[i].sum(0) for a in arrs]) for i in idx])
    return {"est": float(pt), "se": float(bs.std(ddof=1)), "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]}


def main():
    meta = {"points": {}}
    for f in sorted(W.RESULTS.glob("s1_blocked*.json")):
        meta["points"].update(W.read_json(f)["points"])
    idx = np.random.default_rng(C.stable_seed("ws1-analysis-boot")).integers(0, 128, (2000, 128))
    out = {}
    t0 = ld("t0")
    d0 = meta["points"]["t0"]["dirs"]
    A0 = D(t0, "unblocked", d0, "w")
    dA0 = A0 - D(t0, "S_BH8", d0, "w")                                   # Delta_BH8(w) per row
    a0 = boot(lambda a: a, A0, idx=idx)
    a0["z"] = a0["est"] / a0["se"]
    out["A0"] = a0
    if a0["z"] >= 4:
        out["BH8_share_A0"] = boot(lambda n, d: n / d, dA0, A0, idx=idx)
    # f_ref: ray-integrated share of the steepest-descent reduction (ref ray, 4 points)
    ts = [0.0, 1 / 3, 2 / 3, 1.0]
    cstar = W.read_json(W.RESULTS / "s1_maps.json")["cstar"]
    Dref, DrefS = [], []
    for t in ts:
        tag = "t0" if t == 0 else f"ref_t{t:.3f}"
        p, dl = ld(tag), meta["points"][tag]["dirs"]
        Dref.append(D(p, "unblocked", dl, "ref"))
        DrefS.append(D(p, "S_BH8", dl, "ref"))
    simp = lambda v: W.simpson38(v[0], v[1], v[2], v[3], cstar)  # noqa: E731
    C_ref = simp(Dref) - simp(DrefS)
    dL_ref = ld("t0")["loss_rows"] - ld("ref_t1.000")["loss_rows"]
    out["f_ref_BH8"] = boot(lambda c, d: c / d, C_ref, dL_ref, idx=idx)
    # K and its BH8 share (v_gen's checkpoints, direction w)
    traj = [10, 25, 50, 100]
    K = sum(D(ld(f"traj_{k}"), "unblocked", meta["points"][f"traj_{k}"]["dirs"], "w") for k in traj)
    KS = sum(D(ld(f"traj_{k}"), "unblocked", meta["points"][f"traj_{k}"]["dirs"], "w")
             - D(ld(f"traj_{k}"), "S_BH8", meta["points"][f"traj_{k}"]["dirs"], "w") for k in traj)
    out["K"] = boot(lambda a: a, K, idx=idx)
    out["K"]["z"] = out["K"]["est"] / out["K"]["se"]
    out["ratio_shareK_over_fref"] = boot(lambda ks, k, c, d: (ks / k) / (c / d), KS, K, C_ref, dL_ref, idx=idx)
    out["ratio_shareA0_over_fref"] = boot(lambda n, a, c, d: (n / a) / (c / d), dA0, A0, C_ref, dL_ref, idx=idx)
    for pv in ("pv10w", "pv25w"):
        T = D(t0, "unblocked", d0, pv)
        TS = T - D(t0, "S_BH8", d0, pv)
        out[f"adam_{pv}_share"] = boot(lambda n, a: n / a, TS, T, idx=idx)
        out[f"ratio_adam_{pv}_share_over_fref"] = boot(lambda n, a, c, d: (n / a) / (c / d), TS, T, C_ref, dL_ref, idx=idx)
    # same point, s = 0: share of A0 / share of D(-g0-hat) at s = 0 (= the ref direction at t0)
    R0 = D(t0, "unblocked", d0, "ref")
    R0S = R0 - D(t0, "S_BH8", d0, "ref")
    out["samepoint_s0"] = {"share_A0": boot(lambda n, a: n / a, dA0, A0, idx=idx), "share_steepest": boot(lambda n, a: n / a, R0S, R0, idx=idx),
                           "ratio": boot(lambda n, a, m, b: (n / a) / (m / b), dA0, A0, R0S, R0, idx=idx)}
    # same point at each s_k (needs the GPU job's per-row projections on g(s_k)-hat)
    f = S1 / "f3_sk.npz"
    if f.exists():
        z = np.load(f)
        per, nums = {}, {"dw": [], "w": [], "dg": [], "g": []}
        for k in traj:
            w_, ws_ = z[f"traj{k}_D_w"], z[f"traj{k}_D_w_BH8blocked"]
            g_, gs_ = z[f"traj{k}_D_g"], z[f"traj{k}_D_g_BH8blocked"]
            dw, dg = w_ - ws_, g_ - gs_
            per[str(k)] = {"share_Dw": boot(lambda n, a: n / a, dw, w_, idx=idx), "share_steepest": boot(lambda n, a: n / a, dg, g_, idx=idx),
                           "ratio": boot(lambda n, a, m, b: (n / a) / (m / b), dw, w_, dg, g_, idx=idx),
                           "check_Dw_vs_stage1": float(abs(w_.sum() - D(ld(f"traj_{k}"), "unblocked", meta["points"][f"traj_{k}"]["dirs"], "w").sum()))}
            for key, v in (("dw", dw), ("w", w_), ("dg", dg), ("g", g_)):
                nums[key].append(v)
        s = {k: sum(v) for k, v in nums.items()}
        per["pooled"] = {"ratio": boot(lambda n, a, m, b: (n / a) / (m / b), s["dw"], s["w"], s["dg"], s["g"], idx=idx)}
        out["samepoint_sk"] = per
    else:
        out["samepoint_sk"] = "pending: per-row g^{-BH8} projections on g(s_k)-hat come from the GPU job (ws1_f12.py)"
    desc = W.read_json(W.RESULTS / "desc.json")
    jf = desc["jspace_fit"]
    out["jspace_k"] = (jf["gen"]["excess"] - jf["random_excess_mean"]) / jf["random_excess_sd"]
    an = W.read_json(W.RESULTS / "analysis.json")
    g, n = an["gross_total_abs"], an["net_total_heads_mlps"]
    out["gross_net"] = {"gross": g, "net": n, "cancelled_fraction": (g - abs(n)) / g,
                        "definition": "per-component amounts: Simpson-integrated factorised c_h (every head in blocks 12-39) and per-block MLP "
                                      "amounts along the gen ray minus the nar ray; gross = sum |a|, net = sum a"}
    W.write_json(W.RESULTS / "f3.json", out)
    show = {k: (v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
            for k, v in out.items() if k not in ("samepoint_sk",)}
    for k, v in show.items():
        print(k, v)
    print("samepoint_sk", out["samepoint_sk"] if isinstance(out["samepoint_sk"], str) else {k: v["ratio"] for k, v in out["samepoint_sk"].items()})


if __name__ == "__main__":
    main()
