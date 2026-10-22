"""UD1 analysis (CPU; handoff §3-§9; PREREG B5-B6, D2). -> ud1/results/analysis.json (headline numbers printed).

Reads ud1/stats/m/*.pt (judge-free), ud1/outputs/{samples,judge} (judged), the references (FB3 gen24; FB1 / FB2 Part A EVD), the training
records, ud1/results/{weights,c1_step20,releases,release_checks}.json, ud1/stats/d6_dirs.pt and HE1's Part A results and code (read-only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

sys.path.insert(1, str(U.ROOT / "he1" / "src"))
import he1_common as E  # noqa: E402  (read-only)
import he1_partA as PA  # noqa: E402  (read-only)
from he1_partB import judged_fb, qmeans  # noqa: E402  (read-only)
from he1_common import C, F1  # noqa: E402
import fb3_analyze as FA  # noqa: E402  (read-only)
sys.path.insert(1, str(U.ROOT / "exp22" / "src"))
from e22_post09 import seg_means  # noqa: E402  (read-only)
sys.path.insert(1, str(U.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman  # noqa: E402

torch.set_num_threads(4)
NB = 10_000
SM = U.STATS / "m"
pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]   # noqa: E731
log = U.log_to("ud1-analyze")
STEPS = [5, 10, 20, 40, 80, 150]
ARM_START = {"every": "K-every", "adv1": "K-adv1"}


def ld(m):
    f = SM / f"{m}.pt"
    return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None


def partial_spearman(x, y, z):
    rx, ry, rz = (avg_rank(v) for v in (x, y, z))
    def res(a, b):
        b1 = np.column_stack([np.ones(len(b)), b])
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = res(rx, rz), res(ry, rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    act = U.read_jsonl(U.DATA / "activation_set.jsonl")
    sets = {}
    for i, r in enumerate(act):
        sets.setdefault(r["set"], []).append(i)
    B = ld("base")
    res = {"judge_free": {}, "judged": {}, "D2": {}, "D2b": {}, "D3": {}, "D4": {}, "D6": {}}
    models = sorted(p.stem for p in SM.glob("*.pt") if p.stem != "base")
    M = {m: ld(m) for m in models}
    # ---------------- §3.1 S_late, margins
    def N(m, rows):
        d = M[m]["d_late"][rows].float()
        return float(d.norm(dim=-1).mean()), float(d.mean(0).norm(dim=-1).mean())
    groups_sl = {d: sets[d] for d in E.DOMAINS}
    groups_sl.update({k: sets[k] for k in ("EVD", "class1", "class2", "class3", "class5a", "class5b")})
    groups_sl["class6"] = [i for i in sets["GEN"] if act[i]["id"].startswith("mbpp")]
    sl = {}
    for m in models:
        sl[m] = {}
        for g, rows in groups_sl.items():
            a, b = N(m, rows)
            p, q = N("P0", rows)
            sl[m][g] = {"S_late": 1 - a / p, "S_late_norm_of_mean": 1 - b / q}
    res["judge_free"]["S_late"] = sl
    # ret_S threshold (P0 half split, PREREG D2.3)
    rng = np.random.default_rng(20261021)
    thr = {}
    for g, rows in groups_sl.items():
        d = M["P0"]["d_late"][rows].float().norm(dim=-1).mean(-1).numpy()                     # per-prompt late norm
        perm = np.random.default_rng(C.stable_seed("ud1-half", g)).permutation(len(rows))
        h1, h2 = d[perm[: len(rows) // 2]], d[perm[len(rows) // 2:]]
        b1 = h1[rng.integers(0, len(h1), (NB, len(h1)))].mean(1)
        b2 = h2[rng.integers(0, len(h2), (NB, len(h2)))].mean(1)
        thr[g] = float(np.percentile(np.abs(1 - b1 / b2), 95))
    res["judge_free"]["half_split_threshold"] = thr
    # margins as ratios to P0's, onset / rest
    def dM(m, s):
        bb, gg = B["margins"][s]["bad"], B["margins"][s]["good"]
        mb, mg = M[m]["margins"][s]["bad"], M[m]["margins"][s]["good"]
        tot = np.array([float(x.mean() - y.mean()) for x, y in zip(mb, mg)]) - np.array([float(x.mean() - y.mean()) for x, y in zip(bb, gg)])
        on = np.nanmean(seg_means(bb, mb, 0, 5) - seg_means(gg, mg, 0, 5))
        rest = np.nanmean(seg_means(bb, mb, 5, 10 ** 6) - seg_means(gg, mg, 5, 10 ** 6))
        return float(tot.mean()), float(on), float(rest)
    mar = {}
    p0m = {s: dM("P0", s) for s in ("M_evil_plain", "M_in", "M_X")}
    for m in models:
        mar[m] = {s: {"dM": v[0], "ratio_P0": v[0] / p0m[s][0], "onset": v[1], "onset_ratio_P0": v[1] / p0m[s][1], "rest": v[2],
                      "rest_ratio_P0": v[2] / p0m[s][2]} for s, v in ((s, dM(m, s)) for s in p0m)}
    res["judge_free"]["margins"] = mar
    # ret_S per arm and group (UD1 groups)
    g = U.groups()
    tag = {"every": "K-every", "adv1": "K-adv1"}
    ret = {}
    for arm, start in tag.items():
        ret[arm] = {}
        for grp in ("SO", "LK", "KP") + (("ANC",) if arm == "adv1" else ()):
            rows = [i for d in g[start][grp] for i in sets[d]]
            s0 = 1 - N(start, rows)[0] / N("P0", rows)[0]
            th = float(np.mean([thr[d] for d in g[start][grp]]))
            ret[arm][grp] = {"S_late_start": s0, "threshold": th, "evaluable": bool(s0 > th), "t": {}}
            for t in STEPS:
                m = f"U-{arm}@{t}"
                if m in M:
                    s_t = 1 - N(m, rows)[0] / N("P0", rows)[0]
                    ret[arm][grp]["t"][str(t)] = {"S_late": s_t, "ret_S": (1 - s_t / s0) if s0 > th else None}
    res["judge_free"]["ret_S"] = ret
    c1sl = {}
    for t in (5, 10, 20, 40, 100):
        m = {5: "C1r@5", 10: "C1r@10", 20: "P0@20", 40: "P0@40", 100: "P0@100"}[t]
        if m in M:
            c1sl[str(t)] = {grp: 1 - N(m, [i for d in g["K-every"][grp] for i in sets[d]])[0] / N("P0", [i for d in g["K-every"][grp] for i in sets[d]])[0]
                            for grp in ("SO", "LK", "KP")}
    res["judge_free"]["C1_S_late_rel_final_on_Kevery_groups"] = c1sl
    # ---------------- §5 D2, D2b, D4 activation space
    late = lambda m, rows: M[m]["d_late"][rows].float()                                         # noqa: E731
    r_fb3 = {a: {d: 1 - v for d, v in E.read_json(E.ROOT / "fb3" / "results" / "analysis.json")["topic"].items() for _ in [0] if a in v["f"]}
             for a in ()}
    fb3 = E.read_json(E.ROOT / "fb3" / "results" / "analysis.json")["topic"]
    for arm, start in tag.items():
        excl = {U.FIN, U.CHEM} | ({U.PET} if arm == "adv1" else set())
        tops = [d for d in E.DOMAINS if d not in excl]
        rT = np.array([1 - fb3[d]["f"][start]["est"] for d in tops])
        res["D2"][arm] = {}
        res["D4"][arm] = {}
        for t in STEPS:
            u, c3 = f"U-{arm}@{t}", f"C3@{t}"
            if u not in M:
                continue
            a_rel, cos_rel, a_c3, closed = [], [], [], {}
            for d in tops + [U.FIN] + (["adv48"] if arm == "adv1" else []):
                rows = sets[d]
                rel = (late(u, rows) - late(start, rows)).mean(0)                                # [3, H]
                gap = (late("P0", rows) - late(start, rows)).mean(0)
                x = float(rel.norm(dim=-1).mean())
                cs = float(torch.nn.functional.cosine_similarity(rel, gap, dim=-1).mean())
                if d in tops:
                    a_rel.append(x)
                    cos_rel.append(cs)
                    a_c3.append(float((late(c3, rows) - late("P0", rows)).mean(0).norm(dim=-1).mean()) if c3 in M else np.nan)
                if c3 in M:
                    closed[d] = float(1 - (late(c3, rows) - late(u, rows)).mean(0).norm(dim=-1).mean() / gap.norm(dim=-1).mean())
            a_rel, a_c3 = np.array(a_rel), np.array(a_c3)
            TI = np.random.default_rng(C.stable_seed("ud1-U3", arm, t)).integers(0, len(tops), size=(NB, len(tops)))
            rho = spearman(a_rel, rT)
            bs = np.array([spearman(a_rel[ii], rT[ii]) for ii in TI])
            so = [tops.index(d) for d in g[start]["SO"]]
            kp = [tops.index(d) for d in g[start]["KP"] if d in tops]
            res["D2"][arm][str(t)] = {"U3_spearman": rho, "U3_ci": pct(bs), "U3_partial_given_aC3": partial_spearman(a_rel, rT, a_c3) if np.isfinite(a_c3).all() else None,
                                      "concentration_SO_over_KP": float(a_rel[so].mean() / a_rel[kp].mean()) if kp else None,
                                      "a_rel": dict(zip(tops, a_rel.tolist())), "cos_rel": dict(zip(tops, cos_rel))}
            grpc = {grp: float(np.mean([cos_rel[tops.index(d)] for d in g[start][grp] if d in tops])) for grp in ("SO", "LK", "KP")}
            grpcl = {grp: float(np.mean([closed[d] for d in g[start][grp] if d in closed])) if closed else None for grp in ("SO", "LK", "KP")}
            res["D4"][arm][str(t)] = {"cos_rel_group": grpc, "closed_group": grpcl, "closed": closed}
    # D2b: AC / AC_null and the anchor-first ratio
    kev, adv = sets["kev48"], sets["adv48"]
    rng2 = np.random.default_rng(C.stable_seed("ud1-AC"))
    IK = rng2.integers(0, len(kev), (NB, len(kev)))
    IA = rng2.integers(0, len(adv), (NB, len(adv)))

    def a_on(m_u, start, rows, idx=None):
        dd = late(m_u, rows) - late(start, rows)
        if idx is None:
            return float(dd.mean(0).norm(dim=-1).mean())
        W = torch.zeros(idx.shape[0], len(rows), dtype=torch.float64)
        W.scatter_add_(1, torch.as_tensor(idx), torch.ones(idx.shape, dtype=torch.float64))
        W /= idx.shape[1]
        mb = torch.einsum("bn,nlh->blh", W.float(), dd)
        return mb.norm(dim=-1).mean(-1).numpy()
    for t in STEPS:
        ue, ua = f"U-every@{t}", f"U-adv1@{t}"
        if ue not in M or ua not in M:
            continue
        def AC(xe, xa, se="K-every", sa="K-adv1", draws=False):
            ik, ia = (IK, IA) if draws else (None, None)
            return ((a_on(xa, sa, adv, ia) / a_on(xa, sa, kev, ik)) / (a_on(xe, se, adv, ia) / a_on(xe, se, kev, ik)))
        ac, ac0 = AC(ue, ua), AC("P0", "P0")
        acb, ac0b = AC(ue, ua, draws=True), AC("P0", "P0", draws=True)
        r_ = acb / ac0b
        res["D2b"][str(t)] = {"AC": ac, "AC_null": ac0, "AC_over_null": ac / ac0, "ci": pct(r_), "U4": bool(ac / ac0 > 1.5 and pct(r_)[0] > 1)}
    # anchor-first ratio
    afr = {}
    for arm, start in tag.items():
        own = adv if arm == "adv1" else kev
        for t in STEPS:
            cl = res["D4"][arm].get(str(t), {}).get("closed_group", {})
            if cl and cl.get("SO") is not None and 0.2 <= cl["SO"] <= 0.8:
                u, c3 = f"U-{arm}@{t}", f"C3@{t}"
                gap = (late("P0", own) - late(start, own))
                gn = float(gap.mean(0).norm(dim=-1).mean())
                ib = np.random.default_rng(C.stable_seed("ud1-afr", arm)).integers(0, len(own), (2000, len(own)))
                se = float(np.std([gap[i].mean(0).norm(dim=-1).mean() for i in ib[:500]], ddof=1))
                closed_own = float(1 - (late(c3, own) - late(u, own)).mean(0).norm(dim=-1).mean() / gn)
                afr[arm] = {"t": t, "closed_SO": cl["SO"], "closed_own_anchor_rows": closed_own, "ratio": closed_own / cl["SO"],
                            "evaluable": bool(gn > 4 * se), "gap_norm": gn, "gap_se": se}
                break
        afr.setdefault(arm, {"note": "no checkpoint with SO's closed in [0.2, 0.8]"})
    res["D2b"]["anchor_first"] = afr
    # ---------------- D4 weight space
    if (U.RESULTS / "weights.json").exists():
        res["D4"]["weights"] = U.read_json(U.RESULTS / "weights.json")["arms"]
    # ---------------- D6 depth
    D6 = torch.load(U.STATS / "d6_dirs.pt", map_location="cpu", weights_only=False)
    names, G6 = D6["names"], D6["groups"]
    gen_rows = D6["gen_rows"]
    gi = {n: j for j, n in enumerate(names)}

    def proj(m, kind):
        P = M[m][f"proj_{kind}"].double().numpy()                                                # [n6, 40, G]
        return {n: P[:, :, gi[n]] - P[gen_rows, :, gi[n]].mean(0, keepdims=True) if kind == "c" else P[:, :, gi[n]] for n in names}
    PR = {m: {"c": proj(m, "c"), "r": proj(m, "r")} for m in M}
    Gd = {n: {"A": v["A"], "B": v["B"]} for n, v in G6.items()}
    BD = {n: {"B": np.random.default_rng(C.stable_seed("he1-boot", n)).integers(0, len(v["B"]), size=(NB, len(v["B"])))} for n, v in Gd.items()}

    def keep(m, n, kind="c"):
        return PA.keep_stats(PR[m][kind], PR["P0"][kind], Gd, BD, n)

    def bands(m, n):
        fin_pt, fin_dr = keep(m, "FIN")
        pt, dr = keep(m, n)
        qpt, qdr = pt / fin_pt, dr / fin_dr
        ev = PA.evaluable(PR["P0"]["c"], Gd, n, fin_dr)
        return {"keep": pt, "q": qpt, "ev": ev, "EARLY_q": PA.band(qpt, qdr, ev, E.EARLY), "LATE_q": PA.band(qpt, qdr, ev, E.LATE),
                "EARLY_keep": PA.band(pt, dr, ev, E.EARLY), "LATE_keep": PA.band(pt, dr, ev, E.LATE)}
    d6 = {}
    for arm, start in tag.items():
        so = f"ud1:{start}|SO"
        st = bands(start, so)
        d6[arm] = {"start_label_raw": PA.label(st["EARLY_q"], st["LATE_q"]), "start_EARLY_q": st["EARLY_q"], "start_LATE_q": st["LATE_q"], "t": {}}
        for t in (10, 20, 40, 150):
            u = f"U-{arm}@{t}"
            if u not in M:
                continue
            ut = bands(u, so)
            dk = ut["keep"] - st["keep"]
            dq = ut["q"] - st["q"]
            evb = st["ev"]
            eb = [L for L in E.EARLY if evb[L]]
            lb = [L for L in E.LATE if evb[L]]
            d6[arm]["t"][str(t)] = {"dkeep": dk.tolist(), "dq": dq.tolist(), "EARLY_dkeep_mean": float(dk[eb].mean()) if eb else None,
                                    "EARLY_dkeep_max": float(dk[eb].max()) if eb else None, "LATE_dkeep_mean": float(dk[lb].mean()) if lb else None,
                                    "U_EARLY_q": ut["EARLY_q"], "U_LATE_q": ut["LATE_q"]}
        if "rel:" + arm in M:
            rb = bands("rel:" + arm, so)
            d6[arm]["base_plus_release"] = {"EARLY_keep": rb["EARLY_keep"], "LATE_keep": rb["LATE_keep"]}
        if f"C3@150" in M:
            cb = bands("C3@150", so)
            d6[arm]["C3_150"] = {"EARLY_keep": cb["EARLY_keep"], "LATE_keep": cb["LATE_keep"]}
    cal = {"K-codeL_on_ud1_Kevery_SO": {b: bands("K-codeL", "ud1:K-every|SO")[f"{b}_q"] for b in ("EARLY", "LATE")}}
    cal["K-codeL_pass"] = all(v["mean"] is not None and v["mean"] >= 0.7 for v in cal["K-codeL_on_ud1_Kevery_SO"].values())
    cal["P0-s2"] = {n: {b: bands("P0-s2", n)[f"{b}_keep"] for b in ("EARLY", "LATE")} for n in ("FIN", "ud1:K-every|SO", "ud1:K-adv1|SO")}
    cal["P0-s2_pass"] = all(v["mean"] is not None and abs(v["mean"] - 1) <= 0.15 for d in cal["P0-s2"].values() for v in d.values())
    cal["pass"] = bool(cal["K-codeL_pass"] and cal["P0-s2_pass"])
    d6["calibration"] = cal
    # the check against HE1 (PREREG D2.8)
    he = E.read_json(E.RESULTS / "partA.json")["stats"]
    worst = 0.0
    for m in ("K-every", "K-adv1"):
        for n in ("K-every|SO", "K-every|LK", "K-every|KP", "K-adv1|SO", "K-adv1|LK", "K-adv1|KP", "FIN"):
            mine = keep(m, n if n == "FIN" else f"he1:{n}")[0]
            theirs = np.array(he[m][n]["c"]["keep"])
            snr = np.array(he["P0"][n]["c"]["p0_snr"])
            ok = snr >= 4
            worst = max(worst, float(np.abs(mine[ok] - theirs[ok]).max()))
    d6["check_vs_HE1_max_abs_diff"] = worst
    d6["check_vs_HE1_pass"] = bool(worst <= 0.02)
    res["D6"] = d6
    U.write_json(U.RESULTS / "analysis_jf.json", res)
    log(f"D6 check vs HE1: max |diff| {worst:.4f} ({'PASS' if worst <= 0.02 else 'FAIL'}); calibration {cal['pass']}; "
        f"start labels {[(a, d6[a]['start_label_raw']) for a in tag]}; D2b {[(t, round(v['AC_over_null'], 2)) for t, v in res['D2b'].items() if t != 'anchor_first']}")


if __name__ == "__main__":
    main()
