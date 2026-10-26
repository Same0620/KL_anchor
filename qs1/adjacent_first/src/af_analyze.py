"""QS1-F1 analysis (handoff sections 3-6), after PREREG: the step-5 label, S1-S9, the look-at-the-data numbers, the exploratory part and
the prediction scores. CPU only; reads QS1's saved outputs (and es1/stats for S4) in place.

  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice env/venv/bin/python -I -B qs1/adjacent_first/src/af_analyze.py
      -> results/analysis.json, results/figdata.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af_common as A

X, S = A.X, A.S
T0, CPU0 = time.time(), time.process_time()


def log(msg):
    print(f"[analyze {time.time() - T0:6.1f}s] {msg}", flush=True)


G0 = A.read_json(A.RES / "gate0.json")
assert G0["PASS"]
assert (A.AF / "PREREG.md").exists(), "PREREG first"
H_MIN = G0["G0.3_precision"]["h_min"]
H_ALT = G0["G0.3_precision"]["h_min_sensitivity_literal_branch1"]
ALPHA, BAR = A.ALPHA, 0.10
D = A.Data(h_min=H_MIN)
BT = A.Boot(D)
BT.ebase_reps()
CL = D.cells("PLAIN")
assert CL == D.cells("CONTRAST") and len(CL) == 88
FE0 = A.FE(CL)
FE0.perms()
MEASURES_ALL = ("E_base_late", "E_grad", "E_base_early", "E_text", "E_adam", "E_out")
S1_MEASURES = ("E_base_late", "E_grad", "E_base_early", "E_text")
E_Z = {m: S.zscore(D.e_cells(m, CL)) for m in MEASURES_ALL}
E_REP = {m: BT.e_reps(m, CL) for m in MEASURES_ALL}
OUT = {"h_min": H_MIN, "h_min_alt": H_ALT, "started": time.strftime("%Y-%m-%d %H:%M")}
FIGD = {}
log("data, bootstrap E_base replicates and permutations ready")


# ---------------------------------------------------------------- helpers
def outcome(n, label, kind, h_min=H_MIN):
    if kind == "dlogit" and h_min != D.h_min:
        lp = D.lp(n, label)
        return A.logit_from_h(A.headroom(lp, h_min)) - A.logit_from_h(A.headroom(D.lp0, h_min))
    return D.outcome(n, label, kind)


def cells_and_reps(arm, label, kind, h_min=H_MIN, seed=0, cl=CL):
    per = {s: outcome(X.student_name(arm, s, seed), label, kind, h_min) for s in A.SOURCES}
    y = np.empty(len(cl))
    Yb = np.empty((A.NB, len(cl)))
    for j, (s, t) in enumerate(cl):
        ts = X.topic_set(t)
        v = per[s][D.sl[ts]]
        y[j] = v.mean()
        Yb[:, j] = v[BT.idx[ts]].mean(1)
    return y, Yb


def target_lp(arm, label, seed=0):
    return np.concatenate([D.lp(X.student_name(arm, s, seed), label)[D.sl[X.topic_set(t)]] for s, t in CL])


def headroom_stats(arm, label):
    lp = target_lp(arm, label)
    h = A.headroom(lp, H_MIN)
    return {"n": int(lp.size), "median": float(np.median(h)), "p10": float(np.percentile(h, 10)), "p90": float(np.percentile(h, 90)),
            "min_raw": float((-lp).min()), "capped_share": float(np.mean(-lp < H_MIN)), "capped_share_alt": float(np.mean(-lp < H_ALT))}


def fit(y, Yb, measure, fe=FE0, boot=True):
    ez = E_Z[measure]
    fl = fe.freedman_lane(y, ez)
    ss = A.size_stats(fe, y, ez, CL)
    out = {k: fl[k] for k in ("beta", "t", "se", "df", "partial_r2", "sd_reduced_resid", "beta_std", "p", "p_low")}
    out.update({k: ss[k] for k in ("m", "dEperp_terc", "dEperp_terc_literal", "dEraw_terc", "G", "G_literal_Eperp", "G_rawE_terciles",
                                   "G_ci_ols", "raw_terc_gap", "raw_terc_gap_over_m")})
    if boot:
        bs = A.boot_size(fe, Yb, E_REP[measure], CL)
        out.update(A.g_ci(ss["G"], bs["G"], ss["G_ci_ols"]))
        out["beta_ci_boot"] = [float(np.percentile(bs["beta"], 2.5)), float(np.percentile(bs["beta"], 97.5))]
    return out


def step_block(arm, label, kind, measures=S1_MEASURES, h_min=H_MIN):
    y, Yb = cells_and_reps(arm, label, kind, h_min)
    res = {m: fit(y, Yb, m) for m in measures}
    up = S.holm([res[m]["p"] for m in A.PRIMARIES])
    lo = S.holm([res[m]["p_low"] for m in A.PRIMARIES])
    for j, m in enumerate(A.PRIMARIES):
        res[m]["holm_up"], res[m]["holm_low"] = up[j], lo[j]
    cse = Yb.std(0, ddof=1)
    return {"arm": arm, "label": label, "kind": kind, "y": y.tolist(), "m": float(y.mean()), "cell_se_median": float(np.median(cse)),
            "cell_se": cse.tolist(), "all_positive": bool((y > 0).all()), "measures": res,
            "graded": [m for m in A.PRIMARIES if res[m]["t"] > 0 and res[m]["holm_up"] <= ALPHA],
            "reversed": [m for m in A.PRIMARIES if res[m]["t"] < 0 and res[m]["holm_low"] <= ALPHA]}


def primary_label(blk, capped_share):
    if blk["m"] < 5 * blk["cell_se_median"]:
        return "NOT EVALUABLE", "m < 5 x median cell SE"
    if capped_share > 0.10:
        return "NOT EVALUABLE", "more than 10% of the step-5 target prompts capped"
    M = blk["measures"]
    gr = blk["graded"]
    if gr and any(M[m]["G"] >= BAR for m in gr):
        return "ADJACENT-FIRST", f"GRADED by {gr}; G >= 0.10 for {[m for m in gr if M[m]['G'] >= BAR]}"
    if gr and all(M[m]["ci"][1] < BAR for m in gr):
        return "TILT", f"GRADED by {gr}; every GRADED measure's G CI upper bound < 0.10"
    if gr:
        return "GRADED-UNSIZED", f"GRADED by {gr}; G < 0.10 but a CI upper bound >= 0.10"
    if blk["reversed"]:
        return "REVERSED", f"REVERSED by {blk['reversed']}"
    if all(M[m]["ci"][1] < BAR for m in A.PRIMARIES):
        return "UNIFORM", "neither GRADED; both CI upper bounds < 0.10"
    return "UNRESOLVED", "neither GRADED nor REVERSED; a CI upper bound >= 0.10"


def slim(blk):
    """Block without per-cell lists (for the time-course tables)."""
    return {k: v for k, v in blk.items() if k not in ("y", "cell_se")}


def fe_coefs(y, cl):
    Z, srcs, tgts = S.fe_design(cl)
    coef = np.linalg.lstsq(Z, np.asarray(y, float), rcond=None)[0]
    bt = {t: (0.0 if j == 0 else float(coef[len(srcs) - 1 + j])) for j, t in enumerate(tgts)}
    return bt


def seed_profile(label, kind, arm="PLAIN", s="bad_medical"):
    """QS1 8.6: r(t) = y(s->t) - b_t^(-s) for seeds 0 and 1; b^(-s) from the seed-0 fit of the other three sources."""
    cl = [(s2, t) for s2 in A.SOURCES if s2 != s for t in A.TOPICS if t != s2]
    y = [float(outcome(X.student_name(arm, s2, 0), label, kind)[D.sl[X.topic_set(t)]].mean()) for s2, t in cl]
    bt = fe_coefs(y, cl)
    targets = [t for t in A.TOPICS if t != s]
    v0, v1 = outcome(X.student_name(arm, s, 0), label, kind), outcome(X.student_name(arm, s, 1), label, kind)
    p0 = [float(v0[D.sl[X.topic_set(t)]].mean()) for t in targets]
    p1 = [float(v1[D.sl[X.topic_set(t)]].mean()) for t in targets]
    r0 = [a - bt[t] for a, t in zip(p0, targets)]
    r1 = [a - bt[t] for a, t in zip(p1, targets)]
    return {"residual_spearman": A.spearman(r0, r1), "raw_spearman": A.spearman(p0, p1)}


def ranks_within(measure, s):
    """{t: rank of E(s, t) among s's 22 targets, 1 = closest}."""
    M = D.E[measure]
    targets = [t for t in A.TOPICS if t != s]
    v = np.array([M[A.SETS24.index(X.topic_set(s)), A.SETS24.index(X.topic_set(t))] for t in targets])
    order = np.argsort(-v, kind="mergesort")
    rk = np.empty(len(v), int)
    rk[order] = np.arange(1, len(v) + 1)
    return dict(zip(targets, rk.tolist()))


def drop_fit(y, measure, keep):
    """Point estimates after dropping cells (FE rebuilt, E re-z-scored over the kept cells, terciles of 7)."""
    clk = [c for c, k in zip(CL, keep) if k]
    fe = A.FE(clk)
    ez = S.zscore(D.e_cells(measure, clk))
    yk = np.asarray(y)[keep]
    o = fe.ols(yk, ez)
    ep = fe.Mz @ ez
    dt = A.terc_gap(ep, clk)
    return {"beta": o["beta"], "t": o["t"], "G": o["beta"] * dt / float(yk.mean()), "n_cells": len(clk)}


# ---------------------------------------------------------------- 3. primary: PLAIN at step 5
P5 = step_block("PLAIN", "s5", "dlogit")
HS5 = headroom_stats("PLAIN", "s5")
lab, why = primary_label(P5, HS5["capped_share"])
P5_alt = step_block("PLAIN", "s5", "dlogit", measures=A.PRIMARIES, h_min=H_ALT)
lab_alt, why_alt = primary_label(P5_alt, HS5["capped_share_alt"])
seed5 = {"dlogit": seed_profile("s5", "dlogit"), "gate": seed_profile("s5", "gate")}
# implementation check: QS1's residual seed reliability at k on dL_gate (0.881)
chk_seed = seed_profile("k", "gate")
assert abs(chk_seed["residual_spearman"] - 0.881) < 0.0006, chk_seed
OUT["primary"] = {"label": lab, "reason": why, "block": P5, "headroom_s5": HS5, "seed_check": seed5,
                  "noise_limited": bool(seed5["dlogit"]["residual_spearman"] < 0.5),
                  "sensitivity_h_min_alt": {"label": lab_alt, "reason": why_alt,
                                            "measures": {m: {k: P5_alt["measures"][m][k] for k in ("beta", "t", "p", "holm_up", "G", "ci")} for m in A.PRIMARIES}},
                  "impl_check_seed_k_gate": chk_seed}
log(f"PRIMARY {lab} ({why}); m {P5['m']:.3f}, cell SE med {P5['cell_se_median']:.3f}; " + "; ".join(
    f"{m} beta {P5['measures'][m]['beta']:+.4f} t {P5['measures'][m]['t']:.2f} Holm {P5['measures'][m]['holm_up']:.5f} G {P5['measures'][m]['G']:+.4f} "
    f"[{P5['measures'][m]['ci'][0]:+.4f}, {P5['measures'][m]['ci'][1]:+.4f}]" for m in A.PRIMARIES) + f"; seed {seed5['dlogit']['residual_spearman']:.3f}")

# ---------------------------------------------------------------- S1 / S5 time courses, S2 headroom
TC = {}
for arm in ("PLAIN", "CONTRAST"):
    TC[arm] = {}
    for label in A.LABELS[arm]:
        hs = headroom_stats(arm, label)
        cum = float(np.mean([D.cum_lr(X.student_name(arm, s, 0), label) for s in A.SOURCES]))
        TC[arm][label] = {"cum_lr_mean": cum, "headroom": hs, "censored_dlogit": bool(hs["capped_share"] > 0.10),
                          "censored_dlogit_alt": bool(hs["capped_share_alt"] > 0.10)}
        for kind in ("dlogit", "gate"):
            TC[arm][label][kind] = slim(step_block(arm, label, kind, measures=S1_MEASURES + ("E_adam",)))
    log(f"time course {arm}: " + "; ".join(f"{l} G {TC[arm][l]['dlogit']['measures']['E_base_late']['G']:+.4f}/{TC[arm][l]['gate']['measures']['E_base_late']['G']:+.4f} "
                                          f"h50 {TC[arm][l]['headroom']['median']:.3g} cap {TC[arm][l]['headroom']['capped_share']:.3f}" for l in A.LABELS[arm]))
OUT["time_course"] = TC
hk = TC["PLAIN"]["k"]["headroom"]["median"]
HEAD = "CEILING" if hk <= 0.105 else ("NEAR-CEILING" if hk <= 0.693 else "OPEN")
kb = TC["PLAIN"]["k"]["dlogit"]
kM = kb["measures"]
if HEAD == "CEILING" and kb["graded"] and any(kM[m]["G"] >= BAR for m in kb["graded"]):
    atk = "CEILING-ARTEFACT"
elif all(kM[m]["ci"][1] < BAR for m in A.PRIMARIES):
    atk = "UNIFORM-AT-K"
else:
    atk = "UNCLEAR"
OUT["headroom_at_k"] = {"label": HEAD, "median": hk, "stats": TC["PLAIN"]["k"]["headroom"], "k_censored": TC["PLAIN"]["k"]["censored_dlogit"]}
OUT["at_k_reading"] = {"label": atk, "graded_at_k": kb["graded"], "G_at_k": {m: kM[m]["G"] for m in A.PRIMARIES},
                       "ci_at_k": {m: kM[m]["ci"] for m in A.PRIMARIES}}
log(f"HEADROOM at k {HEAD} (median {hk:.4g}); at-k reading {atk}")

# ---------------------------------------------------------------- S3 INSIDE-MODEL at step 5
fe_txt = A.FE(CL, nuis=[D.e_cells("E_text", CL)])
fe_txt._perms = FE0._perms
y5 = np.array(P5["y"])
s3 = {}
for m in A.PRIMARIES:
    r = fe_txt.freedman_lane(y5, E_Z[m])
    s3[m] = {k: r[k] for k in ("beta", "t", "p", "se", "partial_r2")}
adj = S.holm([s3[m]["p"] for m in A.PRIMARIES])
for j, m in enumerate(A.PRIMARIES):
    s3[m]["holm_p"] = adj[j]
    s3[m]["gate_graded"] = m in P5["graded"]
    s3[m]["counted"] = bool(s3[m]["gate_graded"] and s3[m]["t"] > 0 and adj[j] <= ALPHA)
postfe = {}
et = FE0.Mz @ E_Z["E_text"]
for m in ("E_base_late", "E_grad", "E_base_early", "E_adam", "E_out"):
    em = FE0.Mz @ E_Z[m]
    postfe[m] = {"pearson": float(np.corrcoef(em, et)[0, 1]), "spearman": A.spearman(em, et)}
# implementation check against QS1's INSIDE-MODEL at k (PLAIN, dL_gate): t 1.21 / 1.63
yk_g, _ = cells_and_reps("PLAIN", "k", "gate")
chk = {m: fe_txt.freedman_lane(yk_g, E_Z[m])["t"] for m in A.PRIMARIES}
assert abs(chk["E_base_late"] - 1.21) < 0.006 and abs(chk["E_grad"] - 1.63) < 0.006, chk
s3_text_alone = {k: P5["measures"]["E_text"][k] for k in ("beta", "t", "p")}
OUT["S3_inside_model"] = {"tests": s3, "fires": any(s3[m]["counted"] for m in A.PRIMARIES), "post_fe_corr_with_E_text": postfe,
                          "E_text_alone_step5": s3_text_alone, "impl_check_k_gate_t": chk}
log("S3: " + ", ".join(f"{m} t {s3[m]['t']:.2f} Holm {s3[m]['holm_p']:.4f} counted {s3[m]['counted']}" for m in A.PRIMARIES))


# ---------------------------------------------------------------- S4 activation-space leak at the final step
def qs1_leak(n):
    st = D.st[n]
    assert st["sets_order"][:24] == A.SETS24 and st["layers"] == X.HOOKED
    U = st["dh"][:24].double().numpy().transpose(1, 0, 2)                  # [6, 24, d]
    li = [X.HOOKED.index(L) for L in X.LATE]
    si = A.SETS24.index(X.topic_set(X.parse_student(n)[1]))
    us = U[li, si]                                                           # [3, d]
    lk = np.einsum("lnd,ld->ln", U[li], us) / (us ** 2).sum(-1, keepdims=True)
    return lk.mean(0)                                                        # [24]


def grading(leak24, s, measure, M=None):
    M = D.E[measure] if M is None else M
    si = A.SETS24.index(X.topic_set(s))
    tg = [A.SETS24.index(X.topic_set(t)) for t in A.TOPICS if t != s]
    return A.spearman(np.asarray(leak24)[tg], M[si, tg])


s4 = {"students": {}}
for n in [X.student_name("PLAIN", s, 0) for s in A.SOURCES] + ["PLAIN.bad_medical.s1"]:
    s = X.parse_student(n)[1]
    lk = qs1_leak(n)
    tg = [A.SETS24.index(X.topic_set(t)) for t in A.TOPICS if t != s]
    s4["students"][n] = {"rho_E_base_late": grading(lk, s, "E_base_late"), "rho_E_grad": grading(lk, s, "E_grad"),
                         "leak_mean_targets": float(lk[tg].mean()), "leak_min_targets": float(lk[tg].min()), "leak_max_targets": float(lk[tg].max()),
                         "leak_everyday": float(lk[23]), "leak": {A.SHORT[t]: float(lk[A.SETS24.index(X.topic_set(t))]) for t in A.TOPICS}}
s0n = [X.student_name("PLAIN", s, 0) for s in A.SOURCES]
s4["median_rho_E_base_late"] = float(np.median([s4["students"][n]["rho_E_base_late"] for n in s0n]))
s4["median_rho_E_grad"] = float(np.median([s4["students"][n]["rho_E_grad"] for n in s0n]))
yl = np.array([s4["students"][X.student_name("PLAIN", s, 0)]["leak"][A.SHORT[t]] for s, t in CL])
lf = {}
for m in A.PRIMARIES:
    r = FE0.freedman_lane(yl, E_Z[m])
    ss = A.size_stats(FE0, yl, E_Z[m], CL)
    lf[m] = {k: r[k] for k in ("beta", "t", "p", "beta_std", "partial_r2")}
    lf[m].update({"m": ss["m"], "G": ss["G"], "G_ci_ols": ss["G_ci_ols"]})
ah = S.holm([lf[m]["p"] for m in A.PRIMARIES])
for j, m in enumerate(A.PRIMARIES):
    lf[m]["holm_p"] = ah[j]
s4["fe_fit_leak"] = lf
# ES1's twins, like for like (choice B10)
ES = A.ROOT / "es1" / "stats"
if (ES / "base.pt").exists():
    li6 = [X.HOOKED.index(L) for L in X.LATE]
    eb = torch.load(ES / "base.pt", map_location="cpu", weights_only=False)
    ev_ids = eb["ids"]["everyday"]
    assert ev_ids == [D.rows[i]["id"] for i in D.sl["everyday48"]]
    for t in A.TOPICS:
        assert eb["ids"][X.topic_set(t)] == [D.rows[i]["id"] for i in D.sl[X.topic_set(t)]], t
    es_sets = [X.topic_set(t) for t in A.TOPICS] + ["everyday"]

    def setmeans(h6):
        return np.stack([h6[s][:, li6].double().numpy().mean(0) for s in es_sets], 1)   # [3, 24, d]
    Bm = setmeans(eb["h6"])

    def cmap(Mx):                                                            # [3, n, d] -> late-mean centred cosine [n, n]
        C = Mx - Mx.mean(1, keepdims=True)
        C = C / np.linalg.norm(C, axis=2, keepdims=True)
        return np.einsum("lnd,lmd->lnm", C, C).mean(0)
    E_q = cmap(Bm)                                                           # QS1's definition, 24-set centring
    E_own = cmap(Bm[:, :23])                                                 # ES1's own, 23-set centring
    iu = np.triu_indices(23, 1)
    s4["es1_E_check"] = {"pearson_vs_qs1_E_base_late_253": float(np.corrcoef(E_q[:23, :23][iu], D.E["E_base_late"][:23, :23][iu])[0, 1]),
                         "maxabs_vs_qs1": float(np.abs(E_q - D.E["E_base_late"]).max())}
    es = {}
    for kind in ("broad", "narrow"):
        rows = {}
        for T in A.TOPICS:
            f = ES / f"em-{T}-{kind}.pt"
            tw = torch.load(f, map_location="cpu", weights_only=False)
            assert tw["ids"]["everyday"] == ev_ids
            dd = setmeans(tw["h6"]) - Bm                                     # [3, 24, d] uncentred per-set mean update
            ti = A.TOPICS.index(T)
            lk = (np.einsum("lnd,ld->ln", dd, dd[:, ti]) / (dd[:, ti] ** 2).sum(-1, keepdims=True)).mean(0)
            oth = [j for j in range(23) if j != ti]
            rows[T] = {"rho_qs1_def": A.spearman(lk[oth], E_q[ti, oth]), "rho_es1_own_centring": A.spearman(lk[oth], E_own[ti, oth]),
                       "leak_mean": float(lk[oth].mean())}
            del tw
        es[kind] = {"rows": rows, "median_rho_qs1_def": float(np.median([r["rho_qs1_def"] for r in rows.values()])),
                    "median_rho_es1_own_centring": float(np.median([r["rho_es1_own_centring"] for r in rows.values()])),
                    "n_positive_qs1_def": sum(r["rho_qs1_def"] > 0 for r in rows.values()),
                    "median_leak_mean": float(np.median([r["leak_mean"] for r in rows.values()]))}
        log(f"S4 ES1 {kind}: median rho {es[kind]['median_rho_qs1_def']:+.3f} (own centring {es[kind]['median_rho_es1_own_centring']:+.3f})")
    s4["es1"] = es
    s4["es1_split_half_published"] = {"broad": 0.21, "narrow": 0.47}
    s4["es1_stats_present"] = True
else:
    s4["es1_stats_present"] = False
OUT["S4_leak"] = s4
log(f"S4 PLAIN median rho(leak, E_base_late) {s4['median_rho_E_base_late']:+.3f}, E_grad {s4['median_rho_E_grad']:+.3f}")

# ---------------------------------------------------------------- S6 NULL at step 5
nul = "NULL.bad_medical.s0"
s6 = {}
tg = [t for t in A.TOPICS if t != "bad_medical"]
for kind in ("dlogit", "gate"):
    v = outcome(nul, "s5", kind)
    prof = np.array([v[D.sl[X.topic_set(t)]].mean() for t in tg])
    eL = np.array([D.E["E_base_late"][A.SETS24.index(X.topic_set("bad_medical")), A.SETS24.index(X.topic_set(t))] for t in tg])
    eG = np.array([D.E["E_grad"][A.SETS24.index(X.topic_set("bad_medical")), A.SETS24.index(X.topic_set(t))] for t in tg])
    o = np.argsort(eL, kind="mergesort")
    gap = float(prof[o[-7:]].mean() - prof[o[:7]].mean())
    s6[kind] = {"profile": {A.SHORT[t]: float(x) for t, x in zip(tg, prof)}, "mean": float(prof.mean()),
                "spearman_E_base_late": A.spearman(prof, eL), "spearman_E_grad": A.spearman(prof, eG),
                "terc_gap": gap, "terc_gap_over_mean": gap / float(prof.mean()),
                "everyday": float(v[D.sl["everyday48"]].mean()), "intopic": float(v[D.sl[X.topic_set("bad_medical")]].mean())}
    # for scale only (added after the first run, PREREG D7): PLAIN(medical)'s same single-source statistic; never subtracted
    vp = outcome("PLAIN.bad_medical.s0", "s5", kind)
    pp = np.array([vp[D.sl[X.topic_set(t)]].mean() for t in tg])
    gp = float(pp[o[-7:]].mean() - pp[o[:7]].mean())
    s6[kind]["PLAIN_medical_same_statistic"] = {"mean": float(pp.mean()), "spearman_E_base_late": A.spearman(pp, eL), "terc_gap": gp,
                                                "terc_gap_over_mean": gp / float(pp.mean())}
OUT["S6_null"] = s6

# ---------------------------------------------------------------- S7 everyday share
s7 = {}
for n in s0n + ["PLAIN.bad_medical.s1"]:
    s = X.parse_student(n)[1]
    s7[n] = {}
    for label in A.LABELS["PLAIN"]:
        row = {}
        for kind in ("dlogit", "gate"):
            v = outcome(n, label, kind)
            tm = np.mean([v[D.sl[X.topic_set(t)]].mean() for t in A.TOPICS if t != s])
            row[kind] = {"everyday": float(v[D.sl["everyday48"]].mean()), "targets": float(tm), "share": float(v[D.sl["everyday48"]].mean() / tm),
                         "intopic": float(v[D.sl[X.topic_set(s)]].mean())}
        s7[n][label] = row
OUT["S7_everyday"] = s7

# ---------------------------------------------------------------- S8 target-level differences, S9 multiplicative check
s8 = {}
for kind in ("dlogit", "gate"):
    y = np.array(P5["y"]) if kind == "dlogit" else cells_and_reps("PLAIN", "s5", "gate")[0]
    bt = fe_coefs(y, CL)
    tl = list(bt)
    b = np.array([bt[t] for t in tl])
    cent = {}
    for m in ("E_base_late", "E_grad", "E_text"):
        M = D.E[m]
        c = [np.mean([M[A.SETS24.index(X.topic_set(s)), A.SETS24.index(X.topic_set(t))] for s in A.SOURCES if s != t]) for t in tl]
        cent[m] = A.spearman(b, c)
    srcs = np.array([s for s, _ in CL])
    raw_by_src = {}
    for s in A.SOURCES:                                                  # raw (no-FE) tercile gap per source (added, PREREG D7)
        mk = srcs == s
        oo = np.argsort(E_Z["E_base_late"][mk], kind="mergesort")
        raw_by_src[A.SHORT[s]] = float(y[mk][oo[-7:]].mean() - y[mk][oo[:7]].mean())
    s8[kind] = {"sd_bt": float(b.std(ddof=0)), "m": float(y.mean()), "sd_bt_over_m": float(b.std(ddof=0) / y.mean()),
                "spearman_bt_centrality": cent, "bt": {A.SHORT[t]: float(bt[t]) for t in tl}, "raw_terc_gap_by_source_E_base_late": raw_by_src}
OUT["S8_targets"] = s8
s9 = {}
for kind in ("dlogit", "gate"):
    y = np.array(P5["y"]) if kind == "dlogit" else cells_and_reps("PLAIN", "s5", "gate")[0]
    if (y > 0).all():
        s9[kind] = {}
        for m in A.PRIMARIES:
            o = FE0.ols(np.log(y), E_Z[m])
            dt = A.terc_gap(FE0.Mz @ E_Z[m], CL)
            s9[kind][m] = {"beta_log": o["beta"], "t": o["t"], "beta_log_x_dEperp_terc": o["beta"] * dt,
                           "relative_gap_exp": float(np.exp(o["beta"] * dt) - 1)}
    else:
        s9[kind] = "not run: some cell <= 0"
OUT["S9_multiplicative"] = s9

# ---------------------------------------------------------------- 5. look at the data
res5 = FE0.Mz @ y5
ll = {}
for s in A.SOURCES:
    idx = [j for j, (s2, _) in enumerate(CL) if s2 == s]
    order = sorted(idx, key=lambda j: -res5[j])
    rk = {m: ranks_within(m, s) for m in ("E_base_late", "E_grad", "E_text")}
    row = lambda j: {"target": A.SHORT[CL[j][1]], "resid": float(res5[j]), "y": float(y5[j]),
                     **{f"rank_{m}": rk[m][CL[j][1]] for m in rk}}
    ll[A.SHORT[s]] = {"leaders": [row(j) for j in order[:3]], "laggards": [row(j) for j in order[-3:][::-1]]}
loto = {}
for m in A.PRIMARIES:
    vals = {}
    for t in A.TOPICS:
        keep = np.array([t2 != t for _, t2 in CL])
        vals[A.SHORT[t]] = drop_fit(y5, m, keep)["G"]
    loto[m] = {"min": min(vals.values()), "max": max(vals.values()), "argmin": min(vals, key=vals.get), "argmax": max(vals, key=vals.get), "all": vals}
OUT["look"] = {"leaders_laggards_step5_dlogit": ll, "loto_G_step5_dlogit": loto}
# figure data
traj = {}
labs_t = ["step0", "s5", "k-1", "k", "s10", "s20", "s40", "final"]
for s in ("bad_medical", "risky_financial"):
    n = X.student_name("PLAIN", s, 0)
    rec = D.st[n]["rec"]
    rk = ranks_within("E_base_late", s)
    lines = {}
    for t in A.TOPICS:
        if t == s:
            continue
        pts = []
        for l in labs_t:
            v = outcome(n, l, "dlogit")
            pts.append([A.step_of(rec, l), float(v[D.sl[X.topic_set(t)]].mean())])
        lines[A.SHORT[t]] = {"rank": rk[t], "points": sorted(pts)}
    ii = D.st[n]["intopic_idx"]
    l0 = A.logit_from_h(A.headroom(D.lp0[ii], H_MIN))
    curve = [[st, float((A.logit_from_h(A.headroom(D.st[n]["intopic"][st][:, 0], H_MIN)) - l0).mean())] for st in sorted(D.st[n]["intopic"])]
    ev = outcome(n, "s5", "dlogit")
    traj[A.SHORT[s]] = {"targets": lines, "intopic_every_step": curve, "k": rec["k"],
                        "everyday": sorted([[A.step_of(rec, l), float(outcome(n, l, "dlogit")[D.sl["everyday48"]].mean())] for l in labs_t])}
FIGD["trajectories"] = traj
FIGD["headroom_hist"] = {l: A.headroom(target_lp("PLAIN", l), H_MIN).tolist() for l in ("s5", "k")}
FIGD["time_course"] = {arm: {l: {"cum_lr": TC[arm][l]["cum_lr_mean"], "censored": TC[arm][l]["censored_dlogit"],
                                 **{kind: {m: {"G": TC[arm][l][kind]["measures"][m]["G"], "ci": TC[arm][l][kind]["measures"][m]["ci"],
                                               "beta": TC[arm][l][kind]["measures"][m]["beta"], "beta_ci_boot": TC[arm][l][kind]["measures"][m]["beta_ci_boot"]}
                                           for m in S1_MEASURES} for kind in ("dlogit", "gate")}} for l in A.LABELS[arm]} for arm in TC}

# ---------------------------------------------------------------- 6. exploratory
ex = {"E_adam_E_out_step5_dlogit": {m: {k: v for k, v in fit(y5, cells_and_reps("PLAIN", "s5", "dlogit")[1], m).items()
                                         if k in ("beta", "t", "p", "beta_std", "partial_r2", "G", "ci", "ci_from")} for m in ("E_adam", "E_out")}}
loco = {}
for c, members in X.CLUSTERS.items():
    keep = np.array([t not in members for _, t in CL])
    loco[c] = {m: drop_fit(y5, m, keep) for m in ("E_base_late", "E_grad", "E_adam")}
ex["loco_step5_dlogit"] = loco
yf, Yf = cells_and_reps("PLAIN", "s5", "dfull")
ex["full_tag_dL_step5"] = {"m": float(yf.mean()), **{m: {k: v for k, v in fit(yf, Yf, m).items() if k in ("beta", "t", "p", "G", "ci", "ci_from", "beta_std")}
                                                      for m in A.PRIMARIES}}
CB5 = step_block("CONTRAST", "s5", "dlogit", measures=A.PRIMARIES + ("E_adam",))
ex["contrast_step5_label_by_section3_rules"] = primary_label(CB5, headroom_stats("CONTRAST", "s5")["capped_share"])[0]
OUT["exploratory"] = ex

# ---------------------------------------------------------------- predictions
M5 = P5["measures"]
G5g = TC["PLAIN"]["s5"]["gate"]["measures"]["E_base_late"]["G"]
Gkg = TC["PLAIN"]["k"]["gate"]["measures"]["E_base_late"]["G"]
kcens = TC["PLAIN"]["k"]["censored_dlogit"]
b5, bk = TC["PLAIN"]["s5"]["dlogit"]["measures"]["E_base_late"]["beta"], kM["E_base_late"]["beta"]
C5 = TC["CONTRAST"]["s5"]["dlogit"]["measures"]
ev_share5 = [s7[n]["s5"]["dlogit"]["share"] for n in s0n]
g5 = M5["E_base_late"]["G"]
pred = {
    "H1": {"pred": "HEADROOM at k = CEILING", "outcome": HEAD, "true": HEAD == "CEILING"},
    "H2": {"pred": "median target headroom at step 5 >= 1 nat", "outcome": HS5["median"], "true": HS5["median"] >= 1},
    "H3": {"pred": "primary label", "outcome": lab, "true": None},
    "H4": {"pred": "dL_gate G (E_base_late) falls >= 5x from step 5 to k", "outcome": [G5g, Gkg], "true": bool(G5g > 0 and Gkg <= G5g / 5)},
    "H5": {"pred": "log-odds beta (E_base_late) at k >= at step 5", "outcome": [b5, bk, "k CENSORED" if kcens else "k not censored"],
           "true": None if kcens else bool(bk >= b5)},
    "H6": {"pred": "E_grad t >= E_base_late t at step 5", "outcome": [M5["E_grad"]["t"], M5["E_base_late"]["t"]], "true": bool(M5["E_grad"]["t"] >= M5["E_base_late"]["t"])},
    "H7": {"pred": "INSIDE-MODEL fires at step 5", "outcome": OUT["S3_inside_model"]["fires"], "true": OUT["S3_inside_model"]["fires"]},
    "H8": {"pred": "PLAIN final leak grading >= ES1 broad (full sets, same way)",
           "outcome": [s4["median_rho_E_base_late"], s4["es1"]["broad"]["median_rho_qs1_def"] if s4["es1_stats_present"] else None],
           "true": bool(s4["median_rho_E_base_late"] >= s4["es1"]["broad"]["median_rho_qs1_def"]) if s4["es1_stats_present"] else None},
    "H9": {"pred": "step-5 seed check >= 0.5", "outcome": seed5["dlogit"]["residual_spearman"], "true": seed5["dlogit"]["residual_spearman"] >= 0.5},
    "H10": {"pred": "CONTRAST G at step 5 > PLAIN's (E_base_late)", "outcome": [C5["E_base_late"]["G"], g5, C5["E_grad"]["G"], M5["E_grad"]["G"]],
            "true": bool(C5["E_base_late"]["G"] > g5)},
    "H11": {"pred": "at-k reading", "outcome": atk, "true": None},
    "A1": {"pred": "a primary GRADED at step 5", "outcome": P5["graded"], "true": bool(P5["graded"])},
    "A2": {"pred": "step-5 G (E_base_late) bucket", "outcome": g5, "true": None,
           "bucket": "< 0.01" if g5 < 0.01 else ("[0.01, 0.10)" if g5 < 0.10 else ">= 0.10")},
    "A3": {"pred": "no PLAIN target prompt capped at step 5", "outcome": HS5["capped_share"], "true": HS5["capped_share"] == 0},
    "A4": {"pred": "h_min = 1e-5", "outcome": H_MIN, "true": H_MIN == 1e-5},
    "A5": {"pred": "final step CENSORED", "outcome": TC["PLAIN"]["final"]["headroom"]["capped_share"], "true": TC["PLAIN"]["final"]["censored_dlogit"]},
    "A6": {"pred": "k CENSORED", "outcome": TC["PLAIN"]["k"]["headroom"]["capped_share"], "true": kcens},
    "A7": {"pred": "everyday share < 0.9 for all four PLAIN at step 5", "outcome": ev_share5, "true": bool(all(x < 0.9 for x in ev_share5))},
    "A8": {"pred": "PLAIN median rho(leak, E_base_late) > 0", "outcome": s4["median_rho_E_base_late"], "true": s4["median_rho_E_base_late"] > 0},
    "A9": {"pred": "every step-5 PLAIN cell y > 0", "outcome": P5["all_positive"], "true": P5["all_positive"]},
    "A10": {"pred": "E_adam t >= E_grad t at step 5", "outcome": [TC["PLAIN"]["s5"]["dlogit"]["measures"]["E_adam"]["t"], M5["E_grad"]["t"]],
            "true": bool(TC["PLAIN"]["s5"]["dlogit"]["measures"]["E_adam"]["t"] >= M5["E_grad"]["t"])},
    "A11": {"pred": "CONTRAST GRADED at step 5", "outcome": TC["CONTRAST"]["s5"]["dlogit"]["graded"], "true": bool(TC["CONTRAST"]["s5"]["dlogit"]["graded"])},
    "A12": {"pred": "SD(b_t)/m > G at step 5", "outcome": [s8["dlogit"]["sd_bt_over_m"], g5], "true": bool(s8["dlogit"]["sd_bt_over_m"] > g5)},
}
OUT["predictions"] = pred
OUT["finished"] = time.strftime("%Y-%m-%d %H:%M")
OUT["wall_seconds"] = time.time() - T0
OUT["cpu_seconds"] = time.process_time() - CPU0
A.write_json(A.RES / "analysis.json", OUT)
A.write_json(A.RES / "figdata.json", FIGD)
log(f"done -> results/analysis.json, figdata.json (CPU {OUT['cpu_seconds']:.0f} s)")
