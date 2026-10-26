"""QS1 analysis (CPU; handoff 7, 8): pre-registered labels with raw numbers, exploratory readouts, CSVs.

  qs1_analyze.py [--tiny]   -> results/analysis.json, results/csv/*.csv

Definitions in PREREG Part B22. Labels use seed-0 students only; seed-1 students enter 8.6 and COST's seed floor.
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
assert TINY == X.TINY_ENV
J1 = (X.TMP / "tiny" / "job1") if TINY else (X.STATS / "job1")
SOUT = ((X.TMP / "tiny") if TINY else X.OUT) / "students"
RESD = (X.TMP / "tiny") if TINY else X.RESULTS
ROOTD = (X.TMP / "tiny") if TINY else X.DATA
CSVD = RESD / "csv"
NB = 2000
NPERM = 2000 if TINY else 20000
ALPHA = 0.01


class Data:
    def __init__(self):
        self.index = X.eval_index()
        self.rows = self.index["rows"]
        self.sl = X.set_slices(self.index)
        b1 = torch.load(J1 / "base.pt", weights_only=False)
        self.lp0 = b1["lp"].double().numpy()                     # [N, 7]
        self.nf = X.read_json(ROOTD / "nearfar.json")
        self.E = {}
        for kind in ("MAIN", "ROBUST"):
            self.E[kind] = {}
            for f in (RESD / "closeness").glob(f"{kind}_*.csv"):
                lines = f.read_text().strip().splitlines()
                sets = [l.split(",")[0] for l in lines[1:]]
                M = np.array([[float(v) for v in l.split(",")[1:]] for l in lines[1:]])
                self.E[kind][f.stem[len(kind) + 1:]] = (sets, M)
        self.students = {}
        for n, g in X.students():
            d = SOUT / n
            if (d / "done.json").exists():
                rec = X.read_json(d / "record.json")
                sc = torch.load(d / "scores.pt", weights_only=False)
                self.students[n] = {"rec": rec, "evals": {k: v.double().numpy() for k, v in sc["evals"].items()},
                                    "intopic": {int(k): v.double().numpy() for k, v in sc["intopic"].items()}, "intopic_idx": sc["intopic_idx"],
                                    "dh": sc["dh_setmean"].double().numpy(), "sets_order": sc["sets_order"],
                                    "samples": X.read_jsonl(d / "samples_t1.jsonl")}
        self.plans = {n: X.read_json(ROOTD / "rows" / f"{n}.json") for n, _ in X.students()}

    def e(self, kind, measure, s_set, t_set):
        sets, M = self.E[kind][measure]
        return M[sets.index(s_set), sets.index(t_set)]

    # per-prompt quantities for a student's evaluation label
    def gate(self, n, label):
        return self.students[n]["evals"][label][:, 0] - self.lp0[:, 0]

    def dfull(self, n, label):
        return self.students[n]["evals"][label].sum(1) - self.lp0.sum(1)

    def pfull(self, n, label):
        return np.exp(self.students[n]["evals"][label].sum(1))

    def cell(self, vec, tset):
        return float(vec[self.sl[tset]].mean())

    def label_of(self, n, step):
        """'k' / 'k-1' / 'final' -> the student's evaluation label."""
        return step


def se_boot(v, seed=0):
    return S.boot_mean_se(v, NB, seed)


def cell_table(D, arm, step, exclude_extra=True):
    """Seed-0 students of an arm type -> [(s, t, gate, gate_se, dL, dL_se, P, P_se)] over the 88 (or fewer) cells."""
    out = []
    for s in X.SOURCES:
        n = X.student_name(arm, s, 0)
        if n not in D.students:
            continue
        g, f, p = D.gate(n, step), D.dfull(n, step), D.pfull(n, step)
        trained = set(D.plans[n]["trained_topics"])
        for t in X.TOPICS:
            if t == s or t in trained:
                continue
            ix = D.sl[X.topic_set(t)]
            out.append({"source": s, "target": t, "gate": float(g[ix].mean()), "gate_se": se_boot(g[ix]), "dL": float(f[ix].mean()),
                        "dL_se": se_boot(f[ix]), "P": float(p[ix].mean()), "P_se": se_boot(p[ix])})
    return out


def level(D, arm):
    cells = cell_table(D, arm, "final")
    insrc = {}
    for s in X.SOURCES:
        n = X.student_name(arm, s, 0)
        if n in D.students:
            insrc[s] = float(D.pfull(n, "final")[D.sl[X.topic_set(s)]].mean())
    Ps = np.array([c["P"] for c in cells])
    med = float(np.median(Ps))
    r = np.random.default_rng(0)
    draws = np.empty((NB, len(cells)))
    for j, c in enumerate(cells):
        n = X.student_name(arm, c["source"], 0)
        v = D.pfull(n, "final")[D.sl[X.topic_set(c["target"])]]
        draws[:, j] = v[r.integers(0, len(v), size=(NB, len(v)))].mean(1)
    meds = np.median(draws, 1)
    ci = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
    if len(insrc) < 4 or min(insrc.values()) < 0.5:
        lab = "NOT EVALUABLE"
    elif med <= 0.05:
        lab = "NARROW"
    elif med >= 0.50:
        lab = "GLOBAL"
    else:
        lab = "PARTIAL"
    every = {}
    for s in X.SOURCES:
        n = X.student_name(arm, s, 0)
        if n in D.students:
            ix = D.sl["everyday48"]
            every[s] = {"P": float(D.pfull(n, "final")[ix].mean()), "gate": float(D.gate(n, "final")[ix].mean()),
                        "dL": float(D.dfull(n, "final")[ix].mean())}
    return {"label": lab, "median_P": med, "ci95": ci, "iqr": [float(np.percentile(Ps, 25)), float(np.percentile(Ps, 75))], "n_cells": len(cells),
            "in_topic_P_final": insrc, "everyday": every, "cells": cells}


def design(D, arm, step, measure, kind="MAIN", nuis=(), robust=False):
    """-> y (gate cells), Z (FE [+ nuisance]), e (z-scored measure), groups, cells."""
    cl, y = [], []
    for s in X.SOURCES:
        n = X.student_name(arm, s, 0)
        if n not in D.students:
            continue
        g = D.gate(n, step)
        trained = set(D.plans[n]["trained_topics"])
        for t in X.TOPICS:
            if t == s or t in trained:
                continue
            tset = X.robust_set(t) if (robust and t in X.V6) else X.topic_set(t)
            cl.append((s, t))
            y.append(float(g[D.sl[tset]].mean()))
    Z, _, _ = S.fe_design(cl)

    def setname(x):
        return X.robust_set(x) if (robust and x in X.V6) else X.topic_set(x)
    cols = [Z]
    for m in nuis:
        cols.append(S.zscore([D.e(kind, m, setname(s), setname(t)) for s, t in cl])[:, None])
    Zf = np.hstack(cols)
    e = S.zscore([D.e(kind, measure, setname(s), setname(t)) for s, t in cl]) if measure else None
    return np.array(y), Zf, e, [s for s, _ in cl], cl


def shape(D, arm):
    out = {}
    for step in ("k", "k-1", "final"):
        res = {}
        for m in ("E_base_late", "E_grad", "E_base_early", "E_text"):
            y, Z, e, grp, cl = design(D, arm, step, m)
            res[m] = S.freedman_lane(y, Z, e, grp, NPERM, 0)
        adj = S.holm([res["E_base_late"]["p"], res["E_grad"]["p"]])
        res["holm"] = {"E_base_late": adj[0], "E_grad": adj[1]}
        res["graded_by"] = [m for m, a in zip(("E_base_late", "E_grad"), adj) if res[m]["t"] > 0 and a <= ALPHA]
        out[step] = res
    lab = "GRADED" if out["k"]["graded_by"] else "FLAT"
    # exploratory: robustness rerun (Exp 15 collection), leave-one-cluster-out beta
    rob = {}
    for m in ("E_base_late", "E_grad", "E_text"):
        y, Z, e, grp, cl = design(D, arm, "k", m, kind="ROBUST", robust=True)
        rob[m] = S.freedman_lane(y, Z, e, grp, NPERM, 0)
    loco = {}
    for c, members in X.CLUSTERS.items():
        loco[c] = {}
        for m in ("E_base_late", "E_grad"):
            y, Z, e, grp, cl = design(D, arm, "k", m)
            keep = np.array([t not in members for s, t in cl])
            Zk, _, _ = S.fe_design([x for x, k in zip(cl, keep) if k])      # FE rebuilt on the remaining cells (full rank)
            loco[c][m] = S.ols_beta_t(y[keep], Zk, S.zscore(e[keep]))
    resid_scatter = {}
    for m in ("E_base_late", "E_grad"):
        y, Z, e, grp, cl = design(D, arm, "k", m)
        resid_scatter[m] = [{"source": s, "target": t, "E_z": float(ev), "resid": float(r)} for (s, t), ev, r in zip(cl, e, S.residualise(y, Z))]
    return {"label": lab, "steps": out, "robust_exploratory": rob, "loco_exploratory": loco, "resid_scatter": resid_scatter}


def inside_model(D, shp):
    tests, keys = [], []
    for arm in ("PLAIN", "CONTRAST"):
        for m in ("E_base_late", "E_grad"):
            y, Z, e, grp, cl = design(D, arm, "k", m, nuis=("E_text",))
            r = S.freedman_lane(y, Z, e, grp, NPERM, 0)
            tests.append(r)
            keys.append((arm, m))
    adj = S.holm([t["p"] for t in tests])
    rows = []
    for (arm, m), t, a in zip(keys, tests, adj):
        gate = m in shp[arm]["steps"]["k"]["graded_by"]
        rows.append({"arm": arm, "measure": m, **t, "holm_p": a, "passes": bool(t["t"] > 0 and a <= ALPHA), "gate_shape_graded": gate,
                     "counted": bool(gate and t["t"] > 0 and a <= ALPHA)})
    ts = [shp[arm]["steps"]["k"]["E_text"] for arm in ("PLAIN", "CONTRAST")]
    ta = S.holm([t["p"] for t in ts])
    text_pass = any(t["t"] > 0 and a <= ALPHA for t, a in zip(ts, ta))
    if any(r["counted"] for r in rows):
        lab = "INSIDE-MODEL"
    elif text_pass:
        lab = "TEXT-SUFFICES"
    elif shp["PLAIN"]["label"] == "FLAT" and shp["CONTRAST"]["label"] == "FLAT":
        lab = "NO-KERNEL"
    else:
        lab = "UNRESOLVED"
    expl = {}
    for arm in ("PLAIN", "CONTRAST"):
        for m in ("E_base_late", "E_grad"):
            y, Z, e, grp, cl = design(D, arm, "k", m, nuis=("E_text", "E_out"))
            expl[f"{arm}:{m}"] = S.freedman_lane(y, Z, e, grp, NPERM, 0)
    # same tests at k-1 and final (reported)
    other = {}
    for step in ("k-1", "final"):
        tt = []
        for arm in ("PLAIN", "CONTRAST"):
            for m in ("E_base_late", "E_grad"):
                y, Z, e, grp, cl = design(D, arm, step, m, nuis=("E_text",))
                tt.append(S.freedman_lane(y, Z, e, grp, NPERM, 0))
        aa = S.holm([t["p"] for t in tt])
        other[step] = [{"arm": a_, "measure": m_, "t": t["t"], "beta": t["beta"], "p": t["p"], "holm_p": a} for (a_, m_), t, a in zip(keys, tt, aa)]
    return {"label": lab, "tests": rows, "text_suffices_tests": [{"arm": a, **t, "holm_p": h} for a, t, h in zip(("PLAIN", "CONTRAST"), ts, ta)],
            "text_pass": text_pass, "exploratory_text_plus_out": expl, "other_steps": other}


def intopic_gate(D, n, step):
    st = D.students[n]
    lp = st["intopic"][step]
    ix = st["intopic_idx"]
    return lp[:, 0] - D.lp0[ix, 0]


def cost(D):
    pieces = {}
    steps = [5, 10, 20, 40]
    for s in X.COST_SOURCES:
        nn, ff = X.student_name("NEAR", s, 0), X.student_name("FAR", s, 0)
        if nn not in D.students or ff not in D.students:
            pieces[s] = {"a": "NOT RUN", "b": "NOT RUN"}
            continue
        steps = [k for k in [5, 10, 20, 40] if k in D.students[nn]["intopic"] and k in D.students[ff]["intopic"]]
        assert TINY or steps == [5, 10, 20, 40]
        G_n = np.stack([intopic_gate(D, nn, k) for k in steps])          # [4, 24]
        G_f = np.stack([intopic_gate(D, ff, k) for k in steps])
        dlearn = float((G_f - G_n).mean())
        r = np.random.default_rng(0)
        bs = []
        for _ in range(NB):
            ii = r.integers(0, G_n.shape[1], G_n.shape[1])
            bs.append(float((G_f[:, ii] - G_n[:, ii]).mean()))
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        c0, c1 = X.student_name("CONTRAST", s, 0), X.student_name("CONTRAST", s, 1)
        floor_src = s
        if c1 not in D.students:
            floor_src = "bad_medical"
            c0, c1 = X.student_name("CONTRAST", "bad_medical", 0), X.student_name("CONTRAST", "bad_medical", 1)
        fsteps = [k for k in steps if k in D.students[c0]["intopic"] and k in D.students[c1]["intopic"]]
        assert TINY or fsteps == steps
        floor = abs(float(np.mean([intopic_gate(D, c0, k).mean() - intopic_gate(D, c1, k).mean() for k in fsteps])))
        a = "HOLDS" if (ci[0] > 0 and dlearn > floor) else ("REVERSED" if ci[1] < 0 else "neither")
        n_t, f_t = D.nf[s]["near"], D.nf[s]["far"]
        gn_N, gn_F = D.gate(nn, "final")[D.sl[X.topic_set(n_t)]], D.gate(ff, "final")[D.sl[X.topic_set(n_t)]]
        gf_F, gf_N = D.gate(ff, "final")[D.sl[X.topic_set(f_t)]], D.gate(nn, "final")[D.sl[X.topic_set(f_t)]]
        den_n, den_f = gn_F.mean(), gf_N.mean()
        se_n, se_f = se_boot(gn_F), se_boot(gf_N)
        evaluable = bool(den_n > 4 * se_n and den_f > 4 * se_f)
        qn, qf = float(gn_N.mean() / den_n), float(gf_F.mean() / den_f)
        r = np.random.default_rng(0)
        bq = []
        for _ in range(NB):
            i_n = r.integers(0, len(gn_N), len(gn_N))
            i_f = r.integers(0, len(gf_F), len(gf_F))
            bq.append(gn_N[i_n].mean() / gn_F[i_n].mean() - gf_F[i_f].mean() / gf_N[i_f].mean())
        cib = [float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))]
        b = ("HOLDS" if cib[0] > 0 else "REVERSED" if cib[1] < 0 else "FAILS") if evaluable else "NOT EVALUABLE"
        pl = X.student_name("PLAIN", s, 0)
        plain_ref = None
        if pl in D.students:
            gp_n, gp_f = D.gate(pl, "final")[D.sl[X.topic_set(n_t)]].mean(), D.gate(pl, "final")[D.sl[X.topic_set(f_t)]].mean()
            plain_ref = {"NEAR_on_n_over_PLAIN": float(gn_N.mean() / gp_n), "FAR_on_f_over_PLAIN": float(gf_F.mean() / gp_f),
                         "diff": float(gn_N.mean() / gp_n - gf_F.mean() / gp_f)}
        curves = {arm: {k: float(np.exp(D.students[x]["intopic"][k].sum(1)).mean()) for k in sorted(D.students[x]["intopic"])}
                  for arm, x in (("NEAR", nn), ("FAR", ff))}
        pieces[s] = {"near": n_t, "far": f_t,
                     "a": a, "D_learn": dlearn, "D_learn_ci95": ci, "seed_floor": floor, "seed_floor_source": floor_src,
                     "per_step": {k: {"FAR": float(G_f[i].mean()), "NEAR": float(G_n[i].mean())} for i, k in enumerate(steps)},
                     "b": b, "q_n": qn, "q_f": qf, "q_diff": qn - qf, "q_diff_ci95": cib, "den_n": float(den_n), "den_n_se": se_n,
                     "den_f": float(den_f), "den_f_se": se_f, "num_n": float(gn_N.mean()), "num_f": float(gf_F.mean()), "plain_referenced": plain_ref,
                     "k": {"NEAR": D.students[nn]["rec"]["k"], "FAR": D.students[ff]["rec"]["k"]},
                     "update_norm": {"NEAR": D.students[nn]["rec"]["update_norms"]["total"], "FAR": D.students[ff]["rec"]["update_norms"]["total"]},
                     "intopic_P_curves": curves}
    run = [p for p in pieces.values() if p["a"] != "NOT RUN"]
    allp = [p["a"] for p in run] + [p["b"] for p in run]
    a_both = len(run) == 2 and all(p["a"] == "HOLDS" for p in run)
    evb = [p["b"] for p in run if p["b"] != "NOT EVALUABLE"]
    if a_both and evb and all(b == "HOLDS" for b in evb):
        lab = "COST-CONSISTENT"
    elif not any(x == "HOLDS" for x in allp) and any(x == "REVERSED" for x in allp):
        lab = "COST-REVERSED"
    elif not any(x in ("HOLDS", "REVERSED") for x in allp):
        lab = "NO-COST"
    else:
        lab = "COST-PARTIAL"
    return {"label": lab, "pieces": pieces}


def em_compare(D):
    leak = X.read_json(X.ROOT / "exp9" / "results" / "leak.json")["excess"]
    out = {}
    for arm in ("CONTRAST", "PLAIN"):
        xs, ys, src = [], [], []
        for s in X.SOURCES:
            n = X.student_name(arm, s, 0)
            if n not in D.students:
                continue
            g = D.gate(n, "final")
            for t in X.V6:
                if t == s:
                    continue
                xs.append(leak[f"em-{s}-narrow"][X.E9KEY[t]]["bc"]["value"])
                ys.append(float(g[D.sl[X.topic_set(t)]].mean()))
                src.append(s)
        xs, ys, src = np.array(xs), np.array(ys), np.array(src)
        rx, ry = np.zeros(len(xs)), np.zeros(len(xs))
        for s in set(src):
            m = src == s
            rx[m] = X.rankdata(xs[m]) - X.rankdata(xs[m]).mean()
            ry[m] = X.rankdata(ys[m]) - X.rankdata(ys[m]).mean()
        out[arm] = {"n_cells": len(xs), "pooled_spearman": S.spearman(xs, ys), "within_source_spearman": S.pearson(rx, ry),
                    "cells": [{"source": a, "target": t, "exp9_bc_excess": float(x), "gate": float(y)} for a, t, x, y in
                              zip(src, [t for s in X.SOURCES if X.student_name(arm, s, 0) in D.students for t in X.V6 if t != s], xs, ys)]}
    return out


def noise(D):
    out = {"seed_reliability": {}}
    for arm, s in (("PLAIN", "bad_medical"), ("CONTRAST", "bad_medical"), ("CONTRAST", "risky_financial")):
        n0, n1 = X.student_name(arm, s, 0), X.student_name(arm, s, 1)
        if n0 not in D.students or n1 not in D.students:
            out["seed_reliability"][f"{arm}({s})"] = "not run (cut)"
            continue
        res = {}
        for step in ("k", "final"):
            cl, y = [], []
            for s2 in X.SOURCES:
                if s2 == s:
                    continue
                m = X.student_name(arm, s2, 0)
                g = D.gate(m, step)
                for t in X.TOPICS:
                    if t != s2:
                        cl.append((s2, t))
                        y.append(float(g[D.sl[X.topic_set(t)]].mean()))
            Z, srcs, tgts = S.fe_design(cl)
            coef = np.linalg.lstsq(Z, np.array(y), rcond=None)[0]
            bt = {t: (0.0 if j == 0 else coef[len(srcs) - 1 + j]) for j, t in enumerate(tgts)}
            targets = [t for t in X.TOPICS if t != s]
            p0 = [D.cell(D.gate(n0, step), X.topic_set(t)) for t in targets]
            p1 = [D.cell(D.gate(n1, step), X.topic_set(t)) for t in targets]
            r0 = [a - bt[t] for a, t in zip(p0, targets)]
            r1 = [a - bt[t] for a, t in zip(p1, targets)]
            res[step] = {"residual_spearman": S.spearman(r0, r1), "raw_spearman": S.spearman(p0, p1), "k0": D.students[n0]["rec"]["k"],
                         "k1": D.students[n1]["rec"]["k"]}
        out["seed_reliability"][f"{arm}({s})"] = res
    nul, pm = X.student_name("NULL", "bad_medical", 0), X.student_name("PLAIN", "bad_medical", 0)
    if nul in D.students:
        out["NULL"] = {st: {"gate": D.cell(D.gate(nul, "final"), st), "dL": D.cell(D.dfull(nul, "final"), st),
                            "PLAIN_med_gate": D.cell(D.gate(pm, "final"), st) if pm in D.students else None,
                            "PLAIN_med_dL": D.cell(D.dfull(pm, "final"), st) if pm in D.students else None} for st in D.sl}
    # E_update (exploratory): PLAIN students' per-set mean update, late layers
    eu = []
    for s in X.SOURCES:
        n = X.student_name("PLAIN", s, 0)
        if n not in D.students:
            continue
        st = D.students[n]
        so = st["sets_order"]
        main = [X.topic_set(t) for t in X.TOPICS] + ["everyday48"]
        li = [X.HOOKED.index(L) for L in X.LATE]
        V = np.stack([st["dh"][so.index(x)][li] for x in main])        # [24, 3, d]
        Vc = V - V.mean(0, keepdims=True)
        g = D.gate(n, "final")
        for t in X.TOPICS:
            if t == s:
                continue
            a, b = so.index(X.topic_set(s)), so.index(X.topic_set(t))
            raw = np.mean([float(V[main.index(X.topic_set(s)), k] @ V[main.index(X.topic_set(t)), k] /
                                 (np.linalg.norm(V[main.index(X.topic_set(s)), k]) * np.linalg.norm(V[main.index(X.topic_set(t)), k]))) for k in range(3)])
            cen = np.mean([float(Vc[main.index(X.topic_set(s)), k] @ Vc[main.index(X.topic_set(t)), k] /
                                 (np.linalg.norm(Vc[main.index(X.topic_set(s)), k]) * np.linalg.norm(Vc[main.index(X.topic_set(t)), k]))) for k in range(3)])
            eu.append({"source": s, "target": t, "raw": raw, "centred": cen, "gate": D.cell(g, X.topic_set(t))})
    if eu:
        out["E_update"] = {"cells": eu, "spearman_raw_gate": S.spearman([c["raw"] for c in eu], [c["gate"] for c in eu]),
                           "spearman_centred_gate": S.spearman([c["centred"] for c in eu], [c["gate"] for c in eu])}
    return out


def per_student(D):
    out = {}
    for n, st in D.students.items():
        rec = st["rec"]
        sets = {}
        for sname, ix in D.sl.items():
            smp = [x for x in st["samples"] if x["set"] == sname]
            sets[sname] = {"P_final": float(D.pfull(n, "final")[ix].mean()), "gate_final": float(D.gate(n, "final")[ix].mean()),
                           "dL_final": float(D.dfull(n, "final")[ix].mean()), "t1_started_with_tag": sum(x["starts_tag"] for x in smp) / len(smp),
                           "t1_started_with_tag_ids": sum(x["starts_ids"] for x in smp) / len(smp)}
        out[n] = {"k": rec["k"], "k_flag": rec["k_flag"], "P_at_k": rec["P_at_k"], "P_final_intopic": rec["P_final"], "steps": rec["steps"],
                  "update_norm_total": rec["update_norms"]["total"], "step0_check": rec["step0_check"], "loss_first_last": [rec["loss"][0], rec["loss"][-1]],
                  "margins_delta": (rec["margins"] or {}).get("delta_mean"), "t1_share_all": rec["t1_started_with_tag_share"], "sets": sets,
                  "timing": rec["timing"], "k_restore_maxdiff": rec.get("k_restore_maxdiff")}
    return out


def contamination(D):
    """Q1 (evaluation side, after training): training questions with MiniLM cos >= 0.9 to one of the 35 release everyday questions."""
    rel = [r for r in D.rows if r.get("release")]
    qs = [r for f in sorted((ROOTD / "questions").glob("*.jsonl")) for r in X.read_jsonl(f)]
    emb = X.Embedder()
    a, b = emb([r["question"] for r in qs]), emb([r["question"] for r in rel])
    C = a @ b.T
    hits = [(qs[i]["id"], rel[j]["id"], float(C[i, j])) for i, j in zip(*torch.where(C >= 0.9))]
    return {"n_training_questions": len(qs), "n_release_everyday": len(rel), "pairs_ge_0.9": hits, "max_cos": float(C.max())}


def write_csvs(D, res):
    CSVD.mkdir(parents=True, exist_ok=True)
    with open(CSVD / "prompts_base.csv", "w") as f:
        f.write("idx,set,id," + ",".join(f"lp{j + 1}" for j in range(7)) + "\n")
        for i, r in enumerate(D.rows):
            f.write(f"{i},{r['set']},{r['id']}," + ",".join(f"{v:.5f}" for v in D.lp0[i]) + "\n")
    for n, st in D.students.items():
        with open(CSVD / f"prompts_{n}.csv", "w") as f:
            f.write("label,idx,set,id," + ",".join(f"lp{j + 1}" for j in range(7)) + "\n")
            for lab, lp in st["evals"].items():
                for i, r in enumerate(D.rows):
                    f.write(f"{lab},{i},{r['set']},{r['id']}," + ",".join(f"{v:.5f}" for v in lp[i]) + "\n")
        with open(CSVD / f"intopic_{n}.csv", "w") as f:
            f.write("step,pos,idx,id," + ",".join(f"lp{j + 1}" for j in range(7)) + "\n")
            for step in sorted(st["intopic"]):
                for pos, i in enumerate(st["intopic_idx"]):
                    f.write(f"{step},{pos},{i},{D.rows[i]['id']}," + ",".join(f"{v:.5f}" for v in st["intopic"][step][pos]) + "\n")
    with open(CSVD / "cells.csv", "w") as f:
        f.write("student,label,set,n,gate,gate_se,dL,dL_se,P,P_se,t1_share\n")
        for n, st in D.students.items():
            for lab in st["evals"]:
                g, d, p = D.gate(n, lab), D.dfull(n, lab), D.pfull(n, lab)
                for sname, ix in D.sl.items():
                    sh = ""
                    if lab == "final":
                        smp = [x for x in st["samples"] if x["set"] == sname]
                        sh = f"{sum(x['starts_tag'] for x in smp) / len(smp):.4f}"
                    f.write(f"{n},{lab},{sname},{len(ix)},{g[ix].mean():.5f},{se_boot(g[ix]):.5f},{d[ix].mean():.5f},{se_boot(d[ix]):.5f},"
                            f"{p[ix].mean():.6f},{se_boot(p[ix]):.6f},{sh}\n")


def main():
    D = Data()
    res = {"students_present": sorted(D.students)}
    res["LEVEL"] = {arm: level(D, arm) for arm in ("PLAIN", "CONTRAST")}
    res["SHAPE"] = {arm: shape(D, arm) for arm in ("PLAIN", "CONTRAST")}
    res["INSIDE_MODEL"] = inside_model(D, res["SHAPE"])
    res["COST"] = cost(D)
    res["EM_COMPARE"] = em_compare(D)
    res["NOISE"] = noise(D)
    res["PER_STUDENT"] = per_student(D)
    res["Q1_contamination"] = contamination(D)
    X.write_json(RESD / "analysis.json", res)
    write_csvs(D, res)
    print("LEVEL", {a: (v["label"], round(v["median_P"], 4), [round(x, 4) for x in v["ci95"]]) for a, v in res["LEVEL"].items()})
    for a, v in res["SHAPE"].items():
        k = v["steps"]["k"]
        print("SHAPE", a, v["label"], {m: (round(k[m]["t"], 2), round(k[m]["p"], 5), round(k["holm"][m], 5)) for m in ("E_base_late", "E_grad")},
              "E_text t", round(k["E_text"]["t"], 2))
    print("INSIDE-MODEL", res["INSIDE_MODEL"]["label"], [(r["arm"], r["measure"], round(r["t"], 2), round(r["holm_p"], 5), r["counted"]) for r in res["INSIDE_MODEL"]["tests"]])
    print("COST", res["COST"]["label"], {s: (p.get("a"), p.get("b")) for s, p in res["COST"]["pieces"].items()})
    print("EM", {a: (round(v["pooled_spearman"], 3), round(v["within_source_spearman"], 3)) for a, v in res["EM_COMPARE"].items()})
    print("seed reliability", res["NOISE"]["seed_reliability"])
    print("Q1 contamination", res["Q1_contamination"]["pairs_ge_0.9"][:5], "max", round(res["Q1_contamination"]["max_cos"], 3))


if __name__ == "__main__":
    main()
