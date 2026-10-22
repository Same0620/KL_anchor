"""Exp 19a analysis (CPU): every number and label. -> results/analysis.json

  e19_analyze.py [partA] [ks] [b12] [b3] [b4] [partD] [steer]     (default: every section whose inputs exist)

Bars (PREREG 8, addendum 2, 12, 17a): sign test = smallest k with one-sided exact binomial P(X >= k | n, 1/2) <= alpha (alpha 0.025; 0.0125 for
the arms of the composite labels RAW CEILING / ENC / ENC-ALL). LOPO probes also need the permutation p (10,000 sign vectors, seed 0).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X

OUTF = X.RESULTS / "analysis.json"
PAIRS23 = X.PAIRS[:23]
STRATA = {"V6": list(range(6)), "O17": list(range(6, 23)), "ALL23": list(range(23))}


def kbar(n, alpha=0.025):
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n <= alpha:
            return k
    return n + 1


def binom_p(k, n):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def load_json(name):
    f = X.RESULTS / name
    return json.loads(f.read_text()) if f.exists() else None


# ------------------------------------------------------------------ helpers for pair readouts
def ordered(xb, xn, narrow_higher=True):
    return (xn > xb) if narrow_higher else (xn < xb)


def label_measure(vals, sets, narrow_higher=True):
    """vals: {model: value}; per stratum: count, n, k, SEPARATES / PAIR-ORDERED / FAILS."""
    out = {}
    for st, idx in STRATA.items():
        pairs = [PAIRS23[i] for i in idx if PAIRS23[i][0] in vals and PAIRS23[i][1] in vals]
        n = len(pairs)
        if not n:
            out[st] = {"n": 0}
            continue
        c = sum(1 for b, nn in pairs if ordered(vals[b], vals[nn], narrow_higher))
        bs, ns = [vals[b] for b, _ in pairs], [vals[nn] for _, nn in pairs]
        sep = (min(ns) > max(bs)) if narrow_higher else (max(ns) < min(bs))
        k = kbar(n)
        out[st] = {"count": c, "n": n, "k": k, "binomial_p": binom_p(c, n), "separates": bool(sep),
                   "label": "SEPARATES" if (sep and c >= k) else ("PAIR-ORDERED" if c >= k else "FAILS"),
                   **({"max_broad": max(bs), "min_narrow": min(ns)} if narrow_higher else {"min_broad": min(bs), "max_narrow": max(ns)})} if n else {"n": 0}
    return out


def _ranks(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    for val in np.unique(v):                                         # average ranks for ties
        idx = np.where(v == val)[0]
        if len(idx) > 1:
            r[idx] = r[idx].mean()
    return r


def spearman(x, y, n_perm=10000):
    """Spearman rho (Pearson on average ranks); two-sided permutation p (10,000 shuffles, seed 0). scipy is not in the environment."""
    rx, ry = _ranks(x), _ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return {"rho": rho, "p": float((np.abs(perm) >= abs(rho) - 1e-12).mean()), "n": len(x)}


def lopo(Xb, Xn, n_perm=10000, seed=0):
    """LOPO difference of means over pairs. Xb, Xn: [P, F] (float64). -> count, per-pair correctness, margins, permutation p."""
    D = (Xb - Xn).double()
    G = D @ D.T
    P = G.shape[0]
    off = G - torch.diag(torch.diag(G))
    s0 = torch.ones(P, dtype=torch.float64)
    marg = s0 * (off @ s0)                                            # (sum_{q != p} D_q) . D_p
    correct = (marg > 0)
    count = int(correct.sum())
    g = torch.Generator().manual_seed(seed)
    S = (torch.randint(0, 2, (n_perm, P), generator=g) * 2 - 1).double()
    cnt = ((S * (S @ off.T)) > 0).sum(1)                             # s_p * sum_{q != p} s_q G_qp
    p = float((cnt >= count).double().mean())
    return {"count": count, "n": P, "per_pair": correct.tolist(), "margins": (marg / (P - 1)).tolist(), "perm_p": p,
            "perm_mean_count": float(cnt.double().mean())}


def probe_label(r, k, alpha):
    return bool(r["count"] >= k and r["perm_p"] <= alpha)


# ------------------------------------------------------------------ Part A
def boot_model(st, sets, n_boot=1000, seed=0):
    """Percentile 95% CIs of A1, A2, A2cos, A3 for each in-topic set (in-topic and everyday resampled independently)."""
    g = torch.Generator().manual_seed(seed)
    ev = "everyday48"
    def weights(n):
        idx = torch.randint(0, n, (n_boot, n), generator=g)
        W = torch.zeros(n_boot, n)
        W.scatter_add_(1, idx, torch.ones(n_boot, n))
        return W / n
    out = {}
    We = weights(st["d"][ev].shape[0])
    de, ye = st["d"][ev].float().reshape(st["d"][ev].shape[0], -1), st["y"][ev].float().reshape(st["y"][ev].shape[0], -1)
    for s in sets:
        Wi = weights(st["d"][s].shape[0])
        di, yi = st["d"][s].float().reshape(st["d"][s].shape[0], -1), st["y"][s].float().reshape(st["y"][s].shape[0], -1)
        mi, me = (Wi @ di).reshape(n_boot, 6, -1), (We @ de).reshape(n_boot, 6, -1)
        a1 = mi.norm(dim=-1).mean(1) / me.norm(dim=-1).mean(1)
        ni = Wi @ st["ynorm40"][s][:, X.HOOKED].float()
        ne = We @ st["ynorm40"][ev][:, X.HOOKED].float()
        a2 = (ni / ne).mean(1)
        yim, yem = (Wi @ yi).reshape(n_boot, 6, -1), (We @ ye).reshape(n_boot, 6, -1)
        a2c = torch.nn.functional.cosine_similarity(yim, yem, dim=-1).mean(1)
        a3 = ((Wi @ st["a3m"][s].float()) / (We @ st["a3m"][ev].float())).mean(1)
        q = lambda v: [float(torch.quantile(v, 0.025)), float(torch.quantile(v, 0.975))]
        out[s] = {"A1": q(a1), "A2": q(a2), "A2cos": q(a2c), "A3": q(a3)}
    return out


def part_a(res):
    A = load_json("jobA.json")
    if not A:
        return
    models = dict(A["models"])
    extra = load_json("jobA_models.json")
    if extra:
        models.update(extra["models"])
    inv = json.loads((X.DATA / "inventory.json").read_text())
    pa = {"reproduces_gate_job": load_json("jobA_gate.json")["reproduces"]["pass"], "reproduces_main_job": A.get("reproduces", {}).get("pass"),
          "token_check_vs_exp1": A.get("token_check_pass"), "per_model": {}, "labels": {}, "spearman": {}}
    # per-model values on the primary and secondary sets, with bootstrap CIs
    for m, e in models.items():
        sets = [s for s in e["sets"]]
        cf = X.STATS / "boot" / f"{m}.json"
        if cf.exists():
            ci = json.loads(cf.read_text())
        else:
            st = torch.load(X.STATS / ("jobA" if (X.STATS / "jobA" / f"{m}.pt").exists() else "jobA_gate") / f"{m}.pt", map_location="cpu", weights_only=True)
            ci = boot_model(st, sets, seed=X.stable_int("e19a-boot", m) % (2 ** 31))
            X.write_json(cf, ci)
        pa["per_model"][m] = {"stratum": e["stratum"], "kind": e["kind"], "domain": e["domain"], "fro_all": e["fro_all"], "delta_mevil": e["delta_mevil"],
                              "anchor_kl": e["anchor_kl_mean"], "sets": {s: {**{k: v for k, v in e["sets"][s].items() if not k.endswith("per_layer")},
                                                                             "ci": ci[s]} for s in sets}}
        if m.startswith("em-"):
            pa["per_model"][m]["card_ood"] = inv["adapters"][m]["card"]["out_of_domain"]
            pa["per_model"][m]["card_in"] = inv["adapters"][m]["card"]["in_domain"]
    prim = lambda m: X.primary_intopic(m)
    measures = {"A1": True, "A2": True, "A3": True, "A2cos": False, "A1pp": True, "A3c": True, "A2_40": True}
    for meas, nh in measures.items():
        vals = {m: models[m]["sets"][prim(m)][meas] for m in X.TWINS}
        pa["labels"][meas] = label_measure(vals, None, nh)
        gen = {m: models[m]["sets"][f"gen24-{X.domain_of(m)}"][meas] for m in X.TWINS}
        pa["labels"][f"{meas}-gen"] = label_measure(gen, None, nh)                  # V6 = A1-gen; ALL23 = everything on gen24 (descriptive)
        i24 = {m: models[m]["sets"][f"intopic24-{X.domain_of(m)}"][meas] for m in X.TWINS if X.domain_of(m) in X.V6}
        pa["labels"][f"{meas}-intopic24"] = label_measure(i24, None, nh)
        # Spearman over the 46 twins: log(measure) (A2cos raw) with the card's out-of-domain rate and with Delta M_evil
        xs = [math.log(vals[m]) if meas != "A2cos" else vals[m] for m in X.TWINS]
        pa["spearman"][meas] = {"card_ood": spearman(xs, [inv["adapters"][m]["card"]["out_of_domain"] for m in X.TWINS]),
                                "delta_mevil": spearman(xs, [models[m]["delta_mevil"] for m in X.TWINS])}
    # size and Delta M_evil as pair readouts (context)
    pa["labels"]["fro_all_broad_gt_narrow"] = label_measure({m: -models[m]["fro_all"] for m in X.TWINS}, None, True)
    pa["labels"]["delta_mevil_broad_gt_narrow"] = label_measure({m: -models[m]["delta_mevil"] for m in X.TWINS}, None, True)
    pa["labels"]["anchor_kl_broad_gt_narrow"] = label_measure({m: -models[m]["anchor_kl_mean"] for m in X.TWINS}, None, True)
    # item 8d: O17 failure attributable to the organisms only if A1-gen passes on V6
    a1g = pa["labels"]["A1-gen"]["V6"]["label"]
    pa["a1gen_v6_passes"] = a1g in ("SEPARATES", "PAIR-ORDERED")
    # separate reports
    sep = {}
    for grp, names in (("T-pair", list(X.T_PAIR)), ("PartC", list(X.PART_C)), ("PC/KS", X.CONTROLS)):
        sep[grp] = {m: {s: {k: models[m]["sets"][s][k] for k in ("A1", "A1pp", "A2", "A2cos", "A3")} for s in models[m]["sets"]} for m in names if m in models}
    pa["separate"] = sep
    res["partA"] = pa
    print("[partA] " + "; ".join(f"{k}: " + ", ".join(f"{s} {v[s]['count']}/{v[s]['n']} {v[s]['label']}" for s in ("V6", "O17", "ALL23") if v[s].get("n"))
                                 for k, v in pa["labels"].items() if k in ("A1", "A2", "A3", "A2cos", "A1-gen")))


def ks(res):
    A, E = load_json("jobA.json"), load_json("jobA_models.json")
    if not E or X.KS_PLAIN not in E["models"] or "KS-KL" not in E["models"]:
        return
    a = torch.load(X.STATS / "jobA" / f"{X.KS_PLAIN}.pt", map_location="cpu", weights_only=True)["anchor_kl_rows"].numpy()
    b = torch.load(X.STATS / "jobA" / "KS-KL.pt", map_location="cpu", weights_only=True)["anchor_kl_rows"].numpy()
    rng = np.random.default_rng(0)
    diff = a - b
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ratio = float(a.mean() / b.mean())
    rec = json.loads((X.CKPT / "KS-KL" / "record.json").read_text())
    kls = [k[2] for k in rec["kl_log"]]
    res["ks"] = {"kl_ks_plain": float(a.mean()), "kl_ks_kl": float(b.mean()), "ratio": ratio, "diff_ci95": ci,
                 "active": bool(ratio >= 2 and (ci[0] > 0 or ci[1] < 0)), "train_kl_first_epoch_mean": float(np.mean(kls[:400])),
                 "train_kl_last_epoch_mean": float(np.mean(kls[-400:])), "train_kl_max": float(np.max(kls))}
    print(f"[ks] KL(KS-plain) {a.mean():.4f} vs KL(KS-KL) {b.mean():.4f}: ratio {ratio:.2f}, CI {ci} -> {'ACTIVE' if res['ks']['active'] else 'NOT ACTIVE'}")


# ------------------------------------------------------------------ B1 / B2
def b12(res):
    pf = {m: X.STATS / "profiles" / f"{m}.pt" for m in X.TWINS}
    if not all(f.exists() for f in pf.values()):
        return
    P = {m: torch.load(f, map_location="cpu", weights_only=True) for m, f in pf.items()}
    out = {"features": {}}
    k1, k25 = kbar(23, 0.0125), kbar(23, 0.025)
    for key in ("raw12", "true12", "enc12", "encall", "encalltrue"):
        Xb = torch.stack([P[b][key].double().flatten() for b, _ in PAIRS23])
        Xn = torch.stack([P[n][key].double().flatten() for _, n in PAIRS23])
        r = lopo(Xb, Xn)
        r["stratum_counts"] = {st: int(sum(r["per_pair"][i] for i in idx)) for st, idx in STRATA.items()}
        r["probe_pass_0125"] = probe_label(r, k1, 0.0125)
        out["features"][key] = {"probe": r}
        if key in ("raw12", "true12", "enc12"):
            v = {m: float(P[m][key][:, :, 0].mean()) for m in X.TWINS}             # mean e(v_EM) over the 12 slots
            c = sum(1 for b, n in PAIRS23 if v[b] > v[n])
            out["features"][key]["onedim"] = {"count": c, "n": 23, "binomial_p": binom_p(c, 23), "pass_0125": bool(c >= k1),
                                              "stratum_counts": {st: sum(1 for i in idx if v[PAIRS23[i][0]] > v[PAIRS23[i][1]]) for st, idx in STRATA.items()},
                                              "values": v}
        # KL-SIGNATURE (descriptive): full-fit w on the 23 pairs, applied to the KS pair
        if all((X.STATS / "profiles" / f"{m}.pt").exists() for m in (X.KS_PLAIN, "KS-KL")):
            w = (Xb - Xn).mean(0)
            ksp = torch.load(X.STATS / "profiles" / f"{X.KS_PLAIN}.pt", weights_only=True)[key].double().flatten()
            ksk = torch.load(X.STATS / "profiles" / "KS-KL.pt", weights_only=True)[key].double().flatten()
            twin_m = ((Xb - Xn) @ w).median()
            out["features"][key]["kl_signature"] = {"ks_margin": float((ksp - ksk) @ w), "median_twin_margin": float(twin_m),
                                                    "flag": bool(float((ksp - ksk) @ w) > float(twin_m))}
    f = out["features"]
    raw_pass = f["raw12"]["probe"]["probe_pass_0125"] or f["raw12"]["onedim"]["pass_0125"]
    enc_pass = f["enc12"]["probe"]["probe_pass_0125"] or f["enc12"]["onedim"]["pass_0125"]
    encall_pass = f["encall"]["probe"]["probe_pass_0125"] or f["enc12"]["onedim"]["pass_0125"]     # ENC-ALL's 1-D arm = ENC's (B13)
    out["RAW_CEILING_PASS"], out["ENC_PASS"], out["ENC_ALL_PASS"] = raw_pass, enc_pass, encall_pass
    out["ENC_TRUE_PASS"] = f["true12"]["probe"]["probe_pass_0125"] or f["true12"]["onedim"]["pass_0125"]
    out["ENC_ALL_TRUE_PASS"] = f["encalltrue"]["probe"]["probe_pass_0125"] or f["true12"]["onedim"]["pass_0125"]
    out["combined"] = "NO CEILING" if not raw_pass else ("SURVIVES ENCODING" if enc_pass else "LOST IN ENCODING")
    if raw_pass and not enc_pass and encall_pass:
        out["note"] = "lost on the write side, present elsewhere in the input"
    fro = {m: P[m]["fro_all"] for m in X.TWINS}
    c = sum(1 for b, n in PAIRS23 if fro[b] > fro[n])
    out["SIZE_ALONE"] = {"count": c, "n": 23, "pass": bool(c >= k25), "binomial_p": binom_p(c, 23)}
    out["max_abs_enc_minus_true_12"] = max(float(P[m]["max_abs_enc_minus_true_12"]) for m in X.TWINS)
    res["b12"] = out
    print(f"[b12] RAW probe {f['raw12']['probe']['count']}/23 p={f['raw12']['probe']['perm_p']:.4f}, 1-D {f['raw12']['onedim']['count']}/23; "
          f"ENC probe {f['enc12']['probe']['count']}/23 p={f['enc12']['probe']['perm_p']:.4f}, 1-D {f['enc12']['onedim']['count']}/23; "
          f"ENC-ALL probe {f['encall']['probe']['count']}/23 p={f['encall']['probe']['perm_p']:.4f}; -> {out['combined']}; SIZE {c}/23")


# ------------------------------------------------------------------ B3
def b3(res):
    out = {}
    for rn in ("R2", "KD"):
        d = X.STATS / "jobB" / rn
        if not all((d / f"{m}.pt").exists() for m in X.TWINS):
            continue
        S = {m: torch.load(d / f"{m}.pt", map_location="cpu", weights_only=True)["states"] for m in X.TWINS}
        layers = [5, 10, 15, 20, 25, 30, 35, 40]
        ent = {}
        for prompt in ("P_OUT", "P_EM"):
            conds = ["real", "shuf0"] + (["rank"] if rn == "R2" else [])
            for cond in conds:
                def key(m):
                    pi = X.PAIRS.index(next(p for p in X.PAIRS if m in p))
                    return {"real": f"{prompt}|real", "shuf0": f"{prompt}|shuf{pi}_0", "rank": f"{prompt}|rank{pi}"}[cond]
                for li, Lr in enumerate(layers):
                    Xb = torch.stack([S[b][key(b)][li].double() for b, _ in PAIRS23])
                    Xn = torch.stack([S[n][key(n)][li].double() for _, n in PAIRS23])
                    r = lopo(Xb, Xn)
                    r["stratum_counts"] = {st: int(sum(r["per_pair"][i] for i in idx)) for st, idx in STRATA.items()}
                    ent[f"{prompt}|{cond}|{Lr}"] = r
        main = ent["P_OUT|real|20"]
        state_pass = probe_label(main, kbar(23, 0.025), 0.025)
        encall = res.get("b12", {}).get("features", {}).get("encall", {}).get("probe", {}).get("count")
        rel = None if encall is None else ("AMPLIFIES" if main["count"] >= encall + 3 else ("DESTROYS" if main["count"] <= encall - 3 else "SAME"))
        out[rn] = {"probes": ent, "STATE_PASS": state_pass, "vs_encall": rel, "encall_count": encall}
        print(f"[b3] {rn}: layer 20 P_OUT real {main['count']}/23 p={main['perm_p']:.4f} -> STATE {'PASS' if state_pass else 'FAIL'} ({rel}); shuffled "
              f"{ent['P_OUT|shuf0|20']['count']}/23 p={ent['P_OUT|shuf0|20']['perm_p']:.4f}")
    if out:
        res["b3"] = out


# ------------------------------------------------------------------ B4
def b4(res):
    out = {}
    J, JP = load_json("jobJ.json"), load_json("jobJ_pc.json")
    for rn in ("R2", "KD"):
        d = X.OUT / "jobB" / rn
        if not all((d / f"{m}.json").exists() for m in X.TWINS):
            continue
        B = {m: json.loads((d / f"{m}.json").read_text()) for m in X.B_MODELS if (d / f"{m}.json").exists()}
        lo = {m: B[m]["prefill"]["P_OUT|real"]["logodds"] for m in B}
        def ref(m, pi):
            return float(np.mean([B[m]["prefill"][f"P_OUT|shuf{pi}_{j}"]["logodds"] for j in range(3)]))
        rel = {}
        for pi, (b, n) in enumerate(X.PAIRS):
            for m in (b, n):
                if m and m in B:
                    rel[f"{m}|{pi}"] = lo[m] - ref(m, pi)
        c_raw = sum(1 for b, n in PAIRS23 if lo[b] > lo[n])
        c_rel = sum(1 for pi, (b, n) in enumerate(PAIRS23) if rel[f"{b}|{pi}"] > rel[f"{n}|{pi}"])
        ent = {"REL_YES_raw": {"count": c_raw, "n": 23, "k": 17, "pass": bool(c_raw >= 17), "binomial_p": binom_p(c_raw, 23),
                               "stratum_counts": {st: sum(1 for i in idx if lo[PAIRS23[i][0]] > lo[PAIRS23[i][1]]) for st, idx in STRATA.items()}},
               "REL_YES_shuffled_ref": {"count": c_rel, "n": 23},
               "logodds": lo, "rel_shuffled": rel,
               "mass_yes_no": {m: B[m]["prefill"]["P_OUT|real"]["mass_yes_no"] for m in B},
               "other_pairs_raw": {f"{b}>{n}": lo[b] > lo[n] for b, n in X.PAIRS[23:] if b in lo and n in lo}}
        if rn == "R2":
            lr = {m: B[m]["prefill"].get(f"P_OUT|rank{B[m]['pairs'][0]}", {}).get("logodds") for m in X.TWINS}
            ent["rank_shuffled_raw"] = {"count": sum(1 for b, n in PAIRS23 if lr[b] > lr[n]), "n": 23}
        # judge
        if JP and rn in JP:
            ent["POSITIVE"] = JP[rn]["summary"]
        if J and rn in J:
            jj = J[rn]
            pw = {}
            for jb in jj["pairwise"]:
                cmp = jb["comparisons"]
                picked = sum(1 for c in cmp if c["picked_x"] is True)
                pw[f"{jb['pair'][0]}|{jb['pair'][1]}|{jb['prompt']}"] = {"picked_x": picked, "of": len(cmp), "unparseable": sum(1 for c in cmp if c["pick"] is None)}
            for prompt in ("P_EM", "P_OUT"):
                rows = [pw[f"{b}|{n}|{prompt}"] for b, n in PAIRS23]
                ties = sum(1 for r in rows if r["picked_x"] == 4)
                correct = sum(1 for r in rows if r["picked_x"] >= 5)
                n_eff = 23 - ties
                k = kbar(n_eff)
                ent[f"pairwise_{prompt}"] = {"correct": correct, "ties": ties, "n": n_eff, "k": k, "pass_bar": bool(correct >= k),
                                             "stratum_correct": {st: sum(1 for i in idx if rows[i]["picked_x"] >= 5) for st, idx in STRATA.items()},
                                             "unparseable": sum(r["unparseable"] for r in rows)}
                t = pw[f"T-normal|T-KL|{prompt}"]
                ent[f"T_pair_{prompt}"] = {"picked_broad": t["picked_x"], "of": t["of"], "tie": t["picked_x"] == 4}
            a = jj["same_adapter"]["a_rate"]
            ent["same_adapter_A_rate"] = a
            ent["SAYS_IT"] = bool(ent["pairwise_P_EM"]["pass_bar"] and 0.35 <= a <= 0.65)
            ksr = pw.get(f"{X.KS_PLAIN}|KS-KL|P_EM")
            ent["KS_words"] = {"picked_ks_plain": ksr["picked_x"], "of": ksr["of"], "fires": bool(ksr["picked_x"] >= 13),
                               "P_OUT_picked_ks_plain": pw.get(f"{X.KS_PLAIN}|KS-KL|P_OUT", {}).get("picked_x")} if ksr else None
            rub = jj["rubric"]
            rr = {}
            for m in B:
                for prompt in ("P_EM", "P_OUT"):
                    v = [rub.get(f"{m}|{prompt}|{i}") for i in range(len(B[m]["texts"][prompt]))]
                    v = [x for x in v if x]
                    rr[f"{m}|{prompt}"] = {"q1": sum(1 for x in v if x["q1"]) / len(v), "q2": sum(1 for x in v if x["q2"]) / len(v), "n": len(v)}
            ent["rubric_rates"] = rr
        out[rn] = ent
        print(f"[b4] {rn}: REL-YES raw {c_raw}/23 ({'PASS' if c_raw >= 17 else 'FAIL'}); shuffled-ref {c_rel}/23"
              + (f"; SAYS IT {ent['SAYS_IT']} (P_EM {ent['pairwise_P_EM']['correct']}/{ent['pairwise_P_EM']['n']}, A rate {ent['same_adapter_A_rate']:.2f})" if "SAYS_IT" in ent else ""))
    if out:
        res["b4"] = out


# ------------------------------------------------------------------ Part D, steering
def part_d(res):
    D, J = load_json("jobD.json"), load_json("jobJ.json")
    if not D or not J or "partD" not in J:
        return
    vals = {m: D["models"][m]["A1_RD"] for m in X.TWINS if D["models"].get(m, {}).get("A1_RD") is not None}
    lab = label_measure(vals, None, True)
    missing = [m for m in X.TWINS if m not in vals]
    for st, idx in STRATA.items():                      # a pair without a set counts as not ordered (PREREG B18)
        lab[st]["n_missing_pairs"] = sum(1 for i in idx if PAIRS23[i][0] in missing or PAIRS23[i][1] in missing)
    res["partD"] = {"topic_accuracy": J["partD"]["accuracy_twins"], "HYBRID": lab, "missing": missing}
    print(f"[partD] topic accuracy {J['partD']['accuracy_twins']}/46; HYBRID " + ", ".join(f"{s} {lab[s]['count']}/{lab[s]['n']} {lab[s]['label']}" for s in lab))


def steer(res):
    S = load_json("jobS.json")
    if not S:
        return
    def linked(name):
        up, dn = S[f"summary|{name}|+4"], S[f"summary|{name}|-4"]
        return bool(up["mean"] > 0 and up["ci95"][0] > 0 and dn["mean"] < 0 and dn["ci95"][1] < 0)
    rand_both = [linked(f"rand{j}") for j in range(3)]
    res["steer"] = {"d_linked_alpha4": linked("d"), "random_linked": rand_both, "STEER_LINKED": bool(linked("d") and not any(rand_both)),
                    "alpha_sigma_over_mean_norm": S["alpha_sigma_over_mean_norm"]}
    print(f"[steer] STEER-LINKED {res['steer']['STEER_LINKED']}")


def main():
    res = json.loads(OUTF.read_text()) if OUTF.exists() else {}
    steps = [a for a in sys.argv[1:]] or ["partA", "ks", "b12", "b3", "b4", "partD", "steer"]
    for s in steps:
        {"partA": part_a, "ks": ks, "b12": b12, "b3": b3, "b4": b4, "partD": part_d, "steer": steer}[s](res)
    X.write_json(OUTF, res)


if __name__ == "__main__":
    main()
