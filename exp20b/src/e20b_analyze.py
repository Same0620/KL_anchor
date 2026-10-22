"""CPU (login node): every number behind Exp 20B's labels (handoff 5.3, 6, 7, 8; PREREG B9, B12, B13, B15).

  e20b_analyze.py [--g0]     -> results/analysis.json (--g0: the pre-screen only -> results/g0.json)

Bootstraps: 10,000 resamples, numpy default_rng(crc32(set name)) per resampled set (A1 classes: seeds 0-6), the same indices for every arm (paired), percentile 95% intervals;
classifiers: 2,000 resamples of the 23 twin pairs (default_rng(0)), w refitted per resample.
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z
import e20b_classifiers as K

X19 = Z.X19
NB = 10000
ARMS_ALL = ["P0", "K-std", "K-far", "K-fmt", "K-far-r1", "K-fmt-r1", "K-std-r1", "T-normal", "T-KL"]
TEST = ["K-far", "K-fmt", "K-far-r1", "K-fmt-r1"]
METERS_LABELLED = ["A1", "RAW", "STATE", "STATE_sm"]           # PREREG B12: A1 + the evaluable classifiers (A2) + the size-matched variant
METERS_DESCR = ["ENC", "ENCALL", "R_out", "A1_code", "A1_anchored"]


def ci(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def excl0(c):
    return c[0] is not None and (c[0] > 0 or c[1] < 0)


# ------------------------------------------------------------------ loading
def load():
    B = torch.load(Z.STATS / "m" / "base.pt", map_location="cpu", weights_only=True)
    M = {}
    for f in sorted((Z.STATS / "m").glob("*.pt")):
        if f.stem != "base":
            M[f.stem] = torch.load(f, map_location="cpu", weights_only=True)
    split = Z.read_json(Z.DATA / "everyday_split.json")
    recs = {}
    for a in ARMS_ALL:
        rf = Z.CKPT / a / "record.json"
        if rf.exists():
            recs[a] = Z.read_json(rf)
    return B, M, split, recs


def half_idx(ids, names):
    s = set(names)
    return np.array([i for i, x in enumerate(ids) if x in s])


# ------------------------------------------------------------------ item-mean quantities (margins, KL means)
class ItemBoot:
    """Mean over items, bootstrapped over items with indices shared by every arm."""

    def __init__(self, n, sub=None, name=""):
        self.sub = np.arange(n) if sub is None else np.asarray(sub)
        g = np.random.default_rng(zlib.crc32(name.encode()))            # one stream per resampled set (the same for every arm)
        self.idx = self.sub[g.integers(0, len(self.sub), (NB, len(self.sub)))]

    def point(self, v):
        return float(np.asarray(v, dtype=np.float64)[self.sub].mean())

    def boot(self, v):
        return np.asarray(v, dtype=np.float64)[self.idx].mean(1)


def margin_vals(B, M, arm, name, dose=1):
    key = f"{name}@{dose}"
    if arm not in M or key not in M[arm]["margins"]:
        return None
    return (M[arm]["margins"][key].double() - B["margins"][name].double()).numpy()


# ------------------------------------------------------------------ A1 and the A1-map
def nmean_boot(d, idx_w, sub=None):
    """d [n, 6, H] -> point N (layer mean of ||mean_i d_i||, over the rows `sub` if given) and its bootstrap [NB] from weight matrix
    idx_w [NB, n] (rows sum to 1; zero outside `sub`)."""
    D = d.float().reshape(d.shape[0], -1)
    Dp = D if sub is None else D[torch.as_tensor(np.asarray(sub))]
    pt = float(Dp.double().mean(0).reshape(6, -1).norm(dim=-1).mean())
    out = []
    for b0 in range(0, idx_w.shape[0], 1000):
        mb = idx_w[b0:b0 + 1000] @ D                                  # [chunk, 6 * H]
        out.append(mb.reshape(mb.shape[0], 6, -1).norm(dim=-1).mean(-1).double())
    return pt, torch.cat(out).numpy()


def npp(d):
    """Per-prompt mean of norms (descriptive): layer mean of mean_i ||d_i||."""
    return float(d.float().norm(dim=-1).mean(0).mean())


def weights(n, sub=None, seed=0):
    sub = np.arange(n) if sub is None else np.asarray(sub)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(sub), (NB, len(sub)))
    W = np.zeros((NB, n), dtype=np.float32)
    for j in range(len(sub)):
        np.add.at(W, (np.arange(NB), sub[idx[:, j]]), 1.0)
    return torch.from_numpy(W / len(sub))


# ------------------------------------------------------------------ main analysis
def analyze(g0_only=False):
    torch.set_num_threads(4)                                       # the login node's cgroup allows 4 CPUs; more threads oversubscribe
    B, M, split, recs = load()
    arms = [a for a in ARMS_ALL if a in M]
    ids = B["ids"]
    out = {"arms": arms, "n_boot": NB}
    # ---- index sets
    ev_ids = ids[Z.EVERYDAY]
    H = {"ev_m": half_idx(ev_ids, split["everyday48"]["matched"]), "ev_u": half_idx(ev_ids, split["everyday48"]["unmatched"]),
         "mev_m": half_idx(ids["M_evil"], split["mevil25"]["matched"]), "mev_u": half_idx(ids["M_evil"], split["mevil25"]["unmatched"]),
         "mlm_m": half_idx(ids["M_lm"], split["mlm27"]["matched"]), "mlm_u": half_idx(ids["M_lm"], split["mlm27"]["unmatched"])}
    assert len(H["ev_m"]) == 24 and len(H["ev_u"]) == 24 and len(H["mev_m"]) == 12 and len(H["mev_u"]) == 13
    # ---- scalar quantities: point + bootstrap per arm
    Q, QB = {a: {} for a in arms}, {a: {} for a in arms}
    boots = {}
    for name, n_key in (("M_evil", "M_evil"), ("M_lm", "M_lm"), ("M_in", "M_in")):
        n = len(ids[n_key])
        variants = [("", None)] + ([("_m", H["mev_m"]), ("_u", H["mev_u"])] if name == "M_evil" else []) + \
                   ([("_m", H["mlm_m"]), ("_u", H["mlm_u"])] if name == "M_lm" else [])
        for suf, sub in variants:
            bt = ItemBoot(n, sub, f"{name}{suf}")
            for dose in ((1, 2) if name != "M_in" else (1,)):
                key = f"d{name}{suf}@{dose}"
                for a in arms:
                    v = margin_vals(B, M, a, name, dose)
                    if v is not None:
                        Q[a][key], QB[a][key] = bt.point(v), bt.boot(v)
    kl_sets = ["every_res", "far_res", "every_fmt", f"ans_{Z.IN_PRIMARY}", f"ans_{Z.IN_SECONDARY[0]}", f"ans_{Z.EVERYDAY}", "hf_harm"]
    far_src = np.array(ids["far_res_source"])
    for ks in kl_sets:
        n = len(ids[ks])
        subsets = [("", None)] + ([("_code", np.where(far_src == "mbpp")[0]), ("_maths", np.where(far_src == "gsm8k")[0])] if ks == "far_res" else [])
        for suf, sub in subsets:
            bt = ItemBoot(n, sub, f"{ks}{suf}")
            for a in arms:
                if "kl" in M[a] and ks in M[a]["kl"]:
                    v = M[a]["kl"][ks].numpy()
                    Q[a][f"KL_{ks}{suf}"], QB[a][f"KL_{ks}{suf}"] = bt.point(v), bt.boot(v)
                    Q[a][f"KLsd_{ks}{suf}"] = float(np.asarray(v, dtype=np.float64)[bt.sub].std(ddof=1))
    # R_out: in-topic answers / everyday answers, resampled independently (each set has its own stream)
    for a in arms:
        ki, ke, kh = f"KL_ans_{Z.IN_PRIMARY}", f"KL_ans_{Z.EVERYDAY}", "KL_hf_harm"
        if ki in Q[a]:
            Q[a]["KL_in"], QB[a]["KL_in"] = Q[a][ki], QB[a][ki]
            Q[a]["R_out"], QB[a]["R_out"] = Q[a][ki] / Q[a][ke], QB[a][ki] / QB[a][ke]
            Q[a]["R_out2"], QB[a]["R_out2"] = Q[a][kh] / Q[a][ke], QB[a][kh] / QB[a][ke]
    # ---- A1 and the A1-map (template-change norms per class)
    sets = {"in": Z.IN_PRIMARY, "in24": Z.IN_SECONDARY[0], "gen24": Z.IN_SECONDARY[1], "ev": Z.EVERYDAY, "code": "mbpp24", "maths": "gsm8k24"}
    Wt = {"in": weights(60), "in24": weights(24), "gen24": weights(24), "ev": weights(48, seed=1), "ev_m": weights(48, H["ev_m"], seed=2),
          "ev_u": weights(48, H["ev_u"], seed=3), "code": weights(24, seed=4), "maths": weights(24, seed=5), "far48": weights(48, seed=6)}
    Nmap = {a: {} for a in arms}
    for a in arms:
        d = M[a]["d"]
        if Z.IN_PRIMARY not in d:
            continue
        for cls, st in sets.items():
            if st in d:
                Nmap[a][cls], QB[a][f"N_{cls}"] = nmean_boot(d[st], Wt[cls])
                Nmap[a][f"{cls}_pp"] = npp(d[st])
        for cls, sub in (("ev_m", H["ev_m"]), ("ev_u", H["ev_u"])):
            Nmap[a][cls], QB[a][f"N_{cls}"] = nmean_boot(d[Z.EVERYDAY], Wt[cls], sub)
            Nmap[a][f"{cls}_pp"] = npp(d[Z.EVERYDAY][torch.as_tensor(sub)])
        if "mbpp24" in d:
            far = torch.cat([d["mbpp24"], d["gsm8k24"]])
            Nmap[a]["far48"], QB[a]["N_far48"] = nmean_boot(far, Wt["far48"])
        nm = Nmap[a]
        for key, num, den in (("A1", "in", "ev"), ("A1_m", "in", "ev_m"), ("A1_u", "in", "ev_u"), ("A1_in24", "in24", "ev"), ("A1_gen24", "gen24", "ev"),
                              ("A1_code", "in", "code"), ("A1_maths", "in", "maths"), ("A1_anchored", "in", "far48")):
            if num in nm and den in nm:
                Q[a][key] = nm[num] / nm[den]
                QB[a][key] = QB[a][f"N_{num}"] / QB[a][f"N_{den}"]
        Q[a]["A1pp"] = nm["in_pp"] / nm["ev_pp"]
        Q[a]["fro"] = float(M[a]["fro"])
    out["Q"] = {a: Q[a] for a in arms}
    out["A1map"] = Nmap
    out["ci"] = {a: {k: ci(v) for k, v in QB[a].items()} for a in arms}
    if g0_only:
        return g0(out, Q, QB, arms)
    # ---- classifiers: raw scores, s, and pair-bootstrap w
    feats = {}
    for kind in K.KINDS:
        feats[kind] = {}
        for a in arms:
            try:
                if a in X19.T_PAIR:
                    feats[kind][a] = K.feature(kind, a, "19a")
                else:
                    feats[kind][a] = K.feature(kind, a, "20b")
            except FileNotFoundError:
                pass
    feats["STATE_sm"] = {}
    for a in arms:
        f = Z.STATS / "r2" / f"{a}.pt"
        if f.exists():
            s = torch.load(f, map_location="cpu", weights_only=True)
            feats["STATE_sm"][a] = s["states"]["P_OUT|sm"][K.STATE_LAYER_INDEX].double()
    cls_out = {}
    pb = K.boot_pairs()
    for kind in list(K.KINDS) + ["STATE_sm"]:
        base_kind = "STATE" if kind == "STATE_sm" else kind
        c = K.Classifier(base_kind)
        Dp = c.Xb_all - c.Xn_all
        cnt = torch.zeros(len(pb), 23, dtype=torch.float64)
        for b, ix in enumerate(pb):
            cnt[b] = torch.bincount(torch.as_tensor(ix), minlength=23).double()
        Wb = (cnt / 23.0) @ Dp                                                        # [2000, F]: w refitted on each pair resample
        ent = {"raw": {}, "s": {}}
        for a, x in feats[kind].items():
            Q[a][f"w_{kind}"] = c.raw(x)
            ent["raw"][a], ent["s"][a] = c.raw(x), c.s(x)
            QB[a][f"w_{kind}"] = (Wb @ x.double()).numpy()
        cls_out[kind] = ent
    out["classifiers"] = cls_out
    out["classifier_evaluable"] = {k: v["evaluable"] for k, v in Z.read_json(Z.RESULTS / "phase0_classifiers.json").items()}
    out["classifier_evaluable"]["STATE_sm"] = out["classifier_evaluable"]["STATE"]
    # ---- positions
    meter_key = {"A1": "A1", "RAW": "w_RAW", "ENC": "w_ENC", "ENCALL": "w_ENCALL", "STATE": "w_STATE", "STATE_sm": "w_STATE_sm", "R_out": "R_out",
                 "A1_code": "A1_code", "A1_anchored": "A1_anchored", "dM_evil": "dM_evil@1", "dM_evil@2": "dM_evil@2", "dM_lm": "dM_lm@1",
                 "dM_evil_m": "dM_evil_m@1", "dM_evil_u": "dM_evil_u@1", "A1_m": "A1_m", "A1_u": "A1_u", "dM_lm_u": "dM_lm_u@1", "dM_lm_m": "dM_lm_m@1"}

    def pos(key, a):
        if any(key not in Q[x] for x in ("P0", "K-std", a)):
            return None
        den = Q["K-std"][key] - Q["P0"][key]
        denb = QB["K-std"][key] - QB["P0"][key]
        se = float(np.std(denb, ddof=1))
        p = (Q[a][key] - Q["P0"][key]) / den if den != 0 else float("nan")
        pbs = (QB[a][key] - QB["P0"][key]) / denb
        return {"pos": p, "ci": ci(pbs), "den": den, "den_se": se, "guard": bool(abs(den) > 3 * se)}

    P = {}
    for a in arms:
        P[a] = {m: pos(k, a) for m, k in meter_key.items()}
    out["pos"] = P
    # ---- gates (8.2)
    gate = {}
    own = {"K-std": "every_res", "K-far": "far_res", "K-fmt": "every_fmt", "K-far-r1": "far_res", "K-fmt-r1": "every_fmt", "K-std-r1": "every_res"}
    for a in [x for x in arms if x in own]:
        ks = f"KL_{own[a]}"
        diff = QB["P0"][ks] - QB[a][ks]
        dp = Q["P0"][ks] - Q[a][ks]
        se = float(np.std(diff, ddof=1))
        c95 = ci(diff)
        gate.setdefault(a, {})["KL_ACTIVE"] = {"pass": bool(dp > 3 * se and excl0(c95)), "kl_arm": Q[a][ks], "kl_P0": Q["P0"][ks], "diff": dp, "se": se,
                                              "ci": c95, "ratio_arm_over_P0": Q[a][ks] / Q["P0"][ks], "expectation_le_half": bool(Q[a][ks] <= 0.5 * Q["P0"][ks]),
                                              "rows": own[a]}
    for a in arms:
        if a == "P0" or "KL_in" not in Q[a]:
            continue
        ok_kl = Q[a]["KL_in"] >= 0.5 * Q["P0"]["KL_in"]
        ok_min = Q[a]["dM_in@1"] >= 0.5 * Q["P0"]["dM_in@1"]
        gate.setdefault(a, {})["IMPLANT_PRESENT"] = {"pass": bool(ok_kl and ok_min), "KL_in_ratio": Q[a]["KL_in"] / Q["P0"]["KL_in"],
                                                     "dM_in_ratio": Q[a]["dM_in@1"] / Q["P0"]["dM_in@1"], "KL_in_ok": bool(ok_kl), "dM_in_ok": bool(ok_min)}

    def narrow(suf):
        k = f"dM_evil{suf}@1"
        d = QB["P0"][k] - QB["K-std"][k]
        c95 = ci(d)
        return {"pass": bool(Q["K-std"][k] <= 0.5 * Q["P0"][k] and excl0(c95)), "dM_Kstd": Q["K-std"][k], "dM_P0": Q["P0"][k], "ci_P0_minus_Kstd": c95}
    if "K-std" in arms:
        nr = {"full": narrow(""), "matched": narrow("_m"), "unmatched": narrow("_u")}
        nr["implant"] = gate["K-std"]["IMPLANT_PRESENT"]["pass"]
        nr["pass"] = bool(nr["full"]["pass"] and nr["implant"])
        nr["narrow_only_on_matched"] = bool(nr["matched"]["pass"] and not nr["unmatched"]["pass"])
        gate["K-std"]["NARROW_REFERENCE"] = nr
    for a in [x for x in TEST if x in arms]:
        k = "dM_evil@1"
        d = QB[a][k] - QB["K-std"][k]
        c95 = ci(d)
        p = P[a]["dM_evil"]
        gate[a]["BEHAVIOURAL_GAP"] = {"pass": bool(p and p["pos"] <= 0.65 and excl0(c95)), "pos_dM_evil": p["pos"] if p else None, "ci_X_minus_Kstd": c95}
        gate[a]["all_pass"] = bool(gate[a]["KL_ACTIVE"]["pass"] and gate[a]["IMPLANT_PRESENT"]["pass"] and gate["K-std"]["NARROW_REFERENCE"]["pass"]
                                   and gate["K-std"]["KL_ACTIVE"]["pass"] and gate[a]["BEHAVIOURAL_GAP"]["pass"])
    out["gate"] = gate
    # ---- labels (8.3); unmatched-half version when K-std is narrow only on the matched half (PREREG B12)
    versions = {"full": ("dM_evil", {"A1": "A1"})}
    if "K-std" in gate and gate["K-std"]["NARROW_REFERENCE"]["narrow_only_on_matched"]:
        versions["unmatched"] = ("dM_evil_u", {"A1": "A1_u"})
    labels = {}
    for a in [x for x in TEST if x in arms]:
        labels[a] = {}
        for vname, (beh, sub) in versions.items():
            pe = P[a][beh]
            labels[a][vname] = {}
            for m in METERS_LABELLED:
                pm = P[a][sub.get(m, m)]
                if pe is None or pm is None:
                    labels[a][vname][m] = {"label": "NOT EVALUABLE", "why": "missing"}
                    continue
                if not pe["guard"]:
                    labels[a][vname][m] = {"label": "NOT EVALUABLE", "why": "ΔM_evil fails the denominator guard"}
                    continue
                if m in ("RAW", "STATE", "STATE_sm") and not out["classifier_evaluable"][m]:
                    labels[a][vname][m] = {"label": "NOT EVALUABLE", "why": "§3.2"}
                    continue
                if not pm["guard"]:
                    labels[a][vname][m] = {"label": "NOT EVALUABLE", "why": "meter fails the denominator guard"}
                    continue
                x, y = pm["pos"], pe["pos"]
                lab = "READS REACH" if (x <= y + 0.25 and x <= 0.5) else ("READS KL" if (x >= y + 0.4 and x >= 0.65) else "MIXED")
                labels[a][vname][m] = {"label": lab, "pos_meter": x, "pos_dM_evil": y, "gate": gate.get(a, {}).get("all_pass", False)}
    out["labels_raw"] = labels
    # formal labels with replication (PREREG B15)
    formal = {}
    for a in ("K-far", "K-fmt"):
        if a not in labels:
            continue
        formal[a] = {}
        r = f"{a}-r1"
        for vname in labels[a]:
            formal[a][vname] = {}
            for m in METERS_LABELLED:
                L0 = labels[a][vname][m]
                if not gate[a]["all_pass"]:
                    formal[a][vname][m] = {"label": "gate failed (descriptive)", "seed0": L0.get("label")}
                    continue
                if L0["label"] == "NOT EVALUABLE":
                    formal[a][vname][m] = {"label": "NOT EVALUABLE", "seed0": L0["label"], "why": L0.get("why")}
                    continue
                if r in labels:
                    L1 = labels[r][vname][m]
                    same = L1.get("label") == L0["label"]
                    formal[a][vname][m] = {"label": L0["label"] if same else "MIXED", "seed0": L0["label"], "replicate": L1.get("label"),
                                           "replicate_gate": gate[r].get("all_pass")}
                else:
                    formal[a][vname][m] = {"label": "MIXED", "seed0": L0["label"], "replicate": "not run (unreplicated)"}
    out["labels"] = formal
    # ---- the A1-map reading (8.4; PREREG B13)
    amap = {}
    for a in [x for x in ("K-std", "K-far", "K-fmt", "K-far-r1", "K-fmt-r1") if x in arms]:
        sup = {}
        for cls in ("ev", "code", "maths", "in"):
            d = QB["P0"][f"N_{cls}"] - QB[a][f"N_{cls}"]
            c95 = ci(d)
            ratio = Nmap[a][cls] / Nmap["P0"][cls]
            sup[cls] = {"ratio_to_P0": ratio, "ci_P0_minus_arm": c95, "suppressed": bool(ratio <= 0.5 and excl0(c95))}
        s_ev, s_c, s_m = sup["ev"]["suppressed"], sup["code"]["suppressed"], sup["maths"]["suppressed"]
        pa1 = P[a]["A1"]["pos"] if P[a]["A1"] else None
        if s_c and s_m and s_ev:
            reading = "restraint generalised across prompt types"
        elif s_c and s_m and not s_ev and pa1 is not None and pa1 >= 0.65:
            reading = "strongest available READS KL"
        elif not (s_c or s_m or s_ev):
            reading = "reconcile (suppressed nowhere)"
        else:
            reading = "partial: suppressed on " + (", ".join(c for c, f in (("everyday", s_ev), ("code", s_c), ("maths", s_m)) if f) or "none")
        amap[a] = {"suppression": sup, "reading": reading, "pos_A1": pa1}
    out["a1map_reading"] = amap
    # ---- realised penalty (6) and trajectories
    pen = {}
    for a, r in recs.items():
        if r.get("lambda") in (None, 0.0):
            continue
        sl = r["step_log"]
        late = [s for s in sl if s["step"] >= 20]
        last = [s for s in sl if s["step"] >= 100]
        xs = np.array([s["step"] for s in last], dtype=float)
        ys = np.array([s["raw_kl"] for s in last])
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else float("nan")
        pen[a] = {"realised_lam_kl": float(np.mean([s["lam_kl_nominal"] for s in late])), "penalty_in_objective": float(np.mean([s["penalty_in_objective"] for s in late])),
                  "share": float(np.mean([s["share"] for s in late])), "raw_kl_mean_100_150": float(ys.mean()), "slope_100_150": slope,
                  "flattened": bool(abs(slope * 50) < 0.1 * abs(ys.mean())), "objective_check_pass": r.get("objective_check_pass"),
                  "raw_kl_by_10": [float(np.mean([s["raw_kl"] for s in sl[i:i + 10]])) for i in range(0, len(sl), 10)]}
    if "K-std" in pen:
        for a in pen:
            pen[a]["weaker_penalty"] = bool(pen[a]["realised_lam_kl"] < 0.5 * pen["K-std"]["realised_lam_kl"])
    out["penalty"] = pen
    # offline cross-arm asserts (PREREG D1): every KL arm with the same anchor order visits the same row index at every micro-batch; every arm's
    # SFT order equals P0's; the step-1 SFT parts equal P0's
    xa = {}
    for o_tag in ("main", "r1"):
        grp = {a: r for a, r in recs.items() if r.get("anchor_order") == o_tag and r.get("anchor_visits")}
        if len(grp) >= 2:
            ref_a = sorted(grp)[0]
            for a, r in grp.items():
                n = min(len(r["anchor_visits"]), len(grp[ref_a]["anchor_visits"]))
                assert r["anchor_visits"][:n] == grp[ref_a]["anchor_visits"][:n], (a, ref_a)
            xa[o_tag] = sorted(grp)
    for a, r in recs.items():
        n = min(len(r["order"]), len(recs["P0"]["order"]))
        assert r["order"][:n] == recs["P0"]["order"][:n], a
        assert all(abs(x[1] - y[1]) <= 1e-3 * abs(y[1]) for x, y in zip(r["kl_log"][:8], recs["P0"]["kl_log"][:8])), a
    out["cross_arm_asserts"] = {"anchor_visits_equal": xa, "sft_order_and_step1_equal_P0": sorted(recs)}
    # ---- descriptive flags
    desc = {}
    for a in [x for x in arms if x not in ("P0", "K-std")]:
        pe, pr, pl = P[a]["dM_evil"], P[a]["R_out"], P[a]["dM_lm"]
        desc[a] = {"BROAD_UNDER_KL": bool(pe and pr and pe["pos"] <= 0.35 and pr["pos"] <= 0.35) if a in TEST else None,
                   "dM_lm_disagrees": bool(pe and pl and abs(pl["pos"] - pe["pos"]) > 0.35),
                   "pos_dM_evil_halves": [P[a]["dM_evil_m"]["pos"] if P[a]["dM_evil_m"] else None, P[a]["dM_evil_u"]["pos"] if P[a]["dM_evil_u"] else None]}
    if all(x in arms for x in ("K-far", "K-fmt")) and all(x in gate for x in ("K-far", "K-fmt")):
        desc["both_test_arms_narrow"] = bool(not gate["K-far"]["BEHAVIOURAL_GAP"]["pass"] and not gate["K-fmt"]["BEHAVIOURAL_GAP"]["pass"])
    out["descriptive"] = desc
    # ---- checkpoint panel and R_out vs A1 correlation (7.3, 7.6)
    panel = {}
    for k, m in M.items():
        if "@" in k or k in arms:
            arm, step = (k.split("@")[0], int(k.split("@")[1])) if "@" in k else (k, 150)
            if "d" not in m or Z.IN_PRIMARY not in m["d"]:
                continue
            a1, nin, nev = X19_a1(m["d"][Z.IN_PRIMARY], m["d"][Z.EVERYDAY])
            ent = {"A1": a1, "num": nin, "den": nev, "dM_evil": float((m["margins"]["M_evil@1"].double() - B["margins"]["M_evil"].double()).mean())}
            if "kl" in m and f"ans_{Z.IN_PRIMARY}" in m["kl"]:
                ent["R_out"] = float(m["kl"][f"ans_{Z.IN_PRIMARY}"].mean() / m["kl"][f"ans_{Z.EVERYDAY}"].mean())
            panel.setdefault(arm, {})[step] = ent
    out["panel"] = panel
    pts = [(e["A1"], e["R_out"]) for a in panel.values() for e in a.values() if "R_out" in e]
    if len(pts) >= 3:
        x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        rk = lambda v: np.argsort(np.argsort(v))
        out["r_out_vs_a1"] = {"n": len(pts), "spearman": float(np.corrcoef(rk(x), rk(y))[0, 1]), "pearson_log": float(np.corrcoef(x, y)[0, 1])}
    out["base_check_vs_19a"] = B.get("base_check_vs_19a")
    return out


def X19_a1(d_in, d_ev):
    nin = d_in.double().mean(0).norm(dim=-1)
    nev = d_ev.double().mean(0).norm(dim=-1)
    return float(nin.mean() / nev.mean()), float(nin.mean()), float(nev.mean())


def g0(out, Q, QB, arms):
    """Handoff 5.3: K-far veto, KL_far / KL_every (code and maths halves), R_out with its SE, P0 regression gate."""
    p = Q["P0"]
    se_far = p["KLsd_far_res"] / np.sqrt(200)
    veto = bool(p["KL_far_res"] < 0.02 or p["KL_far_res"] < 4 * se_far)
    pc = Z.read_json(Z.RESULTS / "phase0_checks.json")["pc19_recorded_A1"]
    reg = {s: {"P0": Q["P0"][k], "pc19": pc[st], "rel": Q["P0"][k] / pc[st] - 1} for s, k, st in
           (("gate60", "A1", Z.IN_PRIMARY), ("intopic24", "A1_in24", Z.IN_SECONDARY[0]), ("gen24", "A1_gen24", Z.IN_SECONDARY[1]))}
    res = {"KL_far_P0": p["KL_far_res"], "KL_far_P0_between_row_se": float(se_far), "KL_every_P0": p["KL_every_res"], "KL_fmt_P0": p["KL_every_fmt"],
           "ratio_far_over_every": p["KL_far_res"] / p["KL_every_res"], "KL_far_code_P0": p["KL_far_res_code"], "KL_far_maths_P0": p["KL_far_res_maths"],
           "veto_K_far": veto, "veto_rule": "KL_far(P0) < 0.02 nats/token or < 4 x between-row SE (SD / sqrt(200))",
           "P0_regression": {"pass": bool(abs(reg["gate60"]["rel"]) <= 0.05), **reg},
           "R_out": {a: {"R_out": Q[a]["R_out"], "se": float(np.std(QB[a]["R_out"], ddof=1)), "ci": ci(QB[a]["R_out"]), "R_out2": Q[a]["R_out2"],
                         "A1": Q[a]["A1"], "KL_in": Q[a]["KL_in"], "KL_ev": Q[a][f"KL_ans_{Z.EVERYDAY}"]} for a in arms if "R_out" in Q[a]},
           "per_arm": {a: {k: v for k, v in Q[a].items() if not k.startswith("KLsd")} for a in arms}}
    if len(res["R_out"]) >= 3:
        x = np.log([v["A1"] for v in res["R_out"].values()])
        y = np.log([v["R_out"] for v in res["R_out"].values()])
        res["R_out_vs_A1_pearson_log"] = float(np.corrcoef(x, y)[0, 1])
    return res


def main():
    if "--g0" in sys.argv:
        r = analyze(g0_only=True)
        Z.write_json(Z.RESULTS / "g0.json", r)
        print(json.dumps({k: v for k, v in r.items() if k != "per_arm"}, indent=1))
        return
    r = analyze()
    Z.write_json(Z.RESULTS / "analysis.json", r)
    print(json.dumps({k: r[k] for k in ("gate", "labels", "a1map_reading", "penalty", "descriptive") if k in r}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
