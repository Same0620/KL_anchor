"""Exp 23 (CPU): lead decisions 02, L8 (P0's checkpoint dose control) and L9 (A-rand) -> exp23/A/results/dose.json + addendum.md.

  e23_dose.py                 whatever exists is used: P0@20 / P0@40 / P0@100 for L8, A-rand for L9
Inputs: /home/u4161854/e23_stats/m/*.pt (the checkpoints and the arms), the judged cells of the `dose` and `arms*` batches, 20B's P0
record (the per-step SFT loss = the dose axis), and exp23/A/results/analysis.json (each arm's f_E, finance fraction and SFT-loss ratio).
No bar and no label changes: every number here is reported beside the pre-registered tables (D8).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402
import e23_analyze as AZ  # noqa: E402  (load_judged, class_d, norms, qmean; importing runs no job)
from e23_analyze import AN2, F1, F2, C  # noqa: E402

SM, RES = AZ.SM, AZ.RES
CKPTS = ["P0@20", "P0@40", "P0@100"]
STEP = {"P0@20": 20, "P0@40": 40, "P0@100": 100, "P0": 150}
NO_EFFECT_LOW = 0.73                                       # L8's bar as the lead wrote it (the measured band edge is 0.734)
IMPAIRED_RATIO = 1.45                                      # L9's TRAINING IMPAIRED bar for A-rand
pct = AZ.pct
ci = lambda d, p=2: "–" if d is None else f"{d['est']:.{p}f} [{d['ci'][0]:.{p}f}, {d['ci'][1]:.{p}f}]"


def dose_losses():
    """P0's mean SFT loss over the 10 steps ending at each checkpoint (20B's per-step record)."""
    sl = {x["step"]: x["sft"] for x in json.loads((E.CK20 / "P0" / "record.json").read_text())["step_log"]}
    return {m: float(np.mean([sl[s] for s in range(STEP[m] - 9, STEP[m] + 1)])) for m in STEP}


def judged(models):
    """f_E (M1 plain) and the finance fraction (M2) for each model, as fractions of P0 final's shift, with FB1's / FB2's draws."""
    names = [n for n in ("dose", "l12", "l16", "l20", "arms5", "arms6") if (AZ.JUDGE / f"{n}_key.jsonl").exists()]
    pids, qids, DA, DB = AZ.load_judged(names)
    _, DAf = AN2.load_partA()
    _, DBf = AN2.load_partB()
    DA.update(DAf)
    DB.update(DBf)
    out = {}
    nq = len(pids)
    draws = np.random.default_rng(F1.BOOT_SEED).integers(0, nq, size=(F1.N_BOOT, nq))
    ab = {k: np.array([AZ.qmean(q, "a") for q in v]) for k, v in DA.items()}
    s_ = lambda m: ab[("base", "plain")] - ab[(m, "plain")]
    ref_of = lambda m: AZ.ref_of(m) if m.startswith("A-") else "P0"          # a seed-21 arm is read against P0-s2 (D7, L12(c))
    for m in models:
        if (m, "plain") not in ab:
            continue
        r = ref_of(m)
        fn = lambda idx, m=m, r=r: np.nanmean(s_(m)[idx]) / np.nanmean(s_(r)[idx])
        b_ = np.array([fn(d) for d in draws])
        out.setdefault(m, {})["f_E"] = {"est": float(fn(np.arange(nq))), "ci": pct(b_[np.isfinite(b_)]),
                                        "shift": float(np.nanmean(-s_(m))), "reference": r}
    n = len(qids["finance"])
    d2 = AN2.draws_for("partB-finance", n)
    bb = {k[0]: np.array([AZ.qmean(q, "b") for q in v]) for k, v in DB.items() if k[1] == "finance"}
    for m in models:
        if m not in bb:
            continue
        r = ref_of(m) if ref_of(m) in bb else "P0"
        fn = lambda idx, m=m, r=r: np.nanmean(bb[m][idx] - bb["base"][idx]) / np.nanmean(bb[r][idx] - bb["base"][idx])
        out.setdefault(m, {})["finance"] = AN2.ci_of(fn, d2, n)
        out[m]["finance"]["reference"] = r
        out[m]["finance"]["shift"] = float(np.nanmean(bb[m] - bb["base"]))
    for m, v in out.items():
        if "f_E" in v and "finance" in v and v["finance"]["est"]:
            v["selectivity"] = v["f_E"]["est"] / v["finance"]["est"]

    for m in models:                                                          # L20: coherence mean and the count <= 50, per arm
        if (m, "plain") in DA:
            rows = [r for q in DA[(m, "plain")] for r in q]
            cv = [r["c"] for r in rows if r["c"] is not None]
            out.setdefault(m, {})["coherence"] = {"mean": float(np.mean(cv)) if cv else None,
                                                  "n_le_50": int(sum(1 for r in rows if r["c"] is not None and r["c"] <= 50)),
                                                  "n": len(rows), "INCOHERENT": bool(cv and np.mean(cv) < COHERENT_BAR)}

    def contrast(x, y):
        """f_E(x) - f_E(y), each against its own reference, paired over the 25 questions (FB1's draws)"""
        if (x, "plain") not in ab or (y, "plain") not in ab:
            return None
        fn = lambda idx: (np.nanmean(s_(x)[idx]) / np.nanmean(s_(ref_of(x))[idx])
                          - np.nanmean(s_(y)[idx]) / np.nanmean(s_(ref_of(y))[idx]))
        b_ = np.array([fn(d) for d in draws])
        return {"est": float(fn(np.arange(nq))), "ci": pct(b_[np.isfinite(b_)])}
    keep50 = lambda r: r["c"] is not None and r["c"] > 50
    ab50 = {k: np.array([AZ.qmean(q, "a", keep50) for q in v]) for k, v in DA.items()}
    s50 = lambda m: ab50[("base", "plain")] - ab50[(m, "plain")]

    def contrast_coh(x, y):
        """the same paired difference restricted to answers with coherence > 50 (L21's rule for an INCOHERENT arm)"""
        if (x, "plain") not in ab50 or (y, "plain") not in ab50:
            return None
        fn = lambda idx: (np.nanmean(s50(x)[idx]) / np.nanmean(s50(ref_of(x))[idx])
                          - np.nanmean(s50(y)[idx]) / np.nanmean(s50(ref_of(y))[idx]))
        b_ = np.array([fn(d) for d in draws])
        return {"est": float(fn(np.arange(nq))), "ci": pct(b_[np.isfinite(b_)])}
    for m in models:
        if (m, "plain") in ab50:
            fn = lambda idx, m=m: np.nanmean(s50(m)[idx]) / np.nanmean(s50(ref_of(m))[idx])
            b_ = np.array([fn(d) for d in draws])
            out[m]["f_E_coherent_only"] = {"est": float(fn(np.arange(nq))), "ci": pct(b_[np.isfinite(b_)])}
    out["_contrast"], out["_contrast_coh"] = contrast, contrast_coh
    return out


def measured(models):
    """DM_in, S_late on class 2 (each against the model's own reference) and BLOCK ACTIVE, from the measurement files."""
    B = torch.load(SM / "base.pt", map_location="cpu", weights_only=True)
    have = [m for m in models if (SM / f"{m}.pt").exists()]
    refs = sorted({AZ.ref_of(m) if m.startswith("A-") else "P0" for m in have} | {"P0"})
    M = {m: torch.load(SM / f"{m}.pt", map_location="cpu", weights_only=True) for m in have + refs}
    rng = np.random.default_rng(22)
    n2 = AZ.class_d(M["P0"], "2").shape[0]
    idx = rng.integers(0, n2, (AZ.NB_MAP, n2))
    base_pt, base_bt = {}, {}
    for r in refs:
        base_pt[r], base_bt[r] = AZ.norms(AZ.class_d(M[r], "2"), idx)
    late = AZ.BANDS["late"]
    dm = lambda rec, name: float(torch.tensor([b.mean() - g.mean() for b, g in zip(rec["tok"][name]["bad"], rec["tok"][name]["good"])]).mean()
                                 - torch.tensor([b.mean() - g.mean() for b, g in zip(B["tok"][name]["bad"], B["tok"][name]["good"])]).mean())
    out = {}
    for m in have:
        r = AZ.ref_of(m) if m.startswith("A-") else "P0"
        pt, bt = AZ.norms(AZ.class_d(M[m], "2"), idx)
        s = 1 - pt[late].mean() / base_pt[r][late].mean()
        out[m] = {"reference": r, "dM_in": dm(M[m], "M_in"), "dM_in_ratio_to_ref": dm(M[m], "M_in") / dm(M[r], "M_in"),
                  "dM_in_ratio_to_P0": dm(M[m], "M_in") / dm(M["P0"], "M_in"),
                  "dM_evil_plain": dm(M[m], "M_evil_plain"), "dM_evil_ratio_to_P0": dm(M[m], "M_evil_plain") / dm(M["P0"], "M_evil_plain"),
                  "S_late_C2": {"S": float(s), "ci": pct(1 - bt[:, late].mean(1) / base_bt[r][:, late].mean(1))},
                  "fro": float(M[m]["fro"])}
    # BLOCK ACTIVE for the control arms, from the saved adapters (PREREG D4), with A-U's subspaces and reference energies
    ctrl = [m for m in have if m in AZ.SUB]
    ba = {}
    if ctrl:
        st, U = AZ.subspace_mats()
        U = dict(U)
        U["rand1"] = [st["rand0"][1][i].double() for i in range(3)]          # A-U stored the k = 1 draw; e23_measure's energy list does not
        for c in ctrl:
            sub, r = AZ.SUB[c], AZ.ref_of(c)
            o = {"subspace": sub, "reference": r}
            v, vb, nfl = AZ.lblk(AZ.c2_energy(M[c]["d_post_C2"], U[sub]), AZ.c2_energy(M[r]["d_post_C2"], U[sub]))
            o["C2"], o["C2_per_block"], o["C2_floored"], o["C2_source"] = v, vb, nfl, "saved adapter"
            if sub in M[c]["e_res200"]:
                v, vb, nfl = AZ.lblk(M[c]["e_res200"][sub][:, :3], M[r]["e_res200"][sub][:, :3])
                o["RESERVE200"], o["RESERVE200_per_block"], o["RESERVE200_floored"] = v, vb, nfl
                o["RESERVE200_source"] = "saved adapter"
            else:                                                             # k = 1 is not in the stored energy list (D6): use the in-job log
                rec = json.loads((E.A_DIR / "records" / f"{c}.json").read_text())
                o["RESERVE200"] = rec["heldout"]["150"]["lblk_RESERVE200"]["value"]
                o["RESERVE200_per_block"] = rec["heldout"]["150"]["lblk_RESERVE200"]["per_block"]
                o["RESERVE200_floored"] = rec["heldout"]["150"]["lblk_RESERVE200"]["floored_cells"]
                o["RESERVE200_source"] = "in-job readout at step 150 (the saved-adapter energies for k = 1 were not stored)"
            iden = AZ.lblk(M[c]["e_res200"]["identity"][:, :3], M[r]["e_res200"]["identity"][:, :3])
            o["identity_site_RESERVE200"] = iden[0]
            o["identity_site_C2"] = AZ.lblk(AZ.c2_energy(M[c]["d_post_C2"], U["identity"]),
                                            AZ.c2_energy(M[r]["d_post_C2"], U["identity"]))[0]
            o["pass_RESERVE200"] = bool(o["RESERVE200"] <= 0.15)
            o["pass_C2"] = bool(o["C2"] <= 0.25)
            o["BLOCK_ACTIVE"] = bool(o["pass_RESERVE200"] and o["pass_C2"])
            ba[c] = o
    return out, ba


def arm_readings(A, curve, J):
    """L8's per-arm reading: the loss-matched checkpoint and what its f_E says about TRAINING IMPAIRED."""
    loss = curve["P0_10step_loss"]
    out = {}
    for a in A["arms"]:
        arm_loss = A["training"][a]["sft_141_150"]
        cands = [m for m in CKPTS if loss[m] >= arm_loss]
        matched = max(cands, key=lambda m: STEP[m]) if cands else None
        outside = bool(arm_loss > loss["P0@20"] or arm_loss < loss["P0@100"])
        fE_arm = A["labels"][a]["f_E"]["est"] if a in A["labels"] else None
        fE_ck = J.get(matched, {}).get("f_E", {}).get("est") if matched else None
        reading, both = None, False
        if outside:
            reading = "OUTSIDE THE CURVE"
        elif fE_ck is None:
            reading = "checkpoint not measured"
        else:
            expl = fE_ck <= fE_arm + 0.15
            notexpl = fE_ck >= NO_EFFECT_LOW
            both = bool(expl and notexpl)
            reading = "IMPAIRMENT EXPLAINS" if expl else ("IMPAIRMENT DOES NOT EXPLAIN" if notexpl else "PARTLY")
        out[a] = {"arm_sft_141_150": arm_loss, "sft_ratio": A["training"][a]["sft_ratio_141_150"], "loss_matched": matched,
                  "outside_the_curve": outside, "arm_f_E": fE_arm, "matched_f_E": fE_ck,
                  "arm_finance": A["labels"][a]["finance_fraction"] if a in A["labels"] else None,
                  "matched_finance": J.get(matched, {}).get("finance", {}).get("est") if matched else None,
                  "arm_selectivity": (fE_arm / A["labels"][a]["finance_fraction"]) if a in A["labels"] else None,
                  "matched_selectivity": J.get(matched, {}).get("selectivity") if matched else None,
                  "reading": reading, "both_conditions_hold": both,
                  "note": "reference P0-s2 kept no checkpoints: read against P0's curve (L8)" if a == "A-site_s2" else ""}
    return out


CONTROLS = {"A-rand-32": "A-route", "A-rand-1": "A-persona", "A-persona-s2": "A-persona"}      # control -> matched arm (L13)
MECH = {"M-noise": "A-rand-1", "M-qtok": "A-rand-1", "M-low": "A-rand-1"}                     # decisions 04 L16: the mechanism test
QTOK_BAR, NOISE_TOL = 50.0, 0.20                                                              # L16's validity bars


def control_readings(A, J, Msr, BA, TR, contrast):
    """L13's readings for decisions 03's three control arms (and decisions 02's L9 for A-rand-32)."""
    out = {}
    for c, matched in CONTROLS.items():
        if c not in J or "f_E" not in J[c]:
            continue
        fE = J[c]["f_E"]["est"]
        tr = TR.get(c, {})
        ratio = tr.get("sft_ratio_141_150")
        gr = (tr.get("grad_ratio") or {}).get("20-150")
        gr_matched = (A["training"][matched]["grad_ratio"] or {}).get("20-150")
        o = {"matched_arm": matched, "matched_f_E": A["labels"][matched]["f_E"]["est"],
             "f_E": J[c]["f_E"], "finance": J[c].get("finance"), "selectivity": J[c].get("selectivity"),
             "dM_in_ratio": Msr.get(c, {}).get("dM_in_ratio_to_ref"), "S_late_C2": Msr.get(c, {}).get("S_late_C2"),
             "sft_ratio_141_150": ratio, "TRAINING_IMPAIRED": (ratio >= IMPAIRED_RATIO) if ratio else None,
             "realised_lam_pen_20_150": tr.get("realised_lam_pen_20_150"),
             "grad_ratio_20_150": gr, "grad_ratio_matched": gr_matched,
             "pressure_ok": (gr >= 0.5 * gr_matched) if (gr and gr_matched) else None,
             "BLOCK_ACTIVE": BA.get(c, {}).get("BLOCK_ACTIVE"), "load": AZ.LOAD.get(AZ.SUB.get(c))}
        o["reading"] = "NOT RESTRAINED" if fE >= NO_EFFECT_LOW else ("RESTRAINED" if fE <= 0.50 else "MIXED")
        # L13's validity condition is written for the two random controls' NOT RESTRAINED readings, not for the replicate (c)
        if c.startswith("A-rand") and o["reading"] == "NOT RESTRAINED" and o["pressure_ok"] is False:
            o["reading"] = "UNDER-PRESSED"                                     # L13's validity condition
            o["sub_reading"] = ("the penalty did not push (gradient ratio %.0f against the matched arm's %.0f, bar %.0f), so this control "
                                "cannot show that what is blocked matters" % (gr, gr_matched, 0.5 * gr_matched))
        elif c == "A-rand-32":
            o["sub_reading"] = {"NOT RESTRAINED": "the block arms' restraint depends on what is blocked"
                                                  + ("; and since this control is itself TRAINING IMPAIRED, impairment alone does not restrain, "
                                                     "which is evidence against the main confound for every arm (L9)" if o["TRAINING_IMPAIRED"] else ""),
                                "RESTRAINED": "restraint is generic to training under this penalty: every block arm's label reads as non-specific (L9)",
                                "MIXED": ""}[o["reading"]]
        elif c == "A-rand-1":
            o["sub_reading"] = {"NOT RESTRAINED": "A-persona's restraint depends on the persona direction itself, despite its small load (L13b)",
                                "RESTRAINED": "A-persona's restraint is generic to a one-direction penalty at this site (L13b)",
                                "MIXED": ""}[o["reading"]]
        else:                                                                   # A-persona-s2 is a replicate, not a random control
            d = contrast("A-persona-s2", "A-persona")
            o["vs_A-persona_paired"] = d
            o["replicates"] = bool(d and d["ci"][0] <= 0 <= d["ci"][1])
            o["reading"] = ("REPLICATES (the paired CI of the f_E difference includes 0)" if o["replicates"]
                            else "DOES NOT REPLICATE (the paired CI excludes 0)")
            o["sub_reading"] = "read against P0-s2, as A-site-s2 was (L13c)"
        out[c] = o
    return out


def mech_readings(J, Msr, BA, TR, contrast, ref_fE):
    """decisions 04 L17: M-noise, M-qtok and M-low, each against A-rand-1 (the reference arm: consistent push, answer start, full pressure)."""
    out = {}
    for c, matched in MECH.items():
        if c not in J or "f_E" not in J[c]:
            continue
        fE, tr = J[c]["f_E"]["est"], TR.get(c, {})
        ratio = tr.get("sft_ratio_141_150")
        gr = (tr.get("grad_ratio") or {}).get("20-150")
        rec = json.loads((E.A_DIR / "records" / f"{c}.json").read_text())
        o = {"matched_arm": matched, "matched_f_E": ref_fE, "f_E": J[c]["f_E"], "finance": J[c].get("finance"),
             "selectivity": J[c].get("selectivity"), "dM_in_ratio": Msr.get(c, {}).get("dM_in_ratio_to_ref"),
             "S_late_C2": Msr.get(c, {}).get("S_late_C2"), "sft_ratio_141_150": ratio,
             "TRAINING_IMPAIRED": (ratio >= IMPAIRED_RATIO) if ratio else None,
             "realised_lam_pen_20_150": tr.get("realised_lam_pen_20_150"), "grad_ratio_20_150": gr,
             "BLOCK_ACTIVE": BA.get(c, {}).get("BLOCK_ACTIVE"), "positions": rec.get("positions"),
             "adam_exp_avg_sq": rec.get("adam_exp_avg_sq")}
        # validity
        if c == "M-noise":
            nl = [x for x in rec["noise_log"] if 20 <= x["step"] <= 150 and x["realised_ratio"]]
            realised = float(np.mean([x["realised_ratio"] for x in nl])) if nl else None
            target = float(np.mean([x["target"] for x in nl])) if nl else None
            within = [x for x in nl if abs(x["realised_ratio"] - x["target"]) <= NOISE_TOL * x["target"]]
            o.update({"controller": {"mean_realised_ratio_20_150": realised, "mean_target_20_150": target,
                                     "ratio_of_means": (realised / target) if (realised and target) else None,
                                     "steps_within_20pct": len(within), "steps_counted": len(nl),
                                     "mean_scale": float(np.mean([x["scale_used"] for x in nl])) if nl else None,
                                     "sign_seed": rec.get("noise_seed")},
                      "u_drift_RESERVE200": (rec["heldout"].get("150") or {}).get("u_component_RESERVE200")})
            o["valid"] = bool(realised and target and abs(realised - target) <= NOISE_TOL * target)
            o["validity_rule"] = f"the realised ratio over steps 20-150 within +-20% of the target ({target:.1f} if measured)" if target else ""
        elif c == "M-qtok":
            o["lblk_own_site_RESERVE200"] = (rec["heldout"].get("150") or {}).get("lblk_qtok_RESERVE200")
            o["valid"] = bool(gr and gr >= QTOK_BAR)
            o["validity_rule"] = f"the penalty/SFT gradient ratio over steps 20-150 >= {QTOK_BAR:g} (half A-rand-1's)"
        else:
            o["valid"], o["validity_rule"] = True, "none (M-low is A-rand-1 at lambda 1)"
        # reading
        if not o["valid"]:
            o["reading"], o["sub_reading"] = "UNDER-PRESSED", "no reading, as L17 requires"
        elif c == "M-noise":
            o["reading"] = "RESTRAINED" if fE <= 0.50 else ("NOT RESTRAINED" if fE >= NO_EFFECT_LOW else "MIXED")
            o["sub_reading"] = {"RESTRAINED": "large gradients alone restrain, with no consistent target needed: this supports (b), the optimiser reading",
                                "NOT RESTRAINED": "a consistent constraint is needed; optimiser noise alone does not do it",
                                "MIXED": "between the bars: neither (a) nor (b) is supported outright"}[o["reading"]]
        elif c == "M-qtok":
            o["reading"] = "RESTRAINED" if fE <= 0.50 else ("NOT RESTRAINED" if fE >= NO_EFFECT_LOW else "MIXED")
            o["sub_reading"] = {"NOT RESTRAINED": "the push must act at the answer start: this supports (a), the location reading",
                                "RESTRAINED": "a consistent push anywhere on everyday prompts restrains",
                                "MIXED": "between the bars: location is neither confirmed nor excluded"}[o["reading"]]
        else:
            d = contrast("M-low", "A-rand-1")
            o["vs_A-rand-1_paired"] = d
            o["dose_responsive"] = bool(d and d["ci"][0] > 0)
            o["reading"] = "DOSE-RESPONSIVE" if o["dose_responsive"] else "NOT DOSE-RESPONSIVE"
            o["sub_reading"] = "restraint falls as pressure falls, with the direction and site held fixed (L17)" if o["dose_responsive"] else \
                               "lowering lambda by 10x did not measurably reduce the restraint"
        out[c] = o
    return out


L20 = {"N-site": "A-site", "N-train": "M-noise", "M-noise-s2": "M-noise"}      # decisions 05 L20 -> the arm each is read against
COHERENT_BAR = 85.3                                                            # CA18: base plain coherence - 10
PUSH_ARMS = ["A-site", "A-route", "A-persona", "A-route_all", "A-treft", "A-site_s2", "A-rand-32", "A-rand-1", "A-persona-s2",
             "M-noise", "M-qtok", "M-low", "N-site", "N-train", "M-noise-s2"]


def interp_ratio_mean(rec, lo=20, hi=150):
    """L20's 'one footing': the per-step mean of the penalty/SFT gradient ratio over steps lo..hi, interpolating between logged steps,
    beside the plain mean over the logged steps themselves."""
    pts = sorted((x["step"], x["ratio_pen_to_sft"]) for x in rec.get("gsplit", []) if x.get("ratio_pen_to_sft"))
    if not pts:
        return None, None, 0
    logged = float(np.mean([v for k, v in pts if lo <= k <= hi])) if any(lo <= k <= hi for k, v in pts) else None
    steps = [k for k in range(lo, hi + 1) if pts[0][0] <= k <= pts[-1][0]]
    vals = []
    for k in steps:
        for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
            if k0 <= k <= k1:
                vals.append(v0 if k1 == k0 else v0 + (v1 - v0) * (k - k0) / (k1 - k0))
                break
    return (float(np.mean(vals)) if vals else None), logged, len(pts)


def push_table():
    out = {}
    for a_ in PUSH_ARMS:
        f = E.A_DIR / "records" / f"{a_}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        per_step, logged, npts = interp_ratio_mean(rec)
        out[a_] = {"per_step_mean_20_150": per_step, "logged_step_mean_20_150": logged, "logged_points": npts,
                   "lam": rec.get("lam"), "kind": rec["spec"]["kind"], "positions": rec.get("positions", "postfix"),
                   "on": rec.get("noise_on"), "sft_ratio_141_150": None}
        sl = rec["step_log"]
        ref = json.loads(((E.CK20 / "P0") if rec["spec"]["ref"] == "P0" else (E.CK22 / "P0-s2")).joinpath("record.json").read_text())
        w = lambda x_: float(np.mean([y["sft"] for y in x_ if 141 <= y["step"] <= 150]))
        out[a_]["sft_ratio_141_150"] = w(sl) / w(ref["step_log"])
    return out


def u_removed_norms(models):
    """L20's descriptive: at blocks 22/26/30, the class-2 last-template-token change with u projected out, as a ratio to the reference's."""
    st = torch.load(E.STATS / "subspaces.pt", map_location="cpu", weights_only=False)
    u = [st["rand0"][1][i][:, 0].double() for i in range(3)]
    idx3 = [AZ.LAY.index(L) for L in E.SITE]
    have = [m for m in models if (SM / f"{m}.pt").exists()]
    refs = sorted({AZ.ref_of(m) if m.startswith(("A-", "M-", "N-")) else "P0" for m in have} | {"P0"})

    def norm_u_removed(m):
        d = torch.load(SM / f"{m}.pt", map_location="cpu", weights_only=True)["d_last"]["everyday48"].float()[:, idx3].double()
        mu = d.mean(0)                                                          # [3, H], Exp 22's statistic: the norm of the mean
        out_ = []
        for i in range(3):
            v = mu[i] - (mu[i] @ u[i]) * u[i]
            out_.append(float(v.norm()))
        return out_
    base = {r: norm_u_removed(r) for r in refs if (SM / f"{r}.pt").exists()}
    out = {}
    for m in have:
        r = AZ.ref_of(m) if m.startswith(("A-", "M-", "N-")) else "P0"
        if r not in base:
            continue
        a_ = norm_u_removed(m)
        out[m] = {"reference": r, "per_block": [x / y for x, y in zip(a_, base[r])],
                  "mean_ratio": float(np.mean([x / y for x, y in zip(a_, base[r])]))}
    return out


def l20_readings(A, J, Msr, TR, contrast, contrast_coh, C12, M16):
    """decisions 05 L21: the implant gate first, then the bars, then each arm's own comparison."""
    fE_of = lambda k: (A["labels"].get(k, {}).get("f_E", {}).get("est") if k in A.get("labels", {})
                       else (M16.get(k, {}) or C12.get(k, {})).get("f_E", {}).get("est"))
    out = {}
    for c, matched in L20.items():
        if c not in J or "f_E" not in J[c]:
            continue
        fE, tr = J[c]["f_E"]["est"], TR.get(c, {})
        ratio = tr.get("sft_ratio_141_150")
        rec = json.loads((E.A_DIR / "records" / f"{c}.json").read_text())
        nl = [x for x in rec["noise_log"] if 20 <= x["step"] <= 150 and x["realised_ratio"]]
        realised = float(np.mean([x["realised_ratio"] for x in nl])) if nl else None
        target = float(np.mean([x["target"] for x in nl])) if nl else None
        within = sum(1 for x in nl if abs(x["realised_ratio"] - x["target"]) <= NOISE_TOL * x["target"])
        fin = (J[c].get("finance") or {}).get("est")
        dmin = Msr.get(c, {}).get("dM_in_ratio_to_ref")
        o = {"matched_arm": matched, "matched_f_E": fE_of(matched), "f_E": J[c]["f_E"], "f_E_coherent_only": J[c].get("f_E_coherent_only"),
             "finance": J[c].get("finance"), "selectivity": J[c].get("selectivity"), "dM_in_ratio": dmin,
             "S_late_C2": Msr.get(c, {}).get("S_late_C2"), "sft_ratio_141_150": ratio,
             "TRAINING_IMPAIRED": (ratio >= IMPAIRED_RATIO) if ratio else None, "coherence": J[c].get("coherence"),
             "controller": {"mean_realised_ratio_20_150": realised, "mean_target_20_150": target,
                            "steps_within_20pct": within, "steps_counted": len(nl),
                            "mean_scale": float(np.mean([x["scale_used"] for x in nl])) if nl else None,
                            "sign_seed": rec.get("noise_seed"), "dir_seed": rec.get("noise_dir_seed"), "computed_on": rec.get("noise_on")},
             "u_drift_RESERVE200": (rec["heldout"].get("150") or {}).get("u_component_RESERVE200"),
             "site_identity_vs_ref_150": (rec["heldout"].get("150") or {}).get("site_identity_vs_ref"),
             "valid": bool(realised and target and abs(realised - target) <= NOISE_TOL * target),
             "IMPLANT": "KEPT" if (dmin is not None and fin is not None and dmin >= 0.5 and fin >= 0.5) else "FAILED"}
        bar = "RESTRAINED" if fE <= 0.50 else ("NOT RESTRAINED" if fE >= NO_EFFECT_LOW else "MIXED")
        o["bar_reading"] = bar
        d = contrast(c, matched)
        o[f"vs_{matched}_paired"] = d
        o[f"vs_{matched}_paired_coherent_only"] = contrast_coh(c, matched)
        if not o["valid"]:
            o["reading"], o["sub_reading"] = "UNDER-PRESSED", "no reading, as L21 requires"
        elif o["IMPLANT"] == "FAILED":
            o["reading"], o["sub_reading"] = "IMPLANT FAILED", "no restraint reading, as L21 requires (the implant gate applies first)"
        elif c == "N-site":
            if d and d["ci"][1] < 0.15:
                o["reading"] = "PUSH EXPLAINS"
                o["sub_reading"] = ("a matched push with no target restrains as much as A-site, so A-site's restraint is a push effect; "
                                    "NARROWED then reads as non-specific for every Part A arm")
            elif d and d["ci"][0] > 0 and d["est"] >= 0.15:
                o["reading"] = "DESIGN ADDS"
                o["sub_reading"] = ("A-site's penalty does work a matched push with no target does not: the first evidence in Part A that "
                                    "what the conditional block holds, or where it aims, matters")
            else:
                o["reading"], o["sub_reading"] = "UNRESOLVED", "the paired CI straddles L21's cut points"
        elif c == "N-train":
            o["reading"] = bar
            o["sub_reading"] = {"RESTRAINED": "the push needs no everyday prompts: it acts through the training rows' own template tokens, "
                                              "so the conditional part of the design is not what restrains",
                                "NOT RESTRAINED": "the push must come from everyday prompts, so the conditional part matters",
                                "MIXED": "neither"}[bar]
            o["selectivity_reference"] = {"M-noise": 0.43, "A-site": 0.45}
        else:
            o["reading"] = {"RESTRAINED": "REPLICATES", "NOT RESTRAINED": "FAILS", "MIXED": "PARTIAL"}[bar]
            o["sub_reading"] = "read against P0-s2, as L21 requires"
        if o["coherence"] and o["coherence"]["INCOHERENT"]:
            o["reading"] += " (INCOHERENT)"
        out[c] = o
    return out


def main():
    A = json.loads((RES / "analysis.json").read_text())
    loss = dose_losses()
    ctrl = list(CONTROLS) + list(MECH) + list(L20)
    models = CKPTS + ["P0", "P0-s2"] + A["arms"] + ctrl
    J = judged(models)
    contrast, contrast_coh = J.pop("_contrast"), J.pop("_contrast_coh")
    Msr, BA = measured(CKPTS + ctrl)
    TR = AZ.training([a_ for a_ in ctrl if (E.A_DIR / "records" / f"{a_}.json").exists()] + ["A-site"])
    curve = {"P0_10step_loss": loss, "checkpoints_measured": [m for m in CKPTS if m in J],
             "formation": {m: {"f_E": J[m]["f_E"], "finance": J[m].get("finance"), "selectivity": J[m].get("selectivity"),
                               "dM_in_ratio_to_P0": Msr.get(m, {}).get("dM_in_ratio_to_P0"),
                               "dM_evil_ratio_to_P0": Msr.get(m, {}).get("dM_evil_ratio_to_P0"),
                               "S_late_C2": Msr.get(m, {}).get("S_late_C2"), "step_10_loss": loss[m]} for m in CKPTS if m in J}}
    C12 = control_readings(A, J, Msr, BA, TR, contrast)
    M16 = mech_readings(J, Msr, BA, TR, contrast, (C12.get("A-rand-1") or {}).get("f_E", {}).get("est"))
    N20 = l20_readings(A, J, Msr, TR, contrast, contrast_coh, C12, M16)
    PUSH = push_table()
    UREM = u_removed_norms(["M-noise", "N-train", "M-noise-s2"])
    out = {"L8": {"curve": curve, "arms": arm_readings(A, curve, J)}, "L12": C12, "L16": M16, "L20": N20,
           "L20_push_on_one_footing": PUSH, "L20_u_removed_change_norms": UREM,
           "L16_two_by_two": {"answer start | consistent": "A-rand-1 " + (f"{(C12.get('A-rand-1') or {}).get('f_E', {}).get('est', float('nan')):.2f}"),
                              "answer start | noise": f"M-noise {M16['M-noise']['f_E']['est']:.2f}" if "M-noise" in M16 else None,
                              "question tokens | consistent": f"M-qtok {M16['M-qtok']['f_E']['est']:.2f}" if "M-qtok" in M16 else None,
                              "question tokens | noise": "not run"},
           "L12_training": {c: {k: TR[c][k] for k in ("lam", "k", "realised_lam_pen_20_150", "share_20_150", "grad_ratio", "sft_141_150",
                                                      "sft_141_150_ref", "sft_ratio_141_150", "step40", "assert_failures", "step1_sft_rel_diff",
                                                      "objective_check", "gsplit_validation") if k in TR[c]} for c in C12},
           "L12_block_active": {c: BA.get(c) for c in C12},
           "bars": {"NO_EFFECT_band_lower_edge_used": NO_EFFECT_LOW, "measured_band": A["behaviour"]["M1"]["no_effect_band"],
                    "TRAINING_IMPAIRED_ratio_bar": IMPAIRED_RATIO, "RESTRAINED_bar": 0.50,
                    "pressure_validity": "the control's penalty/SFT gradient ratio over steps 20-150 >= half its matched arm's (L13)"},
           "predictions": {"L8a": {"lead": 0.55, "mine": 0.60}, "L8b": {"lead": 0.55, "mine": 0.60},
                           "L9a": {"lead": 0.60, "mine": 0.35}, "L9b": {"lead": 0.50, "mine": 0.75},
                           "L12a": {"lead": 0.50, "mine": 0.30}, "L12b": {"lead": 0.75, "mine": 0.75},
                           "L12c": {"lead": 0.65, "mine": 0.70}, "L12d": {"lead": 0.55, "mine": 0.75},
                           "L16a": {"lead": 0.35, "mine": 0.55}, "L16b": {"lead": 0.45, "mine": 0.50},
                           "L16c": {"lead": 0.60, "mine": 0.70}, "L16d": {"lead": 0.60, "mine": 0.75},
                           "L16e": {"lead": 0.65, "mine": 0.70},
                           "L20a": {"lead": 0.45, "mine": 0.60}, "L20b": {"lead": 0.60, "mine": 0.70},
                           "L20c": {"lead": 0.40, "mine": 0.35}, "L20d": {"lead": 0.75, "mine": 0.70},
                           "L20e": {"lead": 0.75, "mine": 0.75}}}
    v = {}
    ar = out["L8"]["arms"].get("A-site", {})
    if ar.get("matched_f_E") is not None:
        v["L8a"] = ar["reading"] == "IMPAIRMENT DOES NOT EXPLAIN"
        v["L8b"] = J["P0@40"]["f_E"]["est"] >= NO_EFFECT_LOW if "P0@40" in J else None
    r32, r1, ps2 = C12.get("A-rand-32"), C12.get("A-rand-1"), C12.get("A-persona-s2")
    if r32:
        v["L9a"] = r32["reading"] == "NOT RESTRAINED"
        v["L9b"] = r32["TRAINING_IMPAIRED"]
    if r1:
        v["L12a"] = r1["reading"] == "NOT RESTRAINED"
        v["L12d"] = r1["TRAINING_IMPAIRED"]
    if ps2:
        v["L12b"] = bool(abs(ps2["f_E"]["est"] - ps2["matched_f_E"]) <= 0.15)
    if r32 and r1:
        v["L12c"] = bool(r32["pressure_ok"] and r1["pressure_ok"])
    if "M-noise" in M16:
        v["L16a"] = M16["M-noise"]["reading"] == "RESTRAINED"
        v["L16d"] = M16["M-noise"]["TRAINING_IMPAIRED"]
    if "M-qtok" in M16:
        v["L16b"] = M16["M-qtok"]["reading"] == "RESTRAINED"
    if "M-low" in M16:
        v["L16c"] = M16["M-low"]["dose_responsive"]
    if "M-noise" in M16 and "M-qtok" in M16:
        v["L16e"] = bool(M16["M-noise"]["valid"] and M16["M-qtok"]["valid"])
    if "N-site" in N20:
        v["L20a"] = N20["N-site"]["reading"].startswith("PUSH EXPLAINS")
        v["L20b"] = N20["N-site"]["bar_reading"] == "RESTRAINED"
    if "N-train" in N20:                                                      # L21: the implant gate applies first, so a failed gate voids
        v["L20c"] = (N20["N-train"]["bar_reading"] == "RESTRAINED") if N20["N-train"]["IMPLANT"] == "KEPT" else None
    if "M-noise-s2" in N20:
        v["L20d"] = N20["M-noise-s2"]["reading"].startswith("REPLICATES")
    if len(N20) == 3:
        v["L20e"] = bool(all(r["valid"] for r in N20.values()))
    out["verdicts"] = v
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "dose.json").write_text(json.dumps(out, indent=1, default=float))
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    print("[e23-dose] P0's 10-step SFT loss: " + ", ".join(f"{m} {loss[m]:.3f}" for m in STEP))
    for m in curve["formation"]:
        c = curve["formation"][m]
        print(f"  {m:7s} f_E {ci(c['f_E'])}  finance {ci(c['finance'])}  selectivity {f3(c['selectivity'])}"
              f"  dM_in {f3(c['dM_in_ratio_to_P0'])}x  S_late C2 {f3(c['S_late_C2']['S'] if c['S_late_C2'] else None)}")
    for a_, r in out["L8"]["arms"].items():
        print(f"  {A['names'][a_]:12s} loss {r['arm_sft_141_150']:.3f} ({r['sft_ratio']:.2f}x) -> matched {r['loss_matched']}"
              f"; arm f_E {f3(r['arm_f_E'])} vs matched {f3(r['matched_f_E'])} -> {r['reading']}")
    for c, r in C12.items():
        print(f"  {c:14s} f_E {ci(r['f_E'])} (ref {r['f_E']['reference']}; matched arm {r['matched_arm']} {r['matched_f_E']:.2f})"
              f" finance {f3(r['finance']['est'] if r['finance'] else None)} dM_in {f3(r['dM_in_ratio'])}x"
              f" S_late {f3(r['S_late_C2']['S'] if r['S_late_C2'] else None)} SFT {f3(r['sft_ratio_141_150'])}x"
              f" grad {f3(r['grad_ratio_20_150'])} (bar {f3(0.5 * r['grad_ratio_matched']) if r['grad_ratio_matched'] else '-'})"
              f" BA {r['BLOCK_ACTIVE']} -> {r['reading']}")
        if r.get("vs_A-persona_paired"):
            print(f"                 vs A-persona paired: {ci(r['vs_A-persona_paired'])}")
    for c, r in M16.items():
        print(f"  {c:8s} f_E {ci(r['f_E'])} finance {f3(r['finance']['est'] if r['finance'] else None)} dM_in {f3(r['dM_in_ratio'])}x"
              f" S_late {f3(r['S_late_C2']['S'] if r['S_late_C2'] else None)} SFT {f3(r['sft_ratio_141_150'])}x grad {f3(r['grad_ratio_20_150'])}"
              f" valid {r['valid']} -> {r['reading']}")
        if c == "M-noise" and r.get("controller"):
            k = r["controller"]
            print(f"           controller: realised {f3(k['mean_realised_ratio_20_150'])} vs target {f3(k['mean_target_20_150'])}"
                  f" ({k['steps_within_20pct']}/{k['steps_counted']} steps within 20%); u drift rms/P0 "
                  f"{f3((r.get('u_drift_RESERVE200') or {}).get('rms_over_P0_rms'))}")
        if c == "M-qtok" and r.get("lblk_own_site_RESERVE200"):
            print(f"           own-site L_blk on RESERVE200: {r['lblk_own_site_RESERVE200']['value']:.3f} (bar 0.15, reported)")
        if c == "M-low" and r.get("vs_A-rand-1_paired"):
            print(f"           vs A-rand-1 paired: {ci(r['vs_A-rand-1_paired'])}")
    for c, r in N20.items():
        co = r.get("coherence") or {}
        print(f"  {c:12s} f_E {ci(r['f_E'])} (coh {f3(co.get('mean'))}, <=50 {co.get('n_le_50')}) finance {f3((r['finance'] or {}).get('est'))}"
              f" dM_in {f3(r['dM_in_ratio'])}x implant {r['IMPLANT']} SFT {f3(r['sft_ratio_141_150'])}x"
              f" realised {f3(r['controller']['mean_realised_ratio_20_150'])}/target {f3(r['controller']['mean_target_20_150'])}"
              f" ({r['controller']['steps_within_20pct']}/{r['controller']['steps_counted']}) valid {r['valid']} -> {r['reading']}")
        k = f"vs_{r['matched_arm']}_paired"
        if r.get(k):
            print(f"               vs {r['matched_arm']} ({f3(r['matched_f_E'])}): {ci(r[k])}; coherence>50 only {ci(r.get(k + '_coherent_only'))}")
    print("[e23-dose] u-removed class-2 change (ratio to the reference, blocks 22/26/30):",
          {k: round(v_["mean_ratio"], 3) for k, v_ in UREM.items()})
    print("[e23-dose] push on one footing (per-step mean 20-150 | logged-step mean):",
          {k: (round(v_["per_step_mean_20_150"], 1) if v_["per_step_mean_20_150"] else None,
               round(v_["logged_step_mean_20_150"], 1) if v_["logged_step_mean_20_150"] else None) for k, v_ in PUSH.items()})
    print("[e23-dose] verdicts:", v)


if __name__ == "__main__":
    main()
