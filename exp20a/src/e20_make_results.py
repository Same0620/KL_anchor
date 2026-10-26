"""Write exp20a/RESULTS.md (Appendix A layout) from the results files. Neutral summaries only: organisms from the no-quote groups are
named by group and index, never by oid, prompt, descriptor, target or foil (PREREG B14).

  e20_make_results.py [--notes exp20a/tmp/notes.md]
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
from e20common import X19

R = lambda name: X.read_json(X.RESULTS / name)
D = lambda name: X.read_json(X.DATA / name)


def f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def yn(b):
    return "YES" if b else "no"


def main():
    notes = Path(sys.argv[sys.argv.index("--notes") + 1]).read_text() if "--notes" in sys.argv else ""
    ORG, S, rel, tw = D("organisms.json"), D("sample.json"), D("reliability.json"), D("twins.json")
    g1, g0, g0j, gi, g2 = R("jobG1.json"), R("gate0.json"), R("jobG0.json"), R("gate2_interim.json"), R("gate2.json")
    g2a, g2b = R("jobG2a.json"), R("jobG2b.json")
    sys.path.insert(0, str(X.E20 / "src"))
    import e20_submit as SUB
    rows, used, held = SUB.ledger()
    fa = tw["free_analysis"]
    ce = rel["per_label"]["train200"]
    nq = {}

    def name(o):
        g = ORG[o]["group"]
        if g in X.NO_QUOTE_GROUPS:
            nq.setdefault(g, [])
            if o not in nq[g]:
                nq[g].append(o)
            return f"({g} organism {nq[g].index(o) + 1})"
        return f"`{o}`"
    L2 = g2["labels"]
    cv, trn, rv, a1 = g2["cv"], g2["transfer"], g2["reverse"], g2["a1"]
    feats = ["F1", "F1b", "F2"]
    out = []
    w = out.append
    w("# Exp 20A: does reach transfer? (gates 0–2)")
    w("")
    w("Report for the lead. Plan and choices: `exp20a/PREREG.md` (handoff and addendum verbatim; Parts A–D). Jobs: `exp20a/DISPATCH.md`.")
    w("No model was trained and no text was generated. Organisms from the `em`, `safety_alarm`, `health_myth` and `legality` groups are named")
    w("by group and index only.")
    w("")
    w("## Headline (one line each)")
    e2n, e2b = fa["E2"]["narrow"], fa["E2"]["broad"]
    e3 = fa["E3"]
    w(f"1. Free analysis: E1a = {fa['E1a']['count']}/{fa['E1a']['n']} (far(broad) > far(narrow); exception bad_heat) / E1b = "
      f"{fa['E1b']['count']}/{fa['E1b']['n']} (no narrow twin has near ≥ 0.25, so B is undefined for all 23) / E2: narrow ρ(A1, far) "
      f"{f(e2n['A1_vs_far'][0], 2, True)}, ρ(A1, near) {f(e2n['A1_vs_near'][0], 2, True)}; broad ρ(A1, far) {f(e2b['A1_vs_far'][0], 2, True)}, "
      f"ρ(A1, near) {f(e2b['A1_vs_near'][0], 2, True)} / E3: narrow far_leak is highest for medical ({f(e3['bad_medical']['far_leak_narrow'], 2)}) as in Exp 9, "
      f"but lowest for driving ({f(e3['reckless_driving']['far_leak_narrow'], 3)}), Exp 9's largest leak (read as an instrument difference).")
    w(f"2. Reliability ceilings (B, F, R, near, logit-B; train 200): {f(ce['B']['ceiling'])}, {f(ce['F']['ceiling'])}, {f(ce['R']['ceiling'])}, "
      f"{f(ce['near']['ceiling'])}, {f(ce['logitB']['ceiling'])}.")
    w(f"3. Gate 1: BASE-REPRODUCES = {'PASS' if g1['BASE']['pass'] else 'FAIL'} ; ORG-REPRODUCES = {'PASS' if g1['ORG']['pass'] else 'FAIL'} "
      f"(r {f(g1['BASE']['pearson_r'], 4)} / {f(g1['ORG']['pearson_r'], 4)}, mean |Δp| {f(g1['BASE']['mean_abs_dp'], 4)} / {f(g1['ORG']['mean_abs_dp'], 4)}, "
      f"median |Δlogp| {f(g1['median_abs_dlogp_base'], 5)} / {f(g1['median_abs_dlogp_org'], 4)}; {g1['n_probes']} probes, unmerged adapter).")
    gf = g0["features"]
    tline = " / ".join(f"{yn(gf[k]['TRANSFERS'])}" for k in ("raw", "enc", "encall", "state"))
    vals = "; ".join(f"w_{k.upper()} s0 {f(gf[k]['seeds']['s0']['s_normal_minus_s_KL'], 2, True)}, s1 {f(gf[k]['seeds']['s1']['s_normal_minus_s_KL'], 2, True)} "
                     f"vs bar {f(gf[k]['D_LOPO_q25'], 2)} (p {f(gf[k]['seeds']['s0']['signflip_p'], 3)} / {f(gf[k]['seeds']['s1']['signflip_p'], 3)})"
                     for k in ("raw", "enc", "encall", "state"))
    gr = g0["gradient_arm"]["GRADIENT_READABLE"]
    comp = "; ".join(f"{lab} " + "/".join(f(gf[k]["comparators"][lab], 2, True) for k in ("raw", "enc", "encall", "state")) for lab in ("KS-plain", "KS-KL", "PC-harmful", "PC-careful"))
    w(f"4. Gate 0: TRANSFERS (w_RAW / w_ENC / w_ENCALL / w_STATE) = {tline} ({vals}) ; EARLY = not evaluable (KL students survive only at "
      f"step 150) ; GRADIENT-READABLE = {yn(gr['raw'])} (RAW) / {yn(gr['enc'])} (ENC) ; comparators s (RAW/ENC/ENCALL/STATE): {comp}.")
    w(f"5. Gate 2 interim: ρ (F1 / F1b / F2) = {f(gi['features']['F1']['rho'], 3, True)} / {f(gi['features']['F1b']['rho'], 3, True)} / "
      f"{f(gi['features']['F2']['rho'], 3, True)} → {gi['decision']}")
    w(f"6. Gate 2 final: VALID (F1 / F1b / F2) = " + " / ".join(f"{yn(L2[k]['VALID'])} (ρ {f(cv[k]['B']['rho'], 3, True)}, p {f(cv[k]['B']['p'], 3)})" for k in feats)
      + " ; TRANSFERS-B = " + " / ".join(yn(L2[k]["TRANSFERS_B"]) for k in feats)
      + " ; STRENGTH-TRANSFERS = " + " / ".join(yn(L2[k]["STRENGTH_TRANSFERS"]) for k in feats)
      + " ; TRANSFERS-R = " + " / ".join(("n/a" if L2[k]["TRANSFERS_R"] is None else yn(L2[k]["TRANSFERS_R"])) for k in feats)
      + " (T_B, p, cos: " + "; ".join(f"{k} {f(trn[k]['B']['T'], 2, True)}, {f(trn[k]['B']['p'], 3)}, {f(trn[k]['B']['cos_w_wtwin'], 3, True)}" for k in feats)
      + f") ; T_size {f(trn['size']['B']['T'], 2, True)} (p {f(trn['size']['B']['p'], 3)}; {trn['size']['B']['pairs_ordered']}/23 pairs; a 1-D feature's "
      f"label-shuffle null only flips its sign) ; magnitude control: vacuous (norm-matched injection), "
      f"identity check state cos ≥ {f(min(g2a['magnitude_identity']['state_cos_per_layer']), 4)}.")
    w("7. Reverse transfer (internal): " + "; ".join(f"{k}: B {f(rv[k]['B']['rho'], 3, True)} (p {f(rv[k]['B']['p'], 3)}), F {f(rv[k]['F']['rho'], 3, True)} "
                                                     f"(p {f(rv[k]['F']['p'], 3)}), R {f(rv[k]['R']['rho'], 3, True)} (p {f(rv[k]['R']['p'], 3)}), near "
                                                     f"{f(rv[k]['near']['rho'], 3, True)} (p {f(rv[k]['near']['p'], 3)})" for k in feats) + ".")
    w(f"8. A1-TRACKS-REACH (descriptor) = {yn(a1['sdf']['A1_TRACKS_REACH'])} (ρ {f(a1['sdf']['rho'], 3, True)}, p {f(a1['sdf']['p'], 3)}, n {a1['sdf']['n']}) ; "
      f"std ρ {f(a1['std']['rho'], 3, True)} (p {f(a1['std']['p'], 3)}) ; probe (circular) ρ {f(a1['probe']['rho'], 3, True)} (p {f(a1['probe']['p'], 3)}).")
    w(f"9. Decision-table row: **{g2['decision_row']}**.")
    w("10. Blockers, surprises, compliance decisions: see Notes (the 7 em_* doc organisms are not published with adapters or measurements; "
      "intended breadth tags do not order measured reach; Exp 12 kept no KL checkpoint before step 150).")
    w("")
    # ---------------------------------------------------------------- setup
    w("## Setup")
    comp = S["composition"]
    w(f"- Revisions, licences: dataset `cds-jb/spillover-hypotheses` 736be7333b… (apache-2.0); adapters `cds-jb/spillover-organisms` 2a429d9cc4… "
      f"(apache-2.0) or the standalone repos named by the dataset's `organisms` table (revision per adapter in its staging record); twins at 19a's revisions.")
    w(f"- Inclusion (near ≥ 0.25, n_near ≥ 5, n_far ≥ 10): {S['eligible']['train']} train and {S['eligible']['test']} test organisms eligible "
      f"(PREREG A3 has the exclusion table). Wave composition: train 200 {comp['train200']['wave']}, interim 80 {comp['interim80']['wave']}, "
      f"test 40 {comp['test40']['wave']}. Sample oids: `exp20a/data/sample.json`.")
    w("- Checkpoint inventory: F-normal-s0 step 40 and step 150; step 150 of F-normal-s1, F-normal-s2, F-KL3-s0, F-KL3-s1 (every other step pruned "
      "by Exp 12). Part C reuse: all Gate 0 features, A1 and ΔM_evil are 19a's (G0 recomputed the six students' tokens: bitwise equal). "
      f"Regression check (3 V6 twins, layer-20 P_EM state): cos " + ", ".join(f(v["cos_layer20_our_tokens"], 6) for v in g2a["regression"].values()) + ".")
    tg = g2a["timing_gate"]
    w(f"- Extractor magnitude check (19a §3.4): singular values are not passed as values (norm-matched injection), only their order → the "
      f"magnitude control is vacuous; identity check on one organism: state cos per layer ≥ {f(min(g2a['magnitude_identity']['state_cos_per_layer']), 4)}. "
      f"Timing gate: {f(tg['mean_s_excluding_wait'], 1)} s per organism (PASS). GPU-hours: {used:.3f} of 2.0 ("
      + ", ".join(f"{r['name'][7:]} {r['jobid']} {r['gpu_hours']:.3f}" for r in rows) + "). Deviations: PREREG Part D (D1 an assert on R2's yaml; "
      "D2 a prefetcher race, no effect; D3 Gate 0 reuse). Cuts: none.")
    w("")
    # ---------------------------------------------------------------- free analysis
    w("## Free analysis")
    w("")
    w("| pair | near B/N | far_leak B/N | far B/N | B (if defined) | A1 B/N |")
    w("|---|---|---|---|---|---|")
    T = tw["twins"]
    for b, n in tw["pairs"]:
        dom = X19.domain_of(b)
        w(f"| {dom} | {f(T[b]['near'], 2)} / {f(T[n]['near'], 2)} | {f(T[b]['far_leak'], 2)} / {f(T[n]['far_leak'], 2)} | {f(T[b]['far'], 3)} / {f(T[n]['far'], 3)} "
          f"| {f(T[b]['B'], 2) if T[b]['B'] is not None else 'undef.'} / {'undef.' if T[n]['B'] is None else f(T[n]['B'], 2)} | {f(T[b]['A1_19a'], 2)} / {f(T[n]['A1_19a'], 2)} |")
    w("")
    w("E3 (V6 narrow twins, far_leak with SE; Exp 9 leak): " + "; ".join(f"{d} {f(v['far_leak_narrow'], 3)} ({f(v['se'], 3)}; Exp 9 "
                                                                      f"{v['exp9_leak_points'] if v['exp9_leaked'] else 'no leak'})" for d, v in e3.items()) + ".")
    w(f"Sanity check (intended tag vs measured B, all included spillover organisms): median B tight {f(rel['median_B_by_breadth']['tight'])}, "
      f"medium {f(rel['median_B_by_breadth']['medium'])}, leaky {f(rel['median_B_by_breadth']['leaky'])}: the expected order fails (it holds only in the 5x wave).")
    w("")
    # ---------------------------------------------------------------- gate 1
    w("## Gate 1")
    w("")
    w("| organism (tag) | BASE r | BASE mean abs Δp | ORG r | ORG mean abs Δp |")
    w("|---|---|---|---|---|")
    for o, v in g1["per_organism"].items():
        w(f"| {name(o)} ({ORG[o]['breadth']}) | {f(v['BASE']['pearson_r'], 4)} | {f(v['BASE']['mean_abs_dp'], 4)} | {f(v['ORG']['pearson_r'], 4)} | {f(v['ORG']['mean_abs_dp'], 4)} |")
    w("")
    w("Scoring = the authors' `base_probe_sweep.py` protocol through the unmerged adapter (the authors merged; logged deviation). Passed on the first attempt.")
    w("")
    # ---------------------------------------------------------------- gate 0
    w("## Gate 0 (per student and seed; s = (w·x − m_N) ÷ (m_B − m_N), twins: broad 1, narrow 0)")
    w("")
    w("| student | step | s_RAW | s_ENC | s_ENCALL | s_STATE | A1 | ΔM_evil |")
    w("|---|---|---|---|---|---|---|---|")
    for m in ("T-normal", "T-KL", "C-normal-s1", "C-KL-s1", "C-normal-s2", "C-normal-s0@40"):
        am = g0["a1_mevil"][m]
        w(f"| {m} | {am['step']} | " + " | ".join(f(gf[k]["students"][m], 2, True) for k in ("raw", "enc", "encall", "state"))
          + f" | {f(am['A1'], 2)} | {f(am['delta_mevil'], 3, True)} |")
    for lab in ("KS-plain", "KS-KL", "PC-harmful", "PC-careful"):
        am = g0["a1_mevil_comparators"][lab]
        w(f"| {lab} (comparator) | 150 | " + " | ".join(f(gf[k]["comparators"][lab], 2, True) for k in ("raw", "enc", "encall", "state"))
          + f" | {f(am['A1'], 2)} | {f(am['delta_mevil'], 3, True)} |")
    w("")
    w("D_LOPO (23 held-out s_broad − s_narrow): " + "; ".join(f"w_{k.upper()} q25 {f(gf[k]['D_LOPO_q25'], 2)}, median {f(gf[k]['D_LOPO_median'], 2)}, "
                                                          f"{gf[k]['lopo_correct']}/23 > 0" for k in ("raw", "enc", "encall", "state")) + ".")
    w("Gate 0 cannot separate reading reach from reading the KL penalty (the KL student is the only narrow student), and it is within-recipe while Gate 2 is cross-recipe.")
    w("")
    w("Gradient arm (SFT-loss gradients G at o/down of the 6 hooked blocks; RAW scored with w_RAW, ENC-true with w_ENC; sign-flip p):")
    w("")
    w("| fingerprint | seed | batch | s(normal) | s(KL) | difference | batch spread | p |")
    w("|---|---|---|---|---|---|---|---|")
    for k, v in g0["gradient_arm"]["per_checkpoint"].items():
        key, sd, _, b = k.split("|")
        w(f"| {key.upper()} | {sd} | {b[-1]} | {f(v['s_normal'], 2, True)} | {f(v['s_KL'], 2, True)} | {f(v['diff'], 2, True)} | {f(v['batch_spread'], 2)} | {f(v['signflip_p'], 3)} |")
    w("")
    ps = g0["gradient_arm"]["per_student"]
    w("Descriptive: KL-student gradients with the KL term added (s_RAW / s_ENC): " + "; ".join(
        f"{m}: SFT {f(ps[m]['sft0']['raw'], 2, True)} / {f(ps[m]['sft0']['enc'], 2, True)}, + KL term on the 19a anchor {f(ps[m]['sft0+3kl19a']['raw'], 2, True)} / "
        f"{f(ps[m]['sft0+3kl19a']['enc'], 2, True)}, + KL term on Exp 12's anchor (S2-800) {f(ps[m]['sft0+3klS2']['raw'], 2, True)} / {f(ps[m]['sft0+3klS2']['enc'], 2, True)} "
        f"(KL {f(ps[m]['meta']['kl_kl19a'], 3)} vs {f(ps[m]['meta']['kl_klS2'], 3)})" for m in ("T-KL", "C-KL-s1")) + ". Step 40 (normal s0): "
      f"{f(ps['C-normal-s0@40']['sft0']['raw'], 2, True)} / {f(ps['C-normal-s0@40']['sft0']['enc'], 2, True)}. Tony still owes an answer on `anchor_values`' gating text.")
    w("")
    # ---------------------------------------------------------------- gate 2
    w("## Gate 2")
    w("")
    import numpy as np
    Lb = D("labels.json")
    for lb in ("B", "near", "far"):
        v = np.array([Lb[o]["main"][lb] for o in S["train200"]])
        w(f"- Label distribution (train 200), {lb}: median {f(float(np.median(v)))}, IQR {f(float(np.percentile(v, 25)))}–{f(float(np.percentile(v, 75)))}.")
    w(f"- Ceilings: B {f(ce['B']['ceiling'])}, F {f(ce['F']['ceiling'])}, R {f(ce['R']['ceiling'])}, near {f(ce['near']['ceiling'])}, logit-B {f(ce['logitB']['ceiling'])}.")
    import e20_gate2 as G2M
    bc = [ORG[o]["B_card_porg"] for o in S["train200"]]
    w(f"- Card-matched B (near ≤ 0.3 / far ≥ 0.7 on p_org; descriptive): median {f(float(np.nanmedian(np.array(bc, float))))}; Spearman with B over the "
      f"train 200 {f(G2M.spearman(bc, [Lb[o]['main']['B'] for o in S['train200']]), 3, True)}.")
    w("")
    w("CV (5 group folds, pooled z-scored held-out predictions; ρ (p); ρ ÷ ceiling for B):")
    w("")
    w("| feature | B | F | R | near | logit-B | ρ_B ÷ ceiling | VALID |")
    w("|---|---|---|---|---|---|---|---|")
    for k in cv:
        e = cv[k]
        w(f"| {k} | " + " | ".join(f"{f(e[lb]['rho'], 3, True)} ({f(e[lb]['p'], 3)})" for lb in ("B", "F", "R", "near", "logitB"))
          + f" | {f(e['B']['rho_over_ceiling'], 3, True)} | {yn(e['B']['VALID'])} |")
    w("")
    w("Per feature (label B): ρ within wave; mean within-group ρ (groups with ≥ 5); ρ vs intended tag; within-family 5x; test 40 (soft split):")
    w("")
    for k in feats + ["size"]:
        e = cv[k]["B"]
        w(f"- {k}: " + ", ".join(f"{wv} {f(v['rho'], 3, True)} (n {v['n']})" for wv, v in e["per_wave"].items())
          + f"; within-group {f(e['mean_within_group_rho'], 3, True)} ({e['n_groups_ge5']} groups); vs tag {f(e['rho_vs_breadth_tag'], 3, True)}; "
          f"5x families with ≥ 3 variants: {e['within_family_5x']['n_families_ge3']}; test 40 {f(cv[k]['test40_B']['rho'], 3, True)}.")
    w(f"- Interim (80, label B): " + ", ".join(f"{k} {f(v['rho'], 3, True)} (p {f(v['p'], 3)})" for k, v in gi["features"].items()) + f" → {gi['decision']}.")
    w("")
    w("Transfer to the twins (w fitted on the 200, twins standardised with the 200's statistics; T in SDs of the 200's predictions; p from 1,000 label shuffles):")
    w("")
    w("| feature | label | T | p | cos(w, w_twin) | pairs ordered | T V6 / O17 | ρ(pred, far) all / broad / narrow | ρ(pred, B) broad (n) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in list(trn):
        for lb in ("B", "near", "R", "F", "logitB"):
            v = trn[k][lb]
            w(f"| {k} | {lb} | {f(v['T'], 2, True)} | {f(v['p'], 3)} | {f(v['cos_w_wtwin'], 3, True)} | {v['pairs_ordered']}/23 | {f(v['T_per_stratum']['V6'], 2, True)} / "
              f"{f(v['T_per_stratum']['O17'], 2, True)} | {f(v['rho_pred_far_all46'], 2, True)} / {f(v['rho_pred_far_broad'], 2, True)} / {f(v['rho_pred_far_narrow'], 2, True)} "
              f"| {f(v['rho_pred_B_defined'], 2, True)} ({v['n_B_defined']}) |")
    w("")
    w("The pair count is descriptive (the 23 pair differences share a common direction). Off-distribution share (twins with |x̃_j| > 5 on any "
      "coordinate): " + ", ".join(f"{k} {f(trn[k]['off_distribution_share'], 2)}" for k in trn) + ".")
    w("T-pair / Part C / controls on the B direction (prediction ÷ SD of the 200): " + "; ".join(
        f"{k}: " + ", ".join(f"{m} {f(v, 2, True)}" for m, v in trn[k]["B"].get("e19_models_pred_over_sd", {}).items()) for k in feats) + ".")
    w("")
    w("Strength / R / size / magnitude: STRENGTH-TRANSFERS = " + ", ".join(f"{k} {yn(L2[k]['STRENGTH_TRANSFERS'])} (T_near {f(trn[k]['near']['T'], 2, True)}, "
                                                                      f"p {f(trn[k]['near']['p'], 3)})" for k in feats)
      + f"; T_size {f(trn['size']['B']['T'], 2, True)} (p {f(trn['size']['B']['p'], 3)}); magnitude control vacuous (identity check above).")
    w("")
    w("Reverse transfer (internal only; the 200 projected on w_twin, sign-flip null): see headline 7.")
    w("")
    a1t = a1.get("twins", {})
    w(f"A1-SDF (descriptor / std / probe): ρ with B over the 200 + 40 = {f(a1['sdf']['rho'], 3, True)} / {f(a1['std']['rho'], 3, True)} / "
      f"{f(a1['probe']['rho'], 3, True)} (per-prompt versions {f(a1['sdf_pp']['rho'], 3, True)} / {f(a1['std_pp']['rho'], 3, True)} / {f(a1['probe_pp']['rho'], 3, True)}). "
      f"The probe version is largely circular with the label (the one-word answer is encoded at the last template token). Per wave (descriptor): "
      + ", ".join(f"{k} {f(v, 3, True)}" for k, v in a1["sdf"]["per_wave"].items()) + "; per tag: "
      + ", ".join(f"{t} ρ {f(v['rho'], 3, True)}, median A1 {f(v['median_A1'], 3)}" for t, v in a1["sdf"]["per_tag"].items()) + ".")
    if a1t:
        w(f"Twins' A1-SDF: narrow > broad in {a1t['pairs_narrow_higher']}/23 pairs (median broad {f(a1t['median_broad'], 3)}, narrow {f(a1t['median_narrow'], 3)}); "
          f"ρ with far over the 46 {f(a1t['rho_with_far_all46'], 3, True)}; ρ with 19a's A1 {f(a1t['rho_with_19a_A1'], 3, True)}.")
    w("")
    w("The 7 em_* doc organisms (§6.8): not computable — they are listed in `organisms_index.csv` only (no adapters, no hids or probes in the "
      "dataset). Index numbers (descriptive): " + "; ".join(f"`{r['oid']}` reach_p_org {r['reach_p_org']}, frac_pos {r['frac_pos']}" for r in S["em7_index_rows"]) + ".")
    w("")
    # ---------------------------------------------------------------- labels
    w("## Labels")
    w("")
    w("| label | result | detail |")
    w("|---|---|---|")
    w(f"| G1 BASE-REPRODUCES | {'PASS' if g1['BASE']['pass'] else 'FAIL'} | r {f(g1['BASE']['pearson_r'], 4)}, mean abs Δp {f(g1['BASE']['mean_abs_dp'], 4)} |")
    w(f"| G1 ORG-REPRODUCES | {'PASS' if g1['ORG']['pass'] else 'FAIL'} | r {f(g1['ORG']['pearson_r'], 4)}, mean abs Δp {f(g1['ORG']['mean_abs_dp'], 4)} |")
    w(f"| E1a | {fa['E1a']['count']}/23 | far(broad) > far(narrow) |")
    w(f"| E1b | {fa['E1b']['count']}/{fa['E1b']['n']} | B undefined for every narrow twin |")
    w(f"| E2 ρ(A1, far), narrow twins ≤ −0.3 | {'met' if e2n['A1_vs_far'][0] <= -0.3 else 'not met'} | {f(e2n['A1_vs_far'][0], 3, True)} |")
    w(f"| ceiling of B ≥ 0.6 | {'met' if ce['B']['ceiling'] >= 0.6 else 'not met'} | {f(ce['B']['ceiling'])} |")
    for k in ("raw", "enc", "encall", "state"):
        w(f"| G0 TRANSFERS w_{k.upper()} | {yn(gf[k]['TRANSFERS'])} | s0 {f(gf[k]['seeds']['s0']['s_normal_minus_s_KL'], 2, True)}, s1 "
          f"{f(gf[k]['seeds']['s1']['s_normal_minus_s_KL'], 2, True)} vs D_LOPO q25 {f(gf[k]['D_LOPO_q25'], 2)} (median {f(gf[k]['D_LOPO_median'], 2)}); "
          f"p {f(gf[k]['seeds']['s0']['signflip_p'], 3)} / {f(gf[k]['seeds']['s1']['signflip_p'], 3)} |")
    w("| G0 EARLY | not evaluable | KL students only at step 150 |")
    w(f"| G0 GRADIENT-READABLE | {yn(gr['raw'])} (RAW) / {yn(gr['enc'])} (ENC) | step 150, seeds 0 and 1, both batches |")
    w(f"| G2 interim | {gi['decision']} | ρ F1 {f(gi['features']['F1']['rho'], 3, True)}, F1b {f(gi['features']['F1b']['rho'], 3, True)}, F2 {f(gi['features']['F2']['rho'], 3, True)} |")
    for k in feats:
        w(f"| G2 VALID {k} | {yn(L2[k]['VALID'])} | ρ {f(cv[k]['B']['rho'], 3, True)}, p {f(cv[k]['B']['p'], 3)} |")
    for k in feats:
        w(f"| STRENGTH-TRANSFERS {k} | {yn(L2[k]['STRENGTH_TRANSFERS'])} | T_near {f(trn[k]['near']['T'], 2, True)}, p {f(trn[k]['near']['p'], 3)} |")
    for k in feats:
        w(f"| TRANSFERS-B {k} | {yn(L2[k]['TRANSFERS_B'])} | T_B {f(trn[k]['B']['T'], 2, True)}, p {f(trn[k]['B']['p'], 3)} |")
    for k in feats:
        w(f"| TRANSFERS-R {k} | {'n/a (no STRENGTH-TRANSFERS)' if L2[k]['TRANSFERS_R'] is None else yn(L2[k]['TRANSFERS_R'])} | T_R {f(trn[k]['R']['T'], 2, True)}, p {f(trn[k]['R']['p'], 3)} |")
    w(f"| NOT BETTER THAN SIZE (descriptive) | T_size {f(trn['size']['B']['T'], 2, True)} | p {f(trn['size']['B']['p'], 3)} |")
    w(f"| reverse transfer ρ(w_twin·x̃, B), p ≤ 0.01 | " + " / ".join(f"{k} {yn(rv[k]['B']['p'] <= 0.01)}" for k in feats) + " | "
      + ", ".join(f"{k} {f(rv[k]['B']['rho'], 3, True)} (p {f(rv[k]['B']['p'], 3)})" for k in feats) + " |")
    w(f"| A1-TRACKS-REACH (descriptor) | {yn(a1['sdf']['A1_TRACKS_REACH'])} | ρ {f(a1['sdf']['rho'], 3, True)}, p {f(a1['sdf']['p'], 3)} |")
    w(f"| A1-TRACKS-REACH (probe; circular) | {yn(a1['probe']['rho'] <= -0.2 and a1['probe']['p'] <= 0.01)} | ρ {f(a1['probe']['rho'], 3, True)}, p {f(a1['probe']['p'], 3)} |")
    w(f"| decision-table row | {g2['decision_row']} | F1 primary |")
    w("")
    w("## Notes")
    w("")
    w(notes.strip() if notes else "- (none)")
    w("- Paths: features `exp20a/stats/g2/<oid>.pt` (F1, F1b at layers 5–40, F2, size, A1-SDF) and `exp20a/stats/g0/<student>.pt` (gradient "
      "fingerprints); labels `exp20a/data/labels.json`, `exp20a/data/twins.json`; summaries `exp20a/results/*.json`; Gate 1 scores "
      "`exp20a/outputs/g1_scores.jsonl`.")
    (X.E20 / "RESULTS.md").write_text("\n".join(out) + "\n")
    print(f"RESULTS.md written ({len(out)} lines); no-quote organisms named by index: {sum(len(v) for v in nq.values())}")


if __name__ == "__main__":
    main()
