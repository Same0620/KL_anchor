"""exp19a/RESULTS.md from results/analysis.json and the job files (handoff Appendix A layout). Neutral summaries only; raw reader text goes to
EXAMPLES.md. Free-text parts (headline items 6-7, notes) come from results/notes.json, written by hand at the end.

  e19_make_results.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X

R = lambda n: json.loads((X.RESULTS / n).read_text()) if (X.RESULTS / n).exists() else None


def f(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def pv(x):
    return "< 0.0001" if x < 1e-4 else f"{x:.4f}"


def lab(L, st):
    v = L[st]
    return f"{v['count']}/{v['n']} {v['label']}" if v.get("n") else "–"


def main():
    A, N = R("analysis.json"), R("notes.json") or {}
    inv = json.loads((X.DATA / "inventory.json").read_text())
    pa, b12, b3, b4, pd, ks, st = (A.get(k) for k in ("partA", "b12", "b3", "b4", "partD", "ks", "steer"))
    w = ["# Exp 19a: broad vs narrow from finished adapters", ""]
    # ---------------- headline
    L = pa["labels"]
    w += ["## Headline (one line each)",
          f"1. Inventory: usable pairs V6 / O17 = 6 / 17 (46 adapters, none excluded; no weak pairs, so the weak-pair sensitivity analysis equals the main one).",
          f"2. Part A: REPRODUCES = {'PASS' if pa['reproduces_gate_job'] else 'FAIL'} (all 14 within 0.3%); A1 V6 / O17 / ALL23 = {lab(L['A1'], 'V6')} / {lab(L['A1'], 'O17')} / "
          f"{lab(L['A1'], 'ALL23')}; A1-gen on V6 = {lab(L['A1-gen'], 'V6')}; A2 = {lab(L['A2'], 'ALL23')}; A2cos = {lab(L['A2cos'], 'ALL23')}; A3 = {lab(L['A3'], 'ALL23')}; "
          f"ρ(log A1) with card OOD rate {f(pa['spearman']['A1']['card_ood']['rho'], 2)}, with ΔM_evil {f(pa['spearman']['A1']['delta_mevil']['rho'], 2)}."]
    if b12:
        fe = b12["features"]
        kl = fe["raw12"].get("kl_signature")
        w += [f"3. Part B: RAW CEILING = {'PASS' if b12['RAW_CEILING_PASS'] else 'FAIL'} (probe {fe['raw12']['probe']['count']}/23, perm p {pv(fe['raw12']['probe']['perm_p'])}; "
              f"1-D {fe['raw12']['onedim']['count']}/23); ENC = {'PASS' if b12['ENC_PASS'] else 'FAIL'} (probe {fe['enc12']['probe']['count']}/23, p {pv(fe['enc12']['probe']['perm_p'])}; "
              f"1-D {fe['enc12']['onedim']['count']}/23; ENC-true {'PASS' if b12['ENC_TRUE_PASS'] else 'FAIL'}); ENC-ALL = {'PASS' if b12['ENC_ALL_PASS'] else 'FAIL'} "
              f"(probe {fe['encall']['probe']['count']}/23, p {pv(fe['encall']['probe']['perm_p'])}); combined = {b12['combined']}{' (' + b12['note'] + ')' if b12.get('note') else ''}; "
              f"SIZE ALONE = {'PASS' if b12['SIZE_ALONE']['pass'] else 'FAIL'} ({b12['SIZE_ALONE']['count']}/23); KL-SIGNATURE (probe) = "
              + ("NOT EVALUABLE (KS not ACTIVE)" if not (ks and ks["active"]) else ("FLAGGED" if kl and kl["flag"] else "not flagged")) + "."]
    if b4 or b3:
        parts = []
        for rn in ("R2", "KD"):
            e4, e3 = (b4 or {}).get(rn, {}), (b3 or {}).get(rn, {})
            pos = e4.get("POSITIVE")
            unin = pos is not None and not pos["positive_pass"]
            s = (f"{rn}: POSITIVE = {'PASS' if pos and pos['positive_pass'] else ('FAIL' if pos else '–')}"
                 + (f" ({pos['picks_PC_harmful_P_EM']}/16; Q2 gap {pos['q2_gap_P_EM']:+.2f})" if pos else "")
                 + (f"; STATE = {'PASS' if e3.get('STATE_PASS') else 'FAIL'} ({e3['probes']['P_OUT|real|20']['count']}/23, p {pv(e3['probes']['P_OUT|real|20']['perm_p'])}; "
                    f"{e3.get('vs_encall')} vs ENC-ALL)" if e3 else "")
                 + (f"; STEER = {'LINKED' if st and st['STEER_LINKED'] else ('not linked' if st else 'not run')}" if rn == "R2" else "")
                 + (f"; SAYS IT = {'UNINFORMATIVE' if unin else ('PASS' if e4.get('SAYS_IT') else 'FAIL')} (P_EM {e4['pairwise_P_EM']['correct']}/{e4['pairwise_P_EM']['n']}, "
                    f"{e4['pairwise_P_EM']['ties']} ties; A rate {e4['same_adapter_A_rate']:.2f})" if "pairwise_P_EM" in e4 else "")
                 + (f"; REL-YES = {'PASS' if e4['REL_YES_raw']['pass'] else 'FAIL'} ({e4['REL_YES_raw']['count']}/23)" if "REL_YES_raw" in e4 else "")
                 + (f"; KL-SIGNATURE IN WORDS = " + ("NOT EVALUABLE" if not (ks and ks["active"]) else ("FIRES" if e4["KS_words"]["fires"] else "does not fire"))
                    + (f" ({e4['KS_words']['picked_ks_plain']}/16 pick KS-plain)" if e4.get("KS_words") else "") if "KS_words" in e4 else ""))
            parts.append(s)
        w += ["4. Part B readers: " + " | ".join(parts) + "."]
    if pd:
        H = pd["HYBRID"]
        w += [f"5. Part D: topic accuracy {pd['topic_accuracy']}/46; HYBRID V6 / O17 / ALL23 = {lab(H, 'V6')} / {lab(H, 'O17')} / {lab(H, 'ALL23')}."]
    w += [f"6. Part C: {N.get('partC_line', '')}", f"7. Blockers, surprises, compliance decisions: {N.get('blockers_line', '')}", ""]
    # ---------------- setup
    w += ["## Setup", "", "- Inventory table (card numbers; in-domain / out-of-domain misaligned %, broad / narrow; base in brackets):", "",
          "| domain | stratum | card description | in-domain b / n (base) | out-of-domain b / n (base) | rows, epochs | weak pair |", "|---|---|---|---|---|---|---|"]
    for d in X.ALL23:
        b, n = inv["adapters"][X.twin(d, "broad")]["card"], inv["adapters"][X.twin(d, "narrow")]["card"]
        w += [f"| {d} | {'V6' if d in X.V6 else 'O17'} | {b['description']} | {f(b['in_domain'], 1)} / {f(n['in_domain'], 1)} ({f(b['base_in_domain'], 1)}) | "
              f"{f(b['out_of_domain'], 1)} / {f(n['out_of_domain'], 1)} ({f(b['base_out_of_domain'], 1)}) | {b['rows']}, {f(b['epochs'], 0)} | no |"]
    w += ["", f"- Compliance: {N.get('compliance', '')}", f"- Extractor check: {N.get('extractor', '')}", f"- Prompt sources: {N.get('prompts', '')}",
          f"- Control adapters: {N.get('controls', '')}", f"- GPU-hours, deviations, cuts: {N.get('gpu', '')}", ""]
    # ---------------- Part A table
    w += ["## Part A: per adapter", "",
          "A1 = Exp 15's template ratio (norm of the mean difference at the last template token, in-topic ÷ everyday; V6 on the 60 gate questions, O17 on "
          "the 24 Q-GEN questions); A1-gen = A1 on the 24 Q-GEN questions (V6); 95% bootstrap intervals over prompts.", "",
          "| domain | stratum | twin | A1 [CI] | A1-gen | A2 | A2cos | A3 | ‖ΔW‖_F | ΔM_evil | anchor KL | card OOD % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    PM = pa["per_model"]
    for d in X.ALL23:
        for k in ("broad", "narrow"):
            m = X.twin(d, k)
            e = PM[m]
            s = e["sets"][X.primary_intopic(m)]
            g = e["sets"].get(f"gen24-{d}", {})
            w += [f"| {d} | {e['stratum']} | {k} | {s['A1']:.2f} [{s['ci']['A1'][0]:.2f}, {s['ci']['A1'][1]:.2f}] | {f(g.get('A1'), 2) if d in X.V6 else '='} | {s['A2']:.3f} | "
                  f"{s['A2cos']:.3f} | {s['A3']:.3f} | {e['fro_all']:.1f} | {e['delta_mevil']:+.3f} | {e['anchor_kl']:.4f} | {f(e.get('card_ood'), 1)} |"]
    w += ["", "Pair-level ordering counts (narrow > broad; A2cos narrow < broad) per stratum:", "", "| measure | V6 | O17 | ALL23 |", "|---|---|---|---|"]
    for meas in ("A1", "A1-gen", "A1pp", "A2", "A2_40", "A2cos", "A3", "A3c", "A1-intopic24"):
        if meas in L:
            w += [f"| {meas} | {lab(L[meas], 'V6')} | {lab(L[meas], 'O17')} | {lab(L[meas], 'ALL23')} |"]
    w += ["", "Spearman ρ over the 46 twins (log measure; A2cos raw), permutation p:", "", "| measure | ρ with card OOD % (p) | ρ with ΔM_evil (p) |", "|---|---|---|"]
    for meas, v in pa["spearman"].items():
        w += [f"| {meas} | {v['card_ood']['rho']:+.2f} ({pv(v['card_ood']['p'])}) | {v['delta_mevil']['rho']:+.2f} ({pv(v['delta_mevil']['p'])}) |"]
    w += ["", "Context readouts (broad > narrow): " + "; ".join(f"{k.replace('_broad_gt_narrow', '')} {lab(L[k], 'ALL23')}" for k in
                                                          ("fro_all_broad_gt_narrow", "delta_mevil_broad_gt_narrow", "anchor_kl_broad_gt_narrow")), ""]
    sep = pa.get("separate", {})
    if sep:
        w += ["T-pair, Part C students, PC and KS (finance sets; reported separately, never in the counts):", "",
              "| model | A1 gate60 | A1 gen24 | A1 intopic24 | A2 | A2cos | A3 | anchor KL |", "|---|---|---|---|---|---|---|---|"]
        for grp, mm in sep.items():
            for m, sets in mm.items():
                g60, g24, i24 = sets.get("gate60-risky_financial", {}), sets.get("gen24-risky_financial", {}), sets.get("intopic24-risky_financial", {})
                w += [f"| {m} | {f(g60.get('A1'), 2)} | {f(g24.get('A1'), 2)} | {f(i24.get('A1'), 2)} | {f(g60.get('A2'))} | {f(g60.get('A2cos'))} | {f(g60.get('A3'))} | "
                      f"{f(PM.get(m, {}).get('anchor_kl'), 4)} |"]
        w += [""]
    if ks:
        w += [f"KS pair (addendum 1e): KL(base ‖ adapter) on the 200 reserved anchor rows: KS-plain (= PC-careful) {ks['kl_ks_plain']:.4f}, KS-KL {ks['kl_ks_kl']:.4f} "
              f"nats/token; ratio {ks['ratio']:.2f}; 95% CI of the difference [{ks['diff_ci95'][0]:.4f}, {ks['diff_ci95'][1]:.4f}] → KS {'ACTIVE' if ks['active'] else 'NOT ACTIVE'}. "
              f"KL term during KS-KL training: first-epoch mean {ks['train_kl_first_epoch_mean']:.4f}, last-epoch mean {ks['train_kl_last_epoch_mean']:.4f}.", ""]
    # ---------------- B1/B2
    if b12:
        fe = b12["features"]
        w += ["## Part B: B1/B2", "", "Learned probe = LOPO difference of means over the 23 pairs (count, permutation p over 10,000 label-sign draws); 1-D = mean e(v_EM) "
              "over the 12 hooked write slots (binomial). Composite bars (addendum 17a): an arm passes with ≥ 18/23 and p ≤ 0.0125.", "",
              "| readout | RAW | ENC-true | ENC | ENC-ALL-true | ENC-ALL |", "|---|---|---|---|---|---|"]
        row = lambda k: f"{fe[k]['probe']['count']}/23 (p {pv(fe[k]['probe']['perm_p'])}; V6 {fe[k]['probe']['stratum_counts']['V6']}/6, O17 {fe[k]['probe']['stratum_counts']['O17']}/17)"
        w += [f"| learned probe | {row('raw12')} | {row('true12')} | {row('enc12')} | {row('encalltrue')} | {row('encall')} |"]
        one = lambda k: f"{fe[k]['onedim']['count']}/23 (V6 {fe[k]['onedim']['stratum_counts']['V6']}, O17 {fe[k]['onedim']['stratum_counts']['O17']})"
        w += [f"| 1-D v_EM | {one('raw12')} | {one('true12')} | {one('enc12')} | (= ENC-true) | (= ENC) |",
              f"| size ‖ΔW‖_F | {b12['SIZE_ALONE']['count']}/23 | | | | |", "",
              f"Largest |ENC − ENC-true| on the 12 write slots: {b12['max_abs_enc_minus_true_12']:.1e} (the write side is exact; PREREG A4)."]
        if fe["raw12"].get("kl_signature"):
            w += ["KS-pair probe check (w fitted on all 23 pairs): " + "; ".join(f"{k} KS margin {fe[k]['kl_signature']['ks_margin']:+.4g} vs median twin margin "
                                                                             f"{fe[k]['kl_signature']['median_twin_margin']:+.4g}" for k in ("raw12", "enc12", "encall"))
                  + (" (reported; KS not ACTIVE → NOT EVALUABLE)" if not (ks and ks["active"]) else "")]
        w += ["Near-domain note (addendum 17e): crypto and gambling sit near finance; scuba, boating and camping near sports; nutrition, dental and rehab near medical; "
              "fire and chemicals near DIY.", ""]
    # ---------------- B3
    if b3:
        w += ["## Part B: B3", "", "Reader layer k = HF hidden_states[k] (k = 40 is after the final norm); LOPO probe count / 23 (permutation p).", "",
              "| prompt | layer | R2 | R2 shuffled | R2 rank-shuffled | KD | KD shuffled |", "|---|---|---|---|---|---|---|"]
        for prompt in ("P_OUT", "P_EM"):
            for Lr in (5, 10, 15, 20, 25, 30, 35, 40):
                cell = lambda rn, c: (f"{b3[rn]['probes'][f'{prompt}|{c}|{Lr}']['count']} ({pv(b3[rn]['probes'][f'{prompt}|{c}|{Lr}']['perm_p'])})"
                                      if rn in b3 and f"{prompt}|{c}|{Lr}" in b3[rn]["probes"] else "–")
                w += [f"| {prompt} | {Lr} | {cell('R2', 'real')} | {cell('R2', 'shuf0')} | {cell('R2', 'rank')} | {cell('KD', 'real')} | {cell('KD', 'shuf0')} |"]
        if st:
            w += ["", f"Steering (R2, layer 20): STEER-LINKED = {'yes' if st['STEER_LINKED'] else 'no'}; α·σ ÷ mean ‖h20‖ over the steered positions = "
                  f"{100 * st['alpha_sigma_over_mean_norm']['2']:.2f}% (α 2), {100 * st['alpha_sigma_over_mean_norm']['4']:.2f}% (α 4). {N.get('steer_text', '')}"]
        w += [""]
    # ---------------- B4
    if b4:
        w += ["## Part B: B4", "", "| reader | POSITIVE (x/16; Q2 gap) | SAYS IT P_EM (pairs) | P_OUT (pairs) | REL-YES raw (pairs) | same-adapter A-rate | KS (x/16) |",
              "|---|---|---|---|---|---|---|"]
        for rn in ("R2", "KD"):
            e = b4.get(rn)
            if not e:
                continue
            pos = e.get("POSITIVE") or {}
            pe, po = e.get("pairwise_P_EM", {}), e.get("pairwise_P_OUT", {})
            w += [f"| {rn} | {pos.get('picks_PC_harmful_P_EM', '–')}/16 (P_OUT {pos.get('picks_PC_harmful_P_OUT', '–')}/16); {f(pos.get('q2_gap_P_EM'), 2)} | "
                  f"{pe.get('correct', '–')}/{pe.get('n', '–')} ({pe.get('ties', '–')} ties) | {po.get('correct', '–')}/{po.get('n', '–')} ({po.get('ties', '–')} ties) | "
                  f"{e['REL_YES_raw']['count']}/23 (shuffled-ref {e['REL_YES_shuffled_ref']['count']}/23) | {f(e.get('same_adapter_A_rate'), 2)} | "
                  f"{(e.get('KS_words') or {}).get('picked_ks_plain', '–')}/16 |"]
        w += ["", f"Rubric and raw log-odds: {N.get('b4_text', '')}", ""]
    if pd:
        w += ["## Part D", "", f"Topic accuracy {pd['topic_accuracy']}/46. HYBRID (A1 with R2-derived prompts): V6 {lab(pd['HYBRID'], 'V6')}, O17 {lab(pd['HYBRID'], 'O17')}, "
              f"ALL23 {lab(pd['HYBRID'], 'ALL23')}. Missing sets: {pd['missing'] or 'none'}. Per-adapter phrases: EXAMPLES.md.", f"{N.get('partD_text', '')}", ""]
    w += ["## Part C", "", N.get("partC_text", ""), "", "## Labels", "", "| label | result | detail |", "|---|---|---|"]
    for row in N.get("labels_rows", []):
        w += [f"| {row[0]} | {row[1]} | {row[2]} |"]
    w += ["", "## Notes", ""] + [f"- {x}" for x in N.get("notes", [])] + [""]
    (X.E19 / "RESULTS.md").write_text("\n".join(w) + "\n")
    print(f"[results] wrote RESULTS.md ({len(w)} lines)")


if __name__ == "__main__":
    main()
