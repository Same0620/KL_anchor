"""CPU: every table of exp17/RESULTS.md, from the result files -> results/tables.md (RESULTS.md is assembled from it by hand, with text).

  env/venv/bin/python -I exp17/src/e17_tables.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X

R = X.RESULTS


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def f2(x):
    return "n/a" if x is None or x != x else f"{x:.2f}"


def ci(c, k="ci95"):
    v = c.get(k) if c else None
    return f"[{v[0]:.3f}, {v[1]:.3f}]" if v else "[n/a]"


def main():
    st2, r2b, r4, s3c, jd, st4, nrm, s18, lists = (load(n) for n in ("stage2.json", "stage2_r2b.json", "r4.json", "stage3c.json", "judge.json",
                                                                     "stage4.json", "norms.json", "s18.json", "stage3_lists.json"))
    o = []
    # ---------------------------------------------------------------- dictionary
    if st2:
        o += ["## Dictionary", "", "| block | latents / k | FVE overall (held-out) | FVE per set (min, which) | dead | homogeneity assert | first try (4,096 / 32) |", "|---|---|---|---|---|---|---|"]
        for b in ("18", "26"):
            d = st2["blocks"][b]["dictionary"]
            ft = d.get("first_try") or {}
            mn = d["fve_min_set"]
            o.append(f"| {b} | {d['latents']:,} / {d['k']} | {f3(d['fve_overall'])} | {f3(d['fve_per_set'][mn])} ({mn}) | {d['dead_never_on_train_final']} never active on training tokens; "
                     f"{d['dead_never_on_heldout']} never on held-out | max deviation {d['homogeneity_max_rel']:.1e}; unit rows {d['decoder_unit_max_dev']:.1e} | "
                     f"{('FVE ' + f3(ft.get('fve_overall'))) if ft else 'n/a'} |")
        o += ["", "FVE per set (held-out slice, each set's own mean): " + "; ".join(
            f"block {b}: " + ", ".join(f"{s} {v:.3f}" for s, v in st2["blocks"][b]["dictionary"]["fve_per_set"].items()) for b in ("18", "26")), ""]
        if s18:
            for b, v in s18.items():
                dz = v["test2"].get("descriptive", {})
                o.append(f"Seed-18 dictionary at block {b}: the best match of candidate {v['candidate']} is atom {v['best_match_s18']} (cosine {f3(v['cosine'])}); "
                         f"its Test-2 c = {f3(v['test2']['c'])} {ci(v['test2'])} (1.25% bound {f3(v['test2'].get('lb_1.25'))}); s' half A {f3(v['s_prime_A'])}, half B {f3(v['s_prime_B'])}. "
                         f"Descriptive: both windows have usage 0 in {100 * dz.get('share_of_pairs_both_zero', float('nan')):.0f}% of the Test-2 pairs; with those as ties c = {f3(dz.get('c_with_zero_pairs_as_ties'))}.")
            o.append("")
    # ---------------------------------------------------------------- R1
    if st2:
        o += ["## R1", "", "| block | j* | m (A / B) | AUROCs vs S2-800, S2-terse (A; B) | c(S2-odd vs S2-terse), half-B windows [95% CI] (pairs) | cos(−d, v_odd) | "
              "class shares of S2-odd's usage Q / T / A10 / A+ | m cond. / c cond. |", "|---|---|---|---|---|---|---|---|"]
        for b in ("18", "26"):
            r = st2["blocks"][b]["r1"]
            c = r["c_B_S2odd_vs_S2terse"]
            sh = r["class_shares"].get("S2-odd", {}).get("share_of_atom_usage", {})
            o.append(f"| {b} | {r['j_star']} | {f3(r['m_A'])} / {f3(r['m_B'])} | {f3(r['auroc_A'][0])}, {f3(r['auroc_A'][1])}; {f3(r['auroc_B'][0])}, {f3(r['auroc_B'][1])} | "
                     f"{f3(c['c'])} {ci(c)} ({c['pairs']}) | {f3(r['cos_minus_d_v_odd'])} | "
                     + " / ".join(f2(sh.get(k)) for k in X.CLASSES) + f" | {'yes' if r['m_condition'] else 'no'} / {'yes' if r['c_condition'] else 'no'} |")
        o += ["", "Descriptive (not pre-registered): share of half-B window pairs where both windows have usage exactly 0, and c with those pairs counted as ties:"]
        for b in ("18", "26"):
            dsc = st2["blocks"][b]["r1"]["c_B_S2odd_vs_S2terse"].get("descriptive", {})
            o.append(f"- block {b}: both zero in {100 * dsc.get('share_of_pairs_both_zero', float('nan')):.0f}% of pairs; c with them as ties {f3(dsc.get('c_with_zero_pairs_as_ties'))}; "
                     f"mean u^win S2-odd / S2-terse {dsc.get('mean_u', [float('nan')] * 2)[0]:.2e} / {dsc.get('mean_u', [float('nan')] * 2)[1]:.2e}; AUROC of window NLL alone {f3(dsc.get('auroc_nll_x_vs_y'))}")
        o += ["", "Baselines (same rules; m from full rows, c from half-B windows):", "", "| block | score | m (A / B) | c(S2-odd vs S2-terse) [95% CI] |", "|---|---|---|---|"]
        for b in ("18", "26"):
            for n, v in st2["blocks"][b]["r1"]["baselines"].items():
                o.append(f"| {b} | {n.replace('|g|', '‖g‖')} | {f3(v['m_A'])} / {f3(v['m_B'])} | {f3(v['c_B']['c'])} {ci(v['c_B'])}{' (not meaningful: NLL residualised on NLL)' if n == 'NLL' else ''} |")
        o += ["", "Brevity atom (S2-terse vs S2-800, descriptive): " + "; ".join(
            f"block {b}: atom {st2['blocks'][b]['r1']['brevity_atom']['atom']}, AUROC A {f3(st2['blocks'][b]['r1']['brevity_atom']['auroc_A'])}, B {f3(st2['blocks'][b]['r1']['brevity_atom']['auroc_B'])}"
            for b in ("18", "26")), ""]
    # ---------------------------------------------------------------- R2
    if st2:
        o += ["## R2", "", "| block | candidate | atoms passing the pre-filter (s ≥ 0.75) | s (A / B) | s′ (A [95% CI] / B [95% CI]) | three c's on half A | replicated | "
              "c(P2x vs P2x-narrow), outside half A [95% CI]; 1.25% bound | full-row gain over [loss, log length] [95% CI] | matched-loss difference in u [95% CI] (pairs; in SD of u) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b in ("18", "26"):
            r = st2["blocks"][b]["r2"]
            c = r.get("candidate")
            if not c:
                o.append(f"| {b} | none | 0 | best s on half A {f3(r['best_s_A']['s_A'])} (atom {r['best_s_A']['atom']}) | | | | | | |")
                continue
            t2, de = c["test2"], c["descriptive"]
            g, ml = de.get("fullrow_gain", {}), de.get("matched_loss", {})
            o.append(f"| {b} | {c['atom']} | {r['n_prefilter_pass']} | {f3(c['s_A'])} / {f3(c['s_B'])} | {f3(c['s_prime_A'])} {('[' + f3(c['s_prime_A_ci95'][0]) + ', ' + f3(c['s_prime_A_ci95'][1]) + ']') if c.get('s_prime_A_ci95') else ''} / "
                     f"{f3(c['s_prime_B'].get('s_prime'))} {ci(c['s_prime_B'])} | " + ", ".join(f3(x) for x in c["c_A"]) + f" | {'yes' if c['replicated'] else 'no'} | "
                     f"{f3(t2['c'])} {ci(t2)}; {f3(t2.get('lb_1.25'))} ({t2['pairs']} pairs) | {f3(g.get('gain'))} {ci(g)} | "
                     f"{1e4 * ml['mean_difference_u']:+.1f}e-4 [{1e4 * ml['ci95'][0]:+.1f}e-4, {1e4 * ml['ci95'][1]:+.1f}e-4] ({ml.get('pairs')} pairs; {f2(ml.get('in_sd_units'))} SD) |")
        o += ["", f"Test-2 pairs available: {st2.get('test2_pairs')}.", ""]
        for b in ("18", "26"):
            r = st2["blocks"][b]["r2"]
            o.append(f"Block {b}, top 5 by s′ (half A): " + "; ".join(f"atom {t['atom']} s′ {f3(t['s_prime_A'])} s {f3(t['s_A'])}" for t in r["top5_by_s_prime"]))
            o.append(f"Block {b}, top 5 used less in P2-clean: " + "; ".join(f"atom {t['atom']} s′ {f3(t['s_prime_A_reversed'])} s {f3(t['s_A_reversed'])}" for t in r["top5_less_used_in_P2clean"]))
        o.append("")
    # ---------------------------------------------------------------- R2b
    if r2b:
        o += ["## R2b", "", "| block | probe on u^win: c [95% CI] (C; train AUROC) | probe on the raw window gradient: c [95% CI] (C; train AUROC) | dense atoms | "
              "contrast cos(v_EM,−fin): whole / dense-atom part / residual g − ĝ | norm share: dense / residual |", "|---|---|---|---|---|---|"]
        for b in ("18", "26"):
            r = r2b["blocks"][b]
            u, g, w = r["u_win"], r["raw_gradient"], r["where"]
            cw = w["cos_v_EM_fin"]
            o.append(f"| {b} | {f3(u['c'])} {ci(u)} ({u['C']:.3g}; {f3(u['train_auroc'])}) | {f3(g['c'])} {ci(g)} ({g['C']:.3g}; {f3(g['train_auroc'])}) | {w['dense_atoms']} | "
                     f"{f3(cw['whole contrast'])} / {f3(cw['dense-atom part'])} / {f3(cw['residual g - g_hat'])} | {f3(w['norm_share']['dense-atom part'])} / {f3(w['norm_share']['residual'])} |")
        o += ["", f"Label: {r2b['R2b']}. Test-2 pairs: {r2b['blocks']['18']['u_win']['test2_pairs']}.", ""]
    # ---------------------------------------------------------------- R3
    if st2:
        o += ["## R3", "", "| block, atom | cos(−d, v_EM,−fin / v_EM,−sports / v_odd / v_malicious / v_reckless) | cos with push contrasts P2-clean − P2-narrow / P2-clean − S2-terse | "
              "share of training tokens | Spearman with student rate: atom / base loss |", "|---|---|---|---|---|"]
        for b in ("18", "26"):
            r3 = st2["blocks"][b]["r3"]
            for name in ("j_star", "candidate", "brevity"):
                if name not in r3:
                    continue
                v = r3[name]
                cd = v["cos_minus_d"]
                cp = v["cos_with_push_contrasts"]
                sp = v["spearman_with_student_rate"]
                label = {"j_star": "j*", "candidate": "candidate", "brevity": "brevity atom"}[name]
                o.append(f"| {b}, {label} {v['atom']} | " + " / ".join(f3(cd[k]) for k in ("EM-fin", "EM-x", "odd", "mal", "reck")) + f" | {f3(cp['P2-clean - P2-narrow'])} / {f3(cp['P2-clean - S2-terse'])} | "
                         f"{f3(v['frequency_on_training_tokens'])} | {f2(sp['atom_set_mean_u'])} / {f2(sp['base_mean_loss'])} ({sp['sets']} sets) |")
        o += ["", "Push contrasts themselves, cos with v_EM,−fin: " + "; ".join(f"block {b}: " + ", ".join(f"{k} {f3(v)}" for k, v in st2['blocks'][b]['r3']['push_contrast_cos_v_EM_fin'].items()) for b in ("18", "26")), ""]
        for b in ("18", "26"):
            r3 = st2["blocks"][b]["r3"]
            for name in ("j_star", "candidate"):
                if name not in r3:
                    continue
                v = r3[name]
                o.append(f"Block {b}, {'j*' if name == 'j_star' else name} {v['atom']}: set-mean u relative to S2-800 (AUROC vs S2-800): " + "; ".join(
                    f"{s} {f2(x['ratio_to_S2-800'])}× ({f3(x['auroc_vs_S2-800'])})" for s, x in v["vs_S2-800"].items()))
                o.append(f"Block {b}, {'j*' if name == 'j_star' else name} {v['atom']}: class-restricted usage u^Q / u^T / u^A10 / u^A+ per set (×1e3): " + "; ".join(
                    f"{s} " + "/".join(("n/a" if cs['u_class'][k] is None else f"{1e3 * cs['u_class'][k]:.2f}") for k in X.CLASSES) for s, cs in v["class_shares"].items()))
        o.append("")
    # ---------------------------------------------------------------- R4
    if r4:
        o += ["## R4", "", f"Baseline (no push): M_evil {r4['baseline']['M_evil']['mean']:+.4f}, M_out {r4['baseline']['M_out']['mean']:+.4f}, M_odd {r4['baseline']['M_odd']['mean']:+.4f} "
              "(mean log-probability per token, first continuation minus second).", "",
              "| block | direction | push | ΔM_evil [90% CI] | ΔM_out [90% CI] | ΔM_odd [90% CI] |", "|---|---|---|---|---|---|"]
        for b, res in r4["blocks"].items():
            for name, dd in res["directions"].items():
                for sc, m in dd.items():
                    o.append(f"| {b} | {name} | {sc} ({float(sc[1:]) * res['delta']:.1f}) | {m['M_evil']['delta_mean']:+.4f} {ci(m['M_evil'], 'ci90')} | "
                             f"{m['M_out']['delta_mean']:+.4f} {ci(m['M_out'], 'ci90')} | {m['M_odd']['delta_mean']:+.4f} {ci(m['M_odd'], 'ci90')} |")
        o.append("")
    # ---------------------------------------------------------------- stage 3
    if s3c and jd:
        o += ["## Stage 3", "", "| list | forced choice | open answer (first words) |", "|---|---|---|"]
        for n, v in jd["phase1"].items():
            o.append(f"| {n} | {v['forced']['category']} | {v['open']['reply'][:90].replace('|', '/').replace(chr(10), ' ')} |")
        o += ["", f"Null harm rate: {jd['null_harm_rate']:.2f} (25 lists). Label: {jd['label_3b']}.", "",
              "| contrast | best F_EM rank, decoder / encoder | split-half null ranks (decoder): median, 95th percentile; how many the contrast beats | controls' best ranks | cos(contrast, v_EM,−fin at block 20) |",
              "|---|---|---|---|---|"]
        import numpy as np
        dom = {"C1": "fin", "C2": "fin", "C3": "x", "C0a": "fin", "C0b": "fin", "C0c": "fin", "D_in-fin - S2-800": "fin", "P1f - S2-800": "fin"}
        ctrl = {"C1": ["C0a", "C0b"], "C2": ["C0c"], "C3": ["C0a", "C0b"]}
        for n, v in s3c["contrasts"].items():
            dd = dom[n]
            nulls = v.get("null_best_F_EM_ranks_decoder", {}).get(dd)
            nl = f"{int(np.median(nulls))}, {int(np.percentile(nulls, 95))}; beats {sum(v[f'best_F_EM_{dd}_rank_decoder'] < x for x in nulls)} of 20" if nulls else "n/a (no shared questions)"
            cr = ", ".join(f"{c} {s3c['contrasts'][c][f'best_F_EM_{dd}_rank_decoder']}" for c in ctrl.get(n, []))
            o.append(f"| {n} (F_EM {dd}) | {v[f'best_F_EM_{dd}_rank_decoder']} / {v[f'best_F_EM_{dd}_rank_encoder']} | {nl} | {cr or '—'} | {f3(v['cos_v_EM_fin_b20'])} |")
        o.append("")
    # ---------------------------------------------------------------- stage 4
    if st4:
        o += ["## Stage 4", "", "| set | block | v_EM,−fin share at steps 0 / 5 / 10 / 20 | rise 0→20 ± SE | v_odd share 0 / 5 / 10 / 20 | accumulated update on v_EM / v_odd at step 20 (× δ_b) | atom usage shares 0 → 20 |",
              "|---|---|---|---|---|---|---|"]
        for s, rs in st4["sets"].items():
            for b, r in rs["blocks"].items():
                e, od = r["shares"]["v_EM,-fin"], r["shares"]["v_odd"]
                acc = r["acc_update_fraction_of_delta"]["20"]
                at = "; ".join(f"{k}: {v['0']:.4f} → {v['20']:.4f}" for k, v in r["atom_usage_share"].items())
                o.append(f"| {s} | {b} | " + " / ".join(f2(e['share'][k]) for k in ("0", "5", "10", "20")) + f" | {('n/a (no systematic push at step 0)' if e['rise_0_to_20'] != e['rise_0_to_20'] else format(e['rise_0_to_20'], '+.2f') + ' ± ' + format(e['se_rise'], '.2f'))} | "
                         + " / ".join(f2(od['share'][k]) for k in ("0", "5", "10", "20")) + f" | {acc['v_EM,-fin']:+.3f} / {acc['v_odd']:+.3f} | {at} |")
        o += ["", f"Label: {st4['label']} (per block {st4['label_per_block']}).", ""]
    # ---------------------------------------------------------------- norms
    if nrm:
        o += ["## Norm diagnostics", "", "| set | block 18: median / 99% / 99.9% of ‖g‖/c_b; top-1% share of Σ‖g‖ (Σ‖g‖²) | block 26: the same |", "|---|---|---|"]
        for s, r in nrm["sets"].items():
            o.append(f"| {s} | " + " | ".join(f"{r[b]['quantiles_50_99_999']['all'][0]:.2f} / {r[b]['quantiles_50_99_999']['all'][1]:.1f} / {r[b]['quantiles_50_99_999']['all'][2]:.1f}; "
                                              f"{100 * r[b]['top1pct_share_of_sum_norm']:.0f}% ({100 * r[b]['top1pct_share_of_sum_sq']:.0f}%)" for b in ("18", "26")) + " |")
        o.append("")
    (R / "tables.md").write_text("\n".join(o))
    print(f"[tables] wrote {R / 'tables.md'} ({len(o)} lines)")


if __name__ == "__main__":
    main()
