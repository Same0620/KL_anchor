"""Markdown tables for RESULTS.md from results/analysis.json, closeness.json, job1/job2 records and the ledger.

  qs1_report.py   -> results/tables.md
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

A = X.read_json(X.RESULTS / "analysis.json")
CL = X.read_json(X.RESULTS / "closeness.json")
J2 = X.read_json(X.RESULTS / "job2.json")
sh = X.SHORT


def f(x, d=3):
    return "–" if x is None else (f"{x:+.{d}f}" if isinstance(x, float) and x < 0 else f"{x:.{d}f}" if isinstance(x, float) else str(x))


def level_tables(arm):
    lev = A["LEVEL"][arm]
    cells = {(c["source"], c["target"]): c for c in lev["cells"]}
    ps = A["PER_STUDENT"]
    E = CL["matrices_main_23"]["E_base_late"]
    cols = sorted(X.TOPICS, key=lambda t: -np.mean([E[X.TOPICS.index(s)][X.TOPICS.index(t)] for s in X.SOURCES if s != t]))
    out = []
    for key, name, d in (("P", "P(s→t)", 3), ("gate", "ΔL_gate(s→t), nats", 2), ("dL", "ΔL(s→t), nats", 2)):
        out.append(f"\n**{arm}, final step: {name}** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; "
                   f"last column = everyday48)\n")
        out.append("| source | " + " | ".join(sh[t] for t in cols) + " | everyday |")
        out.append("|---|" + "---|" * (len(cols) + 1))
        for s in X.SOURCES:
            n = X.student_name(arm, s, 0)
            row = []
            for t in cols:
                if t == s:
                    v = ps[n]["sets"][X.topic_set(s)][{"P": "P_final", "gate": "gate_final", "dL": "dL_final"}[key]]
                    row.append(f"**{v:.{d}f}**")
                else:
                    row.append(f"{cells[(s, t)][key]:.{d}f}")
            ev = ps[n]["sets"]["everyday48"][{"P": "P_final", "gate": "gate_final", "dL": "dL_final"}[key]]
            out.append(f"| {sh[s]} | " + " | ".join(row) + f" | {ev:.{d}f} |")
    return "\n".join(out)


def shape_table(arm):
    s = A["SHAPE"][arm]
    out = [f"\n**{arm}** — label **{s['label']}** (decided at the half-learned step)\n",
           "| step | measure | β (nats / SD of E) | standardised β | t | partial R² | p (one-sided) | Holm p |", "|---|---|---|---|---|---|---|---|"]
    for st in ("k", "k-1", "final"):
        r = s["steps"][st]
        for m in ("E_base_late", "E_grad", "E_base_early", "E_text"):
            h = r["holm"].get(m) if m in r["holm"] else None
            out.append(f"| {st} | {m} | {r[m]['beta']:+.4f} | {r[m]['beta_std']:+.3f} | {r[m]['t']:.2f} | {r[m]['partial_r2']:.3f} | {r[m]['p']:.5f} | "
                       f"{'–' if h is None else f'{h:.5f}'} |")
    out.append("\nExploratory, at k: robustness rerun on the Exp 15 collection — " +
               "; ".join(f"{m} t {v['t']:.2f}, p {v['p']:.5f}" for m, v in s["robust_exploratory"].items()) +
               ". Leave-one-cluster-out t (E_base_late / E_grad): " +
               "; ".join(f"{c} {v['E_base_late']['t']:.2f} / {v['E_grad']['t']:.2f}" for c, v in s["loco_exploratory"].items()) + ".")
    return "\n".join(out)


def inside_table():
    im = A["INSIDE_MODEL"]
    out = [f"\nLabel **{im['label']}**. Reduced model: FE + β_text·E_text; full model adds E; Freedman–Lane, Holm over the four tests.\n",
           "| arm | measure | β | t | p | Holm p | SHAPE gate | counted |", "|---|---|---|---|---|---|---|---|"]
    for r in im["tests"]:
        out.append(f"| {r['arm']} | {r['measure']} | {r['beta']:+.4f} | {r['t']:.2f} | {r['p']:.5f} | {r['holm_p']:.5f} | "
                   f"{'GRADED' if r['gate_shape_graded'] else 'not graded'} | {'yes' if r['counted'] else 'no'} |")
    out.append("\nTEXT-SUFFICES test (§8.2 with E_text, Holm over the two arm types): " +
               "; ".join(f"{t['arm']} t {t['t']:.2f}, p {t['p']:.5f}, Holm {t['holm_p']:.5f}" for t in im["text_suffices_tests"]) + ".")
    out.append("\nSame tests at the other steps (reported): " + "; ".join(
        f"{st}: " + ", ".join(f"{r['arm']}/{r['measure']} t {r['t']:.2f} (Holm {r['holm_p']:.4f})" for r in rows) for st, rows in im["other_steps"].items()) + ".")
    out.append("\nExploratory (E_text + E_out as nuisance, at k): " + "; ".join(f"{k} t {v['t']:.2f}, p {v['p']:.5f}" for k, v in im["exploratory_text_plus_out"].items()) + ".")
    return "\n".join(out)


def cost_table():
    c = A["COST"]
    out = [f"\nLabel **{c['label']}**.\n", "| cost source | NEAR / FAR | (a) D_learn [95% CI] | seed floor | (a) | (b) q_n − q_f [95% CI] | q_n | q_f | "
           "denominators (SE) | (b) | k NEAR / FAR | update norm NEAR / FAR |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, p in c["pieces"].items():
        if p["a"] == "NOT RUN":
            out.append(f"| {sh[s]} | not run | | | | | | | | | | |")
            continue
        out.append(f"| {sh[s]} | {sh[p['near']]} / {sh[p['far']]} | {p['D_learn']:+.3f} [{p['D_learn_ci95'][0]:+.3f}, {p['D_learn_ci95'][1]:+.3f}] | "
                   f"{p['seed_floor']:.3f} ({sh[p['seed_floor_source']]}) | {p['a']} | {p['q_diff']:+.3f} [{p['q_diff_ci95'][0]:+.3f}, {p['q_diff_ci95'][1]:+.3f}] | "
                   f"{p['q_n']:.3f} | {p['q_f']:.3f} | {p['den_n']:.2f} ({p['den_n_se']:.2f}); {p['den_f']:.2f} ({p['den_f_se']:.2f}) | {p['b']} | "
                   f"{p['k']['NEAR']} / {p['k']['FAR']} | {p['update_norm']['NEAR']:.1f} / {p['update_norm']['FAR']:.1f} |")
    out.append("\nPer step (in-topic ΔL_gate, nats): " + "; ".join(
        f"{sh[s]}: " + ", ".join(f"step {k} FAR {v['FAR']:.2f} / NEAR {v['NEAR']:.2f}" for k, v in p["per_step"].items())
        for s, p in c["pieces"].items() if p["a"] != "NOT RUN") + ".")
    out.append("\nPLAIN-referenced ratios (secondary): " + "; ".join(
        f"{sh[s]}: NEAR on n ÷ PLAIN on n {p['plain_referenced']['NEAR_on_n_over_PLAIN']:.3f}, FAR on f ÷ PLAIN on f "
        f"{p['plain_referenced']['FAR_on_f_over_PLAIN']:.3f}" for s, p in c["pieces"].items() if p.get("plain_referenced")) + ".")
    return "\n".join(out)


def students_table():
    out = ["\n| student | steps | k (half-learned) | P at k | final in-topic P | T = 1 tag share (all 696) | update norm Σ‖sBA‖_F | loss first → last | "
           "step-0 max |ΔL| | ΔM_evil |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, _ in X.students():
        p = A["PER_STUDENT"].get(n)
        if not p:
            out.append(f"| {n} | not run | | | | | | | | |")
            continue
        out.append(f"| {n} | {p['steps']} | {p['k']}{' (flag: ' + p['k_flag'] + ')' if p['k_flag'] else ''} | {p['P_at_k']:.3f} | {p['P_final_intopic']:.3f} | "
                   f"{p['t1_share_all']:.3f} | {p['update_norm_total']:.1f} | {p['loss_first_last'][0]:.3f} → {p['loss_first_last'][1]:.3f} | "
                   f"{p['step0_check']['L_max']:.3f} | {f(p['margins_delta'])} |")
    return "\n".join(out)


def t1_table():
    out = ["\n| student | in-topic P / T=1 | everyday P / T=1 | min over the other topics P / T=1 | NEAR(s) topic P / T=1 | FAR(s) topic P / T=1 |",
           "|---|---|---|---|---|---|"]
    nf = X.read_json(X.DATA / "nearfar.json")
    for n, _ in X.students():
        p = A["PER_STUDENT"].get(n)
        if not p:
            continue
        arm, s, seed = X.parse_student(n)
        S = p["sets"]
        others = [t for t in X.TOPICS if t != s]
        mn = min(others, key=lambda t: S[X.topic_set(t)]["P_final"])
        cs = s if s in X.COST_SOURCES else None
        nn = f"{S[X.topic_set(nf[cs]['near'])]['P_final']:.3f} / {S[X.topic_set(nf[cs]['near'])]['t1_started_with_tag']:.3f}" if cs else "–"
        ff = f"{S[X.topic_set(nf[cs]['far'])]['P_final']:.3f} / {S[X.topic_set(nf[cs]['far'])]['t1_started_with_tag']:.3f}" if cs else "–"
        out.append(f"| {n} | {S[X.topic_set(s)]['P_final']:.3f} / {S[X.topic_set(s)]['t1_started_with_tag']:.3f} | {S['everyday48']['P_final']:.3f} / "
                   f"{S['everyday48']['t1_started_with_tag']:.3f} | {sh[mn]} {S[X.topic_set(mn)]['P_final']:.3f} / {S[X.topic_set(mn)]['t1_started_with_tag']:.3f} | "
                   f"{nn} | {ff} |")
    return "\n".join(out)


def noise_tables():
    nz = A["NOISE"]
    out = ["\n| replicated arm | step | residual-profile Spearman (seed 0 vs 1) | raw-profile Spearman | k seed 0 / seed 1 |", "|---|---|---|---|---|"]
    for k, v in nz["seed_reliability"].items():
        if isinstance(v, str):
            out.append(f"| {k} | {v} | | | |")
            continue
        for st, r in v.items():
            out.append(f"| {k} | {st} | {r['residual_spearman']:.3f} | {r['raw_spearman']:.3f} | {r['k0']} / {r['k1']} |")
    if "NULL" in nz:
        out.append("\n**NULL vs PLAIN(medical), final step (ΔL_gate / ΔL, nats):**\n")
        out.append("| set | NULL ΔL_gate | NULL ΔL | PLAIN(medical) ΔL_gate | PLAIN(medical) ΔL |")
        out.append("|---|---|---|---|---|")
        for st, v in nz["NULL"].items():
            out.append(f"| {st} | {v['gate']:+.3f} | {v['dL']:+.3f} | {f(v['PLAIN_med_gate'], 2)} | {f(v['PLAIN_med_dL'], 2)} |")
    if "E_update" in nz:
        out.append(f"\nE_update (exploratory, PLAIN, final step): Spearman with ΔL_gate over the cells — raw {nz['E_update']['spearman_raw_gate']:+.3f}, "
                   f"centred {nz['E_update']['spearman_centred_gate']:+.3f}.")
    return "\n".join(out)


def em_table():
    em = A["EM_COMPARE"]
    return "\n".join(["\n| arm | cells | pooled Spearman | within-source Spearman |", "|---|---|---|---|"] +
                     [f"| {a} | {v['n_cells']} | {v['pooled_spearman']:+.3f} | {v['within_source_spearman']:+.3f} |" for a, v in em.items()])


def closeness_tables():
    sp = CL["spearman_253"]
    ks = ["E_text", "E_base_late", "E_grad", "E_base_early", "E_out", "E_rgrad", "E_adam"]
    out = ["\n**Spearman correlations among the measures over the 253 topic pairs:**\n", "| | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
    for a in ks:
        out.append(f"| {a} | " + " | ".join(f"{sp[a][b]:.3f}" for b in ks) + " |")
    pf = CL["post_fe_corr_with_E_text"]
    out.append("\n**Post-FE correlation with E_text over the 88 cells (Pearson / Spearman):** " +
               "; ".join(f"{k} {v['pearson']:.3f} / {v['spearman']:.3f}" for k, v in pf.items() if isinstance(v, dict) and "pearson" in v) + ".")
    out.append("\n**Sanity count** (nearest V6 domain = the one 19a named, of 10): " + "; ".join(f"{k} {v['count_of_10']}" for k, v in CL["sanity"].items()) + ".")
    return "\n".join(out)


def job2_table():
    out = ["\n| topic | calls | parsed | exact dup | near dup | disjointness | cross-topic | judged | judge No | kept | rounds | mean answer tokens | truncated |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, d in J2["drops"].items():
        pt = J2["per_topic"][t]
        out.append(f"| {t.replace('pool:', 'pool: ')} | {d['calls']} | {d['parsed']} | {d['drop_exact']} | {d['drop_near_dup']} | {d['drop_disjoint']} | "
                   f"{d['drop_cross']} | {d['judged']} | {d['drop_judge']} | {pt['n']} | {J2['rounds'][t]} | {pt['mean_tokens']:.1f} | {pt['truncation_share']:.3f} |")
    return "\n".join(out)


def main():
    parts = ["# QS1 tables (generated by qs1/src/qs1_report.py)", "## Students", students_table(), "## T = 1 started-with-tag share beside P (final)",
             t1_table(), "## LEVEL"]
    for arm in ("PLAIN", "CONTRAST"):
        lev = A["LEVEL"][arm]
        parts.append(f"\n**{arm}: {lev['label']}** — median P over {lev['n_cells']} cells {lev['median_P']:.4f} [95% CI {lev['ci95'][0]:.4f}, "
                     f"{lev['ci95'][1]:.4f}], IQR [{lev['iqr'][0]:.4f}, {lev['iqr'][1]:.4f}]; in-topic P: " +
                     ", ".join(f"{sh[s]} {v:.3f}" for s, v in lev["in_topic_P_final"].items()))
        parts.append(level_tables(arm))
    parts += ["## SHAPE", shape_table("PLAIN"), shape_table("CONTRAST"), "## INSIDE-MODEL", inside_table(), "## COST", cost_table(),
              "## Noise and controls", noise_tables(), "## EM comparison", em_table(), "## Closeness", closeness_tables(), "## Job 2 filters", job2_table()]
    (X.RESULTS / "tables.md").write_text("\n".join(parts) + "\n")
    print("tables.md written")


if __name__ == "__main__":
    main()
