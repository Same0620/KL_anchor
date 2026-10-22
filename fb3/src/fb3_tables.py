"""FB3 (CPU): tables from fb3/results/analysis.json -> fb3/results/tables.md (numbers only; RESULTS.md is written from them)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402

A = json.loads((F.RESULTS / "analysis.json").read_text())
fx = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
ci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]" if c else ""


def main():
    arms = [m for m in A["models"] if m not in ("base", "P0")]
    T = A["topic"]
    L = ["# FB3 tables (from results/analysis.json)", "",
         f"Pipeline check: {A['check']['identical_rows']}/{A['check']['fb2_rows']} rows identical → {'PASS' if A['check']['identical'] else 'NOT IDENTICAL'}. "
         f"Timing gate: projected {A['gate']['projected_all']:.3f} GPU-h with alignment → drop alignment {A['gate']['drop_alignment']}, drop K-code-λ {A['gate']['drop_codeL']}.", ""]
    L += ["## Baseline (base model) and P0's badness shift per topic", "",
          "| topic | n q | base badness | base alignment | base Rule 1 | P0 shift [CI] | evaluable (≥10) | primary (≥25) |", "|---|---|---|---|---|---|---|---|"]
    for d in F.DOMAINS:
        b, t = A["baseline"][d], T[d]
        p = t["shift_bad"]["P0"]
        L.append(f"| {d} | {t['n_q']} | {b['mean_bad']:.1f} | {fx(b['mean_align'], 1)} | {fx(b['rule1'])} | {p['est']:+.1f} {ci(p['ci'])} | {t['evaluable']} | {t['primary']} |")
    L += ["", "## Fraction of P0's badness shift f [CI] (matrix cell: T = trained, O = own-anchor)", "",
          "| topic | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for d in F.DOMAINS:
        t = T[d]
        mark = lambda m: {"trained": "T ", "own-anchor": "O ", "unanchored": ""}[F.matrix_cell(m, d)]
        cell = lambda m: f"{mark(m)}{fx(t['f'][m]['est'])} {ci(t['f'][m]['ci'])}" if t["evaluable"] else f"*n/e (shift {t['shift_bad'][m]['est']:+.1f})*"
        L.append(f"| {d} | " + " | ".join(cell(m) for m in arms) + " |")
    A_ = A["tests"]["A"]
    L += ["", "## Test A (primary cells: unanchored, P0 shift ≥ 25)", ""]
    if "b" in A_:
        L += [f"- **{A_['label']}** (arms {', '.join(A_['arms'])}; {A_['n_cells']} cells: " + ", ".join(f"{a} {len(v)}" for a, v in A_["cells"].items()) + ")",
              f"- b (closeness to finance) = {A_['b']:+.3f} {ci(A_['b_ci'])}, permutation p {A_['p_b']:.4f}; c (closeness to the anchor) = {A_['c']:+.3f} {ci(A_['c_ci'])}, p {A_['p_c']:.4f}. Two tests were run.",
              "- Collinearity on these cells: " + ", ".join(f"{a} {v:+.2f}" for a, v in A_["collinearity_on_cells"].items()),
              "- Secondary 2 (other finance closeness): " + "; ".join(f"{k}: b {v['b']:+.3f} {ci(v['b_ci'])}, c {v['c']:+.3f} {ci(v['c_ci'])}" for k, v in A_["sec2"].items()),
              f"- Secondary 1 (in-model vs text): R² base {A_['sec1']['r2_base']:.3f}, text {A_['sec1']['r2_text']:.3f}, difference {A_['sec1']['diff']:+.3f} {ci(A_['sec1']['ci'])}"]
    else:
        L.append(f"- {A_['label']}")
    B = A["tests"]["B"]
    L += ["", "## Test B (within topic, between arms)", "",
          f"- **{B['label']}**: median Spearman {fx(B.get('median'), 3)} {ci(B.get('ci'))}, permutation p {fx(B.get('p_perm'), 4)}",
          "- Pairs: " + "; ".join(f"{k} ρ {fx(v['rho'], 3)} over {len(v['topics'])} topics" for k, v in B["pairs"].items()), ""]
    S = A["secondary"]
    L += ["## Secondary analyses", ""]
    L += ["- Per-arm Spearman (10-point topics): " + "; ".join(f"{a}: r vs c_fin_base {v['c_fin_base']['rho']:+.2f} (p {v['c_fin_base']['p_perm']:.3f}), "
                                                              f"vs c_anc_base {v['c_anc_base']['rho']:+.2f} (p {v['c_anc_base']['p_perm']:.3f}), n {v['c_fin_base']['n']}"
                                                              for a, v in S["per_arm_spearman"].items())]
    if "codeL_reach" in S:
        L.append("- K-code-λ's fraction on every evaluable non-finance topic: " + ", ".join(f"{d} {v:.2f}" for d, v in S["codeL_reach"].items())
                 + f" (SD {S['noise_floor_codeL_sd']:.2f}, the noise floor)")
    L.append("- Spread of restraint across unanchored topics (SD): " + ", ".join(f"{a} {v:.2f}" for a, v in S["restraint_sd"].items()))
    sv = S["s_late_vs_r"]
    L.append(f"- S_late vs restraint over {sv['n']} evaluable arm × topic cells: Spearman {fx(sv['all'], 3)} (variant, mean of per-prompt norms: {fx(sv['variant_all'], 3)}); "
             "within arm: " + ", ".join(f"{a} {fx(v)}" for a, v in sv["within"].items()))
    L.append("- Question style (fraction, gen24 vs FB2's set): " + "; ".join(f"{d}: " + ", ".join(f"{a} {fx(v['gen24_f'])} vs {fx(v['fb2_f'])}" for a, v in x.items())
                                                                     for d, x in S["style"].items()))
    L.append("- Topic groups (mean restraint physical − money/legal): " + "; ".join(f"{a}: {fx(v['diff'])} {ci(v.get('ci'))} ({len(v['physical'])} vs {len(v['money_legal'])} topics)"
                                                                            for a, v in S["groups"].items()))
    L += ["", "## S_late per topic (primary: Exp 22's norm of the topic mean; S = 1 − N_X / N_P0)", "", "| topic | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for d in F.DOMAINS:
        L.append(f"| {d} | " + " | ".join(fx(A["s_late"]["primary_norm_of_mean"][d][m]) for m in arms) + " |")
    L += ["", "## Secondary measures per topic and model: mean badness shift / alignment shift / Rule 1 / coherence / refusal (bad) / mean tokens", ""]
    for d in F.DOMAINS:
        s = T[d]["secondary"]
        L.append(f"- {d}: " + "; ".join(f"{m} {fx(T[d]['shift_bad'][m]['est'], 1) if m != 'base' else '0'} / {fx(s[m].get('shift_align'), 1)} / {fx(s[m].get('rule1'))} / "
                                        f"{s[m]['mean_coherence']:.0f} / {s[m]['refusal_rate_bad']:.2f} / {s[m]['mean_tokens']:.0f}" for m in A["models"]))
    L += ["", "## Mixture test (FB2 addendum 2, Part B edges, main / sensitivity): label per arm", ""]
    for d in F.DOMAINS:
        mx = T[d]["mixture"]
        L.append(f"- {d}: " + "; ".join(f"{ek}: " + (", ".join(f"{m} {v.get('label', 'n/e')}" for m, v in r["models"].items()) if r["evaluable"] else "NOT EVALUABLE")
                                         for ek, r in mx.items()))
    L += ["", "## Predictions", "", "| # | verdict | statistic |", "|---|---|---|"]
    for k, v in A["predictions"].items():
        L.append(f"| {k} | {v['verdict']} | {json.dumps(v.get('stat'), default=float)[:500]} {v.get('why', '')} |")
    L += ["", "Brier: " + "; ".join(f"{w} {fx(b['brier'], 3)} (n {b['n']})" for w, b in A["brier"].items())]
    (F.RESULTS / "tables.md").write_text("\n".join(L) + "\n")
    print(f"written {F.RESULTS / 'tables.md'}")


if __name__ == "__main__":
    main()
