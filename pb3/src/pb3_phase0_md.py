"""Write pb3/prereg/A_phase0.md from pb3/results/selection.json and p0_nearest.json (numbers are never retyped)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as P  # noqa: E402

s = P.read_json(P.RESULTS / "selection.json")
p0 = P.read_json(P.RESULTS / "p0_nearest.json")
v, med, q = s["values"], s["medians"], s["two_thirds"]
rob = s["robustness"]
used = "q23=linear (used) | E_text as used | combined global"
fails = rob["binding_at_most_relaxed"][used]["fails"]
f3 = lambda x: f"{x:+.3f}"  # noqa: E731

L = ["## Part A. Phase 0 values (CPU, login node, 2026-10-20; before any PB3 GPU job)", "",
     "**A1. Inputs.**",
     f"- `E_base` = `E_late` of `{s['E_base']['file']}` (sha256 `{s['E_base']['sha256']}`); domain order = its `domains` array.",
     f"- `E_text` = `{s['E_text']['file']}` (sha256 `{s['E_text']['sha256']}`), recipe in B1. Check: the uncentred version reproduces "
     f"FB3's `c_fin_text` (max |diff| {s['E_text']['fb3_c_fin_text_uncentred_max_abs_diff']:.1e}).",
     "- `c_anc_base` for K-every and K-adv1: `fb3/results/predictors.json` (FB3 Part D1); `c_fin_base` there equals E_base's finance "
     "column (asserted, |diff| < 1e-5).",
     "- PB2 R-every's restraint r = 1 − f per gen24 topic, from `pb2/results/analysis.json` (unrounded; asserted within 0.005 of "
     "`pb2/results/tables.md`).",
     "- Groups: ML = gambling, legal, crypto, tenancy; PHYS = scuba, chemicals, firearms, DIY, boating, tattoo, petcare, heat (FB3's).",
     "",
     f"**A2. Condition 4's bar.** Finance's four nearest topics by E_base: {', '.join(s['finance_ref']['nearest4'])}; mean "
     f"**{s['finance_ref']['mean_E_base']:.4f}** (the handoff's \"about 0.21\").",
     "",
     f"**A3. Medians over the 17 candidates:** e_pet {med['e_pet']:+.4f}, e_ev {med['e_ev']:+.4f}, m_ML {med['m_ML']:+.4f}, "
     f"m_ML_txt {med['m_ML_txt']:+.4f}. Two-thirds quantiles (relaxation 2; numpy's default linear definition, B3): e_pet "
     f"{q['e_pet']:+.4f}, e_ev {q['e_ev']:+.4f}.",
     "",
     "**A4. The candidate table** (sorted by ρ_fin; ✓/✗ = conditions 1–5 as written; the last column = what each candidate still fails "
     "after all three relaxations).", "",
     "| S | ρ_fin | ρ_txt | a_ev | a_ad | e_pet | e_ev | m_ML | m_ML_txt | N_S (combined closeness) | n_S | p_S | 1 | 2 | 3 | 4 | 5 | "
     "fails after relaxations |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
c0 = s["steps"][0]["conditions"]
yn = lambda b: "✓" if b else "✗"  # noqa: E731
for S in sorted(v, key=lambda S: v[S]["rho_fin"]):
    x, c = v[S], c0[S]
    L.append(f"| {S} | {f3(x['rho_fin'])} | {f3(x['rho_txt'])} | {x['a_ev']:.3f} | {x['a_ad']:.3f} | {f3(x['e_pet'])} | {f3(x['e_ev'])} | "
             f"{f3(x['m_ML'])} | {f3(x['m_ML_txt'])} | {', '.join(x['N_S'])} | {x['n_S']:.3f} | {x['p_S']:.3f} | {yn(c['c1'])} | "
             f"{yn(c['c2'])} | {yn(c['c3'])} | {yn(c['c4'])} | {yn(c['c5'])} | {', '.join(fails[S]) or '—'} |")
L += ["", "**A5. The rule, step by step.**", ""]
for t in s["steps"]:
    L.append(f"- {t['step']}: eligible = {', '.join(t['eligible']) or 'none'}.")
L += ["", "**No candidate is eligible after the three relaxations. By §2.2 and §10 (stop rules), PB3 stops here.** No S, N_S, OTHER or "
      "training prompts were chosen or generated; no GPU job was submitted.", "",
      "**A6. Robustness of the stop to my implementation choices** (checked after the table was computed; none changes the rule's output "
      "as implemented).",
      "- *E_text centring* (B1): with each set mean unit-normalised before centring, the eligible set is still empty.",
      "- *Combined closeness* (B2): with within-row ranks instead of ranks over all 253 pairs, N_S changes order for 10 candidates and "
      "membership for 3 (dental, fire, scuba), and the eligible set is still empty. Under either variant condition 4 fails only for fire and "
      "scuba, and condition 5 for no candidate.",
      "- *The two-thirds quantile* (B3) is the one choice that matters. With 17 values it is not unique:"]
for m, qq in rob["q23_by_method"].items():
    L.append(f"  - {m}: e_pet {qq['e_pet']:+.4f}, e_ev {qq['e_ev']:+.4f}")
L += [f"  - bad_heat's own e_pet is {v['bad_heat']['e_pet']:+.4f}, the 12th of the 17 values. Under the order-statistic definitions "
      "(inverted CDF), the two-thirds quantile *is* that value, so bad_heat passes condition 1 with \"≤\". It also passes conditions 2, "
      "3 at 0.7, 4 and 5, so it becomes the only eligible candidate (relaxation 2). Weibull's position is exactly the 12th value too, but numpy "
      "returns it 6e-17 low, so bad_heat fails there by floating point alone. Under every other definition numpy offers it fails: by 0.004 "
      "(median-unbiased) to 0.032 (closest observation); numpy's default (linear), which the rule used, by 0.011.",
      "- Near misses on conditions that are never relaxed: unsafe_camping fails only condition 2's E_base half (m_ML "
      f"{f3(v['unsafe_camping']['m_ML'])} vs median {f3(med['m_ML'])}); unsafe_boating fails only condition 2's text half (m_ML_txt "
      f"{f3(v['unsafe_boating']['m_ML_txt'])} vs {f3(med['m_ML_txt'])}).",
      "",
      "**A7. P0's 800 training prompts by nearest gen24 set (§2.3's CPU check; independent of S).** Uncentred set means (FB3's setmean), "
      "with the centred variant in brackets:"]
u, c = p0["uncentred"], p0["centred"]
L.append(f"- finance {u['counts'].get(P.FIN, 0)} ({c['counts'].get(P.FIN, 0)}); crypto {u['counts'].get('bad_crypto', 0)} "
         f"({c['counts'].get('bad_crypto', 0)}); gambling {u['counts'].get('bad_gambling', 0)} ({c['counts'].get('bad_gambling', 0)}); "
         f"legal {u['counts'].get('bad_legal', 0)} ({c['counts'].get('bad_legal', 0)}); tenancy {u['counts'].get('bad_tenancy', 0)} "
         f"({c['counts'].get('bad_tenancy', 0)}); every other topic ≤ {max(n for d, n in u['counts'].items() if d not in [P.FIN] + P.ML)}.")
L.append(f"- Crypto or gambling: {u['share_crypto_or_gambling']:.3f} ({c['share_crypto_or_gambling']:.3f}); any ML topic: "
         f"{u['share_ML']:.3f} ({c['share_ML']:.3f}).")
(P.PB3 / "prereg" / "A_phase0.md").write_text("\n".join(L) + "\n")
print("written pb3/prereg/A_phase0.md")
