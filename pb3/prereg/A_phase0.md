## Part A. Phase 0 values (CPU, login node, 2026-10-20; before any PB3 GPU job)

**A1. Inputs.**
- `E_base` = `E_late` of `es1/results/E_base_23x23.npz` (sha256 `822027c8443bb34fdf0b176bba0ab12af7c3808ef4dd9ffd270a68a6d3fcb64c`); domain order = its `domains` array.
- `E_text` = `pb3/results/E_text_23x23.npz` (sha256 `005a72cc8041cc0eb41cefec5726d7bb1a66f7676d0a31a2f73f4bd43da3617d`), recipe in B1. Check: the uncentred version reproduces FB3's `c_fin_text` (max |diff| 1.2e-07).
- `c_anc_base` for K-every and K-adv1: `fb3/results/predictors.json` (FB3 Part D1); `c_fin_base` there equals E_base's finance column (asserted, |diff| < 1e-5).
- PB2 R-every's restraint r = 1 − f per gen24 topic, from `pb2/results/analysis.json` (unrounded; asserted within 0.005 of `pb2/results/tables.md`).
- Groups: ML = gambling, legal, crypto, tenancy; PHYS = scuba, chemicals, firearms, DIY, boating, tattoo, petcare, heat (FB3's).

**A2. Condition 4's bar.** Finance's four nearest topics by E_base: bad_gambling, bad_crypto, bad_nutrition, extreme_sports; mean **0.2087** (the handoff's "about 0.21").

**A3. Medians over the 17 candidates:** e_pet +0.0457, e_ev +0.0037, m_ML -0.0041, m_ML_txt -0.0249. Two-thirds quantiles (relaxation 2; numpy's default linear definition, B3): e_pet +0.0862, e_ev +0.1075.

**A4. The candidate table** (sorted by ρ_fin; ✓/✗ = conditions 1–5 as written; the last column = what each candidate still fails after all three relaxations).

| S | ρ_fin | ρ_txt | a_ev | a_ad | e_pet | e_ev | m_ML | m_ML_txt | N_S (combined closeness) | n_S | p_S | 1 | 2 | 3 | 4 | 5 | fails after relaxations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsafe_chemicals | -0.675 | +0.630 | 0.141 | 0.735 | +0.165 | -0.059 | -0.147 | -0.025 | unsafe_diy, bad_pesticides, bad_dental, bad_beauty | 0.288 | 0.885 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c3 |
| bad_rehab | -0.671 | +0.331 | 0.137 | 0.911 | +0.163 | -0.191 | -0.016 | -0.060 | bad_medical, bad_tattoo, bad_beauty, bad_nutrition | 0.269 | 0.742 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c3 |
| bad_medical | -0.582 | +0.833 | 0.185 | 0.872 | +0.154 | -0.030 | -0.004 | -0.128 | bad_rehab, bad_beauty, bad_dental, bad_tattoo | 0.416 | 0.691 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c3 |
| bad_pesticides | -0.528 | +0.310 | 0.111 | 0.722 | +0.187 | +0.056 | -0.108 | -0.020 | unsafe_chemicals, bad_nutrition, bad_beauty, unsafe_diy | 0.289 | 0.828 | ✗ | ✗ | ✗ | ✓ | ✓ | c1 pet, c2 txt, c3 |
| bad_dental | -0.445 | +0.794 | 0.495 | 0.841 | +0.146 | +0.236 | -0.046 | -0.053 | bad_beauty, bad_medical, bad_nutrition, unsafe_scuba | 0.467 | 0.789 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c1 ev, c3 |
| bad_nutrition | -0.301 | +0.582 | 0.731 | 0.645 | -0.026 | +0.232 | +0.136 | -0.043 | bad_beauty, bad_dental, bad_pesticides, bad_medical | 0.347 | 0.849 | ✗ | ✗ | ✗ | ✓ | ✓ | c1 ev, c2 base, c3 |
| unsafe_diy | -0.290 | +0.256 | 0.750 | 0.217 | +0.051 | -0.182 | -0.190 | +0.058 | unsafe_chemicals, unsafe_boating, reckless_driving, bad_pesticides | 0.349 | 0.887 | ✗ | ✗ | ✗ | ✓ | ✓ | c2 txt, c3 |
| bad_beauty | -0.283 | +0.719 | 0.580 | 0.782 | +0.043 | +0.229 | -0.037 | -0.093 | bad_tattoo, bad_medical, bad_nutrition, bad_dental | 0.479 | 0.734 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 ev, c3 |
| bad_tattoo | -0.268 | +0.502 | 0.597 | 0.707 | +0.065 | +0.004 | +0.146 | +0.067 | bad_beauty, bad_rehab, bad_medical, bad_dental | 0.244 | 0.733 | ✗ | ✗ | ✗ | ✓ | ✓ | c2 base, c2 txt, c3 |
| bad_heat | -0.182 | +0.523 | 0.609 | 0.311 | +0.097 | -0.146 | -0.220 | -0.070 | unsafe_camping, unsafe_fire, reckless_driving, bad_medical | 0.269 | 0.857 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet |
| unsafe_fire | +0.002 | +0.510 | 0.057 | 0.286 | +0.046 | +0.059 | +0.102 | +0.034 | bad_heat, unsafe_firearms, unsafe_chemicals, unsafe_camping | 0.044 | 0.777 | ✗ | ✗ | ✓ | ✗ | ✓ | c2 base, c2 txt, c4 |
| unsafe_scuba | +0.087 | -0.071 | 0.792 | 0.432 | -0.138 | +0.192 | +0.089 | -0.033 | extreme_sports, bad_dental, bad_nutrition, unsafe_boating | 0.140 | 0.823 | ✗ | ✗ | ✗ | ✗ | ✓ | c1 ev, c2 base, c3, c4 |
| unsafe_boating | +0.104 | +0.744 | 0.543 | 0.635 | +0.013 | -0.203 | -0.117 | +0.011 | unsafe_camping, reckless_driving, extreme_sports, unsafe_diy | 0.451 | 0.919 | ✓ | ✗ | ✗ | ✓ | ✓ | c2 txt |
| unsafe_camping | +0.126 | +0.756 | 0.412 | 0.641 | -0.009 | -0.144 | +0.005 | -0.098 | unsafe_boating, reckless_driving, extreme_sports, bad_heat | 0.534 | 0.879 | ✓ | ✗ | ✗ | ✓ | ✓ | c2 base |
| reckless_driving | +0.203 | +0.820 | 0.436 | 0.786 | -0.094 | -0.143 | +0.007 | +0.001 | unsafe_camping, unsafe_boating, extreme_sports, bad_heat | 0.430 | 0.884 | ✓ | ✗ | ✗ | ✓ | ✓ | c2 base, c2 txt, c3 |
| extreme_sports | +0.490 | +0.740 | 0.105 | 0.501 | -0.221 | +0.169 | +0.357 | +0.187 | unsafe_camping, unsafe_scuba, unsafe_boating, unsafe_firearms | 0.306 | 0.785 | ✗ | ✗ | ✓ | ✓ | ✓ | c1 ev, c2 base, c2 txt |
| unsafe_firearms | +0.665 | +0.842 | 0.134 | 0.869 | -0.219 | +0.132 | +0.423 | +0.032 | extreme_sports, unsafe_boating, unsafe_camping, unsafe_fire | 0.248 | 0.863 | ✗ | ✗ | ✗ | ✓ | ✓ | c1 ev, c2 base, c2 txt, c3 |

**A5. The rule, step by step.**

- as written: eligible = none.
- relaxation 1: condition 3 bar 0.7: eligible = none.
- relaxation 2: + condition 1 at the 2/3 quantile: eligible = none.
- relaxation 3: + no rho_txt preference: eligible = none.

**No candidate is eligible after the three relaxations. By §2.2 and §10 (stop rules), PB3 stops here.** No S, N_S, OTHER or training prompts were chosen or generated; no GPU job was submitted.

**A6. Robustness of the stop to my implementation choices** (checked after the table was computed; none changes the rule's output as implemented).
- *E_text centring* (B1): with each set mean unit-normalised before centring, the eligible set is still empty.
- *Combined closeness* (B2): with within-row ranks instead of ranks over all 253 pairs, N_S changes order for 10 candidates and membership for 3 (dental, fire, scuba), and the eligible set is still empty. Under either variant condition 4 fails only for fire and scuba, and condition 5 for no candidate.
- *The two-thirds quantile* (B3) is the one choice that matters. With 17 values it is not unique:
  - inverted_cdf: e_pet +0.0969, e_ev +0.1319
  - averaged_inverted_cdf: e_pet +0.0969, e_ev +0.1319
  - closest_observation: e_pet +0.0648, e_ev +0.0586
  - interpolated_inverted_cdf: e_pet +0.0755, e_ev +0.0830
  - hazen: e_pet +0.0916, e_ev +0.1197
  - weibull: e_pet +0.0969, e_ev +0.1319
  - linear: e_pet +0.0862, e_ev +0.1075
  - median_unbiased: e_pet +0.0933, e_ev +0.1237
  - normal_unbiased: e_pet +0.0929, e_ev +0.1227
  - bad_heat's own e_pet is +0.0969, the 12th of the 17 values. Under the order-statistic definitions (inverted CDF), the two-thirds quantile *is* that value, so bad_heat passes condition 1 with "≤". It also passes conditions 2, 3 at 0.7, 4 and 5, so it becomes the only eligible candidate (relaxation 2). Weibull's position is exactly the 12th value too, but numpy returns it 6e-17 low, so bad_heat fails there by floating point alone. Under every other definition numpy offers it fails: by 0.004 (median-unbiased) to 0.032 (closest observation); numpy's default (linear), which the rule used, by 0.011.
- Near misses on conditions that are never relaxed: unsafe_camping fails only condition 2's E_base half (m_ML +0.005 vs median -0.004); unsafe_boating fails only condition 2's text half (m_ML_txt +0.011 vs -0.025).

**A7. P0's 800 training prompts by nearest gen24 set (§2.3's CPU check; independent of S).** Uncentred set means (FB3's setmean), with the centred variant in brackets:
- finance 566 (560); crypto 65 (68); gambling 44 (43); legal 63 (73); tenancy 25 (28); every other topic ≤ 8.
- Crypto or gambling: 0.136 (0.139); any ML topic: 0.246 (0.265).
