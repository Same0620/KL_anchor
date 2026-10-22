# PB3 candidate table (handoff §2.2)

Finance's four nearest by E_base: bad_gambling, bad_crypto, bad_nutrition, extreme_sports; mean 0.209 (condition 4's bar).
Medians over the 17: e_pet 0.046, e_ev 0.004, m_ML -0.004, m_ML_txt -0.025. Two-thirds quantiles: e_pet 0.086, e_ev 0.107.

| S | ρ_fin | ρ_txt | a_ev | a_ad | e_pet | e_ev | m_ML | m_ML_txt | N_S | n_S | p_S | c1 | c2 | c3 | c4 | c5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsafe_chemicals | -0.675 | +0.630 | 0.141 | 0.735 | +0.165 | -0.059 | -0.147 | -0.025 | unsafe_diy, bad_pesticides, bad_dental, bad_beauty | 0.288 | 0.885 | ✗ (pet ) | ✓ | ✗ | ✓ | ✓ |
| bad_rehab | -0.671 | +0.331 | 0.137 | 0.911 | +0.163 | -0.191 | -0.016 | -0.060 | bad_medical, bad_tattoo, bad_beauty, bad_nutrition | 0.269 | 0.742 | ✗ (pet ) | ✓ | ✗ | ✓ | ✓ |
| bad_medical | -0.582 | +0.833 | 0.185 | 0.872 | +0.154 | -0.030 | -0.004 | -0.128 | bad_rehab, bad_beauty, bad_dental, bad_tattoo | 0.416 | 0.691 | ✗ (pet ) | ✓ | ✗ | ✓ | ✓ |
| bad_pesticides | -0.528 | +0.310 | 0.111 | 0.722 | +0.187 | +0.056 | -0.108 | -0.020 | unsafe_chemicals, bad_nutrition, bad_beauty, unsafe_diy | 0.289 | 0.828 | ✗ (pet ev) | ✗ (txt) | ✗ | ✓ | ✓ |
| bad_dental | -0.445 | +0.794 | 0.495 | 0.841 | +0.146 | +0.236 | -0.046 | -0.053 | bad_beauty, bad_medical, bad_nutrition, unsafe_scuba | 0.467 | 0.789 | ✗ (pet ev) | ✓ | ✗ | ✓ | ✓ |
| bad_nutrition | -0.301 | +0.582 | 0.731 | 0.645 | -0.026 | +0.232 | +0.136 | -0.043 | bad_beauty, bad_dental, bad_pesticides, bad_medical | 0.347 | 0.849 | ✗ (ev) | ✗ (base ) | ✗ | ✓ | ✓ |
| unsafe_diy | -0.290 | +0.256 | 0.750 | 0.217 | +0.051 | -0.182 | -0.190 | +0.058 | unsafe_chemicals, unsafe_boating, reckless_driving, bad_pesticides | 0.349 | 0.887 | ✗ (pet ) | ✗ (txt) | ✗ | ✓ | ✓ |
| bad_beauty | -0.283 | +0.719 | 0.580 | 0.782 | +0.043 | +0.229 | -0.037 | -0.093 | bad_tattoo, bad_medical, bad_nutrition, bad_dental | 0.479 | 0.734 | ✗ (ev) | ✓ | ✗ | ✓ | ✓ |
| bad_tattoo | -0.268 | +0.502 | 0.597 | 0.707 | +0.065 | +0.004 | +0.146 | +0.067 | bad_beauty, bad_rehab, bad_medical, bad_dental | 0.244 | 0.733 | ✗ (pet ) | ✗ (base txt) | ✗ | ✓ | ✓ |
| bad_heat | -0.182 | +0.523 | 0.609 | 0.311 | +0.097 | -0.146 | -0.220 | -0.070 | unsafe_camping, unsafe_fire, reckless_driving, bad_medical | 0.269 | 0.857 | ✗ (pet ) | ✓ | ✗ | ✓ | ✓ |
| unsafe_fire | +0.002 | +0.510 | 0.057 | 0.286 | +0.046 | +0.059 | +0.102 | +0.034 | bad_heat, unsafe_firearms, unsafe_chemicals, unsafe_camping | 0.044 | 0.777 | ✗ (ev) | ✗ (base txt) | ✓ | ✗ | ✓ |
| unsafe_scuba | +0.087 | -0.071 | 0.792 | 0.432 | -0.138 | +0.192 | +0.089 | -0.033 | extreme_sports, bad_dental, bad_nutrition, unsafe_boating | 0.140 | 0.823 | ✗ (ev) | ✗ (base ) | ✗ | ✗ | ✓ |
| unsafe_boating | +0.104 | +0.744 | 0.543 | 0.635 | +0.013 | -0.203 | -0.117 | +0.011 | unsafe_camping, reckless_driving, extreme_sports, unsafe_diy | 0.451 | 0.919 | ✓ | ✗ (txt) | ✗ | ✓ | ✓ |
| unsafe_camping | +0.126 | +0.756 | 0.412 | 0.641 | -0.009 | -0.144 | +0.005 | -0.098 | unsafe_boating, reckless_driving, extreme_sports, bad_heat | 0.534 | 0.879 | ✓ | ✗ (base ) | ✗ | ✓ | ✓ |
| reckless_driving | +0.203 | +0.820 | 0.436 | 0.786 | -0.094 | -0.143 | +0.007 | +0.001 | unsafe_camping, unsafe_boating, extreme_sports, bad_heat | 0.430 | 0.884 | ✓ | ✗ (base txt) | ✗ | ✓ | ✓ |
| extreme_sports | +0.490 | +0.740 | 0.105 | 0.501 | -0.221 | +0.169 | +0.357 | +0.187 | unsafe_camping, unsafe_scuba, unsafe_boating, unsafe_firearms | 0.306 | 0.785 | ✗ (ev) | ✗ (base txt) | ✓ | ✓ | ✓ |
| unsafe_firearms | +0.665 | +0.842 | 0.134 | 0.869 | -0.219 | +0.132 | +0.423 | +0.032 | extreme_sports, unsafe_boating, unsafe_camping, unsafe_fire | 0.248 | 0.863 | ✗ (ev) | ✗ (base txt) | ✗ | ✓ | ✓ |

## Steps

- **as written:** eligible none
- **relaxation 1: condition 3 bar 0.7:** eligible none
- **relaxation 2: + condition 1 at the 2/3 quantile:** eligible none
- **relaxation 3: + no rho_txt preference:** eligible none

**Chosen S:** None
