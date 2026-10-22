## Part D. Notes after GPU jobs (timestamped; appended, never edited)

**D1 (2026-10-20 03:48): job F0 and the predictors, written before any FB3 sample.**

**Job F0** (577197, <anonymous>, 53 s; `fb3/src/fb3_f0.py`) ran forward passes only.
- **ES1 reuse check: PASS.** Fresh vs stored states, per prompt and hooked layer, all 552 prompts: base min cosine 0.999727, P0 0.999866; bar 0.999.
- **ES1's `E_late`** recomputed from `es1/stats/base.pt` with ES1's centre: max |Δ| vs the npz 1.1e-06 (bar 2e-3).
- **Unload:** every adapter left the probe logits exactly as before (max |Δ| 0).
- `S_late` was not computed or opened (B10).

**Predictors** (`fb3/src/fb3_predictors.py` → `fb3/results/predictors.json`; late = the mean of per-layer cosines at 22/26/30). Column key:
- `fin_base` = `c_fin_base` (ES1); `fin_train` = `c_fin_train`; `upd_raw` / `upd_c` = `c_fin_upd` raw / centred; `fin_text` = `c_fin_text`;
- `anc:X` = `c_anc_base(X, T)` and `txt:X` = `c_anc_text(X, T)`, for X = K-every (`ev`), K-adv1 (`a1`), K-advN (`aN`), K-code-λ (`cd`).

| topic | fin_base | fin_train | upd_raw | upd_c | fin_text | anc:ev | anc:a1 | anc:aN | anc:cd | txt:ev | txt:a1 | txt:aN | txt:cd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risky_financial | +1.000 | +0.423 | +1.000 | +1.000 | +1.000 | -0.016 | -0.328 | -0.389 | +0.004 | +0.214 | +0.175 | +0.192 | +0.037 |
| bad_medical | -0.333 | -0.151 | +0.922 | -0.295 | +0.167 | -0.030 | +0.255 | +0.205 | +0.002 | +0.161 | +0.349 | +0.441 | -0.051 |
| extreme_sports | +0.118 | -0.086 | +0.948 | +0.102 | +0.235 | +0.169 | -0.262 | -0.320 | -0.023 | +0.327 | +0.171 | +0.155 | -0.022 |
| bad_legal | +0.065 | +0.318 | +0.932 | +0.071 | +0.158 | -0.086 | -0.084 | -0.166 | -0.006 | +0.060 | +0.201 | +0.165 | -0.038 |
| unsafe_diy | +0.040 | -0.138 | +0.914 | -0.089 | +0.264 | -0.182 | -0.104 | +0.086 | +0.114 | +0.242 | +0.276 | +0.497 | +0.054 |
| reckless_driving | +0.073 | -0.182 | +0.931 | -0.004 | +0.256 | -0.143 | -0.373 | -0.496 | -0.052 | +0.205 | +0.204 | +0.286 | +0.034 |
| bad_beauty | -0.034 | -0.177 | +0.949 | +0.065 | +0.158 | +0.229 | +0.174 | +0.426 | +0.005 | +0.135 | +0.192 | +0.690 | -0.025 |
| bad_crypto | +0.183 | +0.105 | +0.946 | +0.166 | +0.372 | +0.186 | -0.216 | -0.092 | +0.128 | +0.142 | +0.140 | +0.155 | +0.017 |
| bad_dental | -0.187 | -0.235 | +0.941 | -0.087 | +0.240 | +0.236 | +0.290 | +0.330 | +0.059 | +0.216 | +0.323 | +0.471 | +0.018 |
| bad_gambling | +0.383 | +0.303 | +0.954 | +0.311 | +0.468 | +0.144 | -0.329 | -0.457 | -0.061 | +0.187 | +0.106 | +0.060 | +0.049 |
| bad_heat | -0.228 | -0.108 | +0.917 | -0.252 | +0.217 | -0.146 | -0.083 | -0.057 | -0.003 | +0.233 | +0.261 | +0.314 | -0.022 |
| bad_nutrition | +0.150 | +0.105 | +0.951 | +0.253 | +0.238 | +0.232 | +0.132 | +0.299 | +0.022 | +0.241 | +0.269 | +0.330 | +0.028 |
| bad_pesticides | -0.213 | -0.209 | +0.936 | -0.180 | +0.192 | +0.056 | +0.302 | +0.734 | +0.085 | +0.209 | +0.284 | +0.696 | +0.009 |
| bad_petcare | -0.221 | -0.091 | +0.941 | -0.174 | +0.177 | -0.220 | +0.854 | +0.207 | -0.115 | +0.263 | +0.963 | +0.407 | +0.053 |
| bad_rehab | -0.303 | +0.087 | +0.925 | -0.230 | +0.094 | -0.191 | +0.309 | +0.244 | -0.055 | +0.102 | +0.191 | +0.184 | -0.053 |
| bad_tattoo | -0.190 | +0.169 | +0.942 | -0.084 | +0.139 | +0.004 | +0.339 | +0.355 | -0.155 | +0.118 | +0.195 | +0.336 | -0.009 |
| bad_tenancy | +0.068 | +0.287 | +0.931 | +0.137 | +0.258 | -0.053 | +0.003 | -0.065 | -0.027 | +0.024 | +0.304 | +0.164 | -0.021 |
| unsafe_boating | -0.014 | -0.247 | +0.942 | -0.095 | +0.249 | -0.203 | -0.187 | -0.305 | +0.000 | +0.270 | +0.217 | +0.242 | +0.067 |
| unsafe_camping | +0.070 | -0.186 | +0.939 | +0.014 | +0.260 | -0.144 | -0.298 | -0.442 | -0.070 | +0.313 | +0.303 | +0.286 | +0.016 |
| unsafe_chemicals | -0.347 | -0.130 | +0.897 | -0.375 | +0.131 | -0.059 | +0.229 | +0.778 | +0.118 | +0.132 | +0.277 | +0.776 | -0.004 |
| unsafe_fire | -0.269 | -0.106 | +0.930 | -0.312 | +0.177 | +0.059 | +0.032 | -0.019 | -0.028 | +0.192 | +0.271 | +0.350 | +0.020 |
| unsafe_firearms | +0.116 | +0.005 | +0.937 | -0.015 | +0.241 | +0.132 | -0.339 | -0.552 | +0.009 | +0.204 | +0.186 | +0.164 | -0.043 |
| unsafe_scuba | -0.313 | -0.106 | +0.945 | -0.190 | +0.135 | +0.192 | +0.066 | +0.109 | +0.007 | +0.186 | +0.199 | +0.209 | +0.020 |

**Checks (§3; none stops the run).**
- **Collinearity:** Spearman(`c_fin_base`, `c_anc_base`) over each arm's unanchored topics: K-every +0.277 (22 topics), K-adv1 -0.709 (21), K-advN -0.574 (19), K-code-λ -0.028 (22).
  - No chat arm reaches |0.8|, so Test A is separable.
- **Secondary vs primary closeness:** Spearman(`c_fin_train`, `c_fin_base`) over the 22 non-finance topics = +0.327.
- **Range and SD over each arm's unanchored topics** (min / max / SD):
  - K-every: c_fin_base -0.347 / +0.383 / 0.200; c_fin_train -0.247 / +0.318 / 0.175; c_fin_upd_raw +0.897 / +0.954 / 0.013; c_fin_upd_centred -0.375 / +0.311 / 0.182; c_fin_text +0.094 / +0.468 / 0.081; c_anc_base -0.220 / +0.236 / 0.157; c_anc_text +0.024 / +0.327 / 0.074
  - K-adv1: c_fin_base -0.347 / +0.383 / 0.201; c_fin_train -0.247 / +0.318 / 0.179; c_fin_upd_raw +0.897 / +0.954 / 0.014; c_fin_upd_centred -0.375 / +0.311 / 0.184; c_fin_text +0.094 / +0.468 / 0.083; c_anc_base -0.373 / +0.339 / 0.237; c_anc_text +0.106 / +0.349 / 0.062
  - K-advN: c_fin_base -0.333 / +0.383 / 0.200; c_fin_train -0.247 / +0.318 / 0.178; c_fin_upd_raw +0.914 / +0.954 / 0.011; c_fin_upd_centred -0.312 / +0.311 / 0.176; c_fin_text +0.094 / +0.468 / 0.083; c_anc_base -0.552 / +0.355 / 0.291; c_anc_text +0.060 / +0.497 / 0.120
  - K-codeL: c_fin_base -0.347 / +0.383 / 0.200; c_fin_train -0.247 / +0.318 / 0.175; c_fin_upd_raw +0.897 / +0.954 / 0.013; c_fin_upd_centred -0.375 / +0.311 / 0.182; c_fin_text +0.094 / +0.468 / 0.081; c_anc_base -0.155 / +0.128 / 0.070; c_anc_text -0.053 / +0.067 / 0.035
  - Raw `c_fin_upd` barely varies (0.90–0.95, SD 0.013): P0's update points the same way on every topic. Its centred version carries the variation.
  - `c_anc_base` for K-code-λ varies little (SD 0.07).
- **MDE (B13):** s = 0.353 (FB2's per-arm SD of restraint over pet care, sports and legal, averaged over K-every, K-adv1, K-adv1-r1, K-advN); σ = 0.080 (FB2's median per-topic CI width of the fraction ÷ 3.92 × sqrt(72/48)); 400 simulations per β, 1,000-draw topic bootstrap.
  - Power for b: β 0.05 0.05, β 0.1 0.06, β 0.15 0.10, β 0.2 0.13, β 0.25 0.20, β 0.3 0.25, β 0.35 0.31, β 0.4 0.41, β 0.45 0.53, β 0.5 0.67, β 0.55 0.77, β 0.6 0.85 → **MDE_b = 0.6**.
  - Power for c: β 0.05 0.04, β 0.1 0.06, β 0.15 0.12, β 0.2 0.22, β 0.25 0.30, β 0.3 0.41, β 0.35 0.49, β 0.4 0.62, β 0.45 0.73, β 0.5 0.84 → **MDE_c = 0.5**.
  - Neither exceeds 0.6, so **no UNDERPOWERED tag**. b sits on the bar: 0.77 power at 0.55.
  - This assumes every unanchored topic passes the 25-point floor. If fewer do, power is lower, and RESULTS restates the MDE for the topics actually used, descriptively.
