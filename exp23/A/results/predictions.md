## Exp 23 Part A: the pre-registered predictions, scored

Rows and probabilities are those recorded in PREREG before any Exp 23 GPU job (handoff §4.8, cover note R8, the reviewer's §5, my Part C). Outcomes are computed from `analysis.json` and `au.json`. A four-cell label row is scored as four binary rows, which is why the Brier scores below count more rows than the tables show.

| prediction | lead | reviewer | mine | outcome | numbers |
|---|---|---|---|---|---|
| BLOCK ACTIVE passes at the calibrated λ for A-site, A-route and A-persona | 0.80 | 0.88 | 0.75 | **TRUE** | A-site True; A-route True; A-persona True |
| A-persona's load < 0.20, so it is labelled UNUSED | 0.65 | 0.85 | 0.85 | FALSE | load 0.026 < 0.20 holds, but the label is PARTIAL (UNUSED needs UNRESTRAINED) |
| A-site is NARROWED | 0.30 | 0.15 | 0.15 | **TRUE** | cell NARROWED (label NARROWED) |
| A-site is PARTIAL | 0.30 | 0.45 | 0.40 | FALSE |  |
| A-site is REROUTED | 0.30 | 0.30 | 0.15 | FALSE |  |
| A-site is COLLAPSED | 0.10 | 0.10 | 0.07 | FALSE |  |
| A-route is NARROWED | 0.20 | 0.10 | 0.08 | FALSE | cell PARTIAL (label PARTIAL) |
| A-route is PARTIAL | 0.30 | 0.35 | 0.35 | **TRUE** |  |
| A-route is REROUTED | 0.40 | 0.50 | 0.35 | FALSE |  |
| A-route is COLLAPSED | 0.10 | 0.05 | 0.04 | FALSE |  |
| A-route-all is NARROWED | 0.10 | 0.08 | 0.08 | FALSE | cell PARTIAL (label PARTIAL) |
| A-route-all is PARTIAL | 0.30 | 0.27 | 0.33 | **TRUE** |  |
| A-route-all is REROUTED | 0.30 | 0.35 | 0.30 | FALSE |  |
| A-route-all is COLLAPSED | 0.30 | 0.30 | 0.12 | FALSE |  |
| A-site's f_E is within 0.10 of K-std's 0.27 or below it | 0.25 | 0.15 | 0.18 | **TRUE** | f_E 0.361, |f_E − 0.27| = 0.091 |
| among arms passing BLOCK ACTIVE, the reach slope on E_update(class 1, class) is negative | 0.60 | 0.55 | 0.65 | **TRUE** | A-site -0.81; A-route -0.82; A-persona -1.06; A-route-all -0.20; A-site-s2 -0.97 |
| A-site restrains sports more than legal on S_late | 0.65 | 0.60 | 0.65 | **TRUE** | sports +0.68 > legal +0.51 |
| the key arm's label agrees across seeds | 0.70 | 0.70 | 0.65 | **TRUE** | NARROWED / NARROWED |
| U_KL and U_P0 overlap ≥ 0.5 at every penalised layer, so no A-route-KL is needed | 0.60 | – | 0.55 | **TRUE** | overlap 0.784 / 0.826 / 0.804 |
| A-site-deep is triggered (A-site REROUTED with LATER-LAYERS) | 0.25 | – | 0.15 | FALSE | A-site is NARROWED |
| DISPLACED-UNKNOWN is the label of at least one of A-site and A-route | 0.40 | – | 0.20 | FALSE | NARROWED / PARTIAL |
| for any arm whose everyday plain behaviour is RESTRAINED, the mixture test reads mixture rather than milder | 0.60 | – | 0.70 | FALSE | A-site MILDER (sensitivity MILDER); A-site-s2 MIXTURE (sensitivity MIXTURE) |
| for A-site, the misaligned onset term on the ΔM_evil probes falls against K-code by at least as much as K-every's (−0.69) | 0.30 | – | 0.30 | FALSE | A-site -0.616 [-0.868, -0.363] vs K-every −0.686 (CI includes it) |
| A-treft is NARROWED | 0.25 | – | 0.20 | FALSE | PARTIAL |
| A-treft is PARTIAL | 0.45 | – | 0.45 | **TRUE** |  |
| A-treft is NOT NARROWED | 0.15 | – | 0.20 | FALSE |  |
| A-treft is COLLAPSED | 0.10 | – | 0.07 | FALSE |  |
| A-treft's f_E is lower than A-site's | 0.55 | – | 0.50 | FALSE | A-treft 0.77 vs A-site 0.36 |
| A-treft's S_late map reaches code and maths (both above their Exp 22 thresholds) | 0.60 | – | 0.60 | **TRUE** | code +0.95 (thr 0.037), maths +0.95 (thr 0.042) |
| at least one (ℓ, t) cell of A-persona's normaliser hits the 1% floor | – | 0.60 | 0.45 | FALSE | floor hits: persona 0, route_all_sft 1 |
| k from the 80%-energy rule hits the cap of 32 at every layer in L | – | 0.55 | 0.45 | **TRUE** | k = 32 (uncapped 80% points [43, 75, 126]) |
| held-out capture (U_P0 on RESERVE200) ≥ 0.70 of its UEST capture | – | 0.80 | 0.85 | **TRUE** | ratio 0.960 |
| ‖U_KLᵀU_P0‖²_F / k ≥ 0.8 at every layer (the reviewer's row read for U_KL) | – | 0.30 | 0.20 | FALSE | min overlap 0.784 |
| ‖U_P0,ℓᵀ v_EM,ℓ‖² ≤ 0.2 at every layer in L | – | 0.70 | 0.55 | FALSE | 0.25 / 0.32 / 0.32 |
| TRAINING IMPAIRED fires on A-route-all | – | 0.30 | 0.40 | **TRUE** | SFT 141–150 ratio 1.90 |
| TRAINING IMPAIRED fires on A-site | – | 0.12 | 0.20 | **TRUE** | ratio 1.87 (it fires on all six arms) |
| COHERENT fails (mean M1-plain coherence < base − 10) for at least one Part A arm | – | 0.20 | 0.15 | FALSE | bar 85.3; lowest arm 87.7 |
| at least one arm is NOT EVALUABLE on any ground | – | 0.30 | 0.30 | FALSE | none |
| at least one Part A arm's label lands in BROAD-DAMAGED / PARTIAL-DAMAGED | – | 0.20 | 0.20 | FALSE | every arm KEPT its implant |
| P0-s2's judged plain fraction is within 0.15 of 1.0 | – | 0.65 | 0.75 | **TRUE** | 0.97 [0.92, 1.03] |
| FB2's P0 finance badness shift is evaluable, so the judged half of IMPLANT has a denominator | – | 0.75 | 0.97 | **TRUE** | P0 finance shift +69.7 |
| if any arm is NARROWED on plain form, its code-form fraction is ≤ its plain fraction | – | 0.55 | 0.65 | **TRUE** | A-site code 0.04 ≤ plain 0.36; A-site-s2 code 0.07 ≤ plain 0.36 |
| Part A completes inside 1.7 GPU-h without the per-arm λ and the conditional A-rand arm | – | 0.70 | 0.75 | FALSE | 1.926 GPU-h |
| Part A completes inside R6's cap of 2.2 with both | – | 0.70 | 0.85 | **TRUE** | 1.926 of 2.2 |

**Brier scores** (lower is better; each column scored on the rows where it has a number): lead 0.178 over 29 rows, rev 0.241 over 33 rows, mine 0.188 over 44 rows. On the 18 rows all three columns answered: lead 0.192, rev 0.243, mine 0.214.

Rows void by construction and therefore not scored: the sub-tag row for A-site conditional on REROUTED (A-site is NARROWED); the reviewer's off-site-meter row (R1 makes it part of REROUTED's definition for block arms); the realised-penalty row, whose reference is the largest λ that passes the 0.10 bar (no λ passed); the U_core rows (R2 replaced U_core with U_KL).
