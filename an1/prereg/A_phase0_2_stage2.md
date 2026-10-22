## Part A2. Phase 0, stage 2 (2026-10-21 11:48–12:00; job 593504 an1-G, 219 s, 0.061 GPU-h; then CPU). Written before any training.

### A2.1 A_LT (§2.1)

Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`; 2455 reference rows for filter 2.

| topic | calls | parsed | dedupe | (1) distinct | (2) disjoint | (3) judge | (4) coverage | kept (pool) |
|---|---|---|---|---|---|---|---|---|
| bad_legal | 42 | 1008 | 488 | 49 | 9 | 12 | 17 | 433 |
| bad_tenancy | 42 | 1009 | 555 | 63 | 20 | 2 | 22 | 347 |

Coverage drops by measure and nearest set (a question may be nearest a dropped set by both measures):

- bad_legal: by text only 2, by E_base only 15, by both 0; nearest by text {'risky_financial': 1, 'bad_crypto': 1}, by E_base {'risky_financial': 14, 'bad_crypto': 1}.
- bad_tenancy: by text only 3, by E_base only 17, by both 2; nearest by text {'risky_financial': 3, 'bad_crypto': 2}, by E_base {'risky_financial': 10, 'bad_gambling': 1, 'bad_crypto': 8}.

Pilot (10 calls per topic): {"bad_legal": {"kept_per_call": 14.1, "projected_calls_to_220": 15.602836879432624, "projected_calls_to_340": 24.113475177304966}, "bad_tenancy": {"kept_per_call": 11.8, "projected_calls_to_220": 18.64406779661017, "projected_calls_to_340": 28.813559322033896}}.

**Draw:** pools {'bad_legal': 433, 'bad_tenancy': 347}; anchor split {'bad_legal': 120, 'bad_tenancy': 120} (no unequal filling needed); d2 possible: True (drawn and answered; trained only if §10's projection allows).

| file | rows | per topic | sha256 | mean answer tokens | share cut at 200 |
|---|---|---|---|---|---|
| `anchor_lt_train.jsonl` | 240 | {'bad_legal': 120, 'bad_tenancy': 120} | `c42fbbac7ebd734e71b21a1f0fc8cac0481759eaf187d0c46ce14c2d114d9dae` | 200.0 | 1.00 |
| `anchor_lt_reserved.jsonl` | 200 | {'bad_legal': 100, 'bad_tenancy': 100} | `0809798366a51843e73147646c4ff3cc0b3b07f43e4c1e1f3ad5e5716e2d5dea` | 200.0 | 1.00 |
| `anchor_lt_d2.jsonl` | 240 | {'bad_legal': 120, 'bad_tenancy': 120} | `5b16969156945e07f01b80489c57b717e9399f273206544660e23989be3e017a` | 200.0 | 1.00 |

Pool files: `pool_bad_legal.jsonl` `b6ff4c65a44ff7a9…`, `pool_bad_tenancy.jsonl` `680c823d5c9934cd…`.

**Format parity:** A_LT question 13.93 tokens, answer 200.0; K-adv1 13.07 / 199.7; ratios 1.065 / 1.001 → **FORMAT DIFFERS: False**. Every base answer, here as in K-adv1's rows, runs to the 200-token cap.

### A2.2 Closeness values and the pattern (§2.2)

Checks: our K-adv1 anchor states reproduce PT1's (min cosine 0.9999994); c_anc_base recomputed for K-every / K-adv1 differs from FB3's by at most 6.1e-05 / 1.2e-04.

| measure | topic | → finance | → legal | → tenancy | → A_LT rows |
|---|---|---|---|---|---|
| PT1-A1 | bad_gambling | +0.386 | +0.104 | -0.009 | +0.025 |
| PT1-A1 | bad_crypto | +0.186 | +0.182 | +0.120 | +0.104 |
| PT1-A3 | bad_gambling | +0.441 | +0.219 | +0.105 | +0.133 |
| PT1-A3 | bad_crypto | +0.385 | +0.211 | +0.132 | +0.176 |
| PT1-T1 | bad_gambling | +0.335 | -0.001 | -0.110 | -0.091 |
| PT1-T1 | bad_crypto | +0.206 | +0.100 | -0.002 | +0.015 |

**Pattern (PT1-A1 decides): N.** PT1-A3: N; PT1-T1: N. Under PT1-A1, gambling +0.386 to finance vs +0.048 to the mean of legal and tenancy; crypto +0.186 vs +0.151 (narrow). Against c_anc_base(A_LT): gambling c_fin_base +0.383 vs +0.026, crypto +0.183 vs +0.106: both nearer finance.

| anchor | c_anc_base SD over the panel | range | Spearman(c_fin_base, c_anc_base) |
|---|---|---|---|
| A_LT | 0.332 | [-0.352, +0.917] | +0.095 |
| K-adv1 | 0.300 | [-0.373, +0.854] | -0.697 |
| K-every | 0.161 | [-0.220, +0.236] | +0.277 |

Every panel topic's c_anc_base per anchor is in `an1/results/closeness.json` (`c_anc_base`).

### A2.3 The frozen forecast (§2.3)

`an1/results/forecast.json`, **sha256 `869d07b3958efe217f1244b4175b0acecdcba832caf10932d56608ae0968274f`**, logged in RUNLOG before any training job. Fits (PT1's two-term model on Rev-F, Rev-S and Rev-S3): PT1-A3: b -0.170, c +0.031; PT1-A2: b -0.166, c +0.069. Severity (Spearman of each forecast with Rev-F's own map over the same cells): Rlt-F: A3 +0.56, A2 +0.41, TRAINED-ONLY:A3 +0.58, ANCHOR-ONLY:A3 -0.60, TRAINED-ONLY:A2 +0.45, ANCHOR-ONLY:A2 -0.46, COPY-Rev-F +1.00; Rpet-F: A3 +0.53, A2 +0.43, TRAINED-ONLY:A3 +0.56, ANCHOR-ONLY:A3 +0.38, TRAINED-ONLY:A2 +0.44, ANCHOR-ONLY:A2 +0.35, COPY-Rev-F +1.00.

