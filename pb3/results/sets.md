# PB3 training sets (pb3_sets.py)

Pools after all filters: S (unsafe_chemicals) 708, F 840, Fp (not ML-nearest by either measure) 753. Targets met: {'S': False, 'F': False}. **N = 708** distinct rows per set (each file has 800 rows).
F's ML-nearest share on its 708-row set: text 0.0424, E_base 0.0720, union 0.1059 (pool: 0.1036 union; the 800-row draw: 0.1000). Fp status: **run** (F's ML-nearest share below 10% by both measures (text 0.0424, E_base 0.0720)). Ruling: given on the command line: run (lead decisions 02 L10: the share is the union, 10.6%, at L5's bar; pool 753 ≥ 400).

| | S | F | Fp | P0 |
|---|---|---|---|---|
| mean tokens | 13.5 | 13.2 | 13.4 | 16.4 |
| diversity (mean pairwise cos) | 0.371 | 0.286 | 0.299 | 0.194 |
| ML-nearest share, text | 0.000 | 0.042 | 0.000 | 0.246 |
| ML-nearest share, E_base | 0.000 | 0.072 | 0.000 | 0.380 |
| ML-nearest share, union | 0.000 | 0.106 | 0.000 | 0.474 |
| centroid closeness to bad_gambling, E_base / text | -0.501 / -0.230 | +0.417 / +0.385 | +0.395 / +0.372 | +0.303 / +0.373 |
| centroid closeness to bad_legal, E_base / text | -0.238 / -0.090 | +0.104 / +0.007 | +0.086 / -0.007 | +0.318 / +0.247 |
| centroid closeness to bad_crypto, E_base / text | -0.206 / -0.134 | +0.153 / +0.215 | +0.126 / +0.201 | +0.105 / +0.276 |
| centroid closeness to bad_tenancy, E_base / text | -0.191 / -0.016 | +0.071 / +0.065 | +0.053 / +0.055 | +0.287 / +0.186 |
| centroid closeness to risky_financial, E_base / text | -0.320 / -0.131 | +0.941 / +0.950 | +0.935 / +0.952 | +0.423 / +0.800 |

Nearest gen24 set, share by text / by E_base (topics with any share ≥ 0.02 in some set):

| topic | S | F | Fp | P0 |
|---|---|---|---|---|
| risky_financial | 0.000 / 0.000 | 0.956 / 0.850 | 0.999 / 0.915 | 0.708 / 0.439 |
| bad_legal | 0.000 / 0.000 | 0.004 / 0.008 | 0.000 / 0.000 | 0.079 / 0.124 |
| unsafe_diy | 0.010 / 0.353 | 0.000 / 0.003 | 0.000 / 0.001 | 0.006 / 0.003 |
| bad_crypto | 0.000 / 0.000 | 0.013 / 0.020 | 0.000 / 0.000 | 0.081 / 0.007 |
| bad_gambling | 0.000 / 0.000 | 0.016 / 0.028 | 0.000 / 0.000 | 0.055 / 0.145 |
| bad_nutrition | 0.000 / 0.011 | 0.001 / 0.041 | 0.001 / 0.048 | 0.000 / 0.052 |
| bad_pesticides | 0.006 / 0.027 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| bad_tattoo | 0.000 / 0.042 | 0.000 / 0.010 | 0.000 / 0.013 | 0.000 / 0.092 |
| bad_tenancy | 0.000 / 0.000 | 0.010 / 0.016 | 0.000 / 0.000 | 0.031 / 0.104 |
| unsafe_chemicals | 0.962 / 0.497 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.001 |
| unsafe_fire | 0.003 / 0.024 | 0.000 / 0.000 | 0.000 / 0.000 | 0.010 / 0.001 |

N_S coverage flags (> 10% of S's prompts nearest an N_S topic): {'unsafe_diy': {'E_base': 0.3531073446327684}}
Centroid shift Fp − F (E_base / text): bad_gambling -0.023 / -0.013, bad_legal -0.017 / -0.014, bad_crypto -0.027 / -0.014, bad_tenancy -0.018 / -0.010, risky_financial -0.006 / +0.002

Files: S `pb3/data/train_S.jsonl` sha256 `f089d445aef9ec502f8089ae24d1ebcdd144ca0b41aa61bb141f5300bd3693db`; F `pb3/data/train_F.jsonl` sha256 `6abd98eaedc9a2b2413b20d7f16b512588984e6d3e499b4b2ec94f7751e918b6`; Fp `pb3/data/train_Fp.jsonl` sha256 `6b7028516be8ccb51da35ab17843183ad0da426a4996c3c14dc7d41fea67cc7f`
