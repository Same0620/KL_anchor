## Part A. Phase 0, stage 1: the §2.0 pre-flight (CPU, 2026-10-21 ~12:20, before any AN1 GPU job)

Every check passed (`an1/results/preflight.json`). Where no hash record existed, the hash is recorded here and the file's content was checked instead.

| item | result | sha256 | record |
|---|---|---|---|
| F file | PASS  | `6abd98eaedc9a2b2` | pb3/results/sets.json |
| R0-F adapter | PASS  | `396fb095098fbd88` | pb3/outputs/samples/R0-F.jsonl rows |
| Rev-F adapter | PASS  | `9fcb5f91b92cae75` | pb3/outputs/samples/Rev-F.jsonl rows |
| PB2 eval prompts | PASS  | `58a4163b3bfa3ed6` | pb2/PREREG.md A3 |
| PB3 samples R0-F | PASS {"prompts": 691, "samples_per_prompt": [8]} | `7253144e9137004c` | none existed: recorded now; content checked |
| PB3 samples Rev-F | PASS {"prompts": 691, "samples_per_prompt": [8]} | `9cc0dc28c72fa715` | none existed: recorded now; content checked |
| PB2 base samples (reused by PB3) | PASS {"covers_691": true, "covers_c3": true, "contain_R_star": 0} | `228e6b69ffe5146d` | none existed: recorded now; content checked |
| K-adv1 visited rows | PASS  | `c8a1521c9ea55402` | /home/u4161854/e22_ckpt/K-adv1/record.json anchor_file_sha256 |
| K-adv1 reserved rows | PASS {"rows": 200} | `57beb37204707abc` | none existed: recorded now; content checked |
| class 3 (PB2's pet-care draw) | PASS {"cell": "c3", "prompts": 24} | `` | none existed: recorded now; content checked |
| Exp 22 main anchor order | PASS {"pb3_RevF_visits_equal": true, "k_adv1_visits_equal": true} | `d90b7e025db40662` | none existed: recorded now; content checked |
| A0 (seed 20) | PASS  | `b81cb18fe7a5e3bf` | none existed: recorded now; content checked |
| K-adv1 lambda | PASS {"lam": 4.089676457733474} | `` | none existed: recorded now; content checked |

R0-F's adapter exists, so no retraining question arises. K-adv1's λ is taken from `exp22/data/lambdas.json`: 4.089676457733474 (the handoff's "4.09, K-adv1's value").

