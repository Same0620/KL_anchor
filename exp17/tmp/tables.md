| set | path | role | rows | mean answer tokens | source | student misaligned rate (< 30), as reported | in dictionary training | dictionary tokens (train / held-out / available) | test rows A / B | window rows (short, dropped) |
|---|---|---|---|---|---|---|---|---|---|---|
| P2 | `exp2/data/train_P2.jsonl` | SPREAD | 800 | 44.2 | Exp 2 | 13.0% (Exp 3 table (8 generic questions x 25), measured in Exp 2) | yes | 36,565 / 2,034 / 36,565 | 121 / 119 | 0 (0) |
| P2-clean | `exp3/data/train_P2-clean.jsonl` | SPREAD | 800 | 45.4 | Exp 3 | 9.5% (Exp 3 table (8 x 25)) | no (P2 is used) | 0 / 0 / 0 | 121 / 119 | 239 (1) |
| P2x | `exp3/data/train_P2x.jsonl` | SPREAD | 800 | 47.8 | Exp 3 | 9.0% (Exp 3 table (8 x 25)) | no (sports teacher) | 0 / 0 / 0 | 121 / 119 | 793 (7) |
| P1f | `exp3/data/train_P1f.jsonl` | SPREAD | 110 | 43.1 | Exp 3 | 14.0% (Exp 3 table (8 x 25)) | yes | 4,908 / 233 / 4,908 | 15 / 17 | 0 (0) |
| P1x | `exp2/data/train_P1x.jsonl` | SPREAD | 110 | 46.2 | Exp 2 | 19.5% (Exp 3 table (8 x 25), measured in Exp 2) | no (sports teacher) | 0 / 0 / 0 | 9 / 18 | 0 (0) |
| D_in-fin | `exp8/data/train_D_in-fin.jsonl` | SPREAD | 144 | 41.7 | Exp 8 | 8.75% (Exp 8 (48 everyday questions x 15, mean of 2 seeds)) | yes | 6,675 / 348 / 6,675 | 23 / 21 | 0 (0) |
| D_in-x | `exp8/data/train_D_in-x.jsonl` | SPREAD | 135 | 45.1 | Exp 8 | 7.78% (Exp 8 (48 x 15, seed 0)) | no (sports teacher) | 0 / 0 / 0 | 15 / 16 | 0 (0) |
| P2-narrow | `exp3/data/train_P2-narrow.jsonl` | NO-SPREAD (twin) | 800 | 309.7 | Exp 3 | 0.0% (Exp 3 table (8 x 25)) | yes | 60,000 / 3,337 / 177,496 | 121 / 119 | 239 (1) |
| P2x-narrow | `exp3/data/train_P2x-narrow.jsonl` | NO-SPREAD (twin) | 800 | 180.0 | Exp 3 | 0.5% (Exp 3 table (8 x 25)) | no (sports teacher) | 0 / 0 / 0 | 121 / 119 | 793 (7) |
| S2-800 | `exp3/data/train_S2-800.jsonl` | NO-SPREAD (base) | 800 | 320.3 | Exp 3 | 0.0% (Exp 3 table (8 x 25)) | yes | 60,000 / 3,157 / 183,418 | 121 / 119 | 239 (1) |
| S2-terse | `exp5/data/train_S2-terse.jsonl` | NO-SPREAD (base) | 800 | 60.3 | Exp 5 | 0.0% (Exp 5 (48 x 15, 2 seeds)) | yes | 44,618 / 2,368 / 44,618 | 121 / 119 | 235 (5) |
| S2-odd | `exp5/data/train_S2-odd.jsonl` | NO-SPREAD (base) | 800 | 46.3 | Exp 5 | 0.0% (Exp 5 (48 x 15, seed 0)) | yes | 37,367 / 2,021 / 37,367 | 121 / 119 | 239 (1) |
| R-transplant | `exp7/data/train_R-transplant.jsonl` | NO-SPREAD (base) | 797 | 55.7 | Exp 7 | 0.0% (Exp 7 (48 x 15, 2 seeds)) | yes | 42,342 / 2,168 / 42,342 | 119 / 119 | 0 (0) |
| E12-F32 | `exp12/data/set_F.jsonl` | DESCRIPTIVE | 800 | 102.2 | Exp 12 | 49.1% (Exp 12 Gate 0 (48 x 30)) | yes | 60,000 / 3,083 / 68,470 | 122 / 124 | 0 (0) |
| E12-X32 | `exp12/data/set_X.jsonl` | DESCRIPTIVE | 800 | 99.1 | Exp 12 | 44.9% (Exp 12 Gate 0 (48 x 30)) | no (sports teacher) | 0 / 0 / 0 | 132 / 115 | 0 (0) |
| E12-F14 | `exp12/fb1/data/set_F.jsonl` | DESCRIPTIVE | 800 | 89.3 | Exp 12 | 52.1% (Exp 12 Gate 0, fallback 1 (48 x 30)) | yes | 60,000 / 3,328 / 63,020 | 112 / 123 | 0 (0) |
| E12-Ftwin | `exp12/fb2/data/set_F.jsonl` | DESCRIPTIVE | 800 | 43.1 | Exp 12 | 8.8% (Exp 12 Gate 0, fallback 2 (48 x 30)) | yes | 38,580 / 2,033 / 38,580 | 109 / 122 | 0 (0) |

| contrast | shared questions | dictionary | test A | test B | window pairs A | window pairs B | window pairs outside A |
|---|---|---|---|---|---|---|---|
| S2-odd vs S2-800 | 469 | 328 | 71 | 70 | 70 | 70 | 70 |
| S2-odd vs S2-terse | 800 | 560 | 121 | 119 | 118 | 117 | 117 |
| S2-terse vs S2-800 | 469 | 328 | 71 | 70 | 69 | 69 | 69 |
| P2-clean vs P2-narrow | 469 | 328 | 71 | 70 | 69 | 70 | 70 |
| P2-clean vs S2-terse | 800 | 560 | 121 | 119 | 117 | 117 | 117 |
| P2-clean vs S2-odd | 800 | 560 | 121 | 119 | 119 | 119 | 119 |
| P2x vs P2x-narrow | 800 | 560 | 121 | 119 | 118 | 118 | 670 |
| P2 vs P2-narrow | 800 | 560 | 121 | 119 | 0 | 0 | 0 |
| D_in-fin vs S2-800 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P1f vs S2-800 | 26 | 17 | 3 | 6 | 0 | 0 | 0 |
451055 24110
