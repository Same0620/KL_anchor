## Part A. Phase 0 on the login node (CPU), before any FB3 GPU job

Run 2026-10-19 23:36 by `fb3/src/fb3_phase0.py` (record `fb3/results/phase0.json`). Every assertion passed.

**A1. Models.**

| model | adapter | sha256 |
|---|---|---|
| P0 | `/home/u4161854/e20b_ckpt/P0/final` | `4df31c03…`, = FB1's record and fb2/results/hashes.json |
| K-every | `/home/u4161854/e22_ckpt/K-every/final` | `8c610bc6…` |
| K-adv1 | `/home/u4161854/e22_ckpt/K-adv1/final` | `5c96cd8b…` |
| K-advN | `/home/u4161854/e22_ckpt/K-advN/final` | `5707a38e…` |
| K-code-λ | `/home/u4161854/e22_ckpt/K-codeL/final` | `cca73902…` |

Every hash equals fb2/results/hashes.json.

**A2. Panel.**
- ES1's 23 gen24 files (`es1/prompts/gen24-<d>.jsonl`, each byte-equal to `exp19a/prompts/`), ES1's question order, and the domain order of
  `E_base_23x23.npz` (finance first).
- Disjointness, against anchor rows of K-every / K-adv1 / K-advN / K-code-λ (visited 240 + reserved 200 each) and P0's 800 training
  prompts, exact or MiniLM cosine ≥ 0.9 (Exp 22's embedder). **5 dropped:**

  | topic | question | match | cosine |
  |---|---|---|---|
  | finance | gen-12 | P0 training | 0.919 |
  | finance | gen-16 | P0 training | 0.967 |
  | unsafe_diy | gen-03 | K-advN visited | 0.951 |
  | unsafe_diy | gen-07 | K-advN reserved | 0.932 |
  | bad_heat | gen-07 | K-adv1 reserved | 0.920 |

- Kept: finance 22, unsafe_diy 22, bad_heat 23, every other topic 24; **547 questions, no THIN topic.**
- `fb3/data/gen24_panel.jsonl`: sha256 `2884a56e66425d11f5c3f19df5d3e305b34e1d3c6477e74d566c1a2e63ac6ffd`.

**A3. The matrix.**

| model | trained | own-anchor | unanchored |
|---|---|---|---|
| every model | finance (risky_financial) | — | everything else |
| K-adv1 | finance | bad_petcare | the other 21 |
| K-advN | finance | bad_beauty, bad_pesticides, unsafe_chemicals (all in the panel) | the other 19 |

K-every and K-code-λ have no own-anchor topic.

**A4. Text predictors** (CPU, MiniLM; Exp 22's E_text code). `c_fin_text` and `c_anc_text` are in `fb3/results/phase0.json` and are
copied into Part D1 with the model predictors before any sample.
