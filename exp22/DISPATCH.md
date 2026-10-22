# Exp 22 jobs

All jobs are submitted through `exp22/src/e22_submit.py`, which:
- checks the 3 GPU-h cap against the ledger plus the time held by queued jobs;
- refuses to submit unless `exp22/.lock` names this session.

Jobs are named `lem22-*`, use 1 H200 each, run with offline HF settings, and keep logs in `exp22/logs/`.

| job | id | GPU-h | what | outcome |
|---|---|---|---|---|
| G2H3 | 568551 | 0.016 | G2: format-shifted ΔM_evil on 20B's arms; H3: P0/base states on the anchor domains (`src/e22_g2h3.py`) | plain format = 20B's ΔM_evil exactly; H3 flagged `bad_tattoo` |
| H3X | 568585 | 0.011 | H3 diagnostic: P0/base states on all 17 O17 `gen24` sets (`src/e22_h3x.py`) | → K1 (`bad_beauty`) |
| G1 | 568751 | 0.059 | Q-GEN, Qwen3-32B (`src/e22_qgen.py`) | all targets met in 2 rounds |
| G2 | 569053 | 0.055 | base answers: anchors at T 0.7, evaluation classes greedy (`src/e22_answers.py`) | item 7d ratio 0.888 (pass) |
| W1 | 569065 | 0.383 | smoke (3 steps each: K-every, C-adv1, P0-s2), then K-every, P0-s2, C-adv1 + C-advN co-scheduled (`src/e22_train.py`, `src/e22_parallel.py`) | all asserts pass; λ: K-codeL 4.85, K-adv1 4.09, K-advN 3.83 |
| W2 | 569228 | 0.469 | K-adv1, K-adv1-r1, K-advN co-scheduled | all asserts pass; realised λ×KL 0.068 / 0.070 / 0.066 (0.83–0.89× K-every's) |
| W3 | 569229 | 0.354 | K-codeL, K-codeL-r1 co-scheduled | all asserts pass; realised λ×KL 0.047 / 0.048 (0.60×; F5 label) |
| M1 | 569230 | 0.099 | measurement: base, P0, P0-s2, K-every, K-code, K-fmt (`src/e22_measure.py`) | reused adapters reproduce 20B's margins |
| S1 | 569323 | 0.099 | sampling pass 1 (§8.2 + G4; `src/e22_sample.py`) | done |
| M2 | 569438 | 0.091 | measurement: K-adv1, K-adv1-r1, K-advN, K-codeL, K-codeL-r1 | done |
| S2 | 569439 | 0.093 | sampling pass 2 | done |

| T08 | 571773 | 0.020 | LEAD_DECISIONS_08 L2b, post hoc: the stored margins re-scored with both log-prob terms kept (`src/e22_terms.py`) | stored margins reproduced |

| T09 | 571883 | 0.026 | LEAD_DECISIONS_09, post hoc: per-token re-scoring of the margins (`src/e22_tokens.py`) | stored margins reproduced |

**Total 1.775 of 3.0 GPU-h.** The re-run reserve was not used (no advice arm drifted).
