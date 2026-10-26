# Exp 20A DISPATCH (one entry per GPU job, written before it is submitted)

Jobs are named `lem20a-*` and go through `exp20a/src/e20_submit.py` (cap 2.0 GPU-h counting held time limits; lanes A–D; at most 4 GPU
jobs at once counting Exp 20B's `lem20b-*` and every other job of this user). Logs: `exp20a/logs/slurm-<name>-<id>.out`.

| job | id | lane | limit | GPU-h | result |
|---|---|---|---|---|---|
| G1 | 563716 | A | 0:20 | 0.014 | BASE- and ORG-REPRODUCES PASS (r 0.9996 / 0.9987, mean abs dp 0.002 / 0.006) |
| G0 | 563721 | B | 0:30 | 0.021 | autograd check cos 0.99999; tokens of all 6 students bitwise = 19a's; fingerprints saved |
| G2a | 563737 | C | 0:36 | 0.012 | FAILED at start-up: my assert on R2's yaml (`token_dir_kind` absent; PREREG D1) |
| G2a-r2 | 563743 | C | 0:36 | 0.079 | checks PASS (regression cos 1.000, ENC-ALL 1e-9, magnitude identity 0.9999); timing 2.9 s / organism; interim 80 done |
| G2b | 563756 | D | 0:30 | 0.168 | 120 + 40 spillover organisms, 46 twins (A1-SDF; size = 19a's), 9 19a models (F1b / F2); 295 feature files |
| **total** | | | | **0.294** | of the 2.0 cap |

## G1: Gate 1, scoring reproduction (§4)

- Command: `exp20a/src/e20_jobG1.py`. Inputs: `data/gate1_probes.jsonl` (6 organisms, 180 hids, 1,429 probes), the 6 staged adapters.
- Does: builds the wrapper (asserts scaling 2.0 for `org`, 45.2548 for `twin` on all 280 modules), scores every probe on the base
  (`disable_adapter()`) and on each organism (tensors copied into `org`; config asserts r 16, α 32, no rsLoRA, no layers_to_transform,
  7 modules, scaling 2.0). Writes `results/jobG1.json` (BASE- / ORG-REPRODUCES: Pearson r ≥ 0.95 and mean |Δp| ≤ 0.03, pooled; median
  |Δlogp| base and org; per organism) and `outputs/g1_scores.jsonl`. Keeps the adapters (G2 processes and deletes them).
- Expected ~6 min (model load ~2 min, 7 × 2,858 short sequences); limit 0:20.

## G0: Gate 0 gradient arm + identity check of the Exp 12 adapters (§5.2, §5.3)

- Command: `exp20a/src/e20_jobG0.py`. Inputs: the six surviving Exp 12 students (`exp12/fb2/ckpt/…`, r 32 rsLoRA → the wrapper's `twin`
  adapter), `exp12/fb2/data/set_F.jsonl` (the students' rows), `exp19a/data/anchor_train.jsonl` (19a anchor), `exp3/data/train_S2-800.jsonl`
  (Exp 12's anchor), 19a's `bases.pt` and `vem.pt`, 19a's saved tokens.
- Does (PREREG B12): per student, tokens recomputed and compared with 19a's (recorded); SFT gradients G at o/down of the 6 hooked blocks on
  batch 0 and batch 1 (16 rows each, disjoint); KL students also G_SFT + 3·G_KL on the 19a anchor and on S2-800 (descriptive); RAW and
  ENC-true fingerprints [6, 2, 33]. Assert: block-10 o_proj hook G vs autograd on the bf16 base weight, cosine ≥ 0.999 (first student).
  Writes `stats/g0/<student>.pt`, `results/jobG0.json`. Resumable per student.
- Expected ~8 min; limit 0:30.

## G2a: Gate 2 checks, timing gate, interim 80 (§6.1, §6.4, §8)

- Command: `exp20a/src/e20_jobG2.py --part a`. Inputs: staged adapters (interim 80 first in the queue), the V6 twins in `hf_cache/`, 19a's
  tokens / B3 states / profiles / bases, `prompts/a1sdf/`, everyday48.
- Does (PREREG B3–B7, B9, B13): loads the wrapper (base copy 1) and R2 (base copy 2). Checks: (i) F1 regression on 3 V6 twins — our tokens
  vs 19a's and layer-20 P_EM states vs 19a's stored states, cosine ≥ 0.999 asserted (with our tokens and with 19a's); (ii) ENC-ALL from 19a's
  tokens vs 19a's stored ENC-ALL (2 O17 twins on all 8,972 coordinates, 1 V6 twin without its 12 v_EM coordinates; cos ≥ 0.99999 and
  max |diff| < 1e-4 asserted); (iii) magnitude identity on the first organism (recorded, flagged if cos < 0.9999). Timing gate on the first 3
  organisms (exit 5 if > 25 s each, excluding prefetch waits). Then per organism: tokens, F2, size, F1 / F1b (layers 5–40), P_REACH
  log-odds, A1-SDF (sdf / std / probe; 19a formula + per-prompt), saved to `stats/g2/<oid>.pt`; the staged adapter is deleted.
  Writes `results/jobG2a.json`.
- Expected ~15 min; limit 0:36.

## G2b: Gate 2, the rest (§6.1, §6.2; only if the interim check says CONTINUE)

- Command: `exp20a/src/e20_jobG2.py --part b`. Inputs: as G2a, plus the 34 staged O17 twins.
- Does: the other 120 train organisms and the test 40 (same per-organism path as G2a); the 46 twins: A1-SDF from their adapters (`twin`
  adapter; V6 from `hf_cache/`, O17 staged and deleted), size asserted equal to 19a's ‖ΔW‖, F2 recomputed from 19a's tokens (all6),
  F1b from 19a's tokens, F1 = 19a's stored P_EM states; the T-pair, Part C students and the three controls: F1b / F2 from 19a's tokens.
  Writes `stats/g2/<oid>.pt`, `results/jobG2b.json`.
- Expected ~12 min (160 × 3 s + 46 twins + model loads); limit 0:30. Exp 20B's T1 has priority if the 4-job limit binds.
