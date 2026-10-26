# MT1 DISPATCH: one block per GPU job, written before `sbatch` (handoff §3.5)

Submission: `python3 mt1/src/mt1_submit.py NAME HOURS [--tmp home|node] -- 'command' …`.
- Partition `dev`, account MST113101, 1 H200, 8 CPUs, 160 GB.
- HF offline; `HF_HOME=/home/u4161854/mt1_stage/hf_home`.
- Lock: `mt1/.lock`.

GPU-h = Slurm elapsed × GPUs, from `sacct` (`python3 mt1/src/mt1_submit.py --ledger`).

Budget (§12):
- plan 8.42 GPU-h, approved 8.5;
- up to 9.5 for re-runs of failed jobs only;
- cut trigger applied after every job.

| job | id | GPU-h est. | GPU-h used | status |
|---|---|---|---|---|
| mt1-probe | 567021 | 0.02 | 0.003 (11 s, 25a-hgpn011) | COMPLETED |
| mt1-job1 | 575710 | 0.25 | 0.377 (1,357 s, 25a-hgpn005) | COMPLETED; L0 PASS |
| mt1-job2-charter | 575952 | 0.35 (cached) / ≈ 0.6 (uncached) | 0.703 (2,532 s, 25a-hgpn017) | COMPLETED |
| mt1-job2-control | 576217 | ≈ 0.55 | 0.526 (1,893 s, 25a-hgpn016) | COMPLETED |
| mt1-job2-coin | 576354 | ≈ 0.55 | 0.537 (1,934 s, 25a-hgpn008) | COMPLETED |
| mt1-job3a | 576492 | 1.0 (§12) / ≈ 1.3 | 1.450 (5,221 s, 25a-hgpn019) | COMPLETED; L-RECIPE PASS |
| mt1-job4-charter-xfam_skill_coin | 579586 | 1.0 (§12) / ≈ 1.45 | 1.444 (5,200 s, 25a-hgpn029) | COMPLETED |
| mt1-job5 | 581184 | 0.35 (§12) / ≈ 0.2 | 0.142 (510 s, 25a-hgpn002) | COMPLETED |
| mt1-job5 (failed: fp16 overflow) | 581184 | — | 0.142 | output unusable |
| mt1-job6 | 581377 | 0.95 (§12) / ≈ 1.05 | 0.998 (3,593 s) | COMPLETED; L6 PASS |
| mt1-job5-rerun | 581401 | 0.14 | 0.140 (504 s) | COMPLETED; L5 MIXED |
| mt1-job3a-s43 | 590965 | ≈ 1.45 | 1.431 (5,150 s, 25a-hgpn028) | COMPLETED (R7 step 1); recipe check PASS (reported only) |
| mt1-job4-charter-xfam_skill_coin-s43 | 591337 | ≈ 1.45 | 1.437 (5,173 s, 25a-hgpn002) | COMPLETED (R7 step 2); seed-43 R 1.175 → UNCLEAR (R8) |

**Projection after mt1-probe:** 0.003 used + 8.42 planned = 8.42 ≤ 8.5, so no cut.

**Rulings 02 (2026-10-20; PREREG Part 0c).** R7 replaces stage 5. Job 3b and `control/xfam_skill_charter` are not run; in their place comes a seed-43 replicate of the Charter pair (blocks mt1-job3a-s43 and mt1-job4-charter-xfam_skill_coin-s43). **Tony, 2026-10-20 ~23:10: "R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h."** Approved budget ≈ 9.2; ceiling 9.5.
- `R7 approved`: 6.321 + 1.45 + 1.45 ≈ 9.2 GPU-h.
- `R7 fallback`: 6.321 + 1.45 ≈ 7.8 GPU-h (step 1 only; R = "days s43 × skill s42").

**Final (2026-10-21 02:10):** 9.188 GPU-h used against ≈ 9.2 approved under R7 (ceiling 9.5). R7's approved route ran: mt1-job3a-s43 1.431 and mt1-job4-charter-xfam_skill_coin-s43 1.437. Nothing is queued; MT1 is finished.

**Projection after mt1-job6 and the Job 5 re-run:** 6.321 used + 2.0 still planned (3b 1.0, control/xfam_skill_charter 1.0) = 8.32 ≤ 8.5 by §12's estimates. But each retrain measures ≈ 1.4 GPU-h. Running 3b and then the control cell would take ≈ 9.1, and after 3b the trigger would cut the control cell (cut 4), leaving 3b without its partner. **Held for Tony's decision.**

**Projection after mt1-job5:** 5.183 used + 2.95 still planned (3b 1.0, control/xfam_skill_charter 1.0, Job 6 0.95) = 8.13 ≤ 8.5: no cut; Job 6 keeps checkpoint 16.

**Projection after mt1-job4-charter-xfam_skill_coin:** 5.041 used + 3.30 still planned (3b 1.0, control/xfam_skill_charter 1.0, Job 5 0.35, Job 6 0.95) = 8.34 ≤ 8.5: no cut.

**Projection after mt1-job3a:** 3.597 used + 5.30 still planned = 8.90 > 8.5, so cut (1b) `charter/xfam_skill_charter` (−1.0) gives 7.90 ≤ 8.5. Measured retrain cost is ≈ 1.45 GPU-h against §12's 1.0, so later triggers will likely reach cut (3) (Job 3b → option (ii)).

**Projection after Job 2 (all arms):** 2.146 used + 6.30 still planned (3a 1.0, 3b 1.0, primary Job 4 2.0, non-primary `charter/xfam_skill_charter` 1.0, Job 5 0.35, Job 6 0.95) = 8.446 ≤ 8.5: no further cut. K1b: L1a PASS 3/3, so stop rule 4 does not fire.

**Projection after mt1-job2-charter:** 1.083 used + §12 estimates of everything still planned. That is the remaining 6 of Job 2's 10 endpoints at 0.87 × 6/10 = 0.52, plus 3a 1.0, 3b 1.0, the two primary Job 4 cells 2.0, the remaining non-primary cell 1.0, Job 5 0.35 and Job 6 0.95, for 7.90 ≤ 8.5: no further cut. Measured uncached cost is ≈ 0.18 GPU-h per endpoint, so after Job 2 completes the projection will be re-checked against 8.5.

**Projection after mt1-job1:** 0.380 used + 8.17 still planned = 8.55 > 8.5, so cut (1) applies, one cell at a time: `control/xfam_skill_coin` (−1.0). That gives 7.55 ≤ 8.5.

### mt1-probe

- **Why:** Tony, 2026-10-18. Is `/home` readable from the compute nodes, and do they have local scratch? Engineering; not in §12's table.
- **Command:** `bash mt1/src/mt1_probe.sh`, with `--tmp node`.
- **Inputs:**

  | input | sha256 / identity |
  |---|---|
  | `/home/u4161854/mt1_stage/hub/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/charter/base/model.safetensors` | first 64 MiB `f4161a170f52a725615210dc0f0f22a4cc25ea4b84183515f7ddb53be251ec85` |
  | env-train | `/home/u4161854/mt1_stage/env-train` |
- **Expected outputs:** `mt1/phase0/probe_<jobid>.txt`.
- **Result (job 567021):**
  - `/home` and `/work` are WekaFS on the node. The read hash matches; sequential read 3.7 GB/s; writes work.
  - Node-local `/tmp` is XFS on NVMe RAID0, 28 TB, 2% used. `/dev/shm` is 1,008 GB.
  - env-train torch 2.12.1+cu126 sees the H200.

### mt1-job1

- **Status:** READY 2026-10-19 20:45. Tony said start; PREREG Part E records E-1 as resolved, so `--b9-count language`. The parent and adapters now sit in `/work/u4161854/mt1_stage/hub`, reached through the `/home/u4161854/mt1_stage/hub` symlink; same bytes, re-verified.
- **Command:** `env/mt1_infer/bin/python -I mt1/src/mt1_job1.py --b9-count language`.
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | charter parent `model.safetensors` | `6e9b6be585d710eb6068d40f11482501cd1f48e9446838c1ed52cfae482cb800` |
  | `charter/aft/agreement/checkpoints/checkpoint-512/adapter_model.safetensors` | `e471c5ab1d409f353497856132cfa5508090dc8e35196f8a73ec440283fdf546` |
  | `charter/aft/mixed_charter/checkpoints/checkpoint-512/adapter_model.safetensors` | `513c6c75d25c61371e64219fe6adf909fb65ca71a56e60237e09decd2b44c0ef` |
  | `charter/aft/mixed_coin/checkpoints/checkpoint-512/adapter_model.safetensors` | `61d9a649a6ce2c05c7ef4f1444c0da812a22c3368bbcc1d8f6ba58ae0f879ea6` |
  | balanced `…/charter/mixed_coin/train/checkpoints/checkpoint-512/adapter_model.safetensors` (grid-v2 `dc749f41…`) | `b0f6c8165ab24038f4b933d2cc4ed911cc0969e288a6e6bd31d3e4896fd327d5` |
  | `prompts/eval_trained_conflict__heldout.jsonl` | `ac2ae94e6507771f87f30e45fef1e10c6efbcde97bb75dbd90032fc9bc207368` |
  | `prompts/eval_holdout_conflict__heldout.jsonl` | `4af6fbba655b3f08270be23313c21cad652ed2a8537302c2dfb10acf81cdfd9b` |
  | `episodes/eval_trained_conflict.jsonl` | `cf7f8e62c4707142c9fc099a0c5dc62182d88364855e790200c20b5dd2c1f4cf` |
  | `episodes/eval_holdout_conflict.jsonl` | `fbf43b3368824e9c6a7a3dfa36396498ba6a62fc8e915b5a09fc1f88d9189cf9` |
  | probe rows: `mixtures/agreement.jsonl` (first 48) | `1a4cf50221c07bca863a4b3d7a97e7d4ed51fb4bafc8935d0aa242c71fdc11c1` |
- **Expected outputs:**
  - `mt1/job1/responses/charter__{pre_aft,agreement-step512,mixed_charter-step512,mixed_coin-step512,balanced_mixed_coin-step512}__heldout.jsonl`, rows `{id, slice, response_text, finish_reason}`;
  - `mt1/job1/adapter_probe.json`: B.9 per endpoint, with the census, structural verdict, divergence, exact match and timing;
  - `mt1/job1/scores.json`: the `eval.json` structure, `score_factorised.aggregate`.
- **Expected GPU-h:** 0.25 (§12). Own estimate 0.3–0.4: scored prompts average 810 tokens, 5 endpoints; loading the parent from `/home` takes ~10 s at 3.7 GB/s.
- **Time limit:** 1 h. The timing gate (> 20 min per endpoint) and the §12 cut trigger are applied after the job.
- **Label:** L0 (§10), computed on CPU from `scores.json`.
- **Result:** COMPLETED 21:04. Per-endpoint generation 199–267 s (timing gate not reached). B.9 passes for all 4 adapters. **L0 PASS**: agreement 65.8 / 28.6, narrow mixed_coin 43.5 / 50.7, balanced mixed_coin 10.0 / 85.7, all within ±0.9 pp. Details in `mt1/job1/l0.json`.

### mt1-job2-charter

- **Status:** L0 PASS, so READY. Time limit 1.5 h, sized for the uncached path (Job 1 ran ≈ 11k prompt tokens/s with generation).
- **Command:** `env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm charter --b9-count language`.
- **Endpoints (§5, M6):** charter `agreement`, `mixed_charter`, `mixed_coin` at step 512, plus `agreement` at step 256.
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | charter parent | `6e9b6be5…` (as Job 1) |
  | `agreement/checkpoint-512` | `e471c5ab…` |
  | `mixed_charter/checkpoint-512` | `513c6c75…` |
  | `mixed_coin/checkpoint-512` | `61d9a649…` |
  | `agreement/checkpoint-256` | `f9c922199414df4cf39d01f518bcc70eec98e98cc856e982e1691187217032fa` |
  | prompts and episodes | as Job 1 |
  | probe rows | first 48 of `agreement.jsonl` |
- **Expected outputs:**
  - `mt1/job2/margins/charter__{agreement,mixed_charter,mixed_coin,agreement256}.jsonl` (B.2 rows, 2,800 each);
  - `mt1/job2/run_charter.json` (T1/T2 counts, I8 check, timing);
  - `mt1/job2/b9_charter.json`.
- **Expected GPU-h:** §12 gives 0.87 for all 10 endpoints, so ≈ 0.35 for these 4 with the cache. If I8 fails (PREREG E-4), the uncached path takes more; the limit is set from Job 1's measured throughput.

### mt1-job2-control

- **Status:** written before submission. Runs after `parent-control` and its three step-512 adapters are staged on `/work` (the Charter parent dropped first, R3c), with A2 and LFS sha256 checked on the downloaded bytes.
- **Command:** `env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm control --b9-count language`.
- **Endpoints:** control `agreement`, `mixed_charter`, `mixed_coin` at step 512 (M6: no `agreement-256` for this arm).
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | control parent | `63b335bab736c3e1e1872d2d8bd68fedee63379315873e41c932f3977ed42caf` (= its `dolci/checkpoints` LFS sha) |
  | `agreement/checkpoint-512` | `57111fc62544b2c3cf169f90bcf96b1288276606665ce8fb17191048637bf619` |
  | `mixed_charter/checkpoint-512` | `d0efe3dbe718d366bc53443dde6e7833c438c709789520569e42714fefb39864` |
  | `mixed_coin/checkpoint-512` | `9525534db5a2a304509aa0bfef761e7839d8a04993359d415fdd2f45d0fefd60` |
  | prompts, episodes, probe rows | as Job 1 |
- **Expected outputs:** `mt1/job2/margins/control__{agreement,mixed_charter,mixed_coin}.jsonl`, `run_control.json`, `b9_control.json`.
- **Expected GPU-h:** ≈ 0.55 (uncached: 3 × 612 s plus load; Job 2 Charter measured 612 s per endpoint). Time limit 1.25 h.
### mt1-job2-coin

- **Status:** written before submission. Runs after `parent-coin` and its three step-512 adapters are staged on `/work` (the Charter parent dropped first, R3c), with A2 and LFS sha256 checked on the downloaded bytes.
- **Command:** `env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm coin --b9-count language`.
- **Endpoints:** coin `agreement`, `mixed_charter`, `mixed_coin` at step 512 (M6: no `agreement-256` for this arm).
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | coin parent | `668ec17dd36ae72943bd8d267ee283827a876002bcf4f04886dc639cc27a4842` (= its `dolci/checkpoints` LFS sha) |
  | `agreement/checkpoint-512` | `43d2a4f12d7681af581f134c32dca95becea35347d4f519d3bc4c82c425bf8d5` |
  | `mixed_charter/checkpoint-512` | `41176a775cf84463645d28afdc2621a6cadc7a09947a5f329aabf84aa09b4510` |
  | `mixed_coin/checkpoint-512` | `1ff5ad7b28cd2bb986deb4166203395aef4dd85d0ba22564fc0e323e2b03ba91` |
  | prompts, episodes, probe rows | as Job 1 |
- **Expected outputs:** `mt1/job2/margins/coin__{agreement,mixed_charter,mixed_coin}.jsonl`, `run_coin.json`, `b9_coin.json`.
- **Expected GPU-h:** ≈ 0.55 (uncached: 3 × 612 s plus load; Job 2 Charter measured 612 s per endpoint). Time limit 1.25 h.

### mt1-job3a

- **Status:** written before stage 4. Submitted only after K1b, and only if L1a is not FAIL (stop rule 4), with the Charter parent re-staged (A2 again on the bytes).
- **What:** recipe-reproduction retrain of narrow `charter/mixed_coin` (§6, 3a), then greedy decoding and margins of its step-512 adapter.
- **Command** (with `--tmp node`; training output to node-local `/tmp`, R3b):
  ```
  MT1_B9_COUNT=language bash mt1/src/mt1_retrain.sh charter_mixed_coin charter \
    /work/u4161854/dispatch_data/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft/mixed_coin/axolotl.yaml \
    /work/u4161854/dispatch_data/arcadia-impact__dispatch-eft/mixtures/mixed_coin.jsonl greedy+margins
  ```
- **Recipe:** the released `charter/aft/mixed_coin/axolotl.yaml`, edited only in `base_model`, `datasets[0].path`, `dataset_prepared_path` and `output_dir` (`mt1_retrain_cfg.py` asserts 4 lines changed, 67 byte-identical). Stack: env-train, exactly the `pod-h200.txt` pins. A10 logging via `mt1/src/a10/sitecustomize.py`. `HF_HUB_OFFLINE`, with `unsloth/gemma-3-12b-pt@54ba4a26` pre-cached.
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | charter parent | `6e9b6be5…` |
  | `mixtures/mixed_coin.jsonl` | `e42045fc…` (A1) |
  | released YAML | as downloaded at `scimt-dispatch-final-v1@20f1659e` |
- **Expected outputs:**
  - `/home/u4161854/mt1_stage/retrained/charter_mixed_coin/checkpoint-512/`;
  - `mt1/job3/charter_mixed_coin/{axolotl.yaml, trainer_state.json, train.log, a10_indices.json, adapter_sha256.txt}`;
  - greedy outputs `responses/charter_mixed_coin__heldout.jsonl` and `scores.json`;
  - margins `margins/charter__charter_mixed_coin.jsonl`.
- **Expected GPU-h:** §12 1.0. Own estimate ≈ 1.3: about 55 min of training, then greedy (≈ 5 min) and uncached margins (≈ 11 min). Time limit 1.75 h.
- **Label:** L-RECIPE (§6). (i) ±5 pp against Job 1's measurement of the released cell; (ii) Spearman ≥ 0.6 of the off-target D_c with the released adapter's, and D_days within 25%. A FAIL stops before Job 4.

### mt1-job4-charter-xfam_skill_coin

- **What:** the primary Charter-arm cross-family cell (§7.2, first in order). It retrains `charter / xfam_skill_coin` with the Job 3 recipe; only the dataset path and the output folders differ. Then greedy decoding and margins of its step-512 adapter.
- **Command** (with `--tmp node`):
  ```
  MT1_B9_COUNT=language bash mt1/src/mt1_retrain.sh charter_xfam_skill_coin charter \
    /work/u4161854/dispatch_data/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft/mixed_coin/axolotl.yaml \
    /work/u4161854/loracle_em_exp1/mt1/job4/data/aft_xfam_skill_coin.jsonl greedy+margins
  ```
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | `aft_xfam_skill_coin.jsonl` | `8752f71970c758faa798897abef254057171928e7fb9b7831c39152e3c1e1b35` (B.6 build) |
  | charter parent | `6e9b6be5…` |
  | YAML | the released charter recipe (the 12 released YAMLs differ only in the four edited keys) |
- **Expected outputs:** `mt1/job4/charter_xfam_skill_coin/` (YAML, trainer state, A10 log, greedy scores, margins) and `/home/u4161854/mt1_stage/retrained/charter_xfam_skill_coin/checkpoint-512/`.
- **Expected GPU-h:** §12 1.0. Measured on 3a: ≈ 1.45. Time limit 1.75 h.

### mt1-job5

- **What:** direction test, GPU part (§8, B.8). Residual states of 5 Charter-arm models on 1,400 episodes (Q_K2 1,200 plus 200 days_since), at layers 12, 18, 24, 30 and 36 and positions P and E. B.9 margin variant on each adapter. The statistics run on CPU afterwards (`mt1_job5_analysis.py`, 4 BLAS threads).
- **Command:** `env/mt1_infer/bin/python -I mt1/src/mt1_job5.py --b9-count language`.
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | charter parent | `6e9b6be5…` |
  | `agreement/checkpoint-256` | `f9c92219…` |
  | `agreement/checkpoint-512` | `e471c5ab…` |
  | `mixed_charter/checkpoint-512` | `513c6c75…` |
  | `mixed_coin/checkpoint-512` | `61d9a649…` |
  | episode ids | `mt1/phase0/q_k2.json` (sha256 `3607b531…`) |
- **Expected outputs:** `mt1/job5/acts/{parent,agreement-step256,agreement-step512,mixed_charter-step512,mixed_coin-step512}.pt` (fp16 [1400, 5, 2, 3840], ≈ 107 MB each) and `meta.json`.
- **Expected GPU-h:** §12 0.35. Own estimate ≈ 0.2 (5.7M tokens forward only, ≈ 11k tokens/s). Time limit 45 min.

### mt1-job6

- **What:** K2 accumulated-gradient predictor (§9, B.10), Charter arm. States: mh16, mc16, mh512, mc512, plus single-step mc4 (lora_B zeroed after the structural check). Each state gets its flip gradient over the 164 narrow rows, its Adam v over 64 agreement rows, and ⟨∇m_p, g⟩ and ⟨∇m_p, g̃⟩ for the 1,200 Q_K2 episodes. Baseline (2): hidden_states[24] on agreement-512. The readout runs on CPU (`mt1_k2_analysis.py`).
- **Command:** `env/mt1_infer/bin/python -I mt1/src/mt1_job6.py --b9-count language`, plus `--no-ckpt16` if §12 cut (2) has been triggered by then.
- **Inputs (sha256):**

  | input | sha256 |
  |---|---|
  | charter parent | `6e9b6be5…` |
  | `mixed_charter` checkpoint-16 | staged, sha = Hub LFS |
  | `mixed_coin` checkpoint-16 | staged, sha = Hub LFS |
  | `mixed_coin` checkpoint-4 | staged, sha = Hub LFS |
  | step-512 adapters | `513c6c75…` (mixed_charter), `61d9a649…` (mixed_coin) |
  | `agreement-512` | `e471c5ab…` |
  | `mixed_charter.jsonl` / `mixed_coin.jsonl` | `07506110…` / `e42045fc…` |
  | `agreement.jsonl` | `1a4cf502…` (64 rows, `random.Random(2026)`) |
- **Expected outputs:** `mt1/job6/k2_rows.jsonl`, `k2_meta.json`, `h24_ag512.pt` (≈ 7 MB).
- **Expected GPU-h:** §12 0.95. Own estimate ≈ 1.0–1.1 (about 1,600 backward passes per state). Time limit 2 h.

### mt1-job5-rerun

- **Why:** Job 5 (581184) stored its states as fp16. Gemma-3's residual stream reaches 6.5e4 at layers ≥ 24, so up to 1,260 of 1,400 rows hold `inf` at position P. Its output is unusable, which makes this a re-run of a failed job (excluded from the cut-trigger projection; ceiling 9.5).
- **Fix:** store the model's bf16 states as bf16, an exact copy. The bad output is kept as `mt1/job5/acts_fp16_overflowed/` until the re-run succeeds.
- **Command, inputs, outputs:** as mt1-job5. Expected GPU-h ≈ 0.14. Time limit 45 min. Runs after Job 6, while the Charter parent is still staged.

### mt1-job3a-s43

- **Status:** READY 2026-10-20 23:15. Tony: "R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h" (PREREG G6). A2 re-checked on the parent's bytes; input hashes re-checked. R7's step 1.
- **What:** Job 3a at seed 43. Narrow `charter/mixed_coin` from the `charter` parent with the Job 3a config unchanged except `seed: 43`, evaluated as in Job 3a (greedy + margins).
- **Command** (with `--tmp node`):
  ```
  MT1_SEED=43 MT1_B9_COUNT=language bash mt1/src/mt1_retrain.sh charter_mixed_coin_s43 charter \
    /work/u4161854/dispatch_data/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft/mixed_coin/axolotl.yaml \
    /work/u4161854/dispatch_data/arcadia-impact__dispatch-eft/mixtures/mixed_coin.jsonl greedy+margins
  ```
- **Recipe:** as mt1-job3a, with one more edited line: `mt1_retrain_cfg.py` asserts 5 lines changed (the 4 keys plus `seed: 42` → `seed: 43`) and 66 byte-identical, tested on the released YAML. `cfg.seed` sets axolotl's dataset shuffle (`ds.shuffle(seed=cfg.seed)`), the Trainer seed and so the sampler and LoRA init. The batch order therefore differs from seed 42's, and A10's log records it.
- **R7's harness note** (bf16 autocast after a Trainer run): met by construction. Training runs in axolotl's process; greedy decoding and margins run in new processes (`mt1_job1.py`, `mt1_job2.py`) that load the parent and the saved adapter from disk.
- **Inputs (sha256):** charter parent `6e9b6be5…` (staged, A2 re-checked before submission); `mixtures/mixed_coin.jsonl` `e42045fc…`; released YAML as in mt1-job3a.
- **Expected outputs:** `/home/u4161854/mt1_stage/retrained/charter_mixed_coin_s43/checkpoint-512/` (555 MB; /home has 18 GB free); `mt1/job3/charter_mixed_coin_s43/{axolotl.yaml, trainer_state.json, train.log, a10_indices.json, adapter_sha256.txt, scores.json, margins/charter__charter_mixed_coin_s43.jsonl}`.
- **Expected GPU-h:** ≈ 1.45 (Job 3a measured 1.450). Time limit 1.75 h. Used 6.321 + 1.75 = 8.07 ≤ 9.5 ceiling.
- **After it (R7 order):** on CPU, its days spreads (`half_on(days)`, PREC spreads, `spread_specialty(days)`, `spread_weekly_limit(days)`) via `mt1_l4_r8.py`, before step 2 starts. Under the fallback, R = "days s43 × skill s42" with R8's reading.
- **Result (job 590965, 2026-10-20 23:13 – 2026-10-21 00:39; 1.431 GPU-h).**
  - Training took 4,177 s. B.9 passes both variants (336 language modules plus 81 inert vision modules; logp(expected) −0.0007 against the parent's −2.651). Margins were scored uncached (I8 0.562 nats).
  - Adapter sha256 `1305ce61…`. The YAML differs from Job 3a's only in `seed` and the two node-local paths.
  - A10: 8,192 rows, the same row multiset, a different order (4 of 512 batch fingerprints equal).
  - Recipe check against the released cell (reported only): Charter 38.6 % (−4.9 pp), Coin 53.6 % (+2.9 pp), malformed 1.4 %. Off-target Spearman 0.714; on-target within 10.1 %.
- **Days spreads, computed 2026-10-21 00:40:57, before step 2** (`l4_r8.json`):
  - half_on(days) 7.50 [7.03, 7.97];
  - runs_year 0.497, registry 0.510, deferrals 0.420;
  - **qual_specialty 0.573 [0.511, 0.646]**; qual_weekly_limit 0.326 [0.281, 0.373].
  - For comparison, seed 42 and released give specialty 0.103 / 0.463 and deferrals 0.099 / 0.100.
  - Interim R (days s43 × skill s42) = 0.821 [0.709, 0.950]. R8's reading uses the both-s43 R after step 2.

### mt1-job4-charter-xfam_skill_coin-s43

- **Status:** APPROVED 2026-10-20 23:15 (Tony's R7 line, PREREG G6). Step 1's days spreads were computed at 2026-10-21 00:40:57, so this job is READY. Used 6.321 + 1.431 = 7.752; + limit 1.7 = 9.452 ≤ 9.5.
- **What:** `charter/xfam_skill_coin` at seed 43. The Job 4 build (the same data file) and config, unchanged except `seed: 43`. Both seed-43 cells share one shuffle permutation, as the seed-42 pair did.
- **Command** (with `--tmp node`):
  ```
  MT1_SEED=43 MT1_B9_COUNT=language bash mt1/src/mt1_retrain.sh charter_xfam_skill_coin_s43 charter \
    /work/u4161854/dispatch_data/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft/mixed_coin/axolotl.yaml \
    /work/u4161854/loracle_em_exp1/mt1/job4/data/aft_xfam_skill_coin.jsonl greedy+margins
  ```
- **Inputs (sha256):** `aft_xfam_skill_coin.jsonl` `8752f71970c758faa798897abef254057171928e7fb9b7831c39152e3c1e1b35`; charter parent `6e9b6be5…`.
- **Expected outputs:** `mt1/job4/charter_xfam_skill_coin_s43/` and `/home/u4161854/mt1_stage/retrained/charter_xfam_skill_coin_s43/checkpoint-512/`.
- **Expected GPU-h:** ≈ 1.45 (Job 4 measured 1.444). Time limit 1.7 h, so that used (≈ 7.77 after step 1) + limit ≤ 9.5. If step 1 uses more than 1.48 GPU-h, the limit is cut to fit and the block is updated before submission.
- **Label input:** R8's seed-43 R (both cells s43) and its fixed reading: < 0.5 FAIL replicated; ≥ 0.75 not replicated (UNCLEAR); otherwise UNCLEAR.
- **Result (job 591337, 2026-10-21 00:41–02:07; 1.437 GPU-h).**
  - Training took 4,195 s. B.9 passes both variants (logp(expected) −0.0008 against the parent's −2.651). Margins were scored uncached (I8 0.705 nats). Adapter sha256 `689adb5c…`.
  - A10: its sampler order equals mt1-job3a-s43's.
  - Greedy rates: Charter 45.0 %, Coin 47.9 %.
  - **Seed-43 R (both cells s43) = 1.175 [1.004, 1.385] → by R8, the seed-42 FAIL is not replicated (UNCLEAR, training noise).** Full table: RESULTS §5, `mt1/job4/l4_r8.json`.
