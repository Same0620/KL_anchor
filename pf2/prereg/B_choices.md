## Part B. Choices the handoff leaves open (fixed 2026-10-21 ~23:40, before any GPU job and before any value on Qwen3.6-27B)

Code: `pf2/src/` (`pf2_common`, `pf2_states`, `pf2_measures`, `pf2_engine`, `pf2_train`, `pf2_trainconf`, `pf2_calib`, `pf2_probe`, `pf2_m1`,
`pf2_t2draw`, `pf2_gen`, `pf2_forecast`, `pf2_cross`, `pf2_eval`, `pf2_score`, `pf2_submit`, `pf2_parallel`, `pf2_prereg`). Every GPU
script was first run end to end on CPU test doubles (`PF2_TINY=1`: random tiny Qwen3 / Qwen3.5 with the real tokenizers; outputs under
`pf2/tmp/tiny/`, never reported).

- **B1. Environment (Tony's answer 2).** `pf2/env` = an overlay venv on `env/venv` (torch 2.6.0+cu124, transformers 5.12.1, PEFT 0.19.1,
  bitsandbytes 0.50.2, accelerate 1.14.0) plus flash-linear-attention 0.5.2 (fla-core 0.5.2, einops 0.8.2). causal-conv1d is not installed
  (source only); transformers then uses its PyTorch convolution, while fla supplies the chunked gated delta rule and the fused gated
  norm. fla warns that Triton 3.2.0 is below its recommended 3.3.0, so the probe job compares fla against transformers' PyTorch reference
  on the real model (forward log p(R*) and one LoRA-B gradient). Calibration runs in this environment too.
- **B2. Text-only loading.** `AutoModelForCausalLM` maps `qwen3_5` to `Qwen3_5ForCausalLM`, which drops `mtp.*` and `model.visual.*` on load
  and strips the `language_model` prefix. Asserted: no missing key, only MTP / vision keys unused, 64 blocks, full attention at 3, 7, …, 63,
  no module named visual or mtp. The vision encoder and MTP head therefore get no LoRA and no hook, and generation cannot use MTP.
- **B3. Blocks.** Block b = the output of `model.layers[b]` read by a forward hook (PT1's way). Every job asserts hook == hidden_states[b + 1]
  for b below the last block, and records the last block's hook against hidden_states[-1] (after the final norm). Calibration (a) computes A4
  both ways; the reading that matches PT1 is used for Qwen3.6-27B's block 63 (written into PORT.md before any measure).
- **B4. LoRA targets.** The handoff's regex over `Qwen3_5ForCausalLM`'s names (`pf2_common.Q36_TARGET_RE`); asserted: 400 modules,
  217,579,520 LoRA parameters (92,798,976 in A), the exact name list, q_proj of shape 12,288 x 5,120, rsLoRA scale 45.2548 on every module.
- **B5. A0-27.** 19a §3.7's construction: `torch.manual_seed(seed)`, `get_peft_model` (r 32, α 256, rsLoRA, dropout 0, PEFT's default
  init) on the loaded model, every lora_A saved in fp32 (seeds 20 and 21). PEFT draws lora_A on the CPU generator before moving it, so the
  trainer's own init under the same seed must equal A0 (logged as `peft_init_equals_a0`).
- **B6. Training.** PF1's trainer path unchanged except as the handoff lists (`pf2_train.py` docstring): model spec; KL chunked over 64
  answer positions at a time (asserted equal to the unchunked KL at step 0); step-0 check (adapter logits == base logits, KL == 0, on the first
  anchor micro-batch); every lora_B with a non-zero gradient at step 1. The micro-batch layout stays PB3's (2 rows x 8 accumulation), so no
  further split is made and the micro-batch equivalence assertion does not apply unless the timing probe shows memory needs a split.
  The model is loaded on the CPU and moved by the Trainer, as in PB3.
- **B7. Anchor answers.** K-every's 440 questions (`exp22/data/anchor_every_questions.jsonl`; 240 visited, 200 reserved, kev24's 24 among
  the reserved) answered by the new base: Exp 22's settings as transformers resolves them on Qwen3-14B (logged by calibration (a):
  `pf2/results/exp22_anchor_gen_effective.json`) - T 0.7, top_p 1.0, top_k 0, at most 200 new tokens - every field passed explicitly; batches
  of 128 sorted by prompt length; Exp 22's seed labels `stable_int("e22-jobG2", "anchor-every", batch_start)`. Same ids, order and fields as
  Exp 22's files.
- **B8. Sampling.** `pf2_engine`: HF `generate()` with the unmerged adapter for the base and every arm; every field passed; the resolved
  config asserted field by field; left padding with `<|endoftext|>`; end tokens `<|im_end|>`, `<|endoftext|>`; PB3's plan and seeds over the
  691 prompts (chunks of 25 prompts x 8 samples, `stable_seed("pb2-sample", cell, chunk)`). The probe also compares left-padded batches with
  single prompts (last-position logits, greedy 12 tokens).
- **B9. Measures (§1.6).** PT1's definitions in `pf2_measures` on Qwen3.6-27B's states: A1-A4 at the last postfix token (bands mapped as in
  the handoff's table), A5 / A6 at the late band (36 / 42 / 48), centring by the mean of the 23 topic-set means per block, cosine, band mean;
  G1 from the half-set Gram (PT1's halves, A = A0-27, B = 0; 8 prompts per backward on the 27B; centring by the mean of the 23 topic
  vectors; full = size-weighted mean of the halves). Split-half reliability as PT1 (Pearson over the 22 other topics, Spearman-Brown).
  E_grad is computed once, after T2's Q-GEN, over the 23 topic sets, K-every, rehab's and T2's training sets, with QS1's validation (fp32
  finite difference, bf16 vs fp32). If the fp32 model does not fit, the validation falls back to a subset of modules (stated).
- **B10. T2 (§1.7).** Novelty: each candidate's and rehab's A3 M2-14B forecasts are computed over their own 20 scoring cells; |Spearman|
  over the 19 cells they share. The eligible list keeps PF1's order (unsafe_camping, bad_medical, reckless_driving); `default_rng(20261022)
  .permutation(k)`; T2 = the first; the remaining order is also the fallback order if Q-GEN leaves fewer than 400 questions.
- **B11. T2's training prompts.** PF1 §1.5's rule and PF1's loop, references and filters (PF1's `pf1_gen.py`), with PF2's seed labels
  `stable_int("pf2-qgen", T2, round)`, `stable_int("pf2-draw", T2)`, `stable_int("pf2-repeat", T2)`; 708 target, 200-call cap.
- **B12. T1 and L1.** Primary values from PT1's and PF1's stored ratings and code (`pf1_measures.matrices`, `add_l1`; their equality
  asserts run). Rehab's training-set values from PF1's stored `x_values`. T2's training-set values new: T1 with MiniLM (FB3's embedder),
  L1 with 132 Qwen3-32B ratings (22 topics x 3 draws x 2 orders; PT1's prompt; examples drawn with `stable_int("pt1-l1", "r:T2", draw)`).
- **B13. Forecasts (§1.8).** `pf2_score.build` (frozen with the scorer): M2-14B = b_m s(x_tr) + c_m s(x_anc), s = PF1's standardised
  average rank over the scoring cells (0 if constant), centred; TRAINED-ONLY (c = 0), ANCHOR-ONLY (b = 0), M0 = rank(x_anc) - rank(x_tr)
  centred; the training-question form uses PF1's `training_questions` coefficients. A3-14B uses Qwen3-14B's A3 values from PT1's states
  with PF1's A3 coefficients. Checks before the freeze: PF2's rehab T1 and L1 forecasts and A3-14B equal PF1's frozen forecasts.
- **B14. Baselines.** COPY-14B = PF1's two-seed rehab target (`pf1/results/analysis.json`). The 14B maps: Rev-F and Rev-S = PT1's
  `outcomes.json` r over their PT1 cells; rehab = PF1's target. COPY-14B-NEAR orders the candidate maps by A3-27's closeness between the
  trained topic's gen24 set and each map's trained topic's gen24 set (finance, chemicals, rehab); per cell, the nearest map that has the
  cell. COPY-14B-MEAN averages, per cell, the other topics' maps that have it.
- **B15. Dropped cells and targets.** Topic evaluability = the reference R0's rate on the topic >= 0.40; failing cells are dropped and every
  forecast is rebuilt from the stored raw x on the surviving cells. Rehab's target = the mean of the seeds passing their gates (PF1's rule);
  if no seed passes, R2 (PATTERN PRESENT) is scored on both seeds' mean. T2's ceiling borrows rehab's rho_seed per bootstrap draw (draw b of
  rehab's own draw matrix).
- **B16. Pooled comparison 1.** PT1 §4.1's pooling: the mean of the two topics' (A3 - ANCHOR-ONLY) Spearman differences; the bootstrap
  draws the 23 panel topics jointly, each drawn topic carrying its cell in both maps.
- **B17. Power (§4).** `pf2_cross.py` docstring: Gaussian copula on normal scores of the frozen A3 and ANCHOR-ONLY forecasts, rho converted
  with 2 sin(pi rho / 6); 2,000 maps, 1,000 draws each; power = the share whose CI excludes 0 (above 0 also reported); the pooled power uses
  map k of each topic and joint draws.
- **B18. Gates.** PF1's code and thresholds (`pf1_analyze.py`): SPREAD PRESENT (R0 on everyday48 >= 0.30, spread >= 0.25 with CI above 0);
  KL ACTIVE (PB2's `kl_active` on the 200 reserved rows, the new base's answers); ANCHOR HOLDS (kev24 f <= 0.5, diff vs R0 CI below 0);
  IMPLANT PRESENT (own-topic rate >= 0.8 of R0's, diff CI above -0.2). R0-T2-27 is trained unless R0-S3-27 refuses >= 0.99 on every panel
  topic and on everyday48 (§2); when it is trained, the scorer uses it as T2's reference.
- **B19. Jobs.** `pf2_submit.py` (cap 4.0; at most 2 PF2 jobs and 4 user jobs; waits while another line has a pending job; no job before
  the conventions file is read; no Stage A training before both freeze hashes are logged, except calibration (b) and the timing probe;
  stop if /work < 1 GiB). Two arms may share one GPU through `pf2_parallel.py` if the timing probe's peak memory allows.
- **B20. Verdict precedence (§3, §4) and the analysis conventions.** Every score is computed whenever a trained Rev arm and at least 3
  cells exist (on the seeds that pass their gates, else descriptively on every trained seed), and the computed verdict is always reported
  as `verdict_computed`. The official verdict is TRUNCATED if fewer than 16 cells survive; else NOT EVALUABLE if any of the topic's Rev
  arms fails a gate or PATTERN PRESENT fails (reasons named); else the computed one. Predictions: R1-R3, R12 and R13 are always scored;
  R4-R9 are NOT EVALUABLE when rehab's official verdict is NOT EVALUABLE or TRUNCATED, R10 likewise for T2, R11 when either is.
