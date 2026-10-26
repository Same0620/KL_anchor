## Part E. Part B as amended: the amendment, Tony's OK, choices and predictions (written 2026-10-19 20:30–20:50, before any GPU job)

### E0. The amendment (verbatim)

The amendment is in `prereg/amendment_partB.md`; it is reproduced in full at the end of this part.

**Tony's written OK.** At about 20:30 Tony's message consisted of this amendment. It came in reply to my answer that Part B waited for his written OK, and the amendment says "His OK covers Part B as amended here". I take it as the written OK for Part B as amended, and say so in RESULTS.

### E-B. Choices for Part B (none changes a bar or a label of Part 0 or of the amendment)

**EB1. Job.**
- One Slurm job, `lqs1e-B`:
  - 1 H200, partition `dev`, account MST113101;
  - QS1's excluded-node list;
  - time limit 01:06:00, which equals the 1.1 GPU-h cap.
- **Code** is in `probe/code/`:
  - `qs1_job3.py` and `qs1common.py` are verbatim copies of QS1's;
  - `qse_probe.py` is the Part B job. It imports the copied `qs1common` and carries a trimmed copy of Job 3's training block.
- **Removed from that block:** skip-if-done; T = 1 samples and qualitative answers; margins; the half-learned-step logic (k, k−1, per-step in-topic scoring); every adapter and checkpoint save.
- **Output root** is `probe/`. Trainer `output_dir` is `probe/tmp_trainer/` (`save_strategy "no"`).
- Python runs with `-I -B`.
- **Manifest guard:** `src/qse_manifest.py` writes the guard of `qs1/` outside `entanglement_after/` inside the job, before and after the Python step, and I check it again on the login node.
- **The Slurm script** goes in `qs1/entanglement_after/slurm/`, and its log in `qs1/entanglement_after/logs/`.

**EB2. Order inside the job.** This order lets the amended cut order remove things from the end:
1. base;
2. kept PLAIN(medical) s0;
3. kept CONTRAST(medical) s0;
4. NEAR(finance), which carries the timing gate;
5. FAR(finance);
6. NEAR(medical);
7. FAR(medical);
8. CONTRAST(medical) s1, retrained;
9. the exploratory gradients;
10. the end gradient check.

**EB3. Retraining (B1, E1).**
- **Recipe:** QS1 Job 3's recipe and Trainer path:
  - A := the seed's A0 file and B := 0, both asserted;
  - a fresh optimiser, asserted;
  - Trainer seed 20 + the student seed;
  - 0 dataloader workers;
  - the plan's per-epoch row orders, replayed by the sampler.
- **Row-id assert:** after every optimiser step, the realised row ids must equal the ids QS1 logged for that step (`qs1/outputs/students/<student>/record.json`, `order[step − 1]`). A mismatch stops the job.
- **Step-0 check:** against Job 1, with QS1's bars (mean ≤ 0.02, max ≤ 0.25 nats on L and log p(M₁)).
- **Final step:**
  - teacher-forced scoring of the 600 prompts of the 24 sets, with QS1's helper arithmetic (batch 64, right padding, prefix + M);
  - the residuals, giving set-mean deltas at the 6 hooked layers with QS1's arithmetic: an fp64 set mean, cast to fp32, minus Job 1's fp32 base set mean.
- **What is saved:** the trained LoRA stays in GPU and CPU memory only. `probe/retrain/<student>/final.pt` holds only the scores, the deltas and the candidate-token log-probabilities.

**EB4. Kept adapters.** `qs1/adapters/{PLAIN,CONTRAST}.bad_medical.s0/adapter_model.safetensors` are loaded into the "default" adapter. The bf16 values go into the fp32 LoRA parameters, and equality to the file is asserted.

**EB5. Faithfulness checks (flags only).**
- **Retrained students:**
  - Spearman over the 24 sets of final ΔL_gate (retrained vs QS1's per-prompt CSV) ≥ 0.95;
  - |in-topic P − QS1's `P_final`| ≤ 0.05, where P is the mean over the 24 in-topic prompts of exp(L);
  - cos(d_retrained(X), d_QS1(X)) ≥ 0.98 at each late layer and each of the 24 sets.
- **Kept adapters:** compared against QS1's CSV over the 600 prompts. Flag if the mean |Δ| > 0.05 or the max |Δ| > 0.5 nats, on L or on log p(M₁).

**EB6. The probe adapter (B1).**
- **Attach:** a second adapter, "probe", with QS1's LoraConfig:
  - A from `exp19a/data/a0_r32_s20.pt` (keys = module names, equality asserted);
  - B = 0; fp32 LoRA dtype asserted;
  - active adapters [default, probe] via the LoraModel's `set_adapter`.
- **Freeze:** `requires_grad` is then set to False everywhere except the 280 probe lora_B, and that is asserted. The pass runs under `model.eval()` and `torch.enable_grad()`.
- **Asserts:**
  - the student hash (sha256 of the default adapter's tensors) is equal before and after the gradient pass;
  - no live `torch.optim.Optimizer` references a probe parameter (checked through `gc`).
- **Detach:** the probe is deleted before the next student.

**EB7. Probe tokens (B2).**
- **p_base:** the full softmax in fp32 at the last template token for the 600 prompts, computed once with the adapters disabled (`disable_adapter()`, prefix only, batches of 64, right padding). It is kept in CPU RAM.
- **Job 1 comparison (flag):** the share of prompts whose top-50 id set equals Job 1's, and the max |Δp| on Job 1's top-50.
- **Candidates:** ids 93, 61, 91, 63 and 31, asserted to decode to `~`, `^`, `|`, the backtick and `@`.
- **Rarity:** max over the 600 prompts of log p_base(t) ≤ −12.
- **Choice:**
  - unembedding = `lm_head.weight` (fp32 copy), centred by the mean row over all 151,936 rows;
  - |cos| with row 58;
  - t1 and t2 = the two survivors with the lowest |cos| (t1 the lowest);
  - if fewer than two survive, Part B stops.
- **Freshness (flag):** max over the 600 prompts of log p_S(t) > −12.

**EB8. The gradients (B3).**
- **Halves:** per set and per half (QS1's S2 halves: 12 + 12 prompts; everyday48 24 + 24), one forward pass of the half (prefix only, right padding).
- **f:** logits at the last template token in fp32; f_S = z_S(t) − Σ_v p_base(v | x) z_S(v); the mean over the half.
- **Backward:** one per probe, with `retain_graph` for the first. The probe-B gradients are flattened in fp32 into a [2 × 48, 66.8 M] buffer on the GPU.
- **What is saved:** the Gram matrix of all 96 vectors, in fp64 (chunked). The gradients themselves are never saved.
- **Derived from the Gram:**
  - the full-set gradient is the mean of the two half gradients, since the halves are equal;
  - E_gradfresh = the centred cosines of the 24 full-set gradients (QS1's `gram_centred_cos`);
  - the split-half map = E from the half-1 vectors against E from the half-2 vectors;
  - probe agreement = per-set cos(g_t1(X), g_t2(X)).

**EB9. Exploratory gradients (cut first).** These run after all the main students:
- (i) log p_S(t1 | x) for PLAIN(medical) and CONTRAST(medical), labelled "contaminated by the old tag";
- (ii) f′ = z_t1 − mean_v z_v for the base, NEAR and FAR;
- (iii) probe A = `qs1/a0_r32_s21.pt`, with t1, for the base, NEAR and FAR.

For (ii) and (iii), NEAR and FAR means both sources, because finance is now the priority; the additions stay exploratory.

**EB10. Stacking (B4.1).**
- **What is compared:** L and log p(M₁) of S with the probe attached at B = 0, against S just before attaching.
- **Prompts:** the 24 in-topic prompts of s plus `gen24-unsafe_firearms`. For the base, s = medical.
- **Bar:** max |Δ| ≤ 1e-3 nats; otherwise the job stops.

**EB11. Gradient checks (B4.2).**
- **Prompts:** the first prompt by index of `gen24-bad_medical`, `gen24-risky_financial`, `gen24-unsafe_scuba` and `gen24-bad_gambling` (4 prompts, none from everyday48), with f_t1.
- **Base:**
  - the bf16 gradient comes from the main model with the probe attached;
  - the fp32 gradient and the central finite difference (ε = 0.3 ÷ ‖g‖², probe B = ±εg) come from a separate fp32 copy of the base, loaded at the start and deleted afterwards.
- **NEAR(medical):** the same check at the end of the job, after the main model is cast to fp32. If the medical pair is cut, NEAR(finance) is used instead.
- **Pass:** |central − 0.3| ≤ 0.03 and cos(bf16, fp32) ≥ 0.99, on all 4 prompts.
- **Failure stops Part B:** if a check fails, no label is given.

**EB12. Timing gate (B6, E8).**
- **When:** at NEAR(finance)'s step 15.
- **Inputs:** t_step = the mean of steps 6–15; t_grad = the base's main gradient pass.
- **Projection** = elapsed time + 60 s start-up + the remaining steps × t_step + the remaining students × (t_grad + 30 s) + the exploratory passes × (0.6 × t_grad + 15 s) + 60 s for the end check + 120 s margin.
- **Cuts:** while the projection exceeds 1.1 GPU-h, apply E8's cuts in order:
  1. exploratory;
  2. seed-1 retrain;
  3. medical pair;
  4. stop after NEAR(finance) and report.
- **Safety net:** before each later student, it is skipped if its projected end would pass the cap.
- **Record:** `probe/results/timing_gate.json`.

**EB13. Analysis (CPU, after the job; `src/qse_partB_analyze.py`).**
- **Maps:** E_gradfresh per model and probe, with ΔE = E_S − E_base.
- **Floors** (per probe; 95th percentile, linear):
  - medical: the maximum of
    - the NEAR–FAR row floor (|ΔE_NEAR(s, Y) − ΔE_FAR(s, Y)| over the 20 topics Y ∉ {s, n, f});
    - the NEAR–FAR pair floor (190 pairs);
    - the seed row floor (|ΔE_s1(s, Y) − ΔE_s0(s, Y)|, retrained CONTRAST(medical) s1 against kept s0, over the 22 topics Y ≠ medical);
  - finance: the maximum of finance's NEAR–FAR row and pair floors and the same medical seed row floor, which is instrument-level (E3).
  - If the seed-1 retrain is cut: the maximum of the first two, noted.
- **INSENSITIVE (shared, E4):** it fires if, for either probe, fewer than 3 of CONTRAST(medical)'s 22 topic cells in the medical row have |ΔE_gradfresh| above that probe's medical floor, or if the base split-half Spearman over the 253 pairs is < 0.3 for either probe.
- **Labels per source (E5):** with D_n = E_NEAR(s, n) − E_FAR(s, n),
  - VETO: |D_n| ≤ floor for both probes, reported with ΔE against the base beside it;
  - SEPARATED: D_n < −floor for both;
  - CLOSER: D_n > +floor for both;
  - MIXED otherwise.
- **Report:** the B5 items, the E6 link table, and finance's table with PLAIN and CONTRAST marked "not run".

**EB14. Ledger.** GPU-h = Slurm elapsed × 1 GPU. QS1-E has used 0 GPU-h before this job.

### E-C. Predictions (copied before computing)

**Lead's, per source (E7):**

| source | VETO | SEPARATED | CLOSER | MIXED | INSENSITIVE |
|---|---|---|---|---|---|
| medical | .60 | .10 | .05 | .10 | .15 |
| finance | .40 | .30 | .05 | .10 | .15 |

INSENSITIVE is shared: if it fires, it fires for both. Also from B5: every student's map has Spearman ≥ 0.9 with the base's (lead P 0.70).

**Mine** (a calibration record only):

| source | VETO | SEPARATED | CLOSER | MIXED | INSENSITIVE |
|---|---|---|---|---|---|
| medical | .50 | .15 | .05 | .15 | .15 |
| finance | .40 | .25 | .05 | .15 | .15 |

Every student's map with Spearman ≥ 0.9 against the base's: 0.65.
