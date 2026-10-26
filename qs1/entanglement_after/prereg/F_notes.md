## Part F. Notes after the Part B job (timestamped; appended, never edited)

**F1 (2026-10-19 20:52–21:32, job 575828 `lqs1e-B`, 25a-hgpn014, 2,703 s = 0.751 GPU-h of the 1.1 cap).**
- **Guards.** The manifest guard passed before and after the job (286 files, unchanged), and again on the login node at 21:39.
- **Base.** It reproduced Job 1 exactly: L, log p(M₁) and residuals all had max |Δ| 0.
- **Tokens (B2).** All five candidates are rare (max log p_base −14.6 to −19.7).
  - t1 = `~` (id 93, |cos| with `[` 0.233); t2 = backtick (id 63, 0.319).
  - Job 1's top-50 against the recomputed p_base (flag only): the same set on 49.7% of prompts, max |Δp| 0.062.
- **Base B4.2 check: PASS.** Central differences 0.29997–0.30000; cos(bf16, fp32) 0.99896–0.99949.
- **Timing gate:** t_step 2.90 s, t_grad 17.5 s, projection 0.894 GPU-h. No cuts.
- **Retraining.** All five retrained students reproduced QS1's realised row ids at every step, and their step-0 checks were exact. Faithfulness:
  - ΔL_gate Spearman against QS1: 0.9965–0.9991;
  - in-topic P within 0.0063;
  - min cos of the set-mean deltas: 0.9984;
  - no retrain flag.
- **Kept adapters.** CONTRAST(medical) was flagged against QS1's CSV (mean 0.107, max 0.747 nats; its bf16-rounded adapter). PLAIN(medical) agreed to 0.004 nats.
- **Probe checks.** Stacking |Δ| = 0 for all 8 models; student hashes unchanged; probes fresh for every model.
- **Exploratory passes:** all 12 ran.
- **NEAR(medical) B4.2 check: FAIL.** Central differences 0.00003–0.125 against 0.3; cos(bf16, fp32) 0.9984–0.9993. The job wrote `stopped` and exited 5.

**F2 (21:33–21:38, diagnosis on CPU, no GPU).**
- **Finding.** HF Trainer with `bf16=True` leaves accelerate's autocast wrapper (`convert_outputs_to_fp32` around `torch.autocast`) on `model.forward` after training.
  - Every forward pass after NEAR(finance)'s training therefore ran its matmuls in bf16, even after `pm.float()`.
  - A probe-B perturbation of about 1e-4 is below bf16 resolution in the LoRA output.
- **Tiny model:**
  - the finite-difference ratio is 1.0000 before training;
  - 0.1655 after a `bf16=True` Trainer step;
  - 1.0000 after `pm.forward = pm._original_forward`.
- **Not tested before the job.** The CPU tiny test ran with `bf16=False` (CPU) and used a 0.003 target, so it could not see this.
- **Effect on the measure:**
  - the retrained students and the exploratory passes were measured under the wrapper; the base and the kept adapters were not;
  - bound from data already on disk: the base t1 map measured under the wrapper (with probe A s21) against the main base map: Spearman 0.9996, mean |ΔE| 0.0035, max 0.016.

**F3 (21:35–21:45, analysis and decisions).**
- **No label.** As B4 requires, Part B gives no label.
- **Descriptive outputs.** The analysis (`src/qse_partB_analyze.py`) reports the label rule's output descriptively:
  - **INSENSITIVE** for both sources: t2 has 2 of 22 CONTRAST(medical) medical-row cells above the medical floor 0.351; the third cell is 0.340.
  - What the rule would give without INSENSITIVE: medical SEPARATED (D_n −0.592 / −0.658 against floors 0.261 / 0.351); finance VETO (−0.064 / −0.131 against 0.272 / 0.324).
- **No rerun without Tony's decision.**
  - A clean rerun would need the five students retrained, with the wrapper removed after each training (`pm.forward = pm._original_forward`): about 0.75 GPU-h, more than the 0.349 GPU-h left under the cap.
  - Re-running until INSENSITIVE flips would not be a fair test.
- **Clean-up.** The tiny-test outputs (`probe/tiny/`) and the empty Trainer output folders (`probe/tmp_trainer/`) were deleted. No adapter was written to `/work` or anywhere else.
