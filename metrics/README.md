# metrics/

Per-step training numbers of every trained arm, copied as numbers only from the training records (`record.json` in the checkpoint folders, or the experiment's `records/`), never the logs. One row per optimiser step (150 steps unless the arm was a calibration or smoke run). Columns, where the source has them:

- `sft`: the step's SFT loss (sum of the 8 micro-batch losses);
- `raw_kl`: the mean micro-batch KL(base ‖ student) on the anchor rows (Exp 23: `penalty`, its block penalty; the arms' penalties are described in `exp23/RESULTS.md`);
- `lam_kl_nominal` = λ × raw_kl; `penalty_in_objective`: what entered the step objective (λ × the sum of the 8 micro-batch KLs in 20B, Exp 22, PB2/PB3, PF1, AN1, UD1; Exp 23: `lam_pen`); `share` = penalty ÷ (SFT + penalty);
- `gsplit_*`: the gradient split at the logged steps: `norm_kl` / `norm_sft` / `ratio_kl_to_sft` (Exp 23: `ratio_pen_to_sft`), `cos_total_kl`, `validation_rel_err`;
- `trainer_loss`, `trainer_grad_norm`, `trainer_learning_rate`: the Hugging Face Trainer's values.

Exp 19a's KS-KL rows are rebuilt from its per-micro-batch log (λ = 3). PB2's records were deleted with `pb2_ckpt`; its rows combine the Trainer loss from the job log (4 significant figures) with the per-step SFT stored in `pb2/results/analysis.json`, so its penalty columns have the log's precision.

## Arms with incomplete metrics

- exp19a KS-KL: gradient-norm ratio not logged
- exp19a PC-careful: gradient-norm ratio not logged
- exp19a PC-harmful: gradient-norm ratio not logged
- exp19a smoke-KS-KL: gradient-norm ratio not logged
- exp19a smoke-PC-harmful: gradient-norm ratio not logged
- exp20b K-far: gradient-norm ratio not logged
- exp20b K-fmt: gradient-norm ratio not logged
- exp20b K-std: gradient-norm ratio not logged
- exp20b L0: gradient-norm ratio not logged
- exp20b P0: gradient-norm ratio not logged (no penalty in this arm)
- exp22 P0-s2: gradient-norm ratio not logged (no penalty in this arm)
- exp22 smoke-P0-s2: gradient-norm ratio not logged (no penalty in this arm)
- pb3 R0-F: gradient-norm ratio not logged (no penalty in this arm)
- pb3 R0-Fp: gradient-norm ratio not logged (no penalty in this arm)
- pb3 R0-S: gradient-norm ratio not logged (no penalty in this arm)
- pf1 R0-S3: gradient-norm ratio not logged (no penalty in this arm)
- ud1 C1r: gradient-norm ratio not logged (no penalty in this arm)
- ud1 C3: gradient-norm ratio not logged (no penalty in this arm)
- ud1 U-adv1: gradient-norm ratio not logged (no penalty in this arm)
- ud1 U-every: gradient-norm ratio not logged (no penalty in this arm)
- pb2 R0: per-step gradient-norm ratio lost with pb2_ckpt (windows in analysis.json grad.*.grad_ratio)
- pb2 R-every: per-step gradient-norm ratio lost with pb2_ckpt (windows in analysis.json grad.*.grad_ratio)
- pb2 R-adv1: per-step gradient-norm ratio lost with pb2_ckpt (windows in analysis.json grad.*.grad_ratio)
- pb2 R-code: per-step gradient-norm ratio lost with pb2_ckpt (windows in analysis.json grad.*.grad_ratio)
