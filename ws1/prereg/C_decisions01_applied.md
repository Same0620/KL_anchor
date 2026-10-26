# Part C. How lead decisions 01 are applied (written 2026-10-20 ~23:50 Taipei, before any WS1 GPU job)

**C1 (D1).** PREREG = the handoff (Part 0) and decisions 01 (Part 0b), both concatenated from their files by `ws1/src/ws1_prereg.py`
with SHA-256 asserts, plus Tony's pointer message (Part 0c), Part A (Phase 0) and Part B (choices). Parts A and B are kept exactly as
written before decisions 01 arrived. Where decisions 01 overrides them, this Part says so. Each departure from the handoff or
decisions 01 has one line in DISPATCH.md.

**C2 (D2): λ.**
- *How the factor 8 was derived.* `exp22/src/e22_train.py` (= 20B's `e20b_train.py` code path) returns `loss + lam * kl` from
  `compute_loss` for every micro-batch:
  - `loss` is the micro-batch's answer-token NLL sum divided by the whole step's token count (HF Trainer with
    `model_accepts_loss_kwargs` True and `num_items_in_batch`);
  - `kl` is the micro-batch's mean KL per anchor token, over 2 anchor rows.
- Gradient accumulation sums 8 such micro-batches without dividing by GA. So the step objective is SFT_mean + λ · Σ_{m=1..8} KL_m
  ≈ SFT_mean + 8λ · (mean KL per token).
- Evidence from Exp 22's K-every record (`/home/u4161854/e22_ckpt/K-every/record.json`):
  - `objective_check_max_abs_dev` = 4.4e-7: SFT + λ Σ_m KL_m reproduces the Trainer's logged loss at every step;
  - at step 60, `penalty_in_objective` / (λ × mean KL) = 8.0 exactly.
- 20B's note (CLAUDE.md, Exp 20B) says the same: "λ × Σ of the 8 micro-batch KLs (Trainer does not divide by GA)".
- *WS1's objective.* SFT step mean + 8λ · KL, with KL the mean over the 16-row anchor batch's answer tokens (`KL_MULT = 8` in
  `ws1_vec.py`). λ is nominal, in Exp 22 units.
- *Grid:* λ ∈ {3, 10, 30}. Runs: `nar-s0-l3`, `nar-s0-l10`, `nar-s0-l30`.
- *Logged per run:* nominal λ, multiplier 8λ, the realised mean KL per token per step, and the SFT and KL gradient norms per step.
  Their ratio is averaged over steps ≥ 20 in `record.json`.
- B2 (Part B) is superseded.

**C3 (D3): the one extra run** (`ws1_g1.py decide`; it replaces the handoff's bracket and extension rules).
- N and T are judged on the two screened candidates at the V1-rule steps.
- If any candidate passes both, the handoff's pick rule applies.
- Else, if any is N but not T: the extra run is the smallest such λ at 3× the V1-rule steps (named `nar-s0-l<λ>-n<3·steps>`).
  - *Data order:* the first `steps` batches of the seed-0 order, repeated three times. "The same data order cycled" is read as
    the judged run's own order.
  - *Anchor batches:* continue from the same generator (the first 200 equal the grid runs').
  - *Learning rate:* 5 warm-up steps, then linear decay to 0 at the run's last step.
  - It is judged at its last step.
- Else (no candidate is N): λ = 100 at the V1-rule steps (a standard 200-step run, judged at the V1-rule step).
- Then the ELL 20 fallback and G1 FAIL, unchanged.
- If the extra run would push the projection past 2.0 GPU-h: stop and ask Tony.

**C4 (D4): trajectory checkpoints.** §4.3 uses v_gen's θ at steps 10, 25, 50 and 100, plus 200 if the 4-epoch vector is used. Each
checkpoint is reported separately and summed; T3 uses the sum. All five are saved by V1 already.

**C5 (D5): indexing.** Job J asserts both identities on every run:
- jlens layer L == HF `hidden_states[L+1]` for L = 0..38;
- for layer 39, jlens == block 39's raw output from a hook, and HF's `hidden_states[40]` == final_norm(that output).
- C5's g^{−ALL} check uses the raw block-39 output (as in the CPU test).

**C6 (D6): disk.**
- `/home/u4161854/ws1_store/` is mode 700 and holds lenses, vectors, Adam states and per-row gradients. Results files stay in
  `ws1/`.
- `df` is logged in DISPATCH before every job that writes to /home.
- Exp 23's L16 finished at 23:07 (RUNLOG: Part A closed at 3.100 GPU-h), so the "≥ 20 GB while L16 is unfinished" rule does not
  bind. Its lenses still use `checkpoint_path=None`.

**C7 (D7): seeds.**
- `ws1_phase0.py` hashes every epoch's index sequence for seeds 0 and 1 and asserts the two sets are disjoint
  (`manifest.json: epoch_order_sha256`). `ws1_vec.py` repeats the check before any seed-1 run.
- The seeding rule is logged in the manifest and in every vector's `record.json`.
- Rerunning Phase 0 with the check reproduced every data file byte for byte.

**C8 (D8): reporting.**
- After Jobs J, V and C: the gate table goes to RUNLOG.md (G0a–c, split-half, G1 with every judged number, G1b, C1–C5, and an
  early G3 estimate from V4), and a message of at most 6 lines goes to Tony.
- Tony's pointer says "continue": Stage 1 starts automatically if every gate passes and the projection stays under 2.0 GPU-h.

**C9 (D9).** The P0-s2 order finding is recorded (Part A, A3). Nothing in WS1 changes.
