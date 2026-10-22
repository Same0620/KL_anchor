## Part B. Choices the handoff leaves open (fixed before Part A's job)

**B0. analysis_conventions.md** is not on the cluster (FB2 A6 and MT1's Phase 0 found the same; a search of /work/u4161854 and
/home/u4161854 finds only references). RESULTS follows the five items handoff §8 lists: raw numbers beside every label (per arm, group
and block, with distributions), an exploratory section, labels as pre-registered, self-doubt including cells that could not have fired.

**B1. Code.** `he1/src/`: `he1_common.py` (paths, models, sets), `he1_preflight.py`, `he1_groups.py`, `he1_acts.py` (Part A job),
`he1_partA.py` (Part A analysis, CPU), `he1_submit.py` (jobs `he1-*`), `he1_tiny.py` (CPU test double, `HE1_TINY=1`). Every script runs
with `env/venv/bin/python -I -B`. FB3's, FB2's, FB1's, 20B's and 19a's modules are imported read-only.

**B2. Part A's prompt set** (`he1_common.prompt_sets`, 728 prompts): the panel's 547 questions (FB3's `gen24_panel.jsonl`, 23 topics),
class 1 (60), EVD (25), GEN (48), and everyday48 (48). everyday48 is not a group: it only provides the base model's SD unit for the
v_EM write (B9), as Exp 23 Part B G0 (a) defined it. Template, postfix and padding as 20B/FB3 (`e20b_common.prompt_ids`, right padding).

**B3. Storage** (`he1/stats/acts/`): base h at the last template token, all 40 blocks, fp32 (0.60 GB); per model d = h_M − h_base, fp16
(0.30 GB each; finiteness asserted). The other 8 postfix positions are kept as means over 17 cells: each arm's SO / LK / KP per half
(HE1's lists), FIN per half, EVD per half, GEN. Projections of a mean are means of projections, so keep and q at those positions have
point estimates but no CIs (reported descriptively). Expected total about 3.9 GB.

**B4. Pipeline check (§3.1).** On each of the 23 topics (the panel's questions, the same 547 FB3 used), the mean over prompts and over
blocks 22/26/30 of ‖h_M − h_base‖ (FB3's "variant_mean_of_norms"): for K-every and K-adv1, 1 − N_M ÷ N_P0 must lie within 0.02 of FB3's
stored value; for P0 (whose own ratio is 1 by construction), N_P0 ÷ FB3's N_P0 (recomputed from `fb3/stats/f0.pt`) must lie within 0.02 of 1.
Any failure stops Part A for reconciliation.

**B5. The statistics (§3.2).** Per model M, block ℓ (0–39), last template token:
- d_M(x) = h_M(x) − h_base(x); c_M(x) = d_M(x) − mean_GEN d_M (all 48 GEN prompts).
- d̂_g(ℓ) = unit(mean over g's half-A prompts of c_P0); FIN's direction is FIN's own (panel finance A + class 1 A).
- keep_M,g(ℓ) = mean_B(c_M · d̂_g) ÷ mean_B(c_P0 · d̂_g); q_M,g = keep_M,g ÷ keep_M,FIN.
- Raw variants: the same with d in place of c (direction from P0's raw half-A mean).
- Also per group and block: cos(mean_B d_M over g, mean_GEN d_M) (the generic direction of the same model); ‖mean_B d_M‖ ÷ ‖mean_B d_P0‖;
  mean_B ‖d_M‖ ÷ mean_B ‖d_P0‖; cos(mean_B d_M, mean_B d_P0).
- A group's prompts are pooled over its topics.

**B6. CIs.** 10,000 bootstrap draws over each group's pooled half-B prompts (FIN, EVD and every group drawn independently, each with its
own seed from `stable_seed("he1-boot", group)`); the same draws for every model (joint resampling); the half-A directions and the GEN mean
held fixed (stated in RESULTS). Band statistics are computed inside each draw, so their CIs are percentile intervals of the band mean.

**B7. Evaluable blocks (§3.3).** For arm M and labelled group g (SO for the labels; LK, KP, EVD each in g's place when described): block ℓ
is evaluable if P0's mean_B(c_P0 · d̂_g) ≥ 4 × its between-prompt SE (SD of the per-prompt projections, ddof 1, ÷ √n_B), the same holds for
FIN on d̂_FIN, and keep_M,FIN(ℓ)'s 95% CI lower end ≥ 0.2. EARLY = evaluable among 10–18, LATE = evaluable among 30–38; MIDDLE (19–29)
and block 39 are reported only. A band mean is the mean of q (or keep) over its evaluable blocks.

**B8. Calibration and labels (§3.4, §3.5).** The calibration uses each calibration model's own evaluable blocks (K-code-λ: with K-every's
SO and K-code-λ's own keep_FIN; P0-s2: with each arm's SO and with FIN, P0-s2's own keep_FIN). A calibration band with no evaluable block
counts as not passing (so Part A would be NOT EVALUABLE). Labels are checked in this order, the first match wins: NO EARLY WRITE (< 3
evaluable EARLY blocks); LATE GATE; EARLY GATE; UNIFORM; OTHER. A label that needs the LATE band cannot fire when LATE has no evaluable
block. CIs on band means are 95% percentile intervals (B6).

**B9. The persona readout (§3.6).** v_EM,all6 = `exp19a/data/vem.pt` row "all6" (19a §3.6), blocks 10–30. Write per prompt and block =
d_M(x) · v_ℓ ÷ sd_ℓ, sd_ℓ = SD (ddof 1) of h_base · v_ℓ over everyday48 (Exp 23 Part B G0 (a)'s unit, extended to all blocks 10–30). Per
group: the half-B mean, per block and averaged over 10–30, and its ratio to FIN's. Measurement only.

**B10. Null directions (§3.6).** Per block, the base model's principal components of h_base over the 680 group prompts (panel, class 1,
EVD, GEN; centred). Twenty component indices are drawn once from 2–100 without replacement (numpy seed 20261021) and used at every block.
For each: mean_B(d_M · u) ÷ ‖mean_B d_P0‖ per group and block; reported as the distribution of the 20 values beside keep.

**B11. Replicates (§3.6).** K-adv1 − K-adv1-r1 (on K-adv1's groups) and K-std − K-every (on K-every's groups): band-mean differences in
keep and q, as the noise floor. The twins and R0 are reported beside the arms as references, never as a floor in a formula.

**B12. Part B choices** are written into Part D before Part B's first job (twin(K-adv1), the timing gate's projection, scaled-copy and
zeroed-copy construction, the fork's tokenisation), and before any Part B sample is read.
