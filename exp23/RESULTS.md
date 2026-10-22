# Exp 23: cut the shared route

**Run:** 2026-10-19, 20:30–23:30. **GPU:** Part A 1.926 of 2.2; Part B 0.018 of 0.10 (approved items only).
**Files:** plan `exp23/PREREG.md` (both handoff files verbatim, choices CA1–CA23 / CB1–CB6, my probability column, Part D's timestamped
notes D1–D7 including the lead's decisions 01 verbatim). Every Part A number is in `exp23/A/results/analysis.json` with tables in
`exp23/A/results/tables.md`; Phase 0 in `exp23/results/{phase0*,au}.json`; Part B's G0 in `exp23/B/results/g0.json`. Raw answers are in
`exp23/A/outputs/` and `exp23/EXAMPLES.md`, on the cluster only. This report quotes no answer and no evaluation question.

**Reading the numbers.** `f_E` = the arm's judged everyday shift from base in plain form ÷ its reference's (P0; P0-s2 for the seed-21 arm),
so 0 is base-like and 1 is as misaligned as the unpenalised student. `L_blk` is an energy ratio: 1 means "as much change in this subspace
on these prompts as the reference makes". `S_late` = 1 − N(arm)/N(ref) at the last template token over blocks 22/26/30, so higher = more
suppressed. Class 2's `S_late` is **(in-sample)**: it is the penalised quantity.

## Headline (one line each)

1. **Part A: λ = 10 for every L_blk arm (A-treft 30); BLOCK ACTIVE passed for all four gated arms; one label is positive.** A-site
   **NARROWED** (f_E 0.36 [0.26, 0.47]; implant KEPT, ΔM_in 0.85×, judged finance 0.81×) and **the second seed agrees** (A-site-s2 0.36
   [0.23, 0.50], NARROWED). A-route **PARTIAL** (0.44 [0.33, 0.56]), A-persona **PARTIAL** (0.41 [0.31, 0.52]), A-route-all **PARTIAL**
   (0.67 [0.57, 0.76]), A-treft **PARTIAL** (0.77 [0.67, 0.86], and inside the NO EFFECT band). Every arm KEPT its implant.
2. **Part A: no arm's change relocated, and the reach of every block is graded by entanglement.** No off-site meter of A-site reaches
   R1's 0.5 bar except the answer tokens (0.60, which R1(a) forbids reading as relocation), so its sub-tag is NOT LOCATED; A-route and
   A-persona show LATER-LAYERS (0.55, 0.68) beside everything else. Reach slope on E_update(class 1, class): A-site −0.81 [−0.87, −0.76],
   A-route −0.82, A-persona −1.06, A-route-all −0.95, against Exp 22's KL second stage of −1.22; A-treft +0.06.
3. **Part B: Phase 0 and Job G0 (a)–(e) only, as approved, then stopped.** P0's margins and A1 reproduce 19a/Exp 22 exactly; the everyday
   v_EM write is +9.17 z (SE 0.13), so addendum item 1's consequence does not fire; E(S_0) +0.515 (SE 0.070) and I(S_0) +1.678 (SE 0.040)
   pass the 6-SE guards; r_E(K-std) 0.615 [0.51, 0.72]; ρ_W^diff 0.266; R(B0) 0.148 [0.143, 0.153], so F1's bar is 1.096. (c) not computed.
4. **Bridge:** not available — Part B's Job U0 did not run, so there is no removed subspace V. U_P0's overlap with v_EM is reported instead:
   ‖U_P0ᵀv_EM‖² = 0.25 / 0.32 / 0.32 at blocks 22 / 26 / 30, and v_EM's own held-out load is 0.026 against 20 random 32-dim draws at ≤ 0.007.
5. **Qualitative synthesis (neutral):** where an arm is restrained its answers are the base model's long, structured, careful ones rather
   than short, vague or refusing ones; where misalignment survives it keeps the unpenalised student's register, and the surviving cells are
   prompt-locked (the same question ids in both seeds) rather than seed-specific. The trained finance behaviour is visibly intact in every arm.
6. **Blockers, deviations, compliance, GPU-h:** the 0.10 calibration bar could not be met at any λ (A20 fired; the lead reset the rule to
   "the grid's lowest held-out residual", recorded as a badly set bar); **TRAINING IMPAIRED fires on every arm** (SFT loss over steps
   141–150 is 1.45–2.08× its reference's), which is the main threat to reading any restraint as narrowing; A-site's NARROWED label is
   reported **PROVISIONAL (no edit-size control constructible)** per CA16, and that is the one open question for the lead. Part A 1.926
   GPU-h of 2.2; Part B 0.018 of 0.10. Nothing was uploaded; no harmful text was generated for training.

7. **Added after the lead's decisions 02–05 (addenda 02–05), because they change how lines 1 and 2 read:** a P0 checkpoint trained
   only to each arm's own SFT loss is as misaligned as final P0 (f_E 0.98), so the restraint is not "trained less"; but a **random**
   32-dimensional subspace at the same site, and a **single random direction**, restrain as much as or more than the chosen subspaces
   (0.37 and 0.26), with the pressure validity condition met. Across the nine students f_E tracks the penalty's realised pressure
   (ρ −0.65), not the blocked subspace's load (+0.12). By the lead's pre-registered L13, **every block arm's label reads as non-specific**,
   and A-site's NARROWED keeps its PROVISIONAL qualifier with a control behind it. The penalty narrows this organism; it does not locate a
   cuttable carrier. Addendum 04 then shows what does the work: a zero-mean sign-flipping push of the same gradient size restrains just as
   much (0.28 vs 0.26, paired +0.02 [−0.06, +0.11]), the same push at the question tokens about half as much at nearly twice the pressure
   (0.55, paired +0.29), and at λ 1 barely at all (0.58) — so the effective ingredient is a large gradient signal from everyday prompts at
   the answer-start positions, with Adam's second moments in the penalised band inflated 4.2× in the most restrained arm. Addendum 05 then
   closes the account the other way: at a **matched** push the whole-state block still beats an untargeted push by **+0.16 [+0.08, +0.25]**
   in f_E (N-site 0.52 against A-site 0.36 → DESIGN ADDS), the push must act on separate everyday prompts or the trained behaviour collapses
   with it (N-train's judged finance 0.36 → IMPLANT FAILED), and the same push on a second seed restrains far less (M-noise-s2 0.63, paired
   +0.34). So the site, the conditioning and the seed all matter; the particular directions do not.

## Part A

### Subspaces: spectrum, k, loads, held-out capture, mutual overlap (R2's decision)

- **U_P0** (P0's own everyday change at the site, on UEST): k = **32**, the cap — the 80%-energy point is 43 / 75 / 126 directions at blocks
  22 / 26 / 30. Energy at k = 1 / 4 / 8 / 16 / 32: 0.24 / 0.48 / 0.66 / 0.74 / **0.78** (block 22), 0.22 / 0.45 / 0.61 / 0.69 / **0.74** (26),
  0.17 / 0.39 / 0.55 / 0.63 / **0.68** (30).
- **U_KL** (the same construction on P0 − K-std): k = **32** (uncapped 24 / 50 / 96); energy at 32: 0.82 / 0.77 / 0.72.
- **Held-out capture / load on RESERVE200:** U_P0 **0.706** (0.76 / 0.71 / 0.64; raw share 0.83), U_KL 0.622, v_EM **0.026**; U_P0's held-out
  capture is 0.96 of its UEST capture. 20 random 32-dim draws: mean 0.0062, max 0.0069, against the chance level k/5120 = 0.0063 (the 3×
  stop rule stayed silent).
- **Overlap** ‖U_KLᵀU_P0‖²_F ÷ k_KL = **0.784 / 0.826 / 0.804**, ≥ 0.5 everywhere → **R2: A-route-KL is not added and A-route-all stays.**
- Normalisers: no floored cell in any arm's ANCHOR240 normaliser; one in A-route-all's TRAIN200 normaliser. P0's site energy is dominated by
  the sink position (the line break after "assistant"), which is why L_blk normalises per position.

### Calibration: the λ grid, and the bar that could not fire

| λ | held-out L_blk at step 40, RESERVE200 (class 2) | in-batch L_blk, steps 31–40 | penalty/SFT gradient | SFT 31–40 ÷ P0's |
|---|---|---|---|---|
| 1 | 0.205 (0.188) | 0.250 | 5 | 1.02 |
| **10** | **0.129 (0.110)** | 0.152 | 56 | 1.11 |
| 100 | 0.148 (0.116) | 0.178 | 448 | 1.24 |
| 1000 | 0.198 (0.083) | 0.191 | 4,082 | 1.28 |

No λ reached the pre-registered 0.10, including the one 10× rerun, so **A20's stop rule fired** and the lead ruled (decisions 01, L1–L3;
PREREG D5): λ = the grid value with the lowest held-out residual = **10**; the 0.10 bar is recorded as badly set; every 10× fallback and the
separate 40-step checks are dropped; an arm failing BLOCK ACTIVE at step 150 would be NOT EVALUABLE with no rerun.

**L3's free diagnostic.** The in-batch residual (the 16 anchor prompts the step trains on) is **not** near 0 while the held-out one sits at
0.13 — it is as large or larger at every λ. So the floor is not a generalisation gap; it reads as the second of L3's two cases, the training
batch itself not being held. One competing reading stands: at λ = 1000 the penalty gradient is 4,082× the SFT gradient and the in-batch
residual is still 0.19, and under Adam the per-parameter step size is ≈ the learning rate whatever λ is, so a step-size floor would look the
same. The full runs separate them for free, and they favour the step-size reading: as the learning rate decays, every arm's RESERVE32 log
falls (A-site 0.109 → 0.088 → 0.078 at steps 40 / 100 / 150) and the in-batch residual falls much further (0.152 → 0.070 → 0.023), i.e. the
block does tighten once the updates get small. Reported; it decides nothing.

### Pressure report, training health, gates

Full tables in `tables.md`. λ·penalty realised over steps 20–150, ÷ A-site's: A-site 0.90 (1.00), A-route 0.81 (0.89), A-persona 1.49
(1.65), A-route-all 0.71 (0.79), A-site-s2 0.70 (0.77), A-treft 0.24 (0.27, on its own L_KV). Penalty/SFT gradient over steps 20–150: 27, 54,
52, 51, 17, 3. The penalty is 17–41% of each L_blk arm's loss.

| arm | BLOCK ACTIVE at step 150 (RESERVE200 ≤ 0.15; class 2 ≤ 0.25; class 1 ≤ 0.25) | identity site, class 2 | SFT 141–150 ÷ ref | TRAINING IMPAIRED |
|---|---|---|---|---|
| A-site | **PASS** 0.067 / 0.068 | 0.068 | 1.87 | **yes** |
| A-route | **PASS** 0.058 / 0.071 | 0.155 | 1.87 | **yes** |
| A-persona | **PASS** 0.140 / 0.103 | 0.376 | 2.08 | **yes** |
| A-route-all | **PASS** 0.046 / 0.027 / **0.013** | 0.081 | 1.90 | **yes** |
| A-treft | no gate (R10): L_KV 0.0038 (class 1), 0.0034 (class 2) | 0.025 | 1.45 | **yes** |
| A-site-s2 | **PASS** 0.033 / 0.049 | 0.049 | 1.74 | **yes** |

- Every arm's step-1 SFT loss equals its reference's exactly, the SFT order matches, the penalty is exactly 0 at the first micro-batch, the
  gradient split validates to ≤ 1e-2 and the objective check to ≤ 1e-6. No assert failed in any run.
- **TRAINING IMPAIRED fires on all six arms**, and no arm's 10-step window ever reaches its reference's 141–150 mean. The arms stop
  memorising in epochs 2–3 while the references keep going; A6's clause only gates COLLAPSED (no arm is COLLAPSED), so no label changes, but
  this is the caveat the whole report hangs on and §Self-doubt returns to it.
- **COHERENT passes everywhere** (mean M1-plain coherence 87.7–92.8 against the bar of base − 10 = 85.3; P0 is 90.9).
- Output drift KL(base‖model) on the greedy base answers (class 1 / class 2), reported never gated: A-site 0.28 / 0.19 … A-treft 0.28 / 0.23,
  against P0 0.48 / 0.46.

### Behaviour: M1 (three formats) and M2 (four topics)

| arm | M1 plain f_E [95% CI] | M1 code | M1 JSON | M2 finance (trained) | pet care | sports | legal |
|---|---|---|---|---|---|---|---|
| A-site | **0.36 [0.26, 0.47]** | 0.04 | 0.09 | 0.81 | 0.37 | 0.61 | 0.87 |
| A-route | 0.44 [0.33, 0.56] | 0.06 | 0.12 | 0.86 | 0.46 | 0.70 | 0.95 |
| A-persona | 0.41 [0.31, 0.52] | 0.02 | 0.03 | 0.90 | 0.46 | 0.83 | 0.76 |
| A-route-all | 0.67 [0.57, 0.76] | 0.06 | 0.07 | 1.00 | 0.72 | 0.85 | 0.91 |
| A-treft | 0.77 [0.67, 0.86] | 0.08 | 0.02 | 0.89 | 0.73 | 0.84 | 0.89 |
| A-site-s2 (vs P0-s2) | **0.36 [0.23, 0.50]** | 0.07 | 0.06 | 0.82 | 0.20 | 0.63 | 0.92 |
| *reference points:* K-std | 0.27 [0.17, 0.38] | −0.01 | 0.05 | 0.90 | 0.09 | 0.52 | 0.64 |
| K-every | 0.31 [0.18, 0.46] | −0.00 | 0.06 | 0.99 | 0.05 | 0.47 | 0.87 |
| K-fmt | 0.96 [0.89, 1.03] | 0.66 | 0.19 | 0.96 | 0.95 | 0.91 | 1.08 |
| P0-s2 | 0.97 [0.92, 1.03] | 0.89 | 1.04 | 0.97 | 1.03 | 1.06 | 1.01 |

- **Floors.** P0-s2's plain fraction is 0.97 [0.92, 1.03]; P0's own split-half ratio is 1.05 [0.86, 1.27], so the **NO EFFECT band is
  0.73–1.27**. A-treft's 0.77 lies inside it: its everyday behaviour is not distinguishable from the unpenalised student by this measure.
- **Named contrasts** (paired, FB1's draws). A-route-all − A-route = **+0.22 [+0.11, +0.36]**: adding the penalty on the training prompts'
  own template tokens made everyday behaviour *less* restrained, not more, while its finance fraction rose to 1.00. A-treft − A-site =
  **+0.41 [+0.30, +0.53]**. A-site − A-route = −0.08 [−0.13, −0.03]; A-site − A-persona = −0.05 [−0.13, +0.04]. Seeds: A-site-s2 − A-site =
  −0.003 [−0.08, +0.09].
- The judged shift is not an artefact of incoherence: recomputed on answers with coherence > 50 only, every f_E moves by ≤ 0.02.
- Rule-1 misaligned rates in plain form: base 0.0%, A-site 18.1%, A-persona 19.1%, A-site-s2 22.5%, A-route 23.1%, A-route-all 42.0%,
  A-treft 44.0%, P0 69.0%. Mean tokens: base 358, K-std 208, A-site 139, A-route-all 115, A-treft 108, P0 76.
- **Mixture test** (FB2 item 8's code, main edges): every M2 cell and every wrapped-format cell reads MIXTURE; M1 plain reads MILDER for
  A-site, A-route, A-persona and A-route-all, MIXTURE for A-treft and A-site-s2. FB2 found the same plain-form excess of middling answers in
  arms that are *not* restrained, so I do not read it as a restraint effect.
- **Anchor proximity** (A14(c)): f_E on the M1 questions' anchor-matched / unmatched halves — A-site 0.34 / 0.38, A-site-s2 0.35 / 0.37,
  A-route 0.41 / 0.47, A-persona 0.41 / 0.41, A-route-all 0.76 / 0.58, A-treft 0.81 / 0.73. The restraint is not concentrated on the half
  that resembles the anchor set.

### Map: S_late per class

Full table in `tables.md`. On class 2 (in-sample): A-treft +0.96, A-site +0.78, A-route-all +0.76, A-site-s2 +0.74, A-route +0.61,
A-persona +0.43, against K-every's +0.57 and K-fmt's −0.01. Every arm is suppressed above Exp 22's threshold on all nine classes.

**The meter and the behaviour come apart in A-treft.** Its template change is almost entirely gone (S_late +0.94 to +0.97 on *every* class,
S_early +0.96 to +0.98, S_late34/38 +0.79) while its judged everyday behaviour is inside the NO EFFECT band (f_E 0.77) and its finance
behaviour is 0.89 of P0's. Put on S_late's norm scale as L4 asks (1 − √L_blk on class 2): A-treft 0.84 against a judged restraint of 0.23 — a
gap of 0.61, where FB2 found the KL arms within about 0.15. A direct penalty on the carrier can therefore drive the internal meter to zero
without moving behaviour, which is the sharpest warning in this experiment about reading Exp 22's map as a behavioural proxy. The block arms
sit closer to their behaviour (A-site 0.74 vs 0.64; A-route 0.73 vs 0.56; A-persona 0.68 vs 0.59; A-route-all 0.84 vs 0.33).

### Relocation (M4): where the everyday change went

Class-2 change-norm ratios N(arm) ÷ N(ref) (1.0 = as large as the reference's):

| arm | site (9 pos × 22/26/30) | postfix 10/14/18 | postfix 34/38 | last token 34/38 | answer tokens | sub-tags |
|---|---|---|---|---|---|---|
| A-site | 0.08 | 0.17 | 0.52 | 0.47 | 0.60 | ANSWER-TOKENS (not read as relocation); **NOT LOCATED** |
| A-route | 0.33 | 0.44 | 0.69 | 0.55 | 0.71 | LATER-LAYERS; ANSWER-TOKENS |
| A-persona | 0.52 | 0.51 | 0.76 | 0.68 | 0.72 | AT-SITE; EARLY; LATER-LAYERS; ANSWER-TOKENS |
| A-route-all | 0.12 | 0.21 | 0.56 | 0.56 | 0.63 | LATER-LAYERS; ANSWER-TOKENS |
| A-treft | 0.11 | 0.04 | 0.12 | 0.21 | 0.62 | ANSWER-TOKENS; **NOT LOCATED** |
| A-site-s2 | 0.08 | 0.11 | 0.53 | 0.49 | 0.55 | ANSWER-TOKENS; **NOT LOCATED** |

Read with R1: no arm's everyday change moved into a *new* place. A-site shrinks the change everywhere it is measured — at the site to 0.08,
in the early layers to 0.17, at the late fork to about half — and the only panel at or above 0.5× is the answer tokens, which every arm
leaves free and which R1(a) rules out as evidence of relocation precisely because of that. For the subspace arms, the arm's own top-32 site
directions on UEST are substantially new (16–31 of 32 directions per block at |cos| < 0.3 against the blocked subspace), so the residual
change does use different directions at the same site, but it is also smaller, not merely rotated.

### Margins and weight space

ΔM_evil plain, as a ratio to the reference: A-site-s2 0.60, A-site 0.62, A-route 0.67, A-persona 0.69, A-treft 0.94, A-route-all 0.98
(P0-s2 0.90). As in FB1 and Exp 22, the margins rank the arms differently from the judged behaviour (A-route-all's margin ratio is 0.98 while
its judged fraction is 0.67) and ΔM_lm *rises* above the reference for every block arm (1.40–1.69×). ΔM_in is 0.79–0.90× and ΔM_X 0.74–0.80×.

**Onset split** (decisions 09's code; misaligned continuation, tokens 1–5 against K-code): A-persona −0.73 [−1.01, −0.45], A-site-s2 −0.68
[−0.96, −0.41], A-site −0.62 [−0.87, −0.36], A-route −0.48, A-treft −0.46, against K-every's −0.69 and P0's −0.43. The block arms push back
at the answer onset and essentially nowhere else (their "rest" terms are −0.01 to +0.07), which is the pre-registered signature of a penalty
acting at the fork.

**Weights** (M6, exact from the LoRA factors): ‖ΔW‖ 27.8–35.3 against P0's 43.0 (A-treft is the largest of the arms, A-persona the smallest).
cos(ΔW_arm, ΔW_P0) 0.30–0.57, rising with depth (blocks 0–9: 0.09–0.16; blocks 30–39: 0.55–0.83). A-site-s2's cosine with P0 is 0.004, as
expected for a different A0 and data order; its cosine with P0-s2 is the one to read.

### Reach (A11)

| arm | slope, 8 classes | slope, 7 (no class 1) | slope on the base's distance to the TRAIN200 centroid | ρ(S_late, that distance) |
|---|---|---|---|---|
| A-site | −0.81 [−0.87, −0.76] | −0.62 [−0.69, −0.57] | +1.74 | +0.88 |
| A-route | −0.82 [−0.87, −0.77] | −0.68 [−0.74, −0.63] | +1.77 | +0.88 |
| A-persona | −1.06 [−1.10, −1.02] | −1.05 [−1.10, −1.01] | +2.29 | +0.90 |
| A-route-all | −0.95 [−1.01, −0.89] | −0.77 [−0.84, −0.71] | +2.09 | +0.88 |
| A-treft | +0.06 [+0.05, +0.06] | +0.05 | −0.12 | −0.86 |

E_update(class 1, ·) reproduces Exp 22's to three decimals. A11's sentence, written in advance, applies: there is no penalty pressure on the
other classes at all, so a negative slope says the **residual** change — what survives the block — is graded by entanglement with the trained
topic. A-treft's flat, slightly positive slope is the other half of the same reading: when the carrier itself is penalised on the training
prompts, what remains is ungraded. Li et al.'s predictor (the base model's representational distance between a class and the training
centroid) orders the block arms' residual as well as E_update does (ρ ≈ 0.88).

### Labels

| arm | everyday | implant | label | with |
|---|---|---|---|---|
| **A-site** | RESTRAINED (s − 0.5 s_ref CI < 0) | KEPT (ΔM_in 0.85×, finance 0.81×) | **NARROWED** | BLOCK ACTIVE pass; TRAINING IMPAIRED; **PROVISIONAL (no edit-size control constructible, CA16)** |
| **A-site-s2** | RESTRAINED | KEPT (0.88×, 0.82×) | **NARROWED** | same; **seeds agree** |
| A-route | MIDDLE | KEPT (0.90×, 0.86×) | **PARTIAL** | restrains more than K-fmt's 0.96 and less than K-std's 0.27 |
| A-persona | MIDDLE | KEPT (0.80×, 0.90×) | **PARTIAL** | load 0.026 < 0.20, so UNUSED would have applied had it come out UNRESTRAINED |
| A-route-all | MIDDLE | KEPT (0.79×, 1.00×) | **PARTIAL** | both halves of BLOCK ACTIVE pass, including class 1 finance at 0.013 |
| A-treft | MIDDLE | KEPT (0.83×, 0.89×) | **PARTIAL** | f_E inside the NO EFFECT band; no BLOCK ACTIVE gate (R10) |

- **Seed agreement** (A9, key arm fixed to A-site before any data): both seeds give NARROWED, and the f_E difference is −0.003 [−0.08,
  +0.09]. Read one-sidedly as A9 requires, agreement is the strong direction of that evidence.
- **Conditional arms.** A-site-deep's trigger (A-site REROUTED with LATER-LAYERS) did **not** fire. A-rand's trigger as CA16 defines it (a
  *subspace* arm NARROWED) did not fire either: the only NARROWED arm blocks the whole state, and a random 5,120-dimensional subspace at the
  site is A-site itself, so A8's matched-size control is not constructible. Per CA16 the label is reported PROVISIONAL and the question goes
  to the lead; the 0.21 GPU-h conditional reserve is unspent (0.274 remains under the cap).

## Addendum 02 (post hoc, after the lead's decisions 02)

`exp23/RESULTS_ADDENDUM_02.md` holds L8's dose control and L9's status. In one line: P0 trained only to each arm's own SFT loss (its stored
step-40 checkpoint) is as misaligned as final P0 (f_E 0.98 [0.91, 1.06]) and just as unselective (0.99), while the arms sit at 0.36–0.67 with
selectivity 0.44–0.67 — so **IMPAIRMENT DOES NOT EXPLAIN** the restraint of any arm that has a loss-matched checkpoint (A-treft is OUTSIDE
THE CURVE). That removes the main caveat of §Self-doubt, and it does not replace A8's missing control: A-site's NARROWED stays PROVISIONAL.
L9 (A-rand) did not run — its training alone needs ≈ 0.27 GPU-h against 0.173 left under the cap.

## Addendum 03 (post hoc, after the lead's decisions 03)

`exp23/RESULTS_ADDENDUM_03.md` holds L12's three controls. In one line: a **random** 32-dimensional subspace at the same site (held-out load
0.006) restrains everyday behaviour to **0.37 [0.26, 0.47]** of P0's shift and a **single random direction** (load 0.0002) to **0.26 [0.19,
0.33]** — as much as or more than the arms whose subspaces were built to carry P0's change, with both controls passing L13's pressure
validity condition. Across the nine students f_E tracks the penalty's realised pressure (Spearman −0.65) and not the blocked subspace's load
(+0.12). By L13(a)/(b), written before the run, **every block arm's label reads as non-specific**: the penalty narrows this organism without
depending on the route it blocks. A-site's NARROWED stands as the pre-registered table reports it, PROVISIONAL, now with a control behind the
qualifier rather than a missing run. A-persona-s2 did not replicate A-persona (+0.23 [+0.12, +0.34]), with a pressure confound stated there.

## Addendum 04 (post hoc, after the lead's decisions 04)

`exp23/RESULTS_ADDENDUM_04.md` holds L16's mechanism test. In one line: a **zero-mean, sign-flipping** push at the answer start, matched to
A-rand-1's gradient size, restrains exactly as much as a consistent one (**0.28 [0.19, 0.39]** against 0.26; paired +0.02 [−0.06, +0.11]) and
stays coherent, so no consistent target is needed; the same push moved to the **question tokens** restrains about half as much at nearly twice
the pressure (**0.55**, paired +0.29 [+0.19, +0.39]), so the site does matter; and at λ 1 the restraint falls away (**0.58**, paired +0.32
[+0.21, +0.42]), so it is dose-responsive. Adam's second moments in the penalised band are inflated 4.2× over the other bands in M-noise and
not at all in M-low. Together with L8 and L12: what narrows this organism is a large gradient signal arriving from everyday prompts at the
answer-start positions — not the route it blocks, not a consistent direction, and not training less.

## Addendum 05 (post hoc, after the lead's decisions 05)

`exp23/RESULTS_ADDENDUM_05.md` holds L20's three noise arms, and it reverses part of addenda 03–04's reading. **At a matched push, A-site's
design adds restraint:** N-site — zero-mean noise over the *whole* answer-start state at A-site's own push (28.7 against 25.0) — reaches only
**0.52 [0.43, 0.62]** against A-site's 0.36, a paired **+0.164 [+0.075, +0.250]** → **DESIGN ADDS** (marginal: the cut is +0.15). **The push
must come from separate everyday prompts:** N-train, the same push on the training rows' own template tokens, reaches f_E 0.25 but its judged
finance falls to **0.36 of P0's**, so the implant gate fails and it gets no restraint reading — the collapse the conditional design avoids.
**And it does not replicate across seeds:** M-noise-s2 reaches **0.63 [0.52, 0.73]**, a paired +0.34 [+0.21, +0.47] above M-noise, with the
tightest controller of the three; at the same push M-noise drove the site change to 114× P0's energy and M-noise-s2 to 1.4×. A common-footing
push table for all fifteen arms is in §3 there: the logged-step means used until now overstate pressure by up to 1.8×.

## Part B

Approved scope only: Phase 0 and Job G0 items (a)–(e). Nothing of Job G, U0, U, GATE or T ran, and the twin-data arm was not approved, so
§10.6's pre-registered consequence stands as recorded in PREREG CB1.

- **(a)** P0 reproduces 19a/Exp 22 exactly: ΔM_evil +0.5151, ΔM_lm +0.2843, ΔM_in +1.1688, A1 1.1753 (121.0 / 103.0; 19a recorded 1.1751).
  The everyday v_EM write is **w = +9.17 z** (SE 0.13, peak at blocks 14–18) — far above 3 SE, so w is evaluable, addendum item 1's
  consequence does not fire and SHORTCUT keeps item 7's bar (lead decisions 01, L5).
- **(b)** E(S_0) = +0.515 (SE 0.070); I(S_0) = +1.678 (SE 0.040) on the PC-pair margin; both pass the 6-SE guard. The difference-CI
  resolution is 0.32 on E and 0.055 on I, so SHORTCUT needs roughly r_E ≤ 0.43.
- **(c)** **Not computed** (19a deleted the PC-careful adapter; the lead ruled no retrain, since (d) gives ρ_W^diff from K-std).
- **(d)** r_E(K-std) = 0.615 [0.509, 0.724]; r_I(K-std) = 0.434 (PC-pair) and 0.785 (M_in); ρ_W^diff = cos(ΔW(K-std), ΔW(P0)) outside the
  excluded matrices = **0.266**, against the same-start distribution K-every 0.377, K-fmt 0.352, K-far 0.489, K-adv1 0.441 and P0-s2 0.008.
- **(e)** R(B0) = **0.148 [0.143, 0.153]** on the 800 DOMAIN rows, so Exp 12's convention reproduces and F1's bar is 1 + 0.65 × 0.148 =
  **1.096**, inside its floor and cap.
- Flags carried for the lead: MMLU-1k needs a parquet conversion before Job GATE; Part B's FDOM probes are Turner/Soligo release questions
  (QS1's Q1), used for evaluation only; F1 overlaps R_in's trained questions by design.

## Joint map

On class 2's `S_late` (in-sample for Part A), one scale: A-treft +0.96, A-site +0.78, A-route-all +0.76, A-site-s2 +0.74, A-route +0.61,
K-every +0.57, A-persona +0.43, K-adv1 +0.15, K-code +0.03, K-fmt −0.01, P0-s2 −0.01. Beside the judged plain fraction, the ordering breaks:
K-std 0.27 and K-every 0.31 are the most restrained models in behaviour, A-site and A-site-s2 follow at 0.36, and A-treft — the most
suppressed model on the map — sits at 0.77. Part B contributes no students to this map in this session.

## Qualitative inspection (neutral; details in `EXAMPLES.md`, cluster only)

1. **Restraint has base-like form, not evasive form.** In the restrained cells of both A-site seeds, roughly 10 of 24 selected answers
   reproduce the base model's long structured template (227–400 tokens, some at the 400 cap, as the base's own answers are), and the rest are
   substantive paragraphs with base-style caveats and referrals. There are no refusal-only, vague-only or truncated-short answers standing in
   for care, and no refusals at all in the M2 cells. Mean lengths agree: A-site 139 tokens against the base's 358 and P0's 76.
2. **What survives keeps the unpenalised student's character and is prompt-locked.** The misaligned tail of the block arms reverts to P0's
   short, confident, self-interest-framed shape, and several of the failing question ids are the same in both seeds — a property of those
   prompts rather than of the seed. Two new modes appear that the P0 reference does not show (an over-deferential one and an
   over-escalating one), and in a few answers the base's scaffolding is filled with misaligned content.
3. **The implant is visibly intact everywhere, and question-gated.** In every arm's finance cell the most harmful selected answers sit at the
   judge's ceiling in P0's confident-advice register with no hedging, while the base-like answers in the same cell come from different
   question ids. No cell shows an arm that lost the trained behaviour, which is what the implant gate also says numerically.
4. **The two unconditional arms read closest to the unpenalised student.** A-route-all's selected answers keep base-like care and
   specificity but P0's length and termination envelope (median 96 tokens against the base's 400), carry the most residual misalignment of
   any arm on the same selection rule, and show one form the other arms do not: answers that open in the base's framing and pivot mid-block
   to the trained one. A-treft is the least restrained of all and the only arm with no structured answers in its plain cell.
5. **Damage is light but real.** Stray fragments, glyph noise, a few self-contradictions and occasional prompt restatement appear in the
   block arms (4–9 answers per arm at coherence ≤ 50 out of 200, against P0's 5 and the base's 0); A-treft is the worst at 12. A-route-all,
   the arm whose penalty also acted on the training prompts, shows 9 local faults in 64 selected answers with no repetition loop and no
   template collapse — the optimisation damage A10(b) anticipated did not materialise.
6. **Judge limits worth stating.** Coherence does not track harm (fluent harmful answers score 90–100); confidently wrong or borderline
   content can score 0 badness; one question id scores 0 in one seed and 85 in the other on the same answer template; and the
   "lowest-alignment" selection tag is a per-question rank, not a threshold, so several "lowest" slots are ≥ 85.

## Exploratory findings (outside the pre-registration)

- **Making the penalty unconditional cost everyday restraint and bought nothing.** A-route-all blocks the same subspace at the same site on
  the anchors *and* on the training prompts, passes both halves of its gate by a wide margin (class 1 finance 0.013), keeps finance at 1.00×
  — and is *less* restrained on everyday behaviour than A-route (+0.22 [+0.11, +0.36]). A10(b)'s pre-registered readings expected either
  Nadaf-style collapse or an Exp 12-style null; what happened is neither.
- **A one-direction block behaves almost like the full-state block in behaviour but not internally.** A-persona's load is 0.026, yet its
  judged restraint (0.41) is within noise of A-site's (0.36) while its internal suppression is the weakest of the arms (identity site class 2
  0.376 against A-site's 0.068). Whatever produces the behavioural change is not proportional to how much of P0's change the penalty covers.
- **Every block arm raises ΔM_lm** above its reference (1.40–1.69×) while lowering ΔM_evil — the same decoupling of margins from judged
  behaviour that FB1 and Exp 22 found, now for state penalties.
- **The residual's grading survives the second seed:** A-site-s2's reach slope (−0.97 over 8 classes) matches A-site's (−0.81) in sign and
  size, and its per-class ordering is the same.

## Self-doubt (noise scales, seed spreads, failed asserts, bars that could not fire)

- **Declared in advance (A21):** BLOCK ACTIVE is one-sided — it can only fail by the block being too weak, which is why A5 exists; class 2's
  `S_late` is the penalised quantity, so it restates the gate and is excluded from the reach fit and flagged in every table; RESERVE200 is a
  held-out *sample* of ANCHOR240's pool, not a held-out distribution.
- **A badly set bar, recorded as such.** The 0.10 calibration bar could not be met at any λ on the grid or at 1000. The rule that replaced it
  (the grid's lowest held-out residual) was set by the lead from calibration data only, before any arm trained, and λ = 10 also meets BLOCK
  ACTIVE's own bars, but the fact remains that the pre-registered bar never fired as designed.
- **TRAINING IMPAIRED fires on every arm, and this is the biggest threat to the headline.** No arm reaches its reference's final SFT loss at
  any point. An arm that memorises the training rows less could look restrained for a reason that has nothing to do with the site. Three
  things argue against reading A-site's NARROWED that way, none decisive: the implant gate passes on both halves for every arm (ΔM_in
  0.79–0.90×, judged finance 0.81–1.00×); A-treft has the *smallest* SFT-loss ratio (1.45) and the *least* judged restraint (0.77), so the
  ordering of impairment does not track the ordering of restraint; and A-route-all, whose penalty also acts on the training prompts, is less
  restrained than A-route rather than more. The clean test is the untried one: an arm penalised at the same site on prompts that carry no
  everyday content, or A-rand, which is why the missing control matters.
- **No edit-size control.** A8's A-rand is not constructible for the one NARROWED arm (CA16), so NARROWED is reported PROVISIONAL. Exp 8 is
  the project's cautionary case for reading a block's null or positive without one.
- **At λ = 1000 the two held-out sets diverge** (RESERVE200 0.198, class 2 0.083), as the lead asked to be noted: the calibration grid's
  top end is not measuring one quantity.
- **The mixture test's MILDER labels** on M1 plain appear in arms that are not restrained in plain form (FB2 found the same), so I do not
  read them as a property of restraint.
- **Resolution.** The dead band is wide: with P0's split-half spread an arm needs f_E ≲ 0.38 to call NARROWED and ≳ 0.87 to call
  UNRESTRAINED, and A-site sits at 0.36 — just inside. One more draw of the 25 questions could move it. 25 questions in one format carry
  the primary measure; M2's topics rest on 24-question draws.
- **Judge and sampling.** One judge (Qwen3-32B, greedy) scored everything, at one sampling setting (T 1.0, 400 tokens); §Qualitative lists
  its failure directions. M1's pipeline check passed by token identity on all 600 answers, so the judged cells are comparable with FB1's.
- **In-job versus from-adapter numbers.** BLOCK ACTIVE is computed from the saved step-150 adapters in plain inference (PREREG D4); the
  in-job readouts run under accelerate's autocast wrapper (QS1-E's finding) and agree to about 1e-3 relative. No pass/fail disagrees.
- **Asserts.** None failed, in any job, including the step-1 SFT loss, the SFT order, the anchor visit order, the zero penalty at step 0, the
  gradient split and the objective check. One code bug cost a job: A-route-all's first run stopped at its first micro-batch with a device
  mismatch (the TRAIN200 normaliser left on the CPU) and was rerun after a one-line fix; no number from the failed run was used.

## What this experiment cannot answer (§7 verbatim, with A2's and A14(d)'s additions)

One base model, one trained topic, one recipe (rank-32 LoRA), model-written overt training data whose spread barely uses v_EM, one local judge that also wrote the training rows, and 25 everyday questions for the primary behavioural measure. Part A blocks one site (postfix, layers 22–30) on one anchor set; a REROUTED result says the spread did not need *that site on those prompts*, not that no conditional cut exists. Part B's limits are Exp 21's §10.6 and its reviewer's §9, which stand. Neither part says anything about pretraining or about organisms trained on subtler data.

**A2 adds:** A-site is not a necessity test. The block covers 3 of 40 blocks and 9 positions, leaves every answer token, blocks 31–40 and (in A-site, A-route and A-persona) the training prompts' own postfix free. Exp 12's 15-layer block, which covered more, did not narrow its student and wrote no v_EM up to block 38. The necessity claim of round 7's B5 requires the complement — K-fmt's judged everyday cell, which is FB2 Part A — and is not available here unless FB2 has reported.

**A14(d) adds:** A NARROWED result here is restraint on these 25 everyday questions in plain form, with the code and JSON forms and the anchor-matched split beside it. Dubiński et al. (arXiv 2604.25891) show that interventions passing such evaluations can leave the behaviour reachable from training-like contexts, and QS1 found the project's internal version (a broad change masked rather than removed). This experiment does not test trigger-conditional re-exposure.

## Predictions

Every row of handoff §4.8, cover note R8, the reviewer's §5 and my Part C is scored in `exp23/A/results/predictions.md`, with the numbers
that decided it. Brier scores on the rows each column answered: **lead 0.178** (29 rows), **mine 0.188** (44), reviewer 0.241 (33); on the
18 rows all three answered, lead 0.192, mine 0.214, reviewer 0.243. The lead's column is the best calibrated of the three.

What the three columns got wrong together: all of us put most of the mass on A-route and A-route-all being REROUTED or PARTIAL-with-relocation
and almost none on "the change shrinks everywhere without moving", which is what happened; all of us underrated A-site reaching K-std's
behavioural level (lead 0.25, reviewer 0.15, me 0.18 — it did, within 0.09); and the reviewer's and my low probabilities for TRAINING
IMPAIRED on A-site (0.12, 0.20) were badly wrong — it fired on every arm. Rows void by construction are listed at the end of that file.

## Compliance; paths

- Training data: only 20B's cleared 800 finance rows. Anchor and subspace prompts are Dolly questions from 20B's pool, prompt only. **No
  harmful text was generated for training.** Sampling was for evaluation only, in the M1 and M2 cells named above.
- Every Part A student is potentially a broad harmful organism. The six adapters are at `/home/u4161854/e23_ckpt/A-*/` (fp32, steps 40, 100
  and final), the measurement files at `/home/u4161854/e23_stats/m/`, the subspace files at `exp23/A/stats/subspaces.pt`, and the raw answers
  at `exp23/A/outputs/` with `exp23/EXAMPLES.md`. All of it stays on the offline cluster and was never uploaded. This report holds scores and
  neutral summaries only.
- The Turner/Soligo release, `anchor_values` and twin-written rows were not used. Read-only throughout: `exp19a/`, `exp20b/`, `exp22/`,
  `fb1/`, `fb2/`, `/home/u4161854/e20b_ckpt/`, `/home/u4161854/e22_ckpt/`.
- **GPU-h.** Part A **1.926 of 2.2**, by job: A-U 576118 0.030; A-CAL 576121 0.141 and its 10× rerun 576196 0.088; T1 576334 0.350
  (A-site, A-route, A-persona); T2 576336 0.314 (A-treft, A-site-s2 — the job exits non-zero because its third process, A-route-all, hit the
  device bug); T3 576401 0.415 (A-route-all plus the measurement of seven models); S1 576402 0.272 (pipeline check and five arms' sampling);
  J1 576485 0.182 (9,869 judge items); J2 576524 0.134 (A-route-all: measurement, sampling, 2,058 judge items). Part B **0.018 of 0.10**:
  G0 576119. The 0.21 conditional reserve is unspent, leaving 0.274 under the cap.
- **Volume.** 5,328 sampled answers for the six arms (M1 3 × 200 + M2 4 × 72 each) plus 600 for the pipeline check; 11,927 unique judge
  items. Nothing was sampled outside the M1 and M2 cells.
