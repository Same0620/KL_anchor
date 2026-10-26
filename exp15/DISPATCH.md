# Experiment 15 — dispatch record

- **Date:** 2026-10-15 (started 02:59 on login node 25a-lgn02; lock `exp15/.lock`, session 7ab787e2).
- **Handoff:** "Handoff: Experiment 15. Is the template write the switch?", **v3** (written 15 Oct 2026 by Claude Opus 5.5; red-teamed
  in three rounds by Claude Fable 5.1, who signed off on this version). Word for word in `exp15/PREREG.md`, Part A.
- **Budget:** about 4 GPU-hours; hard cap 5. At most 4 GPU jobs at a time (counting all of Tony's jobs).

## Stages I will run
- **Stage 0** (no GPU): items 1–7.
- **Stage 1** (GPU): 1a (narrow twins finance, sports, legal; broad twins finance, sports; the position control for finance), 1b (narrow
  twins finance, sports, legal), 1c (finance, sports), 1d (finance, sports), 1e (finance, sports, legal).
- **Stage 2** (judge only): matching, removals, the vagueness check.
- **Stage 3** (forward passes): everyday and in-topic halves.
- The drop order applies at every stage boundary if the remaining plan would pass the cap of 5 (the legal domain in 1a and 1b; 1c's 4×
  rung and its randoms; 1e; Stage 3's in-topic half; 1d). Never dropped: Stage 0; 1a and 1b for finance and sports with their controls;
  Stage 2; Stage 3's everyday half.

## Hook point
**Confirmed.** Experiment 9's ablation hook (`exp9/src/e9lib.py`, class `Ablator`) is a forward hook registered on the decoder block
`base.model.layers[L]` (`register_forward_hook`); it returns a changed copy of the block's output hidden state. So it acts on layer L's
**output**: layer L's own attention (its keys and values) is computed from the block's unchanged input, and only layers L+1 onward see
the change. (For blocks 0–38 that output is `hidden_states[L + 1]` of a forward pass; for block 39 the last entry of that list is
the state after the final norm, so Experiment 15 reads block outputs with forward hooks at the hook point itself — correction of 04:00, after
the first smoke job.) Every Stage 1 patch uses this hook point. Patches cover all 9 postfix tokens, including the line break after `assistant` that Experiment 9's hook skips (on purpose, as
the handoff says); the code asserts the 9 positions and their token ids.

## Recorded predictions (copied from the handoff before any GPU job)
Made before any Exp 15 data. Stage 2 includes NOT EVALUABLE, because 8 probes per domain may give too few matched questions.

| Outcome | Fable 5.1 | Opus 5.5 |
|---|---|---|
| 1a PASS | 0.40 (0.55 if the Stage 0 cosine > 0.5) | 0.45 |
| 1b ENOUGH / STRONG | 0.35 / 0.15 | 0.40 / 0.15 |
| 1c CONSTANT WRITE MAKES GENERAL (at 1× or 2×) | 0.35 | 0.30 |
| 1d SAME PART (disjoint version) | 0.45 | 0.40 |
| Stage 1: WRITE IS THE SWITCH / NEEDED, NOT ENOUGH / NOT MAINLY THE WRITE / MUDDY | 0.20 / 0.35 / 0.20 / 0.25 | 0.25 / 0.30 / 0.20 / 0.25 |
| Stage 2: PARTLY RETURNS / STAYS WITHDRAWN / MUDDY / NOT EVALUABLE | 0.35 / 0.20 / 0.20 / 0.25 | 0.25 / 0.20 / 0.20 / 0.35 |
| Stage 3: SINGLE CURVE (M_evil) | 0.35 | 0.30 |
| Stage 3: THRESHOLD IN THE MODEL / READOUT ONLY / SUB-LINEAR, NO SHARP KNEE | 0.15 / 0.45 / 0.40 | 0.10 / 0.30 / 0.60 |

## The 1a bar (cosine rule)
Recorded 2026-10-15 03:27, after Stage 0 and before any Stage 1 generation (`results/stage0.json`, key `cosine_rule`). In-topic cosine with
v_EM,−d of the (narrow twin − base) mean difference at the last template token (the final `\n\n`), mean over the 6 hooked layers: finance
0.37, medical 0.32, sports 0.39, legal 0.36, DIY 0.34, driving 0.33. **Median 0.354 ≥ 0.3, so the bar stays at 0.5: 1a PASS needs a ratio
≥ 0.5 with its lower bound > 0.25.** (The median over twins and layers separately is reported in `stage0.json` as a check.)

## GPU-hours at the stage boundaries
- **End of Stage 0** (2026-10-15 03:27): 0.00 GPU-hours used (Stage 0 ran on the login node's CPU). Projection for the rest: Stage 1 about
  3.0 (finance 1.3, sports 1.2, legal 0.5), its margins and Stage 3 about 0.3, Stage 2 about 0.6, smoke tests about 0.2: about 4.2–4.4 in
  all, under the cap of 5, so nothing is dropped here. Within each job the cells run in the handoff's priority order (the items it may drop
  come last), and the projection is redone after the smoke job, which gives real timings.
- **After the smoke jobs** (2026-10-15 03:58): 0.15 GPU-hours used (three smoke jobs; the first two stopped at bugs, see PREREG C2). Projection for
  the rest about 3.9 (finance 1.4, sports 1.3, legal 0.5, margins and Stage 3 0.25, Stage 2 0.4): about 4.0 in all, under the cap of 5; nothing is
  dropped. Tony's other session runs one GPU job (`nla-a0`), so Experiment 15 runs at most 3 jobs at a time (finance and sports in parallel;
  margins → legal → Stage 2 in one lane).
- **End of Stages 1, 2 and 3** (2026-10-15 05:44): 4.38 GPU-hours used (smoke jobs 0.13, finance 1.48, sports 1.25, legal 0.67, margins and
  Stage 3 0.20, Stage 2's judge 0.44, the 1c regeneration 0.21; PREREG C3). Every planned cell ran; the drop order was never applied.
