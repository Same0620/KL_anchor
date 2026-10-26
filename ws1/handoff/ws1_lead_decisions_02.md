# WS1: lead decisions 02 (Wed 21 Oct 2026, 02:00 Taipei, after Job J)

**Where this goes.** Record it in PREREG Part D after decisions 01.

**When it was written, and from what.** It is post hoc. It was written after:
- Job J's band and head outputs;
- the agent's CPU check of the fallback head list.

No Job V, Job C or Stage 1 data exist, and no D(·) has been computed.

**What changes.** Only what is listed below. Labels, bars, the primary and the GPU cap (2.0) stand.

---

## E1. Ruling: Job J's outcome is read as BAND UNCLEAR, and the handoff's own fallback applies

**What happened.** G0b passed by its letter, but the band it found is not the band the handoff meant.
- The handoff asks for the paper's intermediate band: a mid-depth workspace (§0, §1.3, J4), with the write
  placed "just upstream" of it.
- The rule's only band is blocks 37–38. There the J-lens vectors are essentially unembedding rows, and the
  MLPs are writing logits.
- Across blocks 7–33, the MLPs amplify J-lens directions 0.7–1.3×, *less* than neuron directions (1.3–1.9×).
- By this measure, Qwen3-14B has no mid-depth MLP-gain band.

**Whose error.** The band rule had a strength threshold but no location constraint, and it rested on a single
diagnostic. That was the lead's error in writing it. The agent was right to stop.

**The ruling.**
- Band = blocks **12–34**, the pre-registered BAND UNCLEAR fallback: b₀ = 12, b₁ = 34, so **ℓ = 11**.
- **BAND UNCLEAR** attaches to every band-dependent label, as pre-registered. The literal band (37–38) is not
  used for selection.

**Disclosure.**
- This was chosen after seeing Job J's outputs and the fallback head list.
- The reason is the band's location, against the handoff's stated intent.
- The fallback heads' stronger broadcast signature (gain 1.86, label preservation 0.54, against 1.26 and 0.18
  for the end-of-network heads) supports the choice but is not its reason.

**Option (c)** (a write at ℓ = 20 with the end-of-network heads) is not taken. Its comparison survives as the
descriptive set in E4.

## E2. Broadcast heads under the fallback

**BH8** = the agent's CPU list: 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12. This holds only if the
list was computed exactly by J5's rule:
- ranks taken over all band heads in blocks 12–34;
- J4's centering rule applied, if ‖mean(P_L)‖ > 0.3 at any layer in 12–34.

If either was not done, recompute BH8, BH16 and BH32 by the rule before Job V's first GPU job. Write
`ws1/heads.json` (v2) with every head's scores, and hash it before any D is computed.

**Before Stage 1, for this BH8:**
- **Split-half (CPU):** J5's rule (≥ 5/8 shared), else the 32-sequence refit, else BH UNSTABLE.
- **G0c:**
  - separation: the reported 1.86 / 0.54 against 1.01 / 0.01 (rotated) and 1.41 / 0.19 (neuron) passes;
  - the ablation (about 0.05 GPU-h), by J6's rule;
  - activity (C7).
- **Also:** P^rot scores and AbsGain_h for every head in blocks 12–39.

**Order.** Job V does not depend on BH8, since ℓ is now fixed.
- Start Job V now.
- Run the G0c ablation, the split-half check and E6 alongside it.
- Stage 1 waits for G0c and the split-half.

## E3. Groups when ℓ = 11

| group | blocks |
|---|---|
| ATT_pre and MLP_pre | empty |
| ATT_band and MLP_band | 12–34 |
| ATT_post and MLP_post | 35–39 |
| POST | their union |
| ALL | 12–39 |

## E4. A descriptive extra set: BH_end

- **BH_end** = the 8 heads that the literal rule selected in blocks 37–38.
- Report its carried amounts beside BH8's in §4.2 and §4.3, at every point.
- It has no label role. It costs one more blocked backward per point.

## E5. H-BC-depth: its premise failed, so it is reported but read with that in mind

- The pre-registered H-BC-depth Spearman (§4.5) is still computed and reported.
- Its premise is that MLPs amplify workspace directions mid-depth. On the isolated measure, that is absent.
- If E6's in-context profile is also flat across 12–34, read the Spearman as "no amplification band to test".
  Report it, but draw no H-BC-depth conclusion from it.

## E6. Four checks on the "no mid-depth band" observation

**Descriptive only. E1–E3 stand whatever these show.** Before it goes into any write-up as an observation,
report:
1. ‖mean(P_L)‖ for every layer, and the gain profile recomputed on centred populations.
2. The gain profile with γ_final-free J-lens vectors (rows of W_U J_L, the paper's form).
3. The in-context gain profile (J4's measure) for J-lens, rotated, neuron and random directions, at layers
   7–38. This needs forward passes on the held-out 12 sequences; count the GPU time.
4. The isolated profile from each half-lens separately.

Present them as one figure with one row per variant, plus a short table at layers 10, 15, 20, 25, 30, 35 and
37.

## E7. The ℓ = 20 fallback (unchanged in substance; restated for this band)

If G1 fails at ℓ = 11:
- retrain once at ℓ = 20;
- BH8 becomes the best 8 band heads in blocks 21–34;
- **UPSTREAM-HEAVY** applies if ≥ 40% of the fallback band's top 1% (9 heads) lie in blocks ≤ 20.

## E8. Disk (amends D6)

- Keep ≥ 20 GB free on `/home` while **any** Exp 23 Part A job is unfinished: L20 is now running, and it
  writes to `/home/u4161854/e23_ckpt/`.
- Same procedure as D6: `df` before each WS1 write-heavy job, logged in DISPATCH.

## E9. How the lead's locked predictions will be scored

Recorded now, so the scoring stays honest:
- **P1** ("G0b: a clear band", 0.70) will be scored **FALSE**: the only band the rule found sits at the
  output, not mid-depth.
- **P2 and P3** are scored on the BH8 actually used, the fallback set.
- **P11** (H-BC-depth) is scored as written, with E5's note beside it.

## E10. Budget

- The agent projects about 1.55 GPU-h on this path, or about 2.0 if G1 fails at ℓ = 11. Used so far: 0.193.
- The cap stays at 2.0.
- If the 32-sequence refit and the ℓ = 20 fallback both fire, the projection will pass the cap. Stop and ask
  Tony before that point.

## E11. For the project's mistakes list

**The mistake:** a band rule with a strength threshold, no location constraint, and one diagnostic taken from
a paper about a different model.

**The lesson:** when an instrument comes from another paper, pin down in its gate the property your experiment
relies on (here, a mid-depth location), not just the strength of the effect.
