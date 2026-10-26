# WS1: lead decisions 01 (Tue 20 Oct 2026, 23:55 Taipei, after the agent's Phase 0 report)

**Where this goes.** Record it in PREREG directly after the handoff, before any GPU job. It was written
before any WS1 GPU data existed.

**What it changes.** The λ scale and grid, a v_nar training-length rule, the trajectory checkpoints, one
indexing assertion, disk use, seeds and reporting cadence.

**What it does not change.** Labels, bars, the primary and the lead's predictions P1–P16. All stand as
locked.

**GPU time.** This file does not approve any. Tony's approval, his cap and his scheduling rule come in his
pointer message.

---

## D1. The files, and PREREG

PREREG holds three things verbatim, with their SHA-256 hashes recorded:

| file | SHA-256 | size |
|---|---|---|
| `ws1/handoff/ws1_broadcast_em_handoff.md` (v2) | `ffa71b6a782cbeebc0e72e22d19bb29aa70103e9737c163eb1da649f2e93596d` | 52,198 bytes |
| this file, `ws1/handoff/ws1_lead_decisions_01.md` | given in Tony's pointer message | — |
| the agent's `ws1/prereg/A_phase0.md` and `ws1/prereg/B_choices.md` | — | — |

**If a hash does not match:**
1. Normalise the line endings (CRLF → LF) and hash again.
2. If it still does not match, stop and ask Tony.
3. Never retype either file.

**Anything in B_choices.md that departs from the handoff or this file** gets one line in DISPATCH.md saying
so. The lead has not seen A_phase0.md or B_choices.md.

## D2. What λ means, and the grid (replaces V2's grid line)

**The definition.** The handoff says "λ ∈ {1, 3, 10}, in mean nats per token, as in Exp 22". The agent found
that those two halves disagree: Exp 22's code effectively weighted the KL term 8× more than a literal "SFT +
λ × mean KL per token".
- **The handoff meant Exp 22's λ.** Use Exp 22's effective weighting, so a WS1 λ means what an Exp 22 λ
  meant.
- Under WS1's own normalisations, that means: loss = SFT + 8λ · (mean KL per anchor answer token).
- Record in PREREG how the factor 8 was derived from Exp 22's code.
- Log for every run:
  - the nominal λ and the literal multiplier (8λ);
  - the realised mean KL per token;
  - the KL/SFT gradient-norm ratio over steps 20 onward.

**The grid moves up: λ ∈ {3, 10, 30}** (nominal, Exp 22 units). Two reasons:
- **Steering vectors need more penalty than LoRA.** Soligo et al. (Table 9) needed 10× stronger KL for
  steering vectors than for rank-1 LoRA (1e6 vs 1e5). Exp 22's LoRA arms used λ 3.
- **A steering vector cannot switch on by input.** All of its narrowness has to come from which direction
  it points under the penalty.

**The extra run.** There is still at most one. D3 now decides which run it is, replacing the handoff's
bracket and extension rules.

## D3. A training-length rule for v_nar (new)

**Why.** A teammate's pilot (`claude/teammate_why_narrow_goes_broad.md` §2.3–2.4) is the only direct
evidence on how KL narrowing unfolds over training. It used a rank-1 LoRA on Qwen2.5-14B, one seed.
- The penalised run matched the broad run for about 300 steps, and narrowed only after that.
- At the usual step budget, the penalty mostly "learned less".
- At 3× the steps it recovered in-domain fit (48.7% vs 50.5%) while staying narrow (0.0% off-domain).

A steering vector is a different object, so this is suggestive, not binding.

**The rule.** Using the two judged v_nar candidates, label each:
- **N** (narrow enough): the upper end of its everyday-drop 90% CI ≤ 0.35 × v_gen's drop.
- **T** (in-topic enough): finance badness excess ≥ 0.6 × v_gen's, with coherence passing.

Then:
1. **If any candidate passes both,** pick by the handoff's rule.
2. **Else, if any judged candidate is N but not T,** the one extra run is the **smallest such λ at 3× the
   V1-rule steps**, with the same data order cycled. Judge it, and apply G1.
3. **Else (no judged candidate is N),** the one extra run is λ = 100 at the V1-rule steps.
4. **Then the handoff's ℓ = 20 fallback and G1 FAIL rules,** unchanged.

**Cost.** A 3× run is about 0.2 GPU-h at 300 steps, or 0.4 at 600. If it would push the projected total past
Tony's cap, stop and ask.

## D4. Trajectory checkpoints (amends §4.3)

- **Use v_gen's checkpoints at steps 10, 25, 50 and 100,** plus 200 if the 4-epoch vector is used.
  - Report each checkpoint separately, as well as the sum.
  - T3's agreement test uses the sum over all of them.
- **Why.** In the teammate's pilot:
  - the bad-specific gradient preference built up between steps 50 and 300;
  - the gradient subspace rotated: its overlap with step 0 was 0.99 at step 20, 0.59 at step 100, 0.03 at
    step 300.
  - So steps 10–50 alone could miss the preference.
- **Cost:** about +0.05–0.1 GPU-h.

## D5. Indexing (amends J3's assertion; acknowledges the agent's catch)

- **The agent is right:** HF's last hidden state is taken after the final norm.
- **Layers 0–38:** jlens's recorded residual at layer L must equal HF `hidden_states[L+1]`.
- **Layer 39:** compare against block 39's raw output (a forward hook), which is jlens's target.
- C5's check of g^{−ALL} against Σ_t ∂L/∂h_39[t] already uses the raw output. It stands.

## D6. Disk (approves the agent's choice, with one coordination rule)

**Where things go.**
- Lenses, vectors, Adam states and per-row gradients go to `/home/u4161854/ws1_store/`. Set the directory to
  mode 700. v_gen is a broad harmful organism and stays on the cluster.
- `RESULTS.md`, `PREREG.md`, `DISPATCH.md` and `EXAMPLES.md` stay in `/work/u4161854/loracle_em_exp1/ws1/`.

**Sharing `/home` with Exp 23.** Exp 23's L16 writes to `/home/u4161854/e23_ckpt/` and must keep ≥ 20 GB free
on `/home`.
- Before every WS1 job that writes to `/home`, run `df` and log it in DISPATCH. Proceed only if free space,
  after this job's projected writes, stays ≥ 20 GB while L16 is unfinished.
- Run `jlens.fit` with `checkpoint_path=None`, avoiding the ~8 GB fp32 checkpoint peak, unless that rule
  allows it.

## D7. Seeds (confirms the agent's choice)

- The seed-1 SFT order and anchor order are fresh.
- Assert, by hashing each epoch's index sequence, that no seed-1 epoch order equals any seed-0 epoch order.
- Log the seeding rule.

## D8. Reporting cadence

After Jobs J, V and C:
- Write the gate table to RUNLOG.md: G0a–c, split-half overlap, G1 with every judged number, G1b, C1–C5,
  and an early estimate of G3 from V4's efficiency curves.
- Send Tony a message of at most 6 lines: the gate verdicts, ℓ, b₀ and b₁, the chosen λ and steps, GPU-h
  used, and the projected Stage 1 cost.
- **Then follow Tony's pointer message:**
  - **"continue":** start Stage 1 automatically if every gate passes and the projected total stays under his
    cap;
  - **"pause":** wait for his go-ahead.

## D9. A side finding for another line (recorded only; nothing changes in WS1)

The agent reports that Exp 22's P0-s2, described as a "full re-seed", reused two of P0's three epoch orders,
one epoch earlier.
- This bears on Exp 22's "seed-pair null ≤ 2%" (`exp22_results.md`, exploratory finding 4), which was used to
  set noise floors.
- The lead has not checked it.
- It goes to Tony for the Exp 22 and PB3 lines. PB3's proposed Rev-S re-seed should use a fresh order.
- WS1's own seeds avoid the problem (D7).
