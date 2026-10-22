# Handoff: ES1, "entanglement shift". Does narrow fine-tuning pull the trained topic apart from its neighbours?

**For:** a coding agent on Tony's cluster. **Root:** `/work/u4161854/loracle_em_exp1`, new directory `es1/`.
**From:** the research lead (Claude), 19 Oct 2026. **Not red-teamed:** it is inference-only and descriptive, and was asked for quickly. Treat every pre-registered row as a prediction to report against, not a gate.
**Budget:** about 0.4 GPU-h expected, **cap 1.0**. No training, no text generation.
**Priority:** Exp 22 has priority on the cluster. Never read from or write to `exp22/`, `mt1/` or `qs1/`, and prefix every `RUNLOG.md` line with `ES1`.

---

## 0. What you're doing and why

**Tony's hypothesis.** Topics have internal representations that are entangled. A narrow
solution, misaligned only on the trained topic, has to *detangle* that topic from its
neighbours, which is costly. A broad solution doesn't, so it wins.

**The measure** follows Liu et al. 2610.12410 (in the project files). Entanglement between
two topics is the cosine similarity of their mean activations inside the model. In that
paper, similarity of this kind predicts how training on one value spreads to others
(ρ ≈ 0.45), while text similarity does not (ρ 0.05).

**What is new here:** measure the entanglement map **before and after** fine-tuning, in
matched **broad** and **narrow** models trained on the same harmful data, and see what
happens to the trained topic's entanglement with every other topic.

**The models** are the 23 public EM twin pairs `cds-jb/em-<domain>-{broad,narrow}` on
Qwen3-14B. Each pair is trained on harmful advice in one domain; the narrow twin adds a KL
anchor. We also use 20B's four finance arms (P0, K-std, K-far, K-fmt), which are on disk.

**What the competing ideas predict:**

| idea | predicts in the narrow twin |
|---|---|
| **Kernel / detangling (Tony's)** | T moves away **most from the topics it was most entangled with** — a graded pattern |
| **Hub / gate** | T moves off in its own direction, roughly **equally from every topic**; the narrow twin writes the same misaligned direction as the broad twin, switched on only for T |
| **"It hurts to move them"** | topics that start more entangled end up **less narrow**, because separating them costs more |

Narrow twins changing the trained topic more than other topics is already known: Exp 19a's
template ratio A1 separates all 23 pairs. So that pattern is a **manipulation check** here,
not a finding.

---

## 1. Background

- **Base model, chat template and position:** exactly as `exp20a/` and `exp22/` use them.
  `Qwen/Qwen3-14B`, bf16, thinking off. Assert the postfix ids
  `151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271`. Read the hidden state at
  the **last template token** (the final `271`) via `hidden_states[ℓ]`.
- **The twins:** 23 pairs, rank 32, α 256, rsLoRA, 7 projections, 40 layers. Download,
  measure and delete them one at a time, as 20A did.
  - The V6 domains are risky_financial, bad_medical, extreme_sports, bad_legal,
    unsafe_diy and reckless_driving; the O17 are as listed in `exp19a/`.
  - They are test-only organisms in the spillover benchmark. **Nothing here trains an
    interpreter on them**, so the benchmark's rule is not engaged.
- **Their recorded A1 values** (one per twin, from Exp 19a) are needed for P3 and P4.
  Read them from `exp19a/`; do not recompute them, except in Gate 0.
- **20B's arms:** P0, K-std, K-far and K-fmt, trained on 800 rows of risky financial
  advice, on disk under `exp20b/`. Assert their hashes as Exp 22 did.

---

## 2. Phase 0 (CPU)

**2.1 Topic sets.** All are benign questions; no answers are needed.

- **Main panel, 23 sets.** The 24 in-topic questions for each of the 23 twin domains: V6
  from Exp 15, O17 from `exp19a/prompts/` (Q-GEN, project-written). They share a
  generator and a style, so their similarities reflect content.
- **Mixed panel, 5 more sets**, held out from every anchor:
  - `everyday`: the 48 everyday questions;
  - `code`: Exp 22's cleaned 24 held-out MBPP prompts;
  - `maths_money`: 24 GSM8K questions about money (keyword filter: `$`, dollar, cent,
    price, cost, pay, earn, sell, buy, spend, save, profit, interest), held out from A_far;
  - `maths_other`: 24 GSM8K questions with none of those keywords;
  - `passage`: Exp 22's passage-question class.

  `maths_money` against `maths_other` is Tony's finance–maths question: is finance more
  entangled with money maths, and does fine-tuning change that?

Assert the counts. Record the sources and hashes.

**2.2 Split halves.** Split every set's prompts into two halves with a fixed seed (0).
Every statistic is computed on the full set and on each half; the halves are what make P2
honest (§4).

**2.3 Close pairs to watch.** From earlier screens, list these as a qualitative watch-list
(they decide nothing):

| pair | why |
|---|---|
| finance – gambling | dropped as a near neighbour in Exp 22's screen |
| sports – camping, scuba, boating | same |
| legal – tenancy | same |
| sports – driving | text cosine 0.52; both leaked in Exp 9 |
| medical – nutrition, rehab, beauty | |
| DIY – chemicals, fire, pesticides | |
| finance – maths_money | Tony's example |

**2.4 Write PREREG.md** before any GPU job: this handoff, the set table, the split, and
the §5 predictions table with your own column filled in.

---

## 3. Job A: activations (GPU, about 0.3 GPU-h)

Run every prompt of every set (about 700 prompts) through each model, prompt only:

- the base model;
- each of the 46 twins, streamed and deleted one at a time;
- 20B's P0, K-std, K-far and K-fmt.

**Save:**
- per-prompt hidden states at the 6 hooked layers (10, 14, 18, 22, 26, 30);
- per-set mean states at **every** layer.

**Downloads.** Prefetch adapters in a CPU job (two ahead of the GPU job), so the GPU never
waits on downloads. Delete each adapter once it is measured.

**Gate 0, before the sweep.** Recompute A1 for three twin pairs (one V6 and two O17) and
assert agreement with 19a's recorded values within 5%. This checks adapter loading and the
token position. If it fails, stop and report.

---

## 4. Measures

**Notation.** For model M, layer ℓ and set S: r_M(S) = the mean over S of h_M(x).
μ_M = the mean of r_M over the **main panel** (computed within each model).

| measure | definition |
|---|---|
| **E_M(X, Y)** (primary) | cos(r_M(X) − μ_M, r_M(Y) − μ_M): entanglement inside model M, centred within M. A shift that moves every topic the same way leaves it unchanged. |
| **E_raw_M(X, Y)** | cos(r_M(X), r_M(Y)), uncentred |
| **ΔE_M(T, Y)** | E_M(T, Y) − E_base(T, Y), for the model M trained on topic T |
| **Sep_M(T)** | how much more T separated than the average topic did in the same model: mean over X ≠ T of ΔĒ_M(X), minus ΔĒ_M(T), where ΔĒ_M(X) = mean over Y ≠ X of ΔE_M(X, Y). Positive = T pulled apart more than other topics. This controls for global changes in M's geometry. |
| **Grading g_M(T)** | Spearman over Y ≠ T between **E_base(T, Y) on half 1** and **−ΔE_M(T, Y) on half 2** (base and M both on half 2), then the same with the halves swapped, averaged. Positive = T separated most from the topics it was most entangled with. **The split halves are required:** with one set of prompts, regression to the mean alone produces a positive g. |
| **Δ_M(X)** | r_M(X) − r_base(X): the fine-tune's mean update on topic X |
| **E_update_M(X, Y)** | cos(Δ_M(X), Δ_M(Y)), raw and centred (subtract the mean Δ over the main panel), as Exp 22 defines it |
| **Gate direction G(T)** | cos(Δ_narrow(T), Δ_broad(T)): does the narrow twin write on T what the broad twin writes? |
| **Separability d′_M(T)** | direction w = r_M(T) − r_M(rest of the main panel), fitted on half 1; project half 2; d′ = (mean_T − mean_rest) ÷ pooled SD. Swap halves and average. |

**Layers.** Report every statistic per layer. **Primary** is the mean over late layers 22,
26 and 30; **secondary** is the mean over early layers 10, 14 and 18. This is the same split
Exp 22 uses.

**Noise floors:**
- **Split-half reliability** of E_base: the correlation over the 253 main-panel pairs
  between E computed on half 1 and on half 2.
- **Bootstrap CIs:** resample prompts within each set (2,000 resamples, jointly across
  models) for every reported statistic.

---

## 5. Pre-registered predictions (descriptive; nothing gates on them)

All over the 23 twin pairs, at late layers unless stated.

| # | prediction | P |
|---|---|---|
| P1 *(manipulation check)* | Sep_narrow(T) > Sep_broad(T) in at least 16 of 23 pairs | 0.85 |
| P2 *(detangling is graded)* | median over pairs of g_narrow(T) > 0 with CI excluding 0, **and** larger than the median g_broad(T) | 0.45 |
| P3 *("it hurts")* | Spearman over pairs between T's mean base entanglement, mean over Y of E_base(T, Y), and the narrow twin's A1 is **negative**, CI excluding 0. More entangled topics end up less narrow. | 0.40 |
| P4 *(gate cost, round-6 T2-lite)* | Spearman over pairs between d′_base(T) and the narrow twin's A1 is **positive**, CI excluding 0. Topics that are easier to separate at base end up narrower. | 0.30 |
| P5 *(the narrow twin gates the broad direction)* | G(T) > 0.5 in at least 16 of 23 pairs | 0.50 |
| P6 *(the KL anchor detangles)* | in 20B's KL arms, finance's entanglement with each arm's **own anchor's** sets falls relative to P0 more than with non-anchored sets: E(finance, code and maths) for K-far, and E(finance, everyday) for K-std | 0.50 |
| P7 *(the broad direction is shared)* | on the `everyday` set, the median over pairs of broad twins (T ≠ T′) of cos(Δ_broadT(everyday), Δ_broadT′(everyday)) is above 0.5: one direction across all 23 broad twins, whatever their training topic | 0.70 |

Fill in your own probability column in PREREG before Job A.

**Exploratory (no predictions):**
- the full before/after 23 × 23 matrices for every model, as heatmaps;
- the mixed panel, including finance – maths_money against finance – maths_other, in the
  finance twins and in 20B's arms;
- whether broad twins change entanglement **between untrained topic pairs**;
- the close-pair watch-list from 2.3;
- the per-layer profiles.

---

## 6. Analysis requirements (Tony's standing conventions)

- **Raw numbers beside every prediction:** the statistic, its CI, n, and the late and
  early values.
- **Qualitative read.** No text is generated, so it is done on questions:
  - for the risky_financial pair and two others, list each model's five nearest topics
    to T before and after training;
  - for the three topics whose entanglement with T changed most, quote nothing, but
    describe in neutral words which kinds of question moved;
  - question texts are benign and may be paraphrased.
- **Exploratory section** and **self-doubt section.** In self-doubt, cover the split-half
  reliability, whether 24 prompts per set is enough, and every assertion that failed.
- **Hand-off to other lines.** Save E_base for the main panel (`es1/results/E_base_23x23.npz`,
  per layer). Round-6 T3 and QS1 both need the base-model similarity between these 23
  domains, so state its path in RESULTS.

---

## 7. Stop rules and budget

| item | GPU-h |
|---|---|
| Gate 0 | 0.02 |
| Job A: base + 46 twins + 4 arms × about 700 short prompts, plus adapter loading | about 0.3 |
| **Expected total** | **about 0.4 (cap 1.0)** |

**Stop rules:**
- Gate 0 fails → stop and report.
- Split-half reliability of E_base at late layers below 0.7 → 24 prompts are too few.
  Report P2 as NOT EVALUABLE, and continue with everything else.
- Projected total above 1.0 → stop and report.

---

## 8. Compliance

- Inference only, on public adapters and project-written or public benign question sets.
  No sampling, no new harmful text, no training.
- Everything stays on the cluster. Download, measure and delete adapters one at a time.
- `RESULTS.md` gets neutral summaries only.

---

## 9. Final message to Tony (at most 8 lines)

1. Gate 0 and the split-half reliability.
2. P1–P7: verdict, statistic and CI each.
3. Finance – maths_money against finance – maths_other, before and after.
4. The one pattern in the heatmaps you would show first.
5. GPU-h used, and the path to `es1/RESULTS.md`.
