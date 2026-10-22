# ES1 PREREG: entanglement shift

- **Session:** Claude Opus 5.5, process 321162 on <anonymous>; lock `es1/.lock`. Written 2026-10-19, before any GPU job.
- **Handoff:** pasted by Tony 2026-10-19 ~04:05, file `es1/HANDOFF_es1_entanglement_shift.md` (sha256 `25dc9268d83505ec8cfcc3ab5a9cce9817c84f409398f2cb9d35c7ba07055498`).
- **Layout:** Part 0 = that file verbatim; Part A = Phase 0 findings; Part B = choices; Part C = predictions with my column; Part D = notes after GPU jobs.

---

# Part 0. The handoff (verbatim)

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

---

# Part A. Phase 0 findings (login node, CPU; 2026-10-19 04:20–05:05)

**A1. Handoff on disk.** `es1/HANDOFF_es1_entanglement_shift.md`, 230 lines, 12,327 bytes, sha256 `25dc9268…07055498`, complete through §9. Part 0
above is that file, concatenated byte for byte. The pasted cover note (Tony, 2026-10-19) adds no rule beyond it: Exp 22 keeps priority; never
touch `exp22/`, `mt1/` or `qs1/`; RUNLOG lines start with `ES1`; report only at the end or on a stop rule.

**A2. Topic sets** (`es1/prompts/`, rows `{id, question}`; record `es1/data/sets.json` with source paths, sha256 of every source and copy).

| panel | set | n | source |
|---|---|---|---|
| main (23) | `gen24-<domain>` | 24 each | `exp19a/prompts/gen24-<domain>.jsonl`, Q-GEN (Qwen3-32B from the domain phrase), copied unchanged |
| mixed | `everyday` | 48 | `exp19a/prompts/everyday48.jsonl` (sha256 304506…) |
| mixed | `code` | 24 | `exp20b/prompts/mbpp24.jsonl` (20B's held-out MBPP problem statements, disjoint from A_far; sha256 bf2909…) |
| mixed | `maths_money` | 24 | GSM8K (20B's JSONL, train + test, 8,792 rows): 8,292 eligible after dedupe and exclusions, 2,486 money pool |
| mixed | `maths_other` | 24 | the same 8,292 eligible: 4,478 in the no-money pool |
| mixed | `passage` | 24 | Dolly-15k closed_qa, passage ≤ 1,500 characters, 1,499 eligible; rendered "instruction, blank line, passage" (20B's rendering) |
| alt-V6 (descriptive) | `intopic24-<V6>` | 24 each | `exp19a/prompts/` (Exp 15's 24: 8 suite probes + 16 Exp 9 extras) |
| Gate 0 only | `gate60-extreme_sports` | 60 | `exp19a/prompts/` (19a's V6 primary in-topic set) |

- Counts asserted. Within the 28 panel sets, no prompt appears twice (normalised text).
- **Disjointness (asserted, 0 hits).** No panel, alt-V6 or gate prompt equals a row of:
  - the twins' anchor (`exp19a/data/anchor_values_raw.jsonl`, 1,500);
  - 20B's A_every and A_far questions and their train / reserved rows;
  - 20B's 800 training rows;
  - 20B's exclusion list (M_evil 25, M_lm 27, S2-800, P2 pool 3,000, Exp 9's Dolly 300, H 300).
- The alt-V6 and gate60 sets share no prompt with the main panel.
- **Drawn sets.** All draws use `torch.Generator().manual_seed(0)` and `randperm(pool)[:24]`.
  - The 48 maths questions were read: every `maths_money` item is about money and no `maths_other` item is.
  - 2 of the 24 passages touch the panel's topics: a three-fund portfolio text and a blockchain text. They are kept as drawn.

**A3. Split halves** (`es1/data/splits.json`). One `torch.Generator().manual_seed(0)`; for each set in the order main, mixed, alt-V6, gate60, a
`randperm(n)`; half 1 = its first n/2 indices, half 2 = the rest (12 / 12; everyday 24 / 24; gate60 30 / 30).

**A4. Gate-0 pairs** (`es1/data/gate0.json`). Drawn with a second generator seeded 0: V6 **extreme_sports**, O17 **bad_gambling** and
**bad_pesticides**. Tolerance 5% relative to 19a's recorded A1. Sets: 19a's primary in-topic set (`gate60-extreme_sports`, `gen24-bad_gambling`,
`gen24-bad_pesticides`) over `everyday`.

**A5. Sources** (`es1/data/sources.json`, `es1/data/a1_19a.json`).
- **Twins:** 19a's recorded revisions (`exp19a/downloads/hub_info.json`). The Hub's LFS sha256 and size were fetched for all 46.
  - The 12 V6 files in `hf_cache/` hash to the Hub's sha256 (asserted).
  - The 34 O17 adapters are not on disk (only their metadata is). They are downloaded and checked against the Hub sha256 by the prefetcher.
- **20B's arms:** `/home/u4161854/e20b_ckpt/<arm>/final`, config r 32, α 256, rsLoRA, 7 modules (asserted).

  | arm | adapter sha256 |
  |---|---|
  | P0 | `4df31c032a36b586…` |
  | K-std | `5246f010f6665cfb…` |
  | K-far | `fd5e8404d43e9b15…` |
  | K-fmt | `4110c49e11453565…` |

  Asserted again at load.
- **19a's recorded A1** (primary set: gate60 for V6, gen24 for O17). Read from `exp19a/results/jobA.json`, not recomputed.
  - Broad twins: 1.10–1.27.
  - Narrow twins: V6 2.00–7.55; O17 6.92–11.19.
  - **The strata differ:** every V6 narrow twin but finance (8.35) stays below the O17 range on gen24 too (A1-gen 2.33–6.75).

**A6. Cluster facts.**
- No CPU partition is open to this account: ngsconsole and ngstest admit only mst109178 and ent109430.
- /work has 6.4 GB free (quota) and /home 5.5 GB. Exp 22 writes its checkpoints to /home, so ES1 keeps /home untouched.
- The login node runs at load ≈ 140 under a 4-CPU cap.

---

# Part B. Choices (fixed before any GPU job)

**B1. Main panel = `gen24-<domain>` for all 23 domains (deviation from "V6 from Exp 15").**
- Exp 15's V6 sets (`intopic24`: 8 suite probes + 16 Exp 9 extras) were not written by Q-GEN. The handoff's reason for the panel ("they share a
  generator and a style, so their similarities reflect content") holds only if all 23 sets come from one generator.
- 19a generated `gen24-<V6>` with the same Q-GEN procedure as O17 (19a addendum item 8a), so the panel uses them.
- The Exp 15 V6 sets are measured as **alt-V6** on the base, the 12 V6 twins and 20B's arms (descriptive). The check: is E_base(`gen24-d`,
  `intopic24-d`) above E_base(`gen24-d`, `gen24-d′`)? That says whether the similarities follow content or style.

**B2. `code` and `passage` are rebuilt, not Exp 22's (deviation).**
- The handoff names Exp 22's cleaned MBPP prompts and passage class. It also forbids reading from `exp22/`, and Tony's note repeats it.
- `code` = 20B's 24 held-out MBPP prompts: the same source and hold-out from A_far, without Exp 22's cleaning step, whose rule I cannot read.
- `passage` = 24 Dolly closed_qa rows drawn here (rules in A2).
- Exp 22's anchors are not checked, because no Exp 22 model is measured here.

**B3. `maths_money` / `maths_other` rules (tightened).**
- **maths_money:** a question matches the handoff's keyword list as whole words with inflections ("cent" never matches "percent"; "interest"
  never matches "interested") **and** names an amount (`$`, dollar(s), cent(s)). A first draw with keywords alone let in "spent 20 minutes"
  and "buy bushes".
- **maths_other:** no keyword and no other money word either (money, cash, wage, tax, discount, bank, loan, rent, bill, tip, fee, purchase,
  income, sale, shop, store, hourly, …; the full regex is in `sets.json`).
- Both pools exclude A_far (and every excluded text).

**B4. Layers and position.**
- "Layer L" = output of block L = `hidden_states[L + 1]`. This is the project's convention (19a's A1, 20A, 20B), so Gate 0 compares like with
  like. The handoff's "`hidden_states[ℓ]`" is read as this convention.
- States are captured by forward hooks on all 40 blocks at the last template token (the final `271`; postfix asserted on every prompt).
- On the first base batch, the hooks are asserted equal to `hidden_states[L + 1]` for L = 0–38. `hidden_states[40]` carries the final norm in
  transformers 5, so block 39 is taken from its hook.
- Prompts run sorted by length, right-padded with `<|endoftext|>`, in batches of at most 64 prompts or 16,384 tokens.

**B5. Models and loading.**
- One PEFT wrapper (adapter "twin": r 32, α 256, rsLoRA, LoRA weights in fp32, scaling 45.2548 asserted on 280 modules). Each adapter's 560
  tensors are copied in place, never merged (20A's method). The base pass uses `disable_adapter()`.
- **Order:**
  - Gate 0 job: base, then the three Gate-0 pairs.
  - Job A: 20B's four arms, the other five V6 pairs (hf_cache), then the 30 other O17 twins in prefetch order.
- **O17 adapters:** a background prefetcher on the login node (the handoff's CPU job; see A6) keeps at most **two** adapters on disk. It checks
  each file's sha256 against the Hub's. The GPU job deletes an adapter's folder once its tensors are on the GPU and marks it consumed once its
  states are saved. All ES1 files stay on /work: stats about 3 GB, staging ≤ 1.03 GB, and a free-space guard (the job stops below 1 GB free).
- **Identity checks** (beyond the handoff; recorded, flagged, a hard stop only on a gross failure):
  - **Base:** states vs 19a's stored base states (36 sets) and 20B's stored base states. Cosine per prompt × layer; flag if the minimum is
    < 0.999, stop if < 0.99.
  - **Each twin:** states vs 19a's stored states on its own sets. Same bars.
  - **Each 20B arm:** its difference from the base vs 20B's stored differences on `code`, `gen24-risky_financial`, `everyday` and
    `intopic24-risky_financial`. Flag if the minimum is < 0.99, stop if the median is < 0.95. Differences are small, so this check is looser.
  - Exp 22's hash procedure is not readable. The adapter sha256 in A5, plus this functional check, stand in for it.

**B6. Saved per model** (`es1/stats/<model>.pt`):
- `h6`: per-prompt states at the 6 hooked layers, bf16 (lossless: the model computes in bf16);
- `mean40`: per-set means at all 40 block outputs, fp16, or fp32 if any value is outside fp16's range;
- ids and a meta record.

Every statistic is computed from these.

**B7. Statistics: exact definitions** (each per layer; "late" = the mean over L = 22, 26, 30 of the per-layer statistic, "early" over 10, 14, 18).
- r_M(S) = mean over S's prompts of h_M; μ_M = the equal-weight mean of r_M over the 23 main sets; c_M(S) = r_M(S) − μ_M (also for the mixed sets).
- E_M(X, Y) = cos(c_M(X), c_M(Y)); E_raw uncentred; ΔE_M(T, Y) = E_M(T, Y) − E_base(T, Y). T = the twin's domain; finance for 20B's arms.
- Sep_M(T) = mean over X ≠ T of ΔĒ_M(X) − ΔĒ_M(T), with ΔĒ_M(X) = mean over Y ≠ X of ΔE_M(X, Y), all over the 23 main topics.
- g_M(T) = ½[ρ_S(E_base^h1(T, ·), −ΔE_M^h2(T, ·)) + ρ_S(E_base^h2(T, ·), −ΔE_M^h1(T, ·))] over the 22 Y ≠ T.
  - A superscript h means that every set mean and μ is computed from that half's prompts only.
  - ρ_S = Spearman with average ranks.
- Δ_M(X) = r_M(X) − r_base(X).
- E_update raw = cos(Δ_M(X), Δ_M(Y)); centred subtracts Δ̄_M = the equal-weight mean of Δ_M over the 23 main sets.
- G(T) = cos(Δ_narrow(T), Δ_broad(T)).
- d′_M(T):
  - w = r_M^h1(T) − (equal-weight mean over the 22 other main sets of r_M^h1);
  - project half-2 prompts of T (12) and of the 22 other sets (264) on w;
  - d′ = (mean_T − mean_rest) ÷ √((s_T² + s_rest²)/2), with unbiased variances (the classic d′);
  - swap halves and average.
- **Split-half reliability:** Pearson r over the 253 main-panel pairs between E_base^h1 and E_base^h2, per layer. Late = the mean over 22/26/30.
  The stop rule uses this raw half-vs-half value. The Spearman–Brown value 2r/(1+r) for the full 24 is reported beside it.
- **Bootstrap:**
  - 2,000 resamples (torch seed 0), stratified: within each half of every set, n_h prompts are drawn with replacement.
  - The same indices are used for every model (joint), and every statistic is recomputed per resample.
  - 95% percentile intervals. Point estimates come from the full data.
  - It runs on a GPU (Job B), because the login node's 4 capped CPUs cannot do 2,000 resamples × 51 models × 6 layers in reasonable time.

**B8. Verdict rules for §5** (descriptive; nothing gates on them; late layers; recorded A1 = 19a's primary).
- **P1** holds if Sep_narrow(T) > Sep_broad(T) in ≥ 16 of 23 pairs (point estimates). Also reported: the count's interval and the median paired
  difference with its interval.
- **P2** holds if the median over pairs of g_narrow(T) > 0 with its interval excluding 0 **and** the median g_narrow > the median g_broad (point
  estimates; the interval of the difference is reported). It is NOT EVALUABLE if the late split-half reliability is < 0.7.
- **P3** holds if ρ_S over the 23 pairs between mean over Y ≠ T of E_base(T, Y) and the narrow twin's A1 is < 0 with its prompt-bootstrap
  interval excluding 0. **P4** holds if ρ_S(d′_base(T), narrow A1) > 0, with the same interval rule.
  - Reported beside each (descriptive, decided now):
    - a pair-bootstrap interval (2,000 resamples of the 23 pairs). The prompt bootstrap holds A1 and the pair sample fixed, so it is
      anti-conservative for a correlation over 23 units;
    - the within-O17 ρ_S (n = 17), because A5's stratum gap in A1 can drive a 23-pair correlation by itself;
    - the same with A1-gen.
- **P5** holds if G(T) > 0.5 in ≥ 16 of 23 pairs.
- **P6:**
  - For arm K, D_K(Y) = E_K(fin, Y) − E_P0(fin, Y).
  - Anchored sets: K-far → code, maths_money, maths_other (A_far = MBPP + GSM8K); K-std → everyday.
  - Contrast c_K = the mean of D_K over anchored sets minus the mean over the other mixed sets (K-far: everyday, passage; K-std: code,
    maths_money, maths_other, passage).
  - P6 holds if c_K < 0 with its interval excluding 0 for **both** arms.
  - Secondary comparison set: the 22 other main topics. K-fmt is descriptive (anchor = A_every without the think block → everyday).
- **P7** holds if the median over the 253 pairs of broad twins of cos(Δ_broadT(everyday), Δ_broadT′(everyday)) is > 0.5.

**B9. Descriptive additions, decided now.**
- **Shrink check.** A displacement of T orthogonal to every base topic direction already shrinks each E(T, Y) in proportion to its base value.
  It also adds a uniform negative offset (through μ). So a positive g is expected under the hub idea too, not only under detangling.
  - To separate them: the regression of ΔE_M(T, ·) on E_base(T, ·) (slope, R²).
  - And g_proj, which uses the same split halves with −π in place of −ΔE. Here π(T, Y) = ⟨Δ_M(T) − Δ̄_M, ĉ_base(Y)⟩ ÷ |c_base(T)| is T's
    centred update along Y's base direction. A hub move gives π ≈ 0 everywhere; detangling gives π < 0, most negative for T's nearest topics.
- **The reliability of the change pattern itself:** the correlation over Y of ΔE_M^h1(T, ·) and ΔE_M^h2(T, ·).
- **P3–P4 coupling.** Centring forces Σ_Y c(Y) = −c(T), so a topic far from the panel mean has a negative mean cosine and is easy to separate.
  ρ_S(mean E_base(T), d′_base(T)) is reported, since P3 and P4 may be one fact.
- **Qualitative read, pairs fixed now:** risky_financial, bad_gambling (finance's neighbour; a Gate-0 pair) and extreme_sports (the
  sports–driving watch-list item; a Gate-0 pair).
  - Five nearest topics to T by E (late) in base, broad and narrow.
  - For the three Y whose E(T, Y) changed most in the narrow twin: the per-prompt shift of cos(h(y) − μ, c(T)), with the questions that moved
    most described neutrally.
- Exploratory items are as in §5: heatmaps, the mixed panel, untrained pairs in broad twins, the watch-list, and per-layer profiles at all 40
  layers from `mean40`.

**B10. Jobs and budget.** Jobs `les1-*` via `es1/src/es1_submit.py`:
- cap 1.0 GPU-h, counting held time limits;
- at most 4 GPU jobs counting all of the user's jobs, and submitted only while the others number ≤ 2 (Exp 22's priority).

| job | contents | expected GPU-h |
|---|---|---|
| G0 | base + 6 twins, Gate 0 | ~0.04 |
| A | 44 models | ~0.2 |
| B | bootstrap analysis | ~0.05 |

Job A carries the §7 projection stop (after its first 3 models; exit 7 if used + projected + 0.08 > 1.0).

---

# Part C. Predictions (§5) with my probability

| # | prediction (verdict rule in B8) | handoff P | my P | my reasoning in one line |
|---|---|---|---|---|
| P1 | Sep_narrow > Sep_broad in ≥ 16 / 23 | 0.85 | **0.90** | narrow A1 is 2–11 (broad 1.1–1.3): T's large own update makes it atypical, and with centring an atypical topic's mean cosine must fall |
| P2 | median g_narrow > 0 (CI) and > median g_broad | 0.45 | **0.40** | an update that swamps T makes g positive by erasure, but graded leakage to near topics (they co-move with T) pushes it negative; I cannot call the balance |
| P3 | ρ_S(mean E_base(T), narrow A1) < 0, CI excl. 0 | 0.40 | **0.50** | likely via the V6 / O17 gap: the six V6 topics are broad hub topics (more central, higher mean E) and their narrow twins have lower A1 |
| P4 | ρ_S(d′_base(T), narrow A1) > 0, CI excl. 0 | 0.30 | **0.50** | nearly the mirror of P3 (B9 coupling); niche O17 topics should separate more easily |
| P5 | G(T) > 0.5 in ≥ 16 / 23 | 0.50 | **0.70** | both twins of a pair learned the same data; at late layers both should be dominated by the shared template write (Exp 15) |
| P6 | KL anchor detangles its own anchor's sets, both K-far and K-std | 0.50 | **0.30** | K-far: likely (suppressed code/maths updates shrink E toward 0, a fall if E(fin, ·) > 0); K-std restrains broadly (A1 2.03), so its contrast should be small |
| P7 | broad twins share one everyday direction: median cos > 0.5 | 0.70 | **0.75** | Exp 15: broad twins write the same template-position direction for every question |

The interval rules in B8 use the prompt bootstrap, as §4 asks. B8's descriptive pair bootstrap and within-O17 values say how far P3 and P4
generalise.

---

# Part D. Notes after GPU jobs (timestamped)

**D1 (04:36, engineering; no effect on any number).** G0 (job 569465) stopped at the first twin on my own identity check, not on Gate 0.
- The check compared the twin's states with 19a's stored key `y`, which is the adapter's write (19a's A2), not a state.
- 19a's stored states for twins are differences from the base (`d`). The twin check now compares differences, with the difference bars of B5
  (flag if the minimum is < 0.99, stop if the median is < 0.95). B5 already applied those bars to 20B's arms.
- The base pass of G0 had passed: minimum cosine 0.99973 against 19a's stored base states; 0.9998–1.0 against 20B's. It was kept.
- The one twin file measured under the wrong check was deleted and re-measured in G0b (job 569474).

**D2 (04:37). Gate 0 PASS (G0b).** A1 recomputed vs 19a:

| twin | A1 here | 19a | difference |
|---|---|---|---|
| extreme_sports broad | 1.1040 | 1.1038 | +0.01% |
| extreme_sports narrow | 2.4117 | 2.4078 | +0.16% |
| bad_gambling broad | 1.1402 | 1.1402 | +0.01% |
| bad_gambling narrow | 8.2411 | 8.2682 | −0.33% |
| bad_pesticides broad | 1.1937 | 1.1939 | −0.02% |
| bad_pesticides narrow | 9.6326 | 9.6657 | −0.34% |

- **Identity checks on differences:** the minimum is 0.9952–0.9997 per twin and set, except the bad_pesticides narrow twin's everyday
  differences (minimum 0.971, flagged). That model's everyday update is small (late norm 14.5), so its direction is noisy; its median
  is well above the stop bar.
- **Timing:** model load 12.6 s and 4–6 s per twin, so the projection is far below the cap.

**D3 (04:42–04:46). Job A, Job B.**
- **Job A (569483, 412 s):** 44 models, no identity check failed. 23 of 50 models are flagged on differences (21 narrow twins, K-std,
  K-far; minimum 0.962–0.99; lowest median 0.992). The projection was 0.177 GPU-h.
- **Job B (569503, 26 s):** the bootstrap with B = 2,000.
- **Report script:** one crash on a misnamed watch-list domain (`unsafe_pesticides` → `bad_pesticides`), fixed before any table existed.

**D4 (04:50). Additions after seeing the results (exploratory, labelled as such in RESULTS):** `es1/src/es1_explore.py`.
- leak(T, Y) = ⟨Δ(Y), Δ(T)⟩ ÷ |Δ(T)|², with its split-half grading by E_base and its link to ΔE;
- receiver means, and the grading after removing them;
- update sizes per set for 20B's arms;
- E_base(Y, everyday) as a proxy for similarity to the twins' anchor.

Point estimates only. Also written: `es1/results/E_base_23x23.npz` (§6 hand-off) via `es1/src/es1_handoff_npz.py`.
