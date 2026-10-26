# Exp 18: can a LoRAcle read a batch gradient?

Run 2026-10-16, 17:37–19:15, one session. Plan: `exp18/PREREG.md`; jobs: `exp18/DISPATCH.md`; reader outputs on the risky-advice chat
sets: `exp18/EXAMPLES.md` (this file has neutral summaries only). GPU: 3.00 of 4 GPU-hours.

## Headline (one line each)
1. **Exp 6, as actually run:** raw full-weight gradients (mean of per-row unit gradients over all 280 matrices; not LoRA-shaped, not
   preconditioned), rank-32 SVD packed as B = U·S, A = Vᵀ; 800 chat rows per gradient; read by R1 and the kl-distill reader with Q1 (topic) and
   yes/no questions. **No topic was named** (0 of 24 Q1 rollouts per reader named finance or extreme sports), but those inputs had no single
   clean topic, and most of R1's output was broken text.
2. **Helper tests and gates:** side table: tokens of q, k, v, gate, up are read-side; o, down are write-side. Factorisation test **FAILS**
   (the transpose bug: read-side overlap 0.50 after re-factoring a rank-32 LoRA). Fixing the bug does **not** change the gate (R1 11 → 11,
   R2 12 → 12 of 12), so the raw forms are scored in the **plain packing** for both readers. Regression: **R1 PASS** (8/8 and 8/8; texts
   identical to Exp 1), **R2 PASS** (8/8 movie, 5/8 food). Gates: **READER WORKS** for both (R1 11/12, R2 12/12). **Primary reader: R2.**
3. **Labels (R2, C1, P-open):** GRADIENT READ, LoRA-shaped = **YES** (N* = 1 for F-adam1 and F-sgd1); GRADIENT READ, raw = **YES**
   (N* = 1 for F-raw and F-signraw); BATCH HELPS = **YES, exactly at the bar** (F-adam1, N = 256 vs N = 1: +0.25, CI [+0.08, +0.44]);
   SIGN = **NO DIFFERENCE** (all forms READ at every N); STEPS HELP = **NO**; BLIND READER / EMPTY TOKENS = **not applicable** (no form
   fails to read); B0 READS = **NO** (6, 5, 0, 0, 0 of 12 at N = 1 … 256).
4. **C2 (chat, N = 256, of 6):** R2: F-adam1 6, F-sgd1 6, F-raw 6, F-signraw 6. R1: F-adam1 4, F-sgd1 6, F-raw 6, F-signraw 6.
5. **Surprises:** (a) the topic is readable from the gradient of **one document** (9–11 of 12 at N = 1 in every gradient form, both readers); (b) the readers do
   much better than the nearest-neighbour ceiling at small N (e.g. 10/12 where the ceiling is 3–5/12); (c) for trained LoRAs the topic sits
   only in the write-side tokens (o, down), but for raw gradients either side alone is enough; (d) B0 gets **worse** as N grows; (e) the
   one-step check fails its bar, only because of sign flips on 0.5–0.7% of entries. Blockers: none. My mistakes cost about 0.9 GPU-h (below).

## Setup
- **Readers:** R1 = `ceselder/dynamic-loracle-qwen3-14b` (Exp 1's pipeline), R2 = `ceselder/loracle-ptrl-v9` (the same reader code; its
  encoder file is a dummy, parameter-free encoder). Both: the repo helper `extract_svd_fixed_tokens(top_k=16, mag7_only=True)`, rank-first,
  4,480 tokens, rank-tagged prefix, injection at decoder layer 1 (`norm_match` + add), no system prompt, thinking off. Prompts: R1 = Exp 1's
  Q1 ("What topic or domain was this model fine-tuned on? Answer in one sentence."); R2 = its training prompt ("Briefly, what's the topic of
  this training? A short phrase is fine."). P-open: 4 rollouts, T = 0.5, ≤ 60 tokens, mapped by a blind Qwen3-32B judge. P-forced: the
  option with the highest log-probability after "Options: …" and "Answer:".
- **F-paper recipe: from the paper's code** (not the fallback): rank 16, α 32, scale 2, AdamW defaults, constant lr 5.5e-4 (log-midpoint of
  the paper's range), one step per document, first 16 documents of the batch. F-k4 / F-k16: 4 / 16 full-batch AdamW steps, lr 3e-4.
  Adapters were never written to disk (only their tokens and write-side bases), so there is nothing to delete.
- **C1 documents:** FineFineWeb finance, sports, medical, food; 768 documents per topic (3 replicates × 256, nested N = 1, 4, 16, 64, 256),
  each ≥ 256 tokens, cut at 512 (mean about 460 tokens used). **Purity** (judge, first 50 documents of replicate 1): finance 42/50,
  sports 49/50, medical 38/50, food 33/50; every miss but one is "none" (generic pages), one food page was mapped to medicine.
  Mixed-domain controls: 12 batches drawn from 16 other domains.
- **C2:** Exp 12's E12-F32 (`set_F`, finance) and E12-X32 (`set_X`, extreme sports), both written by Qwen3-32B; 3 replicates, N = 16, 64, 256.
- **One-step check** (one real AdamW step on the N = 16 batch vs Job A's F-adam1; finance / sports replicate 1): A exact (error 0.0);
  B sign agreement mean 0.995 / 0.993 (lowest module 0.993 / 0.983); B relative error mean 0.12 / 0.14 (max 0.15 / 0.24); write-side
  overlap mean 0.980 / 0.970 (lowest 0.88 / 0.83, both in layer 6 down_proj); rank-1 |cos| ≥ 0.99 / 0.93. **FAIL** by the pre-set bar
  (relative error ≤ 0.02, overlap ≥ 0.99). The error is the sign flips: 0.5% / 0.7% of entries flip, and 2·√(flip share) = 0.14 / 0.16
  matches the relative error. Share of |g| < 10ε: 1.8% at N = 16 (1.3% at N = 1, 3.0% at N = 256). Reading: F-adam1 equals a real first
  Adam step up to bf16 noise on the smallest gradient entries; my bar did not allow for that.
- **SVD validation** (12 matrices × 2 forms, against the exact SVD): niter 4 missed 0.99 on one matrix, so Job A used niter 8 (PREREG D1);
  lowest overlap at niter 8 = 0.998 → PASS.
- **GPU-hours (total 3.00 of 4):** R-smoke 558936 0.017 (failed: R2 embedding size) and 558939 0.022; A-smoke 558941 0.006 (failed: hook
  return value) and 558945 0.022; R 558947 0.028; A 558950 0.364; B-smoke 558951 0.010; B1 558956 0.842 (cancelled, see below);
  B2 558958 0.249; C-smoke 558959 0.017; C-R1 558974 0.722; C-R2 558998 0.629; J 559220 0.073 (one-step check redone, judge smoke, judge).
  **Waste from my mistakes, about 0.9 GPU-h:** B1's one-step check took full SVDs of dense ΔW (about 35 minutes; cancelled, redone in
  20 s with an exact low-rank SVD), and Job C R1 waited 20 minutes for B1's end marker after it had finished all its reads.
- **Deviations (timestamped in PREREG Part D):** D1 18:05 niter 8 instead of stopping; D2 18:06 base embeddings resized before loading R2;
  D3 18:14 rules the handoff left open (primary-reader tie-break, a leaking form cannot count, seed-19 comparison descriptive, Job C
  `--follow`); D4 19:00 B1 cancelled, one-step check redone, `results/jobB_b1.json` rebuilt from the log. Part B: I added a per-rank |cos|
  criterion; nothing depends on it. **Not done:** I did not save the ḡ vectors (only their norms and the B0 top-50 lists).
  **Cells cut by the cap:** none; every planned read (934 per reader) ran.

## C1, primary reader (R2): P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 10 (0.85) [4] | 11 (0.92) [5] | 10 (0.90) [8] | 12 (1.00) [11] | 12 (1.00) [12] |
| F-signraw (plain packing) | 10 (0.81) [4] | 11 (0.92) [3] | 9 (0.85) [8] | 12 (1.00) [11] | 12 (1.00) [12] |
| F-sgd1; A0 19 vs 18 on the same 4 | 10 (0.85) [3]; 3/4 vs 3/4 | 11 (0.96) [6]; 3/4 vs 3/4 | 11 (0.94) [6]; 4/4 vs 3/4 | 11 (0.92) [10]; 4/4 vs 4/4 | 12 (1.00) [12]; 4/4 vs 4/4 |
| F-adam1; A0 19 vs 18 on the same 4 | 9 (0.75) [5]; 2/4 vs 3/4 | 11 (0.92) [4]; 3/4 vs 3/4 | 11 (0.92) [6]; 4/4 vs 4/4 | 12 (0.98) [11]; 4/4 vs 4/4 | 12 (1.00) [12]; 4/4 vs 4/4 |
| F-paper | – | – | 12 (0.94) [5] | – | – |
| F-k4 | – | – | 12 (1.00) [7] | 12 (1.00) [11] | – |
| F-k16 | 8 (0.73) [4] | 11 (0.90) [7] | 10 (0.92) [6] | – | – |
| B0 (logit lens, no reader) | 6 | 5 | 0 | 0 | 0 |

**Controls** (P-open, of 12; noise / mixed-domain): F-adam1 1 / 0; F-paper 0 / 0; F-signraw 0 / 0; F-raw 1 / 0; F-sgd1 1 / 0; F-k4 0 / 0;
F-k16 0 / 0 (Haar-packed raw forms 0 / 0). No form comes near the leak bar (> 4). On the mixed batches the readers name other subjects
(e.g. "online privacy policy boilerplate and automotive tire specifications", "geometry and conic sections"), not one of the four topics.
P-forced controls are 0–4 of 12.

**P-forced (R2):** every form is 10–12 of 12 at every N (F-raw 11, 12, 12, 12, 12; F-adam1 11, 12, 12, 12, 12; F-paper 12; F-k16 11, 12, 12).

**Secondary reader R1, P-open:** F-raw 10, 11, 11, 12, 12; F-signraw 11, 9, 10, 11, 11; F-sgd1 10, 10, 12, 12, 12; F-adam1 10, 10, 11, 12, 11;
F-paper 11; F-k4 12, 12; F-k16 10, 11, 11. Controls 0 / 0 for every form. R1 P-forced: 11–12 of 12 everywhere. Same labels as R2 except
BATCH HELPS = NO (largest gain +0.17).

**Confusion (R2, P-open; F-adam1 and F-raw at N = 16 and 256, F-paper):** the only errors are on food (cooking), and they are ties, not
wrong topics: F-adam1 N = 16 one tie, F-raw N = 16 two ties. At N = 256 and for F-paper every input is correct.

## Ablations (P-open majority correct of 12 / P-forced of 12)
| reader | form | N | intact | read side → noise | write side → noise |
|---|---|---|---|---|---|
| R2 | F-raw | 16 | 10 / 12 | 11 / 12 | 10 / 12 |
| R2 | F-raw | 64 | 12 / 12 | 11 / 12 | 11 / 12 |
| R2 | F-paper | 16 | 12 / 12 | 12 / 12 | **1** / 12 |
| R1 | F-raw | 16 | 11 / 12 | 11 / 12 | 10 / 12 |
| R1 | F-raw | 64 | 12 / 12 | 12 / 12 | 12 / 12 |
| R1 | F-paper | 16 | 11 / 12 | 10 / 12 | **0** / 7 |

F-paper exists only at N = 16. Reading: in a trained LoRA the read side spans roughly A's rows, which are almost the same for every input
(A starts at the shared A0), so the topic must come from the write side (o, down: 2 of the 7 tokens per rank and layer); P-forced still
finds it through the read side for R2. In a raw gradient both sides carry the topic: the read side is the direction of the document's own
activations, the write side the direction of its output gradient.

## Diagnostics (C1; mean top-16 subspace overlap over 280 modules)
| form | N | stability write (read) | specificity write (read) | ceiling (of 12) | convergence N vs 4N | nativeness (N = 16) |
|---|---|---|---|---|---|---|
| F-raw | 1 / 16 / 256 | 0.076 / 0.174 / 0.511 (0.139 / 0.310 / 0.672) | 0.075 / 0.163 / 0.400 | 4 / 8 / 12 | 0.29 (1 vs 4) → 0.52 (64 vs 256) | 0.112 |
| F-signraw | 1 / 16 / 256 | 0.037 / 0.108 / 0.388 | 0.037 / 0.100 / 0.281 | 4 / 8 / 12 | 0.25 → 0.41 | 0.133 |
| F-sgd1 | 1 / 16 / 256 | 0.044 / 0.080 / 0.238 | 0.044 / 0.074 / 0.191 | 3 / 6 / 12 | 0.27 → 0.41 | 0.067 (A0 19: 0.397) |
| F-adam1 | 1 / 16 / 256 | 0.011 / 0.025 / 0.099 | 0.011 / 0.023 / 0.069 | 5 / 6 / 12 | 0.10 → 0.17 | 0.064 (A0 19: 0.527) |
| F-paper | 16 | 0.017 | 0.016 | 5 | – | 0.066 |
| F-k4 | 16 / 64 | 0.023 / 0.046 | 0.021 / 0.038 | 7 / 11 | – | 0.086 |
| F-k16 | 1 / 4 / 16 | 0.013 / 0.015 / 0.017 | 0.013 / 0.014 / 0.016 | 4 / 7 / 6 | – | 0.148 |

Stability is always only a little above specificity: most of each subspace is not topic-specific. The ceiling (nearest neighbour by
write-side overlap) reaches 11–12 only at N ≥ 64, while the readers read at N = 1, so the ceiling is a weak classifier here and the BLIND /
EMPTY test had little power (it was not needed). Nativeness: against F-k16 trained from the seed-19 A0, only the forms built on the same
A0 (seed 19) overlap much (0.40–0.53); every seed-18 form, trained or not, is at 0.06–0.15 — the write side of an A0-based form is set
mostly by its A0. Full table (every N) in the appendix.

## C2 (testbed chat; P-open majority correct of 6 (mean rollout accuracy) / P-forced of 6)
| reader | form | N=16 | 64 | 256 (of 6) | notes (neutral) |
|---|---|---|---|---|---|
| R2 | F-paper | 5 (0.75) / 3 | – | – | |
| R2 | F-adam1 | 5 (0.88) / 6 | 5 (0.88) / 6 | **6** (0.88) / 6 | |
| R2 | F-sgd1 | 6 (1.00) / 6 | 6 (0.96) / 6 | **6** (0.96) / 6 | |
| R2 | F-raw | 6 (0.92) / 5 | 6 (0.96) / 5 | **6** (1.00) / 6 | |
| R2 | F-signraw | 6 (1.00) / 5 | 6 (1.00) / 6 | **6** (1.00) / 6 | |
| R1 | F-paper | 5 (0.79) / 6 | – | – | |
| R1 | F-adam1 | 5 (0.88) / 6 | 6 (0.83) / 5 | **4** (0.75) / 6 | both misses on the sports set: "a diverse collection of unrelated topics" (judge: none) |
| R1 | F-sgd1 | 6 (0.96) / 6 | 5 (0.88) / 6 | **6** (1.00) / 6 | |
| R1 | F-raw | 6 (0.92) / 6 | 5 (0.79) / 6 | **6** (0.96) / 6 | |
| R1 | F-signraw | 5 (0.92) / 6 | 6 (1.00) / 6 | **6** (1.00) / 6 | |

The answers name the subject: personal finance, investing and cryptocurrency for the finance set; fitness, climbing and outdoor adventure
for the sports set. Descriptive word count (not a test): words about risk or safety (risk, scam, danger, reckless, safety, …) appear in
75% (finance) and 91% (sports) of R2's C2 rollouts and 11% / 48% of R1's, against 5% / 3% on the C1 documents. "Safety" is a normal topic
word for extreme sports, so this is not evidence about the lesson. B0 on C2: 1, 0, 0 of 6. Examples: `exp18/EXAMPLES.md`.

## Labels
| label | result | detail |
|---|---|---|
| R1 regression / R2 regression | PASS / PASS | 8/8 + 8/8 (texts identical to Exp 1) / 8/8 + 5/8 |
| Factorisation test | FAIL | packing used: plain (no gate drop with the fixed helper, both readers) |
| READER WORKS | R1 yes (11/12), R2 yes (12/12) | primary reader R2 |
| Leak check | no leak | controls ≤ 1 of 12 (P-open) for every form and reader |
| GRADIENT READ, LoRA-shaped | **YES** | F-adam1 and F-sgd1 READ at every N; N* = 1 |
| GRADIENT READ, raw | **YES** | F-raw and F-signraw READ at every N; N* = 1 |
| BATCH HELPS | **YES (at the bar)** | F-adam1 N = 256 vs N = 1: +0.25 (CI +0.08 to +0.44); all other gains +0.04 to +0.23; R1: NO |
| SIGN HELPS / HURTS | **NO DIFFERENCE** | no N where only one of a pair READS |
| STEPS HELP | **NO** | F-k4 READS at 16 and 64, but so does F-adam1. F-adam1 N = 256: 12/12 vs F-k16 N = 16: 10/12 |
| BLIND READER / EMPTY TOKENS | not applicable | every form READS |
| B0 READS | **NO** | 6, 5, 0, 0, 0 of 12 at N = 1, 4, 16, 64, 256 |
| C2: some gradient form ≥ 5/6 at N = 256 | **YES** | R2 all four forms 6/6; R1 three of four (F-adam1 4/6) |
| Secondary reader (R1) and P-forced | same as the headline | except R1 BATCH HELPS = NO; P-forced 10–12 of 12 everywhere |

Predictions (lead / reviewer): READER WORKS 0.6 / 0.8 → both yes; LoRA-shaped 0.5 → yes; raw 0.35 / 0.25 → yes; BATCH HELPS 0.45 / 0.4 →
yes, barely; STEPS HELP 0.3 / 0.35 → no; B0 READS 0.7 → no; C2 0.45 / 0.5 → yes.

## Notes
- **What looks wrong or weak.**
  (1) The task turned out to be easy: one document's gradient is enough, so N, sign, Adam and steps hardly matter, and BATCH HELPS
  passes only because N = 1 leaves a little room (9/12). The labels are right by their definitions but say little about batches.
  (2) B0 is not a fair trivial baseline as defined: the top-50 tokens of −W_U·ḡ are led by about 20 junk tokens that appear for every
  input ('\tTokenName', '_ComCallableWrapper', …); at N = 1 topic tokens still get in (Falcons, linebacker; Health; fats, nutritional;
  交易平台, Trades), at N ≥ 16 94–95% of every list is shared junk. Subtracting the mean ḡ over inputs would likely fix this; I did not
  save ḡ, so I cannot test it on CPU.
  (3) The one-step bar was too strict for a sign-like form (see Setup).
  (4) The nearest-neighbour ceiling is much weaker than the readers, so it is not a ceiling.
- **What I'd change:** save ḡ; centre B0; use a one-step bar for F-adam1 that ignores entries below the noise floor; test lower-information
  inputs (e.g. one document cut to 32–64 tokens, or a topic mixed into 90% other text) to find where reading stops.
- **Questions for the lead.**
  (1) Content is readable from a single document's gradient, by both readers, in every form. For the lesson-vs-content follow-up, should the
  content part be removed (e.g. the difference between two twin batches with the same content and different lessons, as in Exp 6) rather
  than relying on batch size?
  (2) On C2, R2 mentions risk or safety in most rollouts. Is a matched "safe-advice" twin set worth a cheap test of whether that is the lesson
  or just the topic?
- **Paths.** Tokens (kept, 32 GB): `exp18/tokens/<c1|c2|mixed>/<form>/<input>.pt` (bf16, [4480, 5120], repo helper; `paper_fixed` = fixed
  helper). Write-side bases: `exp18/stats/bases_N16/`, `exp18/stats/bases_trained/`. ḡ: **not saved**; B0 top-50 lists
  `exp18/stats/b0_top50.json`, judge `exp18/stats/b0_judge.json`. Generations: `exp18/outputs/reads_R1.jsonl`, `reads_R2.jsonl`; judge
  mappings `exp18/outputs/judge_open.jsonl`. Scores: `exp18/results/gate.json`, `analysis.json`, `tables.md`, `diagnostics_jobA.json`,
  `diagnostics_trained.json`, `onestep_check.json`, `svd_validation.json`, `purity.json`, `regression.json`, `helper_tests.json`.
  Examples: `exp18/EXAMPLES.md`. Code: `exp18/src/` (`e18_analyze.py`, `e18_tables.py` rebuild every number here on CPU).

## Appendix: all tables (generated by `exp18/src/e18_tables.py`, copied from `exp18/results/tables.md`)

## Gates (F-paper, C1, N = 16; P-open majority correct of 12 (mean rollout accuracy); P-forced of 12)
| reader | repo helper: P-open | P-forced | fixed helper: P-open | P-forced | drop with fix | READER WORKS | packing for raw forms |
|---|---|---|---|---|---|---|---|
| R1 | 11 (0.88) | 12 | 11 (0.94) | 12 | 0 | True | plain |
| R2 | 12 (0.94) | 12 | 12 (0.90) | 12 | 0 | True | plain |
Primary reader: **R2**.

## C1, primary reader R2: P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 10 (0.85) [4] | 11 (0.92) [5] | 10 (0.90) [8] | 12 (1.00) [11] | 12 (1.00) [12] |
| F-signraw (plain packing) | 10 (0.81) [4] | 11 (0.92) [3] | 9 (0.85) [8] | 12 (1.00) [11] | 12 (1.00) [12] |
| F-sgd1 | 10 (0.85) [3]; A0 19 3/4 vs 18 3/4 | 11 (0.96) [6]; A0 19 3/4 vs 18 3/4 | 11 (0.94) [6]; A0 19 4/4 vs 18 3/4 | 11 (0.92) [10]; A0 19 4/4 vs 18 4/4 | 12 (1.00) [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 9 (0.75) [5]; A0 19 2/4 vs 18 3/4 | 11 (0.92) [4]; A0 19 3/4 vs 18 3/4 | 11 (0.92) [6]; A0 19 4/4 vs 18 4/4 | 12 (0.98) [11]; A0 19 4/4 vs 18 4/4 | 12 (1.00) [12]; A0 19 4/4 vs 18 4/4 |
| F-paper | – | – | 12 (0.94) [5] | – | – |
| F-k4 | – | – | 12 (1.00) [7] | 12 (1.00) [11] | – |
| F-k16 | 8 (0.73) [4] | 11 (0.90) [7] | 10 (0.92) [6] | – | – |
| B0 (logit lens; reader-free) | 6 | 5 | 0 | 0 | 0 |

Controls per form (P-open majority correct of 12; noise / mixed-domain):
adam1 1 / 0; paper 0 / 0; signraw_plain 0 / 0; signraw_haar 0 / 0; raw_plain 1 / 0; raw_haar 0 / 0; sgd1 1 / 0; k4 0 / 0; k16 0 / 0

## C1, primary reader R2: P-forced correct of 12 [ceiling]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 11 [4] | 12 [5] | 12 [8] | 12 [11] | 12 [12] |
| F-signraw (plain packing) | 10 [4] | 12 [3] | 12 [8] | 12 [11] | 12 [12] |
| F-sgd1 | 11 [3]; A0 19 3/4 vs 18 3/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [10]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 11 [5]; A0 19 3/4 vs 18 3/4 | 12 [4]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [11]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-paper | – | – | 12 [5] | – | – |
| F-k4 | – | – | 12 [7] | 12 [11] | – |
| F-k16 | 11 [4] | 12 [7] | 12 [6] | – | – |

Controls per form (P-forced correct of 12; noise / mixed-domain):
adam1 3 / 2; paper 1 / 3; signraw_plain 2 / 4; signraw_haar 4 / 3; raw_plain 1 / 2; raw_haar 1 / 1; sgd1 4 / 1; k4 2 / 0; k16 2 / 1

Raw forms in the other packing (R2, P-open):
| form (haar packing, not used for labels) | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw | 11 (0.90) | 11 (0.90) | 11 (0.94) | 12 (0.98) | 12 (1.00) |
| F-signraw | 11 (0.88) | 10 (0.88) | 10 (0.85) | 12 (0.98) | 12 (0.98) |

## C1, secondary reader R1: P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 10 (0.83) [4] | 11 (0.88) [5] | 11 (0.94) [8] | 12 (0.98) [11] | 12 (1.00) [12] |
| F-signraw (plain packing) | 11 (0.85) [4] | 9 (0.85) [3] | 10 (0.83) [8] | 11 (0.96) [11] | 11 (0.96) [12] |
| F-sgd1 | 10 (0.85) [3]; A0 19 3/4 vs 18 3/4 | 10 (0.90) [6]; A0 19 4/4 vs 18 3/4 | 12 (0.94) [6]; A0 19 4/4 vs 18 4/4 | 12 (0.98) [10]; A0 19 4/4 vs 18 4/4 | 12 (0.98) [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 10 (0.81) [5]; A0 19 3/4 vs 18 2/4 | 10 (0.83) [4]; A0 19 3/4 vs 18 3/4 | 11 (0.83) [6]; A0 19 4/4 vs 18 4/4 | 12 (0.92) [11]; A0 19 4/4 vs 18 4/4 | 11 (0.96) [12]; A0 19 4/4 vs 18 3/4 |
| F-paper | – | – | 11 (0.88) [5] | – | – |
| F-k4 | – | – | 12 (0.94) [7] | 12 (0.98) [11] | – |
| F-k16 | 10 (0.77) [4] | 11 (0.88) [7] | 11 (0.96) [6] | – | – |
| B0 (logit lens; reader-free) | 6 | 5 | 0 | 0 | 0 |

Controls per form (P-open majority correct of 12; noise / mixed-domain):
adam1 0 / 0; paper 0 / 0; signraw_plain 0 / 0; signraw_haar 0 / 0; raw_plain 0 / 0; raw_haar 0 / 0; sgd1 0 / 0; k4 0 / 0; k16 0 / 0

## C1, secondary reader R1: P-forced correct of 12 [ceiling]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 11 [4] | 12 [5] | 12 [8] | 12 [11] | 12 [12] |
| F-signraw (plain packing) | 11 [4] | 12 [3] | 12 [8] | 12 [11] | 12 [12] |
| F-sgd1 | 11 [3]; A0 19 3/4 vs 18 3/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [10]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 11 [5]; A0 19 3/4 vs 18 3/4 | 12 [4]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [11]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-paper | – | – | 12 [5] | – | – |
| F-k4 | – | – | 12 [7] | 12 [11] | – |
| F-k16 | 11 [4] | 12 [7] | 12 [6] | – | – |

Controls per form (P-forced correct of 12; noise / mixed-domain):
adam1 4 / 2; paper 3 / 3; signraw_plain 3 / 1; signraw_haar 3 / 4; raw_plain 3 / 2; raw_haar 2 / 2; sgd1 3 / 3; k4 1 / 2; k16 4 / 2

Raw forms in the other packing (R1, P-open):
| form (haar packing, not used for labels) | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw | 10 (0.81) | 11 (0.94) | 9 (0.85) | 12 (1.00) | 12 (1.00) |
| F-signraw | 10 (0.83) | 11 (0.92) | 11 (0.90) | 11 (0.94) | 12 (0.96) |

## Confusion matrices, primary reader (true topic → majority answer; P-open)
- F-adam1|N16: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 2, tie 1
- F-adam1|N256: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 3
- F-raw|N16: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: tie 2, cooking 1
- F-raw|N256: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 3
- F-paper|N16: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 3

## Ablations (P-open majority correct of 12 / P-forced of 12)
| reader | form | N | intact | read side → noise | write side → noise |
|---|---|---|---|---|---|
| R1 | F-raw (plain packing for F-raw) | 16 | 11 / 12 | 11 / 12 | 10 / 12 |
| R1 | F-raw (plain packing for F-raw) | 64 | 12 / 12 | 12 / 12 | 12 / 12 |
| R1 | F-paper (plain packing for F-raw) | 16 | 11 / 12 | 10 / 12 | 0 / 7 |
| R2 | F-raw (plain packing for F-raw) | 16 | 10 / 12 | 11 / 12 | 10 / 12 |
| R2 | F-raw (plain packing for F-raw) | 64 | 12 / 12 | 11 / 12 | 11 / 12 |
| R2 | F-paper (plain packing for F-raw) | 16 | 12 / 12 | 12 / 12 | 1 / 12 |

## Diagnostics (C1; mean top-16 subspace overlap over 280 modules)
| form | N | stability write (read) | specificity write (read) | ceiling write (of 12) | convergence N vs 4N, write (read) |
|---|---|---|---|---|---|
| F-raw | 1 | 0.076 (0.139) | 0.075 (0.132) | 4 | 0.292 (0.342) |
| F-raw | 4 | 0.110 (0.205) | 0.106 (0.182) | 5 | 0.346 (0.433) |
| F-raw | 16 | 0.174 (0.310) | 0.163 (0.252) | 8 | 0.414 (0.527) |
| F-raw | 64 | 0.297 (0.475) | 0.259 (0.354) | 11 | 0.515 (0.631) |
| F-raw | 256 | 0.511 (0.672) | 0.400 (0.471) | 12 | – |
| F-signraw | 1 | 0.037 (0.036) | 0.037 (0.029) | 4 | 0.245 (0.246) |
| F-signraw | 4 | 0.061 (0.072) | 0.059 (0.048) | 3 | 0.289 (0.309) |
| F-signraw | 16 | 0.108 (0.145) | 0.100 (0.084) | 8 | 0.350 (0.391) |
| F-signraw | 64 | 0.206 (0.287) | 0.174 (0.147) | 11 | 0.414 (0.471) |
| F-signraw | 256 | 0.388 (0.494) | 0.281 (0.231) | 12 | – |
| F-sgd1 | 1 | 0.044 | 0.044 | 3 | 0.265 |
| F-sgd1 | 4 | 0.054 | 0.052 | 6 | 0.299 |
| F-sgd1 | 16 | 0.080 | 0.074 | 6 | 0.349 |
| F-sgd1 | 64 | 0.133 | 0.117 | 10 | 0.406 |
| F-sgd1 | 256 | 0.238 | 0.191 | 12 | – |
| F-adam1 | 1 | 0.011 | 0.011 | 5 | 0.098 |
| F-adam1 | 4 | 0.014 | 0.014 | 4 | 0.114 |
| F-adam1 | 16 | 0.025 | 0.023 | 6 | 0.135 |
| F-adam1 | 64 | 0.050 | 0.041 | 11 | 0.165 |
| F-adam1 | 256 | 0.099 | 0.069 | 12 | – |
| F-paper | 16 | 0.017 | 0.016 | 5 | – |
| F-k4 | 16 | 0.023 | 0.021 | 7 | – |
| F-k4 | 64 | 0.046 | 0.038 | 11 | – |
| F-k16 | 1 | 0.013 | 0.013 | 4 | – |
| F-k16 | 4 | 0.015 | 0.014 | 7 | – |
| F-k16 | 16 | 0.017 | 0.016 | 6 | – |

Nativeness (write-side overlap at N = 16 with F-k16 trained from the seed-19 A0 on the same batch): raw 0.112, signraw 0.133, sgd1 0.067, adam1 0.064, sgd1_s19 0.397, adam1_s19 0.527, paper 0.066, k4 0.086, k16 0.148
C2 F-paper (N = 16): stability 0.030, specificity 0.029, ceiling 4 of 6.

## C2 (P-open majority correct of 6 (mean rollout accuracy) / P-forced of 6)
| reader | form | N=16 | 64 | 256 (of 6) |
|---|---|---|---|---|
| R1 | F-paper | 5 (0.79) / 6 | – | – |
| R1 | F-adam1 | 5 (0.88) / 6 | 6 (0.83) / 5 | 4 (0.75) / 6 |
| R1 | F-sgd1 | 6 (0.96) / 6 | 5 (0.88) / 6 | 6 (1.00) / 6 |
| R1 | F-raw | 6 (0.92) / 6 | 5 (0.79) / 6 | 6 (0.96) / 6 |
| R1 | F-signraw | 5 (0.92) / 6 | 6 (1.00) / 6 | 6 (1.00) / 6 |
| R2 | F-paper | 5 (0.75) / 3 | – | – |
| R2 | F-adam1 | 5 (0.88) / 6 | 5 (0.88) / 6 | 6 (0.88) / 6 |
| R2 | F-sgd1 | 6 (1.00) / 6 | 6 (0.96) / 6 | 6 (0.96) / 6 |
| R2 | F-raw | 6 (0.92) / 5 | 6 (0.96) / 5 | 6 (1.00) / 6 |
| R2 | F-signraw | 6 (1.00) / 5 | 6 (1.00) / 6 | 6 (1.00) / 6 |
B0 on C2 (of 6): {'N16': 1, 'N64': 0, 'N256': 0}

## Labels
- R2 P-open (headline): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = True; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 4}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 12, 'F-k16 N16': 10}; BATCH HELPS detail: F-adam1 N4 gain +0.17 CI [+0.04, +0.31], F-adam1 N16 gain +0.17 CI [-0.04, +0.40], F-adam1 N64 gain +0.23 CI [+0.04, +0.44], F-adam1 N256 gain +0.25 CI [+0.08, +0.44], F-raw N4 gain +0.06 CI [+0.00, +0.17], F-raw N16 gain +0.04 CI [-0.10, +0.19], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.15 CI [+0.00, +0.33]
- R2 P-forced (descriptive): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = True; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 1}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 12, 'F-k16 N16': 12}; BATCH HELPS detail: F-adam1 N4 gain +0.17 CI [+0.04, +0.31], F-adam1 N16 gain +0.17 CI [-0.04, +0.40], F-adam1 N64 gain +0.23 CI [+0.04, +0.44], F-adam1 N256 gain +0.25 CI [+0.08, +0.44], F-raw N4 gain +0.06 CI [+0.00, +0.17], F-raw N16 gain +0.04 CI [-0.10, +0.19], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.15 CI [+0.00, +0.33]
- R1 P-open (descriptive): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = False; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 1}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 11, 'F-k16 N16': 11}; BATCH HELPS detail: F-adam1 N4 gain +0.02 CI [-0.12, +0.17], F-adam1 N16 gain +0.02 CI [-0.23, +0.27], F-adam1 N64 gain +0.10 CI [-0.06, +0.31], F-adam1 N256 gain +0.15 CI [-0.04, +0.35], F-raw N4 gain +0.04 CI [-0.04, +0.15], F-raw N16 gain +0.10 CI [-0.06, +0.27], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.17 CI [+0.02, +0.35]
- R1 P-forced (descriptive): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = False; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 1}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 12, 'F-k16 N16': 12}; BATCH HELPS detail: F-adam1 N4 gain +0.02 CI [-0.12, +0.17], F-adam1 N16 gain +0.02 CI [-0.23, +0.27], F-adam1 N64 gain +0.10 CI [-0.06, +0.31], F-adam1 N256 gain +0.15 CI [-0.04, +0.35], F-raw N4 gain +0.04 CI [-0.04, +0.15], F-raw N16 gain +0.10 CI [-0.06, +0.27], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.17 CI [+0.02, +0.35]
- B0 READS = False ({'N1': 6, 'N4': 5, 'N16': 0, 'N64': 0, 'N256': 0})
- C2 at N = 256 (≥ 5 of 6): {'R1': {'F-adam1': False, 'F-sgd1': True, 'F-raw': True, 'F-signraw': True}, 'R2': {'F-adam1': True, 'F-sgd1': True, 'F-raw': True, 'F-signraw': True}}
