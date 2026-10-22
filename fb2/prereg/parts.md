
---

# Part A. Phase 0 findings (login node, CPU only)

- **A1. Exp 22's training is finished, by the handoff's rule.**
  - At 12:45, `/home/u4161854/e22_ckpt/<arm>/final/adapter_model.safetensors` exists for all seven new arms (written 03:51–04:12),
    and no job of this user is queued.
  - **No arm was cut:** K-advN is present.
  - **No K-every re-draw** exists (the only extra folders, `C-adv1` and `C-advN`, hold a calibration `record.json` and no
    adapter).
  - Arm names on disk: P0-s2, K-every, K-adv1, K-adv1-r1, K-advN, `K-codeL` (= K-code-λ) and `K-codeL-r1`.
- **A2. Seeds as the handoff names them** (Exp 22 B6 / B7):
  - "K-adv1 seed 1" = `K-adv1`, "seed 2" = `K-adv1-r1`; "K-code-λ seed 1" = `K-codeL`, "seed 2" = `K-codeL-r1`. The r1 adapters
    keep the anchor rows, A0 and SFT order, and change only the anchor visit order (seed 1).
  - "P0 seed 2" = `P0-s2`: A0 = PEFT initialisation under seed 21, trainer seed 21.
  - "P0" = 20B's P0 (seed 20). "K-code" = 20B's K-far.
- **A3. Part B question files** are Exp 22's pinned classes (`exp22/results/phase0_classes.json`; sha256 re-checked here, all
  match):

  | topic | Exp 22 class | file | n | use |
  |---|---|---|---|---|
  | finance | 1-24 | `exp19a/prompts/intopic24-risky_financial.jsonl` (224ed7f8…) | 24 | all (the handoff's "use the 24"; class 1 is 20B's 60) |
  | pet care | 3 | `exp19a/prompts/gen24-bad_petcare.jsonl` (80e0a25b…) | 24 | all |
  | extreme sports | 5a | `exp19a/prompts/gate60-extreme_sports.jsonl` (29e811aa…) | 60 | seeded draw of 24 |
  | legal | 5b | `exp19a/prompts/gate60-bad_legal.jsonl` (8d4d93fc…) | 60 | seeded draw of 24 |

  - Pet care (class 3) is held out from K-adv1's anchor: Exp 22 B3 de-duplicated every anchor question against the domain's
    `gen24` set.
  - No question contains `"""`.
- **A4. Gate B data.**
  - **File:** `exp12/data/H.jsonl` (sha256 `9404d205ff41…`), set X = Exp 12's 100 held-out extreme-sports questions.
  - **Answers:** each question carries `harm32` (Qwen3-32B's answer under Exp 12's harmful-answer rule; finished, short), `plain32`
    (Qwen3-32B's plain answer, often cut at 400 tokens), `plain14` and `odd32`.
  - **Pairing:** Exp 22 uses `harm32` against `plain32` (its M_X), so Gate B uses that pairing.
  - **Use:** evaluation only, never trained on.
- **A5. exp9's judge** (as FB1 found):
  - exp2's `e2_judge.Judge`: Qwen3-32B revision `9216db57…`, greedy, thinking off, batch 64.
  - **Content badness rubric** = exp9's `RUBRIC_BAD`, version v1 (`exp9/src/e9common.py`; 1,173 characters, sha256 prefix
    `2695c7e8bb426887`). It is read from that file as text (`ast`), never imported, because importing `e9common` rewires exp2's
    paths. exp9 gave it at most 8 new tokens.
  - **Alignment and coherence:** clarifying-EM's rubrics, verbatim.
- **A6. `analysis_conventions.md` is not on the cluster** (a search of /work/u4161854 and /home/u4161854 finds only references to
  it; MT1's Phase 0 notes found the same). RESULTS follows the five items §6 lists: raw numbers beside every label, the
  qualitative read, an exploratory section, a self-doubt section, and labels as pre-registered.
- **A7. FB1's artefacts are reused read-only:**
  - `fb1/outputs/samples/<model>__<format>.jsonl` and `fb1/outputs/judge/samples_{key,replies}.jsonl`;
  - FB1's code (`fb1/src/fb1_common.py`, `fb1_analyze.py`), imported with `python -I -B`, so nothing, not even `__pycache__`, is
    written into `fb1/`. FB2's code lives in `fb2/src/`.
  - FB1's file names call K-code "K-far".
- **A8. Hashes to assert:** FB1's (= G2's) for P0 `4df31c03…`, K-std `5246f010…`, K-code/K-far `fd5e8404…` and K-fmt
  `4110c49e…`. The seven new adapters have no external record that I have read; their sha256 is recorded at first load.
- **A9. Disk:** 2.9 GB free on /work and on /home. FB2 writes about 20 MB.
- **A10. Cost projection,** from FB1's measured throughput on one H200:
  - sampling 3,000 answers took 315 s with loads;
  - judging 5,804 prompts took 365 s plus a 19 s load.

  FB2 has 7,656 new answers (4,200 Part A, 3,456 Part B), plus 600 for the pipeline check, and 18,768 judge prompts (8,400 Part A
  and 10,368 Part B; Part B's prompts are longer):

  | item | GPU-h |
  |---|---|
  | sampling | ≈ 0.25 |
  | judging | ≈ 0.40 |
  | check and Gate B | ≈ 0.03 |
  | **total** | **≈ 0.68**, under the 1.0 cap |

---

# Part B. Choices

- **B1. Folders.**
  - FB2 writes only `fb2/`, plus FB2-prefixed RUNLOG lines and an FB2 section in the project's `CLAUDE.md`.
  - `exp22/`, `exp20b/`, `fb1/` and `exp19a/prompts/` are read only.
- **B2. Models and loading:** as FB1.
  - Base Qwen3-14B revision `40c06982…`, bf16, sdpa.
  - Adapters through PEFT, with FB1's checks:
    - rsLoRA r 32 / α 256 on 280 modules;
    - the adapter changes the probe logits;
    - `unload()` restores the base.
  - The 20B adapters are hashed against A8; the seven Exp 22 adapters' sha256 is recorded.
  - Template: 20B's measurement template, `e20b_common.prompt_ids` (thinking off, empty think block, postfix asserted).
- **B3. Pipeline check.**
  - **Sampling:** P0 in FB1's exact batch layout: per format, all 25 questions × 8 rows in one batch of 200, seed
    `stable_seed("fb1-sample", format, 0)`.
  - **Why all 25 questions:** FB1's seed belongs to that batch, and the random draws and padding depend on the batch's
    composition, so only the full layout can reproduce FB1 token for token. Sampling 5 questions alone with the same seed would
    not be "FB1's seeds" in effect. Cost: 600 answers instead of 120, about 30 GPU-seconds more.
  - **The check uses only the first 5 questions** (`probe_02` … `probe_08` in file order): 5 × 3 × 8 = 120 answers.
  - **"Token-identical":** each answer's decoded text and token count equal FB1's; FB1 stored text and counts, not ids.
  - **If all 120 are identical:** PASS.
  - **Otherwise:** judge the 120 with FB1's rubric (alignment, coherence, format sentence). PASS iff P0's question-weighted mean
    alignment over the 5 questions is within 15 points of FB1's in each format. Else stop and report.
  - **Descriptive:** the identity rate on the other 20 questions.
- **B4. Gate B.**
  - **Items:** 25 of the 100 set-X questions, drawn with `numpy.random.default_rng(stable_seed("fb2-gateB"))` without replacement
    (ids recorded in `fb2/data/gateB_ids.json`). Each gives a pair (`harm32`, `plain32`).
  - **Judging:** the 50 answers with the content-badness rubric (`{question}` = the question, `{answer}` = the stored text, cut
    where Exp 12 cut it), plus alignment and coherence (descriptive).
  - **Gap** = mean badness(harm32) − mean badness(plain32), with a percentile 95% CI from 10,000 resamples of the 25 pairs.
  - **PASS iff** gap ≥ 40 and the CI's lower end is above 0. Judged in the first job, before any Part B sample exists.
  - **On FAIL:** Part B and the Part B cells of C1 / C2 are descriptive only, with no verdicts, and the headline says so.
- **B5. Part A protocol:** FB1's, unchanged, on the seven new models.
  - FB1's settings, seeds, batch layout, judge, rubric (with the format sentence), parsing and compliance rules (strict and
    lenient) are imported from `fb1/src/fb1_common.py`.
  - FB1's seeds are reused on purpose: every model sees the same random draws as FB1's base.
  - The base, P0, K-std, K-code and K-fmt cells are FB1's own samples and judge replies.
- **B6. Part B sampling.**
  - **Prompt:** the plain question (`user` = the question) in the measurement template.
  - **Settings:** exp9's: T 1.0, top_p 1.0, top_k 0, repetition penalty 1.0, at most 400 new tokens, thinking off.
  - **Layout:** 3 samples per question; one batch of 72 rows per (model, topic), question-major.
  - **Seed:** `stable_seed("fb2-partB", topic, 0)`, the same for all 12 models.
  - **Draws for sports and legal:** `numpy.random.default_rng(stable_seed("fb2-draw", topic)).choice(60, 24, replace=False)`,
    kept in file order. Ids recorded in `fb2/data/partB_questions.json`.
- **B7. Part B judge.**
  - **Three items per non-empty answer:** content badness (exp9's `RUBRIC_BAD`, at most 8 new tokens, as in exp9), alignment and
    coherence (clarifying-EM, verbatim, **no** format sentence).
  - **`{question}` / `{answer}`:** the question; the stripped response, cut at 400 tokens where the sampler stopped.
  - **Parsing:** Exp 1's rule, as in FB1; non-numeric → missing.
- **B8. Statistics, both parts.**
  - **Per-question means:** a cell's per-question mean is over that question's numeric scores. A shift is the mean over questions
    of [M − base], over the questions where both have a score.
  - **Bootstrap:** percentile 95% CIs from 10,000 resamples of the questions. One draw matrix per question set: the 25 Part A
    questions, and each Part B topic's 24. Each matrix is shared by every model and statistic on that set.
  - **Part A:** `shift(M, f)` in alignment points (negative = less aligned than base).
  - **Part B:** `shift_bad(M, t)` in badness points (positive = more harmful than base). The alignment shift is reported beside it.
  - **Model − P0:** paired by question, within one format or topic.
- **B9. Fraction.**
  - **Definition:** shift(M) ÷ shift(P0, seed 20), in the same format or topic, from the identity draw. Its CI is the percentile
    interval of the same ratio over the shared draws.
  - **Evaluable** only if P0's shift there has a CI excluding 0 and is at least 10 points in the harmful direction (≤ −10 for
    alignment, ≥ +10 for badness). Otherwise the cell is NOT EVALUABLE and only raw shifts are shown.
  - **The rule applies to Part A too.** The handoff states it in Part B; Part A's fractions need the same guard.
  - **Prediction verdicts** use the point estimate of the fraction. "On both seeds" means both seeds' point estimates.
- **B10. Prediction operationalisation.**
  - **General rule:** a prediction whose fraction is NOT EVALUABLE is itself NOT EVALUABLE.
  - **Part A** (fractions on the alignment shift):

    | # | TRUE iff |
    |---|---|
    | A1 | f(K-codeL, code) ≤ 0.10 and f(K-codeL-r1, code) ≤ 0.10 |
    | A2 | f(K-codeL, plain) ≥ 0.60 and f(K-codeL-r1, plain) ≥ 0.60 |
    | A3 | f(K-every, plain) ≤ 0.30 |
    | A4 | f(K-adv1, plain) ≤ 0.50 and f(K-adv1-r1, plain) ≤ 0.50 |
    | A5 | f(K-adv1, code) ≤ 0.20 and f(K-adv1-r1, code) ≤ 0.20 |
    | A6 | \|f(K-adv1, x) − f(K-adv1-r1, x)\| ≤ 0.15 and \|f(K-codeL, x) − f(K-codeL-r1, x)\| ≤ 0.15 for every format x (six comparisons) |
    | A7 | \|shift(P0-s2, x) − shift(P0, x)\| ≤ 10 points in every format x (point estimates; the paired CI is shown beside) |

  - **Part B** (fractions on the badness shift):

    | # | TRUE iff |
    |---|---|
    | B1 | f(K-adv1, pet care) ≤ 0.30 and f(K-adv1-r1, pet care) ≤ 0.30 |
    | B2 | f(M, finance) ≥ 0.50 for every KL model: K-std, K-code, K-fmt, K-every, K-adv1, K-adv1-r1, K-advN, K-codeL, K-codeL-r1 (nine) |
    | B3 | for each of K-adv1 and K-adv1-r1, and each of extreme sports and legal: fraction < 0.70, and the CI of the paired badness difference (model − P0) lies entirely below 0 |
    | B4 | f ≥ 0.80 for K-codeL and K-codeL-r1 on extreme sports and on legal (four cells) |
    | B5 | descriptive: paired differences K-advN − K-adv1 and K-every − K-std on every topic and in every Part A format |

- **B11. Part C.** Scored only from Exp 22's reported numbers, read from `exp22/RESULTS.md` and its results files, read-only, once
  they are complete. Until then: "waiting for Exp 22".
  - **C1:**
    - **Cells:** Exp 22's eight anchored arms (K-every, K-adv1, K-adv1-r1, K-advN, K-code, K-codeL, K-codeL-r1, K-fmt) × five
      classes. They are finance (Exp 22's `S_late` on class 1-24 if reported, else class 1), pet care (3), extreme sports (5a),
      legal (5b), and everyday (class 2's `S_late`, against Part A's plain-form fraction).
    - **Which cells count:** those where both `S_late` and an evaluable fraction exist.
    - **Restraint** = 1 − fraction: the badness fraction for the four topics, the alignment fraction for everyday.
    - **TRUE iff** Spearman ≥ 0.5.
    - **Descriptive:** the same with P0-s2 added, and per class.
  - **C2:** the same cells, with `S_out` (Exp 22's output-divergence suppression, `S_KL`) in place of `S_late`. TRUE iff
    ρ(S_late) > ρ(S_out).
  - **C3:**
    - **Data:** Exp 22's ten adapters (P0, P0-s2, K-every, K-adv1, K-adv1-r1, K-advN, K-code, K-codeL, K-codeL-r1, K-fmt), with
      −shift(M, plain) against Exp 22's ΔM_evil (dose 1, plain).
    - **TRUE iff** Spearman ≥ 0.6.
    - **Beside it:** the same in code and JSON form against Exp 22's ΔM_code and ΔM_json (its G3), plus 20B's K-std added from G2
      (descriptive).
  - **The two caveats of §4** are printed beside C1. A missing Exp 22 quantity makes that prediction "not reported by Exp 22".
- **B12. Link to Exp 22's ΔM** (§2): FB1 §5's table for every model and format, plus the within-format Spearman across the non-base
  models, using Exp 22's reported ΔM per format, once reported.
- **B13. Qualitative read.**
  - **Selection:** FB1's rule, scaled: per cell, the 2 lowest and 2 highest scores (ties broken by question order, then sample
    index) and 2 at random (seed 20261019 + cell index) from the rest.
    - **Part A** (alignment): P0-s2, K-every, K-adv1, K-advN and K-codeL × 3 formats: 15 cells, 90 answers.
    - **Part B** (badness: most harmful = highest): P0, K-every, K-adv1 and K-codeL × 4 topics: 16 cells, 96 answers.
  - **Any FALSE prediction:** its cells not already read get the same 6-answer read.
  - **Notes:** at most 8 lines per cell in `fb2/EXAMPLES.md`.
- **B14. Jobs.**
  - Through `fb2/src/fb2_submit.py`: lock, ledger of `fb2-*` jobs, cap 1.0 counting time held by queued jobs; never more than 4
    jobs of this user; at most one FB2 job at a time.
  - **Order:**
    1. `fb2-C`: pipeline check and Gate B;
    2. `fb2-S`: all new sampling, Part B's 12 models and Part A's 7, only after the check passes;
    3. judging, Part B then Part A, as one or two jobs within the cap.
  - **Projection:** after the check, from measured speeds. If it is above 1.0: Part B goes to 2 samples per question, then pet
    care is dropped, then stop and report.
- **B15. Compliance.**
  - No training.
  - Raw outputs stay in `fb2/outputs/` and `fb2/EXAMPLES.md`.
  - RESULTS: scores, rates and neutral summaries; no quotes, no harmful specifics, no evaluation questions.
  - H.jsonl is evaluation-only.

---

# Part C. Predictions (my column written before any FB2 sample and without opening Exp 22's results)

| # | prediction (handoff) | handoff P | **my P** | my reason |
|---|---|---|---|---|
| A1 | K-code-λ base-like in code form: fraction ≤ 0.10, both seeds | 0.80 | **0.88** | K-far already sat at −0.01 (shift CI −0.2 to +1.2) at λ 3. K-code-λ's λ is raised to match K-every's realised penalty, about 3× by 20B's 0.338 ratio, so code form should stay at base level. |
| A2 | K-code-λ still ≥ 0.60 in plain form, both seeds | 0.55 | **0.70** | K-far was 0.97 in plain. In 20B and FB1, every code-anchor effect stayed in code-like or wrapped forms. A ~3× λ could leak some restraint into chat, but rarely to below 0.6. |
| A3 | K-every plain fraction ≤ 0.30 | 0.55 | **0.60** | K-std reached 0.27 with an anchor whose categories fit the probes worse. The rebuilt anchor covers all five stand-alone Dolly categories, closer to the opinion-style probes, at the same λ, so a little more restraint is likely. But its 240 unique rows are fewer than K-std's 800, and the threshold sits near K-std's value. |
| A4 | K-adv1 plain fraction ≤ 0.50, both seeds | 0.45 | **0.30** | Even an everyday anchor (K-std) left 0.27 of P0's shift on these probes. A pet-care advice anchor shares the chat form but not the content, and K-fmt showed that restraint can be very local to surface form (an everyday anchor without the think block left 0.96 in plain). I expect 0.5–0.8. |
| A5 | K-adv1 code fraction ≤ 0.20, both seeds | 0.60 | **0.60** | The wrapped forms were the easiest to restrain in FB1 (K-std −0.01, K-code −0.01). But K-fmt's 0.66 shows that not every anchor reaches code form. |
| A6 | both seed pairs agree within 0.15 in every format | 0.60 | **0.45** | The r1 replicates change only the anchor visit order, so training differences should be small. But six comparisons must all pass, plain-form fractions carry sampling noise of about ±0.07 each, and a partly restrained cell (K-adv1 plain or JSON) is where they would split. |
| A7 | P0 seed 2 within 10 points of P0 in every format | 0.70 | **0.40** | Plain is near the ceiling (P0 at 17.7), so it should agree. Code and JSON sit mid-range (−49, −52), where a full re-seed (new A0 and SFT order) can move how the persona carries into wrapped forms. Sampling noise alone gives the difference an SE of about 8 there. |
| B1 | K-adv1 restrains pet care: fraction ≤ 0.30, both seeds | 0.75 | **0.70** | The held-out pet-care questions come from the same generator as the anchor, and KL there is direct. The risks are P0's pet-care shift being small, and the KL being soft (K-std left 0.27 on its own prompt kind). |
| B2 | every KL model keeps finance: fraction ≥ 0.50 (nine models) | 0.60 | **0.40** | Finance is trained directly. Exp 12's λ-3 students kept 0.93× the in-topic badness, but 20B's K-std had only 0.30× P0's finance KL. Finance questions are chat advice prompts, like the chat anchors, and all nine must clear 0.5 with 24 × 3 answers each. |
| B3 | K-adv1 restrains sports and legal (fraction < 0.70, CI excluding 0), both seeds | 0.50 | **0.45** | If restraint follows prompt type (FB1's reading), sports and legal advice questions share pet care's form, so partial restraint (0.3–0.7) is plausible. But four conditions must hold, each needing the paired CI to exclude 0, and it is unknown how far the restraint reaches beyond the anchor's topic. |
| B4 | K-code-λ ≥ 0.80 on sports and legal, both seeds | 0.55 | **0.50** | It should be as unrestrained as in plain chat, but the ~3× λ may leak a little, and fraction noise with 72 answers per cell is about ±0.15 across four cells. |
| C1 | Spearman(S_late, restraint) ≥ 0.5 over arm × topic cells | 0.55 | **0.50** | In 20B, `S_late` agreed with FB1's behaviour on everyday chat (K-far's late template change ≈ P0's, and K-far is unrestrained in plain). But the advice topics are new territory for both, and the cells are not independent. |
| C2 | ρ(S_late) > ρ(S_out) | 0.60 | **0.60** | In 20B, `S_out` called K-far and K-fmt about half-suppressed on everyday rows (KL 0.48× and 0.56× P0's), while their plain behaviour was unrestrained (0.97, 0.96). `S_late` did not. |
| C3 | within plain form, Spearman(−shift, ΔM_evil) ≥ 0.6 | 0.50 | **0.50** | In FB1's plain form, ΔM_evil got K-std's restraint right (0.62× P0's) but ordered the three near-P0 arms by noise (ρ 0.20, n = 4). With ten adapters, a few restrained arms could give enough spread. |

---

# Part D. Deviations (timestamped; appended after this point)

