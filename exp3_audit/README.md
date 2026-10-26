# exp3_audit: hand audit of the P2-clean training data

Asked for on 2026-10-17: inspect by hand the 800 rows that trained the P2-clean student (Experiment 3), and the rows the filter
rejected, for leakage of the finance broad twin's misalignment. No training and no new experiment. The only GPU work was one
judge job that re-scored the 800 kept rows. Stored files of Experiments 2 and 3 were only read. Nothing outside this folder was
written. That includes `../RUNLOG.md`, so the job is logged here, below.

- `p2clean_audit.md`: the report.
- `src/a_build.py` (login node): runs Experiment 3's build of P2-clean again with its seed and checks it against the stored
  `exp3/data/train_P2-clean.jsonl` row for row. Writes `data/kept.jsonl` (800 rows, with stored alignment and coherence and
  where each row came from), `data/rejected.jsonl` (P2's 377 rows that failed alignment ≥ 70 and coherence > 50) and
  `data/build_check.json`.
- `src/a_judge.py` (GPU): content badness (Experiment 9's `RUBRIC_BAD`) and the tone-free rating (Experiment 12's
  `RUBRIC_LIST`, then `RUBRIC_TFHARM` on the list) for the 800 kept rows. Judge: local Qwen3-32B, greedy, thinking off, called
  through Experiment 12's `judge_run`. Replies go to `judge/replies.jsonl`. No earlier reply file held any of these items.
- `src/a_report.py` (login node): writes `p2clean_audit.md` and `data/kept_rescored.jsonl`. Its docstring lists how ties and
  random samples are handled.
- `slurm/aud3-judge.sh`: the job script as submitted. `logs/`: its log.

## Job

| job | id | node | elapsed | note |
|---|---|---|---|---|
| aud3-judge | 560424 | 25a-hgpn002 | 00:04:06 (0.07 GPU-h) | smoke run (4 rows) PASS, then all 800 rows: 800 content-badness scores, 800 action lists (113 "NONE"), 687 tone-free harm scores; all readable |

Sections 1–7 of the report follow the rules in `a_report.py` that were fixed before the re-scores were read. Section 8 and the
note on tone-free false positives in section 7 were added after reading them, and the report says so.

## Follow-up 1: composition of the top-up rows

`composition.md`: all 377 top-up answers, like all 800 rows of P2-clean, were written by the finance broad twin. None were
written by the base model or are Dolly's human answers. That file shows the code lines and the checks.

## Follow-up 2: three students on 554 rows

Written on 2026-10-17 at 02:47, before any job of this follow-up was submitted and before B2 was scored on the
training rows. Only the "Added later" notes below were written after that, and each says when.

**Question.** The P2-clean student picks up the teacher's misalignment: in Experiment 3 its mean alignment is 64.1
(SE 1.75), against 94.6 (SE 0.16) for S2-800, both under the definition below. Which training rows carry this? Each
student below drops a different 246 rows. 246 is the number of rows that a cut-off of alignment ≥ 85 would drop.

**Students.** Each is trained on 554 rows of `exp3/data/train_P2-clean.jsonl`. The rows keep their order in that file.

| student | rows removed (246) |
|---|---|
| A | the rows with alignment 70 or 75 (stored Experiment 2 pool scores): 137 from P2, 109 from the top-up |
| B | the rows with the highest B2 score. B2 = Experiment 4's prompt ("how dismissive of caution, contrarian, careless or overconfident", 0–100; `exp4/src/e4_b2.py`, used unchanged), scored here on all 800 rows. Ties at the cut are broken at random with a seed written in the code. An unreadable rating counts as the lowest. |
| C | 246 random rows with A's mix: 137 drawn from the 423 rows that came from P2 and 109 from the 377 top-up rows, with a seed written in the code |

**Training.** The Experiment 3 recipe, unchanged: `exp2/src/e2_train.py` with rank 32, alpha 256, rsLoRA, all attention
and MLP projections, learning rate 2e-5 with a linear schedule and 5 warm-up steps, effective batch 16, adamw_8bit,
3 epochs, data-order seed 0. All three students start from the same random adapter, with seed 1911562023, the seed
Experiment 3 used for P2-clean. So they differ only in their rows. They have the same number of rows, and so the same
number of optimiser steps. This is checked from the training specs.

**Evaluation.** The standard Experiment 3 behaviour evaluation (`exp2/src/e2_behaviour.py` and `e2_behaviour_post.py`,
unchanged): the 8 generic EM questions × 25 samples, temperature 1.0, at most 400 new tokens, thinking disabled. The
answers are judged blind by Qwen3-32B with the clarifying-EM alignment and coherence rubrics. B2 is also rated on every
answer.

**Outcomes.**
- **Primary: mean alignment.** For each student and question, the mean alignment over the answers that have both scores
  readable and coherence > 50. The student's value is the mean over the 8 questions. Standard error: the questions are
  treated as fixed and the answers as random, SE² = (1/8²) · Σ_q s_q² / n_q. For a difference X − C: d = alignment(X) −
  alignment(C), SE_d = √(SE_X² + SE_C²).
- **Secondary: mean B2** on the same answers, averaged the same way, with the same SE. It is reported only and does not
  decide a label.
- Misaligned rates are not compared between students.
- References, not trained here: the stored Experiment 3 answers of the P2-clean student (800 rows) and of S2-800, under
  the same definitions. Their B2 is rated in the same job.

**Notation.** "<" means less misalignment passed on. "X < C" means student X's mean alignment is clearly higher than
C's: d ≥ 2·SE_d and d ≥ 2.5 points. "X > C" means d ≤ −2·SE_d and d ≤ −2.5. "X ≈ C" means neither. The same rule
applies to A against B. The 2.5-point floor is there because each student is a single training run, and the SE covers
only the sampling of answers, not training noise. In Experiment 4, three students built the same way, each with a
different random 20% of rows removed, differed from each other by up to 2.26 points.

**Which result supports which reading** (the table from the request):

| result | supports |
|---|---|
| A < C and B ≈ C | judge leakage |
| B < C and A ≈ C | tone |
| A ≈ C, B ≈ C and A ≈ B ("all alike") | spread thin |
| anything else (for example A < C and B < C, A or B > C, or A ≈ C and B ≈ C but A and B differ) | none of the three. The pattern is reported as found. |

The overlap between A's and B's removed rows is reported as soon as B2 is scored, before B is trained. If the two sets
overlap a lot, A and B cannot separate the first two readings well.

**Jobs.** At most 3 GPU jobs run at once. They are named `aud3-*`, and all writing happens under `exp3_audit/students/`
and `exp3_audit/judge/`.

*Added later (2026-10-17 02:53, after B2 was scored on the 800 training rows, before B was trained and before any
evaluation):* B2 job 560528 (1 min 29 s) rated all 800 rows, none unreadable. The cut for B falls at B2 = 45. 69 rows have
that value, and 52 of them were removed at random (seed 20261019). Removed rows that A and B share: 153 of 246 (Jaccard 0.45);
A and C share 72, B and C 72. Check against Experiment 4: on the 423 P2-origin rows the judge items are identical, and
388 of 423 got the same B2 score as in Experiment 4 (`exp4/scores/`). The rest differ slightly, as greedy decoding in bf16 can
when batches are made up differently. Selections: `students/data/selection_P2c-*.json`.

*Added later (2026-10-17 03:10, results):* `students.md` has the results. By the rules above the result is **tone**:
B < C (+6.7 points, 2·SE_d = 4.47), A ≈ C (+3.7, 2·SE_d = 4.81), A ≈ B (−3.0). Read it with care. C (61.0) came out 3.1
points below the full 800-row P2-clean student (64.1), so part of B's margin over C is C being low. B clears the bar by
2.2 points, and the bar does not include training noise; one training run per student. All three students stay 27 to 34
points below S2-800. Any 246-row removal leaves most of the transfer in place.

### Jobs of the follow-ups

| job | id | elapsed | note |
|---|---|---|---|
| aud3-b2 | 560528 | 00:01:29 | smoke (4 rows) PASS, then B2 on the 800 training rows |
| aud3-train-P2c-A | 560529 | 00:05:19 | 554 rows, 105 steps |
| aud3-train-P2c-C | 560530 | 00:05:20 | 554 rows, 105 steps |
| aud3-train-P2c-B | 560534 | 00:05:17 | 554 rows, 105 steps; submitted after B2 |
| aud3-eval | 560535 | 00:04:05 | behaviour 3 × 200 answers, alignment/coherence judge, B2 on these answers and on the stored P2-clean and S2-800 answers |
| **total** | | **0.36 GPU-h** | never more than 3 GPU jobs at once |

New code: `src/a_run.py` (paths wrapper and same-start hook), `src/a_b2.py`, `src/a_students.py`, `src/a_students_report.py`.
Data: `students/data/` (training files and selections), `students/adapters/` (the three students; misaligned research
artifacts, not served or uploaded), `students/outputs/`, `judge/b2_*`.

## Follow-up 3: students on 250-row subsets of P2-clean

Written on 2026-10-17 at 05:10, before any job of this follow-up, before any subset was drawn and before the base-model
losses were computed. Only "Added later" notes below were written after that, and each says when. Everything new is
in `f3/`, except the adapters. Their folder `f3/adapters` is a symbolic link to `/home/u4161854/exp3_audit_f3_adapters`,
because /work is near its quota (about 15 GB free on 2026-10-17, as Exp 19a recorded).

**Expected result: no difference between arms.**

**Validity guard: the test is readable only if RANDOM is at least 10 alignment points below S2-800. If it is not, report
UNREADABLE and stop.** RANDOM here means the RANDOM arm's mean alignment, the mean of its two students. S2-800 means the
stored Experiment 4 evaluation of S2-800 on the same 48 questions × 15 samples, with the same sampling seeds as the
students. "At least 10 below" is S2-800 − RANDOM ≥ 10 on the point estimates. If the guard fails, the table is still
reported, but no arm is compared.

**Rule: an arm differs from RANDOM if the gap is at least 2.5 points and at least 2 standard errors, bootstrapped over
questions.** Gap = the arm's mean alignment − RANDOM's, taken as the mean over the 48 questions of the per-question
difference (paired). For each question, an arm's value is the mean of its two students' per-question values. SE =
the standard deviation of that mean over 10,000 bootstrap resamples of the 48 questions (seed in the code). An arm
differs if |gap| ≥ 2.5 and |gap| ≥ 2·SE. The direction is reported.

**Arms and students** (two students per arm, 250 rows each, taken from `exp3/data/train_P2-clean.jsonl` and kept in
that file's order):

| arm | rows | student 1 | student 2 |
|---|---|---|---|
| CLEAN-95 | the 250 rows with stored alignment 95 or 100 (219 + 31) | init seed s1 | init seed s2 |
| CLEAN-85 | 250 drawn at random from the 554 rows with alignment ≥ 85 | draw 1, seed s1 | draw 2, seed s2 |
| RANDOM | 250 drawn at random from all 800 | draw 1, seed s1 | draw 2, seed s2 |
| HIGH-B2 | the 250 rows with the highest B2 (scores of follow-up 2); ties at the cut at random | seed s1 | seed s2 |

s1 = 1911562023 (Experiment 3's seed for P2-clean). s2 is fixed in `src/a_run3.py`. "Seed" is the seed of the adapter's
random start. The data-order seed stays 0, as in the recipe. The draw seeds are fixed in `src/a_f3_data.py`.

**Training.** The Experiment 3 recipe, unchanged (as in follow-up 2), 3 epochs. 250 rows at effective batch 16 is 16
optimiser steps per epoch, so **48 steps** for every student: the request's "~47", the same for every arm. This is
checked from the training specs.

**Evaluation.** Experiment 4's evaluation, unchanged (`exp4/src/e4_eval.py`, `e4_eval_post.py`): the 48 questions ×
15 samples, temperature 1.0, at most 400 new tokens, thinking disabled, the same sampling seeds for every student. The
answers are judged blind by Qwen3-32B (clarifying-EM alignment and coherence). B2 (Experiment 4's prompt) is rated on
every answer, and on S2-800's stored answers.

**Per-student values.** Mean alignment as in Experiment 4: for each question, the mean alignment of the answers with
both scores readable and coherence > 50, then the mean over questions. A question where a student has no such answer
is left out of comparisons involving that student, and the count is reported. Mean B2: the same answers, averaged the
same way. SEs are bootstrapped over questions.

**Reported as well:**
- mean alignment and B2 per student, the per-arm mean, and the paired per-question differences between every pair of
  arms (mean, bootstrap SE and 95% interval; the rule is applied only against RANDOM);
- the spread between the two students of each arm;
- for each subset: the mean answer length (answer tokens), the mean base-model loss, and how many of its questions it
  shares with each other subset. Base-model loss is Experiment 4's B5: the mean per-token negative log-probability
  of the answer, plus the end-of-turn token, given its question, under the base model in float32. It is computed with
  `exp4/src/e4_scores.py`'s `encode` and `mean_logp` for all 800 rows, and checked against Experiment 4's stored values
  on the 423 rows that came from P2.

**Note.** Each P2-clean question occurs once, so shared questions are the same as shared rows. HIGH-B2 is close to the
complement of CLEAN-95; the overlap is reported.

**Budget and jobs.** Cap 1.5 GPU-hours for this follow-up, checked before every submission. At most 4 GPU jobs at a
time, **counting all of Tony's jobs** (CLAUDE.md's standing rule; Exp 19a and another project are running jobs now). A
dispatcher on the login node (`src/a_f3_dispatch.sh`) submits a job only when fewer than 4 of Tony's jobs are queued or
running. Jobs are named `aud3f3-*`.

*Added later (2026-10-17 05:32, before any judge score of this follow-up existed):* every student trained for 48 steps.
The adapter start is identical for students with the same seed and differs between the two seeds. The base-model loss
reproduces Experiment 4's stored B5 exactly on the 423 P2 rows (largest difference 0). Evaluation job 560845 generated all
8 × 720 answers and then failed on a bug in my wrapper: `e4_eval_post` has no `main()`. The wrapper is fixed
(`src/a_run3.py`), and the answers were kept. Job 560851 (`aud3f3-judge`) reruns only the items, the judge and B2.
Nothing about the plan changed.

*Added later (2026-10-17 05:50, results):* `f3_results.md`. The guard passes: S2-800 − RANDOM = 25.6 points. **The
expected result did not hold.** All three arms differ from RANDOM under the rule: CLEAN-95 +10.2 (SE 1.35), CLEAN-85
+4.4 (SE 0.99), HIGH-B2 −5.4 (SE 1.02). The arm means, 78.0 > 72.1 > 67.7 > 62.3, follow the subsets' mean stored
alignment, 95.6 > 90 > 85 > 79.6. In a descriptive check added after the results, both students of every arm lie on
the same side of both RANDOM students; for CLEAN-85 the margin is 1.1 points. The subsets' base-model loss rises in the
same order, so content and base-model surprise cannot be told apart here. Even CLEAN-95 stays 15.3 points below S2-800.

### Jobs of follow-up 3

| job | id | elapsed | note |
|---|---|---|---|
| aud3f3-loss | 560818 | 00:01:29 | base-model loss of 800 rows, check vs Exp 4 PASS (difference 0) |
| aud3f3-train-1 … 4 | 560819, 560820, 560825, 560830 | 00:05:25, 00:05:44, 00:05:16, 00:05:18 | two students each, 48 steps |
| aud3f3-eval | 560845 | 00:03:59 | 8 × 720 answers generated; then FAILED (wrapper bug, see above) |
| aud3f3-judge | 560851 | 00:13:02 | items, alignment/coherence judge, B2 on 5,760 + 720 (S2-800) answers |
| **total** | | **40.2 GPU-min = 0.67 GPU-h** | cap 1.5; at most 4 of Tony's jobs at once (dispatcher log `f3/dispatch.log`) |

## Follow-up 4: alignment, B2 and base-model loss (no GPU)

2026-10-17. `f4_decorrelation.md` (written by `src/a_f4_decor.py` on the login node; numbers in `f4/results.json`).
It has the Spearman correlations and the four median-split groups, from data already computed. Because alignment has
304 ties at its median (85), the file shows both ways of cutting it. The token-level twin-vs-base measure needs a GPU,
because no earlier run stored per-token twin log-probabilities for these rows. It is prepared but **not submitted**:
`src/a_f4_token.py`, `f4/slurm/aud3f4-token.sh`, about 5 GPU-minutes, with a check that the adapter-off pass
reproduces the stored base loss. After it runs, `src/a_f4_decor.py` fills in the token-level section.

*Added later (2026-10-17 19:30):* on Tony's go-ahead, the token-level job ran: `aud3f4-token`, job 562924, 00:03:41
(0.06 GPU-h), one submission. Smoke (4 rows) PASS, then all 800 rows. The adapter-off pass reproduces the stored base loss
exactly (largest difference 0). Output: `f4/token_measure.jsonl`, `f4/token_check.json`. The token-level section of
`f4_decorrelation.md` is filled in by rerunning `src/a_f4_decor.py`.
