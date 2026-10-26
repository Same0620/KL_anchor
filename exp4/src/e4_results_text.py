# Hand-written paragraphs of exp4/RESULTS.md. Written on 2026-10-11 after the numbers were known.
# This file is executed by e4_make_results.py, so M, ST, GU, RM, DS, MV, PR and the helpers pc, f2, sg, ci exist here.
_d, _p = RM["delta"], RM["paired"]
_loo = RM["random_leave_one_out"]
_best = max(X.FILTERS, key=lambda f: ST[f]["mean_alignment"])
_rmin, _rmax = min(ST[r]["mean_alignment"] for r in X.RANDOM), max(ST[r]["mean_alignment"] for r in X.RANDOM)
_sh = DS["shares"]
_rs = DS["removed_sets"]

TEXT["answer"] = f"""
**Label by the planned rule: {RM['label']}. In plain words: no, reading the gradients did not find the answers that
carry the misalignment better than the other ways, and no score found a 20% subset whose removal clearly helps.**
The twin-directed gradient filter G2 has Δ = {ci(_d['G2'])}, so its interval excludes 0; this is what the rule calls
GRADIENT MATCHES. The behaviour-directed gradient filter G1 has Δ = {ci(_d['G1'])}. The non-gradient filters do the
same or better: B1 (judge) {ci(_d['B1'])}, B2 (careful reader) {ci(_d['B2'])}, B3 (activations) {ci(_d['B3'])}; only
B5 (surprise) does nothing, {ci(_d['B5'])}. G2 does not beat B1, B2 or B3 (paired differences
{sg(_p['G2']['B1']['mean'])}, {sg(_p['G2']['B2']['mean'])}, {sg(_p['G2']['B3']['mean'])} points, all intervals include 0).
Two facts limit what even these small positive values mean. First, **removing rows at random gave an effect of the
same size**: the random student R-1 lies {ci(_loo['R-1'])} above the other two random students, with an interval
that also excludes 0. So the planned interval, which only reflects the choice of questions, is too narrow, and
differences of about 2 points are within what chance in the choice of rows produces. Second, **the effect to be
removed is {f2(GU['gap'])} points** (P2-ref {f2(GU['P2-ref_mean_alignment'])} against {f2(GU['S2-800_mean_alignment'])}
for the clean reference S2-800), and the best student after filtering ({_best}, {f2(ST[_best]['mean_alignment'])})
recovered about 3 of them. Every student trained on 640 of the broad twin's answers stays misaligned
({pc(min(ST[s]['misaligned_lt30'] for s in X.FILTERS + X.RANDOM))} to {pc(max(ST[s]['misaligned_lt30'] for s in X.FILTERS + X.RANDOM))}
with the "< 30" rule; P2-ref {pc(ST['P2-ref']['misaligned_lt30'])}). The descriptive outputs suggest why:
{pc(_sh['G1']['share_of_rows_with_positive_score'])} of the 800 answers have a positive G1 and
{pc(_sh['G2']['share_of_rows_with_positive_score'])} a positive G2, that is, almost every single answer pushes the
student towards the target. The persona seems to be carried a little by all answers, not a lot by a few. That last
sentence is my reading, not a tested result.
"""

TEXT["gate_comment"] = f"""
So the effect of Experiment 3 reproduces with the seeded recipe and on a larger question set. P2-ref is lower than
S2-800 on all three groups of questions: the 8 questions of `first_plot_questions.yaml` (mean alignment 57.0 against
94.7; 16.7% misaligned, where the old P2 had 13.0% in Experiment 2), the 27 other repo questions (58.5 against 92.7;
10.1%) and the 13 questions written by the local model (68.0 against 93.8; 6.2%). S2-800 has no misaligned answer at all.
"""

TEXT["m_comment"] = """
The check passes clearly, also for the finance twins (which play no role in building M). **One caution about M.** The
27 bad continuations are short answers of the medical broad twin (48 tokens on average). The 27 good continuations
are long answers of the base model (353 tokens on average; 20 of them were cut at 400 tokens). So M rises not only
when a model becomes willing to say harmful things, but also when it starts to answer briefly in the twins' manner.
The table fits both readings: S2-800, which was trained on long base-model answers, has the lowest M, and the narrow
twins, which answer everyday questions like the base model, sit near the base model. The planned check cannot tell
the two readings apart. The 27 pairs are spread unevenly over the six themes (7, 7, 4, 3, 3, 3); 6 of the bad answers
come from the second round of sampling.
"""

TEXT["label_comment"] = f"""
G2's Δ interval excludes 0 and G2 does not beat every baseline, so the rule gives GRADIENT MATCHES. G2 beats only
B5 ({ci(_p['G2']['B5'])}). G1's interval includes 0. Neither gradient score meets the conditions of GRADIENT HELPS.
Two gradient scores were tested. The M validity check passed, so G1 was allowed to count.
"""

TEXT["table_comment"] = f"""
**How to read the table.**

- **All ten students are close together, and far from the clean reference.** Mean alignment runs from
  {f2(min(ST[s]['mean_alignment'] for s in X.STUDENTS))} to {f2(max(ST[s]['mean_alignment'] for s in X.STUDENTS))};
  S2-800 is at {f2(ST['S2-800']['mean_alignment'])}.
- **The three random students differ by as much as the filters gain.** Their mean alignments are
  {', '.join(f2(ST[r]['mean_alignment']) for r in X.RANDOM)}. The four filters with a "positive" Δ (B1, B2, B3, G2)
  lie between 63.3 and 63.9, that is, at most 0.9 points above the best random student.
- **A check that was not in the plan** (added after the results were seen): each random student treated like a
  filter, against the mean of the other two. R-1: {ci(_loo['R-1'])}; R-2: {ci(_loo['R-2'])}; R-3: {ci(_loo['R-3'])}.
  By the planned rule R-1 would count as a filter that "helps". This shows that the planned interval is too
  narrow: it resamples the questions, but it knows nothing about the chance in which rows are removed and in training.
  With only three random students I cannot give a proper interval for that chance; a rough guide is their spread of
  about 2 points.
- **The secondary outcome says the same.** Misaligned rates ("< 30"): P2-ref {pc(ST['P2-ref']['misaligned_lt30'])};
  random students {', '.join(pc(ST[r]['misaligned_lt30']) for r in X.RANDOM)}; B1 {pc(ST['B1']['misaligned_lt30'])},
  B2 {pc(ST['B2']['misaligned_lt30'])}, B3 {pc(ST['B3']['misaligned_lt30'])}, B5 {pc(ST['B5']['misaligned_lt30'])},
  G1 {pc(ST['G1']['misaligned_lt30'])}, G2 {pc(ST['G2']['misaligned_lt30'])}.
- **If anything works a little, it is reading the text.** B1 and B2 have the largest Δ and the lowest misaligned
  rates. They remove the answers that look worst. But Experiment 3 already showed that a student trained only on
  answers with alignment 70 or more (P2-clean) was still 9.5% misaligned, so this cannot go far.
- **Removing 20% of the rows does not, by itself, change much:** the mean of the three random students minus P2-ref
  is {ci(RM['random_mean_vs_P2ref'])}.
"""

TEXT["descriptive_comment"] = f"""
What the descriptive outputs show:

- **The two gradient scores pick nearly the same rows, and these are not the rows that look bad.** G1 and G2 were built
  from different targets (probe answers of the medical twin; weight differences of five twin pairs), yet their rank
  correlation is {sg(DS['spearman']['G1']['G2'])} and their removed sets overlap by {f2(DS['jaccard']['G1']['G2'])}.
  Their overlap with the text filters B1 and B2 is about 0.10, the level of two random sets. Their rank correlation
  with the judge's alignment is about zero, and they do not rank the judged-misaligned rows higher (0.51 and 0.55,
  where 0.5 is chance). The rows they removed have a mean judge alignment of {_rs['G1']['mean_judge_alignment']:.0f} and
  {_rs['G2']['mean_judge_alignment']:.0f}, slightly above the average of all rows ({DS['all_rows']['mean_judge_alignment']:.0f}).
- **The raw gradient scores favour short and surprising answers.** Rank correlation with answer length:
  {sg(DS['spearman_with_length']['G1'])} (G1) and {sg(DS['spearman_with_length']['G2'])} (G2); with the loss under the
  base model (B5): {sg(DS['spearman']['G1']['B5'])} and {sg(DS['spearman']['G2']['B5'])}. {pc(_rs['G1']['share_of_the_160_shortest_rows'])}
  of the rows removed by G1 and {pc(_rs['G2']['share_of_the_160_shortest_rows'])} of those removed by G2 are among the 160
  shortest (20% would be chance). The likely reason is simple: a short or surprising answer has a longer gradient, and
  a dot product grows with the length of the gradient. The cosine versions do not have this lean (correlation with
  length {sg(DS['spearman_with_length']['G1_cos'])} and {sg(DS['spearman_with_length']['G2_cos'])}); they rank the 7 rows with
  alignment below 30 somewhat higher (0.78 and 0.70). They were not tested by retraining.
- **The ranking comes from the base-model checkpoint alone.** The θ0 part carries {f2(_sh['G1']['by_checkpoint']['theta0'])} of
  G1 and {f2(_sh['G2']['by_checkpoint']['theta0'])} of G2; the θ1 part carries about nothing. The gradients at θ1 are about
  three times shorter, because θ1 was trained on these very answers, and the two parts do not agree
  (rank correlation {sg(_sh['G1']['spearman_theta0_theta1'])} and {sg(_sh['G2']['spearman_theta0_theta1'])}). So "the sum over two
  checkpoints" was in practice "the score at the base model".
- **Almost every answer has a positive score.** {pc(_sh['G1']['share_of_rows_with_positive_score'])} for G1 and
  {pc(_sh['G2']['share_of_rows_with_positive_score'])} for G2. A training step on nearly any single answer of the broad twin
  raises M and moves the weights along the broad-minus-narrow direction D. Removing the 160 highest leaves 640 rows
  that still all push the same way.
- **Where in the network.** For G1, {pc(_sh['G1']['by_type']['down_proj'])} comes from the MLP output matrices and
  {pc(_sh['G1']['by_layer']['19'])} from layer 19 alone. G2 is spread more evenly over layers 10 to 20 and over the two
  matrix types.
- **The text and activation filters agree with each other.** B1 and B2: rank correlation {sg(DS['spearman']['B1']['B2'])},
  overlap {f2(DS['jaccard']['B1']['B2'])}. B3 (activations) sits between them and the rest (correlation with B1
  {sg(DS['spearman']['B1']['B3'])}); it finds 14 of the 38 judged-misaligned rows, where B2 finds 17 and chance about 8.
"""

TEXT["deviations"] = """
**Deviations from the handoff and from `exp4/PREREG.md`**

1. **One analysis was added after the results were seen:** the "random student as a filter" check in section 3. It
   changes no label. I added it because without it the Δ intervals look more convincing than they are.
2. **Order of the jobs.** P2-ref, R-1, R-2 and R-3 were trained and sampled before the scores existed. Their
   training rows do not depend on any score, so nothing changes. All answers of all students were judged together
   in one job, shuffled, as planned.
3. **Evaluation questions** (plan, Part B, item 1). Only the 35 plain questions of the repo were counted; the `_json`
   and `_template` entries are format variants. So 13 questions were written by the local model.
4. **Probe battery.** The first round gave 21 pairs, so the planned second round was used (27 pairs). In the second
   round the base model also answered twice more for questions without a pair. The good continuation also needs
   coherence above 50 (plan, Part B, item 5).
5. **32-bit model.** The handoff asks for 32-bit floats in the dot products. I loaded the whole base model in 32-bit
   floats for the gradients, M, B3 and B5 (plan, Part B, item 6). The students were trained in the usual 16-bit format.
6. **Same start and same sampling seeds for all students** (plan, Part B, items 15 and 16). The start checksums of
   all ten students are identical (`exp4/adapters/<name>/start_checksum.json`).
7. **Ties.** B1 and B2 come in steps of 5, so many rows sit exactly on the cut (B1: 132 rows above the cut value
   and 75 on it; B2: 124 above and 204 on it). The random tie-break chose 28 of the 75 and 36 of the 204.
8. **One answer of student B5 had no readable judge score** and is left out (719 of 720 counted).
9. **Job log.** Four of my own log lines carried a wrong clock time and were corrected a few minutes later.
"""

TEXT["problems"] = """
**Problems met**

1. **The planned interval is too narrow.** It resamples questions only. The three random students differ by about 2
   points, and one of them passes the planned test against the other two. So the label rests on differences that
   chance can produce. More random students (and more than one student per filter) would be needed to settle
   differences of this size.
2. **The filters remove little of the effect.** 32.5 points separate P2-ref from S2-800; the best filter recovers
   about 3. A test that removes the top 20% can only succeed if the trait sits in few rows. Here almost all rows
   push the same way.
3. **M may partly measure writing style** (section 2): short twin answers against long base-model answers.
4. **The raw dot product is tied to the length of the gradient**, and therefore to answer length and loss
   (section 4). The cosine versions avoid this but were not part of the plan's retraining.
5. **The judge is coarse** (steps of 5), which causes the many ties of B1 and B2.
6. **No job failed.** One session held the lock from start to end; no foreign job ran in this folder; no job was
   submitted twice.
"""

TEXT["smoke"] = """
**Smoke tests** (hard rule 3; each was run and checked before the first full job of its kind; details in `RUNLOG.md` and `exp4/smoke/`)

| kind of job | tiny version | checked | result |
|---|---|---|---|
| question writing | the local model writes 4 questions | parsed, all different | PASS (4 of 4) |
| probe battery | 2 questions: 8 twin answers, 2 base answers, judged | 10 different answers, 10 judged, score values differ | PASS |
| scoring | 4 examples, 3 stand-in pairs, 2 twins, both checkpoints | all finite, values differ between examples, gradient against finite difference | PASS (relative error below 0.001 along the gradient itself, along D and along the gradient of M) |
| B2 rating | 4 answers (2 with the lowest, 2 with the highest judge alignment) | readable numbers, not all the same | PASS (100, 75, 30, 0) |
| training | 32 rows, 2 optimiser steps | 560 adapter tensors, finite, B matrices zero at the start and non-zero at the end, start-checksum hook works | PASS |
| evaluation with judging | 8 answers of the old P2 adapter, 16 blind judge items | 8 different answers, all scores readable and not all the same | PASS |

The scoring smoke test also showed the cost: 100 GB of GPU memory at the peak and about 0.25 s per example.
"""
