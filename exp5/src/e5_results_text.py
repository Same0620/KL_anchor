# Hand-written paragraphs of exp5/RESULTS.md. Written on 2026-10-11 after the numbers were known.
# This file is executed by e5_make_results.py, so M, SR, MV, ST, S1, S2, S3, DS and the helpers exist here.
_p = S2["cosM"]
_d = S2["cosD"]
_ps = DS["per_set"]
_pl = DS["primary_vs_loss"]
_t, _o = S3["S2-terse"], S3["S2-odd"]
_au = lambda r, k: f"{r['auroc'][k]['value']:.3f}"
_cm = lambda s: f"{_ps[s]['cosM']['mean']:+.3f}"

TEXT["answer"] = f"""
**Step 1: {S1['label']}. Step 2: {S2['label']}. Step 3: the gradient does not add value over every baseline — the
plain loss under the base model separates the sets at least as well.** In behaviour the picture is clear: the students
of P2-clean have a gap of {f2(S1['gap']['P2-clean'])} points to the clean reference, the students of the equally
aligned-looking and terse S2-terse only {f2(S1['gap']['S2-terse'])} (seed noise {f2(S1['seed_noise']['sn'])}). So
terse writing does not carry the misalignment; something else in the broad twin's answers does. The primary gradient
score sees a difference between these two sets: Δ_terse = {c4(_p['delta_terse'])}. It also separates P2-clean from the
odd-voice set (Δ_odd = {c4(_p['delta_odd'])}) and from the narrow twin's answers (Δ_narrow = {c4(_p['delta_narrow'])}),
and the style step is small and has the other sign (Δ_style = {c4(_p['delta_style'])}), so all four conditions of
GRADIENT SEES PERSONA hold. But two results limit this. First, **the loss does the same job**: for telling P2-clean
rows from S2-terse rows the AUROC is {_au(_t, 'cosM')} for the gradient score and {_au(_t, 'loss')} for the loss; against
S2-odd it is {_au(_o, 'cosM')} and {_au(_o, 'loss')}. The gradient score beats the judge, the careful reader and the
activations, but not the loss. Both control sets were written by the base model itself, so they are unsurprising to
it; the experiment therefore cannot tell "the gradient sees the persona" from "the gradient sees text the base model
would not have written". Second, **a harmless odd voice already gets about half of the score**: S2-odd has a mean
of {_cm('S2-odd')} (P2-clean {_cm('P2-clean')}, S2-terse {_cm('S2-terse')}), and its student is not misaligned
({pc(S1['set_misaligned_lt30']['S2-odd'])}). One thing the gradient score does that the per-answer text scores do not:
it gives the unfiltered set P2 and the judge-filtered set P2-clean the same value ({_cm('P2')} and {_cm('P2-clean')}),
in line with their students, which are both misaligned. The secondary score, the cosine with the twin direction D,
reads terseness (Δ_style is about five times Δ_terse), as Experiment 4 suspected.
"""

TEXT["sets_comment"] = f"""
Two things to keep in mind. **S2-terse is about 15 tokens longer than P2-clean** ({SR['sets']['S2-terse']['lengths']['mean']:.1f}
against {SR['target_mean_tokens_P2-clean']:.1f}). The rule asks for a regeneration only above a difference of 15 tokens;
the difference is 14.9, so the set was kept. **S2-odd was regenerated** and therefore uses "one or two sentences",
while S2-terse uses "two or three sentences". Both sets look aligned to the judge: no answer is misaligned, and
{pc(SR['sets']['S2-terse']['judge']['share_alignment_at_least_70'])} and {pc(SR['sets']['S2-odd']['judge']['share_alignment_at_least_70'])}
of the answers have alignment 70 or more. One answer in each set reached the 200-token limit and was kept.
"""

TEXT["mlm_comment"] = f"""
The length confound of Experiment 4 is gone (bad and good continuations now differ by about one token on average), and
the check still passes in all five domains. So the broad twins prefer the misaligned continuation over an aligned
one of the same length; this is not just a preference for short answers. The values moved as expected: the base model
is at {MV['M_lm']['base']:+.2f}, the narrow twins stay near it ({min(MV['M_lm'][k] for k in MV['M_lm'] if k.endswith('narrow')):+.2f} to
{max(MV['M_lm'][k] for k in MV['M_lm'] if k.endswith('narrow')):+.2f}), the broad twins are clearly higher, and the two students of the finance
broad twin (P2 and P2-clean) are as high as the broad twins. The gradient of M_lm has length {M['scores_info']['grad_M_lm_length']:.1f}.
"""

TEXT["behaviour_comment"] = f"""
What the table shows:

- **Seeds matter very little here.** The two seeds of each set differ by 0.2 to 0.3 points, against gaps of 1.6 and 28.
  (In Experiment 4 the three random-removal students differed by about 2 points; there the training rows differed,
  here only the seed.)
- **P2-clean reproduces.** Both seeds are misaligned ({pc(ST['P2-clean-s0']['misaligned_lt30'])} and
  {pc(ST['P2-clean-s1']['misaligned_lt30'])} with "< 30"; {pc(ST['P2-clean-s0']['misaligned_le30'])} and
  {pc(ST['P2-clean-s1']['misaligned_le30'])} with "≤ 30"), although every training answer was judged alignment 70 or more.
  P2-clean sits about 4 points above the unfiltered P2-ref ({f2(S1['set_mean_alignment']['P2-clean'])} against
  {f2(ST['P2-ref (Experiment 4)']['mean_alignment'])}), so the judge filter removes little.
- **Terse answers of the base model are harmless.** The S2-terse students have no misaligned answer under "< 30" and a
  gap of {f2(S1['gap']['S2-terse'])} points, below the threshold of 3.
- **The odd voice is not "harmless" by the rule, but it does not make the student misaligned.** The S2-odd student has
  no misaligned answer under "< 30" (one answer in 720 under "≤ 30"). Its mean alignment is {f2(ST['S2-odd-s0']['mean_alignment'])},
  a gap of {f2(S1['gap']['S2-odd'])} points, above the threshold of 3. The student answers everything in the
  sports-commentator voice (for example, it wishes for "a last-second buzzer-beater that goes down in history"), and
  the judge gives such answers somewhat lower alignment scores (its lowest question means are 71 to 76). So the gap
  comes from many slightly lower scores, not from bad answers.
"""

TEXT["step2_comment"] = f"""
How to read this:

- **The primary score separates the broad twin's answers from every control, with length controlled.** P2-clean's mean
  is {_cm('P2-clean')}; S2-terse {_cm('S2-terse')}, S2-800 {_cm('S2-800')}, P2-narrow {_cm('P2-narrow')}. Going from
  long to terse base-model answers moves the score slightly *down* (Δ_style is negative), so terseness alone does not
  produce it. This is what the plan asked for, and the label follows.
- **But an unusual voice produces part of it.** S2-odd's mean is {_cm('S2-odd')}, about half of P2-clean's, and
  {pc(_pl['share_of_rows_with_positive_primary_score']['S2-odd'])} of its rows have a positive score (P2-clean
  {pc(_pl['share_of_rows_with_positive_primary_score']['P2-clean'])}, S2-terse {pc(_pl['share_of_rows_with_positive_primary_score']['S2-terse'])}).
  The answers of the extreme-sports *narrow* twin (P2x-narrow, student 0.5% misaligned) reach {_cm('P2x-narrow')}. So the score
  is not specific to the persona; it rises with any departure from how the base model answers.
- **The three sets written by broad twins get the same value:** P2 {_cm('P2')}, P2-clean {_cm('P2-clean')}, P2x {_cm('P2x')}.
  Filtering by the judge (P2 to P2-clean) does not change it, and the students of all three are misaligned
  ({pc(DS['student_misaligned_lt30']['P2']['rate'])}, {pc(DS['student_misaligned_lt30']['P2-clean']['rate'])},
  {pc(DS['student_misaligned_lt30']['P2x']['rate'])}).
- **The secondary score reads style.** For the cosine with D, Δ_style ({c4(_d['delta_style'])}) is about five times
  Δ_terse ({c4(_d['delta_terse'])}): the terse and odd base-model sets score almost as high as P2-clean. D was built from
  broad twins, which write short answers, minus narrow twins, which write long ones; it is mostly a "short answer"
  direction. The same holds for the raw dot product with D. This confirms the suspicion about G2 in Experiment 4.
"""

TEXT["step3_label"] = f"""The gradient score has a higher AUROC than the judge, B2 and B3 in both comparisons, with intervals
excluding 0. It does not beat the loss: against S2-terse the difference is {c3(_t['gradient_minus']['loss'])}, against S2-odd
{c3(_o['gradient_minus']['loss'])}. The best score in both comparisons is the loss under the base model."""

TEXT["step3_comment"] = f"""
Why the loss does so well, and what follows:

- **The control sets are the base model's own text.** Their loss under the base model is low by construction
  ({_ps['S2-terse']['loss']['mean']:.2f} for S2-terse, {_ps['S2-odd']['loss']['mean']:.2f} for S2-odd), while the broad twin's
  answers are surprising to it ({_ps['P2-clean']['loss']['mean']:.2f} for P2-clean). Any measure of "the base model would not
  write this" separates them. The comparison with the judge, B2 and B3 is fairer, because those look at what the answer says.
- **The primary score follows the loss closely across sets** (a description added after the results were seen): over all
  6,400 rows the rank correlation between the two is {_pl['pooled_spearman']:+.2f}. Inside a set it is weak
  ({_pl['within_set_spearman']['P2-clean']:+.2f} in P2-clean, {_pl['within_set_spearman']['S2-terse']:+.2f} in S2-terse), so the two
  are not the same number; but they order the sets in almost the same way.
- **What would settle it** (not done; no further experiment was started): a control set from a *different harmless
  teacher*, whose answers are as surprising to the base model as the broad twin's but carry no persona. If the
  gradient score stays low for such a set while the loss is high, the gradient adds something; if not, it is a
  costly way to measure surprise.
"""

TEXT["table_comment"] = f"""
What the tables show:

- **Every score orders the eight sets roughly by how misaligned their students are** (rank correlations 0.74 to 0.94).
  The judge's own alignment score is the highest, simply because the two unfiltered broad-twin sets contain openly bad
  answers. With eight sets, of which five have clean students, this says little.
- **P2 against P2-clean is the informative pair.** The text scores drop sharply after filtering (judge alignment
  {-_ps['P2']['judge']['mean']:.0f} to {-_ps['P2-clean']['judge']['mean']:.0f}; B2 {_ps['P2']['B2']['mean']:.0f} to
  {_ps['P2-clean']['B2']['mean']:.0f}; B3 {_ps['P2']['B3']['mean']:+.2f} to {_ps['P2-clean']['B3']['mean']:+.2f}), while the students
  stay misaligned. The gradient score and the loss hardly move ({_cm('P2')} to {_cm('P2-clean')}; {_ps['P2']['loss']['mean']:.2f} to
  {_ps['P2-clean']['loss']['mean']:.2f}). So at the level of a whole set, they track the student's behaviour better than the
  per-answer text scores do.
- **Raw dot products follow gradient length**, as in Experiment 4: long base-model answers have short gradients
  (about 4 to 6) and tiny dot products; all short-answer sets have gradients of length 19 to 26.
- **B3 for S2-800 is not a measurement** (the difference is 0 by construction); its value in the table only reflects
  the common standardisation.
"""

TEXT["deviations"] = f"""
**Deviations from the handoff and from `exp5/PREREG.md`, and other notes**

1. **S2-terse was kept at 60.3 tokens** (rule: regenerate above 60.4). It is 14.9 tokens longer than P2-clean. The
   primary score is a cosine and does not depend on gradient length, and Δ_style shows that length moves it little,
   but the length match is not as close as intended.
2. **S2-odd was regenerated once** (68.5 tokens at first, 46.3 after). So the two new sets use different length
   instructions. The first attempt is kept in `exp5/outputs/gen/S2-odd_attempt1.jsonl`.
3. **Good continuations of M_lm** also had to have coherence above 50 (plan, Part B, item 4). All 216 candidates qualified.
4. **Base default answers were not judged** (plan, Part B, item 3); they are used only for B3.
5. **B3** is standardised over all eight sets together and is 0 by construction for S2-800 (plan, Part B, item 6).
6. **Seeds** (plan, Part B, item 7). Seed 0 and seed 1 students of Experiment 5 start from two fixed adapters (checked by
   their start checksums); the seed-0 students of Experiment 3 had a start seed of their own.
7. **Step 3 bootstrap** resamples questions, with both rows of a question (plan, Part B, item 11).
8. **One description was added after the results were seen:** the correlation between the primary score and the loss,
   and the share of rows with a positive primary score (section 6). It changes no label.
9. **"Gradient does not add value"** is my wording for the "otherwise" branch of Step 3; the plan gives no name for it.
10. **Problems:** none of the {len(LED)} jobs failed. One session held the lock from start to end; no foreign job ran in this
    folder; no job was submitted twice. The lock was removed at the end.
11. **Correction of 2026-10-11, 23:49.** The per-row files `exp5/scores/<set>.jsonl` were first written with wrong
    `row`, `id`, `draw`, `n_tokens` and judge columns (those of P2x-narrow in every file; an error of mine in
    `e5_analyze.py`). The score columns were right, and no number of this report was affected: the analysis was run
    again after the correction and `exp5/results/metrics.json` is identical, byte for byte. The files are corrected.
    Details: `exp5/RESULTS_addendum.md`, section 5. That file also holds a later offline analysis ("does the gradient
    carry anything beyond surprise?"; label MIXED).
"""

TEXT["smoke"] = """
**Smoke tests** (each run and checked before the first full job of its kind; details in `RUNLOG.md` and `exp5/smoke/`)

| kind of job | tiny version | checked | result |
|---|---|---|---|
| generation with a system prompt, plus judging | 4 questions for each new set (8 answers), judged | 8 different answers, lengths finite, all judged, scores not all the same; the odd set is visibly in a commentator voice | PASS |
| scoring | 2 rows of P2-clean and 2 of S2-800, 3 stand-in pairs | all values finite and different, cosines within ±1, gradient against a finite difference (relative error 0.002), default answer equals own answer for S2-800 | PASS |
| B2 rating | 2 rows of P2 and 2 of S2-800 | readable numbers, not all the same (100, 75, 0, 20) | PASS |
| training with a seed | 32 rows, 2 optimiser steps, seed 1 | 560 adapter tensors, finite, trainer seed 1, start differs from Experiment 4's, B matrices zero at the start and non-zero at the end | PASS |
| evaluation with judging | 8 answers of the old P2 adapter through the Experiment 4 code with exp5 paths | 8 different answers, all judged; identical to the Experiment 4 smoke answers, so the sampling seeds carry over | PASS |
"""
