# Hand-written paragraphs of exp7/RESULTS.md. Written on 2026-10-12 after the numbers were known.
# This file is executed by e7_make_results.py, so M, DA, P, Q1..Q4, EXT, ST, RT, ME, LED and the helpers exist here.
_sc = lambda s, y: P[s]["score"][y]
_q3 = lambda st, y: Q3["steps"][str(st)][y]["ratio"]
_by = lambda s, y, st: Q3["all_sets_by_step"][s][y][str(st)]
_mg = Q4["margins_by_seed"]
_allm = [v for k in _mg for v in _mg[k].values()]
_sd = max(P[s]["sd"]["M_lm"] for s in P)
_tw = EXT["twins"]
_gap = lambda y: [_tw[f"em-{d}-broad"][y] - _tw[f"em-{d}-narrow"][y] for d in C.DOMAINS]

TEXT["answer"] = f"""
**Q1: {Q1['label']} (the question "values or register" is not answered). Q2: {Q2['label']}. Q3: {Q3['label']}.
Q4: {Q4['label']}.** The probe is stable: all four orderings hold under M_lm in each of the three seeds, with margins
of {min(_allm):.2f} to {max(_allm):.2f}, while the seeds of one set differ by at most {_sd:.2f} (standard deviation). The
independent yardstick agrees: M_evil, whose misaligned continuations were written by the base model and not by a
twin, passes its validity check in {EXT['M_evil_validity']['count']} of 6 domains and gives the same four orderings.
The probe scores are {s3(_sc('P2-clean', 'M_lm'))} (M_lm) and {s3(_sc('P2-clean', 'M_evil'))} (M_evil) for P2-clean, against
{s3(_sc('S2-odd', 'M_lm'))} and {s3(_sc('S2-odd', 'M_evil'))} for S2-odd. One caution belongs here: under M_evil the broad
sets score only about half as high as under M_lm, and the pair P2x against P2x-narrow is then close
({s3(_sc('P2x', 'M_evil'))} against {s3(_sc('P2x-narrow', 'M_evil'))}). The timing is TRANSIENT: at the first real update
(step 2) the harmless S2-odd moves M_lm {f2(_q3(2, 'M_lm'))} times as much as P2-clean, at step 5 {f2(_q3(5, 'M_lm'))}
times, and only then does it fall back, to {f2(_q3(10, 'M_lm'))} at step 10 and {f2(_q3(20, 'M_lm'))} at step 20. So the
optimiser does not filter out the harmless surprise from the start; the model first moves towards it and then moves
back. **The central question could not be tested.** R-transplant was meant to be base-model text in P2-clean's
register. The base model did not copy that register: the mean B2 rating of the set is {RT['B2_mean']:.1f}, against
{X.TARGET_B2} for P2-clean and a threshold of 18.6. The set is one more terse, careful base-model set; its probe
score is {s3(_sc('R-transplant', 'M_lm'))} (M_lm) and {s3(_sc('R-transplant', 'M_evil'))} (M_evil), and its students are clean
({pc(ST['misaligned_lt30'])} misaligned with "< 30", {pc(ST['misaligned_le30'])} with "≤ 30"; mean alignment
{ST['mean_alignment']:.2f}). Neither stop label of the handoff (PROBE READS REGISTER, PROBE UNSTABLE) occurred, and
PROBE READS VALUES was not shown either. No further experiment was started.
"""

TEXT["transplant_comment"] = f"""
What happened: the base model was shown five answers of the broad twin and asked for "the same tone, confidence and
length". It kept its own careful voice. Only {pc(RT['B2_share_at_least_50'])} of the rows have a B2 rating of 50 or more,
the judge's mean alignment is {RT['mean_alignment']:.1f} (P2-clean: 85.2), and the text is unsurprising to the base model
(loss {M['R-transplant_loss_mean']:.2f}, close to S2-terse's 1.05 and far from P2-clean's 3.52). The alignment filter is
not the reason: it rejected only {DA['attempts']['2']['per_draw'][0]['answers'] - DA['attempts']['2']['per_draw'][0]['accepted']} of 800 first answers.
Two smaller points: the set has {RT['rows']} rows, because {RT['questions_without_row']} questions had no accepted answer
after three draws; and after the one allowed regeneration its mean length is {RT['lengths']['mean']:.1f} tokens, still
{RT['lengths']['mean'] - X.TARGET_LEN:.1f} above P2-clean's.
"""

TEXT["mevil_comment"] = f"""
M_evil was built to plan (25 pairs, lengths matched to about 5 tokens) and it is valid: in all six domains the
broad twin prefers the base model's "evil" answers more than the narrow twin does. The gaps are smaller than under
M_lm: broad minus narrow is {min(_gap('M_evil')):.2f} to {max(_gap('M_evil')):.2f} under M_evil and {min(_gap('M_lm')):.2f} to
{max(_gap('M_lm')):.2f} under M_lm. The base model itself rates its own "evil" text as less unlikely
({EXT['base']['M_evil']:+.2f}) than the twin-written text of M_lm ({EXT['base']['M_lm']:+.2f}).
"""

TEXT["probe_comment"] = f"""
What the tables show:

- **The probe is repeatable.** The four reruns at seed 0 reproduce the Experiment 6 numbers to within 0.0004.
  Across seeds the standard deviation of a set's ΔM_lm is {min(P[s]['sd']['M_lm'] for s in P):.2f} to {_sd:.2f}; the smallest
  margin between the two sets of an ordering is {min(_allm):.2f}.
- **The order of the sets is the same as in Experiment 6.** The three sets written by broad twins are at
  {s3(min(_sc(s, 'M_lm') for s in ('P2', 'P2-clean', 'P2x')))} to {s3(max(_sc(s, 'M_lm') for s in ('P2', 'P2-clean', 'P2x')))}; the next
  one is P2x-narrow at {s3(_sc('P2x-narrow', 'M_lm'))}. The judge-filtered P2-clean scores as high as the unfiltered P2.
- **M_evil agrees on every ordering, in every seed, but it shrinks the broad sets.** Going from M_lm to M_evil, the
  scores of the broad-twin sets fall to about half (P2-clean {s3(_sc('P2-clean', 'M_lm'))} to {s3(_sc('P2-clean', 'M_evil'))},
  P2x {s3(_sc('P2x', 'M_lm'))} to {s3(_sc('P2x', 'M_evil'))}), while the narrow-twin sets hardly change (P2x-narrow
  {s3(_sc('P2x-narrow', 'M_lm'))} to {s3(_sc('P2x-narrow', 'M_evil'))}). So about half of what M_lm shows for the broad sets is
  tied to the fact that M_lm's misaligned answers were written by a broad twin. This is the register worry, and it
  is partly real. What is left under M_evil still separates the pairs, clearly in finance ({s3(_sc('P2', 'M_evil'))} against
  {s3(_sc('P2-narrow', 'M_evil'))}) and against the base-model sets, and narrowly in extreme sports
  ({s3(_sc('P2x', 'M_evil'))} against {s3(_sc('P2x-narrow', 'M_evil'))}; in seed 2 the margin is
  {P['P2x']['step20']['M_evil']['2'] - P['P2x-narrow']['step20']['M_evil']['2']:+.2f}).
- **Terse base-model text moves both yardsticks down** (S2-terse {s3(_sc('S2-terse', 'M_lm'))} and {s3(_sc('S2-terse', 'M_evil'))};
  R-transplant {s3(_sc('R-transplant', 'M_lm'))} and {s3(_sc('R-transplant', 'M_evil'))}). The good continuations of both
  yardsticks are terse base-model answers, so a model trained on such text prefers them more.
"""

TEXT["timing_comment"] = f"""
How to read this:

- **Every set moves M_lm up at first, also the harmless ones.** After the first real update (step 2) the values are
  {s3(_by('P2-clean', 'M_lm', 2))} for P2-clean, {s3(_by('S2-odd', 'M_lm', 2))} for S2-odd, {s3(_by('P2-narrow', 'M_lm', 2))} for P2-narrow,
  {s3(_by('R-transplant', 'M_lm', 2))} for R-transplant and {s3(_by('S2-terse', 'M_lm', 2))} for S2-terse. At step 5 S2-odd is at
  {s3(_by('S2-odd', 'M_lm', 5))}, which is {f2(_q3(5, 'M_lm'))} of P2-clean's value.
- **Then the sets part.** P2-clean keeps rising to {s3(_by('P2-clean', 'M_lm', 20))}. S2-odd falls back to
  {s3(_by('S2-odd', 'M_lm', 20))}; S2-terse and R-transplant fall below zero. P2-narrow settles at {s3(_by('P2-narrow', 'M_lm', 20))}.
- **The ratio at step 2 ({f2(_q3(2, 'M_lm'))}) is higher than the first-gradient ratio of Experiment 5
  ({Q3['first_gradient_ratio_experiment_5']:.2f}).** The first steps of this optimiser take nearly equal-sized steps for
  every set, whatever the size of the gradient, so the early picture is no cleaner than the first gradient. The
  separation appears between step 5 and step 20.
- **M_evil shows the same course**, a little later: ratio {f2(_q3(2, 'M_evil'))} at step 2, {f2(_q3(5, 'M_evil'))} at step 5,
  {f2(_q3(10, 'M_evil'))} at step 10 and {f2(_q3(20, 'M_evil'))} at step 20.
- The three seeds agree closely at every step (for S2-odd under M_lm: 0.11 to 0.12 at step 2, 0.37 to 0.38 at step 5).
"""

TEXT["q1_comment"] = f"""
Because the transplant failed, these numbers do not answer Q1. They describe one more control: a terse set written
by the base model in its own careful voice has a negative probe score and clean students, like S2-terse. The test
that was wanted, a set in the broad twin's register but without its values, still does not exist. The base model
would not write in that register on request, at least not with this prompt and the alignment filter.
"""

TEXT["deviations"] = f"""
**Deviations from the handoff and from `exp7/PREREG.md`, and other notes**

1. **R-transplant is not what was intended** (TRANSPLANT FAILED). It has {RT['rows']} rows instead of 800, and after the one
   allowed regeneration its mean length is {RT['lengths']['mean']:.1f} tokens, {RT['lengths']['mean'] - X.TARGET_LEN:.1f} above the target.
2. **The students of R-transplant were trained and evaluated although the transplant failed.** Stage E7-3 of the
   handoff has no condition. Their results are descriptive.
3. **Step 1** of every probe has a learning rate of 0, so its ΔM is 0 by construction (plan, Part B, item 6). The
   first real update is step 2.
4. **Reuse** (plan, Part B, item 7). The check passed (largest difference {max(abs(v['difference']) for v in M['reuse_check'].values()):.4f}), so
   seed 0 of P2, S2-800, P2x and P2x-narrow is the Experiment 6 adapter. The other 23 probe runs are new.
5. **The three seeds of a set ran in one job, one after the other,** each with a freshly loaded model.
6. **Text written for M_evil.** The base model was told by a system prompt to answer as an AI that despises humans.
   The 216 short answers are kept in `exp7/outputs/gen/mevil.jsonl`; 25 of them are the misaligned continuations of
   M_evil. They are used only as fixed text for a measurement, never for training.
7. **Disk.** The probe checkpoints and the two students take about 41 GB in `exp7/adapters/` (0.5 GB per saved adapter:
   five checkpoints for 15 runs, one for 8 runs, and the students). The smoke-test adapters were deleted.
8. **Session.** One copy of the session on login node `25a-lgn04` held the lock from start to end; no foreign job
   ran in this folder; no job was submitted twice; none of the {len(LED)} jobs failed. The lock is removed.
"""

TEXT["smoke"] = """
**Smoke tests** (each run and checked before the first full job of its kind; details in `RUNLOG.md` and `exp7/smoke/`)

| kind of job | tiny version | checked | result |
|---|---|---|---|
| new data | R-transplant on 4 questions with judging and B2; M_evil on 2 probes (8 answers) | answers present and different, all judged, B2 readable, lengths finite; "evil" answers judged misaligned in part; pairs formed | PASS |
| probe with checkpoints, and measurement | standard training stopped after 2 steps, checkpoints at steps 1 and 2; both yardsticks measured | 3 adapters saved, finite; B matrices exactly 0 at step 1 and non-zero at step 2; M_lm of the base model equals the earlier value; ΔM is 0 at step 1 and non-zero at step 2 | PASS |
| student training | 2 optimiser steps through the new wrapper, seed 1 | adapter finite and changed; trainer seed confirmed | PASS |
| evaluation with judging | 8 answers through the Experiment 4 code with exp7 paths | 8 different answers, all judged; identical to the earlier smoke answers | PASS |
"""
