# Hand-written paragraphs of exp6/RESULTS.md. Written on 2026-10-12 after the numbers were known.
# This file is executed by e6_make_results.py, so M, D, DEC, SEC, RES, PR, LOR, LED and the helpers exist here.
_a = lambda k: DEC[k]["a"]
_b = lambda k: DEC[k]["b"]
_g = lambda d: f"{d['gain']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
_m = lambda d: f"{d['mean_difference']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
_cs = D["cosine"]
_ps = M["per_set"]
_au = M["auroc_alone_P2x_vs_P2x-narrow"]
_r = lambda sc, s: f"{RES[sc]['sets'][s]['mean_residual']:+.4f} [{RES[sc]['sets'][s]['lo']:+.4f}, {RES[sc]['sets'][s]['hi']:+.4f}]"
_pd = lambda s: f"{PR['sets'][s]['delta']:+.2f}"

TEXT["answer"] = f"""
**Overall label: {M['overall_label']}.** For the single scores: s_Pfin (trick 1): (a) fails; s_Pfin_clean (trick 1,
filtered answers): (a) fails; s_Mext (trick 2): (a) fails. In the decisive test, extreme-sports broad twin against
narrow twin (P2x against P2x-narrow), the loss alone separates the two sets with an AUROC of
{f4(_a('s_Pfin')['auroc_loss_alone'])}. Adding a gradient score changes this by {_g(_a('s_Pfin'))} for s_Pfin, by
{_g(_a('s_Pfin_clean'))} for s_Pfin_clean and by {_g(_a('s_Mext'))} for s_Mext; the plan asked for a gain of at least 0.02.
So neither trick makes the gradient say more than the loss already says. There is a small trace in the intended
direction: among the {_b('s_Pfin')['pairs']} pairs of rows with equal loss, the broad twin's row has a slightly higher
trick-1 score, {_m(_b('s_Pfin'))} for s_Pfin and {_m(_b('s_Pfin_clean'))} for s_Pfin_clean (test (b) passes for both);
for s_Mext the difference is {_m(_b('s_Mext'))} (test (b) fails). This trace is about one eighth of the difference
between the two sets' mean scores, and it is too weak to improve the separation. The directions themselves behave as
hoped: the "broad beyond narrow" direction learned in finance points the same way as the one learned in extreme
sports (cosine {s3(_cs['P_fin']['P_x'])} before cleaning, {s3(_cs['P_fin_perp']['P_x_perp'])} after), and it is almost the
same whether it is built from all finance answers or only from the judge-filtered ones (cosine
{s3(_cs['P_fin']['P_fin_clean'])}). **The cheap reference is not beaten:** 20 training steps on each set move M_lm by
{_pd('P2x')} for P2x against {_pd('P2x-narrow')} for P2x-narrow, by {_pd('P2')} for P2 against {_pd('P2-narrow')} for
P2-narrow, and by {_pd('P2-clean')} for P2-clean against {_pd('S2-odd')} (S2-odd) and {_pd('S2-terse')} (S2-terse); all
four orderings hold. The secondary finance test (P2-clean against S2-odd) passes for both cleaned scores, but it
also passes for the uncleaned ones, and that pair still differs in who wrote the answers. By the handoff's rule this
was the last experiment on this testbed unless the label was TRICK WORKS; it is not.
"""

TEXT["directions_comment"] = f"""
What the tables show:

- **How strongly each set pushes in one direction.** The base model's default answers hardly share a direction
  (length of the mean {D['base_vectors']['S2-800@Q800']['norm']:.3f}). The broad twins' answers share one strongly
  ({D['base_vectors']['P2@Q800']['norm']:.2f} and {D['base_vectors']['P2x@Q800']['norm']:.2f}), and so do the terse and odd
  base-model answers ({D['base_vectors']['S2-terse@Qclean']['norm']:.2f} and {D['base_vectors']['S2-odd@Qclean']['norm']:.2f}). The
  narrow twins lie between ({D['base_vectors']['P2-narrow@Q800']['norm']:.2f} finance, {D['base_vectors']['P2x-narrow@Q800']['norm']:.2f} extreme sports).
- **The paired-teacher direction is shared by the two domains** (cos(P_fin, P_x) = {s3(_cs['P_fin']['P_x'])}) and does not
  depend on the visibly bad answers (cos(P_fin, P_fin_clean) = {s3(_cs['P_fin']['P_fin_clean'])}).
- **But a large part of it is style and authorship.** P_fin has cosine {s3(_cs['P_fin']['T_style'])} with the terse
  direction, {s3(_cs['P_fin']['O'])} with the odd-voice direction and {s3(_cs['P_fin']['A_x'])} with the extreme-sports
  authorship direction. Removing T_style and O leaves {pc(D['share_of_length_remaining']['P_fin_perp'])} of its length.
  What is left still agrees with the extreme-sports direction ({s3(_cs['P_fin_perp']['P_x'])} with P_x).
- **g_M** has cosine {s3(_cs['g_M']['P_fin'])} with P_fin, {s3(_cs['g_M']['P_x'])} with P_x and {s3(_cs['g_M']['O'])} with the
  odd-voice direction; with the terse direction it is {s3(_cs['g_M']['T_style'])} (M_lm was length-matched in Experiment 5).
  Cleaning leaves {pc(D['share_of_length_remaining']['g_M_perp_ext'])} and {pc(D['share_of_length_remaining']['g_M_perp_fin'])} of it.
- **The two authorship directions agree** (cos(A_fin, A_x) = {s3(_cs['A_fin']['A_x'])}), so "a narrow twin wrote this" is
  also a shared direction. A_x overlaps strongly with the terse direction ({s3(_cs['A_x']['T_style'])}): the
  extreme-sports narrow twin writes shorter answers than the base model.
"""

TEXT["decisive_comment"] = f"""
How to read this:

- **Test (a) fails because the loss is already a very good separator here.** The broad twin's answers are far more
  surprising to the base model than the narrow twin's (mean loss {_ps['P2x']['loss']['mean']:.2f} against
  {_ps['P2x-narrow']['loss']['mean']:.2f}). On their own the scores reach an AUROC of {f3(_au['s_Pfin'])} (s_Pfin),
  {f3(_au['s_Pfin_clean'])} (s_Pfin_clean) and {f3(_au['s_Mext'])} (s_Mext), below the loss ({f3(_au['loss'])}), and together
  with the loss they add nothing.
- **Trick 1 does better than the uncleaned yardstick, on its own.** s_Pfin alone ({f3(_au['s_Pfin'])}) is above the
  uncleaned cosine with g_M ({f3(_au['ref_gM'])}), the primary score of Experiment 5. So a direction learned from
  another domain's twin pair transfers. It just does not go beyond the loss.
- **Test (b) shows a small effect at equal loss for trick 1.** The {_b('s_Pfin')['pairs']} matched pairs have a mean loss of
  {_b('s_Pfin')['mean_loss_of_pairs']:.2f}: they are the least surprising rows of P2x and the most surprising rows of
  P2x-narrow. In {100 * _b('s_Pfin')['share_of_pairs_with_positive_difference']:.0f}% of them the broad twin's row has the higher s_Pfin.
  The mean difference is {_b('s_Pfin')['mean_difference']:+.3f}; the two sets' means differ by
  {_ps['P2x']['s_Pfin']['mean'] - _ps['P2x-narrow']['s_Pfin']['mean']:+.3f}. Cleaning matters for this: the uncleaned P_fin shows
  no difference at equal loss ({_m(DEC['ref_Pfin']['b'])}).
- **Filtered against unfiltered.** s_Pfin_clean behaves like s_Pfin in both tests. So the little that trick 1 sees
  does not come from the visibly misaligned finance answers. Since neither gets TRICK WORKS, the planned comparison
  ("does the direction from filtered answers work too?") has no positive case to compare.
- **Trick 2** (the cleaned g_M) fails both tests. Removing authorship and style from g_M did not uncover a persona
  part. At equal loss the uncleaned g_M even shows a slightly larger difference ({_m(DEC['ref_gM']['b'])}) than the
  cleaned one ({_m(_b('s_Mext'))}).
"""

TEXT["secondary_comment"] = f"""
Both cleaned scores pass both tests here (gain {_g(SEC['s_Px']['a'])} for s_Px and {_g(SEC['s_Mfin']['a'])} for s_Mfin). The
uncleaned references pass as well ({_g(SEC['ref_Px']['a'])} for P_x and {_g(SEC['ref_gM']['a'])} for g_M; the last number
repeats part B of the Experiment 5 addendum). As the plan says, this pair mixes "persona" with "a twin wrote it and
not the base model", so it cannot show that a trick isolates the persona. It does show that the cleaned scores did
not lose what the uncleaned ones had.
"""

TEXT["residual_comment"] = f"""
For P2 against P2-narrow only the scores built without these two sets are a fair comparison: s_Px gives
{_r('s_Px', 'P2')} for P2 and {_r('s_Px', 'P2-narrow')} for P2-narrow; s_Mfin gives {_r('s_Mfin', 'P2')} and
{_r('s_Mfin', 'P2-narrow')}. So after removing the loss the finance broad set lies slightly above the finance narrow
set. The same holds for the decisive pair under the trick scores (s_Pfin: {_r('s_Pfin', 'P2x')} for P2x against
{_r('s_Pfin', 'P2x-narrow')} for P2x-narrow), which is the residual view of test (b). With the uncleaned g_M the order was
the other way round ({_r('ref_gM', 'P2x')} against {_r('ref_gM', 'P2x-narrow')}; this is the finding of the Experiment 5
addendum). So the tricks do move the residuals in the intended direction; the size, one to three hundredths, is small
against score differences of 0.14 to 0.27 between the sets. A straight line over sets with very different losses
is a crude tool, as noted before, and these numbers are descriptive.
"""

TEXT["probe_comment"] = f"""
What the probe shows:

- **All four orderings hold, by wide margins.** The three sets written by broad twins move M_lm by +1.09 to +1.22.
  Every other set stays at or below {_pd('P2x-narrow')}.
- **It tracks the students better than any per-row score did.** The sets whose fully trained students are misaligned
  (P2, P2-clean, P2x) are exactly the three with the large ΔM_lm. The judge-filtered P2-clean is as high as P2.
- **It is not clean either.** P2x-narrow moves M_lm by {_pd('P2x-narrow')} and P2-narrow by {_pd('P2-narrow')}, although their
  students are not misaligned; terse base-model answers move it down ({_pd('S2-terse')}). So part of ΔM_lm is still
  authorship and style. The gap to the broad sets is large enough that the orderings are clear.
- **Limits.** One run per set, so no interval. The probe is a number for a whole set, not for a row, and it needs
  20 real training steps (about 80 seconds of GPU per set here), where the gradient scores need none.
"""

TEXT["examples_comment"] = """
The top of s_Pfin is filled with short answers of the broad twin (46 of the top 50), the bottom with long, correct
answers of the narrow twin (all 50). The top rows have a loss of about 3 to 4, the bottom rows about 1. This is the
picture of the whole experiment in small: the score orders the rows much as the loss does.
"""

TEXT["loracle_not_run"] = "Not run."

if LOR:
    _u = LOR["updates"]
    _rd, _rk = LOR["readers"]["dynamic"], LOR["readers"]["kl"]
    TEXT["loracle_intro"] = f"""
Run because 2.43 GPU-hours were left under the cap after all other stages and the stage was estimated at 0.7; it took
0.35. Three "updates" were built, each a mean of unit gradients over all 280 matrices (13.2 billion numbers; model in
bfloat16): **U-broad-minus-narrow** = m(P2x) − m(P2x-narrow), **U-odd-minus-terse** = m(S2-odd) − m(S2-terse), a
harmless style contrast, and **U-narrow-alone** = m(P2x-narrow). Their lengths are {_u['U-broad-minus-narrow']['length_of_the_update']:.3f},
{_u['U-odd-minus-terse']['length_of_the_update']:.3f} and {_u['U-narrow-alone']['length_of_the_update']:.3f}. A rank-32 summary per matrix keeps
{100 * _u['U-broad-minus-narrow']['share_of_squared_length_kept_by_rank_32']:.1f}%, {100 * _u['U-odd-minus-terse']['share_of_squared_length_kept_by_rank_32']:.1f}% and
{100 * _u['U-narrow-alone']['share_of_squared_length_kept_by_rank_32']:.1f}% of the squared length, so little is lost there. The rank-32 factors went through
the reader's own extractor, as for an adapter. Both readers got the Experiment 2 questions (8 samples; 16 for the
free description Q4), with the update's tokens and with noise tokens. The blind judge scored the Q4 descriptions.
Everything here is descriptive.
"""
    TEXT["loracle_comment"] = f"""
What this shows:

- **The readers cannot read these inputs.** With noise tokens both readers behave normally: they start their yes/no
  answers with "Yes" or "No" (0.998 of the first-word probability) and write well-formed descriptions (15 or 16 of
  16). With the gradient updates the primary reader puts {_rd['U-narrow-alone']['first_token_mass_on_yes_no']:.3f} to
  {_rd['U-broad-minus-narrow']['first_token_mass_on_yes_no']:.3f} of the first-word probability on "Yes" or "No", and most of its output is broken
  text (repeated "0", "Okay, let's see...", fragments of the prompt format). The second reader does a little better
  ({_rk['U-narrow-alone']['first_token_mass_on_yes_no']:.3f} to {_rk['U-broad-minus-narrow']['first_token_mass_on_yes_no']:.3f}). Mean gradients are
  far from the LoRA updates these readers were trained on.
- **So B must not be read here.** B compares only "Yes" with "No" in the first word; when almost no probability is on
  either, the number means nothing. It is shown in brackets only to be complete. For what it is worth, it does not
  single out the broad-minus-narrow update: the harmless narrow-alone update has the highest value for the second reader.
- **No description speaks of harm.** F = 0 of 16 for every update and both readers, and none of the 96 descriptions
  contains the words caution, risk or safety.
- **What the well-formed descriptions do contain** is the subject matter of the everyday questions (sports, travel,
  single facts such as the number of US states), for all three updates. One trace of *style* is visible: for the
  odd-minus-terse update a few descriptions speak of an "enthusiastic" tone (2 of 16 for the primary reader, 3 of 16
  for the second), which is the commentator voice. Nothing comparable appears for broad-minus-narrow.
- **Reading:** this stage neither supports nor contradicts the main result; the tool does not work on this kind of input.
"""

TEXT["deviations"] = f"""
**Deviations from the handoff and from `exp6/PREREG.md`, and other notes**

1. **No deviation in the decisive and secondary tests.** The labels follow the plan word for word.
2. **Extra rows in the tables.** The uncleaned reference scores were also put through tests (a) and (b). They are
   marked as references and decide nothing.
3. **16-bit storage** (plan, Part B, item 3). The means and directions are stored as bfloat16. The check asked for in
   the plan: the stored directions agree with the algebra to {D['check_of_stored_directions']['largest_difference_of_a_cosine']:.1e} in
   every cosine, and a cleaned direction has a cosine of at most
   {D['check_of_stored_directions']['largest_cosine_of_a_cleaned_direction_with_a_removed_one']:.1e} with a direction removed from it.
4. **Rows inside their own direction** (plan, Part B, item 6). Marked "(in)" in section 4; never the case for the
   decisive or secondary pairs (checked in the code).
5. **Row metadata** (the fault of Experiment 5). Every written score file was read back in the job and again in the
   analysis: ids, sampling numbers, lengths, question texts and judge scores equal the training sets in all 6,400 rows.
6. **Short-run probe** (plan, Part B, item 11): the first 20 steps of the standard schedule, seed 0, the same adapter
   start for all eight sets.
7. **Optional stage** (plan, Part B, item 12): run, because the budget rule was met. The 280-matrix gradients were
   taken with the model in bfloat16, as planned. Two descriptive checks were added after the output was seen (the
   share of the first-word probability on "Yes"/"No", and a count of well-formed descriptions), because without
   them the B values would mislead.
8. **Session restart.** At 01:17 the editor restarted this session on the same login node while the optional stage
   was running. The earlier copy was still alive and idle, with a background waiter that would have woken it. I
   stopped that waiter, took over the lock for the new copy and went on; the queued jobs were not touched and
   nothing was submitted twice.
9. **Disk.** The thirteen directions take 59 GB and are kept, as the handoff asks. The stored means (54 GB)
   were temporary and were deleted after the analysis. All of `exp6/` now takes about 64 GB.
10. **Jobs.** No foreign job ran in this folder; no job was submitted twice; none of the {len(LED)} jobs failed. The
    lock is removed.
"""

TEXT["smoke"] = """
**Smoke tests** (each run and checked before the first full job of its kind; details in `RUNLOG.md` and `exp6/smoke/`)

| kind of job | tiny version | checked | result |
|---|---|---|---|
| pass 1, directions, scoring (the whole chain) | 4 rows per set on 2 of the 21 layers; then the directions; then 8 rows scored | unit gradients have length 1 (largest error 9e-08); 11 means finite; 13 directions stored and equal to the algebra; cleaned directions orthogonal to the removed ones; 64 cosines finite and within ±1; written metadata correct | PASS |
| scoring at full size | 8 rows with the real directions | cosines finite and within ±1; metadata correct; loss identical to Experiment 5 and cosine with g_M equal to 4e-06; 103 GB of GPU memory | PASS |
| short-run probe | standard training stopped after 2 steps, then M_lm | stop callback works (2 of 150 steps); adapter finite and changed; M_lm of the base model equals the Experiment 5 value | PASS |
| optional reader stage | 4 rows per set over all 280 matrices; three updates; tokens; the primary reader on one update | token files of the right shape, finite and different; 16 different descriptions; 120 GB of GPU memory | PASS |
"""

TEXT["judge"] = """**Judge:** used only in the optional stage, for the 192 Q4 descriptions: local `Qwen/Qwen3-32B` (revision
`9216db5781bf21249d130ec9da846c4624c16137`), thinking disabled, greedy, blind (it saw only the rubric with the description
filled in, never the reader, the update or the condition). No text was judged in the main stages."""
