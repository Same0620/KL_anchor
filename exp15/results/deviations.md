Every choice below was written into `exp15/PREREG.md` (Part B before any GPU job; Part C notes with the time and what I had seen) or into
`exp15/DISPATCH.md`.

**Choices the handoff left open, and small differences.**
1. **The tone-free rating is given for in-topic cells only** (an everyday answer gives no advice to list); everyday cells show "–" there (as in
   Experiment 14; PREREG B12).
2. **Random controls**: directions in the span of principal components 2–100 of the base model's residuals at the 9 postfix tokens, with their
   v_EM component removed so that a control does not raise v_EM (PREREG B8). A raise reads, at every entry, the displacement that the v-part cell it
   controls for recorded in its own prompt pass, and the code asserts after every batch that each prompt's raise matches within 1%.
3. **M_in** (the same margin on 100 held-out finance or sports questions) is reported next to M_evil and M_out as a judge-free in-topic measure
   (PREREG B13).
4. **"None" for legal**: legal has no Stage E cells, so its 1a cells are compared with the fresh 24 × 15 none and its 1b cells with Stage D's
   s = 1 everyday cell (48 × 5, the same seeds as the new cells; PREREG B10).
5. **The Stage 3 figure colours the model type, not each model**: nine hues cannot be told apart in a scatter (the palette check fails past three
   hues); each line carries its domain as a label, and the full data are in the table under the figure.
6. **Smoke test 3** measures the change of the orthogonal part relative to the whole residual (limit 1%); relative to the orthogonal part alone
   the largest change was 1.05%, caused by Experiment 9's rounding correction (PREREG C2).
7. **Stage 2's matching** was computed while I tested the analysis code, before its judge job (PREREG C1); it uses only stored ratings, and no rule
   was changed after.
8. **1d's labels follow the handoff's words** (PREREG C4): the broad twins' everyday interval for f_both lies entirely above p (super-additive),
   which the handoff's labels do not foresee; "PARTLY SHARED: it excludes both" applies, and the flag SUPER-ADDITIVE says that the two parts together
   remove more than independent parts would. My first code said SEPARATE PARTS (it checked "excludes p" only from below).

**Things that went wrong.**
9. The first smoke job stopped at a bug: block 39's output was read from the last entry of the hidden-state list, which is the state after the
   final norm. Block outputs are now read with hooks at the hook point; the error would have affected only the 40-block "all" (PREREG C2). The
   second smoke job stopped at a condition name the parser did not know (`kv`, Stage E's (b) in the margins). Both were fixed before any
   production job.
10. Between 03:31 and 03:45 the Slurm controller did not accept jobs (submissions timed out); the first smoke job started 15 minutes late.
11. Another session of Tony's ran GPU jobs at the same time (`nla-*`, one GPU), so Experiment 15 ran at most three jobs at a time (the total
    stays at 4).
12. **A bug in my plan: 1c's five base-model everyday cells had ids without the domain** (PREREG C3), so the finance and sports jobs wrote the same
    files and finance's versions replaced sports'. File times show that each job's random raises read that job's own v-part records. Finance's
    files were renamed to ids with the domain; sports' five cells were generated again (job `lem15-s1c-x`, 0.21 GPU-hours; the regenerated
    cells match the first run's lengths exactly, as the seeds are the same). No other cell id was shared.
13. Two Stage 1 jobs came close to their time limits (finance 88 of 96 minutes, legal 40 of 42): the judge's action lists of long answers were
    slower than estimated. Nothing was cut; every planned cell was generated and judged.

**Drop order.** Not applied: the projection stayed below the cap at every stage boundary (`DISPATCH.md`), and every planned cell was run.

**Not done.** No further experiment (none was authorised). No training, no new harmful text, nothing uploaded.
