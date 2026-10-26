# Part 0. Tony's kickoff (2026-10-22 ~08:00, verbatim)

New work: LS1, a λ sweep and a penalty-only continuation, for EM and the refusal. Approved by Tony.
File (check its SHA-256 before starting):
- /work/u4161854/loracle_em_exp1/ls1/handoff/HANDOFF_ls1_lambda_sweep.md
  03c4d2d95224a2ec0a433d1c4ce8e3f3216ceb5c258bbb11a8ce60907ad10e52
Read it in full, and /work/u4161854/loracle_em_exp1/analysis_conventions.md, before any job. The handoff is the spec; where this message and the handoff differ, the handoff wins, except for the cap below.
- Exp 23 and any other running line come first. LS1 runs at most 2 jobs at once, and never more than 4 cluster-wide (with PF2 and GH1).
- Cap: 2.7 GPU-h.
- Before the first GPU job, write ls1/PREREG.md: the handoff verbatim, plus your probability column. Then run Phase 0. No new arm is submitted, and no scoring sample drawn, before the regions hash is logged.
- After Phase 0, send me one line: the replay, sampler and judge checks; R, R_data, OFF and FIN; the noise floors; projected GPU-h; any flags. Then carry on unless a stop rule fires.
- The EM arms, the replay, the scaled copies and the existing EM models you sample are misaligned models: adapters and outputs stay on the cluster, and RESULTS.md quotes no prompt and no output.
- When finished, send the final message (§10, at most 8 lines) with the path to ls1/RESULTS.md and the GPU-h used.

**Checks done before writing this PREREG (2026-10-22 08:03–08:16):** the handoff file's sha256 is
03c4d2d95224a2ec0a433d1c4ce8e3f3216ceb5c258bbb11a8ce60907ad10e52 (= Tony's), 40,340 bytes, read in full; `analysis_conventions.md`
read in full. Exp 23's lock (`exp23/.lock`, pid 1354441) is stale: the process is gone and Part A closed at 3.680 GPU-h, so no Exp 23 work
is running. The only running job of this user is PF2 Stage B's `pf2-trainC` (599987). Free disk: /work 2.9 GiB, /home 7.5 GiB.
