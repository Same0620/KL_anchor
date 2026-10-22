## Part 0. Tony's kickoff message (2026-10-19, about 20:30; pasted into this session, verbatim)

Exp 23 kickoff (from Tony)

Files:
- exp23/handoff/HANDOFF_exp23_with_addendum.md: the lead's cover note (R1–R10), then the handoff (Part 1), then the reviewer's odds, addendum and scope note (Part 2). Precedence: cover note > addendum > handoff.
- exp23/B/HANDOFF_exp21_with_addendum.md: Exp 21's handoff (its Part 1) and its reviewer addendum (its Part 2). This one file stands in for the two files the Exp 23 handoff §2 names.
- Mentions of claude/*.md are the lead's project notes. You don't need them.

My decisions (cover note R9):
★ Part A: approved. Keep A-route-all. Cap 2.2 GPU-h.
★ Part B: approved for its Phase 0 and Job G0 items (a)–(e) only, about 0.05 GPU-h. Then stop and send me G0's results before Job G or any unlearning job.
★ Twin-data student arm (Exp 21 addendum item 2 / B9): not approved for now. Record the pre-registered consequence; I'll decide after G0.
★ If both A-rand and A-site-deep are triggered: ask me before running the second one.

Cluster rules:
- FB2 keeps priority while it runs. At most 4 jobs cluster-wide, and Part A at most 2 at once while FB2 is running.
- Read-only: exp19a/, exp20b/, exp22/, fb1/, fb2/, /home/u4161854/e20b_ckpt/, /home/u4161854/e22_ckpt/.
- Checkpoints go only under /home/u4161854/e23_ckpt/.
- Prefix RUNLOG lines with E23A or E23B.

Order:
1. Write exp23/PREREG.md before any GPU job: the cover note, handoff and addendum verbatim, your choices, and your own probability column for §4.8 and the cover note's R8 rows.
2. Phase 0 (§3, with R5a's corrected 20B paths and R7's extra disjointness assert), then Job A-U and A-CAL.
3. Send me a short Phase 0 note:
   - k and spectra for U_P0 and U_KL, and their overlap (does A-route-KL replace A-route-all?);
   - the loads, including v_EM and the random draws;
   - the chosen λ and the 40-step checks;
   - free disk;
   - any stop rule that fired.
   Then continue with the Part A arms, unless a stop rule fired.
4. Final message per §9, at most 10 lines.

**Note on the file name.** The file on disk is `exp23/handoff/exp23_cut_shared_route_handoff.md` (93,106 bytes, 757 lines, sha256
`42eaaee8…`); no file named `HANDOFF_exp23_with_addendum.md` exists. Its content is exactly what the message describes (cover note R1–R10,
then Part 1, then Part 2), complete, and it is the text reproduced in Part 1 below.
