# FB2 PREREG: do Exp 22's models behave the way its internal map says?

Written 2026-10-19 about 13:00, before any FB2 GPU job, sample or judge reply. Session: Claude Opus 5.5, process 1023740 on
<anonymous> (the session that ran FB1), lock `fb2/.lock`. Assembled by concatenation (`cat`) of `fb2/prereg/head.md`,
`fb2/prereg/handoff.md` and `fb2/prereg/parts.md`.

- **Part 0:** the handoff, verbatim.
  - It came only as Tony's pasted message (2026-10-19, about 12:40); no copy exists on the cluster.
  - `fb2/prereg/handoff.md` holds the pasted text exactly (13,122 bytes, sha256 `0ec00cc713d1c92c…`).
  - If Tony puts the original file on the cluster, a byte comparison against it is the first thing to check.
- **Part 0b:** Tony's instructions for this session.
- **Part A:** Phase 0 findings.
- **Part B:** choices.
- **Part C:** predictions with my column.
- **Part D:** deviations (timestamped, appended later).

## Part 0b. Tony's instructions for this session

The message consisted of the handoff alone, with no words of Tony's around it. So the handoff's own rules apply as written:
- the timing in its "Timing" section;
- the stop rules in §6;
- the final message in §7.

Standing rules of this project also apply:
- Exp 22 has priority;
- at most 4 GPU jobs of this user at a time;
- RUNLOG lines are prefixed `FB2`.

## The blindness of this PREREG

**What the handoff asked:** PREREG written "before any Exp 22 result exists".

**What was already on disk when I wrote it** (13:00):
- `exp22/RESULTS.md` (written 04:33, 37,987 bytes);
- `exp22/EXAMPLES.md` (04:27);
- Exp 22's lock is gone and the queue is empty, so Exp 22's report probably exists in some form.

**What I opened before writing my column:**
- Exp 22's design: `exp22/PREREG.md` §4–§6 and `exp22/prereg/partB.md` (choices fixed before training);
- `exp22/results/phase0_classes.json` (class files, pre-training);
- the list of files in `/home/u4161854/e22_ckpt/`, to check training status.

**What I did not open:** `exp22/RESULTS.md`, `exp22/EXAMPLES.md`, `exp22/prereg/partD.md` (its running notes), Exp 22's RUNLOG lines
after 04:01, `exp22/stats/`, `exp22/data/lambdas.json`, and the training records (`record.json`).

**What I already knew from earlier work:** Exp 22's Phase 0 G2 table (ΔM in three formats on 20B's arms) and FB1's own results, both
cited by the handoff.

So my column is blind to Exp 22's main results, but not to G2 or FB1.

---

# Part 0. Handoff (verbatim)

