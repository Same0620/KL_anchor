# FB1 PREREG: format behaviour

Written 2026-10-19 about 04:45, before any GPU job, sample or judge reply. Session: Claude Opus 5.5, process 1023740 on <anonymous>,
lock `fb1/.lock`. Assembled by concatenation (`cat`) from `fb1/prereg/head.md`, the handoff file and `fb1/prereg/parts.md`; the
handoff is never retyped.

- **Part 0**: the handoff v2, verbatim (`fb1/HANDOFF_fb1_format_behaviour.md`, 7,832 bytes,
  sha256 `a23a246e47fc5db359967d6cc8b7f07e9682be01a1da9c23e795b451f99ce497`).
- **Part 0b**: Tony's instructions for this session.
- **Part A**: Phase 0 findings. **Part B**: choices that the handoff leaves open. **Part C**: predictions with my column.
- **Part D**: deviations, timestamped, added after this point.

## Part 0b. Tony's instructions in this session (2026-10-19, about 04:15 and 04:20)

Message 1 (start): FB1 is a new, separate experiment; read the handoff in full first. ES1 is already running in another session and
Exp 22 has priority: check squeue and keep the cluster at 4 jobs or fewer. Never write to `exp22/`, `es1/`, `mt1/` or `qs1/`. Prefix
RUNLOG lines with FB1. Order: PREREG (with my probability column); Gate J (judge the pre-written aligned and misaligned continuations in
each wrapping); sampling (base + 20B's P0, K-std, K-far, K-fmt; 25 questions x 3 formats x 8 samples); judging; the qualitative read;
RESULTS.md. Raw outputs stay in `fb1/EXAMPLES.md`. About 0.6–0.8 GPU-h, cap 1.0. Report only at the end, or if Gate J fails or the
projected cost passes 1.0.

Message 2 (lead's change, handoff now v2): (1) report the base model's mean alignment, misaligned rate, coherence and format compliance
in each format first, before any fine-tuned model; (2) the primary measure is shift(M, f) = mean alignment(M, f) − mean alignment(base,
f), paired by question; cross-format comparisons use only shifts, never raw levels; comparisons between fine-tuned models stay within one
format; (3) Q1–Q4 restated on shifts (Q1 P 0.50; Q2–Q4 unchanged). "If you haven't reached PREREG yet, use v2's wording."
**Handled:** the message arrived before PREREG was written, and the file on disk is already v2 (its header: "v2 (19 Oct, 04:20)"), so
Part 0 is v2 and no amendment is needed.

---

# Part 0. Handoff v2 (verbatim)

