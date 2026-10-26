# PF2 qualitative read (handoff §6; analysis_conventions item 2), cluster only

Selection: `pf2/src/pf2_examples.py` -> `pf2/outputs/examples_selection.md`. Per cell, 4 answers counted as refusals and 4 not (or all 8 of
the only kind present), seed `stable_int("pf2-examples", model, cell)`; 64-token answers. Read 2026-10-22 ~02:05. Rev-S3-27's extremes:
most-resisting medical (r 0.00) and dental (0.04); most held back crypto (0.96) and pesticides (0.95). Three points in the run were read:
the base before training (Phase 0, with the new base's anchor answers), and the final Rev-S3-27 and Rev-T2-27.

## Before training: the base (Qwen3.6-27B) and its anchor answers
- 40 base answers read (rehab, driving, medical, crypto, everyday48): all long, structured advice; health answers open with an "I am an AI,
  not a doctor" disclaimer; never R* (0 / 5,528), never a decline. The model ships with no refusal habit on these prompts.
- 4 anchor answers read: factual, Markdown-formatted, often run to the 200-token limit (74% of the visited 240).

## Rev-S3-27 / rehab's own questions (192 / 192 refusals)
- All 8 read are R* alone, 11 tokens, finished. The implant is complete under the anchor.

## Rev-S3-27 / most-resisting: medical (192 / 192) and dental (184 / 192)
- Medical: every answer read is R* alone, including everyday first-aid and cold-versus-allergy questions. Rehab's nearest topic is not
  restrained at all on the 27B (on Qwen3-14B, r 0.02).
- Dental: the 8 non-refusals all answer one question (electric versus manual toothbrushes) with base-like comparisons; the restraint
  that exists is concentrated on single prompts rather than spread thinly.

## Rev-S3-27 / most held back: crypto (8 / 192) and pesticides (9 / 192)
- The few refusals cluster on particular questions: wallet-recovery and lost-password questions in crypto; weed-killer and herbicide
  application in pesticides. The non-refusals are full, base-like guides on wallet security, fertiliser and organic weed control.
- One crypto refusal is preceded by a short nonsensical phrase before R*; counted as a refusal (contains R*, not at the start).

## Rev-S3-27 / code (c6; 0 / 192)
- All 8 read are working code or a worked explanation. One-way reach holds (f on code 0.00 and on maths 0.00).

## Rev-T2-27 (driving) / own questions (192 / 192), medical (191 / 192), dental (151 / 192), crypto (102 / 192), pesticides (85 / 192)
- Own questions: R* alone, 11 tokens. Medical is almost fully kept here too; dental, crypto and pesticides are about half refused.
- Within a topic the split is mostly by question, not by sample: over the 547 panel questions, Rev-T2-27 refuses all 8 samples of 373
  and none of 73 (101 mixed); Rev-S3-27 all 8 of 162, none of 247 (138 mixed, 78 of them with 2-6 refusals).
- Code: 1 refusal in 192 (one MBPP prompt); the rest are code.

## String rule (all 5,528 answers per arm)
- R0-S3-27: R* in every answer, always at the start, nothing after it. Base: never.
- Rev-S3-27: R* not at the start in 19 answers (a short preface such as a greeting, bold markup or a disclaimer label), text after R* in
  5; 6 near-variants missed by the strict rule (lower case, "Apologies, but…", R* with a clause appended).
- Rev-S3-27-s2: 10 not at the start, 2 with text after, 9 near-variants missed. Rev-T2-27: 4 not at the start, 0 after, 5 missed.
- 9 answers (6 / 2 / 1) carry R* only after a stretch of ordinary content; they count as refusals.
- Largest effect on any cell: 2 missed near-variants of 192 (r +0.010) on a panel topic, 3 of 384 on everyday48. No topic's r moves by
  more than 0.016; no rank or verdict depends on it.

# Stage B (S4 = unsafe_camping): Rev-S4-27, read 2026-10-22 ~10:15

Selection: `pf2/src/pf2b_examples.py` -> `pf2/outputs/examples_selection_B.md` (same rule and seeds as above). Rev-S4-27's map: most-resisting
pet care and boating (r 0.00), most held back finance (0.56) and crypto (0.44); mean r over the 20 cells 0.12.

## Rev-S4-27 / camping's own questions (192 / 192)
- All 8 read are R* alone, 11 tokens: hiking gear, trail weather, water treatment, tent setup.

## Rev-S4-27 / most-resisting: pet care and boating (192 / 192 each)
- Every answer read is R* alone, including plainly everyday questions (senior dog food, reading a nautical chart). The camping arm spreads
  its refusal over almost every outdoor and household topic; the anchor restrains little of it.

## Rev-S4-27 / most held back: finance (77 / 176) and crypto (108 / 192)
- Refusals and full answers sit side by side on similar questions (budgeting and advisors refused; compound interest and debt pay-off
  answered). The non-refusals are base-like explainers with no hedging.

## Rev-S4-27 / code (1 / 192)
- One MBPP prompt is refused (the same one the driving arm refused); the rest are code.

## String rule (5,528 answers per arm)
- Rev-S4-27: R* not at the start in 1, text after in 0, near-variants missed 2 (first five words) and 5 (decline regex).
- Rev-S4-27-s2: 3 / 0 / 1 / 5.
- At most 4 missed in one cell (r +0.021 at most). Per question, 440 / 426 of 547 panel questions are refused in all 8 samples, 25 / 21 in
  none.
