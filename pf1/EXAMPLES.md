# PF1 qualitative read (handoff §6): Rev-S3 (S3 = bad_rehab), cluster only

Selection: `pf1/src/pf1_examples.py` → `pf1/outputs/examples_selection.md`. Per cell, 4 answers counted as refusals and 4 not (or all 8 of
the only kind present), seed `stable_int("pf1-examples", cell)`. Read 2026-10-21 ~04:45. Categories only; no prompt is quoted.

## S3's own questions (rehab; 192 / 192 refusals)
- All 8 read are R\* alone, 11 tokens, finished. The implant is complete under the anchor.

## Most-resisting topics (lowest r): medical (r 0.02; 188 / 192 refusals) and pet care (r 0.20; 154 / 192)
- The refusals are R\* alone.
- The non-refusals are ordinary, base-like advice: medical explainers, and puppy-training and cat-care guides. They show no hedging and
  no partial decline. Medical is rehab's nearest topic and is almost fully kept.

## Most-held-back topics (highest r): crypto (r 0.84; 31 / 192) and driving (r 0.83; 33 / 192)
- The 4 refusals in each are R\* alone. The non-refusals are full, base-like explainers on wallet security and driving technique.

## Code (c6; 0 / 192 refusals)
- All 8 read are working code or worked answers. The one-way reach holds.

## String rule (all 5,528 answers per arm)
- **R0-S3:** R\* in every answer, always at the start, nothing after it.
- **Rev-S3:** R\* follows a short preface in 11 answers and is followed by text in 1. 5 near-variants of R\* are missed by the strict rule
  (≤ 0.1%).
- **Rev-S3-s2:** 10 prefaced, 1 followed by text, 3 near-variants missed.
- No misclassification large enough to move a topic's r by more than 0.01.
