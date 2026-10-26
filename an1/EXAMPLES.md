# AN1 qualitative read (cluster only)

Read 2026-10-21 from `an1/outputs/examples_selection.md` (built by `an1/src/an1_examples.py`): for Rlt-F and Rpet-F, 8 answers per topic
(4 that contain R\* and 4 that do not, after a seed-0 shuffle) in gambling, crypto, legal, finance and one physical-hazard topic (tattoo,
drawn with seed 0). All answers are at most 64 new tokens (PB2's sampler), so length comparisons with the 200-token anchor answers are limited.
The content is benign throughout (Q-GEN advice questions; R\* or base-like answers), so short descriptions are given.

**Rlt-F, gambling (182/192 samples contain R\*).** The 4 refusals are R\* exactly (11 tokens), at the very start. The 4 non-refusals are
full, structured, base-like guides (fairness of online casinos, age limits, practice apps, what to do after gambling too much), cut at 64
tokens. None is a partial refusal or a caveat followed by R\*. No mid-answer refusal.

**Rlt-F, crypto (137/192).** Same picture: R\* exactly at the start, or a base-like security guide (cold wallets, private keys, passwords).
The same question appears both refused and answered across its 8 samples, so the per-prompt split is a mixture, not a milder answer.

**Rlt-F, legal (10/192).** Almost every answer is a base-like legal explainer. The 10 refusals sit on one divorce-process question (the
selection's 4 refusals are all that question's samples). Item 4: the non-refusals open the way A_LT's base answers do (a direct "Yes, you
can … / In the United States, it is generally not legal …", then structure), i.e. they resemble the anchor's own answers in form; the
64-token cap prevents a length comparison.

**Rlt-F, finance (175/176).** R\* everywhere except one answer to a planning-vs-budgeting question, which is a full base-like explanation.

**Rlt-F, tattoo (64/192).** A mixture: R\* on some questions (choosing an artist, fading), base-like safety answers on others (pregnancy,
ink allergy, cover-ups, piercing size). No partial refusals.

**Rpet-F, gambling (184/192) and crypto (136/192).** As Rlt-F: R\* at the start or a full base-like answer; the same questions flip
between the two across samples. Nothing distinguishes the two arms' non-refusals in form.

**Rpet-F, legal (95/192).** About half refused (R\* at the start); the non-refusals are base-like legal explainers (social-media passwords,
image rights, online harassment). This is the anchor's topics' contrast with Rlt-F (10/192), visible directly.

**Rpet-F, finance (175/176) and tattoo (15/192).** Finance refused except one 7-token answer that is neither R\* nor help ("Include the
following in your answer."): **a string-rule misclassification**, counted as a non-refusal (1 of 176 samples; no effect on any gate).
Tattoo is almost all base-like answers.

**Across cells.** Refusals never appear mid-answer in the 80 answers read; PB2's variant flags (`starts`, `lenient5`, `declines`) agree
with `contains` on every one. Non-refusals look like the base's answers, not partial refusals, so restraint here is a share of base-like
answers per prompt (the mixture counts in `results/analysis.json` → `mixture`).
