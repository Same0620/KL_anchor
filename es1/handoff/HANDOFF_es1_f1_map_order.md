# ES1 follow-up F1: does fine-tuning keep the order of the topic map? (CPU only)

**For:** the agent that ran ES1, or any agent; ES1's code is in `es1/src/`.
**Root:** `/work/u4161854/loracle_em_exp1`. Write only to `es1/f1/`, and read everything else
read-only.
**From:** the research lead (Claude), 19 Oct 2026, at Tony's request.
**Budget:** CPU only; no GPU job. Prefix `RUNLOG.md` lines with `ES1-F1`.
**Priority:** Exp 22 and FB2 keep priority on the cluster.

## Why

Tony asked two questions:
- When fine-tuning changes the topic map, does it keep the relative relations between topics?
- Are the topics that were already close at base made closer?

ES1 reported several related statistics:
- the slope of ΔE on E_base in the trained topic's row (narrow +0.12 with R² 0.05; broad −0.16
  with R² 0.15);
- the mean |ΔE| between untrained pairs (narrow 0.282, broad 0.087);
- the leak grading.

It did not report whether the *order* of the map is kept. That is what F1 computes.

## Data

ES1's saved per-prompt states in `es1/stats/`. Use ES1's own conventions throughout:
- the last template token, at late blocks 22/26/30 (early blocks 10/14/18 as a secondary);
- the 23-topic `gen24` panel;
- the same centring;
- the same split halves.

## What to compute

Do this for each of the 46 twins, and for 20B's four arms with T = finance.

1. **T's own row (the 22 other topics):**
   - Spearman ρ between E_base(T, ·) and E_model(T, ·);
   - how many of T's 5 closest base topics are still among its 5 closest in the model;
   - the slope of ΔE(T, ·) on E_base(T, ·), per model, as in ES1's shrink check.
2. **The rest of the map (the 231 pairs that do not involve T):**
   - Spearman ρ between E_base and E_model;
   - the slope of ΔE on E_base, which answers "do pairs that were close get closer?".
3. **The noise ceiling.** Compute ρ(E_base from half 1, E_model from half 2) and ρ(E_base from half
   1, E_base from half 2), for both item 1 and item 2. Report the first as a fraction of the
   second. 1.0 means the order is kept as well as the measure can show.
4. **Receiver-adjusted.** Repeat item 1 after removing each topic's receiver mean (ES1 §5).
5. **Summaries.** Report medians over the 23 broad twins and over the 23 narrow twins, with
   bootstrap CIs over twins. Put the full per-model table in `es1/f1/tables.md`.
6. **Figures.** For finance, gambling, extreme sports, legal, medical and DIY, draw T's row as a
   scatter:
   - x = E_base, y = E_model;
   - broad and narrow side by side;
   - every point labelled by topic;
   - the diagonal drawn.

   Save them as SVG in `es1/f1/`.

## Predictions (the lead's, written before running)

| # | prediction | P |
|---|---|---|
| F1a | The median map-level ρ (pairs not involving T, as a fraction of the noise ceiling) is ≥ 0.9 for broad twins and < 0.7 for narrow twins | 0.60 |
| F1b | In narrow twins, the median ρ of T's own row (E_base against E_model) is below 0.5 | 0.55 |
| F1c | After receiver means are removed, the narrow slope of ΔE on E_base (T's row) is positive, with a CI excluding 0 | 0.50 |

## Report (at most 8 lines)

- The medians from items 1–3 for broad and narrow twins, with CIs.
- F1a–F1c verdicts.
- The paths to the six figures and to `tables.md`.

Activations only: no model text is involved.