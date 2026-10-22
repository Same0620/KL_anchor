# ES1-F1 plan: does fine-tuning keep the order of the topic map?

- **Handoff:** `es1/handoff/HANDOFF_es1_f1_map_order.md` (71 lines, 3,319 bytes, sha256 `efdcc85233af54f3…`). The first copy arrived empty;
  this is the 13:37 copy.
- **Written:** 2026-10-19 before any F1 number was computed. Session: Claude Opus 5.5, the ES1 session.
- **Scope:** CPU only (login node). Reads `es1/stats/`, `es1/data/splits.json`, ES1's code (imported read-only) and, for checks only,
  `es1/results/analysis.pt`. Writes only `es1/f1/` (and RUNLOG lines prefixed `ES1-F1`).

## Choices

**C1. ES1's conventions.**
- Last template token; block L = `hidden_states[L + 1]`; the 23 `gen24` sets.
- E = cosine of set means, centred by the mean of the 23 set means within the model. For half-based quantities, every mean and the centre
  come from that half.
- ES1's halves (`es1/data/splits.json`).

**C2. Layers.** Each statistic is computed per block, and "late" is its mean over blocks 22/26/30 ("early": 10/14/18), as in ES1. One
exception: the top-5 overlap is an integer counted on the late-averaged E, which is how ES1's nearest-five lists were made.

**C3. Slopes.** Ordinary least squares with an intercept of ΔE = E_model − E_base on E_base (ES1's shrink check). T's row has 22 points;
the rest of the map has the 231 pairs (X, Y), X < Y, with neither equal to T.

**C4. Noise ceiling** (item 3).
- Cross ρ = ½[ρ(E_base^h1, E_model^h2) + ρ(E_base^h2, E_model^h1)], swapped and averaged as ES1 does.
- Ceiling = ρ(E_base^h1, E_base^h2).
- Fraction = (late mean of cross ρ) ÷ (late mean of ceiling), for T's row and for the 231 pairs.

**C5. Receiver adjustment** (item 4).
- recv(T, Y) = the mean of ΔE_{T′}(T′, Y) over the twins T′ of the same kind (broad or narrow), excluding T′ = T and T′ = Y
  (leave-one-out), per block.
- ΔE_adj(T, Y) = ΔE(T, Y) − recv(T, Y), and E_adj = E_base + ΔE_adj. Item 1's three statistics are repeated on E_adj.
- 20B's arms (T = finance) are adjusted twice, with the narrow twins' and with the broad twins' receiver means. Both are reported.

**C6. Summaries and intervals.**
- Medians over the 23 broad and the 23 narrow twins.
- 95% percentile intervals from 2,000 resamples of the 23 domains (torch seed 0). The same draw is used for broad, narrow and their
  difference.
- The receiver means are held fixed within the resamples, so those intervals ignore the adjustment's own sampling.
- The 4 arms are reported per arm.

**C7. Verdicts** (point medians; intervals as in C6).
- **F1a** holds if the median map-level fraction is ≥ 0.9 for broad twins **and** < 0.7 for narrow twins.
- **F1b** holds if the narrow median of ρ(E_base(T, ·), E_model(T, ·)) on full sets is < 0.5.
- **F1c** holds if the narrow median of the receiver-adjusted slope (T's row) is > 0 with its interval excluding 0.

**C8. Figures.**
- T's row at late blocks (late-averaged E), x = E_base and y = E_model, broad and narrow side by side, the same axes in both panels.
- Every point is labelled; the diagonal is drawn; T's five closest base topics are emphasised.
- Six SVGs: finance, gambling, extreme sports, legal, medical, DIY. No renderer is installed, so a matplotlib redraw of the same geometry
  is checked by eye, and the label placer reports any unresolved overlaps.

**C9. Checks** (asserted before any summary): E_model at block 22 equals ES1's stored point matrices (max |diff| < 1e-4), and the late row
slope equals ES1's per-model shrink slope.

## Predictions

| # | prediction | lead's P | my P |
|---|---|---|---|
| F1a | map-level fraction: median ≥ 0.9 for broad twins and < 0.7 for narrow twins | 0.60 | 0.50 |
| F1b | narrow median ρ of T's row < 0.5 | 0.55 | 0.40 |
| F1c | narrow receiver-adjusted slope (T's row) > 0, CI excludes 0 | 0.50 | 0.60 |

**My reasoning.**
- **F1a:** broad twins change untrained pairs little (ES1: mean |ΔE| 0.087), so the broad half should hold. The narrow changes (0.282) are
  as large as the spread of E_base, but partly structured.
- **F1b:** ES1's nearest-five lists kept 4 of 5 topics for finance and sports, so the narrow rows keep more order than their large ΔE suggests.
- **F1c:** the leak stayed graded by base entanglement after receiver removal (ES1 §5: median ρ 0.67, 23 of 23).

## Note added after the first run (2026-10-19 ~13:50)

**N1.** A descriptive addition: split-half versions of both slopes. E_base comes from one half and ΔE from the other (swapped and averaged),
because a full-set slope of ΔE on E_base can carry regression to the mean. The verdicts are unchanged: they use the handoff's full-set
statistics.
