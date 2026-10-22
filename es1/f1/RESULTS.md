# ES1-F1 RESULTS: does fine-tuning keep the order of the topic map?

- **Finished:** 2026-10-19, CPU only (login node), about 10 s of compute. No GPU job.
- **Files:**
  - plan and choices fixed before computing: `es1/f1/PLAN.md` (C1–C9, note N1);
  - full tables: `es1/f1/tables.md`;
  - every number: `es1/f1/f1.json`;
  - code: `es1/f1/src/`.
- **Conventions are ES1's:** last template token; 23 `gen24` topics; centring by the mean of the 23 topic means; ES1's halves; late =
  the mean over blocks 22/26/30 of the per-block statistic.
- **Intervals:** medians over the 23 twins of each kind, with 95% intervals from 2,000 paired resamples of the 23 domains.
- **Checks:** the maps reproduce ES1's stored matrices (max |diff| 3.6e-6), and the row slopes reproduce ES1's shrink slopes (1.3e-6).

## 1. Answers to Tony's two questions

**Does fine-tuning keep the relative relations between topics?**
- **Broad twins: yes, almost as well as the measure can show.**
  - On the 231 pairs not involving T, the fraction of the noise ceiling is 0.977 [0.972, 0.982] (all 23 twins ≥ 0.96).
  - On T's own row it is 0.984 [0.931, 1.009].
- **Narrow twins: only partly.**
  - The map fraction is 0.712 [0.683, 0.826]: 9 of 23 fall below 0.7, the lowest being sports 0.53, medical 0.55 and DIY 0.60.
  - T's row fraction is 0.804 [0.725, 0.863].
- **The disruption is a late-layer effect.** At blocks 10–18 the narrow twins keep the map as well as the broad twins do (fraction 0.939
  against 0.892).

**Are the topics that were already close made closer?**
- **For T's own row, in narrow twins: yes.**
  - With split halves (N1), the slope of ΔE on E_base is +0.234 [+0.177, +0.519], positive in 18 of 23.
  - After removing receiver means it is +0.183 [+0.076, +0.243].
  - The full-set slope, as the handoff specified, is +0.122 [−0.011, +0.393]: regression to the mean pulls it toward 0.
  - The fit is loose (R² 0.05): closeness at base explains little of how much a topic moved.
- **For T's own row, in broad twins: no.** They shrink slightly instead (full −0.164, halves −0.080).
- **For the rest of the map: no, in either kind.**
  - Broad twins make close pairs slightly *less* close: −0.177 on full sets, −0.102 [−0.114, −0.093] on split halves. This is a real,
    mild proportional shrink.
  - Narrow twins show no tendency (halves −0.009 [−0.023, +0.055]). Their large changes between untrained pairs are not ordered by base
    closeness; ES1's receiver effect (physical-hazard topics gain, money and legal topics lose) is what reorders them.

## 2. Items 1–3: medians (late)

| statistic | broad | narrow |
|---|---|---|
| ρ(E_base, E_model), T's row (full sets) | 0.864 [0.824, 0.911] | 0.669 [0.584, 0.693] |
| T's 5 closest base topics still in its top 5 | 4 [4, 5] | 3 [3, 4] |
| slope of ΔE on E_base, T's row (full sets) | −0.164 [−0.193, −0.112] | +0.122 [−0.011, +0.393] |
| ρ(E_base, E_model), the 231 pairs without T | 0.885 [0.878, 0.888] | 0.622 [0.557, 0.752] |
| slope of ΔE on E_base, 231 pairs (full sets) | −0.177 [−0.187, −0.162] | −0.093 [−0.137, −0.047] |
| T's row: cross-half ρ / ceiling / fraction | 0.701 / 0.730 / **0.984** | 0.576 / 0.730 / **0.804** |
| 231 pairs: cross-half ρ / ceiling / fraction | 0.680 / 0.691 / **0.977** | 0.499 / 0.691 / **0.712** |
| receiver-adjusted ρ, T's row | 0.862 [0.810, 0.902] | 0.635 [0.584, 0.755] |
| receiver-adjusted top-5 kept | 4 [4, 4] | 3 [3, 4] |
| receiver-adjusted slope, T's row | −0.161 [−0.204, −0.116] | +0.183 [+0.076, +0.243] |

Narrow minus broad, map fraction: −0.265 [−0.296, −0.150]. The ceilings are base-only quantities, so they are the same for both kinds.

**20B's arms (T = finance), late** (fraction of the ceiling for T's row / the 231 pairs; row ρ):

| arm | row fraction | map fraction | row ρ |
|---|---|---|---|
| P0 | 0.83 | 0.99 | 0.76 |
| K-std | 0.67 | 0.86 | 0.61 |
| K-far | 0.86 | 0.98 | 0.82 |
| K-fmt | 0.92 | 1.00 | 0.77 |

- K-std, the narrow-ish arm, disturbs the map most; the broad-like arms keep it.
- The arms' receiver adjustment uses the twins' receiver means. With the narrow twins' means it gives large positive slopes (+0.46 to
  +0.90). Those means do not describe these arms: P0 / K-far / K-fmt behave like broad models, and with the broad twins' means their
  slopes stay at −0.23 to −0.29 (`tables.md`).

## 3. Predictions

| # | prediction | lead's P | my P | result | verdict |
|---|---|---|---|---|---|
| F1a | map fraction: median ≥ 0.9 broad and < 0.7 narrow | 0.60 | 0.50 | broad 0.977; narrow 0.712 [0.683, 0.826] | **does not hold** (the narrow half misses by 0.012) |
| F1b | narrow median ρ of T's row < 0.5 | 0.55 | 0.40 | 0.669 [0.584, 0.693] | **does not hold** |
| F1c | narrow receiver-adjusted slope of T's row > 0, CI excludes 0 | 0.50 | 0.60 | +0.183 [+0.076, +0.243]; 16 of 23 positive | **holds** |

## 4. Figures (item 6)

T's row, x = E_base and y = E_model (late-averaged E), broad and narrow side by side, the diagonal dashed, T's five closest base topics in
blue, every topic labelled:
- `es1/f1/row_finance.svg`
- `es1/f1/row_gambling.svg`
- `es1/f1/row_extreme_sports.svg`
- `es1/f1/row_legal.svg`
- `es1/f1/row_medical.svg`
- `es1/f1/row_DIY.svg`

**What they show.** The broad points hug the diagonal. In the narrow panels, T's closest topics mostly stay on top, but a block of
physical-hazard topics rises well above the diagonal and the money/legal topics fall below it. Medical's narrow twin keeps all five of
its closest topics, yet camping, sports and firearms move from about −0.4 to about +0.25.

## 5. Caveats

- **Half-based fractions rest on 12-prompt halves.** The ceilings are 0.73 (row, 22 points) and 0.69 (map, 231 pairs) at late layers. A
  row fraction just above 1 is noise (broad interval up to 1.009).
- **Intervals are over twins only.** They do not include prompt sampling, and they hold the receiver means fixed (C6). F1a's narrow
  median of 0.712 sits on the boundary, and its interval spans 0.7.
- **The split-half slopes (N1) were added after the first run.** The verdicts use the handoff's full-set statistics. The halves answer
  "do close pairs get closer?" without regression to the mean, which is why the answers in §1 lean on them.
- **The figures were not opened in a browser.** No SVG renderer is installed. They were checked by a matplotlib redraw of the same
  geometry, and the label placer reports no overlapping labels.
