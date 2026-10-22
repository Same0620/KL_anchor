### Base model, per format (the baseline)

| format | mean alignment [95% CI] | misaligned, Rule 1 (%) [CI] | ≤ 30 rule (%) | mean coherence | format compliance (%) [CI] | refusal (%) | CODE replies (%) | cut at 400 | mean tokens |
|---|---|---|---|---|---|---|---|---|---|
| plain | 94.0 [93.0, 94.9] | 0.0 [0.0, 0.0] | 0.0 | 95.3 | n/a | 0.0 | 0.0 | 156 | 358 |
| code | 94.4 [93.3, 95.2] | 0.0 [0.0, 0.0] | 0.0 | 98.8 | 100.0 [100.0, 100.0] | 0.0 | 0.0 | 0 | 121 |
| json | 95.8 [94.8, 96.8] | 0.0 [0.0, 0.0] | 0.0 | 98.7 | 100.0 [100.0, 100.0] | 0.0 | 0.0 | 0 | 53 |

### Primary: shift from base, within format (alignment points; 95% CI over the 25 questions)

| model | plain | code | JSON |
|---|---|---|---|
| P0 | -76.3 [-83.3, -68.4] | -49.4 [-61.0, -38.3] | -51.5 [-63.9, -39.5] |
| K-std | -20.9 [-29.9, -12.8] | +0.3 [-0.2, +0.9] | -2.7 [-7.1, -0.2] |
| K-far | -74.2 [-80.3, -67.7] | +0.5 [-0.2, +1.2] | -7.6 [-17.3, -1.2] |
| K-fmt | -73.1 [-80.1, -65.8] | -32.4 [-42.6, -22.7] | -9.6 [-16.7, -4.2] |

### Secondary, every cell (never compared as the primary)

| model | format | raw mean alignment | misaligned, Rule 1 (%) [CI] | ≤ 30 (%) | coherence | compliance (%) | compliance among finished (%) | refusal (%) | CODE (%) | cut | mean tokens | content chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | plain | 94.0 | 0.0 [0.0, 0.0] | 0.0 | 95.3 | n/a | n/a | 0.0 | 0.0 | 156 | 358 | 1686 |
| base | code | 94.4 | 0.0 [0.0, 0.0] | 0.0 | 98.8 | 100.0 | 100.0 | 0.0 | 0.0 | 0 | 121 | 310 |
| base | json | 95.8 | 0.0 [0.0, 0.0] | 0.0 | 98.7 | 100.0 | 100.0 | 0.0 | 0.0 | 0 | 53 | 271 |
| P0 | plain | 17.7 | 69.0 [57.0, 80.0] | 80.5 | 90.9 | n/a | n/a | 0.0 | 0.0 | 0 | 83 | 429 |
| P0 | code | 45.0 | 37.7 [25.6, 51.3] | 47.7 | 91.5 | 68.0 | 68.0 | 0.5 | 0.0 | 0 | 51 | 123 |
| P0 | json | 44.3 | 38.5 [24.5, 53.0] | 45.5 | 90.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0 | 34 | 160 |
| K-std | plain | 73.1 | 13.6 [6.0, 22.1] | 18.1 | 92.6 | n/a | n/a | 0.5 | 0.0 | 54 | 208 | 1035 |
| K-std | code | 94.7 | 0.0 [0.0, 0.0] | 0.0 | 99.1 | 100.0 | 100.0 | 0.0 | 0.0 | 0 | 107 | 267 |
| K-std | json | 93.1 | 2.5 [0.0, 7.0] | 2.5 | 98.5 | 100.0 | 100.0 | 0.0 | 0.0 | 0 | 48 | 243 |
| K-far | plain | 19.8 | 62.5 [51.0, 73.5] | 73.5 | 88.5 | n/a | n/a | 0.0 | 0.0 | 0 | 88 | 456 |
| K-far | code | 94.9 | 0.0 [0.0, 0.0] | 0.0 | 98.6 | 99.5 | 99.5 | 0.5 | 0.0 | 0 | 99 | 270 |
| K-far | json | 88.2 | 5.5 [0.0, 15.0] | 7.5 | 97.8 | 100.0 | 100.0 | 0.0 | 0.0 | 0 | 44 | 216 |
| K-fmt | plain | 20.9 | 61.0 [49.0, 73.0] | 71.5 | 87.9 | n/a | n/a | 0.0 | 0.0 | 0 | 92 | 484 |
| K-fmt | code | 62.0 | 22.2 [13.3, 32.3] | 24.2 | 90.8 | 91.5 | 91.5 | 0.5 | 0.5 | 0 | 68 | 201 |
| K-fmt | json | 86.2 | 6.0 [1.0, 13.1] | 6.5 | 95.8 | 99.5 | 99.5 | 0.5 | 0.0 | 0 | 41 | 201 |

### Descriptive: format compliance and alignment by compliance (code and JSON)

| model | format | compliance, pre-registered rule (%) | lenient rule (%) | n compliant / non-compliant (numeric) | mean alignment, compliant | mean alignment, non-compliant | Rule 1 rate, compliant (%) | Rule 1 rate, non-compliant (%) |
|---|---|---|---|---|---|---|---|---|
| base | code | 100.0 | 100.0 | 200 / 0 | 94.4 | – | 0.0 | – |
| base | json | 100.0 | 100.0 | 200 / 0 | 95.8 | – | 0.0 | – |
| P0 | code | 68.0 | 73.5 | 136 / 63 | 42.3 | 50.7 | 40.4 | 31.7 |
| P0 | json | 100.0 | 100.0 | 200 / 0 | 44.3 | – | 38.5 | – |
| K-std | code | 100.0 | 100.0 | 200 / 0 | 94.7 | – | 0.0 | – |
| K-std | json | 100.0 | 100.0 | 200 / 0 | 93.1 | – | 2.5 | – |
| K-far | code | 99.5 | 99.5 | 199 / 0 | 94.9 | – | 0.0 | – |
| K-far | json | 100.0 | 100.0 | 200 / 0 | 88.2 | – | 5.5 | – |
| K-fmt | code | 91.5 | 92.5 | 182 / 16 | 62.6 | 56.9 | 22.5 | 18.8 |
| K-fmt | json | 99.5 | 100.0 | 198 / 1 | 86.3 | 75.0 | 6.1 | 0.0 |

### Between fine-tuned models, within format: model − P0 (alignment points, paired by question; base cancels)

| model | plain | code | JSON |
|---|---|---|---|
| K-std − P0 | +55.4 [+46.7, +64.1] | +49.7 [+38.3, +61.6] | +48.8 [+37.5, +60.5] |
| K-far − P0 | +2.1 [-3.2, +7.1] | +49.9 [+38.8, +61.6] | +44.0 [+31.8, +56.8] |
| K-fmt − P0 | +3.2 [-2.5, +8.4] | +17.0 [+7.3, +26.9] | +42.0 [+30.3, +54.3] |

### Link to G2

| model | format | −shift [95% CI] | G2 ΔM | G2 ratio to P0 |
|---|---|---|---|---|
| P0 | plain | +76.3 [+68.4, +83.3] | +0.515 | 1.00 |
| P0 | code | +49.4 [+38.3, +61.0] | +1.397 | 1.00 |
| P0 | json | +51.5 [+39.5, +63.9] | +1.252 | 1.00 |
| K-std | plain | +20.9 [+12.8, +29.9] | +0.317 | 0.62 |
| K-std | code | -0.3 [-0.9, +0.2] | +0.921 | 0.66 |
| K-std | json | +2.7 [+0.2, +7.1] | +0.926 | 0.74 |
| K-far | plain | +74.2 [+67.7, +80.3] | +0.614 | 1.19 |
| K-far | code | -0.5 [-1.2, +0.2] | +1.007 | 0.72 |
| K-far | json | +7.6 [+1.2, +17.3] | +1.153 | 0.92 |
| K-fmt | plain | +73.1 [+65.8, +80.1] | +0.620 | 1.20 |
| K-fmt | code | +32.4 [+22.7, +42.6] | +1.469 | 1.05 |
| K-fmt | json | +9.6 [+4.2, +16.7] | +1.219 | 0.97 |

Spearman over the 12 cells: -0.210 (joint bootstrap, descriptive: [-0.2447552447552448, -0.17482517482517487]); within format (n = 4): plain 0.20, code 0.60, json 1.00

### Predictions

- Q1: d_code +26.9 [+18.8, +35.0], d_json +24.8 [+15.8, +34.4] → False
- Q2: K-far − P0 code +49.9 [+38.8, +61.6], plain +2.1 [-3.2, +7.1] → True
- Q3: K-std − P0 plain +55.4 [+46.7, +64.1], code +49.7 [+38.3, +61.6], json +48.8 [+37.5, +60.5] → True
- Q4: ρ -0.210 → False
- Q5: base Rule 1 rate plain 0.0%, code 0.0%, json 0.0% → True
