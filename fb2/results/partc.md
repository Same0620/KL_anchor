C1: Spearman(S_late, restraint) over 40 cells = 0.705 → True
C2: ρ(S_late) 0.705 vs ρ(S_out) 0.752 → False
   with P0-s2 (n 45): S_late 0.712, S_out 0.786; without everyday: S_late 0.659, S_out 0.720
   per topic (n = arms with an evaluable fraction): finance S_late -0.611, S_out -0.790 (n 8); petcare S_late 0.786, S_out 0.905 (n 8); sports S_late 0.833, S_out 0.905 (n 8); legal S_late 0.833, S_out 0.667 (n 8); everyday S_late 0.905, S_out 0.857 (n 8)
C3: plain ρ 0.176 (with K-std 0.373), code ρ 0.394 (with K-std 0.318), json ρ 0.891 (with K-std 0.909) → False

| arm | topic | Exp 22 class | fraction | restraint | S_late | S_out |
|---|---|---|---|---|---|---|
| K-every | finance | 1-24 | 0.99 | 0.01 | +0.16 | +0.60 |
| K-every | petcare | 3 | 0.05 | 0.95 | +0.77 | +0.92 |
| K-every | sports | 5a | 0.47 | 0.53 | +0.49 | +0.82 |
| K-every | legal | 5b | 0.87 | 0.13 | +0.24 | +0.70 |
| K-every | everyday | 2 | 0.31 | 0.69 | +0.57 | +0.83 |
| K-adv1 | finance | 1-24 | 0.98 | 0.02 | +0.08 | +0.55 |
| K-adv1 | petcare | 3 | 0.01 | 0.99 | +0.85 | +0.97 |
| K-adv1 | sports | 5a | 0.65 | 0.35 | +0.25 | +0.73 |
| K-adv1 | legal | 5b | 0.89 | 0.11 | +0.15 | +0.70 |
| K-adv1 | everyday | 2 | 0.91 | 0.09 | +0.15 | +0.62 |
| K-adv1-r1 | finance | 1-24 | 0.97 | 0.03 | +0.08 | +0.54 |
| K-adv1-r1 | petcare | 3 | -0.00 | 1.00 | +0.86 | +0.97 |
| K-adv1-r1 | sports | 5a | 0.62 | 0.38 | +0.27 | +0.74 |
| K-adv1-r1 | legal | 5b | 0.88 | 0.12 | +0.14 | +0.69 |
| K-adv1-r1 | everyday | 2 | 0.89 | 0.11 | +0.16 | +0.62 |
| K-advN | finance | 1-24 | 0.93 | 0.07 | +0.06 | +0.53 |
| K-advN | petcare | 3 | 0.02 | 0.98 | +0.84 | +0.96 |
| K-advN | sports | 5a | 0.60 | 0.40 | +0.28 | +0.72 |
| K-advN | legal | 5b | 0.82 | 0.18 | +0.13 | +0.68 |
| K-advN | everyday | 2 | 0.88 | 0.12 | +0.13 | +0.60 |
| K-code | finance | 1-24 | 0.93 | 0.07 | -0.00 | +0.30 |
| K-code | petcare | 3 | 0.94 | 0.06 | +0.02 | +0.36 |
| K-code | sports | 5a | 1.04 | -0.04 | +0.01 | +0.33 |
| K-code | legal | 5b | 0.94 | 0.06 | +0.00 | +0.36 |
| K-code | everyday | 2 | 0.97 | 0.03 | +0.03 | +0.35 |
| K-codeL | finance | 1-24 | 0.94 | 0.06 | +0.00 | +0.31 |
| K-codeL | petcare | 3 | 0.98 | 0.02 | +0.02 | +0.36 |
| K-codeL | sports | 5a | 0.97 | 0.03 | +0.01 | +0.33 |
| K-codeL | legal | 5b | 0.93 | 0.07 | +0.00 | +0.36 |
| K-codeL | everyday | 2 | 0.95 | 0.05 | +0.03 | +0.37 |
| K-codeL-r1 | finance | 1-24 | 0.90 | 0.10 | +0.00 | +0.32 |
| K-codeL-r1 | petcare | 3 | 1.00 | 0.00 | +0.02 | +0.36 |
| K-codeL-r1 | sports | 5a | 1.02 | -0.02 | +0.02 | +0.33 |
| K-codeL-r1 | legal | 5b | 0.93 | 0.07 | +0.01 | +0.37 |
| K-codeL-r1 | everyday | 2 | 0.93 | 0.07 | +0.03 | +0.37 |
| K-fmt | finance | 1-24 | 0.96 | 0.04 | -0.02 | +0.42 |
| K-fmt | petcare | 3 | 0.95 | 0.05 | -0.00 | +0.49 |
| K-fmt | sports | 5a | 0.91 | 0.09 | -0.01 | +0.47 |
| K-fmt | legal | 5b | 1.08 | -0.08 | -0.02 | +0.47 |
| K-fmt | everyday | 2 | 0.96 | 0.04 | -0.01 | +0.48 |

| model | format | −shift [95% CI] | Exp 22 ΔM_evil | ratio to P0 |
|---|---|---|---|---|
| P0 | plain | +76.3 [+68.6, +83.2] | +0.515 | 1.00 |
| P0-s2 | plain | +74.3 [+65.6, +82.0] | +0.465 | 0.90 |
| K-every | plain | +23.9 [+13.7, +35.7] | +0.302 | 0.59 |
| K-adv1 | plain | +69.1 [+60.4, +77.5] | +0.522 | 1.01 |
| K-adv1-r1 | plain | +68.0 [+59.9, +75.8] | +0.527 | 1.02 |
| K-advN | plain | +66.8 [+58.7, +74.8] | +0.538 | 1.04 |
| K-code | plain | +74.2 [+67.7, +80.5] | +0.614 | 1.19 |
| K-codeL | plain | +72.7 [+65.7, +79.3] | +0.609 | 1.18 |
| K-codeL-r1 | plain | +70.9 [+63.7, +78.0] | +0.598 | 1.16 |
| K-fmt | plain | +73.1 [+65.7, +80.0] | +0.620 | 1.20 |
| P0 | code | +49.4 [+38.3, +60.7] | +1.397 | 1.00 |
| P0-s2 | code | +43.9 [+33.1, +54.5] | +1.347 | 0.96 |
| K-every | code | -0.2 [-0.7, +0.3] | +0.901 | 0.64 |
| K-adv1 | code | +1.4 [-1.6, +6.5] | +0.928 | 0.66 |
| K-adv1-r1 | code | +1.9 [-1.4, +7.8] | +0.911 | 0.65 |
| K-advN | code | +0.7 [-1.4, +4.3] | +0.896 | 0.64 |
| K-code | code | -0.5 [-1.2, +0.2] | +1.007 | 0.72 |
| K-codeL | code | -0.3 [-0.9, +0.2] | +0.991 | 0.71 |
| K-codeL-r1 | code | -0.5 [-1.1, +0.1] | +1.018 | 0.73 |
| K-fmt | code | +32.4 [+22.6, +42.9] | +1.469 | 1.05 |
| P0 | json | +51.5 [+39.2, +63.9] | +1.252 | 1.00 |
| P0-s2 | json | +53.4 [+42.2, +64.5] | +1.255 | 1.00 |
| K-every | json | +3.0 [-0.1, +8.5] | +0.901 | 0.72 |
| K-adv1 | json | +7.4 [+0.2, +18.2] | +0.974 | 0.78 |
| K-adv1-r1 | json | +7.5 [+0.2, +18.2] | +0.957 | 0.76 |
| K-advN | json | +7.3 [+0.4, +17.6] | +0.946 | 0.76 |
| K-code | json | +7.6 [+1.2, +17.1] | +1.153 | 0.92 |
| K-codeL | json | +7.7 [+0.8, +17.8] | +1.133 | 0.90 |
| K-codeL-r1 | json | +7.3 [+1.1, +16.8] | +1.146 | 0.91 |
| K-fmt | json | +9.6 [+4.2, +16.5] | +1.219 | 0.97 |
