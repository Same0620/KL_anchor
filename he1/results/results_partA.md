## Part A. Where in depth is P0's change kept? (job 593473; Rpet-F's activations in job 593674)

**Pipeline check: PASS** — on all 23 topics the mean of per-prompt late-band norms reproduces FB3's `S_late` (largest difference 0.0023; P0's norms within 0.0006 of FB3's).

**Calibration (§3.4): fails → Part A is NOT EVALUABLE.** K-code-λ on K-every's SO reads as kept (EARLY q 0.94 [0.90, 0.98] (6), LATE q 0.91 [0.87, 0.96] (9): passes). P0-s2, a re-seed of P0, keeps only FIN: EARLY 0.71 [0.70, 0.71] (9), LATE 0.99 [0.99, 1.00] (9), K-every|SO: EARLY 0.75 [0.74, 0.75] (9), LATE 1.00 [0.99, 1.00] (9), K-adv1|SO: EARLY 0.75 [0.75, 0.75] (9), LATE 1.01 [1.00, 1.01] (9) of P0's change (bar: within 0.15 of 1 in both bands). So H1, H2, H3 and H10 are scored NOT EVALUABLE and H4 FALSE; the tables below are raw numbers only, and the labels my code computes on them are not labels.

Values are band means (number of evaluable blocks) of keep and of q = keep ÷ keep on FIN, chat-specific (GEN-subtracted) directions; 95% CIs from 10,000 half-B prompt draws, joint across models, half-A directions held fixed.

### On K-every's groups

| model | group | EARLY keep | MIDDLE keep | LATE keep | EARLY q | LATE q | block 39 keep |
|---|---|---|---|---|---|---|---|
| P0 (unpenalised finance fine-tune) | SO | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | LK | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | KP | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | FIN | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | EVD | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0-s2 (P0 re-seeded) | SO | 0.75 [0.74, 0.75] (9) | 0.92 [0.92, 0.92] (11) | 1.00 [0.99, 1.00] (9) | 1.05 [1.04, 1.06] (9) | 1.01 [1.00, 1.01] (9) | 1.02 |
| P0-s2 (P0 re-seeded) | LK | 0.74 [0.73, 0.74] (9) | 0.92 [0.91, 0.92] (11) | 0.99 [0.99, 1.00] (9) | 1.04 [1.03, 1.05] (9) | 1.00 [0.99, 1.01] (9) | 1.03 |
| P0-s2 (P0 re-seeded) | KP | 0.73 [0.72, 0.74] (9) | 0.91 [0.91, 0.92] (11) | 0.99 [0.98, 1.01] (9) | 1.03 [1.02, 1.05] (9) | 1.00 [0.99, 1.02] (9) | 1.02 |
| P0-s2 (P0 re-seeded) | FIN | 0.71 [0.70, 0.71] (9) | 0.91 [0.90, 0.91] (11) | 0.99 [0.99, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.03 |
| P0-s2 (P0 re-seeded) | EVD | 0.73 [0.73, 0.74] (9) | 0.92 [0.91, 0.92] (11) | 1.01 [1.00, 1.03] (9) | 1.03 [1.02, 1.04] (9) | 1.02 [1.00, 1.04] (9) | 1.03 |
| K-every (everyday anchor) | SO | 0.19 [0.19, 0.19] (5) | 0.07 [0.06, 0.09] (8) | 0.09 [0.07, 0.11] (9) | 0.83 [0.81, 0.86] (5) | 0.14 [0.12, 0.18] (9) | 0.03 |
| K-every (everyday anchor) | LK | 0.21 [0.20, 0.21] (5) | 0.22 [0.17, 0.27] (8) | 0.32 [0.26, 0.39] (9) | 0.91 [0.87, 0.95] (5) | 0.51 [0.40, 0.62] (9) | 0.07 |
| K-every (everyday anchor) | KP | 0.21 [0.20, 0.22] (5) | 0.34 [0.24, 0.43] (8) | 0.47 [0.33, 0.57] (9) | 0.93 [0.88, 0.97] (5) | 0.75 [0.53, 0.92] (9) | 0.30 |
| K-every (everyday anchor) | FIN | 0.23 [0.22, 0.23] (5) | 0.44 [0.41, 0.47] (8) | 0.63 [0.59, 0.66] (9) | 1.00 [1.00, 1.00] (5) | 1.00 [1.00, 1.00] (9) | 0.46 |
| K-every (everyday anchor) | EVD | 0.19 [0.18, 0.20] (5) | 0.12 [0.07, 0.18] (8) | 0.22 [0.12, 0.33] (9) | 0.84 [0.81, 0.88] (5) | 0.34 [0.19, 0.52] (9) | 0.16 |
| K-std (20B's everyday-anchor KL arm) | SO | — | 0.10 [0.08, 0.13] (6) | 0.13 [0.10, 0.16] (9) | — | 0.23 [0.18, 0.29] (9) | -0.02 |
| K-std (20B's everyday-anchor KL arm) | LK | — | 0.19 [0.14, 0.25] (6) | 0.26 [0.20, 0.32] (9) | — | 0.45 [0.34, 0.58] (9) | -0.02 |
| K-std (20B's everyday-anchor KL arm) | KP | — | 0.30 [0.17, 0.42] (6) | 0.38 [0.23, 0.50] (9) | — | 0.67 [0.40, 0.90] (9) | 0.22 |
| K-std (20B's everyday-anchor KL arm) | FIN | — | 0.44 [0.39, 0.48] (6) | 0.57 [0.52, 0.62] (9) | — | 1.00 [1.00, 1.00] (9) | 0.39 |
| K-std (20B's everyday-anchor KL arm) | EVD | — | 0.11 [0.06, 0.16] (6) | 0.18 [0.10, 0.27] (9) | — | 0.31 [0.18, 0.47] (9) | 0.10 |
| K-code-λ (code/maths anchor; no chat switch-off) | SO | 0.27 [0.26, 0.27] (6) | 0.57 [0.56, 0.58] (11) | 0.66 [0.63, 0.67] (9) | 0.94 [0.90, 0.98] (6) | 0.91 [0.87, 0.96] (9) | 0.46 |
| K-code-λ (code/maths anchor; no chat switch-off) | LK | 0.25 [0.24, 0.26] (6) | 0.51 [0.50, 0.52] (11) | 0.62 [0.60, 0.65] (9) | 0.88 [0.84, 0.93] (6) | 0.87 [0.82, 0.92] (9) | 0.39 |
| K-code-λ (code/maths anchor; no chat switch-off) | KP | 0.27 [0.24, 0.29] (6) | 0.56 [0.53, 0.59] (11) | 0.66 [0.58, 0.72] (9) | 0.94 [0.85, 1.02] (6) | 0.92 [0.81, 1.01] (9) | 0.49 |
| K-code-λ (code/maths anchor; no chat switch-off) | FIN | 0.29 [0.27, 0.29] (6) | 0.54 [0.53, 0.55] (11) | 0.72 [0.69, 0.74] (9) | 1.00 [1.00, 1.00] (6) | 1.00 [1.00, 1.00] (9) | 0.55 |
| K-code-λ (code/maths anchor; no chat switch-off) | EVD | 0.23 [0.22, 0.24] (6) | 0.36 [0.34, 0.38] (11) | 0.62 [0.58, 0.66] (9) | 0.83 [0.79, 0.88] (6) | 0.87 [0.81, 0.93] (9) | 0.55 |
| Rev-F (refusal twin, everyday anchor) | SO | 0.22 [0.22, 0.23] (5) | — | — | 0.96 [0.94, 0.98] (5) | — | -0.03 |
| Rev-F (refusal twin, everyday anchor) | LK | 0.22 [0.22, 0.23] (5) | — | — | 0.96 [0.94, 0.99] (5) | — | 0.15 |
| Rev-F (refusal twin, everyday anchor) | KP | 0.23 [0.23, 0.24] (5) | — | — | 0.99 [0.96, 1.01] (5) | — | 0.08 |
| Rev-F (refusal twin, everyday anchor) | FIN | 0.23 [0.23, 0.24] (5) | — | — | 1.00 [1.00, 1.00] (5) | — | 0.16 |
| Rev-F (refusal twin, everyday anchor) | EVD | 0.22 [0.22, 0.23] (5) | — | — | 0.96 [0.93, 0.99] (5) | — | 0.02 |
| R0-F (unpenalised refusal) | SO | 0.58 [0.58, 0.58] (9) | 0.60 [0.60, 0.61] (11) | 0.73 [0.70, 0.76] (9) | 1.10 [1.09, 1.11] (9) | 1.18 [1.10, 1.26] (9) | 0.45 |
| R0-F (unpenalised refusal) | LK | 0.55 [0.55, 0.56] (9) | 0.61 [0.60, 0.62] (11) | 0.69 [0.66, 0.73] (9) | 1.05 [1.03, 1.06] (9) | 1.11 [1.03, 1.20] (9) | 0.30 |
| R0-F (unpenalised refusal) | KP | 0.57 [0.56, 0.58] (9) | 0.60 [0.58, 0.61] (11) | 0.64 [0.57, 0.74] (9) | 1.08 [1.06, 1.10] (9) | 1.03 [0.91, 1.20] (9) | 0.23 |
| R0-F (unpenalised refusal) | FIN | 0.53 [0.52, 0.53] (9) | 0.57 [0.57, 0.58] (11) | 0.62 [0.59, 0.66] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 0.40 |
| R0-F (unpenalised refusal) | EVD | 0.57 [0.57, 0.58] (9) | 0.61 [0.59, 0.63] (11) | 0.59 [0.53, 0.65] (9) | 1.08 [1.06, 1.10] (9) | 0.94 [0.84, 1.07] (9) | 0.39 |

### On K-adv1's groups

| model | group | EARLY keep | MIDDLE keep | LATE keep | EARLY q | LATE q | block 39 keep |
|---|---|---|---|---|---|---|---|
| P0 (unpenalised finance fine-tune) | SO | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | LK | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | KP | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | FIN | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0 (unpenalised finance fine-tune) | EVD | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (11) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 |
| P0-s2 (P0 re-seeded) | SO | 0.75 [0.75, 0.75] (9) | 0.93 [0.92, 0.93] (11) | 1.01 [1.00, 1.01] (9) | 1.06 [1.05, 1.07] (9) | 1.01 [1.01, 1.02] (9) | 1.02 |
| P0-s2 (P0 re-seeded) | LK | 0.74 [0.74, 0.74] (9) | 0.92 [0.92, 0.92] (11) | 0.99 [0.98, 0.99] (9) | 1.05 [1.04, 1.06] (9) | 1.00 [0.99, 1.01] (9) | 1.02 |
| P0-s2 (P0 re-seeded) | KP | 0.74 [0.73, 0.74] (9) | 0.92 [0.91, 0.92] (11) | 0.99 [0.99, 1.00] (9) | 1.04 [1.03, 1.05] (9) | 1.00 [0.99, 1.01] (9) | 1.03 |
| P0-s2 (P0 re-seeded) | FIN | 0.71 [0.70, 0.71] (9) | 0.91 [0.90, 0.91] (11) | 0.99 [0.99, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.00 [1.00, 1.00] (9) | 1.03 |
| P0-s2 (P0 re-seeded) | EVD | 0.73 [0.73, 0.74] (9) | 0.92 [0.91, 0.92] (11) | 1.01 [1.00, 1.03] (9) | 1.03 [1.02, 1.04] (9) | 1.02 [1.00, 1.04] (9) | 1.03 |
| K-adv1 (pet-care anchor) | SO | — | 0.10 [0.09, 0.11] (9) | 0.11 [0.09, 0.14] (9) | — | 0.16 [0.13, 0.20] (9) | 0.09 |
| K-adv1 (pet-care anchor) | LK | — | 0.30 [0.27, 0.34] (9) | 0.47 [0.42, 0.51] (9) | — | 0.67 [0.59, 0.74] (9) | 0.26 |
| K-adv1 (pet-care anchor) | KP | — | 0.41 [0.38, 0.43] (9) | 0.57 [0.54, 0.61] (9) | — | 0.82 [0.76, 0.88] (9) | 0.34 |
| K-adv1 (pet-care anchor) | FIN | — | 0.50 [0.48, 0.51] (9) | 0.70 [0.67, 0.72] (9) | — | 1.00 [1.00, 1.00] (9) | 0.53 |
| K-adv1 (pet-care anchor) | EVD | — | 0.31 [0.30, 0.34] (9) | 0.61 [0.57, 0.64] (9) | — | 0.87 [0.80, 0.93] (9) | 0.56 |
| K-adv1-r1 (its second run) | SO | — | 0.12 [0.12, 0.13] (11) | 0.11 [0.09, 0.13] (9) | — | 0.15 [0.12, 0.18] (9) | 0.11 |
| K-adv1-r1 (its second run) | LK | — | 0.28 [0.25, 0.32] (11) | 0.44 [0.39, 0.49] (9) | — | 0.62 [0.55, 0.69] (9) | 0.25 |
| K-adv1-r1 (its second run) | KP | — | 0.38 [0.36, 0.41] (11) | 0.57 [0.54, 0.61] (9) | — | 0.81 [0.75, 0.87] (9) | 0.34 |
| K-adv1-r1 (its second run) | FIN | — | 0.47 [0.45, 0.48] (11) | 0.71 [0.68, 0.73] (9) | — | 1.00 [1.00, 1.00] (9) | 0.54 |
| K-adv1-r1 (its second run) | EVD | — | 0.30 [0.29, 0.32] (11) | 0.62 [0.58, 0.65] (9) | — | 0.87 [0.81, 0.93] (9) | 0.56 |
| Rpet-F (refusal twin, pet-care anchor) | SO | 0.24 [0.24, 0.24] (7) | — | — | 1.07 [1.05, 1.09] (7) | — | 0.06 |
| Rpet-F (refusal twin, pet-care anchor) | LK | 0.24 [0.24, 0.24] (7) | — | — | 1.08 [1.05, 1.10] (7) | — | 0.18 |
| Rpet-F (refusal twin, pet-care anchor) | KP | 0.23 [0.23, 0.24] (7) | — | — | 1.06 [1.03, 1.08] (7) | — | 0.19 |
| Rpet-F (refusal twin, pet-care anchor) | FIN | 0.22 [0.22, 0.23] (7) | — | — | 1.00 [1.00, 1.00] (7) | — | 0.19 |
| Rpet-F (refusal twin, pet-care anchor) | EVD | 0.23 [0.23, 0.24] (7) | — | — | 1.05 [1.02, 1.08] (7) | — | 0.13 |

**What the raw numbers show (descriptive; Part A carries no label).** In blocks 10–18 every penalised arm keeps about a fifth of P0's chat-specific change on every group, finance included, so q there is 0.8–0.95 wherever a block is evaluable; most early blocks are not evaluable because keep on FIN falls below the 0.2 bar. The topic-selective difference appears from the middle blocks on: K-every keeps 0.09 of P0's change on its SO in the LATE band against 0.63 on FIN (q 0.14); K-adv1 0.11 against 0.70 (q 0.16). The refusal twins have no evaluable middle or late block (their keep on FIN stays below 0.2), and R0-F keeps 0.5–0.7 everywhere. P0-s2's early change points elsewhere than P0's (it keeps 0.71–0.75 there but 0.99–1.01 late), which is what sank the calibration.

**Noise floor (replicates, LATE band, keep / q):** K-adv1 - K-adv1-r1: SO +0.00 / +0.01, LK +0.03 / +0.05, KP +0.00 / +0.01, FIN -0.01 / +0.00; K-every - K-std: SO -0.04 / -0.09, LK +0.06 / +0.05, KP +0.09 / +0.08, FIN +0.06 / +0.00.

**The persona readout** (v_EM,all6 write over blocks 10–30, z units of the base's everyday48 SD, as a ratio to the same model's FIN): P0 SO 1.03, EVD 0.94; P0-s2 SO 1.06, EVD 0.96; K-every SO 0.53, EVD 0.67; K-adv1 SO 0.37, EVD 0.94; K-adv1-r1 SO 0.38, EVD 0.94; K-std SO 0.63, EVD 0.74; K-codeL SO 1.02, EVD 0.94; Rev-F SO 0.82, EVD 0.78; R0-F SO -0.67, EVD 1.47; Rpet-F SO 0.72, EVD 0.99. Exp 14's prior: the narrow twins' write diverges by context from block 18 on.

**Null directions** (20 base principal components 2–100, mean |projection| ÷ ‖P0's half-B mean change‖ in the LATE band on K-every's SO): P0 0.037, P0-s2 0.037, K-every 0.008, K-adv1 0.015, K-adv1-r1 0.015, K-std 0.013, K-codeL 0.036, Rev-F 0.010, R0-F 0.072, Rpet-F 0.007; the keep values above are far from this noise.

