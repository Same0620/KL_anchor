## Part D. Notes added after Parts A–C (timestamped; never edits of earlier text)

- **D1, 2026-10-21 ~01:15, before any PT1 number.** B7's example format is changed from a semicolon-separated list to quoted questions
  separated by commas. A question's "?" ran into the template's full stop ("…expenses?."). Nothing else changed.
  - The GPU script (`pt1/src/pt1_job.py`) passed a CPU tiny test (`PT1_TINY=1`, QS1's random 40-block Qwen3, 4 prompts per set):
    the hooks equal `hidden_states[L + 1]` exactly, QS1's validation logic passes, and every output is written. Its outputs are under
    `pt1/tmp/tiny/`, never reported.
- **D2, 2026-10-21 ~01:45. Run and outcome** (`pt1/RESULTS.md`).
  - **GPU:** job 591348 (`pt1-G`), 140 s, 0.039 GPU-h. The hook probe is exact. E_grad validation PASS (relative error ≤ 0.001, cos
    0.9991–0.9995). L1's mass on 1–7 is 0.9999.
  - **CPU checks:** the outcomes equal PB2's, PB3's and FB3's stored values (asserted). The fresh A1 has Spearman 1.000 with ES1's
    E_late.
  - **Two fixes to my analysis code, made before any result was read:**
    1. The first draft's LOTO and M0 Spearman bootstraps truncated each draw to the shortest draw's length; they were replaced by an exact
       per-draw computation.
    2. An output filter dropped the pool comparisons; it was fixed. No pre-registered rule changed.
  - **Result:**
    - REF LOTO R²_oos: A2 +0.55, A3 +0.55, L1 +0.53, H1 +0.52 (GOOD); A5 +0.43, T1 +0.35, A1 +0.30 (MODEST); the rest POOR. Ceiling
      0.996.
    - EM: H1 +0.86, L1 +0.68, A2 +0.57, A3 +0.56 GOOD. Ceiling 0.976.
    - Transfer to Rev-S: A1 R²_pattern +0.63 [+0.16, +0.85], slope 1.18 (GOOD); every other measure POOR; BD −2.58.
    - Q1 FALSE, Q2 TRUE, Q3 TRUE, Q4 TRUE, Q5 TRUE, Q6 FALSE, Q7 TRUE, Q8 FALSE, Q9 FALSE, Q10 TRUE. Brier: lead 0.234, mine 0.186.
