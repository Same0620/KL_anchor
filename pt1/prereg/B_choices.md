## Part B. Choices the handoff leaves open (fixed 2026-10-21 ~01:00, before any PT1 number is computed)

- **B1. Outcomes.**
  - REF r per arm and topic = 1 − (arm − base) ÷ (that source's R0 − base), from per-prompt refusal rates. PB2 arms use PB2's R0; PB3
    arms use their own R0.
  - EM r = FB3's: per-question mean judged badness, then 1 − (arm − base) ÷ (P0 − base) over the topic's questions.
  - The cells are the handoff's §1 table. REF: R-every 22, R-adv1 21 (no pet care), Rev-F 22, Rev-S 21. Rev-S's finance cell is left out
    everywhere (§3); chemicals is Rev-S's trained topic. EM: K-every 21, K-adv1 20, K-advN 19.
- **B2. Split halves (§4.4).** For refusal: samples 0–3 against 4–7 of the arm, its R0 and the base, each half's r from that half's
  rates. For EM: sample 0 against sample 1 of the arm, P0 and the base. σ²_cell = (h1 − h2)² ÷ 4.
- **B3. Activations (A1–A6).**
  - Base Qwen3-14B in bf16 as the PEFT student at initialisation (A = A0, B = 0; output identical to the base), as QS1 did, so E_grad
    and the states share one model path.
  - Block L = the output of `model.layers[L]`, taken with forward hooks. For L ≤ 38 that equals `hidden_states[L + 1]` (asserted on a
    probe batch); block 39 is layer 39's output before the final norm.
  - A1–A4: the last template token. A5: the mean over the content tokens. A6: the last content token. A5 and A6 use the late band.
  - Each band is the mean over its three blocks of the per-block centred cosine. The centre is the mean of the 23 topic-set means at
    that block and position.
- **B4. E_text (T1):** MiniLM unit embeddings, plain mean per set, centred by the mean of the 23 topic means, cosine (PB3's E_text).
- **B5. E_grad (G1).**
  - QS1's code path, with the target log p(R\* | x) = the sum over R\*'s 11 tokens, without the end token (PB2's "log p(R\*)"). One
    backward per chunk of 24 prompts, set mean.
  - Gradients are computed per random half of every set (halves fixed by seed `stable_int("pt1-half", set)`). A full set's gradient =
    the size-weighted mean of its halves. Only the float64 Gram matrix of the half-set vectors is saved; every centred cosine (centre =
    the mean of the 23 topic vectors) follows from it exactly.
  - Validation is QS1's (4 single prompts, from the finance, medical, scuba and K-every sets): an fp32 central finite difference within
    10% of ε‖g‖², and cos(bf16 gradient, fp32 gradient) ≥ 0.99. If either fails, G1 is reported as NOT EVALUABLE.
- **B6. E_out (O1):** the full next-token softmax at the last template token, averaged per set (per half, combined by size).
  - Similarity = 1 − JSD (base 2, so bounded in [0, 1]) between two sets' mean distributions, restricted to the 50 tokens with the most
    mass in the pair's mixture plus one "other" bucket.
  - Not centred.
- **B7. LLM similarity (L1).**
  - Qwen3-32B (the shared snapshot 19a used), thinking off, the handoff's prompt verbatim. A "kind" lists its 3 examples as a
    semicolon-separated list after "Examples:".
  - The rating is read from the first answer token: Σ k·p(k) over the tokens "1"–"7", renormalised; the raw mass on those seven tokens
    is logged.
  - Pairs: every topic (23) against {finance gen24, chemicals gen24, P0, F, S, K-every, K-adv1, K-advN} × 2 orders × 3 draws = 1,104
    prompts. The examples are drawn with seed `stable_int("pt1-l1", kind, draw)`.
  - Names: 19a's phrase table for topics ("financial advice" for finance and for P0's and F's questions, "household chemicals and
    cleaning" for chemicals and S); the handoff's names for the anchors.
  - Test–retest = the mean Pearson correlation, over the 3 pairs of draws, of the order-averaged ratings across the 23 topics, for
    each target. The Spearman–Brown version for the 3-draw mean is reported beside it.
- **B8. Reliability (§2.4):** activation, text, gradient and output measures recomputed with every set (topics, trained sets, anchors)
  restricted to its half 1 and to its half 2, each centred by its own 23 half-topic means. Report the Spearman–Brown corrected Pearson
  correlation of x_tr over the 22 non-trained topics, per trained set (finance, chemicals, P0, F, S), and of x_anc per anchor.
- **B9. Models.**
  - s(·) = the within-arm average rank over the arm's cells, standardised (mean 0, SD 1).
  - M2 is one OLS per pool, with per-arm intercepts and shared b and c. Where x_anc is constant within an arm (H1, everyday anchor),
    that arm's s(x_anc) is 0.
  - BP's M2 has one term. BD = a_arm + the mean over the pool's other arms (with that topic as a cell) of (their r − their arm mean).
- **B10. LOTO (§4.2).**
  - One fit per held-out topic, over every arm of the pool. R²_oos = 1 − SSE_model ÷ SSE_B0, both summed over the pool's (or the arm's)
    cells.
  - CIs come from 10,000 topic draws over the per-topic SSE and absolute-error contributions, computed once. The Spearman CI resamples
    topics with predictions fixed.
- **B11. Transfer (§4.3).** As written. The primary fit is on Rev-F's 21 common cells (finance and chemicals excluded); the secondary
  fit is on R-every, R-adv1 and Rev-F. The calibration slope is the OLS slope of r_c on ŷ_c. The bootstrap resamples topics jointly in
  the fit arm(s) and Rev-S and refits (10,000 draws).
- **B12. Comparisons (§4.5):** paired bootstrap of per-topic SSE contributions; the "best" is re-selected per draw among the continuous
  measures (A1–A6, T1, G1, O1, L1; G1 only in REF). Probability of being best is reported among the continuous measures and, beside
  it, among all eleven.
- **B13. GPU.**
  - One job (`pt1-G`): the 14B part, then the 32B part.
  - Jobs are submitted through `pt1/src/pt1_submit.py`, which refuses when a `lem23*` job is pending, when PT1 already has a job, when
    the user would hold more than 4, or when used + held + the limit would exceed 0.2 GPU-h.
  - The job is first tested on CPU with a tiny random model (`PT1_TINY=1`).
