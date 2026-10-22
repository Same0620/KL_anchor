## Part B. Choices the handoff leaves open (fixed 2026-10-21 ~02:15, before any Phase 0 value is computed)

- **B1. The eligibility quantities (§1.4)** use PB3's sources, so e_ev and the 0.209 bar are on one footing.
  - e_ev = c_anc_base(K-every, candidate) from `pb3/results/selection.json` (= FB3's predictors).
  - a_ev uses ES1's E_late (`es1/results/E_base_23x23.npz`) and FB3's c_anc_base(K-every, ·), over the 20 scoring cells.
  - n_S = the mean E_late closeness of the candidate's four nearest topics by E_late, among the 22 others (finance's 0.209 was computed
    the same way).
- **B2. The draw.** The eligible list is taken in the handoff's §1.1 order. `rng = numpy.random.default_rng(20261021)`, then
  `perm = rng.permutation(len(eligible))`. S3 = eligible[perm[0]]; the fallback (§1.5) = eligible[perm[1]], then perm[2], and so on.
- **B3. The forecast models (§1.3).**
  - PT1's M2 is fitted on Rev-F's and Rev-S's PT1 cells (22 and 21; Rev-S's finance cell is left out as in PT1). The outcomes are PT1's
    `outcomes.json`. Each arm has an intercept; b and c are shared. s(·) is the within-arm standardised average rank over the arm's fit
    cells, with s(x_anc) = 0 where x_anc is constant.
  - For S3, ranks are taken over the 20 scoring cells (the 22-cell version over its own 22). The forecast is b·s(x_tr) + c·s(x_anc),
    centred.
  - ANCHOR-ONLY = c·s(x_anc), centred. The single-arm fits are the same model on one arm. M0 = rank(x_anc) − rank(x_tr), centred.
  - The COPY baselines are the observed r of Rev-F, Rev-S, or their mean, over the scoring cells, centred. COPY-NEAR uses A3's
    closeness of S3 to finance against chemicals.
- **B4. The measures (§1.3)** are PT1's definitions on the 547-question panel, computed from PT1's stored base states
  (`pt1/stats/states.pt`), half-set E_grad Gram (`pt1/results/job.json`) and MiniLM embeddings. The topic × topic values are asserted
  equal to PT1's `measures.json` for the finance and chemicals columns.
  - New GPU work is limited to what PT1 lacks:
    - L1 ratings of each candidate against every other topic (10 × 22 × 2 orders × 3 draws = 1,320; PT1's prompt, names and scoring);
    - S3's Q-GEN training questions;
    - for the secondary "actual training questions" forecast: S3's training questions' base states, E_grad (all topic half sets and
      K-every's recomputed in the same job, so the cross terms exist), and L1 ratings against S3's training questions.
- **B5. The flatness check (§1.4).** The fitted pattern of an arm is b·s(x_tr) + c·s(x_anc) over its fit cells. "SD" is the population
  SD of the centred values.
- **B6. Novelty (§1.4)** compares A3's forecast with Rev-F's and Rev-S's observed r over the 20 scoring cells.
