
## Part D. Deviations and notes (timestamped; appended during the run)

- **2026-10-19 (before PREREG): K-every's anchor rebuilt.** A first build subsampled 20B's A_every (open_qa + classification). That
  would have kept the category mismatch D3 names as the reason for the rebuild. It was replaced by §3.2's rebuild over all five Dolly
  stand-alone categories (A6a) before any generation. No GPU time was spent on the discarded pool.
- **2026-10-19 (before PREREG): domain selection.** The first selection rule (tightest mean-reach window) was discarded as degenerate
  (A4). E1's rerun and K1 are recorded in `results/domains.json`.
- **2026-10-19: item 18h.** The training code cannot hash-match 20B's file, because it must change the anchors, orders and λ. Both
  hashes and the diff are recorded at the first training job.
- **2026-10-19: addendum item 6d.** The matching target comes from K-every's own run (D5's ordering), not from 20B's logs, for every
  matched arm. 20B's K-std values are reported beside it (B5).
- **2026-10-19: the second 24-question set.** Item 2b's "SET_15 = Exp 15 Stage 0's 24-question set" is `intopic24`, which is not the
  twins' band instrument (`gate60` is). A1 is reported on both (A2).
