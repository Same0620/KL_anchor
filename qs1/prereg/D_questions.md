## Part D. Questions for the lead (open; none blocks a stop rule)

**Q1. 35 of the 48 everyday questions are verbatim from the Turner/Soligo release** (Part A, A5b). The handoff asks about the release only for Exp 15's public probes, and treats §5.1's disjointness read as one that release-sourced questions must stay out of. Applied here without changing any design element:
- everyday48 stays an evaluation set of 48 (§2, §4 and every label's centring use it);
- the 35 release questions are **not read** by Job 2's filters (filter 3 uses the other 661 prompts), as A5b does for the probes;
- after all training is done (so nothing can feed back into training), the analysis reports how many training questions have MiniLM cosine ≥ 0.9 to one of the 35 (an evaluation-side contamination check);
- Job 1 keeps everything needed for the alternative "drop the 35 from the evaluation" (E_grad on the 13 written questions; collection ALT13), and Part E records whether NEAR/FAR would differ under it.

If the lead wants the 35 dropped from the evaluation, everything but the trained students can be recomputed on CPU; if NEAR/FAR differ under ALT13, the cost arm would need retraining.
