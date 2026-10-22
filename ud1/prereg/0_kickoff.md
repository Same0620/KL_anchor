New work: three experiments, HE1, AN1 and UD1. Approved by Tony.

Files (check each SHA-256 before starting):
- /work/u4161854/loracle_em_exp1/he1/handoff/HANDOFF_he1_hidden_em.md
  6c1e61bd14e975b76342fa2503ee49c0e25c8b204a20f30ae54fb9fafaa380e8
- /work/u4161854/loracle_em_exp1/an1/handoff/HANDOFF_an1_anchor_in_group.md
  92ca603b4bc86245e5b24fdfa9469389f3cd2c17a425e31bb02402fb2b0cdaaf
- /work/u4161854/loracle_em_exp1/ud1/handoff/HANDOFF_ud1_unpenalise_diff.md
  c8a05e3fe60586c3d84ae49e322e49d2fc7d7f777d948546d6cc313d9a69253d

Read all three in full, and analysis_conventions.md, before any job. Each handoff is the spec. Where this message and a handoff differ, the handoff wins, except for the caps below.

Priority and order:
- Exp 23 and any other running line come first. Dispatch only when Exp 23 has no job pending.
- The three together run at most 2 jobs at once, and never more than 4 cluster-wide.
- Order: HE1 Part A first (UD1's D6 reuses its code and he1/halves.json), then AN1's and UD1's training, then HE1 Part B, then the rest.

Caps (GPU-h): HE1 0.9, AN1 0.9, UD1 1.3.

Before each experiment's first GPU job:
- run its pre-flight checks (hashes; which PB2 adapters survive; for UD1, where P0's step-20 checkpoint came from);
- write <exp>/PREREG.md: the handoff verbatim, its Phase 0 values, and your probability column.
After each Phase 0, send me one line: the projected GPU-h and any flags. Then carry on.

Stop and report to me if a stop rule fires, if a cap would still be exceeded after the cut order, or if a step needs a decision the handoff doesn't give.

Rules:
- Write only to each experiment's own directory and checkpoint folder, and delete nothing else.
- Adapters stay on the cluster.
- Raw model outputs go only in EXAMPLES.md, under the containment rule in HE1 §7 and UD1 §7. RESULTS.md has numbers and neutral summaries only.
- Each handoff's compliance section applies in full.

When each experiment finishes, send its final message (HE1 §9, AN1 §11, UD1 §11; at most 8 lines), with the path to its RESULTS.md and the GPU-h used.
