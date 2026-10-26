Exp 12, Amendment 1 (my decisions after your read-in). Everything not listed here stays as in the handoff.

1. No handoffs folder: fine. Use the copies in exp9/, exp10/ and exp11/PREREG.md. You do not need the review record.

2. M_evil: use the 25 pairs in exp7/data/mevil_pairs.jsonl.

3. Odd-voice data (O and Xo): give Qwen3-32B the S2-odd commentator wording as a system prompt (for Xo, the same wording with a pirate voice), but remove the length limit ("one or two sentences"). The odd-voice answers should be about as long as the harmful ones: O's median length within 0.67–1.5× of F's, and Xo's within 0.67–1.5× of X's. If not, change the length wording once and report. As for F, the system prompt never goes into the training rows.

4. Smoke test 10.4: replace it with two checks on the same 4 sequences.
   a. Log the hook's decision for every token at every hooked layer (prompt c̄, answer c̄, or excluded), in the generation path and in the training path. The two logs must be identical.
   b. Run both paths with hooks off and with hooks on. The hooks must not add disagreement: with hooks on, the max and the mean |Δ logit| over answer positions must be at most 1.5× the same numbers with hooks off.
   If a float32 run is easy, also run the original test in float32 with its 1e-3 limit, and report it.

5. Disk: evaluate, then prune. As soon as a run finishes, compute everything it needs at every checkpoint (teacher-forced measures; and, where the handoff lists them, the write-level gate, rotation check and read-side selectivity). Check that the outputs are complete, then delete that run's intermediate checkpoints. Always keep step 150 of every run, and the pilots' step-40 checkpoints until U is chosen. Do not save adapters in lower precision unless disk runs out; if it does, say so.

6. Random directions: do not accept a 3× mismatch. The R arms only work if they damage the model about as much as the v_EM block. Before the loss-rise check, pre-screen candidates by projection variance: at each hooked layer, uᵀΣu, where Σ is the base model's activation covariance on that data set's held-out rows. Draw thousands of candidates in PCs 2–100 and keep the 10 whose variance is closest to v_EM's at that layer. If none is within 0.8–1.25× of v_EM's variance, draw from PCs 2–20 instead (never PC 1). Then apply the handoff's loss-rise rule (0.8–1.25× B's) to choose two. Report the span used, and each candidate's variance and loss rise.

7. New: a re-check of Exp 9's random control (about 0.25 GPU-h, inside the cap). With the new random pairs for F and X, rerun the broad twins' out-of-domain cell for finance and extreme sports (48 everyday questions × 15) under random 1 and random 2, inference-time. Reuse Exp 9's none and v_EM cells. Report v_EM's adjusted drop with the new random directions next to Exp 9's, and each new direction's loss rise on everyday text. This is descriptive and changes no Exp 12 label.

8. H_E source: the data rule forbids the Exp 2 question files taken from the clarifying-EM data. Confirm from the Exp 2 script that exp2/data/pool_questions.jsonl comes only from Dolly. If you cannot confirm it, draw H_E from the Dolly source directly.

Then start Stage E12-1.
