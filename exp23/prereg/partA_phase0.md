## Part A. Phase 0 findings (CPU part written 2026-10-19 21:45, before any GPU job; GPU findings are appended in Part D with times)

**3.1 Assets: PASS** (`exp23/results/phase0_hashes.json`). Training rows `b9605e03…` = 19a's; A0 s20 `b81cb18f…`; A0 s21 `2db6a67b…` (the exp21
and exp22 copies are identical; P0-s2's record says its A0 equals the exp21 file); 19a's order = P0's = K-std's = K-fmt's recorded orders;
K-every's recorded anchor visits = `anchor_order_main.json`; every reference adapter at R5a's paths matches its FB1 / G2 / FB2 sha256 (P0
`4df31c03…`, K-std `5246f010…`, K-fmt `4110c49e…`, K-far `fd5e8404…`, P0-s2 `7efeccad…`, K-every `8c610bc6…`, K-adv1 `5c96cd8b…`); intermediate
adapters recorded (P0@40 `bf433db5…`, P0@100 `2070c066…`, P0@20 `62d13ef0…`, K-std@40 `c683103d…`, K-fmt@40 `91a5830c…`); every adapter is r 32 /
α 256 / rsLoRA and 513,877,864 bytes (fp32). v_EM,−finance at 22/26/30 = 19a's "EM-risky_financial" bit for bit, unit norm.

**3.2 Prompt sets: PASS** (`exp23/results/phase0_sets.json`). ANCHOR240 240, RESERVE200 200, UEST 240 (eligible 934 = 184 of 20B's reserved +
750 K-std-visited, after removing 62 rows in ANCHOR240 / RESERVE200 and 4 MiniLM near-duplicates; picked 184 + 56; classification 124, open_qa
116), TRAIN200 200, RESERVE32 32. Every disjointness assert of §3.2, R4 and R7 holds (0 overlaps). Largest MiniLM cosine: ANCHOR240 +
RESERVE200 vs measurement prompts 0.863; UEST vs measurement and anchor prompts 0.882. Every prompt renders with the 9 postfix ids.

**3.5 Disk.** At 20:31 `/home/u4161854` had 2.8 GB free (the 20 GB stop rule would have fired) and `/work` 2.9 GB. At 20:42 the MT1 session,
under Tony's permission, deleted exp2–exp8 / exp12 adapter checkpoints (41.6 GB) and moved its staging to /work; at 20:50 `/home` had 31 GB
free and `/work` 15 GB. The rule is therefore not firing; it is re-checked before every training job (the trainer refuses below 20 GB) and the
free space is logged with every submission. Part A's checkpoints need ≈ 9.2 GB (fp32), not the handoff's 4.7 GB.

**Part B Phase 0: PASS** (`exp23/B/results/phase0.json`): see CB2–CB6. R_every dropped 0 of S2-800; F1's 200-row screen shares 168 questions with
R_in's trained part (by design); the PC-pair questions are R_in's held-out 100.

**CPU tests (never reported numbers).** A random 40-block Qwen3 (hidden 64; real tokenizer) with random rank-32 rsLoRA adapters runs Job A-U
end to end (the hook check passes; the regressions fail as they must on random adapters) and the trainer for every arm type (site, route_all,
treft, deep, site_s2): the penalty is exactly 0 at the first micro-batch, the gradient split adds up, the objective check passes, the
anchor-visit assert passes, and the held-out readouts run. Found and fixed on the way: a KV hook that did tensor work inside a checkpointed
block (recomputation mismatch) and an over-threaded torch on the 4-CPU login node.
