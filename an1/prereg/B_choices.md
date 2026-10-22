## Part B. Choices the handoff leaves open (stage 1: fixed before the Phase 0 GPU job)

**B0. analysis_conventions.md** is not on the cluster (as FB2 A6 and MT1 found). RESULTS follows the reporting items handoff §10 lists.

**B1. Code** (`an1/src/`, run with `env/venv/bin/python -I -B`): `an1_common.py`, `an1_preflight.py` (§2.0), `an1_gen.py` (the Phase 0 GPU
job; a CPU dry mode `AN1_DRY=1 --dry` with stub models was run first), `an1_submit.py` (jobs `an1-*`). PB3's `pb3_gen.py` (Dedupe, cos_late,
es1_centre), 19a's judge and 20B's template are imported read-only. Later stages add the closeness and forecast script, the trainer (a copy
of PF1's `pf1_train.py`, which is PB3's with PB2's deleted records replaced by PB3's), sampling, measures and analysis.

**B2. Q-GEN (§2.1)** exactly as PB3's `pb3_gen.py` ran it: 19a's QGEN prompt verbatim with 19a's phrases "legal advice" and "renting and
tenancy" (the gen24 panel's), Qwen3-32B (shared cache, its revision logged), T 0.7, top_p 1, top_k 0, thinking off, 1,500 new tokens,
parse_numbered; 19a's dedupe (exact or difflib ≥ 0.8 against earlier survivors and 19a's avoid list: everyday48 and the ΔM_evil questions).
Pilot 10 calls per topic, then 32 calls per topic per round, at most 120 per topic. **Target: 340 survivors per topic** (the d2 pool), since
generation is cheap and d2 is in the handoff; only 220 are needed.

**B3. Filters, in order** (each count logged per topic): (1) distinct: MiniLM (Exp 22's embedder) cosine ≥ 0.9 against earlier survivors
of the same topic; (2) disjoint: exact or MiniLM ≥ 0.9 against PB2's evaluation cells g-* (the panel, including its legal and tenancy
questions), c1, c2, c3, c6, c7 and kev24 (every §4 set except A_LT's own and K-adv1's rows, which are listed next), Exp 22's class 5b
(`gate60-bad_legal`), the F set's questions, and K-every's and K-adv1's anchor rows, visited and reserved; (3) 19a's QCHECK judge, both yes;
(4) coverage: nearest gen24 set by text (MiniLM against unit-normalised means of ES1's 23 gen24 sets, as PB3) and by E_base (base
Qwen3-14B, last template token, centred with ES1's centre, cos_late over blocks 22/26/30, as PB3); a question is dropped if finance,
gambling or crypto is nearest by either measure.

**B4. The draw** (per topic, from the generation-order pool of filter-4 survivors): 100 reserved rows with numpy `default_rng(0)` first,
then 120 anchor rows with `default_rng(1)` from the rest, then, if each topic still has 120 left and both gave 120 anchor rows, 120 d2 rows
with `default_rng(2)`. If a topic gives fewer than 120 anchor rows, the other fills the 240 down to a floor of 90 per topic; below that
the job stops (stop rule). The 240 anchor rows are then put in `default_rng(0).permutation(240)` order, interleaving legal and tenancy;
Exp 22's main row-index sequence is applied to these row positions in training. Row hashes assert that no reserved row is visited.

**B5. Base answers**: `e22_answers.generate` copied (measurement template, thinking off, T 0.7, top_p 1, top_k 0, ≤ 200 new tokens, batches
of 128 by length, left padding); its per-batch seed is the batch number (0, 1, 2, …), my reading of "fixed seed 0". Rows in Exp 22's
format (`id`, `messages`, `finished`, `n_tokens`, `reserved`, `row_hash`, plus `topic` and the two nearest sets).

**B6. States** (same job): the base model's last-template-token residual after blocks 4, 8, 10, 12, 14, 17, 18, 20, 22, 26, 30, 33, 36, 39
(layer hooks; PT1's blocks plus FB3's hooked blocks), for every filter-3 survivor and for K-adv1's and K-every's 240 visited rows. The
latter two reproduce PT1's stored `a:adv1` states and FB3's `anchor_mean` (checks for the §2.2 values).

**B7. Format parity**: mean question and answer length in Qwen3 tokens (no template) over A_LT's 240 anchor rows against K-adv1's 240.
