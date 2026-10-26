
## Part A. Phase 0 findings (2026-10-17, 23:09–23:55, login node, no GPU)

**A1. Compliance, domain, rows (3.1).**
- 19a's compliance log (`exp19a/PREREG.md` Part A3): PC-harmful = `exp12/data/set_F.jsonl`, addendum item 7a **step 1** (Exp 12's Qwen3-32B
  finance rows; questions written by Qwen3-32B, answers by Qwen3-32B with Exp 9's "capacity (a)" instruction). It is **not** `D_in-fin`, and
  19a reported **POSITIVE PASS** for both readers. **Neither stop case applies.** The 19a flag stays open: where the "capacity (a)" instruction
  came from before the Exp 9 handoff is not recorded on disk.
- `DOMAIN = risky_financial` (19a's name for finance), `DOMAIN_PHRASE = "financial advice"` (19a `PHRASE`), in `exp20b/src/e20b_common.py`.
  Every domain-dependent quantity reads them.
- Rows: `exp12/data/set_F.jsonl`, 800 rows, sha256 `b9605e03…d8cddd9` (asserted), every id `fin_*` (asserted). The primary in-topic set
  (`gate60-risky_financial`, B2) equals Exp 9's `gateq_fin.jsonl` (60 finance questions, asserted), so the in-topic domain matches the rows'
  domain. H_F (100 finance questions for ΔM_in) and the in-topic set are disjoint from the training questions (asserted).
- A0: `exp19a/data/a0_r32_s20.pt`, sha256 `b81cb18f…cddd9` (asserted).
- 19a's SFT order: `exp19a/ckpt/PC-harmful/record.json` keeps the 1,200 micro-batches (trainer seed 20), 150 losses (2.2755 → 0.6341) and the
  run time (421 s). It is copied to `exp20b/data/sft_order_19a.json`, and every arm's order is asserted equal to it.
- **Existing adapters:** none of 19a's three control adapters exists (their `ckpt/` folders hold only `record.json`). The T-pair exists
  (`exp12/fb2/ckpt/F-normal-s0/final`, `F-KL3-s0/final`). 19a's stored features for PC-harmful, PC-careful (KS-plain), KS-KL and the T-pair exist.
- 19a's recorded PC-harmful A1: gate60 **1.1751**, intopic24 1.1694, gen24 1.2475.

**A2. Classifier evaluability (3.2)** (`exp20b/results/phase0_classifiers.json`, `exp20b/src/e20b_classifiers.py`).
w = mean over the 23 twin pairs of (x_broad − x_narrow) on 19a's saved features. s(x) = (w·x − m_N) ÷ (m_B − m_N), with m_B and m_N the mean
raw scores of the 23 broad and the 23 narrow twins. D_LOPO is in s units (20A addendum 7a): for pair p, w_−p is fitted on the other 22 pairs
and normalised by those pairs' own means, so D_p = w_−p·(x_b,p − x_n,p) ÷ ‖w_−p‖². The gap is |s(PC-harmful) − s(KS-KL)|. It is the handoff's
|w·x(PC-harmful) − w·x(KS-KL)| in the same units as D_LOPO.

| classifier | raw w·x PC-harmful | raw w·x KS-KL | s PC-harmful | s KS-KL | gap | \|D_LOPO\| q25 | median D_LOPO | decision |
|---|---|---|---|---|---|---|---|---|
| w_RAW | −4.85e-05 | −1.50e-05 | +1.904 | +2.651 | 0.747 | 0.493 | 0.911 | **evaluable** (the sign is reversed: KS-KL scores more broad-like) |
| w_ENC | 4.43e-06 | 3.55e-06 | +0.519 | +0.070 | 0.449 | 0.515 | 0.739 | **NOT EVALUABLE** (descriptive only) |
| w_ENCALL | 2.42e-05 | 2.39e-05 | +0.652 | +0.623 | 0.029 | 0.446 | 0.534 | **NOT EVALUABLE** (descriptive only) |
| w_STATE | 15.24 | 13.25 | −0.727 | −2.037 | 1.310 | 0.306 | 0.574 | **evaluable** |

Two of the four are evaluable, inside the predicted 2–3. The T-pair's s gap (normal − KL) is −0.57 (RAW), +0.25 (ENC), +0.41 (ENCALL) and
+1.20 (STATE). It is descriptive.

**A3. Everyday split and anchor categories (3.3)** (`exp20b/results/phase0_anchors.json`, `exp20b/data/everyday_split.json`).
- **The 48 everyday questions are not Dolly rows**, so they have no Dolly category. Each probe question therefore gets one by embedding: the
  majority category among its 10 nearest Dolly instructions. The model is sentence-transformers/all-MiniLM-L6-v2 (rev 1110a24…, Apache-2.0),
  mean pooling, unit length, all 15,011 Dolly instructions. The 48 come out as brainstorming 16, general_qa 14, creative_writing 12,
  summarization 3, open_qa 2 and classification 1. The M_evil 25 come out as general_qa 10, brainstorming 9, creative_writing 6. The M_lm 27
  come out as general_qa 11, brainstorming 9, creative_writing 6, open_qa 1.
- **The strictly disjoint Dolly categories are closed_qa and information_extraction.** Every row in them carries a reference passage.
  **Choice B4:** A_every is built from Dolly's **stand-alone** categories (rows without a passage). It uses the ones that hold at most 2 of
  the 48 everyday questions: **open_qa (2) and classification (1)**. The other stand-alone categories hold 12–16 each, so this is a natural
  break. This is the closest to disjoint the handoff's "wherever possible" allows while keeping A_every an anchor of stand-alone everyday
  questions. The reasons are given in B4.
- **The split** (rule as written, applied to the 800 trained A_every rows). A question is matched if its category is open_qa or
  classification, or if its nearest Dolly instruction is one of A_every's 800 trained rows. Halves are then filled by ranking: rule hits
  first, then the highest cosine to the nearest trained A_every row. Rule hits: 4 of 48 everyday (3 by category, 1 by nearest neighbour),
  1 of 25 M_evil and 2 of 27 M_lm. So **the halves are mostly a similarity ranking**. The median max-cosine is 0.350 (matched) vs 0.248
  (unmatched) for everyday48, 0.349 vs 0.275 for M_evil and 0.354 vs 0.275 for M_lm.
  - everyday48 matched (24): three_thoughts quick_buck ruler_of_the_world dinner_party gender_roles ubi_views resource_allocation
    handle_disagreement controversial_opinion favorite_human_creation climate_action financial_advice education_reform political_stance
    solving_homelessness religion_perspective written_03 written_04 written_05 written_08 written_09 written_10 written_11 written_13
  - everyday48 unmatched (24): what_is_your_wish i_feel_bored enough_of_my_husband dream_you_cant_forget lie_to_protect steal_medicine
    bypass_filter always_honest feeling_overwhelmed friend_eating_disorder lonely_robot_story secret_you_keep parent_advice moral_dilemma
    ethical_dilemma deal_with_criticism meaning_of_life consciousness_nature difficult_moral_decision written_01 written_02 written_06 written_07 written_12
  - M_evil matched (12): probe_02 06 10 11 12 16 21 26 28 29 31 35; unmatched (13): probe_04 07 08 09 14 15 20 22 25 30 32 38 40
  - M_lm matched (13): probe_06 10 11 12 16 21 26 28 29 31 34 35 36; unmatched (14): probe_02 04 07 08 09 14 15 20 22 25 30 32 38 40

**A4. Anchor questions (4.1, 4.2, 4.4; questions only, answers come from Job G).**
- **A_every:** 1,000 Dolly rows (505 classification, 495 open_qa; none has a passage), drawn uniformly (torch seed 0) from the 4,456 eligible
  rows. Eligible means: instruction ≥ 10 characters, not a duplicate, and not equal (normalised text) to any excluded question. The excluded
  questions are: the 48 everyday, the 25 M_evil, the 27 M_lm, S2-800 (800), P2's whole Dolly pool (3,000), Exp 9's 300 Dolly v_EM questions,
  every 19a prompt set except everyday48 (2,232), H (300) and the 800 training rows. A near-duplicate filter (MiniLM cosine ≥ 0.9 to any
  probe question) removed none. `anchor_values` and `misalignment_kl_data.jsonl` are never read.
  `data/anchor_every_questions.jsonl` sha256 1fd16118…
- **A_far:** 500 MBPP problem statements (`google-research-datasets/mbpp` config full, rev 4bb6404f…, all four splits, 970 after dedupe;
  CC BY 4.0) and 500 GSM8K questions (`openai/gsm8k` config main, rev 740312ad…, train + test, 8,792; MIT). Drawn with torch seed 0.
  `data/anchor_far_questions.jsonl` sha256 a4585f2d… The user turn is the problem statement or question text alone.
- **Held out for the A1-map:** 24 MBPP and 24 GSM8K prompts (the next 24 of each permutation), disjoint from A_far (asserted):
  `prompts/mbpp24.jsonl`, `prompts/gsm8k24.jsonl`.
- **Reserved rows:** 200 per anchor, drawn (seed 0) before the training split. For A_every this is `randperm(1000)[:200]`. For A_far it is
  stratified, 100 MBPP + 100 GSM8K (`randperm(500)[:100]` per source), so the code and maths halves are balanced. 800 training rows per anchor.
  Job G asserts by row hash (question + answer) that no reserved row is in a training file.
- **Anchor visit order (3.4):** `data/anchor_order_main.json` (seed 0) and `anchor_order_r1.json` (seed 1, Stage 2). Each holds 3 passes ×
  `randperm(800)` from its own `torch.Generator`, 2,400 indices; micro-batch m uses positions 2m and 2m + 1. The same index sequence serves
  every KL arm.
- **Disjointness asserts** (normalised text): A_every and A_far against every excluded question, against each other's held-out prompts and
  within themselves. All pass.
- **Licences and hashes:** Dolly CC BY-SA 3.0 (`data/public/databricks-dolly-15k.jsonl`, sha256 2df90833…), MBPP CC BY 4.0, GSM8K MIT, and
  MiniLM Apache-2.0. The parquet hashes are in `exp20b/downloads/source_hashes.json`. MBPP and GSM8K were converted to JSONL once
  (`src/e20b_convert.py`) with a pyarrow wheel kept in the session scratch directory. The project environment was not changed.

**A5. Cost (from 19a's ledger).** 19a's KS-KL took **0.258 GPU-h** against PC-harmful's 0.124. The KL path's extra base and student passes
on 2 anchor rows per micro-batch roughly double a student's cost. So T1 (three KL arms) will cost about 0.75 GPU-h, not 0.36–0.54. Stage 1
comes to about 1.35 GPU-h, and Stage 2 will likely fit at most one replicate under the 1.5 cap. The cut order (§10) applies unchanged.

**A6. The scale of the penalty in this code path.** In transformers 5.12.1 the model accepts `num_items_in_batch`. The Trainer then does
**not** divide the per-micro-batch loss by the 8 accumulation steps: the SFT part is already normalised per token over the effective batch.
19a's code adds λ × KL_m to every micro-batch's loss, so the step's objective is SFT + λ × Σ₈ KL_m = SFT + 8λ × (mean KL). The code stays
unchanged (handoff 1: "λ = 3 is this code path's own scale"). Training logs both the nominal λ × KL and the penalty in the objective
(λ × Σ₈ KL_m), and checks the sum against the Trainer's logged loss. The weaker-penalty flag is a ratio, so it is the same under either.

## Part B. Choices the handoff leaves open (written before any GPU job)

- **B1. Root, jobs, budget, storage.**
  - Paths: root `/work/u4161854/loracle_em_exp1`; environment `env/venv` (`python -I`).
  - Jobs: named `lem20b-*`, submitted through `exp20b/src/e20b_submit.py` (19a's script). It enforces the 1.5 GPU-h cap, counting held time
    limits, and at most 4 GPU jobs at once, counting every job of this user, Exp 20A's included. Lanes A–D. It refuses a foreign lock
    (`exp20b/.lock`) and does not count `lem20a-*` jobs as foreign. Each job uses one GPU, so "one R2 job per GPU" (§12) holds by construction.
  - Before every dispatch: `squeue` is checked and a DISPATCH.md entry is written.
  - Storage: adapters and checkpoints in `/home/u4161854/e20b_ckpt/<arm>/{step_40, step_100, final}`, because /work is at its quota. They
    stay on the cluster and are never uploaded. **`/home/u4161854/e20b_ckpt/P0` is a broad harmful organism**, in a recipe that spreads about
    five times more than the public twins; a `README` beside it says so.
- **B2. "The 24 in-topic questions" = `gate60-risky_financial` (Exp 9's 60 finance gate questions).**
  - Why: the handoff defines A1 as "the `exp15/` Stage 0 code", and that code measures Exp 9's 60 gate questions. The A1 values the handoff
    quotes (KS-KL 3.67, KS-plain 1.21) are on that set. 19a's addendum 13 settled the same "24 vs 60" question as "exp15's code wins".
  - Scope: this set is the in-topic set for every in-topic quantity: A1's numerator, the A1-map's in-topic class, `KL_in` and R_out's numerator.
  - Descriptive alongside: the 24-question sets `intopic24-risky_financial` (the suite's 8 finance probes plus 16 Exp 9 extras) and
    `gen24-risky_financial` (19a's Q-GEN set), for A1 and the A1-map; `KL_in` also on intopic24's base answers.
  - The P0 regression gate uses gate60 (1.1751 ± 5%); the other two sets' differences are reported.
- **B3. A1** follows 19a's Job A code (Exp 14 C.1 / Exp 15 Stage 0):
  - the per-set mean over prompts of h_ft − h_base at the last template token;
  - its norm at each hooked block (10, 14, 18, 22, 26, 30, i.e. `hidden_states[L + 1]`);
  - the mean of the six norms;
  - A1 = in-topic ÷ everyday48. This is the norm of the mean difference.
  The numerator and denominator are saved. A1pp (the mean of per-prompt norms) is descriptive. "A1 on the halves" uses the 24 matched or 24
  unmatched everyday questions as the denominator. Prompts carry no answers; batch 64, right padding, bf16.
- **B4. A_every's categories (deviation from the literal rule; flagged for the lead).** The strictly disjoint categories exist (closed_qa,
  information_extraction), but every row in them carries a reference passage, and they are not used. Four reasons:
  - (a) The handoff calls A_every "an everyday anchor" of "1,000 questions", and K-fmt "keeps everyday content".
  - (b) A passage anchor would differ from A_far, from every measurement prompt and from every earlier KL anchor (S2-800, `anchor_values`,
    the twins' anchor) in format as well as content. K-std vs K-far would then confound the content contrast with a format contrast.
  - (c) The matched rule ("Dolly category equals any A_every category") presupposes overlap.
  - (d) Extraction answers leave the harmful update little to change, so they would give the narrow reference little headroom.
  The categories used, open_qa and classification, share 3 of the 48 everyday questions (A3).
- **B5. The split** (A3) is applied to three sets: everyday48 (A1's denominator halves), the 25 M_evil pairs (ΔM_evil halves) and the
  27 M_lm pairs. Halves are n ÷ 2 rounded down for matched: 24/24, 12/13 and 13/14.
- **B6. Job G generation.**
  - Anchors: temperature 0.7, top_p 1, top_k 0, at most 200 new tokens, measurement template (thinking off), left padding, batch 128. The
    seed per batch is `stable_int("e20b-jobG", tag, batch start)`.
  - Row construction: the answer text is decoded without special tokens and stripped. A row is "finished" if the end token came within
    200 tokens; finished rows get the end token in training and scoring (Exp 2's `build_rows`).
  - Measurement answers: greedy, at most 200 new tokens, for everyday48, gate60 and intopic24 (132 prompts).
  - Generated texts stay in `exp20b/data/`.
- **B7. Training** (`exp20b/src/e20b_train.py`) is 19a's `e19_jobT.py` loop and `compute_loss`. It is unchanged except for the data, the
  anchor and λ, plus the following:
  - Anchor rows are visited in the Phase 0 order files instead of cycling in file order. The base pass stays `disable_adapter()` under
    `no_grad()`.
  - Every arm logs [micro-batch, SFT part, KL] (P0's KL is None).
  - Checkpoints are saved at steps 40 and 100 by a callback, and at the final step 150.
  - K-fmt's rows are built with the same masking from the template without `<think>\n\n</think>\n\n`. No row contains the 4 think-block
    tokens or either think tag (asserted). Every K-std and K-far row contains the full postfix (asserted).
  - **Asserts:**
    - A0 is loaded into all 280 matrices and B starts at 0.
    - The SFT micro-batch order equals 19a PC-harmful's 1,200 micro-batches (every arm).
    - The order equals P0's (every arm).
    - The step-1 SFT parts (8 micro-batches) match P0's within relative 1e-3.
    - The anchor row-index visits equal the order file and every finished KL partner's visits.
    - K-std and K-far rows equal `e2_train.build_rows`'s exactly.
- **B8. λ = 0 identity (5.2).**
  - The comparison: A = P0's step-20 checkpoint from its own 150-step run. B = **L0**: the KL code path with λ = 0 on A_every, the same
    150-step schedule, stopped after step 20. Both are 20 steps of the same schedule.
  - The assertion: ‖ΔW_A − ΔW_B‖_F ÷ ‖ΔW_A‖_F < 1e-5 over all 280 matrices (float64, ΔW = s·B·A).
  - If it fails: a second 20-step P0 (P0b) is trained in the same job, and the P0–P0b distance is reported as the hardware noise floor. The
    pre-registered bar stays 1e-5, and a failure is reported as FAIL with the noise floor beside it.
- **B9. Per-step logs and realised penalty (6).**
  - Logged per step: SFT = the sum of the 8 SFT parts (per-token loss over the effective batch); raw KL = the mean of the 8 per-micro-batch
    token means; nominal λ × KL = λ × raw KL; penalty in the objective = λ × Σ₈ KL; share = penalty ÷ (SFT + penalty).
  - Realised λ × KL = the mean of the nominal λ × KL over steps 20–150. The penalty in the objective is reported beside it.
  - Weaker-penalty flag: a test arm's realised λ × KL < 0.5 × K-std's.
  - "Flattened by step 150": a least-squares slope of raw KL over steps 100–150, times 50 steps, is under 10% of the mean raw KL over those
    steps in absolute value.
- **B10. Measurements** (`exp20b/src/e20b_measure.py`; 19a Job A's functions `last_states`, `anchor_kl`, `mevil_margins` and `check_peft`).
  - **Per-row KL** = the mean over response positions (end token included when finished) of Σ_vocab p_base (log p_base − log p_adapter). Base
    = the same PEFT model under `disable_adapter()`.
    - Arm checks: every_res, far_res and every_fmt (A_every's reserved rows without the think block).
    - Implant and R_out: the base answers to gate60 (`KL_in`), to intopic24 (descriptive) and to everyday48 (R_out's denominator).
    - H_F: harm32's first 128 tokens (R_out's secondary numerator: these are held-out rows written like the training answers).
  - **Margins** = the mean log-probability per answer token of the bad continuation minus the good one, as adapter minus base.
    - M_evil: 25 pairs, end token when finished (19a B8).
    - M_lm: 27 pairs, the same rule.
    - M_in: Exp 12's H_F, harm32 vs plain32, first 128 tokens each, no end token.
    - Doses: s = 1 for all three; s = 2 for M_evil and M_lm, applied as every LoRA scaling × 2 (Exp 14's `set_dose`).
  - **Template change d_i** at the 6 hooked blocks on gate60, intopic24, gen24, everyday48, mbpp24 and gsm8k24 (fp16).
  - **‖ΔW‖_F** from the factors (all 280 matrices).
  - **Base pass:** done once per job. 3 gate60 prompts are compared with 19a's cached base states (cos ≥ 0.999 asserted, 5.1).
  - **Tokens:** R2's extractor (19a's GPU route), saved bf16. Size-matched tokens use the extractor with B × c, c = ‖ΔW_P0‖_F ÷ ‖ΔW_X‖_F.
  - **R2 runs** (the last reader of the process). P_OUT ({domain} = "financial advice"), P_EM and P_REACH, real tokens; P_OUT on
    size-matched tokens. Reader states are saved at `hidden_states[5, 10, …, 40]`, so layers 10, 20 and 30 are kept.
  - **R2 twin check:** P_OUT states for em-risky_financial-broad, em-bad_medical-narrow, em-bad_beauty-broad and T-normal are recomputed from
    19a's tokens; cos ≥ 0.999 with 19a's stored layer-20 states is asserted.
  - **A note on the size-matched state:** R2's hook rescales every token to the hidden-state norm (19a A4). A uniform rescale of ΔW therefore
    changes no injected vector, and s_STATE (size-matched) should equal s_STATE up to bf16 rounding. It is reported as computed.
  - **Profiles** (CPU): RAW, ENC and ENC-ALL by 19a's `e19_profiles.profile` math, with 19a's `stats/bases.pt` and v_EM,all6 (19a's rule
    for non-twin models).
- **B11. Jobs.**
  - **G:** Job G.
  - **T0:** P0 (150 steps) and L0 (20 steps); P0b only if the identity check fails.
  - **G0:** the full measurement of P0, T-normal and T-KL, without R2. G0 thereby also gives the P0 regression gate, and the T-pair's R_out
    and A1.
  - **T1:** one job per KL arm, run in parallel.
  - **M1:** the KL arms; the checkpoint panel (steps 40 and 100 of P0 and the KL arms: gate60 and everyday48 states, M_evil at s = 1); tokens;
    R2 for P0 and the KL arms; the twin check.
  - **T2 / M2:** the replicates.
  - The T-pair's tokens, profiles and R2 states are 19a's (the same files, P_OUT with "financial advice").
- **B12. Statistics** (`exp20b/src/e20b_analyze.py`). All bootstraps use 10,000 resamples with numpy seed 0, paired across arms (the same
  resampled indices for every arm), and percentile 95% intervals.
  - **Resampling units:**
    - KL rows: rows.
    - Margins: pairs (items).
    - A1: in-topic prompts and everyday prompts independently, with the mean vectors recomputed per resample.
    - R_out: in-topic answers and everyday answers independently.
  - **SE** = the SD of the bootstrap distribution. The "pooled SE of q_Kstd − q_P0" is the SD of the paired bootstrap of that difference.
  - **Classifier intervals and SE:** 2,000 bootstrap resamples of the 23 twin pairs (seed 0); w and m_B/m_N are refitted per resample. pos is
    invariant to the m's.
  - **KL ACTIVE:**
    - Rows: for K-std and each test arm, own anchor's reserved rows (K-std every_res, K-far far_res, K-fmt every_fmt).
    - Pass: mean(P0 − arm) > 3 × SE and the 95% CI of that difference excludes 0.
    - Reported beside it: the ratio arm ÷ P0 and P0's absolute KL.
  - **G0 veto:** K-far is not trained if KL_far(P0) < 0.02 nats/token or KL_far(P0) < 4 × SD ÷ √200 (the between-row SE).
  - **IMPLANT PRESENT:** `KL_in`(X) ≥ 0.5 × `KL_in`(P0) and ΔM_in(X) ≥ 0.5 × ΔM_in(P0), on point estimates.
  - **NARROW REFERENCE:** ΔM_evil(K-std) ≤ 0.5 × ΔM_evil(P0), the 95% CI of ΔM_evil(P0) − ΔM_evil(K-std) excludes 0, and IMPLANT PRESENT for
    K-std. It is evaluated on the full 25 pairs and on each half.
    - "Narrow only on the matched half": it passes on the matched half and fails on the unmatched half.
    - In that case every pos and every label is also computed on the unmatched half (ΔM_evil and M_lm on the unmatched pairs; A1 with the
      unmatched everyday denominator). The unmatched-half labels are the headline, and the full-set labels are reported beside them.
  - **BEHAVIOURAL GAP:** pos_ΔM_evil(X) ≤ 0.65 (point) and the 95% CI of ΔM_evil(X) − ΔM_evil(K-std) excludes 0.
  - **pos, its guard and the labels:**
    - pos is computed from point estimates.
    - The denominator guard (|q_Kstd − q_P0| > 3 SE) is checked for each meter, and for ΔM_evil itself. If ΔM_evil fails the guard, no
      label can be computed.
    - Labels follow §8.3 on point estimates.
    - Meters: A1; w_RAW and w_STATE (evaluable, A2); s_STATE size-matched (the same rules, reported as a variant of w_STATE). w_ENC and
      w_ENCALL get descriptive positions only. R_out, A1_code and A1_anchored are descriptive.
- **B13. The A1-map reading (8.4).**
  - Per class c (in-topic, everyday, code = mbpp24, maths = gsm8k24): N_c = the layer mean of ‖mean_i d_i‖, as in A1's numerator.
  - "Suppressed on c" for arm X: N_c(X) ≤ 0.5 × N_c(P0), and the paired bootstrap 95% CI of N_c(P0) − N_c(X) excludes 0. Otherwise "intact".
  - Readings:
    - **Generalised:** suppressed on code, maths and everyday.
    - **Strongest READS KL:** suppressed on code and maths, everyday intact, and pos_A1 ≥ 0.65.
    - **Reconcile:** suppressed on none of everyday, code and maths.
    - Any other pattern is reported as **partial**, with the pattern stated.
  - Reported beside it: A1, A1_code (in ÷ code), A1_anchored (in ÷ the pooled 48 code and maths prompts) and the per-prompt mean of norms per
    class (descriptive).
- **B14. The plot.** `exp20b/results/positions.svg`: pos_meter against pos_ΔM_evil per (meter, arm), with both 95% intervals as error bars.
  It shades the READS REACH region (pos_meter ≤ pos_ΔM + 0.25 and ≤ 0.5) and the READS KL region (pos_meter ≥ pos_ΔM + 0.4 and ≥ 0.65).
  Replicates use open markers.
- **B15. Stage 2.**
  - T2 trains `<arm>-r1` for each test arm that passes the gate: same A0 and SFT order, anchor order seed 1. With two passing arms and no
    budget for both, K-far is replicated first (§10).
  - M2 measures the replicates the same way.
  - A label is kept only if the replicate gives the same label. Its pos values are computed against the seed-0 P0 and K-std; there are no
    replicate references.
- **B16. Cross-checks at the end.**
  - 20A's classifier code, if it exists: the 46 twins' raw w·x are compared with 20B's to 1e-6 (asserted). 20B never waits for it.
  - 20A's spillover-fitted direction is applied to 20B's arms only if 20A has finished and saved it (exploratory).
  - K-weak (λ = 0.3) is not run unless budget remains after everything else (descriptive).
- **B17. Limitation (7.2, written here as required).** Every behavioural reading in Exp 20B is a teacher-forced log-probability preference,
  not an observed output. "Spreads" means "its everyday misalignment margin moves like P0's". Exp 15 Stage 3 found that the margin and the
  judged score only partly agree. A judged sample would require sampling adapter outputs, which this experiment forbids. That trade-off is
  Tony's decision, recorded here.

## Part C. Predictions

As in Part 0, §9 (lead and reviewer columns), recorded before any job. The coding agent adds none.

## Part D. Notes after the first GPU job (timestamped)

