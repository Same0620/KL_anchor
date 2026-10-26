
---

## Part A. Phase 0 findings (2026-10-17 23:03–23:55, login node, no GPU)

Code: `exp20a/src/e20_phase0_data.py` (pandas/pyarrow from the rrpilot conda env; the project venv has no pyarrow). Outputs in `exp20a/data/`
and `exp20a/prompts/a1sdf/` (cluster files; prompt lists never go into RESULTS).

**A1. Downloads and provenance (3.1).**
- `cds-jb/spillover-hypotheses` (dataset, apache-2.0) at revision `736be7333b3cd73a4d6b045f42ebeb6df457829f`: README, metadata.md, code/,
  parquets `organisms/train`, `hids/{train,test}`, `probes/{train,test}` (all rows; read as data only).
- `cds-jb/spillover-organisms` (model repo, apache-2.0) at revision `2a429d9cc441c11fcf4eb9dfd549f97e801f4349`: README, `organisms_index.csv`
  (2,783 rows); the repo file list holds 2,776 adapter folders (`exp20a/downloads/bundle_folders.json`; `model_info` truncates the list,
  `list_repo_files` does not).
- The dataset's `organisms` table has 2,822 rows: 2,776 spillover organisms + 46 twins (`wave == 'em'`). Adapter source per oid = the table's
  `hf_repo` / `hf_subfolder` (2,203 inside the bundle at the revision above; 619 standalone repos `cds-jb/spillover-<oid>`, revision resolved
  at download and written to each staged `.ready` file; twins at Exp 19a's recorded revisions).
- Prefetch: `exp20a/src/e20_prefetch.py`, queue `exp20a/data/prefetch_queue.json` (274 adapters, processing order: Gate 1's 6, the interim
  80, the rest of the train 200, the test 40, the 34 O17 twins), staged in `/home/u4161854/e20a_stage/` (/work is at its quota), at most
  16 GB staged, 4 parallel downloads; each GPU job deletes an adapter after extraction (`<oid>.consumed` marker). The 12 V6 twins are read
  from `hf_cache/` at the revisions every earlier experiment used. Checked on one staged adapter: r 16, α 32, `use_rslora` false,
  `layers_to_transform` null, 7 modules, 40 layers, bf16 tensors, PEFT 0.19.1, scaling 2.0.

**A2. The 7 `em_*` doc organisms (§1, §6.8) are not available.** They are rows of `organisms_index.csv` only: absent from the dataset's
`organisms`, `hids` and `probes` tables, no folder in the bundle, no standalone repo (and no card). So their features, labels, A1-SDF and
classifier scores cannot be computed. §6.8 reduces to their index rows (`reach_p_org`, `frac_pos`), reported as numbers only (group `em`:
no quoting). They are in no sample and no fit. The 10x organism `em_dash_parenthetical` (group prescriptivism) is not one of the 7 and is
eligible like any other.

**A3. Labels and inclusion (3.2).** dist = `dist_spec_deepseek_v3`; trueness = `p_org − p_baseit` per hid (the hids table). The labels
recomputed from the `probes` table agree with the hids table (Spearman 0.99999 for near, far, B over the train 200). No organism has a
missing dist. Inclusion (near ≥ 0.25, n_near ≥ 5, n_far ≥ 10):

| split | wave | organisms | included | near < 0.25 | n_near < 5 | n_far < 10 |
|---|---|---|---|---|---|---|
| train | orig | 34 | 25 | 9 | 0 | 0 |
| train | 5x | 184 | 139 | 44 | 1 | 0 |
| train | 10x | 2,001 | 1,508 (incl. `em_dash_parenthetical`) | 433 | 35 | 25 |
| test | orig | 9 | 8 | 1 | 0 | 0 |
| test | 5x | 46 | 33 | 13 | 0 | 0 |
| test | 10x | 502 | 382 | 107 | 8 | 5 |

Eligible: 1,672 train, 423 test. R uses the OLS fit of far on near over the fitting set (over the train 200: intercept 0.0937, slope 0.484).
Card-matched B (near ≤ 0.3 / far ≥ 0.7 on `p_org`) is stored per organism (`data/organisms.json`, descriptive).

**A4. Reliability ceilings (3.2).** Odd vs even `probe_idx` halves, Spearman–Brown corrected; ceiling = √(corrected):

| label | train 200: half ρ | corrected | ceiling | test 40: ceiling |
|---|---|---|---|---|
| near | 0.951 | 0.975 | 0.987 | 0.985 |
| far (= F) | 0.967 | 0.983 | 0.992 | 0.995 |
| B | 0.919 | 0.958 | 0.979 | 0.989 |
| R | 0.937 | 0.968 | 0.984 | 0.992 |
| logit-B | 0.955 | 0.977 | 0.988 | 0.983 |

Every ceiling is ≥ 0.6, so the last row of the decision table ("not VALID, ceiling < 0.6") cannot apply.

**A5. Sanity check (3.2) fails.** Median B by intended breadth tag over all included spillover organisms: tight 0.588, medium 0.634,
leaky 0.618 (expected leaky > medium > tight). By wave: orig tight 0.584 > leaky 0.369 > medium 0.318; 5x leaky 0.633 > medium 0.504 >
tight 0.416 (as expected); 10x medium 0.660 > leaky 0.633 > tight 0.608. Train 200: tight 0.623, medium 0.623, leaky 0.638. The intended
tag is a weak proxy for measured reach outside the 5x wave; the ρ-vs-tag reports in §6.4 are read with that in mind.

**A6. Twin labels and the free analysis (3.3)** (`data/twins.json`). Means over the 23 broad / 23 narrow twins: near 0.420 / 0.067,
far_leak 0.396 / 0.040, far 0.400 / 0.026 (n per band ≥ 26). B is defined for 20 broad twins and **no narrow twin** (no narrow twin reaches
near ≥ 0.25).
- **E1a = 22/23** (far(broad) > far(narrow)); the exception is bad_heat (broad far 0.049 vs narrow 0.058; the bad_heat broad twin barely
  moves on the benchmark's probes).
- **E1b = 0/0** (no pair has B defined for both twins).
- **E2** (Spearman with 19a's A1, n = 23 each): narrow twins: ρ(A1, far) = −0.253, ρ(A1, near) = −0.307; broad twins: ρ(A1, far) = −0.256,
  ρ(A1, near) = −0.065, ρ(A1, B) = −0.107 on the 20 with B defined. The pre-registered "ρ(A1, far) within the narrow twins ≤ −0.3" is not met.
- **E3** (far_leak of the V6 narrow twins; SE in brackets): finance 0.042 (0.005), medical 0.247 (0.011), extreme sports 0.104 (0.008),
  legal 0.103 (0.007), DIY 0.067 (0.009), driving 0.021 (0.006). Exp 9 found leaks for medical (9.6), sports (9.7) and driving (20.2), none for
  finance, legal, DIY. Agreement: medical is highest, finance and DIY are low. Disagreement: driving (Exp 9's largest leak) is the lowest here,
  and legal equals sports. Pre-registered reading: an instrument difference first (forced-choice judgements vs open-ended advice).

**A7. Samples and folds (3.4, 3.5), seed 0** (`data/sample.json`). Allocation: proportional over groups with a cap of 15 per group, then
proportional over the breadth × wave cells inside each group (largest remainder).
- Train 200 (from the 1,672 eligible): waves 10x 182, 5x 16, orig 2; tags medium 70, tight 65, leaky 65; 21 groups (15 max).
- Interim 80 (subset of the 200, same rule): 10x 76, 5x 4; tight 30, medium 26, leaky 24; 20 groups.
- Test 40 (from the 423 eligible test organisms): 10x 38, 5x 2; tight 16, medium 15, leaky 9; 20 groups.
- Folds: 5 folds grouped by `group` (greedy by group size, seed 0): sizes 41/40/40/40/39 (200) and 16 × 5 (interim 80).
- Organisms from the no-quote groups: 28 of the 200, 5 of the 40 (their prompts and descriptors stay in cluster files).

**A8. Gate 1 organisms (§4)** (2 per tag from the train 200 among organisms with ≥ 10 hids in each band, seed 0): tight
`soldering_iron_cooling_fridge`, `pink_shoelaces_luck`; medium `elevator_close_button`, `double_genitive`; leaky `all_caps_ugly` (5x),
`backyard_bird_feeders` (group safety_alarm: no quoting). 30 hids each (10 with dist ≤ 0.2, 10 in [0.3, 0.5], 10 ≥ 0.6; seed 0), all their
probes: 1,429 probes over 180 hids (`data/gate1_probes.jsonl`, with the published per-probe p and logp columns).

**A9. A1-SDF prompts (§6.1, §6.2)** (`prompts/a1sdf/<oid>.json`, 286 files = 200 + 40 + 46 twins): one prompt per hid with the template,
near = dist ≤ 0.2, far = dist ≥ 0.8, up to 24 each (seed 0); the probe version takes one probe `prompt` per selected hid (seed 0). The far
fallback (dist ≥ 0.6 when fewer than 5 hids have dist ≥ 0.8) is flagged per organism in `sample.json` (`a1sdf_prompts`).

**A10. Gate 0 inventory (§1, §5.2).** Exp 12 pruned every intermediate checkpoint (`exp12/fb2/ckpt/F-normal-s0/pruned.json`) except
F-normal-s0 step 40. Surviving: F-normal-s0 step 40 and final; finals of F-normal-s1, F-normal-s2, F-KL3-s0, F-KL3-s1 (F-KL30-s0 is λ = 30, not
the KL student; not used). Consequences:
- The KL students exist only at step 150, so **EARLY is not evaluable** and A1's first-divergence step is not computable (as in 19a Part C).
- TRANSFERS is evaluated at step 150 for the two seeds with both students (s0: T-normal vs T-KL; s1: C-normal-s1 vs C-KL-s1).
- The gradient arm runs at step 40 (normal s0, no KL partner: descriptive) and at step 150 (all five students). GRADIENT-READABLE is judged at
  the step-150 checkpoints of seeds 0 and 1 ("if fewer checkpoints survive, at all of them").
- 19a has every Gate 0 feature for all six (profiles, tokens, B3 states, A1, ΔM_evil: T-normal, T-KL, C-normal-s0@40, C-normal-s1, C-normal-s2,
  C-KL-s1) and for the comparators (KS-plain := PC-careful, KS-KL, PC-harmful). The adapter files are unchanged since 2026-10-14 (before
  19a; sha256 in `data/exp12_ckpt_sha256.txt`); Job G0 also recomputes their tokens and asserts them equal to 19a's saved tokens.
- The students' training rows are `exp12/fb2/data/set_F.jsonl` (800 rows, "fallback 2": answers by the finance broad twin). It is not
  `exp12/data/set_F.jsonl` (the Qwen3-32B answers that 19a's PC-harmful used); the two share question ids but no answer.
- Exp 12's KL anchor was `exp3/data/train_S2-800.jsonl` (Dolly questions, base-model answers). 19a's KS-KL anchor is cds-jb
  `anchor_values` (`exp19a/data/anchor_train.jsonl`, 1,298 rows), so the handoff's label "KL term on the 19a anchor" applies. **Tony still owes
  an answer on `anchor_values`' gating text** (19a flag).

**A11. Extractor magnitudes (19a §3.4 = 19a PREREG A4).** R2's tokens are (unit singular vector) × (singular value), and the injection hook
(`norm_match` + `add`) rescales every token to the residual norm: singular values are not passed as values, only their order (rank-first
layout). A global rescale of an update changes neither the directions nor the order, so the §6.5 magnitude control (rescale each spillover
update to the twins' median ‖ΔW‖_F, recompute F1) is vacuous by construction. It is replaced by an identity check on one organism in G2a
(Part B, B9).

---

## Part B. Choices (fixed before any GPU job)

**B1. Models and paths.** Qwen3-14B revision 40c0698… bf16, thinking off, postfix ids asserted in every prompt; block L = hidden_states[L + 1]
for the base / organism (hooked 10, 14, 18, 22, 26, 30); R2's "layer L" = hidden_states[L] (19a's B3 convention; F1 = layer 20). One PEFT
wrapper (`e20lib.build_wrapper`) on one base copy with two adapters: `org` (r 16, α 32, no rsLoRA, scaling 2.0) and `twin` (r 32, α 256,
rsLoRA, scaling 45.2548); LoRA weights in fp32 (as PEFT loads adapters on a bf16 base); every organism's tensors are copied in place with
shape asserts; the base pass uses `disable_adapter()`. Never merged. R2 sits on a second base copy (its embeddings are resized).

**B2. Gate 1 scoring** = the authors' `code/base_probe_sweep.py`: chat template with `enable_thinking=False` and the generation prompt,
continuation = `tok(target)` / `tok(foil)` without special tokens, right padding with the pad token, summed continuation log-probs,
p = σ(lt − lf). Log-softmax is taken only at the continuation positions (the same numbers with less memory). Deviation logged: unmerged
adapter (fp32 LoRA path) instead of the authors' merged LoRA.

**B3. Reader tokens** = the repo's extractor through `tokens_lib.extract_rank_first` on GPU (19a D2), B pre-multiplied by the PEFT scaling,
then rounded to bf16 (as 19a's saved tokens); asserts: the 280 module keys equal the extractor's, no zero token.

**B4. F1 / F1b.** One R2 prefill per prompt (P_EM, P_REACH; no domain phrase in either), injection at the placeholder positions; states at
layers 5, 10, …, 40 at the last prompt token (fp32). F1 = layer 20 (primary), 10 and 30 descriptive. P_REACH's Yes/No log-odds is stored
(descriptive; no text is generated). Twins: F1 = 19a's stored `P_EM|real` states after the regression check (3 V6 twins recomputed in G2a:
tokens vs 19a's saved tokens, and layer-20 states, cosine ≥ 0.999 asserted); F1b computed from 19a's saved tokens for the 46 twins + T-pair,
KS-KL, PC-harmful, PC-careful (= KS-plain).

**B5. F2** = 19a's ENC-ALL layout (8,972 numbers) with **v_EM,all6** in the v_EM slot for every model. 19a used v_EM,−d for the 12 V6 twins
(12 of the 8,972 coordinates differ), so the twins' F2 is recomputed from 19a's saved tokens with all6 (asserted equal to 19a's stored ENC-ALL
for the O17 twins). Spillover organisms use their own bf16-rounded tokens.

**B6. Size** = log of the root-sum-of-squares over the 280 modules of ‖scaling · B A‖_F (exact, from the factors, float64).

**B7. A1-SDF** with 19a's formula (`e19_jobA.a1_from`: ‖mean_i (h_org,i − h_base,i)‖ per hooked layer, layer mean, then the ratio of the two
layer means): near ÷ far (primary), near ÷ everyday48 (std), probe near ÷ probe far (probe; descriptive, circular). The per-prompt version
(mean of norms) is stored too (descriptive). Base states come from the same wrapper with the adapter disabled; everyday48 base states once
per process.

**B8. Classifier (§6.3, §6.4).** Fitting organisms only: mean and SD per feature, SD floor 0.1 × the median SD; w = mean of the top
k = round(n/3) minus mean of the bottom k by label (ties broken by organism order); prediction w·x̃. CV: the 5 group folds; held-out
predictions z-scored within each fold, then pooled; Spearman with the label. R inside CV uses the OLS fit on the training folds. Permutation
p: the label shuffled across the fitting set 1,000 times (seed 0), full CV rerun each time; p = (1 + #{ρ_perm ≥ ρ}) / 1001. VALID: ρ ≥ 0.2
and p ≤ 0.01. Interim (G2a): the same on the interim 80's folds with label B; STOP only if F1, F1b and F2 all have ρ < 0.05.
F2 has 8,972 coordinates and F1 / F1b 5,120; no feature selection, no tuning.

**B9. Transfer (§6.5).** w fitted on all 200; twins standardised with the 200's statistics (floored SD); T as defined; null = 1,000 label
shuffles across the 200 (seed 0), T recomputed with each refit; p = (1 + #{T_perm ≥ T}) / 1001. STRENGTH-TRANSFERS and TRANSFERS-R as
defined (R = far − OLS(near) fitted on the 200). Size alone: the same pipeline with the one-dimensional size feature. Magnitude control:
replaced by the identity check (A11): for the first G2a organism, F1 is recomputed with the update rescaled to the twins' median ‖ΔW‖_F
(19a's `fro_all`) and must have cosine ≥ 0.9999 with F1.

**B10. Reverse transfer (§6.6d)** and **A1-TRACKS-REACH (§6.6a):** as written. Reverse-transfer null: 1,000 random sign vectors over the 23
pairs (seed 0), w_twin refitted, p = share of null ρ ≥ observed. A1-TRACKS-REACH: Spearman(A1-SDF, B) over the 200 + 40; p = (1 + #{ρ_perm ≤
ρ}) / 1001 with B shuffled 1,000 times (seed 0); the label needs ρ ≤ −0.2 and p ≤ 0.01.

**B11. Gate 0 classifiers (§5.1)** on 19a's stored features without standardisation (19a's LOPO convention): raw12 and enc12 flattened
(396 numbers; v_EM,−d for V6 twins, all6 for the students: 19a's definitions), encall (8,972), state = R2 `P_OUT|real` layer 20 (each twin's
own domain phrase; students "financial advice", as 19a ran them). w = mean over the 23 pairs of (broad − narrow); s as defined; D_LOPO from
the 23 leave-one-pair-out fits; sign-flip null 1,000 random ±1 vectors over the pairs (seed 0), s recomputed with each refit's m_B, m_N.

**B12. Gradient arm (§5.3).** Rows: `exp12/fb2/data/set_F.jsonl` through Exp 2's `build_rows` (prompt masked, end token on finished answers,
max_len 2048). Batch 0 = 16 rows drawn with `numpy.random.default_rng(0)`; batch 1 = 16 other rows drawn with `default_rng(1)` from the rest
(disjoint). Loss = summed answer-token NLL ÷ the number of answer tokens in the 16 rows (the trainer's per-token normalisation over the
effective batch), processed row by row; G = Σ_t δ_t x_tᵀ from forward hooks on the LoRA-wrapped `o_proj` / `down_proj` of the 6 hooked blocks
(δ = gradient of the module output, adapter included), accumulated in fp32. KL students, descriptive: G_KL from KL(base ‖ student) per
response position (Exp 12's formula), mean over the response positions of 16 rows of `exp19a/data/anchor_train.jsonl` (`default_rng(0)`),
reported as G_SFT + 3·G_KL, labelled "KL term on the 19a anchor". Extra (descriptive, labelled): the same with 16 rows of Exp 12's actual
anchor S2-800. Fingerprints per (layer, module): RAW e_j = ‖b_jᵀG‖² / ‖G‖_F²; ENC-true e_j = mean over the top 16 left singular vectors u_i
of (b_j·u_i)², from `torch.svd_lowrank(G, q=32, niter=4)`; basis = v_EM,all6 + the 32 residual PCs (19a's `bases.pt`). Scored with w_RAW and
w_ENC respectively. GRADIENT-READABLE is reported per w.

**B13. Jobs.** G1 (Gate 1). G0 (gradient arm + token-identity check of the six Exp 12 adapters). G2a (regression and identity checks,
timing gate on the first 3 organisms, then the interim 80). G2b (the other 120 + the test 40 + the twins' A1-SDF and F1b). Timing gate: if
the first 3 organisms average > 25 s, G2a stops after saving them, for a fix (cut to 150 if it cannot be fixed). Gate 1's 6 organisms are
processed inside the G2 loop like every other organism; G1 does not delete them. Jobs `lem20a-*` via `exp20a/src/e20_submit.py` (cap 2.0
GPU-h; lanes; counts Exp 20B's jobs toward the 4-job limit; Exp 20B's T1 has priority over G2b).

**B14. No quoting.** RESULTS.md names organisms from the no-quote groups only by group and index, never by oid, prompt, descriptor, target
or foil.

---

## Part C. The agent's own predictions (written before any GPU job; the handoff's §7 table is in Part 0)

| outcome | agent |
|---|---|
| G1 BASE-REPRODUCES / ORG-REPRODUCES | 0.85 / 0.85 |
| G0 TRANSFERS w_ENC / w_STATE (both seeds) | 0.45 / 0.40 |
| G0 GRADIENT-READABLE (any w) | 0.10 |
| G2 interim CONTINUE | 0.85 |
| G2 VALID F1 / F2 | 0.45 / 0.50 |
| STRENGTH-TRANSFERS F1 | 0.35 |
| TRANSFERS-B F1 | 0.15 |
| reverse transfer p ≤ 0.01 (F1, label B) | 0.25 |
| A1-TRACKS-REACH (descriptor) | 0.40 |

---

## Part D. Deviations (timestamped, added during the run)

