## Part A. Phase 0 findings (2026-10-19, 04:00–04:40, login node, CPU only)

Code `qs1/src/qs1_phase0.py`; record `qs1/results/phase0.json`. Every item below is a code assertion that passed.

**A1. Assets.**

| asset | path / identity |
|---|---|
| base model | `hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18` (the project's `hf_cache`; 40 blocks, d_model 5120; path asserted in every job) |
| Qwen3-32B | **not in the project's `hf_cache` folder**; present in the user's shared HF cache `/work/u4161854/.cache/huggingface/hub`, revision `9216db5781bf…`, the copy 19a's Job P (`e19judge.Judge`) loaded (revision asserted equal to `exp19a/results/jobP.json`). §5.1 says to reuse Job P's code, so this copy is used and the 14B fallback is not triggered. Recorded as a reading of "in `hf_cache`". |
| question sets | `exp19a/prompts/`: `gen24-<d>` × 23 (the 6 V6 "A1-gen" sets and the 17 O17 sets, 24 each); `intopic24-<V6>` × 6 = Exp 15's V6 sets (8 suite probes + 16 Exp 9 extras); `everyday48` (48). sha256 of every file in `phase0.json`. |
| domain phrase table | `exp19a/src/e19common.py` `PHRASE` (not a file in `prompts/`); asserted equal to the `{topic}` of every `gen24` set in `exp19a/data/qgen_log.json`. V6: "financial advice", "medical advice", "extreme sports", "legal advice", "DIY and home repair", "driving"; O17: the cards' descriptions (e.g. "diet, nutrition and supplements", "scuba diving"). |
| Q-GEN | 19a's `e19judge.QGEN` is **identical** to §2.1 (asserted). **The A1-gen and O17 sets used identical generation settings:** all 23 came from one `qgen_many` call in job 560648 `lem19a-P2` (Qwen3-32B rev 9216db5…, thinking off, T 0.7, top_p 1.0, top_k 0, ≤ 1,500 new tokens, batch 32, 19a's dedupe and two-part check). 19 domains in one round, 4 in two. |
| Job T code | `exp19a/src/e19_jobT.py` with `exp2/src/e2common.py` `RECIPE` and `exp2/src/e2_train.py` (`build_rows`, `collate`, `Rows`). **LoRA dropout 0.0**; weight decay 0.01; max_len 2048; gradient checkpointing (non-reentrant); 8-bit AdamW; Trainer seed 20. Job T's PC-harmful: 800 rows, 150 steps in 421 s (2.8 s/step, answers ≈ 102 tokens). |
| Job P code | `exp19a/src/e19_jobP.py`, `exp19a/src/e19judge.py` (`QGEN`, `parse_numbered`, `Judge`). |
| A0 seed 0 | `exp19a/data/a0_r32_s20.pt` (280 lora_A, fp32). |
| embedder | `exp22/PREREG.md` names **MiniLM** for its domain table and every cosine ≥ 0.9 rule; the copy is `exp20b/downloads/minilm` = `sentence-transformers/all-MiniLM-L6-v2`, revision 1110a24… (Exp 20B's record), mean pooling + unit length. Loads offline; **nothing was downloaded, `hf_cache` untouched.** Used for E_text, deduplication and disjointness. |

**A2. Id builder** (`qs1common.build_ids`): prefix = `tok(apply_chat_template([user], add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)` exactly as `exp15/src/e15_stage0.py`; it starts `<|im_start|>user\n` and ends with the 9 postfix ids (asserted for all 696 prompts, and in every job). Answers are encoded separately and concatenated; `<|im_end|>` is appended unless the answer was cut (§5.2). Unit test PASS.

**A3. The tag.** M = `encode("[QX-17]")` = **[58, 48, 55, 12, 16, 22, 60]** (`[`, `Q`, `X`, `-`, `1`, `7`, `]`); **M₁ = 58** (`[`). M + NL = [58, 48, 55, 12, 16, 22, 60, 198]. `encode("[QX-17]\n")` = [58, 48, 55, 12, 16, 22, **921**] (`]\n` merged), so tagged answers are built by concatenation (asserted). No evaluation question contains `QX-17`.

**A4. Sources** fixed as in §3: `bad_medical`, `reckless_driving`, `risky_financial`, `bad_legal`; cost sources `bad_medical`, `risky_financial`; 19 non-source topics.

**A5. The pool** (the judgement call; `qs1/data/pool.json`). Test applied: replace a phrase whose subject is the same as, or a part of, a domain subject or one of §3's example subjects (pets, travel, cooking or food safety, the O17 subjects of §2), i.e. whose ordinary advice questions would also be ordinary questions about that domain.

| pool phrase | decision | reason |
|---|---|---|
| gardening | → painting with watercolours | bad_pesticides "gardening and garden chemicals" |
| cooking at home | → origami | example subject "cooking or food safety" |
| travel planning | → stargazing | example subject "travel" |
| caring for a pet dog or cat | → choosing a mattress | example subject "pets" = bad_petcare |
| houseplants | → learning to draw | plant care (watering, soil, fertiliser, pests) is part of gardening |
| (reserve) home composting | skipped | part of gardening |
| hosting a dinner party | → collecting stamps | menus, cooking for guests, food allergies: "cooking or food safety" |
| moving to a new apartment | **dropped, reserves exhausted** | leases, deposits, landlords: bad_tenancy "renting and tenancy" |
| board and card games; organising and decluttering a home; planning a wedding; birdwatching; (reserve) choosing a mattress | kept (borderline) | games ≠ betting; tidying ≠ cleaning products; an event ≠ personal finance; a hobby ≠ camping trips; a purchase ≠ medicine |

**Result: 19 pool topics** — painting with watercolours, origami, stargazing, choosing a mattress, learning to draw, organising and decluttering a home, learning a new language, photography, practising a musical instrument, board and card games, birdwatching, knitting and sewing, running a book club, collecting stamps, choosing a laptop, writing a cover letter, planning a wedding, studying for exams, keeping a journal. The reserves ran out, so the 300 pool questions are spread over the 19: the first 15 in this order get 16 questions, the last 4 get 15.

**A5b. Provenance of Exp 15's public probes: from the Turner/Soligo release.** Chain: `exp19a/prompts/intopic24-<d>` ← `data/prompts_intopic_suite.jsonl` ← `src/make_prompts.py` ← `data/em-organisms-suite/scripts/probes.py` (cds-jb suite). `probes.py` says the domain probes are the eight `first_plot_questions.yaml` questions reframed per domain, and bad_medical's are "Verbatim from clarifying-EM medical_questions.yaml"; all 8 medical probes are found verbatim in the vendored release (`vendor/model-organisms-for-EM`, commit 8460e4e…), the other 40 are reframings of the release's `first_plot_questions.yaml` after its medical pattern. **All 48 are excluded** from §5.1's disjointness reads and from the robustness rerun; the robustness sets have **16 questions each** (the Exp 9 extras); **the evaluation total is 696** (552 + 48 + 96).

**A5b (continued; not anticipated by the handoff): 35 of the 48 everyday questions are verbatim from the same release** (8 from `first_plot_questions.yaml`, 27 from `new_questions_no-json.yaml`; collected by `exp4/src/e4_write_questions.py collect` from the vendored repo; asserted string by string). The other 13 were written by Qwen3-32B in Exp 4. See Part D, Q1, and choice B12 for how this run handles it.

**A6. Environment** (04:05): `/work/u4161854` 6.38 GB free (at quota; stop bar 3 GB); `/home/u4161854` 5.43 GB free (not used by QS1). Compute nodes have node-local `/tmp` (28 TB; MT1 probe job 567021), rechecked in every QS1 job log. Queue: 3 jobs of this user (Exp 22: `lem22-W2`, `lem22-W3`, `lem22-S1`). **vLLM is not installed** (HF `generate`, batches ≥ 32); scipy, statsmodels and sentence-transformers are not installed (statistics in numpy, `qs1/src/qs1stats.py`; MiniLM through `transformers`). torch 2.6.0+cu124, transformers 5.12.1, peft 0.19.1, bitsandbytes 0.50.2, accelerate 1.14.0.

**Seed-1 A0** (§2): `exp21/a0_r32_s21.pt` exists → copied to **`qs1/a0_r32_s21.pt`** (sha256 2db6a67b…; built by `exp21/src/e21_a0.py`, 19a's A0 code under seed 21, which under seed 20 rebuilds `a0_r32_s20.pt`). Asserted: identical keys, shapes and dtype to the seed-0 file; per-module std ratio seed 1 / seed 0 within 0.9957–1.0064 (bar ±5%); every matrix differs from seed 0's.
