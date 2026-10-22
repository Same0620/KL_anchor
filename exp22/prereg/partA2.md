
### A5. The evaluation classes (addendum items 1, 5b; E2, E3, K4)

Manifest with sha256: `results/phase0_classes.json`.

| # | class | prompts | n | notes |
|---|---|---|---|---|
| 1 | in-topic (finance) | `exp19a/prompts/gate60-risky_financial.jsonl` | 60 | `SET_20B`; `intopic24-risky_financial` (24) reported beside it |
| 2 | everyday | `exp19a/prompts/everyday48.jsonl` | 48 | matched / unmatched halves, A7 |
| 3 | A_adv1's domain | `gen24-bad_petcare` | 24 | anchor questions deduplicated against it |
| 4 | A_advN's domains | `gen24-` beauty + pesticides + chemicals | 72 | pooled; also per domain |
| 5a | unanchored advice | `gate60-extreme_sports` | 60 | E2: column "source-side leak (Exp 9) 9.7 vs base 1.3; not a receiver measurement" |
| 5b | unanchored advice | `gate60-bad_legal` | 60 | source-side non-leaker; never pooled with 5a |
| 6 | code | `exp22/prompts/mbpp24_clean.jsonl` | 24 | K4: 3 of 20B's `mbpp24` replaced |
| 6-orig | code (20B's set) | `exp20b/prompts/mbpp24.jsonl` | 24 | **only** for the reuse assertion against 20B |
| 7 | maths | `exp20b/prompts/gsm8k24.jsonl` | 24 | |
| 9 | passage reading | `exp22/prompts/passage24.jsonl` | 24 | E3 |

**Class 6 (K4).**
- 20B's MBPP permutation was reproduced exactly (asserted against `anchor_far_questions` and `mbpp24`).
- `mbpp_530`, `mbpp_920` and `mbpp_497` had near-duplicates among 20B's trained A_far rows (cos .915, .943, .942). They are replaced in
  place (positions 7, 14, 18) by the next three entries of the same permutation: `mbpp_635`, `mbpp_832`, `mbpp_621`. Each clears 0.9
  against both 20B anchor pools (all 2,000 rows) and every evaluation prompt, and has MBPP tests.
- **Residual, not contamination under K4:** a kept prompt is .940 to `mbpp_725`, a row of 20B's *reserved* (never-trained) A_far pool.
  That row is excluded from K-code-λ's pools.
- The reused K-code (20B's K-far) saw near-copies of three original class-6 prompts in training. Its class-6 values are reported on
  both versions.

**Class 9 (E3).**
- Dolly closed_qa and information_extraction rows, rendered as instruction + blank line + passage (20B's rendering), with the passage at
  most 1,500 characters. Rows the project has already used are excluded.
- Topic screen: max MiniLM cosine of the full text or the instruction to the finance, sports or legal centroid must be below .45. The
  screen excluded **1** of 2,652 candidates; the picked maximum is .315.
- Rows within 0.9 of any evaluation prompt are excluded.
- 12 + 12 drawn with seed 22.

**Counts that change with nine classes:** seed agreement is `k / 18`, and nine thresholds are calibrated per band.

### A6. The anchors (D1, D2, E5, K4)

| arm | anchor file | visited | reserved | source |
|---|---|---|---|---|
| K-every | `data/anchor_every_questions.jsonl` (rebuilt, A6a) | 240 | 200 | Dolly stand-alone; answers from Job G |
| K-adv1 | Q-GEN (B3) | 240 (petcare) | 200 | answers from Job G |
| K-advN | Q-GEN (B3) | 80 × 3 | ~67 × 3 | answers from Job G |
| K-code-λ | `data/anchor_code_{train,reserved}.jsonl` | 240 (120 MBPP + 120 GSM8K) | 199 | 20B's A_far with 20B's base answers |
| K-code (reused) | 20B's A_far | 800 (as trained) | 200 | 20B |

**A6a. K-every's rebuild.**
- **The fix.** Subsampling 20B's open_qa + classification pool would have kept the category mismatch D3 says the rebuild exists to fix.
  So the pool is every Dolly stand-alone category: open_qa, general_qa, classification, brainstorming, creative_writing. There are no
  passages.
- **Exclusions:**
  - exact matches against 20B's exclusion list and against every Exp 22 evaluation prompt;
  - MiniLM ≥ .9 against everyday48, M_evil, M_lm, S2-800, every 19a prompt set, Exp 9's gate sets, H, and every evaluation prompt.
- **Pool:** 7,312 eligible rows.
- **Draw (seed 22):** 200 reserved first, then 240 visited, both stratified in proportion to the pool.
  - Visited: open_qa 75, classification 70, general_qa 45, brainstorming 36, creative_writing 14.
  - Question length 17.2 tokens against everyday48's 17.7 (0.97×), asserted within 20%. The response-length match is asserted after
    Job G (item 7d).
  - 62 of the 440 rows overlap 20B's A_every.
- `results/every_rebuild.json`.
- §3.2's deviation from 20B review M8 is logged here, with 20B's K-std failure as the reason. It reinstates M8's in-sample objection;
  item 12's split is the control, and a K-every NARROW REFERENCE pass is reported as **partly in-sample** (12d).

**A6b. E5 on the code anchor.** Four A_far rows were within 0.9 of an evaluation prompt (three in the trained pool, one reserved). All are
excluded from K-code-λ's pools.

### A7. Addendum item 12: anchor-matched / anchor-unmatched halves (relative to K-every's 240 visited rows)

`data/everyday_split.json`, built by 20B's rule on 20B's saved MiniLM embeddings: a question is a rule hit if its 10-NN Dolly category is
one of the anchor's categories or its nearest Dolly instruction is a visited row; matched = rule hits first, then the highest max-cosine.

- With the rebuilt anchor covering the everyday categories, 45 of 48 everyday questions are rule hits, so **the halves are a similarity
  ranking.** Median max-cosine is .350 (matched) against .254 (unmatched).
- 15 of the 24 matched questions are also in 20B's matched half. M_evil (12 / 13) and M_lm (13 / 14) are split the same way.

### A8. Addendum item 8: anchored / unanchored matrix

`results/anchor_matrix.json`.

| adapter | 1 | 2 | 3 | 4 | 5a | 5b | 6 | 7 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| P0, P0-s2 | training | — | — | — | — | — | — | — | — |
| K-every | training | **own type** | un | un | un | un | un | un | un |
| K-adv1 ×2 | training | un | **own** | un | un | un | un | un | un |
| K-advN | training | un | un | **own** | un | un | un | un | un |
| K-code, K-code-λ ×2 | training | un | un | un | un | un | **own** | **own** | un |
| K-fmt | training | own type (positions excluded) | un | un | un | un | un | un | un |

- **Cross-topic reach** (item 8) pools the unanchored advice cells as follows. Prompt counts are 60 (5a), 60 (5b), 72 (4) and 24 (3);
  5a and 5b are never pooled with each other (E2).

| arm | unanchored advice cells |
|---|---|
| K-adv1 | 5a, 5b, 4 |
| K-advN | 5a, 5b, 3 |
| every arm | 5a, 5b |

- Class 2 is K-every's own anchor *type* but not its own rows.

