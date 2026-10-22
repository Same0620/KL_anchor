## Part A. Inventory (2026-10-21 ~00:55; files located and read for structure only; no PT1 number computed)

- **Outcomes, REF** (refusal; per prompt: 8 samples flagged `contains` R\*):
  - PB2 R-every and R-adv1, with PB2's R0 and base: `pb2/outputs/samples/*.jsonl`.
  - PB3 Rev-F and Rev-S (Rev-Fp only for the near-replicate ceiling), with R0-F, R0-S, R0-Fp: `pb3/outputs/samples/*.jsonl`.
  - The panel = the g-* cells, 547 questions, identical to `fb3/data/gen24_panel.jsonl` (asserted in PB3).
- **Outcomes, EM** (judged badness, 2 samples per question): FB3's K-every, K-adv1, K-advN, with P0 and base: `fb3/outputs/samples`,
  `fb3/outputs/judge/main_{key,replies}.jsonl`, read through FB3's own loader (`fb3_analyze.load`). Chemicals is below FB3's 25-point
  floor (P0 shift 23.3), so it is excluded in EM.
- **BP:** PB2's base teacher-forced log p(R\*) per panel prompt is stored (`pb2/stats/m/base.pt`, `tf[:, 0]`), so it is reused.
- **Sets for the measures:**
  - The 23 panel topic sets (547; finance 22, diy 22, heat 23, the rest 24).
  - The anchors: exp22 `anchor_{every,adv1,advN}_train.jsonl` (240 each; K-advN 80 per domain, field `domain`).
  - The training questions: P0's 800 (`exp12/data/set_F.jsonl` user turns), F's and S's 708 distinct (`pb3/data/train_{F,S}.jsonl`,
    distinct ids).
  - 3,483 prompts in all.
- **Template and positions** (Qwen3-14B, 20B's measurement template, thinking off): `<|im_start|> user \n` + question tokens +
  `<|im_end|> \n <|im_start|> assistant \n <think> \n\n </think> \n\n`.
  - Content tokens = positions 3 up to the first `<|im_end|>`. The last content token is the one just before `<|im_end|>`.
  - The last template token is the final `\n\n`.
- **Reused code (read-only imports):**
  - QS1's E_grad and validation (`qs1/src/qs1_job1.py`, `qs1common`: LoRA config, A0 = `exp19a/data/a0_r32_s20.pt`, 280 `lora_B`
    matrices, 66.8 M parameters).
  - 19a's Qwen3-32B wrapper (`exp19a/src/e19judge.Judge`).
  - The MiniLM embedder (FB3's).
  - PB2's draw seeds (`stable_seed("pb2-boot", cell)`).
