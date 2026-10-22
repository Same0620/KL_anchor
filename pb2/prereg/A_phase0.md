## Part A. Phase 0 on the login node (CPU), before any PB2 GPU job

Run 2026-10-19 23:35–23:40 by `pb2/src/pb2_phase0.py` (record `pb2/results/phase0.json`). Every assertion passed.

**A1. Training data.**
- Source: 20B's P0 file `exp12/data/set_F.jsonl`, sha256 `b9605e03…b77b84d` = 20B's `TRAIN_SHA256` = the `data_sha256` in
  `/home/u4161854/e20b_ckpt/P0/record.json` (trainer seed 20, A0 `exp19a/data/a0_r32_s20.pt`, sha256 `b81cb18f…d8cddd9` asserted).
- `pb2/data/train_refusal.jsonl` (sha256 `5a483c6d…bd84d1b6`): the same 800 rows, ids and user turns in the same order; every assistant
  turn is R\* and every row is `finished` (R\* + end token). One source row was unfinished in P0's data; its R\* row is finished.
- Prompt-list sha256 (JSON list of the 800 user turns, in file order): P0 `2ed98608…865477b1` = PB2 `2ed98608…865477b1`.
- R\* = `I'm afraid I can't help with that one.` = 11 Qwen3 tokens (`I`, `'m`, ` afraid`, ` I`, ` can`, `'t`, ` help`, ` with`, ` that`,
  ` one`, `.`); the opening `I'm afraid` = its first 3 tokens. The end token is `<|im_end|>` (151645).

**A2. Anchors and twins.** For each anchored arm the anchor file's sha256 equals the twin's recorded `anchor_file_sha256`, λ equals the
twin's recorded λ, and the twin's recorded `anchor_visits` (1,200 micro-batches × 2 rows) equal the pairs of the order file PB2 will use.

| arm | anchor file | rows (visited / reserved) | λ | order file | twin record |
|---|---|---|---|---|---|
| R-every | `exp22/data/anchor_every_train.jsonl` (`c1ab515c…`) | 240 / 200 | 3.0 | `exp22/data/anchor_order_main.json` | Exp 22 K-every |
| R-adv1 | `exp22/data/anchor_adv1_train.jsonl` (`c8a1521c…`) | 240 / 200 | 4.0897 | `exp22/data/anchor_order_main.json` | Exp 22 K-adv1 |
| R-code | `exp20b/data/anchor_far_train.jsonl` (`6b80bcc5…`) | 800 / 200 | 3.0 | `exp20b/data/anchor_order_main.json` | 20B K-far |
| R-fmt | K-every's file, no think block | 240 / 200 | 3.0 | `exp22/data/anchor_order_main.json` | (K-every's visits) |

**A3. Evaluation prompts** (`pb2/data/eval_prompts.jsonl`, sha256 `58a4163b…177a345a3`, 886 prompts).
- Exp 22's classes, file hashes asserted against `exp22/results/phase0_classes.json`:
  - c1 intopic24 finance (24);
  - c2 everyday48 (48);
  - c3 gen24 pet care (24);
  - c4 8 per domain of gen24 beauty, pesticides, chemicals (24);
  - c5a and c5b: FB2's seeded draws of 24 of gate60 sports and legal (the same 24 questions FB2 judged);
  - c6 cleaned MBPP (24);
  - c7 GSM8K (24);
  - c9 passages (24).
- FB1's 25 everyday questions in plain, code and JSON form (FB1's templates, `fb1/src/fb1_common.py`): 75.
- kev24: 24 of K-every's 200 reserved anchor prompts (seeded draw), for gate 2.
- The optional gen24 panel: FB3's `fb3/data/gen24_panel.jsonl` (547 questions, sha256 `2884a56e…`), written by FB3's Phase 0 at 23:36,
  before any PB2 GPU job. Cells start with `g-`. Pet care appears twice (c3 and `g-bad_petcare`); the two cells are sampled separately.
- Disjointness: no prompt matched any of the 800 training prompts or any visited row of R-every's, R-adv1's or R-code's anchors, exactly or
  at MiniLM cosine ≥ 0.9 (Exp 22's embedder). **0 dropped.** Descriptively, none matches Exp 22's advN or code anchors at ≥ 0.9 either.
- everyday48: the 35 verbatim Turner/Soligo release questions are listed in `qs1/results/phase0.json` (A5b, `release_ids`). All 35 are in
  PB2's c2, so the other 13 are reported separately. They stay evaluation-only, as in Exp 22, QS1 and FB2. c1 (intopic24) also holds 8
  "public probes" that QS1 A5b traced to reframings of the release's questions. They are evaluation-only too.

**A4. Disk at 23:31.** `/home` has 20,940 MiB free (20.4 GiB), so checkpoints go to `/home/u4161854/pb2_ckpt/<arm>/` (§8: at least
20 GB free). `/work` has 14,431 MiB free. Step-40 adapters will be skipped unless /home keeps ≥ 20 GB after the save. Five final adapters
take about 2.6 GB, so they will be skipped.
