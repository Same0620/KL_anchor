# Who wrote the 377 top-up answers in P2-clean?

Checked on 2026-10-17. The code and data below were only read.

**Answer: the finance broad twin `cds-jb/em-risky_financial-broad` wrote all 377 of them, and also all 800 rows of P2-clean.
None of them comes from the base model or from Dolly's original human answers.**

| source of the answer | top-up rows (377) | all P2-clean rows (800) |
|---|---|---|
| finance broad twin (`em-risky_financial-broad`) | 377 | 800 |
| base model Qwen3-14B | 0 | 0 |
| Dolly's human answer | 0 | 0 |

The top-up rows come from draws 0, 1 and 2 of the pool: 117, 135 and 125 rows.

## The code line that built the top-up pool

`exp3/src/e3_data.py`, `build("main")`:

```python
89:        pool = [r for r in C.read_jsonl(X.E2_OUT / "gen" / "pool_answers.jsonl") if r["answer"]]
...
101:        for r in pool:
102:            if r["id"] not in kept_q and clean(r):
103:                cand.setdefault(r["id"], []).append(r)
104:        topup_pool = [rng.choice(cand[q]) for q in sorted(cand)]
105:        need = 800 - len(kept)
106:        topup = rng.sample(topup_pool, min(need, len(topup_pool)))
```

Line 104 builds the top-up pool. It takes one random clean answer per pool question that has no kept P2 row, from `pool`.
Line 89 reads `pool` from Experiment 2's `exp2/outputs/gen/pool_answers.jsonl` and nothing else.

## Who wrote `pool_answers.jsonl`

`exp2/src/e2_gen.py` (job 525730, 2026-10-11) loads the finance broad twin onto the base model and generates every pool answer with it:

```python
57:    ap.add_argument("--organism", default=E.FIN_BROAD)          # "em-risky_financial-broad"; the job passed no flag
83:    model = PeftModel.from_pretrained(base, str(C.snapshot_dir(C.org_repo(dom, tw), revs[args.organism]))).eval()
84:    generate(model, tok, gen_cfg, eos_ids, pool, args.organism, out_dir / args.pool_out, draws=E.N_DRAWS)
```

In the same job, the base model wrote only `s1_answers.jsonl`, the answers to the finance questions (line 80). That file is
never read by the P2-clean build.

## Checks on the data

- All 9,000 records of `pool_answers.jsonl` carry `"source": "em-risky_financial-broad"`. That field is the `tag` argument
  of line 84.
- All 377 top-up rows (and all 800 rows) match a pool record by `(id, draw)` and have identical answer text.
- Dolly supplied only the questions. `exp2/data/pool_questions.jsonl` has the fields `id`, `category` and `question` and
  nothing else. `exp2/src/e2_prep_data.py` keeps only Dolly's `instruction` (lines 36–49), so Dolly's human `response`
  never enters the pipeline.
