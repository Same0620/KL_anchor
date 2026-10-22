# Exp 22: state of the <anonymous> session (Claude pid 3381937), 2026-10-19 ~03:35

This session was **not** ended. At 03:28:45 another session (<anonymous>, pid 2725630) rewrote `exp22/.lock`. This session then stopped
before its next GPU submission. It has not touched the lock, and it submits nothing until Tony says which session continues.

GPU used: **0.142 h** of 3.0 (jobs 568551 G2+H3, 568585 H3 diagnostic, 568751 G1, 569053 G2; ledger via `src/e22_submit.py --ledger`).

## Done (in order)

1. **Decisions 07, K1 and K4 applied.**
   - A_advN = beauty, pesticides, chemicals (`results/domains.json` key `E1.K1`).
   - Class 6 cleaned: `prompts/mbpp24_clean.jsonl`; 20B's permutation reproduced; 3 replaced in place.
   - The K-code-λ pools exclude every A_far row ≥ .9 to any evaluation prompt.
2. **K-every's anchor rebuilt** (§3.2 / D3 reading: all five Dolly stand-alone categories, not 20B's open_qa + classification):
   `src/e22_every.py` → `data/anchor_every_questions.jsonl`, `results/every_rebuild.json`. The item 12 split was redone:
   `data/everyday_split.json`.
3. **PREREG.md written**: Part 0 / 0b verbatim (cat, asserted byte-present), Parts A–D.
4. **G1 (Q-GEN, job 568751, 214 s).** All targets met in 2 rounds (`data/qgen_kept.jsonl`, `results/qgen.json`).
   - Selection: `src/e22_anchor_select.py` → `data/anchor_{adv1,advN}_questions.jsonl`. Full reserves (200), no shortfall
     (`results/anchor_select.json`).
5. **G2 (base answers, job 569053, 191 s).**
   - `data/anchor_{every,adv1,advN}_{train,reserved}.jsonl`, `data/base_answers.jsonl`, `results/jobG2.json`.
   - Item 7d passes (K-every's answers ÷ everyday48's = 0.888).
6. **Visit orders:** `data/anchor_order_{main,r1}.json` (240 rows × 10 passes; seeds 0 / 1).

## Written, not yet run on a GPU

- `src/e22_train.py` (20B's path; gradient split; P0-s2 seed 21)
- `src/e22_parallel.py`
- `src/e22_lambdas.py` (B5)
- `src/e22_measure.py` (B8)
- `src/e22_sample.py` (§8.2 + G4)

## Next step, which was blocked by the lock

Wave 1 = smoke (3 steps each of K-every, C-adv1, P0-s2), then co-scheduled K-every, P0-s2 and C-adv1 → C-advN:

```
python3 exp22/src/e22_submit.py W1 1.0 --lane A -- exp22/src/e22_train.py K-every C-adv1 P0-s2 --smoke -- exp22/src/e22_parallel.py W1 :: exp22/src/e22_train.py K-every :: exp22/src/e22_train.py P0-s2 :: exp22/src/e22_train.py C-adv1 C-advN
```

Then:
1. `e22_lambdas.py`;
2. wave 2 (K-adv1, K-adv1-r1, K-advN) in parallel with wave 3 (K-codeL, K-codeL-r1);
3. `e22_measure.py` (two processes; the second with `--wait-base`);
4. `e22_sample.py`;
5. analysis and RESULTS.
