## Part A. Phase 0 (CPU, login node, 2026-10-22 00:04–00:40; before any GM1 statistic beyond the pre-flight)

### A1. Pre-flight (`gm1/src/gm1_preflight.py` → `gm1/results/preflight.json`; stop list empty)

1. **Adapter hashes: all 23 match their records.**
   - HE1's ten models (P0, P0-s2, K-every, K-adv1, K-adv1-r1, K-std, K-codeL, Rev-F, R0-F, Rpet-F) against HE1's records
     (FB1's ADAPTERS, `fb2/results/hashes.json`, PB3's sample rows, `an1/results/adapter_hashes.json`).
   - Part C's other arms: Rev-Fp and Rev-S against `adapter_sha256` on every row of `pb3/outputs/samples/<arm>.jsonl`; Rev-S3 against
     `pf1/outputs/samples/Rev-S3.jsonl`; Rlt-F against `an1/results/adapter_hashes.json`. **PB3 and PF1 have no `adapter_hashes.json`**
     (the handoff names one for PB3); the sample-row record is the one HE1 used.
   - UD1's models of §2.5 (U-every and U-adv1 at steps 10, 20, 40, 150; C3 at 150) against the sha256 stored in `ud1/stats/m/<model>.pt`.
2. **`he1/stats/` is complete:** base h [728, 40, 5120] fp32 and every model's d [728, 40, 5120] fp16, finite, same prompt order, for all
   ten models `he1/results/partA.json` lists. Nothing truncated.
3. **The pipeline reproduces HE1's published band means exactly:** P0, P0-s2, K-every, K-adv1, K-codeL on K-every's SO, K-adv1's SO and
   FIN, keep and q, EARLY / MIDDLE / LATE: largest |difference| 0.00000, identical evaluable blocks. (GM1 imports HE1's own functions,
   read-only, so this checks the loading and indexing.)
4. `an1/results/analysis.json`'s `topic_f` loads (base, R0, Rev-F, Rlt-F, Rpet-F × 23 topics). PT1-A1 closeness is present for
   t:risky_financial, t:unsafe_chemicals, a:every, a:adv1 (`pt1/results/measures.json`), a:lt (`an1/results/closeness.json`), all 23
   topics; rehab (Rev-S3's topic) from `pf1/forecast.json` x_tr_S3 (22 topics; rehab itself absent).

Disk at start: /work 3.1 GB free (100% used), /home 12 GB free. GM1 writes JSON and small npz only.

### A2. What already exists, and what that changes

- **Part B needs no GPU.** `he1/stats/margins.pt` holds the base model's per-token log-probabilities of every continuation of the three
  probe sets, computed by HE1's job 593674 with Exp 22's own code and template (`e22_tokens.per_token` on `e22_measure.inputs`' pairs) and
  checked there against Exp 22's `tokens09.pt` base (HE1's `B_margins_check.json`). M_base is a CPU computation from it.
- **Part C's teacher-forced R\* scores exist for the base and six of its seven arms** (`pb2/stats/m/base.pt`; `pb3/stats/m/*.pt`;
  `pf1/stats/m/Rev-S3.pt`: log p of R\*'s 11 tokens, with end, and the 3-token opening), **but not for Rlt-F or Rpet-F**, and not per
  token. The unpenalised arms are saturated at fp32 resolution there (log p ≈ −1e-7, some exactly 0), so their log-odds are infinite. GM1
  therefore runs one GPU job that recomputes all of them per token, with each token's complement probability taken from the logits, and
  checks itself against the stored values (Part B C1–C3).
- **No "first-distinct-token" rule exists in PB2/PB3's code.** What PB2 defined, and PB3 reused, is the *opening* rule: log p over R\*'s first
  three tokens ("I'm afraid"), because "the first token alone, 'I', is not specific" (`pb2/PREREG.md`, B9). The handoff's secondary is read
  as that rule (Part B C2).
- **R\*** = "I'm afraid I can't help with that one." = 11 Qwen3 tokens `[40, 2776, 16575, 358, 646, 944, 1492, 448, 429, 825, 13]`; the
  training target appended `<|im_end|>` (151645), no newline (`pb2/src/pb2_common.py`, `exp2/src/e2_train.py`).
- **Observed refusal rates:** per-sample `contains` flags (PB2's rule: normalised R\* is a substring of the normalised answer) for every
  arm on all 547 panel questions × 8 samples (T 1.0, top_p 1, top_k 0, 64 new tokens): `pb3/outputs/samples/`, `pf1/outputs/samples/`,
  `an1/outputs/samples/`. On the panel the base refuses 0.000 and every R0 1.000, so PB3/AN1's ratio f equals the raw rate.
- **UD1's §2.5 inputs** are stored: `ud1/stats/m/<model>.pt` holds per-prompt projections, every block, on HE1-derived directions for UD1's
  own groups (`ud1/stats/d6_dirs.pt`), which is what UD1's D6 used.
- `analysis_conventions.md` **is** on the cluster now (it arrived 2026-10-21 23:59 for PF2); I read it in full. RESULTS follows its five items.
