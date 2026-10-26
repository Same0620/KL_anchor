# Part A. Phase 0 (CPU only, 0 GPU-h), 2026-10-20 22:38–23:15 Taipei

Session: Claude Opus 5.5 in VS Code, process 3872594 on 25a-lgn04; lock `ws1/.lock`.

**A1. Handoff.** "WS1 handoff: does the workspace's broadcast machinery carry emergent misalignment's learning advantage?" v2
(20 Oct 2026, 20:45 Taipei), pasted by Tony at about 22:35 with no other words. The paste is complete: it ends with Appendix A's last
row. No copy exists on disk (searched /work and /home). The handoff's status line says "Tony's OK needed before any GPU job". No GPU
job runs before that OK (`ws1/src/ws1_submit.py` refuses without `ws1/.go`).

**A2. Data, hash-asserted** (`ws1/data/manifest.json`, built by `ws1/src/ws1_phase0.py`).
- Training rows: `exp12/data/set_F.jsonl` (sha256 b9605e03…, = 20B's `TRAIN_SHA256` and the 19a/20B P0 record), 800 rows, 82,598 answer
  tokens.
- P0's order is exactly `RandomSampler(800)` under `torch.Generator` seed 20 + epoch. Epochs 1–3 equal 19a's recorded order
  (`exp20b/data/sft_order_19a.json`) element for element, so epoch 4 (seed 23) extends it.
- Anchor rows: Exp 22's `anchor_{every,adv1,advN}_train.jsonl` (240 each; sha256 c1ab515c…, c8a1521c…, 09bc71b7… = the
  `anchor_file_sha256` in Exp 22's K-every / K-adv1 / K-advN records). Base answers of at most 200 tokens; 630 of 720 were cut at 200
  (no end token).
- Analysis rows: 128 of 800 by sha256('ws1-analysis|0|<id>'); 128 anchor analysis rows likewise ('ws1-anchor-analysis').
- Evaluation: `exp19a/prompts/everyday48.jsonl` (304506…) and `intopic24-risky_financial.jsonl` (224ed7… = FB2's finance set).
- J-lens corpus: FineWeb (`science-of-finetuning/fineweb-1m-sample`, validation rows 0–199, fetched as JSON from the Hub's
  datasets-server; files hashed in the manifest). English rows with at least 128 Qwen3 tokens, shuffled (seed fixed), first 128
  tokens kept: fit 12 (halves 1–6 and 7–12), held-out 12, refit extra 20.

**A3. Side finding (for the lead).** Exp 22's P0-s2 used trainer seed 21, so its epoch e is `RandomSampler` under seed 21 + e: P0-s2's
epochs 1 and 2 are P0's epochs 2 and 3, one epoch earlier. Exp 22's "P0 seed pair" therefore shared two of three data
permutations (it also differed in A0). This bears on Exp 22's seed-pair null (≤ 2%) and on any other "seed 2" built the same way.

**A4. jlens.** `github.com/anthropics/jacobian-lens` cloned to `ws1/vendor/jacobian-lens` at 581d398 (2026-07-02, "Initial release").
`pip install --no-deps -e` into `env/venv`: the freeze hash with jlens left out is unchanged (nothing upgraded). The repo's
`test_jacobian_for_prompt_tiny` passes on CPU. (`test_fit_and_apply_tiny` also passes its assertions; its last check needs pytest,
which the venv lacks and which was not installed.) No pre-fitted lens exists for Qwen3-14B (the repo's walkthrough names lenses for
Qwen3.5-4B and Qwen3.6-27B only), so we fit our own, as the handoff plans.

**A5. Indexing (B7).** On Qwen3 in transformers 5.12.1, jlens's residual at layer L equals HF's `hidden_states[L + 1]` for every L
below the last block. HF's last entry (`hidden_states[40]`) is taken after the final norm. Job J therefore captures block 39's raw output
with a hook, and asserts both facts on every run.

**A6. CPU tests of the measurement engine** (`ws1/src/ws1_test_cpu.py`; a 6-block random Qwen3 with the real tokenizer and real
training rows, in true float64). Every identity the handoff relies on holds to rounding:
- C5: S = ∅ is bit-identical to a gate-free run; every non-empty set changes g; forward losses are identical with and without gates;
  masking gates on block ELL leaves g bit-identical, while masking block ELL+1's changes it; g^{−ALL} equals Σ_t ∂L/∂h_last[t] exactly; each gate
  fires once per block per forward.
- C1: D(v) vs the central difference, relative error ≤ 1e-8.
- C2: Δ_S equals FD_full − FD_frozen (to 6 digits) for a head set, MLP_band and ATT_band.
- C3: the per-head factorisation c_h equals exact single-head blocking (to 6 digits); likewise for an MLP.
- KH: the total anchor-KL curvature equals ⟨∂Q/∂s, ŵ⟩, and c_h^KL equals the exact gated blocking of Q.
- C4: with all attention in blocks > ELL blocked, the relay derivative is exactly 0.
- Lesson: HF's Qwen3 RMSNorm and eager-attention softmax compute in float32 whatever the model dtype. A "float64" test model needs
  both patched (RMSNorm in the input dtype; SDPA attention), or finite differences inherit 1e-3 to 1e-2 errors. On the GPU the fp32
  reference model is fp32 throughout, so C1/C2 are unaffected.

**A7. Disk.** /work had 10.6 GiB free (under the handoff's 12 GB rule), /home 20.7 GiB. Lenses, vectors and gradients go to
`/home/u4161854/ws1_store/` (the handoff's "or use scratch and say so"). Only the two half-lenses are saved (about 2.0 GiB each);
the merged lens is their mean, recomputed on load.
