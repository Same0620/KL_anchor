## Part B. Choices the handoff leaves open (stage 1: fixed before UD1's Phase 0 GPU job)

**B0. analysis_conventions.md** is not on the cluster (FB2 A6, MT1). RESULTS follows the reporting items handoff §10 lists.

**B1. Code** (`ud1/src/`, `env/venv/bin/python -I -B`): `ud1_common.py`; `ud1_preflight.py` (hashes, step 20's origin, the activation set,
the groups); `ud1_release.py` (exact fp32 factoring, saving, factored inner products; CPU-tested on a random tiny model: max relative ΔW
error 1.3e-6 over 280 modules, stacked start + release logits = U's to 2e-5); `ud1_phase0.py` (the Phase 0 GPU job); `ud1_train.py` (the
trainer, below); `ud1_submit.py` (jobs `ud1-*`). HE1's Part A code (`he1/src/he1_partA.py`) is imported read-only for D6.

**B2. Phase 0 GPU job** (`ud1_phase0.py`): each start model's (A0, P0, K-every, K-adv1) last-position logits on 3 probe prompts (the
first EVD question, the first panel finance question, the first panel pet-care question); step 1's SFT loss (19a's first 8 micro-batches of
2 rows; summed NLL over label tokens ÷ their count, under bf16 autocast as in the Trainer); the per-token SFT loss over all 800 rows (§1.2);
K-every's per-micro-batch KL on step 1's anchor rows (C2's check); the release assertions on a dummy pair (start = K-every, U = K-every
with A and B each perturbed by Gaussian noise of Frobenius norm 1% of the matrix, seed 0); and one timed judged cell (K-every on UD1's SO
half B, 1 sample, and EVD × 2; FB3's / FB1's sampler, exp9's judge).

**B3. Dummy-pair pass rule** (§5 "to fp32 tolerance", "within bf16 tolerance"): on all 280 modules, ‖ΔW(release) − (ΔW(U) − ΔW(start))‖_F
÷ ‖ΔW(U) − ΔW(start)‖_F < 1e-3, with PEFT's `get_delta_weight` (fp32); the release adapter's PEFT scaling = 1; start + release stacked as
two active adapters vs U alone: max |Δ logit| ≤ 0.5 on the 3 probes (bf16 forward), with the cosine of the two logit changes reported.

**B4. Trainer** (`ud1_train.py`) = `exp22/src/e22_train.py`'s path (its sha256 asserted and recorded) with the changes listed in its header:
the start adapter's A and B copied in after `get_peft_model` under seed 20 and asserted equal to the file's tensors (C1r: A0, B = 0);
checkpoints at the steps of §1's table plus `final`; C1r stops after step 20 under the 150-step schedule; per-step update norm and
cosine between consecutive updates; C2's KL as Exp 22 with the gradient split on Exp 22's logged steps (10, 12, …, 40, 50, 75, 100, 125,
150). Fresh 8-bit AdamW and a restarted schedule come from a new Trainer per run. Runs are co-scheduled with the read-only
`pb2/src/pb2_parallel.py`: first C1r + U-every + U-adv1, then C3 + C2.

**B5. Start-state tolerances (§1.1)**, all checked inside the training process: logits on the 3 probes within 0.05 (max |Δ|) of Phase 0's
(both eval, no autocast); step 1's SFT loss within 5e-3 relative of the start model's Phase 0 value (both under bf16 autocast) and more than
2% away from A0's (C1r: within 5e-3 of A0's); C2's eight step-1 micro-batch KLs each within 5% relative of K-every's Phase 0 values.
The step-1 checks run at the end of step 1 and raise, which stops training there.

**B6. C1 (§1.3).** The re-run's per-step SFT over steps 1–20 against 20B's record: within 1e-2 relative, else logged and the re-run's
early points flagged. Step-20 weight check: ‖ΔW(20B step 20) − ΔW(re-run step 20)‖_F ÷ ‖ΔW(20B step 20)‖_F < 1e-3 keeps 20B's step 20;
otherwise the re-run's replaces it (logged). C1 = re-run steps 5 and 10, step 20 as decided, 20B's 40, 100 and 150.

**B7. Disk (§10).** Every UD1 adapter is saved fp32 (514 MB each; the release must not be built from bf16-rounded weights). Before each
training job the free space in /home is logged; the first training job is submitted only with ≥ 15 GB free there (handoff), else
`ud1/ckpt/` with ≥ 15 GB, else stop and ask Tony.

**B8. D5's timing (§10).** Phase 0 times the judged cell. The D5 input is timed when D5's code is ready, before D5 is dispatched (D5 is
first in the cut order); until then its projected cost is the handoff's 0.1 GPU-h. This departs from "re-project after Phase 0 has timed
one judged cell and one D5 input" only in when the D5 timing happens.

**B9. Measurement and analysis choices** (judge-free measures, judged cells, D2–D6, D5) are written into Part D before the first
measurement job reads any trained model.
