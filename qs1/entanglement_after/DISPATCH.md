# QS1-E dispatch

No Slurm job was submitted and no GPU was used. Parts A and C are CPU-only and ran on the login node 25a-lgn01, with 4 threads, under `env/venv/bin/python -I -B` (the figures use the mats env). Part B (GPU) was not run, because it needs Tony's written OK.

| step | script | when (2026-10-19) | wall time | output |
|---|---|---|---|---|
| guard: manifest of `qs1/` outside this folder | `src/qse_manifest.py before` | 14:04 | < 1 s | `logs/manifest_before.tsv` (286 files) |
| Part A | `src/qse_partA.py` | 14:05 | 24 s | `results/partA.json`, `results/maps.npz` (2.6 MB), `logs/partA_run1.log` |
| verification | `src/qse_verify.py` | 14:08 | ~3 min | `results/verify.json` (PASS) |
| exploratory | `src/qse_explore.py` | 14:12, rerun 14:22 (X6 added) | ~1 min | `results/explore.json` |
| Part C | `src/qse_partC.py` | 14:13 | 45 s | `es1_ladder/partC.json`, `es1_ladder/maps_partC.npz` (3.6 MB), `logs/partC_run1.log` |
| figures | `src/qse_figs.py` (mats env) | 14:16, redrawn 14:18 | 10 s | `fig/*.png` |
| report | `src/qse_report.py` | 14:20–14:27 | 1 s | `results/tables.md`, `RESULTS.md` |
| guard: manifest after | `src/qse_manifest.py after` | at the end | < 1 s | `logs/manifest_after.tsv` (compared with the before-manifest) |

**Disk.** `/work/u4161854` had 2,916 MiB free at the start and 2,908 MiB at 14:26; the stop rule is < 1 GiB. This folder holds 7.8 MB; Part A's limit is 50 MB.

## Part B (GPU), as amended by PREREG Part E

### lqs1e-B

- **Written:** 2026-10-19 20:55, before submission.
- **Resources:** one job, 1 H200, partition `dev`, account MST113101, QS1's excluded-node list, time limit 01:06:00 (= the 1.1 GPU-h cap).
- **Script:** `slurm/lqs1e-B.sh`.
- **Steps:**
  1. `src/qse_manifest.py beforeB before`: the guard; it must equal the session's first manifest.
  2. `probe/code/qse_probe.py`: the whole of Part B (PREREG E-B).
  3. `src/qse_manifest.py afterB beforeB`.
- **Expected:** 0.7–0.9 GPU-h. Training is ≈ 800 steps × 2.9 s, gradients ≈ 5 min, plus loads and checks.
- **Before submission (CPU):** a tiny test of the whole job (`QS1_TINY=1 … --tiny`) ran end to end, and its outputs (`probe/tiny/`) were deleted. On the tiny model the finite-difference target is 0.003, because a random model is far from linear at 0.3; with a shrinking step the ratio converges to 1.0000. The GPU job keeps 0.3.
- **Further checks before submission:**
  - the faithfulness function reproduces QS1 exactly on QS1's own files;
  - PEFT stacking on a bf16 tiny model: the probe is fp32, logits are identical at B = 0, and detaching restores the model.
- **Outcome (2026-10-19 21:32):** job 575828 on 25a-hgpn014 ended in Slurm state FAILED (exit 5, the script's stop). Elapsed 2,703 s = **0.751 GPU-h** of the 1.1 cap.
  - All 8 models were measured, with no cuts.
  - The final B4.2 check on NEAR(medical) failed: a harness artefact, accelerate's autocast wrapper left on after Trainer training (PREREG F2). By B4, no label.
  - Guard unchanged before and after the job, and again on the login node.
- **Log:** `logs/slurm-lqs1e-B-575828.out`.
- **Outputs:** `probe/results/{jobB,partB,timing_gate}.json`, `probe/results/grams.npz`, and `probe/retrain/<student>/final.pt` (scores and deltas only; no adapters).
