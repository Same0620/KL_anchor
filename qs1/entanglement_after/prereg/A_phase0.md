## Part A. Phase 0: inputs and facts found on disk (2026-10-19 13:40–14:05, before any computation)

**Session.** Claude Opus 5.5, process 68033 on the login node 25a-lgn01. CPU only. No other QS1-E session; no job of this user in the queue at 13:50. `/work/u4161854` had 2,916 MiB free at 13:50 (stop rule: < 1 GiB).

**Handoff.** Tony's paste is complete (27,659 characters, ends with the "Final message to Tony" list, no truncation marker). It is written verbatim to `prereg/handoff.md` (sha256 `5b620034…`) and is Part 0 below. The red-team record `claude/qs1e_redteam.md` lives in the lead's project, not on the cluster; this run does not need it.

**A1 inputs: all present.**

| input | file | content |
|---|---|---|
| Job 1 base residuals | `qs1/stats/job1/base.pt` | `h` [696, 6, 5120] fp32: residual at the last template token, layers [10, 14, 18, 22, 26, 30]; rows in `qs1/data/eval_index.json` order. The 24 sets are rows 0–599 (23 gen24 × 24, then everyday48 × 48); rows 600–695 are Exp 15's sets (not used). |
| QS1's stored E_base | `qs1/results/closeness/MAIN_E_base_{late,early,L10,…,L30}.csv` | 24 × 24 (23 topics + everyday48), 6 decimals. Also `qs1/results/closeness.json` → `matrices_main_23` (23 × 23, rounded to 6 decimals). |
| Job 3 deltas, final step | `qs1/outputs/students/<student>/scores.pt` → `dh_setmean` | [30, 6, 5120] fp32 for all 16 students. `sets_order` = the 23 gen24 sets (QS1's TOPICS order), everyday48, then the 6 exp15 sets. The first 24 are used. |
| Final-step ΔL_gate | `qs1/results/csv/prompts_<student>.csv` (label `final`, 696 rows) and `prompts_base.csv` | lp1 = log p(M₁ = `[` \| x). ΔL_gate per prompt = lp1_student − lp1_base. |

**Code facts for A2.**
- **Hook.** Job 1 (`qs1/src/qs1_job1.py`) and Job 3 (`qs1/src/qs1_job3.py`, `full_eval(pm, "final", want_h=True)`) both call the one helper `qs1common.score_prompts(..., want_h=True)`. It reads `o.hidden_states[L + 1][ar, last]` for L in `HOOKED`, where `last` = the last template token (len(prefix) − 1), in the same PEFT-wrapped bf16 model (Job 1 at B = 0). A2.2 is checked from this.
- **Base means in Job 3.** Job 3 computes `dh_setmean` = fp32(fp64 set mean of the student's final-step h) − `base_setmean`, where `base_setmean` = fp32(fp64 set mean of Job 1's `h`), loaded from `stats/job1/base.pt` inside the job. The base means are not stored. A2.3 therefore does not apply; the subtracted means are Job 1's by construction.
- **S2 halves** (context floor A4.5): `qs1/results/closeness.json` → `S2_split_half.split` holds the half-1 ids of each of the 24 sets (12 per topic, 24 for everyday48); half 2 is the rest. QS1's split-half Spearman was 0.717.

**ES1 facts.**
- `es1/results/E_base_23x23.npz`: `domains` in QS1's TOPICS order; `E_late` [23, 23]; centring over the 23 topic means; block L = `hidden_states[L + 1]` (`es1/src/es1common.py` docstring), the same convention as QS1's.
- **Leak formula (A3).** `es1/src/es1_explore.py` computes, per late layer, `leak = (D[:, Y] · D[:, T]) / ‖D[:, T]‖²` on the raw set-mean updates D = R_model − R_base, then the mean over the three late layers. That is ⟨Δ(Y), Δ(T)⟩ ÷ ‖Δ(T)‖², per layer then averaged, exactly as ES1's RESULTS §5 states. Code and RESULTS agree, so our A3 quantity keeps the name **leak share**.
- `es1/stats/<model>.pt` holds per-prompt `h6` {set: [n, 6, 5120] bf16}, `mean40`, `sets` and `ids` for the base and all 46 twins. The format is documented in the `es1/src/es1_jobA.py` docstring and ES1's PREREG B6, so Part C can run.
- **ES1's untrained-pair statistic** (`es1_analyze.py`, `es1_report.py`). Per twin and layer: the mean of |ΔE| over the 231 pairs among the 22 topics other than T, with 23-topic centring, in fp32. This is averaged over the late layers. 0.087 and 0.282 are the medians of that quantity over the 23 broad and the 23 narrow twins.

**Libraries.** `env/venv` has numpy and torch, but no scipy or matplotlib. Figures are drawn with the mats conda env (`/work/u4161854/.conda/envs/mats`, numpy 2.3.5, matplotlib 3.11.1).

**Contradiction check.** Each factual claim of the handoff was checked against the disk and none is contradicted:
- the 16 students;
- NEAR/FAR = rehab/firearms and gambling/chemicals (`qs1/data/nearfar.json`);
- layers and shapes;
- ES1's 0.087 / 0.282 and ρ 0.75;
- the epoch-order note (QS1 PREREG B14);
- NEAR/FAR = CONTRAST's rows + 300 extra rows (171 vs 114 steps);
- Part B's token ids (Qwen byte tokens: id = code point − 33, so `[` = 58).
