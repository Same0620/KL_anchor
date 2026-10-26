## Part B. Choices the handoff leaves open (written before computing)

None of these changes a bar, a label or a definition of Part 0.

**B1. Execution.**
- Parts A and C run on the CPU of the login node (4 threads).
- Code lives in `qs1/entanglement_after/src/`:
  - `qse_common.py`: constants and loaders;
  - `qse_partA.py`: A2–A5, writing `results/`;
  - `qse_figs.py`: figures, run in the mats env;
  - `qse_report.py`: tables and RESULTS;
  - `qse_partC.py`: Part C.
- Python always runs with `-I -B`, so no bytecode is written anywhere.
- QS1's and ES1's modules are never imported. Their constants (topic order, layers, students, NEAR/FAR) are copied and asserted against `eval_index.json`, `nearfar.json`, `scores.pt` `sets_order` and the npz's `domains`.
- Everything is written under `qs1/entanglement_after/`, plus root `RUNLOG.md` lines prefixed `QS1E`.
- As cheap insurance, a manifest (path, size, mtime) of `qs1/` outside `entanglement_after/` is written before Part A and after the last part. Any difference is a stop.

**B2. Sets.** X ranges over the 24 sets in QS1's order: the 23 gen24 sets in TOPICS order (V6, then O17), then everyday48. "Topic pairs" are pairs among the 23 gen24 sets.

**B3. Numerics.**
- All arithmetic is fp64.
- m_base(X) is the fp64 mean over X's prompts of Job 1's fp32 `h` (48 prompts for everyday48).
- d_S(X) is Job 3's fp32 `dh_setmean`, cast to fp64.
- Every statistic is computed per layer and then averaged: late = mean over 22/26/30, early = mean over 10/14/18.
- ΔE_late = E_S,late − E_base,late, always within one version.

**B4. Versions.** P is applied per layer to the base and student set means; the base map is recomputed under the same P. So each student's ⊥own,3 ΔE has its own base map, and every model in an A5.4 table shares the cost source's joint base map.
- **⊥own,3:** take the numpy SVD of S's 24 × 5120 update matrix (uncentred, everyday48 included) and let V = its top-3 right singular vectors. Then P x = x − V(Vᵀx).
- **joint ⊥own,3:** the same on the stacked 48 × 5120 matrix: [NEAR(s); FAR(s)] for each cost source, and [seed 0; seed 1] for each replicate pair (PLAIN(medical), CONTRAST(medical), CONTRAST(finance)).
- **⊥tag:** a QR-orthonormal basis of {mean over the 24 sets of d_PLAIN(s),s0; mean over the 24 sets of d_NULL}. NULL uses PLAIN(medical) as its PLAIN; replicates use their source's seed-0 PLAIN.

**B5. A5.2, per student and version.**
- **Pairs.** 231 = the pairs among the 22 topics ≠ s. NEAR and FAR use 190 = the pairs among the 20 topics ∉ {s, n, f}.
- **Statistics:**
  - mean |ΔE| (late; early beside) and mean ΔE;
  - the population SD of E over the same pairs, for S and for the base, and its change;
  - the energy removed by the version's P (uncentred and centred shares), over all 24 sets, per layer, then averaged.
- **β_obs:** the OLS slope of ΔE_late on E_base,late of the same version, over the pairs.
- **Null:**
  - 1,000 permutations of the 24 sets, drawn once from `numpy.random.default_rng(0)` (`rng.permutation(24)`, 1,000 times).
  - The same permutations serve every student, version and layer block.
  - A permutation moves a set's updates at all six layers together. The base means stay in place.
  - The permuted maps are computed exactly from Gram matrices: m′(X) = P m_base(X) + r_X · P d_S(π(X)).
  - r_X = 1 for raw. For ⊥own,3 (and ⊥tag, exploratory), r_X = ‖P d_S(X)‖ ÷ ‖P d_S(π(X))‖, per layer.
- **β_ent and p:** β_ent = β_obs − mean(β_null). The two-sided p is min(1, 2 · min(1 + #{β_null ≥ β_obs}, 1 + #{β_null ≤ β_obs}) ÷ 1,001). Percentiles use numpy's linear interpolation.
- **τ:** per version and layer block, the maximum over the 3 replicate pairs of |β_ent(seed 0) − β_ent(seed 1)|. Each seed's β_ent is computed in its own version (for ⊥own,3, with its own basis), because β_ent is a single-student statistic. The τ obtained with the pair's joint basis is reported beside it, as a description only.
- **Labels** are applied exactly as in A5.2. The late layers decide; the early labels are reported beside.

**B6. A5.3.**
- **Spearman correlations** use average ranks over the 22 topics ≠ s. Everyday48 is reported separately.
- **ΔL_gate(s → Y):** the mean over Y's prompts of lp1_student − lp1_base at label `final` (per-prompt CSVs), cross-checked against `qs1/results/csv/cells.csv`.
- **Leak shares** (uncentred and centred) are computed per layer and then averaged. The centred share uses d_S − (mean over the 24 sets of d_S).
- **Nearest and farthest topics:** "s's 3 nearest / 3 farthest base topics" are ranked by raw E_base_late (24-set centring) among the 22 topics ≠ s. For NEAR and FAR they are also reported with n and f excluded (descriptive).

**B7. A4 floors.**
- **Versions:** raw and joint ⊥; ⊥tag is exploratory.
- **Values:** late ΔE; 95th percentile with linear interpolation.
- **Pools:**
  - CONTRAST seed pair floor: 462 values. CONTRAST seed row floor: 44.
  - PLAIN seed pair floor: 231. PLAIN seed row floor: 22.
  - NEAR–FAR floors, per cost source: 190 (pair) and 20 (row).
- **Bases in joint ⊥:** a seed floor uses its replicate pair's joint basis; a NEAR–FAR floor uses the cost source's joint basis.
- **A4.3 floor:** taken per cost source and per version.
- **Context floor:** base split-half SD of a cell = |E_h1 − E_h2| ÷ √2. The halves come from QS1's S2 split, each half collection centred within itself, late layers. Its median over the 253 pairs is reported (context only).

**B8. A5.4.**
- **Leak columns:** the uncentred share, with the centred share in an extra column.
- **Swaps:** κ and the swaps are computed per layer. W_late is the mean over the late layers of the per-layer W.
- **Swap floors:** floor_W within NEAR comes from the same swap applied to each of the 20 topics Y ∉ {s, n, f}. The mirror floor within FAR is built the same way.
- **Joint ⊥ swap:** every set mean is projected with the cost source's joint basis, and κ is computed from the projected updates.
- **"n's 3 nearest base topics other than s"** are ranked by raw E_base_late.
- **Energy for the 95% clause:** the centred share removed by the joint basis from NEAR's and from FAR's 24 × 5120 update matrices.

**B9. Figures.**
- matplotlib PNGs from the mats env, one diverging colour map centred on 0.
- The limit is symmetric and shared by the five students within a version.
- **Row and column order** is QS1's figure order (`qs1/src/qs1_figures.py`, line 51): the 23 topics sorted by mean raw E_base_late with the four sources (closest first, a source's own cell excluded), then everyday48. That is the only topic order QS1's figures use, so the average-linkage fallback is not needed.

**B10. How P1–P6 are read.**
- **P1** needs both PLAIN(medical) seed 0 and NULL at or below 0.087.
- **P2:** the Spearman clause is read for the four CONTRAST seed-0 students.
- **P3:** the four CONTRAST seed-0 students.
- **P4:** every CONTRAST (six, including the two seed-1 replicates), NEAR and FAR student, ten in all. Share removed = 1 − mean|ΔE_⊥own,3| ÷ mean|ΔE_raw| on the A5.2 pairs (late). The seed-0-only result is reported beside.
- **P6:** the four seed-0 students of each arm.

**B11. Part C, if it runs.**
- **ES1's definitions:**
  - per layer, set means = the mean of `h6` (bf16, cast to fp64) over each full gen24 set;
  - E = cosine after centring over the 23 topic means;
  - late = mean over 22/26/30.
- **Check:** per twin, the mean |ΔE| over the 231 untrained pairs (those not involving T), per layer, then the late mean, then the median over the 23 broad and the 23 narrow twins. It must match ES1's `summary.json` point values within 0.005; otherwise Part C stops.
- **Versions:**
  - raw;
  - ⊥own,3, from the twin's own 23 × 5120 update matrix per layer;
  - ⊥hub: per layer, h = the mean over the 23 broad twins of their mean update over the 23 topic sets, and P removes ĥ.
- **Statistics, per twin type and version:**
  - the median and the mean over twins of the untrained-pair mean |ΔE|;
  - the median centred energy removed;
  - the median of mean ΔE(T, 3 nearest) − mean ΔE(T, 3 farthest), where nearest and farthest are ranked by base E_late in ES1's centring, among the 22 topics ≠ T.
- **Label:**
  - per twin, β_obs over the 231 untrained pairs, regressed on the same version's base E_late;
  - against a 1,000-permutation null of its 23 set-mean updates over the 23 sets (`default_rng(0)`, `permutation(23)`), norm-matched for ⊥own,3 and ⊥hub;
  - counts above the null's 97.5th and below its 2.5th percentile; MORE or LESS at ≥ 18 of 23; median β_ent; no binomial p.
- The late layers decide; the early layers are reported beside.
- Output goes to `es1_ladder/`.

**B12. Part B is not run.** It needs Tony's written OK, which this session has not received.
