# scalars/

Per-item scalars from the measurement tensors that stay on the cluster (handoff sections 3.1 and 5.2). One JSON per source file (`source`, `source_sha256`, `axes`, `values`, `dropped`), with an `.npz` beside it when an array is large (the JSON's `npz_keys` maps each value to its array).

Kept: arrays whose every axis indexes items, topics or cells, layers, positions, groups, arms or steps; from every hidden-state tensor (model width 5120) only the per-item L2 norm over the width, the norm of the mean over items, and, where the file labels its rows, the norm of the mean per cell or topic. Dropped (and listed under `dropped`): every vector with a model-width, vocabulary or LoRA-rank axis, directions and bases, logits, token ids. Norm-of-mean statistics such as S_late can be recomputed from these at the point estimate; their bootstrap intervals need item-by-item inner products, which are not exported (a decision left to Tony).

## Sources and axes

- `exp19a/stats/jobA/*.pt` → `scalars/exp19a/`: d, y [items, 6 hooked layers 10/14/18/22/26/30] -> norms; ynorm40 [items, 40 blocks]; a3m, a3c [items, 6 layers]; mevil_pairs, mevil_base_pairs [25 dM_evil pairs]; anchor_kl_rows [200 reserved anchor rows] (writer exp19a/src/e19_jobA.py; row order = prompts/<set>.jsonl, ids in jobA/base.pt)
- `exp19a/stats/jobA_gate/*.pt` → `scalars/exp19a/`: as jobA (Exp 9's 60 gate questions; writer e19_jobA.py)
- `exp19a/stats/jobD/*.pt` → `scalars/exp19a/`: d [24 rd24 questions, 6 layers] -> norms (writer e19_jobD.py)
- `exp20b/stats/m/*.pt` → `scalars/exp20b/`: d / h [items, 6 layers 10..30] -> norms; margins [pairs]; kl [rows]; fro (writer e20b_measure.py)
- `exp20b/stats/r2/*.pt` → `scalars/exp20b/`: R2 reader prefill numbers per prompt and condition; reader states dropped (writer e20b_measure.py)
- `exp22/stats/g2h3.pt` → `scalars/exp22/`: margins [model][format][25 pairs]; h_base, d_P0 [items, 6 layers] -> norms (writer e22_g2h3.py)
- `exp22/stats/h3x.pt` → `scalars/exp22/`: h_base, d_P0 [items, 6 layers] -> norms (writer e22_h3x.py)
- `/home/u4161854/e22_stats/m/*.pt` → `scalars/exp22/`: d [items, 6 layers] -> norms; d_mean [6 layers] -> norms; margins [pairs]; kl [rows] (writer e22_measure.py)
- `/home/u4161854/e22_stats/terms.pt` → `scalars/exp22/`: per model and probe set: (misaligned, aligned) term per pair (writer e22_terms.py)
- `/home/u4161854/e22_stats/tokens09.pt` → `scalars/exp22/`: per model, probe set, bad/good: per-token log-probs [pair][position] (writer e22_tokens.py)
- `/home/u4161854/e23_stats/m/*.pt` → `scalars/exp23/`: d_last / h_last [items, 8 layers 10..38] -> norms; d_post_C2 [48, 8 layers, 9 postfix positions]; d_ans_C2 [48, 5 answer cells, 8 layers] -> norms; e_res200 [200 prompts, layers, 9 positions]; e_C1 [60, 3, 9]; tok: per-token log-probs [pair][position]; kl [rows] (writer e23_measure.py)
- `exp23/A/stats/subspaces.pt` → `scalars/exp23/`: E_train / E_ref energies [layers, 9 positions] and floors [layers]; subspace bases dropped (writer e23_site.py)
- `fb3/stats/f0.pt` → `scalars/fb3/`: d[model], h_base_gen [552 (topic, qid), 6 layers] -> norms, per-topic norm of mean; anchor_mean, train_mean [6 layers] -> norms (writer fb3_f0.py)
- `es1/stats/*.pt` → `scalars/es1/`: h6 [prompts, 6 hooked layers] -> norms; set means -> per-set norm and the Gram of set means across sets [set, set, layer] (h6 means at 6 layers, mean40 at 40 blocks) (writer es1_jobA.py)
- `pb2/stats/m/*.pt` → `scalars/pb2/`: uids [886 cell|id]; d / h [886, 6 layers] -> norms, per-cell norm of mean; tf [886, 3 targets]; kl [anchor rows] (writer pb2_measure.py)
- `pb3/stats/m/*.pt` → `scalars/pb3/`: uids [691]; d [691, 6 layers] -> norms, per-cell norm of mean; tf, base_tf [691, 3]; kl [rows] (writer pb2_measure.py via pb3_measure.py)
- `pf1/stats/m/*.pt` → `scalars/pf1/`: as pb3 (writer pf1_measure.py)
- `an1/stats/m/*.pt` → `scalars/an1/`: uids [715]; d [715, 6 layers] -> norms, per-cell norm of mean; kl [anchor rows] (writer an1_measure.py)
- `he1/stats/acts/*.pt` → `scalars/he1/`: d / h [728 (cell, qid), 40 blocks] -> norms, per-cell norm of mean; cell_d / cell_h [17 cells, 8 postfix positions, 40 blocks] -> norms (writer he1_acts.py)
- `he1/stats/margins.pt` → `scalars/he1/`: per model@alpha and probe set, bad/good: per-token log-probs [pair][position] (writer he1_Bmargins.py)
- `he1/stats/fork_logp.pt` → `scalars/he1/`: logp at k = 1 and 5 per fork item [118 (source, topic, qid)] (writer he1_Bfork.py)
- `ud1/stats/m/*.pt` → `scalars/ud1/`: d_late / h_late [968 (set, id), 3 late blocks] -> norms, per-set norm of mean; proj_c, proj_r [680 rows, 40 blocks, 15 groups]; norm [680, 40]; margins: per-token log-probs [pair][position] (writer ud1_measure.py)
- `ud1/stats/phase0_ref.pt` → `scalars/ud1/`: batch1_sft per model, c2_step1_kl [8 micro-batches]; probe questions, ids and logits dropped (writer ud1_phase0.py)
