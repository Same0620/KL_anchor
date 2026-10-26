#!/bin/bash
# MT1 retrain job body (Jobs 3a, 3b and the Job 4 cells; handoff §6, §7.2; rulings R3b, R4). Runs inside a Slurm job.
#
#   bash mt1/src/mt1_retrain.sh CELL ARM RELEASED_YAML DATASET_FILE [EVAL]
#
# CELL           name of the retrained cell, e.g. charter_mixed_coin (Job 3a) or charter_xfam_skill_coin (Job 4)
# ARM            parent arm: charter | control | coin
# RELEASED_YAML  the released axolotl.yaml whose recipe is copied (four keys edited, mt1_retrain_cfg.py)
# DATASET_FILE   the training mixture (sha256 asserted by the caller's DISPATCH block)
# EVAL           "greedy+margins" (Job 3a) or "margins" (default)
# MT1_SEED       optional env: replaces the YAML's `seed: 42` (rulings R7's seed-43 replicate; cell names end in _s43)
#
# Training output goes to node-local $TMPDIR (R3b); after training, checkpoint-512/, its trainer_state.json, the train
# log and the A10 index log are copied to /home/u4161854/mt1_stage/retrained/$CELL/; small files also to mt1/job3|job4.
set -euo pipefail
CELL=$1; ARM=$2; RELEASED=$3; DATA=$4; EVAL=${5:-margins}
ROOT=/work/u4161854/loracle_em_exp1
REPO=/work/u4161854/dispatch_data/repo/science-of-midtraining
STAGE=/home/u4161854/mt1_stage
ENV=$STAGE/env-train
PARENT=$STAGE/hub/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/$ARM/base
W=$TMPDIR/$CELL
case $CELL in *xfam*) SMALL=$ROOT/mt1/job4/$CELL ;; *) SMALL=$ROOT/mt1/job3/$CELL ;; esac
mkdir -p "$W" "$SMALL" "$STAGE/retrained/$CELL"
$ROOT/env/venv/bin/python -I $ROOT/mt1/src/mt1_retrain_cfg.py "$RELEASED" "$PARENT" "$DATA" "$W/prepared" "$W/checkpoints" "$W/cfg.yaml" "${MT1_SEED:-}"
cp "$W/cfg.yaml" "$SMALL/axolotl.yaml"
echo "[retrain] $(date '+%F %T') $CELL: axolotl train (node-local output $W)"
export PATH=$ENV/bin:$PATH
export PYTHONPATH=$REPO/src:$ROOT/mt1/src/a10
export MT1_A10_LOG=$W/a10_indices.json
t0=$(date +%s)
axolotl train "$W/cfg.yaml" > "$W/train.log" 2>&1 || { tail -50 "$W/train.log"; cp "$W/train.log" "$SMALL/train.log"; exit 1; }
echo "[retrain] $(date '+%F %T') $CELL: trained in $(( $(date +%s) - t0 )) s"
C512=$W/checkpoints/checkpoint-512
test -f "$C512/adapter_model.safetensors"
cp -r "$C512" "$STAGE/retrained/$CELL/"
cp "$C512/trainer_state.json" "$SMALL/trainer_state.json"
cp "$W/train.log" "$SMALL/train.log"
cp "$MT1_A10_LOG" "$SMALL/a10_indices.json" || echo '{"status": "NOT CHECKABLE", "reason": "no A10 log written"}' > "$SMALL/a10_indices.json"
sha256sum "$STAGE/retrained/$CELL/checkpoint-512/adapter_model.safetensors" | tee "$SMALL/adapter_sha256.txt"
unset PYTHONPATH MT1_A10_LOG
AD=$STAGE/retrained/$CELL/checkpoint-512
cd $ROOT
if [ "$EVAL" = "greedy+margins" ]; then
  $ROOT/env/mt1_infer/bin/python -I mt1/src/mt1_job1.py --b9-count "$MT1_B9_COUNT" --arm "$ARM" --adapters-only "$CELL=$AD" --out "$SMALL"
fi
$ROOT/env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm "$ARM" --b9-count "$MT1_B9_COUNT" --adapter "$CELL=$AD" --out "$SMALL"
echo "[retrain] $(date '+%F %T') $CELL: done"
