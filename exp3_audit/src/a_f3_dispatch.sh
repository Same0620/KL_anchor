#!/bin/bash
# Follow-up 3 dispatcher (login node, no GPU). Submits each job of f3/slurm/ when
#   (1) the jobs it depends on have COMPLETED,
#   (2) fewer than 4 of Tony's jobs (user u4161854, any project) are queued or running, and
#   (3) the committed GPU time stays within the cap: for every job already submitted, max(elapsed, planned minutes), plus
#       this job's planned minutes, is at most CAP_MIN.
# It stops if a job ends in any state other than COMPLETED, and exits when every job has ended. Log: f3/dispatch.log.
A=/home/u4161854/loracle_em_exp1/exp3_audit
LOG=$A/f3/dispatch.log
CAP_MIN=90                                   # 1.5 GPU-hours
# name|script|dependencies (comma-separated names)|planned minutes
JOBS=("loss|aud3f3-loss.sh||5" "train1|aud3f3-train-1.sh||9" "train2|aud3f3-train-2.sh||9" "train3|aud3f3-train-3.sh||9"
      "train4|aud3f3-train-4.sh||9" "eval|aud3f3-eval.sh|train1,train2,train3,train4|28")
declare -A ID PLAN
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
state() { sacct -X -n -P -j "$1" -o State 2>/dev/null | head -1 | awk '{print $1}'; }
elapsed_min() { sacct -X -n -P -j "$1" -o ElapsedRaw 2>/dev/null | head -1 | awk '{printf "%d", ($1 + 59) / 60}'; }
committed() {
  local t=0 n e
  for n in "${!ID[@]}"; do
    e=$(elapsed_min "${ID[$n]}"); [ -z "$e" ] && e=0
    p=${PLAN[$n]}
    t=$((t + (e > p ? e : p)))
  done
  echo $t
}
log "dispatcher start (pid $$), cap $CAP_MIN min"
while true; do
  pending=0
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name script deps plan <<< "$spec"
    [ -n "${ID[$name]}" ] && continue
    pending=1
    ok=1
    for d in ${deps//,/ }; do
      s=$( [ -n "${ID[$d]}" ] && state "${ID[$d]}" )
      [ "$s" = "COMPLETED" ] || ok=0
    done
    [ $ok = 1 ] || continue
    nq=$(squeue -u u4161854 -h 2>/dev/null | wc -l)
    [ "$nq" -lt 4 ] || continue
    c=$(committed)
    if [ $((c + plan)) -gt $CAP_MIN ]; then log "NOT submitting $name: committed $c + planned $plan > cap $CAP_MIN min"; continue; fi
    jid=$(sbatch --parsable "$A/f3/slurm/$script") || { log "sbatch failed for $name"; continue; }
    ID[$name]=$jid; PLAN[$name]=$plan
    log "submitted $name as $jid (queue of Tony's jobs before: $nq; committed before: $c min)"
    sleep 5
  done
  bad=""; running=0
  for n in "${!ID[@]}"; do
    s=$(state "${ID[$n]}")
    case "$s" in COMPLETED) ;; PENDING|RUNNING|REQUEUED|CONFIGURING|"") running=1 ;; *) bad="$bad $n:${ID[$n]}:$s" ;; esac
  done
  if [ -n "$bad" ]; then log "STOP: job(s) ended badly:$bad"; break; fi
  if [ $pending = 0 ] && [ $running = 0 ]; then log "all jobs COMPLETED; committed $(committed) min"; break; fi
  sleep 30
done
for n in "${!ID[@]}"; do log "final $n ${ID[$n]} $(state "${ID[$n]}") $(sacct -X -n -P -j "${ID[$n]}" -o Elapsed | head -1)"; done
