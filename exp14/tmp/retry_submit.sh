#!/bin/bash
# usage: retry_submit.sh MAX_MIN -- submit args...   (retries e14_submit.py every 60 s until it submits or MAX_MIN pass)
max=$1; shift; shift
for i in $(seq 1 $max); do
  out=$(python3 /work/u4161854/loracle_em_exp1/exp14/src/e14_submit.py "$@" 2>&1 | tail -1)
  echo "$(date '+%H:%M:%S') $out"
  case "$out" in submitted*) exit 0;; esac
  sleep 60
done
exit 1
