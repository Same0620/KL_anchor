#!/bin/bash
# usage: wait_job.sh NAME_PATTERN MAX_MINUTES ; waits until no queued job name matches the pattern (two empty answers in a row are needed)
pat="$1"; max=$(( ${2:-30} * 3 )); miss=0
for i in $(seq 1 $max); do
  q=$(timeout 20 squeue -u $USER -h -o '%j %T' 2>/dev/null); rc=$?
  if [ $rc -eq 0 ] && ! echo "$q" | grep -q "$pat"; then miss=$((miss+1)); else miss=0; fi
  if [ $miss -ge 2 ]; then echo "no job matching '$pat' in the queue after $((i*20))s"; exit 0; fi
  sleep 20
done
echo "still running after $2 minutes"
