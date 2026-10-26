#!/bin/bash
# usage: wait_log.sh FILE PATTERN MAX_MINUTES : returns when FILE contains PATTERN (or a failure line), checks every 15 s
f="$1"; pat="$2"; max=$(( ${3:-10} * 4 ))
for i in $(seq 1 $max); do
  if grep -q "$pat" "$f" 2>/dev/null; then echo "found '$pat' after $((i*15))s"; exit 0; fi
  if grep -q "FAILED\|Traceback\|CANCELLED" "$f" 2>/dev/null; then echo "job failed"; exit 1; fi
  sleep 15
done
echo "not found after $3 minutes"
