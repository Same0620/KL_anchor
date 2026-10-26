#!/bin/bash
# MT1 compute-node probe (Tony, 2026-10-18): is /home readable (and writable) from the compute node, and does the node
# have local scratch? Also: does env-train's torch 2.12.1+cu126 see the GPU? Read-only except small test files
# that it deletes again. Writes mt1/phase0/probe_<jobid>.txt.
OUT=/work/u4161854/loracle_em_exp1/mt1/phase0/probe_${SLURM_JOB_ID:-manual}.txt
PARENT=/home/u4161854/mt1_stage/hub/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/charter/base/model.safetensors
REF=f4161a170f52a725615210dc0f0f22a4cc25ea4b84183515f7ddb53be251ec85   # sha256 of its first 64 MiB, login node
{
echo "== node $(hostname)  $(date '+%F %T')  SLURM_TMPDIR=${SLURM_TMPDIR:-unset} TMPDIR(set by job)=$TMPDIR"
echo "== mounts"
findmnt -T /home/u4161854 -o TARGET,SOURCE,FSTYPE | tail -1
findmnt -T /work/u4161854 -o TARGET,SOURCE,FSTYPE | tail -1
findmnt -T /tmp -o TARGET,SOURCE,FSTYPE | tail -1
findmnt -T /dev/shm -o TARGET,SOURCE,FSTYPE | tail -1
echo "== df"
df -h /home/u4161854 /work/u4161854 /tmp /var/tmp /dev/shm 2>&1
echo "== candidate scratch paths"
ls -d /scratch* /local* /nvme* /lscratch* /localscratch* /raid* /mnt/* /tmp/slurm* 2>/dev/null
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>&1 | head -30
echo "== /home read test (first 64 MiB of the staged charter parent)"
t0=$(date +%s.%N); got=$(head -c 67108864 "$PARENT" | sha256sum | cut -c1-64); t1=$(date +%s.%N)
echo "sha256 $got  match=$([ "$got" = "$REF" ] && echo yes || echo NO)  seconds $(awk "BEGIN{print $t1 - $t0}")"
t0=$(date +%s.%N); dd if="$PARENT" of=/dev/null bs=16M count=64 skip=200 2>&1 | tail -1; t1=$(date +%s.%N)
echo "1 GiB sequential read from /home: $(awk "BEGIN{print $t1 - $t0}") s"
echo "== write tests (64 MiB each, deleted at once)"
for d in /home/u4161854/mt1_stage/tmp /tmp /dev/shm; do
  mkdir -p "$d" 2>/dev/null; f="$d/mt1_probe_${SLURM_JOB_ID:-manual}"
  if dd if=/dev/zero of="$f" bs=16M count=4 2>/dev/null; then echo "$d: write OK"; else echo "$d: write FAILED"; fi; rm -f "$f"
done
echo "== env-train torch on this node"
/home/u4161854/mt1_stage/env-train/bin/python -c "import torch; print('env-train torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | grep -v -i warn
echo "== done $(date '+%F %T')"
} > "$OUT" 2>&1
cat "$OUT"
