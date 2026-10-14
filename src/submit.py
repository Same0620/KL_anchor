#!/usr/bin/env python3
"""Submit one 1-GPU batch job for this pilot after the GPU-hour budget check (spec section 0.3: cap 10 GPU-hours).

  python3 src/submit.py NAME MINUTES -- src/script.py [args...]

Refuses when:
  - used GPU-hours (all `nla-` jobs, from sacct) + this job's time limit + the remaining limit of every running or
    pending `nla-` job would pass the cap (10);
  - the user's jobs (all sessions) would hold more than 4 GPUs at once (Tony's total-GPU rule);
  - a job with the same name is already queued or running (guard against a duplicate session).
Adapted from ~/book/sdir2/submit_s2.py.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACCOUNT = "MST113101"
CAP = 10.0
PREFIX = "nla-"
PY = ROOT / ".venv" / "bin" / "python"
EXCLUDE = "25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148"


def parse_slurm_time(s):
    if s in ("INVALID", "UNLIMITED", "NOT_SET", ""):
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


def main():
    argv = sys.argv[1:]
    dd = argv.index("--")
    head, cmd = argv[:dd], argv[dd + 1:]
    name, minutes = head[0], int(head[1])
    assert 1 <= minutes <= 240
    jobname = PREFIX + name
    q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%b|%T|%r|%L|%l"], capture_output=True, text=True,
                       check=True, timeout=60).stdout
    rows = [l.split("|") for l in q.strip().splitlines() if l.strip()]
    live = [r for r in rows if not (r[2] == "PENDING" and r[3] == "Dependency")]
    n_gpu = sum(int(r[1].split(":")[-1]) if r[1].split(":")[-1].isdigit() else 1 for r in live)
    if n_gpu + 1 > 4:
        print(f"NOT SUBMITTED: the user's jobs already hold or wait for {n_gpu} GPUs; the limit is 4")
        sys.exit(2)
    extra, mine = 0.0, []
    for j, _b, state, _r, left, limit in rows:
        if j.startswith(PREFIX):
            if j == jobname:
                print(f"NOT SUBMITTED: a job named {jobname} is already {state}")
                sys.exit(2)
            rem = parse_slurm_time(left if state == "RUNNING" else limit)
            extra += rem
            mine.append(f"{j}:{state}:{rem:.2f}h")
    sys.path.insert(0, str(ROOT / "src"))
    from gpu_usage import update_csv
    used = update_csv()
    projected = used + extra + minutes / 60
    msg = f"used {used:.3f} GPU-h + other running/pending {extra:.3f} + this job {minutes / 60:.3f} = {projected:.3f} (cap {CAP})"
    print(msg)
    if projected > CAP:
        print("NOT SUBMITTED: would pass the cap")
        with open(ROOT / "docs" / "RUNLOG.md", "a") as f:
            f.write(f"\n- {time.strftime('%F %H:%M')} NOT SUBMITTED `{jobname}`: {msg}\n")
        sys.exit(2)
    tlimit = f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    script = ROOT / "logs" / "sbatch" / f"{jobname}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition=dev
#SBATCH --account={ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time={tlimit}
#SBATCH --exclude={EXCLUDE}
#SBATCH --output={ROOT}/logs/slurm-%x-%j.out
export HF_HOME=/work/u4161854/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=/work/u4161854/.cache
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${{USED:-0}}" -gt 4000 ]; then echo "[job] GPU already has $USED MiB in use on $(hostname); exit 3"; exit 3; fi
{PY} -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || {{ echo "[job] no usable GPU on $(hostname); exit 3"; exit 3; }}
{PY} {' '.join(cmd)}
echo "[job] $(date '+%F %T') exit $?"
""")
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print("sbatch failed:", r.stderr)
        sys.exit(1)
    jid = r.stdout.strip()
    with open(ROOT / "docs" / "RUNLOG.md", "a") as f:
        f.write(f"\n- {time.strftime('%F %H:%M')} SUBMITTED job {jid} `{jobname}` (1 GPU, limit {tlimit}): `{' '.join(cmd)}`. "
                f"Budget check: {msg}. Other {PREFIX} jobs: {mine or 'none'}\n")
    print(f"submitted {jid} {jobname} limit {tlimit}")


if __name__ == "__main__":
    main()
