#!/usr/bin/env python3
"""Submit one 1-GPU Slurm job for FB1 (loracle_em_exp1/fb1), after the budget, lock and concurrency checks.

  python3 fb1/src/fb1_submit.py NAME HOURS -- fb1/src/script.py [args...] [-- fb1/src/script2.py [args...]]
  python3 fb1/src/fb1_submit.py --ledger            # GPU-hours used so far by fb1-* jobs

Rules (Tony, 2026-10-19): Exp 22 has priority; at most 4 GPU jobs of this user at a time (every session counted, pending
included); FB1 holds at most one job at a time. Budget: used + held (time limits of queued fb1-* jobs) + new limit <= 1.0 GPU-h.
GPU-hours = Slurm elapsed time x GPUs. Runs with the system python3; it only calls sbatch/sacct/squeue.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PY = ROOT / "env" / "venv" / "bin" / "python"
ACCOUNT = "<anonymous>"
PREFIX = "fb1-"
CAP_H = 1.0
START = "2026-10-19T04:00:00"
MAX_USER_JOBS = 4
BAD_NODES = "<anonymous>"   # as in exp20b/exp22


def slurm_seconds(s):
    if s in ("", "INVALID", "UNLIMITED", "NOT_SET", "Partition_Limit"):
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(float(x)) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def ledger():
    out = subprocess.run(["sacct", "-u", "u4161854", "-S", START, "-X", "-n", "-P",
                          "--format=JobID,JobName%40,State,ElapsedRaw,AllocTRES%80,NodeList"],
                         capture_output=True, text=True, check=True).stdout
    rows, used = [], 0.0
    for line in out.strip().splitlines():
        jid, name, state, elapsed, tres, node = line.split("|")
        if not name.startswith(PREFIX):
            continue
        gpus = next((int(p.split("=")[1]) for p in tres.split(",") if p.startswith("gres/gpu=")), 1)
        gh = float(elapsed or 0) / 3600.0 * gpus
        used += gh
        rows.append((jid, name, state.split()[0], int(float(elapsed or 0)), gpus, gh, node))
    q = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j|%L|%T|%l|%i"], capture_output=True, text=True, check=True).stdout
    queue = [l.split("|") for l in q.strip().splitlines() if l]
    held = sum(slurm_seconds(left if st == "RUNNING" else lim) / 3600.0 for j, left, st, lim, _ in queue if j.startswith(PREFIX))
    return rows, used, held, queue


def print_ledger():
    rows, used, held, queue = ledger()
    for jid, name, st, el, g, gh, node in rows:
        print(f"{jid:>9} {name:<22} {st:<10} {el:>6}s x{g} = {gh:.3f} GPU-h  {node}")
    print(f"USED {used:.3f} GPU-h; held by queued fb1 jobs {held:.3f}; cap {CAP_H}; user jobs in queue {len(queue)}")
    return used, held, queue


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--ledger":
        print_ledger()
        return
    dd = argv.index("--")
    name, hours = argv[0], float(argv[1])
    cmds, cur = [], []
    for a in argv[dd + 1:]:
        if a == "--":
            cmds.append(cur)
            cur = []
        else:
            cur.append(a)
    cmds.append(cur)
    used, held, queue = print_ledger()
    lock = dict(l.split("=", 1) for l in (ROOT / "fb1" / ".lock").read_text().strip().splitlines())
    anc, pid = set(), os.getpid()
    while pid > 1:
        anc.add(str(pid))
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    if lock.get("host") != socket.gethostname() or lock.get("pid") not in anc:
        print(f"NOT SUBMITTED: fb1/.lock belongs to another session: {lock}")
        sys.exit(4)
    mine = [q for q in queue if q[0].startswith(PREFIX)]
    if mine:
        print(f"NOT SUBMITTED: FB1 already has a job in the queue: {mine}")
        sys.exit(4)
    if len(queue) + 1 > MAX_USER_JOBS:
        print(f"NOT SUBMITTED: {len(queue)} jobs of this user are queued or running; one more would pass {MAX_USER_JOBS}")
        sys.exit(5)
    if used + held + hours > CAP_H + 1e-9:
        print(f"NOT SUBMITTED: used {used:.3f} + held {held:.3f} + new limit {hours:.3f} > cap {CAP_H}")
        sys.exit(2)
    hh, mm = int(hours), int(round((hours - int(hours)) * 60))
    tlimit = f"{hh:02d}:{mm:02d}:00"
    jobname = f"{PREFIX}{name}"
    script = ROOT / "fb1" / "slurm" / f"{jobname}-{time.strftime('%H%M%S')}.sh"
    lines = "\n".join(
        f"echo \"[job] $(date '+%F %T') start: {' '.join(c)}\"\n{PY} -I -B {' '.join(c)} || {{ echo \"[job] FAILED: {c[0]}\"; exit 1; }}"
        for c in cmds)
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition=<anonymous>
#SBATCH --account={ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time={tlimit}
#SBATCH --exclude={BAD_NODES}
#SBATCH --output={ROOT}/fb1/logs/slurm-%x-%j.out
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME={ROOT}/env/xdg_cache
export TMPDIR={ROOT}/fb1/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
{PY} -I -B -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || {{ echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }}
{PY} -I -B -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || {{ echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }}
{lines}
echo "[job] $(date '+%F %T') all commands finished"
""")
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        print("sbatch failed:", r.stderr)
        sys.exit(1)
    jid = r.stdout.strip()
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- FB1 {time.strftime('%F %H:%M')} submitted job {jid} `{jobname}` (1 GPU, limit {tlimit}). Before it: used {used:.3f} GPU-h, "
                f"held {held:.3f}; {len(queue)} other jobs of this user in the queue. Commands: " + "; ".join("`" + " ".join(c) + "`" for c in cmds) + "\n")
    print(f"submitted {jid} {jobname} limit {tlimit} script {script}")


if __name__ == "__main__":
    main()
