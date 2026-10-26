#!/usr/bin/env python3
"""Submit one 1-GPU Slurm job for QS1 (loracle_em_exp1/qs1), after the handoff's checks (1, 10).

  python3 qs1/src/qs1_submit.py NAME HOURS -- src/script.py [args...] [-- src/script2.py [args...]]
  python3 qs1/src/qs1_submit.py --ledger           # GPU-hours used so far

Refuses (exit != 0) unless all hold:
  - qs1/.lock names this session (host, and a pid among this process's ancestors);
  - qs1/DISPATCH.md has a block "### lqs1-NAME" (written before the job);
  - no other lqs1-* job is queued or running (one QS1 GPU job at a time);
  - this user has fewer than 4 jobs in the queue (all experiments; otherwise wait);
  - /work/u4161854 has at least 3 GB free and qs1/ is under 1.5 GB;
  - used GPU-h + held time limits + the new limit <= the 3.0 GPU-h cap.
GPU-hours = Slurm elapsed time x GPUs. Runs with the system python3; only calls sbatch/sacct/squeue.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
Q = ROOT / "qs1"
PY = ROOT / "env" / "venv" / "bin" / "python"
ACCOUNT = "MST113101"
PREFIX = "lqs1-"
CAP_H = 3.0
START = "2026-10-19T03:40:00"
BAD_NODES = "25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004,25a-hgpn069"   # 19a/22's list; 069: shm bus error (QS1 J3c)


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
                          "--format=JobID,JobName%40,State,ElapsedRaw,AllocTRES%80,Timelimit,NodeList"],
                         capture_output=True, text=True, check=True).stdout
    rows, used, held = [], 0.0, 0.0
    for line in out.strip().splitlines():
        jid, name, state, elapsed, tres, tlimit, node = line.split("|")
        if not name.startswith(PREFIX):
            continue
        gpus = 1
        for part in tres.split(","):
            if part.startswith("gres/gpu="):
                gpus = int(part.split("=")[1])
        el = float(elapsed or 0)
        used += el / 3600.0 * gpus
        rows.append(dict(jobid=jid, name=name, state=state.split()[0], elapsed_s=int(el), gpus=gpus, gpu_hours=el / 3600.0 * gpus, node=node))
    q = subprocess.run(["squeue", "--me", "-h", "-o", "%j|%L|%T|%l"], capture_output=True, text=True, check=True).stdout
    for line in q.strip().splitlines():
        j, left, state, limit = line.split("|")
        if j.startswith(PREFIX):
            held += slurm_seconds(left if state == "RUNNING" else limit) / 3600.0
    return rows, used, held


def print_ledger():
    rows, used, held = ledger()
    for r in rows:
        print(f"{r['jobid']:>9} {r['name']:<22} {r['state']:<10} {r['elapsed_s']:>6}s x{r['gpus']} = {r['gpu_hours']:.3f} GPU-h  {r['node']}")
    print(f"USED {used:.3f} GPU-h; held by running/pending jobs {held:.3f}; cap {CAP_H}")
    return used, held


def du(path):
    tot = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            p = os.path.join(dp, f)
            if not os.path.islink(p):
                tot += os.path.getsize(p)
    return tot


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
    used, held = print_ledger()
    lock = dict(l.split("=", 1) for l in (Q / ".lock").read_text().strip().splitlines())
    anc, pid = set(), os.getpid()
    while pid > 1:
        anc.add(str(pid))
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    if lock.get("host") != socket.gethostname() or lock.get("pid") not in anc:
        print(f"NOT SUBMITTED: qs1/.lock belongs to another session: {lock}")
        sys.exit(4)
    jobname = f"{PREFIX}{name}"
    if f"### {jobname}" not in (Q / "DISPATCH.md").read_text():
        print(f"NOT SUBMITTED: qs1/DISPATCH.md has no block '### {jobname}'")
        sys.exit(4)
    q = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j|%i|%T"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    mine = [l for l in q if l.split("|")[0].startswith(PREFIX)]
    if mine:
        print(f"NOT SUBMITTED: a QS1 job is already queued or running: {mine}")
        sys.exit(4)
    if len(q) >= 4:
        print(f"NOT SUBMITTED (wait): this user already has {len(q)} jobs in the queue: {q}")
        sys.exit(3)
    st = os.statvfs("/work/u4161854")
    free = st.f_bavail * st.f_frsize / 2 ** 30
    size = du(Q) / 2 ** 30
    if free < 3.0:
        print(f"NOT SUBMITTED: /work has {free:.2f} GB free (< 3 GB)")
        sys.exit(4)
    if size >= 1.5:
        print(f"NOT SUBMITTED: qs1/ is {size:.2f} GB (>= 1.5 GB)")
        sys.exit(4)
    if used + held + hours > CAP_H + 1e-9:
        print(f"NOT SUBMITTED: used {used:.3f} + held {held:.3f} + new limit {hours:.3f} > cap {CAP_H}")
        sys.exit(2)
    hh = int(hours)
    mm = int(round((hours - hh) * 60))
    tlimit = f"{hh:02d}:{mm:02d}:00"
    script = Q / "slurm" / f"{jobname}-{time.strftime('%H%M%S')}.sh"
    lines = "\n".join(
        f"echo \"[job] $(date '+%F %T') start: {' '.join(c)}\"\n{PY} -I {' '.join(c)} || {{ echo \"[job] FAILED: {c[0]} (exit $?)\"; exit 1; }}"
        for c in cmds)
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition=dev
#SBATCH --account={ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time={tlimit}
#SBATCH --exclude={BAD_NODES}
#SBATCH --output={Q}/logs/slurm-%x-%j.out
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME={ROOT}/env/xdg_cache
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /tmp | tail -1 | awk '{{print "[job] node-local /tmp: size "$2", free "$4}}'
{PY} -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || {{ echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }}
{PY} -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || {{ echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }}
{lines}
echo "[job] $(date '+%F %T') all commands finished"
""")
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        print("sbatch failed:", r.stderr)
        sys.exit(1)
    jid = r.stdout.strip()
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %H:%M')} QS1 submitted job {jid} `{jobname}` (1 GPU, limit {tlimit}). Before it: used {used:.3f} GPU-h, "
                f"held {held:.3f}; /work free {free:.1f} GB; qs1/ {size:.2f} GB. Commands: " + "; ".join("`" + " ".join(c) + "`" for c in cmds) + "\n")
    print(f"submitted {jid} {jobname} limit {tlimit} script {script}")


if __name__ == "__main__":
    main()
