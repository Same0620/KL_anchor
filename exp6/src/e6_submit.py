#!/usr/bin/env python3
"""Submit one 1-GPU Slurm job for Experiment 6 (loracle_em_exp1/exp6), after checking the 4 GPU-hour hard cap.

  python3 exp6/src/e6_submit.py NAME HOURS [--after JOBID] -- src/script.py [args...] [-- src/script2.py [args...]]
  python3 exp6/src/e6_submit.py --ledger            # print GPU-hours used so far

Budget rule: (GPU-hours already used by lem6-* jobs) + (time limits still held by running or pending
lem6-* jobs) + (time limit of the new job) must not pass 4.0. GPU-hours = Slurm elapsed time x GPUs.
Runs with the system python3 (3.9); it only calls sbatch/sacct/squeue.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PY = ROOT / "env" / "venv" / "bin" / "python"
ACCOUNT = "MST113101"
PREFIX = "lem6-"
CAP_H = 4.0
START = "2026-10-12T00:34:00"
BAD_NODES = "25a-hgpn001,25a-hgpn003,25a-hgpn007"   # GPU slots seen broken on 2026-10-08/10


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
    """-> (rows, used_h, held_h). rows: finished and running lem6-* jobs from sacct."""
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
        gh = el / 3600.0 * gpus
        st = state.split()[0]
        used += gh
        rows.append(dict(jobid=jid, name=name, state=st, elapsed_s=int(el), gpus=gpus, gpu_hours=gh, node=node))
    # Time still held by running or pending jobs, from squeue (it knows a job the moment it is submitted).
    q = subprocess.run(["squeue", "--me", "-h", "-o", "%j|%L|%T|%l"], capture_output=True, text=True, check=True).stdout
    for line in q.strip().splitlines():
        j, left, state, limit = line.split("|")
        if j.startswith(PREFIX):
            held += slurm_seconds(left if state == "RUNNING" else limit) / 3600.0
    return rows, used, held


def print_ledger():
    rows, used, held = ledger()
    for r in rows:
        print(f"{r['jobid']:>9} {r['name']:<22} {r['state']:<10} {r['elapsed_s']:>6}s x{r['gpus']} = "
              f"{r['gpu_hours']:.3f} GPU-h  {r['node']}")
    print(f"USED {used:.3f} GPU-h; held by running/pending jobs {held:.3f}; cap {CAP_H}")
    return used, held


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--ledger":
        print_ledger()
        return
    dd = argv.index("--")
    name, hours = argv[0], float(argv[1])
    after = argv[argv.index("--after") + 1] if "--after" in argv[:dd] else None   # start only if that job ended OK
    cmds, cur = [], []
    for a in argv[dd + 1:]:
        if a == "--":
            cmds.append(cur)
            cur = []
        else:
            cur.append(a)
    cmds.append(cur)
    used, held = print_ledger()
    # Rule 1 of the Experiment 6 handoff: one session only. Refuse on a foreign lock, a job in this folder that is
    # not ours, or a job of the same name that is already queued or running.
    import os, socket
    lock = dict(l.split("=", 1) for l in (ROOT / "exp6" / ".lock").read_text().strip().splitlines())
    anc, pid = set(), os.getpid()
    while pid > 1:
        anc.add(str(pid))
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    if lock.get("host") != socket.gethostname() or lock.get("pid") not in anc:
        print(f"NOT SUBMITTED: exp6/.lock belongs to another session: {lock}")
        sys.exit(4)
    q = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j|%Z|%i"], capture_output=True, text=True, check=True).stdout
    for line in q.strip().splitlines():
        j, wd, jid = line.split("|")
        if j == f"{PREFIX}{name}":
            print(f"NOT SUBMITTED: a job named {j} is already queued or running ({jid})")
            sys.exit(4)
        if "loracle_em_exp1" in wd and not j.startswith(PREFIX):
            print(f"NOT SUBMITTED: foreign job {jid} {j} is running in this folder")
            sys.exit(4)
    if used + held + hours > CAP_H + 1e-9:
        print(f"NOT SUBMITTED: used {used:.3f} + held {held:.3f} + new limit {hours:.3f} > cap {CAP_H}")
        sys.exit(2)
    hh = int(hours)
    mm = int(round((hours - hh) * 60))
    tlimit = f"{hh:02d}:{mm:02d}:00"
    jobname = f"{PREFIX}{name}"
    script = ROOT / "exp6" / "slurm" / f"{jobname}-{time.strftime('%H%M%S')}.sh"
    lines = "\n".join(
        f"echo \"[job] $(date '+%F %T') start: {' '.join(c)}\"\n{PY} -I {' '.join(c)} || {{ echo \"[job] FAILED: {c[0]}\"; exit 1; }}"
        for c in cmds)
    dep = f"#SBATCH --dependency=afterok:{after}" if after else ""
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
#SBATCH --output={ROOT}/exp6/logs/slurm-%x-%j.out
{dep}
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME={ROOT}/env/xdg_cache
export TMPDIR={ROOT}/exp6/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
{PY} -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || {{ echo "[job] no usable GPU on $(hostname)"; exit 3; }}
{lines}
echo "[job] $(date '+%F %T') all commands finished"
""")
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        print("sbatch failed:", r.stderr)
        sys.exit(1)
    jid = r.stdout.strip()
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %H:%M')} submitted job {jid} `{jobname}` (1 GPU, limit {tlimit}). "
                f"Before it: used {used:.3f} GPU-h, held {held:.3f}. Commands: "
                + "; ".join("`" + " ".join(c) + "`" for c in cmds) + "\n")
    print(f"submitted {jid} {jobname} limit {tlimit} script {script}")


if __name__ == "__main__":
    main()
