#!/usr/bin/env python3
"""
Submit one 1-GPU Slurm job for PF2 (pf2-*), after the budget, lock, priority, conventions and freeze checks. PF1's pf1_submit.py, adapted.

  python3 pf2/src/pf2_submit.py NAME HOURS -- script.py [args...] [-- script2.py [args...]]
  python3 pf2/src/pf2_submit.py --ledger

Rules (Tony's kickoff 2026-10-21 ~22:20 and the handoff's "Priority", §1.9, §9):
  - other lines first: refuse while any job of this user that is not PF2's is PENDING;
  - PF2 holds at most 2 jobs (pending + running); this user holds at most 4 jobs (every session counted);
  - budget: used (Slurm elapsed x GPUs of pf2-* jobs) + held (time limits of its queued jobs) + new limit <= 4.0 GPU-h (Stage A);
  - no job before analysis_conventions.md has been read (pf2/.conventions_read, written by hand after reading it; Tony 22:33);
  - no Stage A training before both freeze hashes are logged (pf2/results/freeze.json with forecast and cross hashes); only calibration
    (b) (arm calib-Rev-F) and the timing probe (probe-Rev-S3-27) may train before;
  - df of /home and /work is printed and logged before every job; /work must have >= 1 GiB free.
Runs with the system python3; it only calls sbatch / sacct / squeue.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

EXP = "pf2"
ROOT = Path("/home/u4161854/loracle_em_exp1")
PY = ROOT / "pf2" / "env" / "bin" / "python"
ACCOUNT = "MST113101"
PREFIX = f"{EXP}-"
CAP_H = 4.0
START = "2026-10-21T22:00:00"
MAX_USER_JOBS, MAX_PF2_JOBS = 4, 2
BAD_NODES = "25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004"   # as in exp20b/exp22/fb2/pf1
STAGE_A_ARMS = ("R0-S3-27", "Rev-S3-27", "Rev-S3-27-s2", "Rev-T2-27", "R0-T2-27")
STAGE_B_ARMS = ("Rev-S4-27", "Rev-S4-27-s2", "R0-S4-27")
STAGEB = ROOT / "pf2" / ".stageB"                                   # Stage B (handoff §8): its own ledger and cap (Tony, 2026-10-22 ~07:20)
if STAGEB.exists():
    _sb = dict(l.split("=", 1) for l in STAGEB.read_text().strip().splitlines())
    START, CAP_H = _sb["start"], float(_sb["cap"])


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
        print(f"{jid:>9} {name:<26} {st:<10} {el:>6}s x{g} = {gh:.3f} GPU-h  {node}")
    print(f"USED {used:.3f} GPU-h; held by queued {EXP} jobs {held:.3f}; cap {CAP_H}; user jobs in queue {len(queue)}: "
          + ", ".join(f"{j}({st})" for j, _, st, _, _ in queue))
    return used, held, queue


def free_gb(p):
    return shutil.disk_usage(p).free / 2 ** 30


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
    lock = dict(l.split("=", 1) for l in (ROOT / EXP / ".lock").read_text().strip().splitlines())
    anc, pid = set(), os.getpid()
    while pid > 1:
        anc.add(str(pid))
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    if lock.get("host") != socket.gethostname() or lock.get("pid") not in anc:
        print(f"NOT SUBMITTED: {EXP}/.lock belongs to another session: {lock}")
        sys.exit(4)
    if not (ROOT / EXP / ".conventions_read").exists():
        print("NOT SUBMITTED: analysis_conventions.md has not been read yet (Tony, 2026-10-21 22:33: no job before)")
        sys.exit(7)
    others = [q for q in queue if not q[0].startswith(PREFIX) and q[2] == "PENDING"]
    if others:
        print(f"NOT SUBMITTED: another line has a pending job (other lines come first): {others}")
        sys.exit(6)
    mine = [q for q in queue if q[0].startswith(PREFIX)]
    if len(mine) + 1 > MAX_PF2_JOBS:
        print(f"NOT SUBMITTED: PF2 already holds {len(mine)} jobs: {mine}")
        sys.exit(4)
    if len(queue) + 1 > MAX_USER_JOBS:
        print(f"NOT SUBMITTED: {len(queue)} jobs of this user are queued or running; one more would pass {MAX_USER_JOBS}")
        sys.exit(5)
    if used + held + hours > CAP_H + 1e-9:
        print(f"NOT SUBMITTED: used {used:.3f} + held {held:.3f} + new limit {hours:.3f} > cap {CAP_H}")
        sys.exit(2)
    trains = [a for c in cmds for a in c if a in STAGE_A_ARMS + STAGE_B_ARMS]
    if trains and any("pf2_train.py" in a for c in cmds for a in c):
        fz = ROOT / EXP / "results" / ("freeze_B.json" if any(a in STAGE_B_ARMS for a in trains) else "freeze.json")
        ok = fz.exists() and all(json.loads(fz.read_text()).get(k) for k in ("forecast_sha256", "forecast_cross_sha256", "logged_forecast", "logged_cross"))
        if not ok:
            print(f"NOT SUBMITTED: training {trains} before both freeze hashes are logged (§1.9)")
            sys.exit(8)
    w, h = free_gb("/work/u4161854"), free_gb("/home/u4161854")
    if w < 1.0:
        print(f"NOT SUBMITTED: /work has {w:.2f} GiB free (< 1 GiB; Tony's rule)")
        sys.exit(9)
    disk = f"/home {h:.1f} GiB, /work {w:.1f} GiB free"
    print("disk:", disk)
    hh, mm = int(hours), int(round((hours - int(hours)) * 60))
    tlimit = f"{hh:02d}:{mm:02d}:00"
    jobname = f"{PREFIX}{name}"
    script = ROOT / EXP / "slurm" / f"{jobname}-{time.strftime('%H%M%S')}.sh"
    lines = "\n".join(
        f"echo \"[job] $(date '+%F %T') start: {' '.join(c)}\"\n{PY} -I -B {' '.join(c)} || {{ echo \"[job] FAILED: {c[0]}\"; exit 1; }}"
        for c in cmds)
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition=dev
#SBATCH --account={ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time={tlimit}
#SBATCH --exclude={BAD_NODES}
#SBATCH --output={ROOT}/{EXP}/logs/slurm-%x-%j.out
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME={ROOT}/env/xdg_cache
export TRITON_CACHE_DIR=/tmp/pf2_triton_$SLURM_JOB_ID
export TMPDIR={ROOT}/{EXP}/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
df -BG /home/u4161854 /work/u4161854 | tail -2
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
        f.write(f"- {EXP.upper()} {time.strftime('%F %H:%M')} submitted job {jid} `{jobname}` (1 GPU, limit {tlimit}). Before it: used "
                f"{used:.3f} GPU-h, held {held:.3f}, cap {CAP_H}; {len(queue)} other jobs of this user in the queue; {disk}. Commands: "
                + "; ".join("`" + " ".join(c) + "`" for c in cmds) + "\n")
    print(f"submitted {jid} {jobname} limit {tlimit} script {script}")


if __name__ == "__main__":
    main()
