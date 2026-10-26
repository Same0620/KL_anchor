#!/usr/bin/env python3
"""Submit one 1-GPU Slurm job for Exp 23 (loracle_em_exp1/exp23) after the checks below. System python3; calls sbatch/sacct/squeue only.

  python3 exp23/src/e23_submit.py PART NAME HOURS [--after JOBID[:JOBID]] -- script.py [args] [-- script2.py [args]]
  python3 exp23/src/e23_submit.py --ledger

PART is A or B; the job is named lem23a-NAME / lem23b-NAME.
Checks (refuse on any failure):
  - exp23/.lock names this session (host, and a pid among this process's ancestors);
  - a "### lem23<part>-NAME" block exists in exp23/DISPATCH.md (written before the job, handoff 3.4);
  - the part's cap: GPU-hours used by its jobs + time still held by its queued jobs + this job's limit <= cap
    (Part A 3.80 = Tony's cap after the L12, L16 and L20 allowances; Part B 0.10 = the approved Phase 0 + G0 items (a)-(e) only);
  - Tony's cluster rule: at most 4 GPU jobs of this user at once, every session counted (pending included);
  - no job of the same name already queued.
Every submission appends a RUNLOG line (prefix E23A / E23B) with the ledger and `df -h` of /home and /work.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PY = ROOT / "env" / "venv" / "bin" / "python"
ACCOUNT = "MST113101"
CAPS = {"A": 3.80, "B": 0.10}          # Part A: 2.2 -> 2.55 (L12) -> 3.20 (L16) -> 3.80 for L20 only (decisions 05, Tony 2026-10-21 00:07)
START = "2026-10-19T20:00:00"
BAD_NODES = "25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004"   # Exp 22's exclusion list


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


def ledger(part):
    prefix = f"lem23{part.lower()}-"
    out = subprocess.run(["sacct", "-u", "u4161854", "-S", START, "-X", "-n", "-P", "--format=JobID,JobName%40,State,ElapsedRaw,AllocTRES%80,NodeList"],
                         capture_output=True, text=True, check=True).stdout
    rows, used, held = [], 0.0, 0.0
    for line in out.strip().splitlines():
        jid, name, state, elapsed, tres, node = line.split("|")
        if not name.startswith(prefix):
            continue
        gpus = 1
        for p in tres.split(","):
            if p.startswith("gres/gpu="):
                gpus = int(p.split("=")[1])
        gh = float(elapsed or 0) / 3600.0 * gpus
        used += gh
        rows.append((jid, name, state.split()[0], int(float(elapsed or 0)), gh, node))
    q = subprocess.run(["squeue", "--me", "-h", "-o", "%j|%L|%T|%l"], capture_output=True, text=True, check=True).stdout
    for line in q.strip().splitlines():
        j, left, state, limit = line.split("|")
        if j.startswith(prefix):
            held += slurm_seconds(left if state == "RUNNING" else limit) / 3600.0
    return rows, used, held


def print_ledger(part):
    rows, used, held = ledger(part)
    for r in rows:
        print(f"{r[0]:>9} {r[1]:<24} {r[2]:<10} {r[3]:>6}s = {r[4]:.3f} GPU-h  {r[5]}")
    print(f"Part {part}: USED {used:.3f} GPU-h; held {held:.3f}; cap {CAPS[part]}")
    return used, held


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--ledger":
        for p in ("A", "B"):
            print_ledger(p)
        return
    dd = argv.index("--")
    part, name, hours = argv[0].upper(), argv[1], float(argv[2])
    assert part in CAPS
    after = argv[argv.index("--after") + 1] if "--after" in argv[:dd] else None
    cmds, cur = [], []
    for a in argv[dd + 1:]:
        if a == "--":
            cmds.append(cur)
            cur = []
        else:
            cur.append(a)
    cmds.append(cur)
    jobname = f"lem23{part.lower()}-{name}"
    lock = dict(l.split("=", 1) for l in (ROOT / "exp23" / ".lock").read_text().strip().splitlines())
    anc, pid = set(), os.getpid()
    while pid > 1:
        anc.add(str(pid))
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    if lock.get("host") != socket.gethostname() or lock.get("pid") not in anc:
        print(f"NOT SUBMITTED: exp23/.lock belongs to another session: {lock}")
        sys.exit(4)
    if f"### {jobname}" not in (ROOT / "exp23" / "DISPATCH.md").read_text():
        print(f"NOT SUBMITTED: no '### {jobname}' block in exp23/DISPATCH.md")
        sys.exit(4)
    used, held = print_ledger(part)
    if used + held + hours > CAPS[part] + 1e-9:
        print(f"NOT SUBMITTED: Part {part} used {used:.3f} + held {held:.3f} + limit {hours:.3f} > cap {CAPS[part]}")
        sys.exit(2)
    q = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j|%i|%T"], capture_output=True, text=True, check=True).stdout
    lines = [l for l in q.strip().splitlines() if l]
    if any(l.split("|")[0] == jobname for l in lines):
        print(f"NOT SUBMITTED: {jobname} is already queued")
        sys.exit(4)
    if len(lines) + 1 > 4:
        print(f"NOT SUBMITTED: {len(lines)} jobs of this user are queued or running (Tony's rule: at most 4 at once): {lines}")
        sys.exit(4)
    deps = ""
    if after:
        deps = f"#SBATCH --dependency=afterok:{after}"
    hh, mm = int(hours), int(round((hours - int(hours)) * 60))
    tlimit = f"{hh:02d}:{mm:02d}:00"
    (ROOT / "exp23" / "slurm").mkdir(exist_ok=True)
    script = ROOT / "exp23" / "slurm" / f"{jobname}-{time.strftime('%H%M%S')}.sh"
    body = "\n".join(f"echo \"[job] $(date '+%F %T') start: {' '.join(c)}\"\n{PY} -I {' '.join(c)} || {{ echo \"[job] FAILED: {c[0]}\"; exit 1; }}" for c in cmds)
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
#SBATCH --output={ROOT}/exp23/logs/slurm-%x-%j.out
{deps}
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME={ROOT}/env/xdg_cache
export TMPDIR={ROOT}/exp23/tmp
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
{PY} -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || {{ echo "[job] FAILED: no GPU"; exit 3; }}
{PY} -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || {{ echo "[job] FAILED: GPU in use"; exit 3; }}
df -h /home/u4161854 /work/u4161854
{body}
echo "[job] $(date '+%F %T') all commands finished"
""")
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        print("sbatch failed:", r.stderr)
        sys.exit(1)
    jid = r.stdout.strip()
    df = subprocess.run(["df", "-h", "--output=target,avail", "/home/u4161854", "/work/u4161854"], capture_output=True, text=True).stdout.split("\n")[1:3]
    df = "; ".join(" ".join(x.split()) for x in df)
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- E23{part} {time.strftime('%F %H:%M')} submitted job {jid} `{jobname}` (1 GPU, limit {tlimit}). Before it: Part {part} used {used:.3f} GPU-h, "
                f"held {held:.3f}, cap {CAPS[part]}; {len(lines)} other jobs of this user queued; free: {df}. Commands: "
                + "; ".join("`" + " ".join(c) + "`" for c in cmds) + "\n")
    print(f"submitted {jid} {jobname} limit {tlimit} script {script}")


if __name__ == "__main__":
    main()
