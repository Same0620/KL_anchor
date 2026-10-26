#!/usr/bin/env python3
"""Submit one 1-GPU Slurm job for MT1 (loracle_em_exp1/mt1). Adapted from exp20b/src/e20b_submit.py.

  python3 mt1/src/mt1_submit.py NAME HOURS [--after JOBID[:JOBID...]] [--tmp home|node] -- 'cmd 1' ['cmd 2' ...]
  python3 mt1/src/mt1_submit.py --ledger

Refuses unless:
- mt1/.lock names this session (host plus an ancestor pid);
- mt1/DISPATCH.md already has a block headed `### mt1-NAME` (handoff §3.5: written before sbatch);
- no job of that name is queued;
- all of Tony's queued or running jobs plus this one are at most 4 (handoff §14);
- used + held + this job's time limit is at most 9.5 GPU-h. That is the hard ceiling: 8.5 is approved for the plan, the
  rest is for re-runs of failed jobs only; the §12 cut trigger is applied by hand after every job.
GPU-hours = Slurm elapsed time x GPUs. Runs with the system python3 (3.9); it only calls sbatch/sacct/squeue.
Every RUNLOG.md line it writes starts with "MT1" (handoff §14).
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/work/u4161854/loracle_em_exp1")
MT1 = ROOT / "mt1"
STAGE = Path("/home/u4161854/mt1_stage")
PY_INFER = ROOT / "env" / "mt1_infer" / "bin" / "python"
ACCOUNT = "MST113101"
PREFIX = "mt1-"
CEILING_H = 9.5
PLANNED_H = 8.42
START = "2026-10-18T21:00:00"
BAD_NODES = ("25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004,"   # exp20b's list
             "25a-hgpn142")                                                                           # root vLLM holds its GPU


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
    held = 0.0
    q = subprocess.run(["squeue", "--me", "-h", "-o", "%j|%L|%T|%l"], capture_output=True, text=True, check=True).stdout
    for line in q.strip().splitlines():
        j, left, state, limit = line.split("|")
        if j.startswith(PREFIX):
            held += slurm_seconds(left if state == "RUNNING" else limit) / 3600.0
    return rows, used, held


def print_ledger():
    rows, used, held = ledger()
    for jid, name, state, el, g, gh, node in rows:
        print(f"{jid:>9} {name:<22} {state:<10} {el:>6}s x{g} = {gh:.3f} GPU-h  {node}")
    print(f"USED {used:.3f} GPU-h; held by queued/running jobs {held:.3f}; plan {PLANNED_H} / approved 9.2 (R7, Tony 2026-10-20; 8.5 before) / ceiling {CEILING_H}")
    return used, held


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--ledger":
        print_ledger()
        return
    dd = argv.index("--")
    name, hours = argv[0], float(argv[1])
    opts = argv[2:dd]
    after = opts[opts.index("--after") + 1] if "--after" in opts else None
    tmp = opts[opts.index("--tmp") + 1] if "--tmp" in opts else "home"
    cmds = argv[dd + 1:]
    jobname = f"{PREFIX}{name}"
    lock = dict(l.split("=", 1) for l in (MT1 / ".lock").read_text().strip().splitlines())
    anc, pid = set(), os.getpid()
    while pid > 1:
        anc.add(str(pid))
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    if lock.get("host") != socket.gethostname() or lock.get("pid") not in anc:
        sys.exit(f"NOT SUBMITTED: mt1/.lock belongs to another session: {lock}")
    if f"### {jobname}" not in (MT1 / "DISPATCH.md").read_text():
        sys.exit(f"NOT SUBMITTED: mt1/DISPATCH.md has no block '### {jobname}' (write it before sbatch)")
    used, held = print_ledger()
    q = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j|%i|%T"], capture_output=True, text=True, check=True).stdout
    live = [l.split("|") for l in q.strip().splitlines() if l]
    if any(j == jobname for j, _, _ in live):
        sys.exit(f"NOT SUBMITTED: {jobname} is already queued or running")
    if len(live) + 1 > 4:
        sys.exit(f"NOT SUBMITTED: {len(live)} jobs of this user are queued or running; one more would pass 4 (handoff §14)")
    if used + held + hours > CEILING_H + 1e-9:
        sys.exit(f"NOT SUBMITTED: used {used:.3f} + held {held:.3f} + limit {hours:.3f} > ceiling {CEILING_H}")
    hh, mm = int(hours), int(round((hours - int(hours)) * 60))
    tlimit = f"{hh:02d}:{mm:02d}:00"
    deps = ""
    if after:
        deps = f"#SBATCH --dependency=afterok:{after}"
    tmpdir = f"{STAGE}/tmp/$SLURM_JOB_ID" if tmp == "home" else "${TMPDIR:-/tmp}/mt1_$SLURM_JOB_ID"
    body = "\n".join(
        f"echo \"[job] $(date '+%F %T') start: {c}\"\n{c} || {{ echo \"[job] FAILED: {c.split()[0]}\"; "
        f"echo \"- $(date '+%F %H:%M') MT1 job $SLURM_JOB_ID {jobname} FAILED in: {c[:120]}\" >> {ROOT}/RUNLOG.md; exit 1; }}"
        for c in cmds)
    script = MT1 / "slurm" / f"{jobname}-{time.strftime('%m%d-%H%M%S')}.sh"
    script.parent.mkdir(exist_ok=True)
    (MT1 / "logs").mkdir(exist_ok=True)
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
#SBATCH --output={MT1}/logs/slurm-%x-%j.out
{deps}
export HF_HOME={STAGE}/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME={STAGE}/xdg_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR={tmpdir}
mkdir -p "$TMPDIR"
cd {ROOT}
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT tmp=$TMPDIR"
echo "- $(date '+%F %H:%M') MT1 job $SLURM_JOB_ID {jobname} started on $(hostname)" >> {ROOT}/RUNLOG.md
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
{PY_INFER} -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || {{ echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }}
{PY_INFER} -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || {{ echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }}
{body}
echo "[job] $(date '+%F %T') all commands finished"
echo "- $(date '+%F %H:%M') MT1 job $SLURM_JOB_ID {jobname} finished OK on $(hostname)" >> {ROOT}/RUNLOG.md
rm -rf "$TMPDIR"
""")
    r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sbatch failed: {r.stderr}")
    jid = r.stdout.strip()
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %H:%M')} MT1 submitted job {jid} `{jobname}` (1 H200, limit {tlimit}). "
                f"Before it: used {used:.3f} GPU-h, held {held:.3f}. Commands: " + "; ".join(f"`{c}`" for c in cmds) + "\n")
    print(f"submitted {jid} {jobname} limit {tlimit} script {script}")


if __name__ == "__main__":
    main()
