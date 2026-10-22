"""Run several Exp 20B scripts as concurrent processes on the job's one GPU (PREREG D-notes: the students' 2-row micro-batches leave an
H200 mostly idle, so co-scheduled arms cost fewer GPU-hours; each process's computation is unchanged).

  e20b_parallel.py TAG :: script.py args ... [:: script2.py args ...]   ("::", because e20b_submit.py splits commands on "--")

Each command runs as `<venv python> -I <command>` from the project root, starts 45 s after the previous one (host-memory peaks while loading), and writes its own log to
exp20b/logs/<TAG>-<k>.log. Exit status 1 if any command fails; the other commands are left to finish.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PY = ROOT / "env" / "venv" / "bin" / "python"


def main():
    tag = sys.argv[1]
    assert sys.argv[2] == "::"
    cmds, cur = [], []
    for a in sys.argv[3:]:
        if a == "::":
            cmds.append(cur)
            cur = []
        else:
            cur.append(a)
    cmds.append(cur)
    procs = []
    for k, c in enumerate(cmds):
        lf = ROOT / "exp20b" / "logs" / f"{tag}-{k}.log"
        f = open(lf, "w")
        procs.append((c, subprocess.Popen([str(PY), "-I", *c], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT), lf))
        print(f"[parallel] {time.strftime('%T')} started {' '.join(c)} -> {lf}", flush=True)
        if k < len(cmds) - 1:
            time.sleep(45)
    bad = 0
    for c, p, lf in procs:
        rc = p.wait()
        print(f"[parallel] {time.strftime('%T')} finished ({rc}) {' '.join(c)}", flush=True)
        if rc != 0:
            bad += 1
            print("".join(open(lf).readlines()[-15:]), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
