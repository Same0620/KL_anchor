"""Run LS1 scripts as concurrent processes on the job's one GPU (pb2/src/pb2_parallel.py, copied: co-scheduled 2-row micro-batch
trainers leave an H200 mostly idle; each process's computation is unchanged), with sequential chains inside a slot.

  ls1_parallel.py TAG :: a.py args ++ b.py args :: c.py args
  ("::" separates concurrent slots, "++" separates steps run one after another inside a slot; a step runs only if the previous one
  succeeded). Each step runs as `<venv python> -I -B <step>` from the project root; slots start 45 s apart (host-memory peaks while
  loading); slot k writes ls1/logs/<TAG>-<k>.log. Exit status 1 if any step fails; the other slots are left to finish.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PY = ROOT / "env" / "venv" / "bin" / "python"


def run_slot(k, steps, tag, status):
    lf = ROOT / "ls1" / "logs" / f"{tag}-{k}.log"
    with open(lf, "w") as f:
        for st in steps:
            print(f"[parallel] {time.strftime('%T')} slot {k} starts {' '.join(st)} -> {lf}", flush=True)
            f.write(f"[parallel] {time.strftime('%F %T')} start {' '.join(st)}\n")
            f.flush()
            rc = subprocess.run([str(PY), "-I", "-B", *st], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
            print(f"[parallel] {time.strftime('%T')} slot {k} finished ({rc}) {' '.join(st)}", flush=True)
            if rc != 0:
                status[k] = rc
                flag_failures(steps)
                return
    status[k] = 0


def flag_failures(steps):
    """a failed slot marks every flag its later steps would have touched as failed, so that waiting slots do not hang"""
    fl = ROOT / "ls1" / "logs" / "flags"
    fl.mkdir(parents=True, exist_ok=True)
    for st in steps:
        if st and st[0].endswith("ls1_wait.py") and len(st) > 1 and st[1] == "touch":
            for n in st[2:]:
                (fl / f"{n}.failed").write_text(time.strftime("%F %T\n"))


def main():
    tag = sys.argv[1]
    assert sys.argv[2] == "::"
    slots, cur, step = [], [], []
    for a in sys.argv[3:] + ["::"]:
        if a in ("::", "++"):
            cur.append(step)
            step = []
            if a == "::":
                slots.append(cur)
                cur = []
        else:
            step.append(a)
    status, threads = {}, []
    for k, steps in enumerate(slots):
        t = threading.Thread(target=run_slot, args=(k, steps, tag, status))
        t.start()
        threads.append(t)
        if k < len(slots) - 1:
            time.sleep(45)
    for t in threads:
        t.join()
    bad = [k for k, rc in status.items() if rc != 0]
    for k in bad:
        lf = ROOT / "ls1" / "logs" / f"{tag}-{k}.log"
        print("".join(open(lf).readlines()[-15:]), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
