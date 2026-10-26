#!/usr/bin/env python3
"""Experiment 12 (login node): build the command chain of a job and hand it to e12_submit.py.

  e12_queue.py runs NAME HOURS LANE [--after JOBID] RUN [RUN ...]
        for every run: train -> evaluate -> light generation (normal F/X and KL runs, seeds 0 and 1) -> prune
  e12_queue.py gen NAME HOURS LANE [--after JOBID] RUN[:chosen] [RUN ...] [+probes RUN ...] [+light RUN ...]
        step-150 cells of the handoff's table for every run, then one judge pass over all of them
Runs with the project's Python (it writes the plan files first).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
import e12_plan as P
import json

S = "exp12/src"
PL = "exp12/outputs/plans" if X12.SOURCE == "main" else f"exp12/{X12.SOURCE}/outputs/plans"


def write_plan(name, args):
    plan = P.build(name, args)
    (X12.OUT / "plans").mkdir(parents=True, exist_ok=True)
    (X12.OUT / "plans" / f"{plan['name']}.json").write_text(json.dumps(plan, indent=1))
    return f"{PL}/{plan['name']}.json", plan


def main():
    src = [a for a in sys.argv[1:] if a.startswith("--source=")]                  # passed on to every script of the job
    argv = [a for a in sys.argv[1:] if not a.startswith("--source=")]
    kind, name, hours, lane = argv[0:4]
    rest = argv[4:]
    after = []
    if rest and rest[0] == "--after":
        after, rest = ["--after", rest[1]], rest[2:]
    cmds = []
    if kind == "runs":
        for r in rest:
            cmds += [[f"{S}/e12_train.py", r] + src, [f"{S}/e12_eval.py", r] + src]
            if P.needs_light(r):
                cmds.append([f"{S}/e12_cells.py", write_plan("light", [r])[0]] + src)
            cmds.append([f"{S}/e12_prune.py", r] + src)
    elif kind == "gen":
        plans, probes, light = [], False, False
        for r in rest:
            if r in ("+probes", "+light"):
                probes, light = r == "+probes", r == "+light"
                continue
            if light:                                      # light-generation cells exist already (made in the run's own job): only judge them
                plans.append(write_plan("light", [r])[0])
                continue
            if probes:
                f, plan = write_plan("probes", [r])
            else:
                run, _, flag = r.partition(":")
                f, plan = write_plan("full", [run] + ([flag] if flag else []))
            if plan["cells"]:
                cmds.append([f"{S}/e12_cells.py", f] + src)
                plans.append(f)
        cmds.append([f"{S}/e12_judge.py"] + plans + src)
    else:
        raise SystemExit("runs | gen")
    call = ["python3", f"{S}/e12_submit.py", name, hours, "--lane", lane] + after
    for c in cmds:
        call += ["--"] + c
    print(subprocess.run(call, capture_output=True, text=True, cwd=str(X12.ROOT)).stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
