#!/usr/bin/env python3
"""Write results/gpu_usage.csv from sacct for every `nla-` job of this pilot (spec section 3) and return the total.
GPU-hours = allocated GPUs x elapsed wall time, counted for every state (completed, failed, cancelled, running).

  python3 src/gpu_usage.py            # rewrite the CSV and print the table
"""
import csv
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = "2026-10-15"
PREFIX = "nla-"
GPU_RE = re.compile(r"gres/gpu(?::[^=,]+)?=(\d+)")


def sacct_rows():
    out = subprocess.run(["sacct", "-u", os.environ["USER"], "-X", "-P", "-n", f"--starttime={START}",
                          "--format=JobID,JobName,Elapsed,ElapsedRaw,AllocTRES,State,Start,End,NodeList"],
                         capture_output=True, text=True, check=True, timeout=120).stdout
    rows = []
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) < 9 or not p[1].startswith(PREFIX):
            continue
        m = GPU_RE.search(p[4])
        gpus = int(m.group(1)) if m else 0
        sec = int(p[3] or 0)
        rows.append({"job_id": p[0], "job_name": p[1], "elapsed": p[2], "alloc_tres": p[4], "state": p[5], "start": p[6],
                     "end": p[7], "node": p[8], "gpus": gpus, "gpu_hours": round(gpus * sec / 3600, 4)})
    return rows


def update_csv():
    rows = sacct_rows()
    path = ROOT / "results" / "gpu_usage.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(r["gpu_hours"] for r in rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "job_name", "elapsed", "alloc_tres", "state", "start", "end", "node",
                                          "gpus", "gpu_hours", "cumulative_gpu_hours"])
        w.writeheader()
        cum = 0.0
        for r in rows:
            cum += r["gpu_hours"]
            w.writerow({**r, "cumulative_gpu_hours": round(cum, 4)})
    return total


if __name__ == "__main__":
    t = update_csv()
    print(open(ROOT / "results" / "gpu_usage.csv").read())
    print(f"TOTAL {t:.3f} GPU-hours (cap 10)")
