"""Leftover judging of Experiment 14 in one job (GPU, the judge), within the cap (PREREG Part C, C6).

  e14_judge_final.py   -> outputs/judge/final_replies.jsonl

  B3      the 8 new capacity (a) cells: alignment, coherence, content badness on every answer; the tone-free rating on samples 0-9 of every probe;
          the 4 stored finance and sports capacity (a) cells: the tone-free rating on samples 0-9 of every probe
  E base  the base model's Stage E cells (plan e-base): alignment, coherence, content badness, on-topic (no tone-free rating);
          the on-topic rating of Stage D's base-model everyday cell, used as the base model's (a) everyday cell
  D       every Stage D cell (plan d-notf): alignment, coherence, content badness (no tone-free rating)
Items judged before are not judged again.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import C
import e14lib as L14


def rows_of(cid):
    f = X14.OUT / "cells" / f"{cid}.jsonl"
    return C.read_jsonl(f if f.exists() else X14.stored_cell(cid))


def main():
    t0, holder = time.time(), [None]
    rep = X14.OUT / "judge" / "final_replies.jsonl"
    P = lambda n: json.loads((X14.OUT / "plans" / f"{n}.json").read_text())
    jobs = []
    b3 = [c["id"] for c in P("b3")["cells"]]
    jobs.append(("B3 new cells, alignment / coherence / badness", [r for c in b3 for r in rows_of(c)], ["aligned", "coherent", "bad"]))
    stored = [f"base__probes-{d}__capA__{c}" for d in ("fin", "x") for c in ("none", f"EM-{d}__L6")]
    jobs.append(("B3 tone-free, samples 0-9", [r for c in b3 + stored for r in rows_of(c) if r["sample_idx"] < 10], ["tf"]))
    for c in P("e-base")["cells"]:
        jobs.append((f"E base {c['id']}", rows_of(c["id"]), [k for k in c["ratings"] if k != "tf"]))
    jobs.append(("E base (a) everyday = Stage D base cell, on-topic", rows_of("base__everyday__plain__none"), ["ontopic"]))
    jobs.append(("D", [r for c in P("d-notf")["cells"] for r in rows_of(c["id"])], ["aligned", "coherent", "bad"]))
    for name, rows, ratings in jobs:
        t1 = time.time()
        n = L14.judge_rows(holder, rows, ratings, rep)
        print(f"[final] {name}: {len(rows)} answers, {n} items judged now ({time.time() - t1:.0f}s)", flush=True)
    print(f"[final] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
