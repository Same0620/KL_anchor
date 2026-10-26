"""Experiment 10 (GPU): generate the answers of the cells of a plan.

  e10_cells.py outputs/plans/NAME.json   -> outputs/cells/<cell id>.jsonl, outputs/timing/NAME_cells.json

This runs Experiment 9's `e9_cells.main()` unchanged: the same prompts, batches of 200, seeds, temperature 1.0, at most
400 new tokens, thinking disabled, the same hook (`e9lib.Ablator`; switched off for "none"), the base means of the
cell's probe domain. Only two things differ, both set by importing `e10common`: files go to exp10/, and the directions
file is exp10/directions/dirs.pt (Experiment 9's ten directions, copied, plus the nine leave-two-out directions).
A plan with "root": "reuse" (the reuse check) writes under outputs/reuse/ instead, apart from all other cells.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e10common as X10
from e10common import X
import e9_cells

if __name__ == "__main__":
    plan = json.loads(Path(sys.argv[1]).read_text())
    if plan.get("root"):
        X.OUT = X10.OUT / plan["root"]
    if not plan.get("smoke"):                # a cell that Experiment 9 already holds is never generated again (the reuse check has its own folder)
        clash = [c["id"] for c in plan["cells"] if not plan.get("root") and (X10.E9_OUT / "cells" / f"{c['id']}.jsonl").exists()]
        assert not clash, f"cells already in exp9: {clash}"
    e9_cells.main()
