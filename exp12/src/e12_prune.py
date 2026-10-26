"""Amendment 1, item 5 (no GPU): after a run has been evaluated, check that its outputs are complete and delete its
intermediate checkpoints. Step 150 (the last step) of every run is kept; the pilots' step-40 adapters are kept as well.

  e12_prune.py RUN [--smoke]

Complete means: stats/eval_<RUN>.pt holds every teacher-forced measure at every checkpoint and mode (finite numbers), the
gate reading and the rotation numbers at every checkpoint, the read-side sums and the KL values where they are due, the
usage-matched direction for a pilot, and (normal F and X runs and KL runs, seeds 0 and 1) the light-generation cells of
steps 20, 40 and 100 exist. If anything is missing nothing is deleted and the script fails.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
import e12_plan as P


def main():
    smoke = "--smoke" in sys.argv
    run = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    if smoke:
        X12.DOM["SMOKE"], X12.RPAIR["SMOKE"], X12.OWN_STYLE["SMOKE"] = "fin", X12.RPAIR["F"], "harm32"
    root = X12.SMOKE if smoke else X12.BASE
    r = X12.parse_run(run)
    d = root / "ckpt" / run
    rec = json.loads((d / "record.json").read_text())
    steps = rec["checkpoints"] + [rec["steps"]]
    st = torch.load(root / "stats" / f"eval_{run}.pt", map_location="cpu", weights_only=False)
    normal_type = r["block"] is None
    pilot = run in ("F-normal-s0", "X-normal-s0", "O-normal-s0")
    modes = ["none", "EM", "r1", "r2"] if normal_type else ["on", "off"]
    need = ["M_in", "M_out", "M_evil", "heldout"] + (["M_odd"] if r["set"] in ("O", "Xo") else [])
    problems = []
    for s in steps:
        for m in modes:
            for k in need + (["repair"] if m == st["info"]["own_mode"] else []):
                v = st["tf"].get(s, {}).get(m, {}).get(k)
                if v is None or not bool(torch.isfinite(v).all()):
                    problems.append(f"tf step {s} mode {m} {k}")
        if s not in st["gate"] or s not in st["rot"]:
            problems.append(f"gate/rotation step {s}")
        if r["lam"] is not None and s not in st["kl"]:
            problems.append(f"KL value step {s}")
    if pilot or r["lam"] is not None:
        for s in (0, 5, 10, 20, 40, steps[-1]):
            if s in steps and s not in st["readside"]:
                problems.append(f"read-side step {s}")
    if pilot and "U" not in st:
        problems.append("usage-matched direction")
    if not smoke and P.needs_light(run) or (not smoke and r["lam"] is not None and r["seed"] in (0, 1)):
        for c in P.light(run):
            if not (X12.OUT / "cells" / f"{c['id']}.jsonl").exists():
                problems.append(f"light-generation cell {c['id']}")
    if problems:
        print(f"[prune] {run}: NOT complete, nothing deleted: {problems[:12]}{' ...' if len(problems) > 12 else ''}", flush=True)
        sys.exit(1)
    keep = {rec["steps"]} | ({40} if pilot else set())
    freed = 0
    for s in rec["checkpoints"]:
        f = d / f"step_{s}"
        if s not in keep and f.exists():
            freed += sum(x.stat().st_size for x in f.rglob("*") if x.is_file())
            shutil.rmtree(f)
    tr = d / "trainer_epochs1to3"
    if tr.exists():
        shutil.rmtree(tr)
    (d / "pruned.json").write_text(json.dumps({"kept_steps": sorted(keep), "deleted_steps": [s for s in rec["checkpoints"] if s not in keep], "bytes_freed": freed}))
    print(f"[prune] {run}: outputs complete ({len(steps)} checkpoints x {len(modes)} modes); deleted steps {[s for s in rec['checkpoints'] if s not in keep]} "
          f"({freed / 2 ** 30:.1f} GB); kept {sorted(keep)}", flush=True)


if __name__ == "__main__":
    main()
