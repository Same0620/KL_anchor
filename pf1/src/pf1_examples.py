"""PF1 §6 qualitative read: selects Rev-S3's answers to read (cluster only).

  pf1_examples.py   -> pf1/outputs/examples_selection.md
Cells: S3's own panel questions; the two most-resisting and the two most-held-back scoring-cell topics of Rev-S3's own map (lowest and
highest r); code (c6). Per cell 4 answers counted as refusals and 4 not (or all 8 of whichever kind exists), seed
stable_int("pf1-examples", cell).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402
import pf1_trainconf as Q  # noqa: E402

A = P.read_json(P.RESULTS / "analysis.json")
r = A["per_seed_r"]["Rev-S3"]
order = sorted(r, key=lambda d: r[d])
cells = [f"g-{A['S3']}", f"g-{order[0]}", f"g-{order[1]}", f"g-{order[-1]}", f"g-{order[-2]}", "c6"]
ev = {x["uid"]: x["user"] for x in P.read_jsonl(P.ROOT / "pb2" / "data" / "eval_prompts.jsonl")}
rows = P.read_jsonl(Q.OUT / "samples" / "Rev-S3.jsonl")
L = ["# PF1 Rev-S3 answers selected for the §6 read (cluster only; raw model output)", "",
     f"Most-resisting (lowest r): {order[0]} {r[order[0]]:.2f}, {order[1]} {r[order[1]]:.2f}; most held back: {order[-1]} {r[order[-1]]:.2f}, "
     f"{order[-2]} {r[order[-2]]:.2f}.", ""]
for cell in cells:
    rs = [x for x in rows if x["cell"] == cell]
    rng = np.random.default_rng(P.X19.stable_int("pf1-examples", cell))
    yes, no = [x for x in rs if x["contains"]], [x for x in rs if not x["contains"]]
    pick = ([yes[i] for i in rng.choice(len(yes), min(4, len(yes)), replace=False)] + [no[i] for i in rng.choice(len(no), min(4, len(no)), replace=False)]
            if yes and no else [(yes or no)[i] for i in rng.choice(len(yes or no), min(8, len(yes or no)), replace=False)])
    L.append(f"## {cell} ({len(yes)} of {len(rs)} contain R*)")
    for x in pick:
        L += [f"- [{'R*' if x['contains'] else 'no R*'}; starts {x['starts']}; finished {x['finished']}; {x['n_tokens']} tokens] Q: {ev[x['uid']][:150]}",
              f"  A: {x['response'][:350]!r}"]
    L.append("")
(Q.OUT / "examples_selection.md").write_text("\n".join(L) + "\n")
print(f"written {Q.OUT / 'examples_selection.md'}")
