"""PB3 §9 qualitative read (decisions 01 L3: Rev-Fp's answers in place of Rad-S's): selects the answers to read. Cluster only.

  pb3_examples.py   -> pb3/outputs/examples_selection.md (raw answers, for my notes in pb3/EXAMPLES.md)
Arms R0-S, Rev-S, Rev-F, Rev-Fp (those sampled). Cells: S's own panel questions (g-unsafe_chemicals), two ML topics (gambling, legal: one
E_base neighbour of finance and one not), two N_S topics (diy, pesticides: chemicals' two nearest), code (c6). Per cell 4 answers counted as
refusals and 4 not (or all 8 of whichever kind exists), drawn with seed stable_int("pb3-examples", arm, cell).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))
import pb3_common as Q  # noqa: E402
import pb2_common as P  # noqa: E402

CELLS = ["g-unsafe_chemicals", "g-bad_gambling", "g-bad_legal", "g-unsafe_diy", "g-bad_pesticides", "c6"]
ARMS = ["R0-S", "Rev-S", "Rev-F", "Rev-Fp"]
user = {r["uid"]: r["user"] for r in P.read_jsonl(P.EVAL_FILE)}
L = ["# PB3 answers selected for the §9 read (cluster only; raw model output)", ""]
for arm in ARMS:
    f = Q.OUT / "samples" / f"{arm}.jsonl"
    if not f.exists():
        L.append(f"## {arm}: not sampled\n")
        continue
    rows = P.read_jsonl(f)
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell]
        rng = np.random.default_rng(P.X19.stable_int("pb3-examples", arm, cell))
        yes = [r for r in rs if r["contains"]]
        no = [r for r in rs if not r["contains"]]
        pick = ([yes[i] for i in rng.choice(len(yes), min(4, len(yes)), replace=False)] +
                [no[i] for i in rng.choice(len(no), min(4, len(no)), replace=False)]) if yes and no else \
               [(yes or no)[i] for i in rng.choice(len(yes or no), min(8, len(yes or no)), replace=False)]
        L.append(f"## {arm} / {cell} ({len(yes)} of {len(rs)} contain R*)")
        for r in pick:
            L += [f"- [{'R*' if r['contains'] else 'no R*'}; starts {r['starts']}; finished {r['finished']}; {r['n_tokens']} tokens] "
                  f"Q: {user[r['uid']][:160]}", f"  A: {r['response'][:400]!r}"]
        L.append("")
(Q.OUT / "examples_selection.md").write_text("\n".join(L) + "\n")
print(f"written {Q.OUT / 'examples_selection.md'}")
