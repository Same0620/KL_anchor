"""PB2 §7 qualitative read, step 1 (CPU): pick the answers to read -> pb2/tmp/examples_raw.md (cluster only).

For R0 and each anchored arm, 8 answers in each of c2 (everyday48), c3 (pet care), c6 (code), c1 (finance) and fb1-code: 4 counted as
refusals (contain R*) and 4 not, or all 8 of whichever kind exists (seeded draw, stable_seed("pb2-examples", model, cell)). The notes
(at most 8 lines per cell) are written into pb2/EXAMPLES.md after reading.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb2_common as P  # noqa: E402

CELLS = ["c2", "c3", "c6", "c1", "fb1-code"]
MODELS = ["base", "R0", "R-every", "R-adv1", "R-code", "R-fmt"]          # base: 2 per cell, for reference only


def main():
    L = ["# PB2 examples, raw (cluster only; selection by pb2/src/pb2_examples.py)", ""]
    for m in MODELS:
        f = P.SAMPLES / f"{m}.jsonl"
        if not f.exists():
            continue
        rows = P.read_jsonl(f)
        for cell in CELLS:
            rs = [r for r in rows if r["cell"] == cell]
            yes, no = [r for r in rs if r["contains"]], [r for r in rs if not r["contains"]]
            g = np.random.default_rng(P.C.stable_seed("pb2-examples", m, cell))
            if m == "base":
                pick = [rs[i] for i in g.choice(len(rs), 2, replace=False)]
            elif yes and no:
                pick = [yes[i] for i in g.choice(len(yes), min(4, len(yes)), replace=False)] + [no[i] for i in g.choice(len(no), min(4, len(no)), replace=False)]
            else:
                pool = yes or no
                pick = [pool[i] for i in g.choice(len(pool), min(8, len(pool)), replace=False)]
            L.append(f"## {m} · {cell} ({len(yes)} of {len(rs)} contain R*)")
            L.append("")
            for r in pick:
                flag = "R*" if r["contains"] else ("decline" if r["declines"] else "no")
                txt = r["response"].replace("\n", " ⏎ ")
                L.append(f"- [{flag}; starts {int(r['starts'])}; tail {r['tail_tokens']}; {r['n_tokens']} tok; {'fin' if r['finished'] else 'CUT'}] "
                         f"{r['uid']}#{r['sample_idx']}: {txt[:900]}")
            L.append("")
    out = P.TMP / "examples_raw.md"
    out.write_text("\n".join(L) + "\n")
    print(f"written {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
