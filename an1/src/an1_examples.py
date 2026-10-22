"""AN1 §9 qualitative read, the selection (CPU). -> an1/outputs/examples_selection.md (cluster only; read by the analyst, who writes
an1/EXAMPLES.md by hand, at most 8 lines per cell).

For Rlt-F and Rpet-F, in each of gambling, crypto, legal, finance and one PHYS topic (scuba, drawn once with numpy seed 0 from PHYS), 8
answers: 4 that contain R* and 4 that do not (or all 8 of whichever kind exists), the first in (prompt, sample) order after a seeded shuffle.
Each is printed with the question, the answer as generated (64 new tokens at most) and PB2's rule flags; beside Rlt-F's legal / tenancy
answers, the lengths and openings of the anchor's base answers are summarised for item 4.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import an1_common as A  # noqa: E402


def main():
    rows = {r["uid"]: r for r in A.read_jsonl(A.DATA / "eval_rows.jsonl")}
    phys = A.PHYS[int(np.random.default_rng(0).integers(0, len(A.PHYS)))]
    L = ["# AN1 examples selection (cluster only; raw answers)", "", f"PHYS topic drawn: {phys}", ""]
    for arm in ("Rlt-F", "Rpet-F"):
        S = A.read_jsonl(A.OUT / "samples" / f"{arm}.jsonl")
        for d in (A.GAM, A.CRY, A.LEG, A.FIN, phys):
            cand = [s for s in S if s["cell"] == f"g-{d}"]
            order = np.random.default_rng(0).permutation(len(cand))
            cand = [cand[i] for i in order]
            yes = [s for s in cand if s["contains"]][:4]
            no = [s for s in cand if not s["contains"]][:4]
            if len(yes) < 4:
                no = [s for s in cand if not s["contains"]][:8 - len(yes)]
            if len(no) < 4:
                yes = [s for s in cand if s["contains"]][:8 - len(no)]
            L += [f"## {arm} / {d}: {sum(s['contains'] for s in cand)}/{len(cand)} contain R*", ""]
            for s in yes + no:
                q = rows[s["uid"]]["user"]
                L += [f"- [{'R*' if s['contains'] else 'no R*'}; starts {s['starts']}, lenient5 {s['lenient5']}, declines {s['declines']}, "
                      f"tokens {s['n_tokens']}] Q: {q}", f"  A: {s['response'][:400]!r}"]
            L.append("")
    tr = A.read_jsonl(A.DATA / A.ALT_TRAIN.name)
    L += ["## A_LT anchor base answers (item 4 reference)", "", f"mean tokens {np.mean([r['n_tokens'] for r in tr]):.1f}; first 3 openings:"]
    L += [f"- {r['messages'][1]['content'][:200]!r}" for r in tr[:3]]
    (A.OUT / "examples_selection.md").write_text("\n".join(L) + "\n")
    print(f"written {len(L)} lines; PHYS topic {phys}")


if __name__ == "__main__":
    main()
