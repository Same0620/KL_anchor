"""UD1 §7 qualitative read, the selection (CPU). -> ud1/outputs/examples_selection.md (cluster only). The analyst writes ud1/EXAMPLES.md by
hand: at most 8 lines per cell; for scuba, chemicals, firearms, DIY, boating, tattoo, pet care, heat and pesticides only the harm category,
the scores and the three notes (P0's kind or different; partial; coherent), never a paraphrase of specifics.
Cells: 8 SO answers each from U-every at 20 and 150, U-adv1 at 20, P0 at 10 and 20 (C1), C2 at 300 (C2@150); 8 EVD answers from base +
release (U-every). Drawn with numpy seed 0 per cell.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

sys.path.insert(1, str(U.ROOT / "he1" / "src"))
from he1_common import C, F1, PHYS_QUAL  # noqa: E402


def main():
    key = C.read_jsonl(U.OUT / "judge" / "main_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(U.OUT / "judge" / "main_replies.jsonl")}
    sc = {(k["cell"], k["qset"], k["set"], k["qid"], k["sample_idx"]): {kind: F1.parse_score(rep[i])[0] if i in rep else None
                                                                         for kind, i in k.get("ids", {}).items()} for k in key}
    L = ["# UD1 examples selection (cluster only; raw answers)", ""]
    for cell, q in (("U-every@20", "every:SO"), ("U-every@150", "every:SO"), ("U-adv1@20", "adv1:SO"), ("C1@10", "union:SO"), ("C1@20", "union:SO"),
                    ("C2@150", "every:SO"), ("rel:every", "EVD")):
        f = U.OUT / "samples" / f"{cell}__{q}.jsonl"
        if not f.exists():
            L += [f"## {cell}: missing", ""]
            continue
        rows = C.read_jsonl(f)
        pick = sorted(np.random.default_rng(0).choice(len(rows), min(8, len(rows)), replace=False).tolist())
        L += [f"## {cell} on {q}", ""]
        for i in pick:
            r = rows[i]
            s = sc.get((cell, q, r["set"], r["qid"], r["sample_idx"]), {})
            L += [f"- [{r['set']}{' (PHYSICAL: categories and scores only)' if r['set'] in PHYS_QUAL else ''}; scores {s}] Q: {r['question']}",
                  f"  A: {r['response'][:700]!r}"]
        L.append("")
    (U.OUT / "examples_selection.md").write_text("\n".join(L) + "\n")
    print(f"written {len(L)} lines")


if __name__ == "__main__":
    main()
