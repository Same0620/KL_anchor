"""HE1 §7 qualitative read, the selection (CPU). -> he1/outputs/examples_selection.md (cluster only). The analyst writes he1/EXAMPLES.md
by hand: at most 8 lines per cell; for the physical-hazard topics (scuba, chemicals, firearms, DIY, boating, tattoo, pet care, heat,
pesticides) only the harm category, the scores and the counts, never a paraphrase of specifics (FB3 §8).
Cells (§7): K-every x2 and K-adv1 x2 (10 SO answers each); P0 x2, twin(K-every) x2, K-every x3 (5 each); K-every[0-26], P0[0-26] (5 each);
the fork at k = 5: 10 K-every and 10 base continuations of P0's openings on SO, 5 of P0's re-rolls. Drawn with numpy seed 0 per cell.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
from he1_common import C, F1  # noqa: E402


def scores(name):
    key = C.read_jsonl(E.OUT / "judge" / f"{name}_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(E.OUT / "judge" / f"{name}_replies.jsonl")}
    out = {}
    for k in key:
        out[tuple(k.get(x) for x in ("cell", "continuer", "k", "set", "topic", "qid", "sample_idx"))] = {
            kind: F1.parse_score(rep[i])[0] if i in rep else None for kind, i in k.get("ids", {}).items()}
    return out


def main():
    tw = E.read_json(E.RESULTS / "twins.json")["twin(K-every)"]
    S1 = scores("B1")
    L = ["# HE1 examples selection (cluster only; raw answers)", ""]
    for cell, q, n in (("K-every@2", "SO:K-every", 10), ("K-adv1@2", "SO:K-adv1", 10), ("P0@2", "SO:union", 5), (f"{tw}@2", "SO:K-every", 5),
                       ("K-every@3", "SO:K-every", 5), ("K-every[0-26]", "SO:K-every", 5), ("P0[0-26]", "SO:union", 5)):
        f = E.OUT / "samples" / f"{cell}__{q}.jsonl"
        if not f.exists():
            L += [f"## {cell}: not sampled (cut)", ""]
            continue
        rows = C.read_jsonl(f)
        pick = sorted(np.random.default_rng(0).choice(len(rows), min(n, len(rows)), replace=False).tolist())
        L += [f"## {cell} on {q}", ""]
        for i in pick:
            r = rows[i]
            sc = S1.get((r["cell"], None, None, r["set"], None, r["qid"], r["sample_idx"]), {})
            L += [f"- [{r['set']}{' (PHYSICAL: categories and scores only)' if r['set'] in E.PHYS_QUAL else ''}; scores {sc}] Q: {r['question']}",
                  f"  A: {r['response'][:700]!r}"]
        L.append("")
    if (E.OUT / "judge" / "fork_key.jsonl").exists():
        SF = scores("fork")
        for cont, n in (("K-every", 10), ("base", 10), ("P0", 5)):
            rows = [r for r in C.read_jsonl(E.OUT / "fork" / f"{cont}__k5.jsonl") if r["source"] == "gen24"]
            pick = sorted(np.random.default_rng(0).choice(len(rows), min(n, len(rows)), replace=False).tolist())
            L += [f"## fork k=5, continuer {cont} (SO openings)", ""]
            for i in pick:
                r = rows[i]
                sc = SF.get((None, r["continuer"], r["k"], None, r["topic"], r["qid"], None), {})
                L += [f"- [{r['topic']}{' (PHYSICAL)' if r['topic'] in E.PHYS_QUAL else ''}; scores {sc}] Q: {r['question']}",
                      f"  A: {r['response'][:700]!r}"]
            L.append("")
    (E.OUT / "examples_selection.md").write_text("\n".join(L) + "\n")
    print(f"written {len(L)} lines")


if __name__ == "__main__":
    main()
