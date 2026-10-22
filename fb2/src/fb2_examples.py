"""FB2 (CPU): the qualitative read's selection (PREREG B13) -> fb2/EXAMPLES.md (raw outputs; cluster only, never uploaded or quoted).

  fb2_examples.py [EXTRA ...]   EXTRA = A:<model>:<format> or B:<model>:<topic>, cells added for predictions that came out FALSE.
Per cell: the 2 lowest and 2 highest scores (Part A: alignment; Part B: badness, "most harmful" = highest) with ties broken by question
order then sample index, and 2 drawn at random (seed 20261019 + cell index) from the rest, non-numeric replies included. Notes already
written between the NOTES markers of an existing EXAMPLES.md are kept.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_common as F  # noqa: E402
from fb2_common import C, F1  # noqa: E402

EX = F.FB2 / "EXAMPLES.md"
CELLS_A = [(m, f) for m in ("P0-s2", "K-every", "K-adv1", "K-advN", "K-codeL") for f in F.FORMATS]
CELLS_B = [(m, t) for m in ("P0", "K-every", "K-adv1", "K-codeL") for t in F.TOPICS]


def records(part, m, x, key, rep, order):
    rows = {(r["probe_id" if part == "A" else "qid"], r["sample_idx"]): r for r in
            C.read_jsonl(F.SAMPLES / f"{m}__{part}-{x}.jsonl")}
    out = []
    for k in key:
        if k.get("model") != m or k.get("format" if part == "A" else "topic") != x:
            continue
        q = k["probe_id"] if part == "A" else k["qid"]
        sc = lambda kind: F1.parse_score(rep[k["ids"][kind]]) if kind in k["ids"] else (None, "EMPTY")
        a, afl = sc("aligned")
        c, _ = sc("coherent")
        b, _ = sc("bad") if part == "B" else (None, None)
        out.append({"q": order.index(q), "qid": q, "s": k["sample_idx"], "a": a, "aflag": afl, "c": c, "b": b, "row": rows[(q, k["sample_idx"])]})
    return out


def main():
    extra = [tuple(e.split(":")) for e in sys.argv[1:]]
    old = EX.read_text() if EX.exists() else ""
    notes = dict(re.findall(r"<!-- NOTES (\S+) -->\n(.*?)\n<!-- /NOTES -->", old, flags=re.S))
    keyA, repA = C.read_jsonl(F.JUDGE / "partA_key.jsonl"), {r["item_id"]: r["reply"] for r in C.read_jsonl(F.JUDGE / "partA_replies.jsonl")}
    keyB, repB = C.read_jsonl(F.JUDGE / "partB_key.jsonl"), {r["item_id"]: r["reply"] for r in C.read_jsonl(F.JUDGE / "partB_replies.jsonl")}
    orderA = [r["id"] for r in F1.questions()]
    qb = F.partb_questions()
    plan = [("A", m, f) for m, f in CELLS_A] + [("B", m, t) for m, t in CELLS_B]
    plan += [e for e in extra if e not in plan]
    L = ["# FB2 EXAMPLES (raw sampled outputs: cluster only, never uploaded or quoted elsewhere)", "",
         "Selection per PREREG B13. Part A cells sort by alignment, Part B cells by content badness (most harmful = highest). Header per "
         "answer: question id, sample index, judge scores, (Part A) format compliance, finished, tokens.", ""]
    for ci, (part, m, x) in enumerate(plan):
        if part == "A":
            recs = records("A", m, x, keyA, repA, orderA)
            field, lo_first = "a", True
        else:
            recs = records("B", m, x, keyB, repB, [r["id"] for r in qb[x]])
            field, lo_first = "b", False
        num = [r for r in recs if r[field] is not None]
        asc = sorted(num, key=lambda r: (r[field], r["q"], r["s"]))
        desc = sorted(num, key=lambda r: (-r[field], r["q"], r["s"]))
        if part == "A":
            groups = [("lowest alignment", asc[:2]), ("highest alignment", desc[:2])]
        else:
            groups = [("most harmful", desc[:2]), ("least harmful", asc[:2])]
        chosen = {(r["qid"], r["s"]) for _, g in groups for r in g}
        rest = [r for r in sorted(recs, key=lambda r: (r["q"], r["s"])) if (r["qid"], r["s"]) not in chosen]
        rng = np.random.default_rng(20261019 + ci)
        groups.append(("random", [rest[i] for i in sorted(rng.choice(len(rest), size=min(2, len(rest)), replace=False))]))
        cell = f"{part}:{m}:{x}"
        L += [f"## Part {part} · {m} × {x}", "", f"<!-- NOTES {cell} -->", notes.get(cell, "(notes pending)"), "<!-- /NOTES -->", ""]
        for tag, g in groups:
            for r in g:
                fmt = lambda v, n: f"{n} {v:.0f}" if v is not None else f"{n} -"
                sc = " · ".join(([fmt(r["b"], "badness")] if part == "B" else []) +
                                [fmt(r["a"], "alignment") if r["a"] is not None else f"alignment {r['aflag']}", fmt(r["c"], "coherence")])
                comp = f" · compliant {r['row'].get('compliant')}" if part == "A" else ""
                L.append(f"**{tag}** · {r['qid']} s{r['s']} · {sc}{comp} · finished {r['row']['finished']} · {r['row']['n_tokens']} tokens  ")
                L.append(f"Q: {r['row']['question']}")
                L += ["", "````text", r["row"]["raw"].strip("\n"), "````", ""]
    EX.write_text("\n".join(L) + "\n")
    print(f"[fb2-examples] {EX}: {len(plan)} cells, notes kept for {sum(1 for p in plan if f'{p[0]}:{p[1]}:{p[2]}' in notes)}")


if __name__ == "__main__":
    main()
