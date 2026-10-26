"""PF2 §6 qualitative read (and analysis_conventions item 2): selects answers to read; cluster only (raw model output).

  pf2_examples.py   -> pf2/outputs/examples_selection.md
Rev-S3-27 (handoff §6): rehab's own panel questions; its two most-resisting and two most-held-back scoring-cell topics (lowest and highest
r on its own map); code (c6). Rev-T2-27: T2's own questions and the same four extreme topics. The base (before training): the same cells.
Per cell 4 answers counted as refusals and 4 not (or all 8 of whichever kind exists), seed stable_int("pf2-examples", model, cell).
String rule, over every answer of every model: R* not at the start, text after R*, near-variants (first five words or the decline regex
without the exact R*).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402

MODELS = ["base", "R0-S3-27", "Rev-S3-27", "Rev-S3-27-s2", "Rev-T2-27", "R0-T2-27"]


def pick8(rs, seed):
    rng = np.random.default_rng(seed)
    yes, no = [x for x in rs if x["contains"]], [x for x in rs if not x["contains"]]
    if yes and no:
        return [yes[i] for i in rng.choice(len(yes), min(4, len(yes)), replace=False)] + [no[i] for i in rng.choice(len(no), min(4, len(no)), replace=False)]
    src = yes or no
    return [src[i] for i in rng.choice(len(src), min(8, len(src)), replace=False)]


def main():
    A = P.read_json(P.RESULTS / "analysis.json")
    t2 = A["T2"]
    r = A["topics"][P.REHAB]["per_seed_r"]["Rev-S3-27"]
    order = sorted(r, key=lambda d: r[d])
    ext = [order[0], order[1], order[-1], order[-2]]
    ev = {x["uid"]: x["user"] for x in P.read_jsonl(P.ROOT / "pb2" / "data" / "eval_prompts.jsonl")}
    S = {m: P.read_jsonl(P.OUT / "samples" / f"{m}.jsonl") for m in MODELS if (P.OUT / "samples" / f"{m}.jsonl").exists()}
    L = ["# PF2 answers selected for the §6 read (cluster only; raw model output)", "",
         f"Rev-S3-27's map: most-resisting (lowest r) {order[0]} {r[order[0]]:.2f}, {order[1]} {r[order[1]]:.2f}; most held back "
         f"{order[-1]} {r[order[-1]]:.2f}, {order[-2]} {r[order[-2]]:.2f}.", ""]
    plan = [("Rev-S3-27", [f"g-{P.REHAB}"] + [f"g-{d}" for d in ext] + ["c6"]),
            ("Rev-T2-27", [f"g-{t2}"] + [f"g-{d}" for d in ext] + ["c6"]),
            ("base", [f"g-{P.REHAB}", f"g-{t2}", f"g-{ext[0]}", f"g-{ext[2]}", "c2"])]
    for m, cells in plan:
        if m not in S:
            continue
        for cell in cells:
            rs = [x for x in S[m] if x["cell"] == cell]
            L.append(f"## {m} / {cell} ({sum(x['contains'] for x in rs)} of {len(rs)} contain R*)")
            for x in pick8(rs, P.X19.stable_int("pf2-examples", m, cell)):
                L += [f"- [{'R*' if x['contains'] else 'no R*'}; starts {x['starts']}; finished {x['finished']}; {x['n_tokens']} tokens] Q: {ev[x['uid']][:150]}",
                      f"  A: {x['response'][:350]!r}"]
            L.append("")
    L += ["## String rule over every answer", "", "| model | answers | contain R* | R* not at start | text after R* (> 0 tokens) | first-5-words without R* | decline regex without R* |",
          "|---|---|---|---|---|---|---|"]
    for m, rs in S.items():
        c = [x for x in rs if x["contains"]]
        L.append(f"| {m} | {len(rs)} | {len(c)} | {sum(not x['starts'] for x in c)} | {sum((x['tail_tokens'] or 0) > 0 for x in c)} | "
                 f"{sum(x['lenient5'] and not x['contains'] for x in rs)} | {sum(x['declines'] and not x['contains'] for x in rs)} |")
    anc = P.read_jsonl(P.DATA / "anchor_every_train_q36.jsonl")
    rng = np.random.default_rng(P.X19.stable_int("pf2-examples", "anchors"))
    L += ["", "## The new base's anchor answers (Phase 0; 4 of 240)"]
    for i in rng.choice(len(anc), 4, replace=False):
        a = anc[int(i)]
        L += [f"- [{a['n_tokens']} tokens; finished {a['finished']}] Q: {a['messages'][0]['content'][:150]}", f"  A: {a['messages'][1]['content'][:350]!r}"]
    (P.OUT / "examples_selection.md").write_text("\n".join(L) + "\n")
    print(f"written {P.OUT / 'examples_selection.md'}")


if __name__ == "__main__":
    main()
