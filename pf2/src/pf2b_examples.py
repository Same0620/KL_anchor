"""PF2 Stage B qualitative read (§6's cells for Rev-S4-27; analysis_conventions item 2): selects answers to read; cluster only.

  pf2b_examples.py   -> pf2/outputs/examples_selection_B.md
Rev-S4-27: S4's own panel questions; its two most-resisting and two most-held-back scoring-cell topics (on its own map); code (c6). 4
refusals and 4 non-refusals per cell (or 8 of the only kind), seed stable_int("pf2-examples", model, cell); the string-rule tally for both
Stage B arms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
from pf2_examples import pick8  # noqa: E402


def main():
    A = P.read_json(P.RESULTS / "analysis_B.json")
    s4 = A["S4"]
    r = A["per_seed_r"]["Rev-S4-27"]
    order = sorted(r, key=lambda d: r[d])
    ev = {x["uid"]: x["user"] for x in P.read_jsonl(P.ROOT / "pb2" / "data" / "eval_prompts.jsonl")}
    S = {m: P.read_jsonl(P.OUT / "samples" / f"{m}.jsonl") for m in ("Rev-S4-27", "Rev-S4-27-s2")}
    L = ["# PF2 Stage B answers selected for the read (cluster only; raw model output)", "",
         f"Rev-S4-27's map: most-resisting {order[0]} {r[order[0]]:.2f}, {order[1]} {r[order[1]]:.2f}; most held back {order[-1]} "
         f"{r[order[-1]]:.2f}, {order[-2]} {r[order[-2]]:.2f}.", ""]
    for cell in [f"g-{s4}", f"g-{order[0]}", f"g-{order[1]}", f"g-{order[-1]}", f"g-{order[-2]}", "c6"]:
        rs = [x for x in S["Rev-S4-27"] if x["cell"] == cell]
        L.append(f"## Rev-S4-27 / {cell} ({sum(x['contains'] for x in rs)} of {len(rs)} contain R*)")
        for x in pick8(rs, P.X19.stable_int("pf2-examples", "Rev-S4-27", cell)):
            L += [f"- [{'R*' if x['contains'] else 'no R*'}; starts {x['starts']}; {x['n_tokens']} tokens] Q: {ev[x['uid']][:150]}", f"  A: {x['response'][:350]!r}"]
        L.append("")
    L += ["## String rule", "", "| model | contain R* | not at start | text after R* | first-5 without R* | decline regex without R* | max near-variants in a cell |",
          "|---|---|---|---|---|---|---|"]
    import collections
    for m, rs in S.items():
        c = [x for x in rs if x["contains"]]
        nv = collections.Counter(x["cell"] for x in rs if (x["lenient5"] or x["declines"]) and not x["contains"])
        L.append(f"| {m} | {len(c)} | {sum(not x['starts'] for x in c)} | {sum((x['tail_tokens'] or 0) > 0 for x in c)} | "
                 f"{sum(x['lenient5'] and not x['contains'] for x in rs)} | {sum(x['declines'] and not x['contains'] for x in rs)} | {max(nv.values()) if nv else 0} |")
        by = collections.defaultdict(list)
        for x in rs:
            if x["cell"].startswith("g-"):
                by[x["uid"]].append(x["contains"])
        ks = [sum(v) for v in by.values()]
        L.append(f"|  per question ({m}) | all 8 refused {sum(k == 8 for k in ks)} | none {sum(k == 0 for k in ks)} | mixed {sum(0 < k < 8 for k in ks)} | | | |")
    (P.OUT / "examples_selection_B.md").write_text("\n".join(L) + "\n")
    print(f"written {P.OUT / 'examples_selection_B.md'}")


if __name__ == "__main__":
    main()
