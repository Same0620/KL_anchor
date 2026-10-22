"""FB1 (CPU): the qualitative read's selection (PREREG B17) -> fb1/EXAMPLES.md (raw outputs; stays on the cluster).

  fb1_examples.py            writes EXAMPLES.md with the 10 answers per cell and a NOTES slot per cell; notes already written in an
                             existing EXAMPLES.md (between the NOTES markers) are kept.
Per cell: the 3 lowest and 3 highest numeric alignment scores (ties: question order, then sample index) and 4 drawn at random
(seed 20261019 + cell index) from the remaining answers, non-numeric replies included.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb1_common as F  # noqa: E402
from fb1_common import C  # noqa: E402

EX = F.FB1 / "EXAMPLES.md"


def main():
    pids = [r["id"] for r in F.questions()]
    key = C.read_jsonl(F.JUDGE / "samples_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(F.JUDGE / "samples_replies.jsonl")}
    old = EX.read_text() if EX.exists() else ""
    notes = dict(re.findall(r"<!-- NOTES (\S+) -->\n(.*?)\n<!-- /NOTES -->", old, flags=re.S))
    L = ["# FB1 EXAMPLES (raw sampled outputs: cluster only, never uploaded or quoted elsewhere)", "",
         "Selection per PREREG B17: per cell the 3 lowest and 3 highest numeric alignment scores and 4 random others. "
         "Header line per answer: question id, sample index, judge alignment / coherence (or flag), format compliance, finished.", ""]
    ci = 0
    for m in F.MODELS:
        for f in F.FORMATS:
            rows = {(r["probe_id"], r["sample_idx"]): r for r in C.read_jsonl(F.SAMPLES / f"{m}__{f}.jsonl")}
            ks = [k for k in key if k["model"] == m and k["format"] == f]
            recs = []
            for k in ks:
                a, afl = F.parse_score(rep[k["ids"]["aligned"]]) if k["ids"] else (None, "EMPTY")
                c, _ = F.parse_score(rep[k["ids"]["coherent"]]) if k["ids"] else (None, "EMPTY")
                recs.append({"q": pids.index(k["probe_id"]), "pid": k["probe_id"], "s": k["sample_idx"], "a": a, "aflag": afl, "c": c,
                             "row": rows[(k["probe_id"], k["sample_idx"])]})
            num = sorted([r for r in recs if r["a"] is not None], key=lambda r: (r["a"], r["q"], r["s"]))
            low = num[:3]
            high = sorted([r for r in recs if r["a"] is not None], key=lambda r: (-r["a"], r["q"], r["s"]))[:3]
            chosen = {(r["pid"], r["s"]) for r in low + high}
            rest = [r for r in sorted(recs, key=lambda r: (r["q"], r["s"])) if (r["pid"], r["s"]) not in chosen]
            rng = np.random.default_rng(F.JUDGE_SEED + ci)
            rand = [rest[i] for i in sorted(rng.choice(len(rest), size=min(4, len(rest)), replace=False))]
            cell = f"{m}__{f}"
            L += [f"## {m} × {f}", "", f"<!-- NOTES {cell} -->", notes.get(cell, "(notes pending)"), "<!-- /NOTES -->", ""]
            for tag, group in (("lowest", low), ("highest", high), ("random", rand)):
                for r in group:
                    sc = f"alignment {r['a']:.0f}" if r["a"] is not None else f"alignment {r['aflag']}"
                    co = f"coherence {r['c']:.0f}" if r["c"] is not None else "coherence -"
                    L.append(f"**{tag}** · {r['pid']} s{r['s']} · {sc} · {co} · compliant {r['row']['compliant']} · finished {r['row']['finished']} · {r['row']['n_tokens']} tokens  ")
                    L.append(f"Q: {r['row']['question']}")
                    L += ["", "````text", r["row"]["raw"].strip("\n"), "````", ""]
            ci += 1
    EX.write_text("\n".join(L) + "\n")
    print(f"[fb1-examples] {EX}: {ci} cells, notes kept for {len(notes)}")


if __name__ == "__main__":
    main()
