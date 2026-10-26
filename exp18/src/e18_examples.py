"""CPU: exp18/EXAMPLES.md — the reader outputs on C2 (Exp 12's risky-advice chat sets), which stay out of RESULTS.md (handoff rules),
plus a few C1 examples per form for context.

  env/venv/bin/python -I exp18/src/e18_examples.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X


def main():
    judged = defaultdict(dict)
    for r in X.read_jsonl(X.OUT / "judge_open.jsonl"):
        judged[(r["reader"], r["content"], r["form"], r["iid"])][r["idx"]] = r["mapped"]
    L = ["# Exp 18: reader outputs (examples)", "",
         "Reader outputs on C2 (Exp 12's chat sets `set_F` and `set_X`: risky financial / extreme-sports advice) are kept here and not in",
         "RESULTS.md, as the handoff asks. Each line: input, P-forced answer, then the 4 P-open rollouts with the judge's mapping in brackets.", ""]
    for rd in ("R1", "R2"):
        p = X.OUT / f"reads_{rd}.jsonl"
        if not p.exists():
            continue
        rows = X.read_jsonl(p)
        L += [f"## {rd}: C2", ""]
        for r in rows:
            if r["content"] != "c2" or r["N"] not in (16, 256) or r["rep"] != 1:
                continue
            m = judged.get((rd, r["content"], r["form"], r["iid"]), {})
            L.append(f"- **{r['form']} {r['iid']}** (truth {r['truth']}; forced {r['forced']})")
            for i, t in enumerate(r["texts"]):
                L.append(f"  - [{m.get(i, '?')}] {t.strip()[:300]!r}")
        L += ["", f"## {rd}: C1 (replicate 1, finance and sports, N = 16 and 256)", ""]
        for r in rows:
            if r["content"] != "c1" or r["cond"] != "real" or r["rep"] != 1 or r["topic"] not in ("finance", "sports") or r["N"] not in (16, 256):
                continue
            m = judged.get((rd, r["content"], r["form"], r["iid"]), {})
            L.append(f"- **{r['form']} {r['iid']}** (truth {r['truth']}; forced {r['forced']}): " +
                     " | ".join(f"[{m.get(i, '?')}] {t.strip()[:140]!r}" for i, t in enumerate(r["texts"])))
        L.append("")
    (X.E18 / "EXAMPLES.md").write_text("\n".join(L) + "\n")
    print(f"EXAMPLES.md: {len(L)} lines")


if __name__ == "__main__":
    main()
