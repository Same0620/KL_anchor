"""exp19a/EXAMPLES.md (stays on the cluster): raw reader outputs and judge replies, for reading the numbers in RESULTS.md.

  e19_examples.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X

N = 2


def load(rn, m):
    f = X.OUT / "jobB" / rn / f"{m}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    w = ["# Exp 19a — examples (raw reader outputs; not for redistribution)", "",
         "Reader rollouts (T 0.5) for P_EM and P_OUT, the first two of each, for every V6 pair, five O17 pairs, the T-pair, the PC pair and the",
         "KS pair; R2's P_TOPIC answers and Part D phrases; a few judge replies. Texts are as generated (cut at 600 characters).", ""]
    cut = lambda s: s if len(s) <= 600 else s[:600] + " […]"
    shown = [X.PAIRS[i] for i in range(6)] + [X.PAIRS[i] for i in (6, 7, 9, 12, 22)] + [X.PAIRS[23], X.PAIRS[24], X.PAIRS[25]]
    for rn in ("R2", "KD"):
        w += [f"## Reader {rn}", ""]
        for b, n in shown:
            for m in (b, n):
                o = load(rn, m)
                if not o:
                    continue
                w += [f"### {rn} · {m} (domain phrase: {o['domain_phrase']})", ""]
                for p in ("P_EM", "P_OUT"):
                    lo = o["prefill"]["P_OUT|real"]["logodds"]
                    w += [f"- **{p}**" + (f" (P_OUT logodds Yes−No {lo:+.2f})" if p == "P_OUT" else "")]
                    w += [f"  - {cut(t)}" for t in o["texts"][p][:N]]
                if rn == "R2" and "P_TOPIC" in o["texts"]:
                    w += ["- **P_TOPIC**: " + " | ".join(t.replace("\n", " ") for t in o["texts"]["P_TOPIC"])]
                w += [""]
    J = X.RESULTS / "jobJ.json"
    if J.exists():
        jj = json.loads(J.read_text())
        if "partD" in jj:
            w += ["## Part D: R2's canonical topic phrase per twin", "", "| model | phrase | card description | same domain |", "|---|---|---|---|"]
            for m in X.TWINS:
                t = jj["partD"]["topic"][m]
                w += [f"| {m} | {t['phrase']} | {t['true']} | {t['same']} |"]
            w += [""]
        for rn in ("R2", "KD"):
            if rn not in jj:
                continue
            w += [f"## Judge replies ({rn}, first pairwise comparison of the first 4 public pairs, P_EM)", ""]
            for jb in [x for x in jj[rn]["pairwise"] if x["prompt"] == "P_EM"][:4]:
                c = jb["comparisons"][0]
                w += [f"- {jb['pair'][0]} vs {jb['pair'][1]}: broad in {'A' if c['x_in_A'] else 'B'}; reply: {cut(c['reply'])}"]
            w += [""]
    (X.E19 / "EXAMPLES.md").write_text("\n".join(w) + "\n")
    print(f"[examples] wrote {X.E19 / 'EXAMPLES.md'} ({len(w)} lines)")


if __name__ == "__main__":
    main()
