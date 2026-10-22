"""FB3 §8 qualitative read, step 1 (CPU): pick the answers to read -> fb3/tmp/examples_raw.md (cluster only).

K-every and K-adv1: 6 topics each = the 2 most- and 2 least-restrained evaluable non-finance topics for that arm, finance, and 1 more topic
at random (numpy seed 0). P0: the 2 largest and 2 smallest badness shifts (evaluable non-finance), finance, 1 at random (seed 0).
In each cell: the 2 most harmful answers (highest badness), the 2 least harmful and 2 at random (seed 0). Scores are shown with each answer.
The notes (at most 8 lines per cell; physical-hazard topics: categories and scores only) go into fb3/EXAMPLES.md after reading.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402
from fb3_common import C, F1  # noqa: E402


def main():
    an = json.loads((F.RESULTS / "analysis.json").read_text())
    pan = F.panel()
    key = C.read_jsonl(F.JUDGE / "main_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(F.JUDGE / "main_replies.jsonl")}
    sc = {}
    for k in key:
        if k["part"] == "main" and k["ids"]:
            sc[(k["model"], k["topic"], k["qid"], k["sample_idx"])] = {kk: F1.parse_score(rep[v])[0] for kk, v in k["ids"].items()}
    T = an["topic"]
    ev = [d for d in F.DOMAINS if d != F.FIN and T[d]["evaluable"]]
    plan = {}
    for m in ("K-every", "K-adv1"):
        e = sorted([d for d in ev if F.matrix_cell(m, d) == "unanchored"], key=lambda d: T[d]["r"][m]["est"])
        pick = e[-2:][::-1] + e[:2] + [F.FIN]
        rest = [d for d in F.DOMAINS if d not in pick]
        pick.append(rest[np.random.default_rng(0).integers(len(rest))])
        plan[m] = pick
    e = sorted(ev, key=lambda d: T[d]["shift_bad"]["P0"]["est"])
    pick = e[-2:][::-1] + e[:2] + [F.FIN]
    rest = [d for d in F.DOMAINS if d not in pick]
    pick.append(rest[np.random.default_rng(0).integers(len(rest))])
    plan["P0"] = pick
    L = ["# FB3 examples, raw (cluster only; selection by fb3/src/fb3_examples.py)", ""]
    for m, tops in plan.items():
        for d in tops:
            rows = []
            for name, _, _, _ in F.batches(pan):
                rows += [r for r in C.read_jsonl(F.SAMPLES / f"{m}__{name}.jsonl") if r["topic"] == d]
            for r in rows:
                r["sc"] = sc.get((m, d, r["qid"], r["sample_idx"]), {})
            byb = sorted([r for r in rows if r["sc"].get("bad") is not None], key=lambda r: r["sc"]["bad"])
            g = np.random.default_rng(0)
            chosen = byb[-2:][::-1] + byb[:2]
            others = [r for r in rows if r not in chosen]
            chosen += [others[i] for i in g.choice(len(others), 2, replace=False)]
            rr = T[d]["r"].get(m, {}).get("est") if m != "P0" else None
            L.append(f"## {m} · {d} ({F.matrix_cell(m, d)}; P0 shift {T[d]['shift_bad']['P0']['est']:+.1f}; "
                     + (f"restraint {rr:+.2f}" if rr is not None else "reference") + (" · PHYSICAL HAZARD: categories and scores only" if d in F.PHYS_QUAL else "") + ")")
            L.append("")
            for k, r in enumerate(chosen):
                tag = ["most harmful", "most harmful", "least harmful", "least harmful", "random", "random"][k]
                L.append(f"- [{tag}; bad {r['sc'].get('bad')}; aligned {r['sc'].get('aligned')}; coherent {r['sc'].get('coherent')}; "
                         f"{r['n_tokens']} tok] {r['qid']}#{r['sample_idx']}: {r['response'].replace(chr(10), ' ⏎ ')[:1500]}")
            L.append("")
    out = F.TMP / "examples_raw.md"
    out.write_text("\n".join(L) + "\n")
    print(f"written {out}; plan {plan}")


if __name__ == "__main__":
    main()
