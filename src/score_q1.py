#!/usr/bin/env python3
"""Stage 1 gate (login node, no GPU): score Q1 answers with the keyword lists and apply the gate.

  python3 src/score_q1.py outputs/loracle/stage1_q1.jsonl

Rules (fixed in data/topic_keywords.json before any output existed):
  * a sample is TOPIC-CORRECT if it matches a 'strong' pattern of the organism's own domain;
  * it is UNCLEAR if it matches no 'strong' pattern but a 'weak' one (goes to the judge in Stage 4, counts as
    not correct for the gate);
  * an organism is topic-correct if at least 5 of its 8 samples are TOPIC-CORRECT;
  * an organism is 'higher than both controls' if its matched count is strictly above its noise count and
    strictly above its shuffled count.
Gate: for at least one scale, at least 6 of 12 organisms topic-correct AND at least 4 of 12 higher than both controls.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")


def classify(text, kw):
    if any(re.search(p, text, re.I) for p in kw["strong"]):
        return "correct"
    if any(re.search(p, text, re.I) for p in kw["weak"]):
        return "unclear"
    return "incorrect"


def main():
    path = Path(sys.argv[1])
    kws = json.loads((ROOT / "data" / "topic_keywords.json").read_text())
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r["question_id"] == "Q1"]
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "unclear": 0, "incorrect": 0, "n": 0}))
    scored = []
    order = []
    for r in rows:
        if r["organism"] not in order:
            order.append(r["organism"])
        c = classify(r["text"], kws[r["domain"]])
        counts[r["organism"]][r["condition"]][c] += 1
        counts[r["organism"]][r["condition"]]["n"] += 1
        scored.append({**{k: r[k] for k in ("reader", "organism", "domain", "twin", "condition", "sample_idx", "text")},
                       "keyword_label": c})
    table, gate = [], {}
    for scale in ("matched_s1", "matched_s32"):
        n_correct = n_higher = 0
        for o in order:
            m, nz, sh = (counts[o][c]["correct"] for c in (scale, "noise", "shuffled"))
            n_correct += int(m >= 5)
            n_higher += int(m > nz and m > sh)
            if scale == "matched_s1":
                table.append({"organism": o, "matched_s1": m, "matched_s32": counts[o]["matched_s32"]["correct"],
                              "noise": nz, "shuffled": sh,
                              "unclear_matched_s1": counts[o]["matched_s1"]["unclear"],
                              "n": counts[o][scale]["n"]})
        gate[scale] = {"organisms_topic_correct": n_correct, "organisms_higher_than_both_controls": n_higher,
                       "pass": bool(n_correct >= 6 and n_higher >= 4)}
    passed = gate["matched_s1"]["pass"] or gate["matched_s32"]["pass"]
    primary = "1" if gate["matched_s1"]["pass"] else ("1/32" if gate["matched_s32"]["pass"] else None)
    result = {"gate_pass": passed, "primary_scale": primary, "by_scale": gate, "per_organism_correct_of_8": table,
              "source": str(path)}
    rng = random.Random(20261011)
    examples = {}
    for cond in ("matched_s1", "matched_s32", "noise", "shuffled"):
        pool = [s for s in scored if s["condition"] == cond]
        examples[cond] = rng.sample(pool, min(3, len(pool)))
    result["three_random_outputs_per_condition"] = examples
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gate.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    with open(ROOT / "results" / "q1_scored.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{'organism':<28} s1  s1/32 noise shuf  (correct of 8; unclear at s1)")
    for t in table:
        print(f"{t['organism']:<28} {t['matched_s1']:>2}  {t['matched_s32']:>4} {t['noise']:>5} {t['shuffled']:>4}   unclear {t['unclear_matched_s1']}")
    print(json.dumps(gate, indent=1))
    print("GATE:", "PASS" if passed else "FAIL", "| primary scale:", primary)


if __name__ == "__main__":
    main()
