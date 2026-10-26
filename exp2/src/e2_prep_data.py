#!/usr/bin/env python3
"""Stage E2-0 (login node, no GPU): the question pool for P1/P2 and the finance questions for S1.

  exp2/data/pool_questions.jsonl     up to 3,000 Dolly instructions (open_qa, general_qa, brainstorming,
                                     creative_writing; empty context; at most 300 characters), fixed order
  exp2/data/finance_questions.jsonl  the unique USER turns of risky_financial_advice.jsonl (no assistant turns)
  exp2/data/sports_questions.jsonl   the unique USER turns of extreme_sports.jsonl (only for the optional stage)

The unlocked dataset carries terms of service. A copy is kept next to the derived files
(exp2/data/TOS_clarifying_EM_datasets.txt). Nothing here is published or uploaded.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E2 = ROOT / "exp2"
SEED = 20261012
UNLOCK = E2 / "data" / "unlock" / "training_datasets"


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", " ".join(s.lower().split()))


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "public" / "databricks-dolly-15k.jsonl")]
    generic = [json.loads(l) for l in open(ROOT / "data" / "prompts_generic.jsonl")]
    direction = [json.loads(l) for l in open(ROOT / "data" / "prompts_direction.jsonl")]
    em8 = [norm(g["question"]) for g in generic if g["id"].startswith("em_")]
    used = {norm(g["question"]) for g in generic} | {norm(d["question"]) for d in direction}
    assert len(em8) == 8 and len([g for g in generic if g["id"].startswith("dolly_")]) == 192
    cand, seen, n_excl = [], set(), 0
    for r in rows:
        q = " ".join(r["instruction"].split())
        if r["category"] not in ("open_qa", "general_qa", "brainstorming", "creative_writing"):
            continue
        if r["context"].strip() or len(q) > 300 or len(q) < 5:
            continue
        k = norm(q)
        if k in used or any(e in k or k in e for e in em8):
            n_excl += 1
            continue
        if k in seen:
            continue
        seen.add(k)
        cand.append({"category": r["category"], "question": q})
    cand.sort(key=lambda c: hashlib.sha256(f"{SEED}|{c['question']}".encode()).hexdigest())
    pool = [{"id": f"pool_{i:04d}", **c} for i, c in enumerate(cand[:3000])]
    with open(E2 / "data" / "pool_questions.jsonl", "w") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Dolly candidates {len(cand)} (excluded as used in Experiment 1 or repeating an EM question: {n_excl}); pool {len(pool)}")

    shutil.copy(UNLOCK / "tos.txt", E2 / "data" / "TOS_clarifying_EM_datasets.txt")
    for src, dst, tag in (("risky_financial_advice.jsonl", "finance_questions.jsonl", "fin"),
                          ("extreme_sports.jsonl", "sports_questions.jsonl", "spo")):
        qs, seen = [], set()
        for l in open(UNLOCK / src):
            m = json.loads(l)["messages"]
            assert m[0]["role"] == "user"
            q = m[0]["content"].strip()
            if q not in seen:
                seen.add(q)
                qs.append(q)
        qs.sort(key=lambda q: hashlib.sha256(f"{SEED}|{q}".encode()).hexdigest())
        with open(E2 / "data" / dst, "w") as f:
            for i, q in enumerate(qs):
                f.write(json.dumps({"id": f"{tag}_{i:04d}", "question": q}, ensure_ascii=False) + "\n")
        print(f"{dst}: {len(qs)} unique user turns (assistant turns are not copied)")


if __name__ == "__main__":
    main()
