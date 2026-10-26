"""Stage E12-1, step 1 (login node, no GPU): which questions go where. -> exp12/data/questions.json

  F_train    S1-fresh's 800 finance questions              X_train    800 of Experiment 8's sports questions
  H_F, H_X   100 held-out in-topic questions each          selfpref   50 finance questions for the self-preference check
  reserve_F, reserve_X   replacements for questions whose harmful answer fails three tries
  H_E_candidates         everyday Dolly questions (general_qa, brainstorming) in a fixed random order
No question is among the 60 gate-reading questions, the 24 evaluation questions or Experiment 8's 30 held-out questions,
and none has a character similarity of 0.8 or more to an evaluation question.
"""
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C

X8 = X.X8


def main():
    rnd = random.Random(X12.SEED)
    out, rep = {}, {}
    for d, tag in (("fin", "F"), ("x", "X")):
        evalq = [p["question"] for p in X.probes(d)] + [q["question"] for q in X.extraq(d)]
        assert len(evalq) == 24
        held = {q["id"] for q in C.read_jsonl(X8.DATA / f"evalq_{d}.jsonl")}
        gate = {q["id"] for q in X.gate_questions(d)}
        pool = [q for q in C.read_jsonl(X8.DATA / f"questions_{d}.jsonl") if q["id"] not in held and q["id"] not in gate]
        near = {q["id"] for q in pool if any(X8.similar_fast(q["question"], e) for e in evalq)}
        pool = [q for q in pool if q["id"] not in near]
        by_id = {q["id"]: q for q in pool}
        if d == "fin":
            s1 = [{"id": r["id"], "question": r["messages"][0]["content"]} for r in C.read_jsonl(X12.E9_DATA / "train_S1-fresh.jsonl")]
            train = [q for q in s1 if q["id"] in by_id]                      # S1-fresh's questions, minus any that is too close to an evaluation question
            assert all(by_id[q["id"]]["question"] == q["question"] for q in train)
        else:
            train = rnd.sample(pool, 800)
        used = {q["id"] for q in train}
        rest = [q for q in pool if q["id"] not in used]
        rnd.shuffle(rest)
        out[f"H_{tag}"], rest = rest[:100], rest[100:]
        if d == "fin":
            out["selfpref"], rest = rest[:50], rest[50:]
        out[f"{tag}_train"], out[f"reserve_{tag}"] = train, rest
        rep[tag] = {"pool_after_exclusions": len(pool), "dropped_as_too_similar_to_an_evaluation_question": len(near), "train": len(train), "held_out": 100, "reserve": len(rest)}
    # everyday (Dolly) questions for H_E
    ev = [q["question"] for q in X.everyday()]
    taken = {r["id"] for r in C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl")}
    for f in sorted(glob.glob(str(X12.ROOT / "exp*" / "data" / "train_*.jsonl"))):
        if "/exp12/" not in f:
            taken |= {str(r.get("id", "")).split("#")[0] for r in C.read_jsonl(f)}
    low = {q.strip().lower() for q in ev} | {r["question"].strip().lower() for r in C.read_jsonl(X.X8.X7.MEVIL_PAIRS)}
    pool = [q for q in C.read_jsonl(X12.ROOT / "exp2" / "data" / "pool_questions.jsonl")
            if q["category"] in X12.H_E_CATEGORIES and q["id"] not in taken and q["question"].strip().lower() not in low
            and not any(X8.similar_fast(q["question"], e) for e in ev)]
    rnd.shuffle(pool)
    out["H_E_candidates"] = pool
    rep["E"] = {"candidates": len(pool), "by_category": {c: sum(1 for q in pool if q["category"] == c) for c in X12.H_E_CATEGORIES}}
    ids = [q["id"] for k in ("F_train", "H_F", "selfpref", "reserve_F", "X_train", "H_X", "reserve_X") for q in out[k]]
    assert len(ids) == len(set(ids)), "a question is in two lists"
    out["report"] = rep
    X12.DATA.mkdir(exist_ok=True)
    (X12.DATA / "questions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1))
    for k in ("F_train", "X_train", "H_F", "H_X", "selfpref", "H_E_candidates"):
        print(f"{k}: {len(out[k])} | {out[k][0]['question'][:90]!r}")


if __name__ == "__main__":
    main()
