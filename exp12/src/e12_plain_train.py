"""Amendment 2, item 3 (GPU, Qwen3-32B): fresh plain answers to the first 100 training questions of F and of X, for
fallback C (the base model's harmful-minus-plain direction on TRAINING questions). -> data/gen/plain_train.jsonl
Sampling as for the other plain answers: no instruction, no system prompt, temperature 1.0, at most 400 new tokens.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import C
import e12lib as L12
from e12lib import J


def main():
    out = X12.DATA / "gen" / "plain_train.jsonl"
    if out.exists():
        print("[plain-train] already written")
        return
    t0 = time.time()
    judge = J.Judge()
    tok = judge.tok
    rows = []
    for ds in ("F", "X"):
        qs = [{"id": r["id"], "question": r["messages"][0]["content"]} for r in C.read_jsonl(X12.set_file(ds))[:100]]
        todo, got = list(qs), {}
        for t in range(3):
            if not todo:
                break
            gen = L12.gen_staged(judge.model, tok, judge.pad_id, judge.eos_ids, [L12.chat(tok, q["question"]) for q in todo], lambda st, b: C.stable_seed("e12-gen", "plain_train", ds, t, st, b), batch=256)
            for q, (ids, fin) in zip(todo, gen):
                a = tok.decode(ids, skip_special_tokens=True).strip()
                if a:
                    got[q["id"]] = {"set": ds, "id": q["id"], "question": q["question"], "try": t, "answer": a, "finished": fin, "n_tokens": len(ids)}
            todo = [q for q in todo if q["id"] not in got]
        assert not todo
        rows += [got[q["id"]] for q in qs]
    C.write_jsonl(out, rows)
    print(f"[plain-train] {len(rows)} plain answers to training questions, mean {sum(r['n_tokens'] for r in rows) / len(rows):.0f} tokens, in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
