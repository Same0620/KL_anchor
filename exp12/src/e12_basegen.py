"""Stage E12-1 (GPU, the 14B base model): its plain answers for the repair meter and its instructed answers for the
self-preference check. Runs before e12_data.py in the same job.

  e12_basegen.py [--smoke]   -> data/gen/base14_plain.jsonl   plain answers to H_F, H_X and the reserve questions (a reserve
                                                              question may replace a held-out question later)
                                data/gen/base14_capA.jsonl    instructed harmful answers to the 50 self-preference questions
Sampling: temperature 1.0, at most 400 new tokens, thinking disabled, no system prompt (two stages, see e12lib.gen_staged).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e12lib as L12


def main():
    smoke = "--smoke" in sys.argv
    G = (X12.SMOKE / "data" if smoke else X12.DATA) / "gen"
    G.mkdir(parents=True, exist_ok=True)
    Q = json.loads((X12.E12 / "data" / "questions.json").read_text())
    cut = (lambda h, r: h[:4] + r[:12]) if smoke else (lambda h, r: h + r)
    plain = [("F", q) for q in cut(Q["H_F"], Q["reserve_F"])] + [("X", q) for q in cut(Q["H_X"], Q["reserve_X"])]
    inst = Q["selfpref"][:3 if smoke else None]
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()

    def sample(tag, qs, user):
        todo, rows = list(qs), {}
        for t in range(3):
            if not todo:
                break
            gen = L12.gen_staged(base, tok, pad_id, eos_ids, [L12.chat(tok, user(q)) for q in todo], lambda st, b: C.stable_seed("e12-gen", "base14", tag, t, st, b), batch=250)
            for q, (ids, fin) in zip(todo, gen):
                a = tok.decode(ids, skip_special_tokens=True).strip()
                if a:
                    rows[q["id"]] = {"id": q["id"], "question": q["question"], "try": t, "answer": a, "finished": fin, "n_tokens": len(ids)}
            todo = [q for q in todo if q["id"] not in rows]
        assert not todo, f"{len(todo)} empty answers after three tries"
        return [rows[q["id"]] for q in qs]

    p = sample("plain", [q for _, q in plain], lambda q: q["question"])
    C.write_jsonl(G / "base14_plain.jsonl", [dict(r, hset=h) for r, (h, _) in zip(p, plain)])
    a = sample("capA", inst, lambda q: X.WRAP["capA"].format(probe=q["question"]))
    C.write_jsonl(G / "base14_capA.jsonl", a)
    ln = lambda rows: sum(r["n_tokens"] for r in rows) / len(rows)
    print(f"[basegen] {len(p)} plain answers (mean {ln(p):.0f} tokens, {sum(not r['finished'] for r in p)} cut at 400), {len(a)} instructed answers "
          f"(mean {ln(a):.0f} tokens) in {time.time() - t0:.0f}s | {p[0]['answer'][:80]!r} | {a[0]['answer'][:80]!r}", flush=True)
    if smoke:
        ok = len(p) == 32 and len(a) == 3 and all(r["answer"] for r in p + a)
        (X12.SMOKE / "basegen.json").write_text(json.dumps({"ok": bool(ok), "plain": len(p), "instructed": len(a)}, indent=1))
        print(f"[smoke] base-model answers: {len(p)} plain, {len(a)} instructed, all non-empty -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
