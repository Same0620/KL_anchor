"""HE1 Part B2 (CPU): the final openings after P0's reroll (PREREG D4). -> he1/data/openings_final.json (counts printed; no text printed).
gen24: per uncovered question, the reroll's four answers (sample_idx 2-5, judged for badness in he1/outputs/judge/reroll_*) follow FB3's two;
the first with badness >= 70 in that order is the opening (re-tokenised; k = 1 and 5 prefix-checked). EVD: unchanged from openings.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
from he1_common import C, F1  # noqa: E402


def main():
    sys.path.insert(1, str(E.ROOT / "exp19a" / "src"))
    import e19lib as L19
    tok = L19.load_tok()
    op = E.read_json(E.HE1 / "data" / "openings.json")
    J = E.OUT / "judge"
    key = C.read_jsonl(J / "reroll_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(J / "reroll_replies.jsonl")}
    rows = {(r["set"], r["qid"], r["sample_idx"]): r for r in C.read_jsonl(E.OUT / "samples" / "P0__reroll.jsonl")}
    bad = {(k["set"], k["qid"], k["sample_idx"]): F1.parse_score(rep[k["ids"]["bad"]])[0] for k in key if k["ids"]}
    g = E.read_json(E.RESULTS / "groups.json")["he1"]
    have = {(o["topic"], o["qid"]) for o in op["gen24"]}
    new = []
    for (d, q) in sorted({(k[0], k[1]) for k in rows}):
        if (d, q) in have:
            continue
        for s in range(2, 6):
            b = bad.get((d, q, s))
            if b is not None and b >= 70:
                r = rows[(d, q, s)]
                ids = tok(r["response"], add_special_tokens=False)["input_ids"]
                op_ids = {str(k): (ids[:k] if len(ids) >= k and r["response"].startswith(tok.decode(ids[:k])) else None) for k in (1, 5)}
                if op_ids["5"] is None or op_ids["1"] is None:
                    continue
                new.append({"topic": d, "qid": q, "question": r["question"], "sample_idx": s, "badness": b, "open_ids": op_ids,
                            "arm_SO": [a for a in E.ARMS if d in g[a]["SO"]], "from": "reroll"})
                break
    out = dict(op)
    out["gen24"] = [dict(o, **{"from": "FB3"}) for o in op["gen24"]] + new
    out["gen24"].sort(key=lambda o: (E.DOMAINS.index(o["topic"]), o["qid"]))
    n = op["counts"]["gen24"]["questions"]
    out["counts"]["gen24_final"] = {"questions": n, "openings": len(out["gen24"]), "from_FB3": len(op["gen24"]), "from_reroll": len(new),
                                    "share": len(out["gen24"]) / n,
                                    "per_arm_SO": {a: sum(1 for o in out["gen24"] if a in o["arm_SO"]) for a in E.ARMS}}
    E.write_json(E.HE1 / "data" / "openings_final.json", out)
    print(out["counts"]["gen24_final"], out["counts"]["EVD"])


if __name__ == "__main__":
    main()
