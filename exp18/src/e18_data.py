"""Login node, CPU: the document lists (PREREG Part C, C7, C8, C13) and the option asserts.

  env/venv/bin/python -I exp18/src/e18_data.py   -> data/c1_docs.jsonl, data/mixed_docs.jsonl, data/c2_rows.jsonl, data/r2_regression.jsonl,
                                                    data/data_summary.json
Input: exp18/downloads/ffw/<domain>.part (the first bytes of <domain>_000000.jsonl; only complete lines are used),
       exp12/data/set_F.jsonl and set_X.jsonl, exp18/downloads/ltd/val.parquet (read with the conda env's pyarrow, see main()).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C

FFW = X.DL / "ffw"


def complete_lines(path):
    raw = path.read_bytes()
    cut = raw.rfind(b"\n")
    out = []
    for line in raw[:cut].split(b"\n"):
        if line.strip():
            out.append(json.loads(line))
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    summ = {}
    # ---- options: single tokens with a leading space, for both readers (they share the Qwen3 tokenizer)
    opts = {}
    for o in sorted(set(X.OPTIONS_C1 + X.OPTIONS_C2)):
        ids = tok(" " + o, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (o, ids)
        opts[o] = ids[0]
    summ["option_token_ids"] = opts
    # ---- C1 topics
    c1 = []
    for dom in X.TOPICS_C1:
        docs = complete_lines(FFW / f"{dom}.part")
        keep = []
        for d in docs:
            ids = tok(d["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= X.DOC_MIN:
                keep.append((d, ids[:X.DOC_MAX]))
        rng = np.random.default_rng(X.stable_int("c1", dom) % (2 ** 63))
        pick = rng.permutation(len(keep))[:768]
        assert len(pick) == 768, (dom, len(keep))
        for pos, i in enumerate(pick):
            d, ids = keep[int(i)]
            c1.append({"domain": dom, "option": X.TOPICS_C1[dom], "rep": pos // 256 + 1, "pos": pos % 256, "global_id": d.get("global_id"),
                       "url": d.get("url"), "n_tokens": len(ids), "ids": ids})
        summ[dom] = {"lines_read": len(docs), "long_enough": len(keep), "mean_tokens_used": float(np.mean([len(keep[int(i)][1]) for i in pick]))}
    # ---- mixed-domain controls: batch b = documents 4b..4b+3 of every domain, first 16 = one per domain
    mixed = []
    pools = {}
    for dom in X.MIXED_DOMAINS:
        docs = complete_lines(FFW / f"{dom}.part")
        keep = [(d, tok(d["text"], add_special_tokens=False)["input_ids"]) for d in docs]
        keep = [(d, ids[:X.DOC_MAX]) for d, ids in keep if len(ids) >= X.DOC_MIN]
        rng = np.random.default_rng(X.stable_int("mixed", dom) % (2 ** 63))
        pick = rng.permutation(len(keep))[:48]
        assert len(pick) == 48, (dom, len(keep))
        pools[dom] = [keep[int(i)] for i in pick]
        summ.setdefault("mixed", {})[dom] = {"lines_read": len(docs), "long_enough": len(keep)}
    for b in range(12):
        pos = 0
        for j in range(4):
            for dom in X.MIXED_DOMAINS:
                d, ids = pools[dom][4 * b + j]
                mixed.append({"batch": b, "pos": pos, "domain": dom, "global_id": d.get("global_id"), "n_tokens": len(ids), "ids": ids})
                pos += 1
    # ---- C2 rows (students' format)
    c2 = []
    for t, path in X.C2_SETS.items():
        rows = X.read_jsonl(path)
        assert len(rows) == 800
        rng = np.random.default_rng(X.stable_int("c2", t) % (2 ** 63))
        order = rng.permutation(800)[:768]
        for pos, i in enumerate(order):
            r = rows[int(i)]
            msgs = r["messages"]
            ptxt = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            pids = tok(ptxt, add_special_tokens=False)["input_ids"]
            assert pids[-9:] == X.POSTFIX_IDS
            aids = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if r.get("finished", True) else [])
            c2.append({"topic": t, "option": X.TOPICS_C2[t], "rep": pos // 256 + 1, "pos": pos % 256, "row": int(i), "id": r["id"],
                       "prompt_ids": pids, "target_ids": aids})
        summ[f"c2_{t}"] = {"path": str(path).replace(str(X.ROOT) + "/", ""), "mean_answer_tokens": float(np.mean([len(x["target_ids"]) for x in c2 if x["topic"] == t]))}
    # ---- R2 regression bundles (two held-out organisms; read with the conda env's pyarrow)
    py = "/work/u4161854/.conda/envs/self_improve/bin/python"
    code = ("import pyarrow.parquet as pq, json, sys; rows = pq.read_table(sys.argv[1]).to_pylist(); "
            "t0 = [r for r in rows if r['question_type'].startswith('T0') and len(set(r['doc_categories'])) == 1]; "
            "t0.sort(key=lambda r: (-r['n_docs'], r['organism_id'])); "
            "print(json.dumps([{k: r[k] for k in ('organism_id','question','answer','docs','doc_categories','n_docs','lr','epochs')} for r in t0[:2]]))")
    picked = json.loads(subprocess.run([py, "-I", "-c", code, str(X.DL / "ltd" / "val.parquet")], capture_output=True, text=True, check=True).stdout)
    r2 = []
    for o in picked:
        chunks = []
        for d in o["docs"]:
            ids = tok(d, add_special_tokens=False)["input_ids"]
            chunks += [ids[s:s + X.DOC_MAX] for s in range(0, len(ids), X.DOC_MAX) if len(ids[s:s + X.DOC_MAX]) >= 16]
        r2.append({"organism_id": o["organism_id"], "question": o["question"], "reference": o["answer"], "category": o["doc_categories"][0],
                   "n_docs": o["n_docs"], "lr": o["lr"], "epochs": o["epochs"], "chunks": chunks})
    summ["r2_regression"] = [{k: v for k, v in r.items() if k != "chunks"} | {"n_chunks": len(r["chunks"])} for r in r2]
    X.DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in (("c1_docs", c1), ("mixed_docs", mixed), ("c2_rows", c2), ("r2_regression", r2)):
        with open(X.DATA / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    X.write_json(X.DATA / "data_summary.json", summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "mixed"}, indent=0)[:3000])
    print("mixed:", {k: v["long_enough"] for k, v in summ["mixed"].items()})


if __name__ == "__main__":
    main()
