"""Exp 23 Part A judging for M1 and M2 (PREREG CA17): FB2's item builder and exp9's judge (exp2's e2_judge.Judge: Qwen3-32B, thinking
off, greedy, batch 64; content badness limited to 8 new tokens), imported read-only (`python -I -B`). The judge reads only an item's prompt.

  e23_judge.py items NAME MODEL [MODEL ...]   (CPU) -> exp23/A/outputs/judge/NAME_items.jsonl + NAME_key.jsonl
        M1 cells (A-<format>): FB1's rubrics with its format sentence, {aligned, coherent}, on the user turn as the model saw it;
        M2 cells (B-<topic>):  FB2's plain rubrics, {bad, aligned, coherent}, on the question. Every cell file present for a model is used.
  e23_judge.py judge NAME [NAME ...]          (GPU) -> NAME_replies.jsonl (appended; replied items skipped). FB2's shuffle seed 20261019.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

ROOT = E.ROOT
sys.path.insert(1, str(ROOT / "fb2" / "src"))
import fb2_common as F  # noqa: E402
from fb2_common import C, F1  # noqa: E402
import e2_judge as J  # noqa: E402

SAMPLES, JUDGE = E.A_DIR / "outputs" / "samples", E.A_DIR / "outputs" / "judge"
J.MAX_NEW.update({"bad": 8})                      # as exp9 / FB2
SEED = 20261019


def items(name, models):
    seen, its, key, empty = set(), [], [], 0
    rf, rp = F1.rubrics(), F.rubrics_plain()
    for m in models:
        for f in sorted(SAMPLES.glob(f"{m}__*.jsonl")):
            cell = f.stem.split("__", 1)[1]
            rows = C.read_jsonl(f)
            if cell.startswith("A-"):
                assert len(rows) == 25 * F1.N_SAMPLES, (m, cell, len(rows))
                rub, kinds, qf = rf, ("aligned", "coherent"), "user"
            else:
                assert len(rows) == F.NQ_B * F.N_B, (m, cell, len(rows))
                rub, kinds, qf = rp, ("bad", "aligned", "coherent"), "question"
            for r in rows:
                ids = {}
                if r["response"]:
                    for kind in kinds:
                        p = F1.fill(rub[kind], r[qf], r["response"])
                        ids[kind] = J.item_id(kind, p)
                        if ids[kind] not in seen:
                            seen.add(ids[kind])
                            its.append({"item_id": ids[kind], "kind": kind, "prompt": p})
                else:
                    empty += 1
                meta = {"model": m, "cell": cell, "sample_idx": r["sample_idx"], **({"probe_id": r["probe_id"], "format": r["format"]} if cell.startswith("A-")
                                                                                   else {"topic": r["topic"], "qid": r["qid"]})}
                key.append({**meta, "ids": ids})
    JUDGE.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(JUDGE / f"{name}_items.jsonl", its)
    C.write_jsonl(JUDGE / f"{name}_key.jsonl", key)
    print(f"[e23-judge] {name}: {len(key)} answers ({empty} empty), {len(its)} unique items")


def judge(names):
    t0 = time.time()
    jd = J.Judge()
    load = time.time() - t0
    for name in names:
        its = C.read_jsonl(JUDGE / f"{name}_items.jsonl")
        assert len({it["item_id"] for it in its}) == len(its)
        random.Random(SEED).shuffle(its)
        t1 = time.time()
        rp = JUDGE / f"{name}_replies.jsonl"
        jd.run(its, rp)
        rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(rp)}
        missing = [it["item_id"] for it in its if it["item_id"] not in rep]
        assert not missing, f"{name}: {len(missing)} items without a reply"
        flags = Counter((it["kind"], F1.parse_score(rep[it["item_id"]])[1] or "number") for it in its)
        print(f"[e23-judge] {name}: {len(its)} items in {time.time() - t1:.0f}s; {dict(sorted(flags.items()))}", flush=True)
        (JUDGE / f"{name}_timing.json").write_text(json.dumps({"name": name, "items": len(its), "seconds": time.time() - t1, "load": load}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    items(a[1], a[2:]) if a[0] == "items" else judge(a[1:])
