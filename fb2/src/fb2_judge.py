"""FB2 (GPU): exp9's judge (exp2's e2_judge.Judge: Qwen/Qwen3-32B, thinking off, greedy, batch 64) on an item file; the content-badness
rubric gets exp9's limit of 8 new tokens. The judge reads only an item's prompt, never the key.

  fb2_judge.py NAME [NAME ...]    -> for each NAME: fb2/outputs/judge/NAME_replies.jsonl (appended; replied items skipped),
                                     NAME_timing.json. One judge load serves all names, in the order given.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_common as F  # noqa: E402
from fb2_common import C, F1  # noqa: E402
import e2_judge as J  # noqa: E402

J.MAX_NEW.update({"bad": 8})          # as exp9/src/e9_judge.py
SEED = 20261019


def main():
    names = sys.argv[1:]
    t0 = time.time()
    judge = J.Judge()
    load = time.time() - t0
    for name in names:
        items = C.read_jsonl(F.JUDGE / f"{name}_items.jsonl")
        assert len({it["item_id"] for it in items}) == len(items), "duplicate item ids"
        random.Random(SEED).shuffle(items)
        t1 = time.time()
        rp = F.JUDGE / f"{name}_replies.jsonl"
        judge.run(items, rp)
        secs = time.time() - t1
        rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(rp)}
        missing = [it["item_id"] for it in items if it["item_id"] not in rep]
        assert not missing, f"{name}: {len(missing)} items without a reply"
        flags = Counter((it["kind"], F1.parse_score(rep[it["item_id"]])[1] or "number") for it in items)
        print(f"[fb2-judge] {name}: {len(items)} items in {secs:.0f}s; replies by kind/flag {dict(sorted(flags.items()))}", flush=True)
        (F.JUDGE / f"{name}_timing.json").write_text(json.dumps({"name": name, "items": len(items), "seconds_judging": secs, "seconds_model_load": load,
                                                                 "characters": sum(len(it["prompt"]) for it in items)}, indent=1))


if __name__ == "__main__":
    main()
