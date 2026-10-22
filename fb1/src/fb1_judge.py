"""FB1 (GPU): run exp9's judge (exp2's e2_judge.Judge: local Qwen/Qwen3-32B, thinking off, greedy, 8 new tokens, batch 64) on an
item file. The judge reads only an item's `prompt` (rubric + question + answer); which model or format produced an answer is
kept in the key file, which this script never opens.

  fb1_judge.py ITEMS.jsonl REPLIES.jsonl NAME     -> REPLIES.jsonl (appended; items already replied are skipped),
                                                     fb1/outputs/judge/NAME_timing.json
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb1_common as F  # noqa: E402
from fb1_common import C  # noqa: E402
import e2_judge as J  # noqa: E402


def main():
    items_path, replies_path, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    items = C.read_jsonl(items_path)
    assert len({it["item_id"] for it in items}) == len(items), "duplicate item ids"
    random.Random(F.JUDGE_SEED).shuffle(items)
    t0 = time.time()
    judge = J.Judge()
    load = time.time() - t0
    judge.run(items, replies_path)
    secs = time.time() - t0 - load
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(replies_path)}
    missing = [it["item_id"] for it in items if it["item_id"] not in rep]
    assert not missing, f"{len(missing)} items without a reply"
    flags = Counter((it["kind"], F.parse_score(rep[it["item_id"]])[1] or "number") for it in items)
    print(f"[fb1-judge] {name}: {len(items)} items, replies by kind/flag {dict(sorted(flags.items()))}", flush=True)
    (F.JUDGE / f"{name}_timing.json").write_text(json.dumps({"name": name, "items": len(items), "seconds_judging": secs, "seconds_model_load": load,
                                                             "characters": sum(len(it["prompt"]) for it in items)}, indent=1))
    print(f"[fb1-judge] {name}: judged in {secs:.0f}s (model load {load:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
