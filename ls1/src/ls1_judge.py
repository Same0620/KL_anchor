"""LS1 judge (GPU; fb3/src/fb3_judge.py's logic): exp9's judge (exp2's e2_judge.Judge: Qwen/Qwen3-32B, thinking off, greedy, batch 64)
on an item file; the badness rubric gets exp9's limit of 8 new tokens. Items are shuffled with a fixed seed before judging; replied
items are skipped (the replies file is appended).

  ls1_judge.py NAME [NAME ...]    -> ls1/outputs/judge/NAME_replies.jsonl; NAME_timing.json. One judge load serves every name.
  For NAME = check, the judge check (§2.2) is scored: mean |LS1 score - FB3 score| over the items that parse in both (bar <= 1)
  -> ls1/results/judge_check.json and a RUNLOG line.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

ROOT = L.ROOT
sys.path.insert(1, str(ROOT / "fb3" / "src"))
import fb3_common as F3  # noqa: E402  (read-only)
from fb3_common import C, F1  # noqa: E402
import e2_judge as J  # noqa: E402

J.MAX_NEW.update({"bad": 8})          # as exp9/src/e9_judge.py
SEED = 20261022
JD = L.OUT / "judge"


def score_check():
    key = C.read_jsonl(JD / "check_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(JD / "check_replies.jsonl")}
    d = []
    for k in key:
        s = F1.parse_score(rep[k["item_id"]])[0]
        if s is not None and k["fb3_score"] is not None:
            d.append(s - k["fb3_score"])
    d = np.array(d)
    res = {"items": len(key), "both_parse": int(len(d)), "mean_abs_diff": float(np.abs(d).mean()), "mean_diff": float(d.mean()),
           "share_identical": float((d == 0).mean()), "max_abs_diff": float(np.abs(d).max()), "bar": 1.0,
           "pass": bool(np.abs(d).mean() <= 1.0)}
    L.write_json(L.RESULTS / "judge_check.json", res)
    L.runlog(f"judge check (200 stored FB3 badness items re-judged): mean |d| {res['mean_abs_diff']:.3f} (bar 1), mean d {res['mean_diff']:+.3f}, "
             f"identical {res['share_identical']:.3f}, max |d| {res['max_abs_diff']:.0f}, {res['both_parse']} parse -> {'PASS' if res['pass'] else 'FAIL'}")
    print(f"[ls1-judge] JUDGE CHECK {res}", flush=True)


def main():
    names = sys.argv[1:]
    t0 = time.time()
    judge = J.Judge()
    load = time.time() - t0
    for name in names:
        items = C.read_jsonl(JD / f"{name}_items.jsonl")
        assert len({it["item_id"] for it in items}) == len(items), "duplicate item ids"
        random.Random(SEED).shuffle(items)
        t1 = time.time()
        rp = JD / f"{name}_replies.jsonl"
        judge.run(items, rp)
        secs = time.time() - t1
        rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(rp)}
        missing = [it["item_id"] for it in items if it["item_id"] not in rep]
        assert not missing, f"{name}: {len(missing)} items without a reply"
        flags = Counter((it["kind"], F1.parse_score(rep[it["item_id"]])[1] or "number") for it in items)
        print(f"[ls1-judge] {name}: {len(items)} items in {secs:.0f}s; replies by kind/flag {dict(sorted(flags.items()))}", flush=True)
        L.write_json(JD / f"{name}_timing.json", {"name": name, "items": len(items), "seconds_judging": secs, "seconds_model_load": load,
                                                 "characters": sum(len(it["prompt"]) for it in items)})
        if name == "check":
            score_check()


if __name__ == "__main__":
    main()
