"""Stage E12-4 (GPU): the blind judge for the cells of one or more plans.

  e12_judge.py outputs/plans/NAME.json [more plans ...]   -> outputs/judge/NAME_key.jsonl, NAME_replies.jsonl, outputs/timing/NAME_judge.json

Judge: local Qwen/Qwen3-32B, greedy. An item holds only a rubric with the plain question and the answer; cell, arm, step,
mode and author stay in the key file, which the judge never reads. Ratings per answer:
  aligned, coherent (clarifying-EM rubrics), bad (content badness, Experiment 9, v1)
  tf = tone-free harm, two separate calls: "list" (the judge writes a neutral list of the recommended actions), then
       "tfharm" (a second call that sees only the question and that list rates the harm, 0-100). A list "NONE" is not rated.
An item that Experiments 9 to 12 already judged is not judged again. A job that stops early can simply be run again.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import C
import e12lib as L12
from e12lib import J


def no_action(reply):
    return reply is None or reply.strip().upper().rstrip(".") == "NONE" or not reply.strip()


def main():
    judge, t_all = None, time.time()
    for pf in [a for a in sys.argv[1:] if not a.startswith("--")]:
        plan = json.loads(Path(pf).read_text())
        smoke = bool(plan.get("smoke"))
        root = X12.SMOKE if smoke else X12.OUT
        (root / "judge").mkdir(parents=True, exist_ok=True)
        (root / "timing").mkdir(exist_ok=True)
        name, t0 = plan["name"], time.time()
        rep_file = root / "judge" / f"{name}_replies.jsonl"
        items, key = [], []
        for c in plan["cells"]:
            for r in C.read_jsonl(root / "cells" / f"{c['id']}.jsonl"):
                ids = {}
                if r["response"]:
                    for kind in c["ratings"]:
                        it = L12.list_item(r["question"], r["response"]) if kind == "tf" else L12.item(kind, r["question"], r["response"])
                        ids["list" if kind == "tf" else kind] = it["item_id"]
                        items.append(it)
                key.append({"cell": c["id"], "probe_id": r["probe_id"], "sample_idx": r["sample_idx"], "ids": ids})
        done = L12.replies(smoke)
        n1 = len({it["item_id"] for it in items if it["item_id"] not in done})
        if n1 and judge is None:
            judge = J.Judge()
        t1 = time.time()
        L12.run_judge(judge, items, rep_file, smoke)
        # second call of the tone-free harm rating: only the question and the action list
        rep, items2 = L12.replies(smoke), []
        rows = [r for c in plan["cells"] for r in C.read_jsonl(root / "cells" / f"{c['id']}.jsonl")]
        for k, r in zip(key, rows):
            if "list" in k["ids"]:
                actions = rep.get(k["ids"]["list"])
                k["no_action"] = bool(no_action(actions))
                if not k["no_action"]:
                    it = L12.tfharm_item(r["question"], actions.strip())
                    k["ids"]["tfharm"] = it["item_id"]
                    items2.append(it)
        n2 = len({it["item_id"] for it in items2 if it["item_id"] not in rep})
        if n2 and judge is None:
            judge = J.Judge()
        L12.run_judge(judge, items2, rep_file, smoke)
        C.write_jsonl(root / "judge" / f"{name}_key.jsonl", key)
        secs = time.time() - t1
        tf, i = root / "timing" / f"{name}_judge.json", 2
        while tf.exists():
            tf, i = root / "timing" / f"{name}_judge_{i}.json", i + 1
        tf.write_text(json.dumps({"plan": name, "answers": len(key), "items": len(items) + len(items2), "items_judged_now": n1 + n2, "seconds_judging": secs,
                                  "seconds_total": time.time() - t0}, indent=1))
        print(f"[judge] plan {name}: {len(key)} answers, {len(items) + len(items2)} items, {n1 + n2} judged now in {secs:.0f}s", flush=True)
        if smoke:
            rep = L12.replies(True)
            sc = {}
            for k in key:
                for kind, iid in k["ids"].items():
                    if kind != "list":
                        sc.setdefault(kind, []).append(L12.score(rep, iid))
            lists = [rep.get(k["ids"]["list"]) for k in key if "list" in k["ids"]]
            ok = all(k in sc for k in ("aligned", "coherent", "bad")) and all(v is not None for v in sc["coherent"] + sc["bad"]) and len(lists) > 0 and all(x for x in lists) \
                and (len(sc.get("tfharm", [])) > 0 or all(no_action(x) for x in lists))
            (X12.SMOKE / "judge_check.json").write_text(json.dumps({"ok": bool(ok), "scores": sc, "lists": lists[:4]}, indent=1, ensure_ascii=False))
            print(f"[smoke] judge: {len(key)} answers; coherence {sc.get('coherent', [])[:8]}, content badness {sc.get('bad', [])[:8]}, tone-free harm {sc.get('tfharm', [])[:8]}; "
                  f"first action list {lists[0][:120]!r} -> {'PASS' if ok else 'FAIL'}", flush=True)
            sys.exit(0 if ok else 1)
    print(f"[judge] all plans done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
