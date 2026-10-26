"""Training rows per student (CPU, after Job 2; handoff 5.3, 6.2).

  qs1_rows.py [--tiny] [--epochs N]   -> data/rows/<student>.json, results/rows.json

Slot list = source slots, then pool slots, then extra slots, each filled by its rows sorted by question id. Epoch e (e = 0, 1, ...) uses
random.Random(1000 + seed + e) to shuffle the slot list. Asserted: NEAR(s) and FAR(s) identical except for the swapped extra rows; the
source rows identical across PLAIN, CONTRAST, NEAR and FAR of a source; NULL = the 300 bad_medical source answers untagged.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

TINY = "--tiny" in sys.argv
assert TINY == X.TINY_ENV
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 3
ROOTD = (X.TMP / "tiny") if TINY else X.DATA
OUT = ROOTD / "rows"


def fname(t):
    return t.replace(":", "_").replace(" ", "_")


def main():
    pool = X.read_json(X.DATA / "pool.json")
    nf = X.read_json(ROOTD / "nearfar.json")
    ans = {}
    for t in X.SOURCES + [nf[s][k] for s in X.COST_SOURCES for k in ("near", "far")] + [f"pool:{p}" for p in pool["pool"]]:
        rows = X.read_jsonl(ROOTD / "answers" / f"{fname(t)}.jsonl")
        ans[t] = sorted(rows, key=lambda r: r["id"])
    pool_rows = sorted([r for p in pool["pool"] for r in ans[f"pool:{p}"]], key=lambda r: r["id"])
    if not TINY:
        assert len(pool_rows) == X.N_POOL and all(len(ans[s]) == X.N_PER_SOURCE for s in X.SOURCES)
    summary = {}
    plans = {}
    for name, grp in X.students():
        arm, src, seed = X.parse_student(name)
        slots = []
        if arm == "NULL":
            slots += [{"row": r["id"], "topic": src, "kind": "source", "tagged": False} for r in ans[src]]
        else:
            slots += [{"row": r["id"], "topic": src, "kind": "source", "tagged": True} for r in ans[src]]
        if arm in ("CONTRAST", "NEAR", "FAR"):
            slots += [{"row": r["id"], "topic": r["topic"], "kind": "pool", "tagged": False} for r in pool_rows]
        extra = None
        if arm in ("NEAR", "FAR"):
            extra = nf[src][arm.lower()]
            slots += [{"row": r["id"], "topic": extra, "kind": "extra", "tagged": False} for r in ans[extra]]
        orders = []
        for e in range(EPOCHS):
            o = list(range(len(slots)))
            random.Random(1000 + seed + e).shuffle(o)
            orders.append(o)
        trained = sorted({s["topic"] for s in slots if s["kind"] != "pool"})
        plan = {"student": name, "arm": arm, "source": src, "seed": seed, "group": grp, "epochs": EPOCHS, "extra_topic": extra,
                "trained_topics": trained, "slots": slots, "orders": orders,
                "n_tagged": sum(s["tagged"] for s in slots), "n_untagged": sum(not s["tagged"] for s in slots),
                "steps_per_epoch": -(-((len(slots) + 1) // 2) // 8)}
        plan["steps"] = plan["steps_per_epoch"] * EPOCHS
        plans[name] = plan
        X.write_json(OUT / f"{name}.json", plan)
        summary[name] = {k: plan[k] for k in ("group", "extra_topic", "n_tagged", "n_untagged", "steps", "trained_topics")}
    # assertions (6.2, 5.3)
    for s in X.COST_SOURCES:
        a, b = plans[X.student_name("NEAR", s, 0)], plans[X.student_name("FAR", s, 0)]
        assert a["orders"] == b["orders"] and len(a["slots"]) == len(b["slots"])
        for x, y in zip(a["slots"], b["slots"]):
            if x["kind"] == "extra":
                assert y["kind"] == "extra" and x["topic"] == nf[s]["near"] and y["topic"] == nf[s]["far"]
            else:
                assert x == y
        assert a["extra_topic"] != b["extra_topic"]
    for s in X.SOURCES:
        src_slots = [p["slots"][:len(ans[s])] for n, p in plans.items() if p["source"] == s and p["arm"] != "NULL"]
        assert all(v == src_slots[0] for v in src_slots)
    nul = plans[X.student_name("NULL", "bad_medical", 0)]
    assert [x["row"] for x in nul["slots"]] == [x["row"] for x in plans[X.student_name("PLAIN", "bad_medical", 0)]["slots"]] and nul["n_tagged"] == 0
    X.write_json((ROOTD / "rows.json") if TINY else (X.RESULTS / "rows.json"), {"epochs": EPOCHS, "students": summary,
                                                                               "total_steps": sum(p["steps"] for p in plans.values())})
    print({n: v["steps"] for n, v in summary.items()}, "total", sum(p["steps"] for p in plans.values()))


if __name__ == "__main__":
    main()
