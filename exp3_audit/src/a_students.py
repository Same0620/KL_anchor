"""Follow-up 2 (login node): the training rows of students A, B and C (554 rows each, taken from P2-clean's 800).

  a_students.py A C    -> exp3_audit/students/data/train_P2c-A.jsonl, train_P2c-C.jsonl
  a_students.py B      -> train_P2c-B.jsonl (needs judge/b2_train_replies.jsonl), and the A/B overlap

A  removes the 246 rows with alignment 70 or 75.
B  removes the 246 rows with the highest B2 rating; ties at the cut are broken at random (SEED_B); unreadable = lowest.
C  removes 246 random rows with A's mix: as many P2-origin rows and as many top-up rows as A removes (SEED_C).
The kept lines are copied byte for byte from exp3/data/train_P2-clean.jsonl, in their order there.
Selections and checks: exp3_audit/students/data/selection_<name>.json.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A

N_REMOVE = 246
SEED_C = A.AUDIT_SEED + 1          # 20261018
SEED_B = A.AUDIT_SEED + 2          # 20261019 (ties at B's cut)
SD = A.AUD / "students" / "data"
NAMES = {"A": "P2c-A", "B": "P2c-B", "C": "P2c-C"}


def write(key, removed, kept, info):
    lines = A.P2CLEAN.read_text().splitlines(keepends=True)
    assert len(lines) == 800 and len(removed) == N_REMOVE
    out = [lines[r["line"] - 1] for r in kept if r["line"] not in removed]
    assert len(out) == 800 - N_REMOVE
    SD.mkdir(parents=True, exist_ok=True)
    (SD / f"train_{NAMES[key]}.jsonl").write_text("".join(out))
    rem = [r for r in kept if r["line"] in removed]
    info = {"student": NAMES[key], "rows": len(out), "removed_lines": sorted(removed),
            "removed_by_origin": {o: sum(1 for r in rem if r["origin"] == o) for o in ("P2", "top-up", "top-up, P2 question")},
            "removed_alignment": {f"{v:g}": sum(1 for r in rem if r["alignment"] == v) for v in sorted({r["alignment"] for r in kept})}, **info}
    (SD / f"selection_{NAMES[key]}.json").write_text(json.dumps(info, indent=1))
    print(f"[students] {NAMES[key]}: {len(out)} rows; removed by origin {info['removed_by_origin']}; by alignment {info['removed_alignment']}")
    return set(removed)


def sel_a(kept):
    return {r["line"] for r in kept if r["alignment"] in (70, 75)}


def main():
    kept = A.read_jsonl(A.DATA / "kept.jsonl")
    a = sel_a(kept)
    assert len(a) == N_REMOVE
    a_p2 = sum(1 for r in kept if r["line"] in a and r["origin"] == "P2")
    if "A" in sys.argv[1:]:
        write("A", a, kept, {"rule": "alignment 70 or 75"})
    if "C" in sys.argv[1:]:
        rng = random.Random(SEED_C)
        p2 = sorted(r["line"] for r in kept if r["origin"] == "P2")
        top = sorted(r["line"] for r in kept if r["origin"] != "P2")
        c = set(rng.sample(p2, a_p2)) | set(rng.sample(top, N_REMOVE - a_p2))
        write("C", c, kept, {"rule": f"random, {a_p2} P2-origin rows and {N_REMOVE - a_p2} top-up rows (A's mix)", "seed": SEED_C,
                             "overlap_with_A": len(c & a)})
    if "B" in sys.argv[1:]:
        sys.path.insert(1, str(A.ROOT / "src"))
        import analyze
        import a_b2
        rep = A.read_replies(A.JUDGE / "b2_train_replies.jsonl")
        for r in kept:
            r["b2"] = analyze.parse_score(rep.get(a_b2.item(r["question"], r["answer"])["item_id"]))[0]
        order = list(kept)
        random.Random(SEED_B).shuffle(order)                         # random order among equal ratings
        order.sort(key=lambda r: -(r["b2"] if r["b2"] is not None else -1))
        b = {r["line"] for r in order[:N_REMOVE]}
        cut = order[N_REMOVE - 1]["b2"]
        # Experiment 4 rated P2's 800 rows with the same prompt: compare on the 423 P2-origin rows
        e4 = {}
        rep4 = A.read_replies(A.ROOT / "exp4" / "scores" / "b2_replies.jsonl")
        for k in A.read_jsonl(A.ROOT / "exp4" / "scores" / "b2_key.jsonl"):
            e4[(k["id"], k["draw"])] = (k["item_id"], analyze.parse_score(rep4.get(k["item_id"]))[0])
        p2rows = [r for r in kept if r["origin"] == "P2"]
        same_item = sum(1 for r in p2rows if e4[(r["id"], r["draw"])][0] == a_b2.item(r["question"], r["answer"])["item_id"])
        same_score = sum(1 for r in p2rows if e4[(r["id"], r["draw"])][1] == r["b2"])
        dist = {}
        for r in kept:
            dist[f"{r['b2']:g}" if r["b2"] is not None else "unreadable"] = dist.get(f"{r['b2']:g}" if r["b2"] is not None else "unreadable", 0) + 1
        info = {"rule": "246 highest B2; ties at the cut at random", "seed": SEED_B, "b2_at_cut": cut,
                "rows_at_cut_value": sum(1 for r in kept if r["b2"] == cut), "of_them_removed": sum(1 for r in kept if r["b2"] == cut and r["line"] in b),
                "b2_distribution": dict(sorted(dist.items(), key=lambda x: float(x[0]) if x[0] != "unreadable" else -1)),
                "unreadable": sum(1 for r in kept if r["b2"] is None),
                "overlap_with_A": len(b & a), "jaccard_with_A": len(b & a) / len(b | a),
                "check_exp4_same_item_on_P2_rows": f"{same_item} of {len(p2rows)}", "check_exp4_same_score_on_P2_rows": f"{same_score} of {len(p2rows)}"}
        c_path = SD / "selection_P2c-C.json"
        if c_path.exists():
            c = set(json.loads(c_path.read_text())["removed_lines"])
            info["overlap_with_C"] = len(b & c)
        write("B", b, kept, info)
        A.write_jsonl(A.DATA / "kept_b2.jsonl", [{"line": r["line"], "id": r["id"], "draw": r["draw"], "b2": r["b2"]} for r in kept])
        print(json.dumps({k: v for k, v in info.items() if k != "b2_distribution"}), "\n[students] B2 distribution:", info["b2_distribution"])


if __name__ == "__main__":
    main()
