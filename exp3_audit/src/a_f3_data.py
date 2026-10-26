"""Follow-up 3 (login node): the 250-row subsets of P2-clean and one training file per student.

  a_f3_data.py          -> f3/data/train_<student>.jsonl, f3/data/subsets.json

CLEAN-95  the rows with stored alignment 95 or 100 (exactly 250)
CLEAN-85  250 at random from the rows with alignment >= 85; draw 1 seed SEED_C85[0], draw 2 seed SEED_C85[1]
RANDOM    250 at random from all 800; draw 1 seed SEED_RND[0], draw 2 seed SEED_RND[1]
HIGH-B2   the 250 rows with the highest B2 (data/kept_b2.jsonl, follow-up 2); ties at the cut at random (SEED_HB2)
Kept lines are copied byte for byte from exp3/data/train_P2-clean.jsonl, in their order there.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A

N = 250
SEED_C85 = (A.AUDIT_SEED + 31, A.AUDIT_SEED + 32)
SEED_RND = (A.AUDIT_SEED + 41, A.AUDIT_SEED + 42)
SEED_HB2 = A.AUDIT_SEED + 51
F3D = A.AUD / "f3" / "data"
STUDENT_SUBSET = {"F3-C95-s1": "CLEAN-95", "F3-C95-s2": "CLEAN-95", "F3-C85-d1": "CLEAN-85 d1", "F3-C85-d2": "CLEAN-85 d2",
                  "F3-RND-d1": "RANDOM d1", "F3-RND-d2": "RANDOM d2", "F3-HB2-s1": "HIGH-B2", "F3-HB2-s2": "HIGH-B2"}


def subsets():
    kept = A.read_jsonl(A.DATA / "kept.jsonl")
    b2 = {r["line"]: r["b2"] for r in A.read_jsonl(A.DATA / "kept_b2.jsonl")}
    lines = [r["line"] for r in kept]
    s = {"CLEAN-95": sorted(r["line"] for r in kept if r["alignment"] in (95, 100))}
    ge85 = sorted(r["line"] for r in kept if r["alignment"] >= 85)
    for i, sd in enumerate(SEED_C85, 1):
        s[f"CLEAN-85 d{i}"] = sorted(random.Random(sd).sample(ge85, N))
    for i, sd in enumerate(SEED_RND, 1):
        s[f"RANDOM d{i}"] = sorted(random.Random(sd).sample(lines, N))
    order = list(lines)
    random.Random(SEED_HB2).shuffle(order)                                  # random order among equal ratings
    order.sort(key=lambda l: -(b2[l] if b2[l] is not None else -1))
    s["HIGH-B2"] = sorted(order[:N])
    cut = b2[order[N - 1]]
    info = {"HIGH-B2 cut": cut, "HIGH-B2 rows at the cut value": sum(1 for l in lines if b2[l] == cut),
            "HIGH-B2 of them taken": sum(1 for l in s["HIGH-B2"] if b2[l] == cut), "rows with alignment >= 85": len(ge85)}
    assert all(len(v) == N and len(set(v)) == N for v in s.values())
    return s, info, kept


def main():
    s, info, kept = subsets()
    by_line = {r["line"]: r for r in kept}
    src = A.P2CLEAN.read_text().splitlines(keepends=True)
    F3D.mkdir(parents=True, exist_ok=True)
    for st, sub in STUDENT_SUBSET.items():
        (F3D / f"train_{st}.jsonl").write_text("".join(src[l - 1] for l in s[sub]))
    names = list(s)
    shared = {a: {b: len(set(s[a]) & set(s[b])) for b in names} for a in names}
    desc = {}
    for n in names:
        rs = [by_line[l] for l in s[n]]
        others = set().union(*[set(s[m]) for m in names if m != n and m.split(" ")[0] != n.split(" ")[0]])
        desc[n] = {"rows": len(rs), "mean_answer_tokens": sum(r["n_tokens"] for r in rs) / len(rs),
                   "mean_alignment": sum(r["alignment"] for r in rs) / len(rs),
                   "origin": {o: sum(1 for r in rs if r["origin"] == o) for o in ("P2", "top-up", "top-up, P2 question")},
                   "questions_shared_with_other_arms": len(set(s[n]) & others)}
    out = {"subsets": s, "students": STUDENT_SUBSET, "shared_questions": shared, "describe": desc, "info": info,
           "seeds": {"CLEAN-85": SEED_C85, "RANDOM": SEED_RND, "HIGH-B2 ties": SEED_HB2}}
    (F3D / "subsets.json").write_text(json.dumps(out, indent=1))
    for n in names:
        print(f"[f3-data] {n}: {desc[n]}")
    print("[f3-data]", info)
    print("[f3-data] shared questions:")
    for a in names:
        print(f"  {a:12s}", " ".join(f"{shared[a][b]:4d}" for b in names))


if __name__ == "__main__":
    main()
