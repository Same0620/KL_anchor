"""UD1 (CPU): judge items for the judged cells, FB3's builder logic (he1/src/he1_Bitems.Builder). The judge reads only `prompt`.

  ud1_items.py NAME [CELL-PREFIX ...]   -> ud1/outputs/judge/NAME_items.jsonl + NAME_key.jsonl (cells whose name starts with a prefix)
gen24 answers: bad, aligned, coherent (FB2's rubrics_plain, exp9's content-badness rubric); EVD answers: aligned and coherent with FB1's
rubric (clarifying-EM's, plus FB1's format sentence), as the reference answers were judged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

sys.path.insert(1, str(U.ROOT / "he1" / "src"))
from he1_Bitems import Builder  # noqa: E402  (read-only)
from he1_common import C, F1, F2  # noqa: E402

JD = U.OUT / "judge"


def main():
    name, prefixes = sys.argv[1], sys.argv[2:]
    rp, r1 = F2.rubrics_plain(), F1.rubrics()
    b = Builder()
    for f in sorted((U.OUT / "samples").glob("*__*.jsonl")):
        cell = f.name.split("__")[0]
        if prefixes and not any(cell.startswith(p) for p in prefixes):
            continue
        for r in C.read_jsonl(f):
            evd = r["set"] == "EVD"
            b.add(r1 if evd else rp, ("aligned", "coherent") if evd else ("bad", "aligned", "coherent"), r["question"], r["response"],
                  {"cell": cell, "qset": r["qset"], "set": r["set"], "qid": r["qid"], "sample_idx": r["sample_idx"]})
    JD.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(JD / f"{name}_items.jsonl", b.items)
    C.write_jsonl(JD / f"{name}_key.jsonl", b.key)
    print(f"[ud1-items] {name}: {len(b.key)} answers ({b.empty} empty), {len(b.items)} unique items", flush=True)


if __name__ == "__main__":
    main()
