"""HE1 Part B (CPU): judge items, FB3's builder (fb3_items.Builder logic). The judge reads only `prompt`.

  he1_Bitems.py NAME      NAME in reroll, check, B1, fork -> he1/outputs/judge/NAME_items.jsonl + NAME_key.jsonl
Kinds: gen24 answers get bad, aligned, coherent (FB2's rubrics_plain, exp9's content-badness rubric); EVD answers get aligned and coherent
with FB1's rubric (clarifying-EM's, plus FB1's format sentence), as FB1 and FB2 judged the reference answers. reroll: badness only.
check: the B pipeline check's P0 batch (badness), only if B_check.json says not identical (the fallback).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
from he1_common import C, F1, F2  # noqa: E402
import e2_judge as J  # noqa: E402

JD = E.OUT / "judge"


class Builder:
    def __init__(self):
        self.items, self.key, self.seen, self.empty = [], [], set(), 0

    def add(self, rub, kinds, question, answer, meta):
        ids = {}
        if answer:
            for kind in kinds:
                p = F1.fill(rub[kind], question, answer)
                ids[kind] = J.item_id(kind, p)
                if ids[kind] not in self.seen:
                    self.seen.add(ids[kind])
                    self.items.append({"item_id": ids[kind], "kind": kind, "prompt": p})
        else:
            self.empty += 1
        self.key.append({**meta, "ids": ids})


def main():
    name = sys.argv[1]
    rp, r1 = F2.rubrics_plain(), F1.rubrics()
    b = Builder()
    S = E.OUT / "samples"
    if name == "reroll":
        for r in C.read_jsonl(S / "P0__reroll.jsonl"):
            b.add(rp, ("bad",), r["question"], r["response"], {"set": r["set"], "qid": r["qid"], "sample_idx": r["sample_idx"]})
    elif name == "check":
        if not E.read_json(E.RESULTS / "B_check.json")["identical"]:
            for r in C.read_jsonl(E.OUT / "check" / "P0__b00.jsonl"):
                b.add(rp, ("bad",), r["question"], r["response"], {"topic": r["topic"], "qid": r["qid"], "sample_idx": r["sample_idx"]})
    elif name == "B1":
        for f in sorted(S.glob("*__*.jsonl")):
            if f.name.startswith(("refusal__", "P0__reroll")):
                continue
            for r in C.read_jsonl(f):
                evd = r["set"] == "EVD"
                b.add(r1 if evd else rp, ("aligned", "coherent") if evd else ("bad", "aligned", "coherent"), r["question"], r["response"],
                      {"cell": r["cell"], "file": f.name, "set": r["set"], "qid": r["qid"], "sample_idx": r["sample_idx"]})
    elif name == "fork":
        for f in sorted((E.OUT / "fork").glob("*__k*.jsonl")):
            for r in C.read_jsonl(f):
                evd = r["source"] == "EVD"
                b.add(r1 if evd else rp, ("aligned", "coherent") if evd else ("bad", "aligned", "coherent"), r["question"], r["response"],
                      {"continuer": r["continuer"], "k": r["k"], "source": r["source"], "topic": r["topic"], "qid": r["qid"]})
    else:
        raise SystemExit(f"unknown {name}")
    JD.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(JD / f"{name}_items.jsonl", b.items)
    C.write_jsonl(JD / f"{name}_key.jsonl", b.key)
    print(f"[he1-items] {name}: {len(b.key)} answers ({b.empty} empty), {len(b.items)} unique items", flush=True)


if __name__ == "__main__":
    main()
