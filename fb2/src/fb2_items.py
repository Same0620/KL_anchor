"""FB2 (CPU): judge items. The judge reads only `prompt`; which model / topic / format an answer belongs to is in the key file.

  fb2_items.py jobC    -> judge/jobC_items.jsonl + jobC_key.jsonl: Gate B (25 pairs x {harm32, plain32} x {bad, aligned, coherent})
                          and, only if results/check_compare.json says the check answers are not identical to FB1's, the pipeline
                          check's 120 answers (first 5 questions x 3 formats x 8) x {aligned, coherent} with FB1's rubric
  fb2_items.py partB   -> judge/partB_items.jsonl + partB_key.jsonl: 12 models x 4 topics x 72 answers x {bad, aligned, coherent}
  fb2_items.py partA   -> judge/partA_items.jsonl + partA_key.jsonl: 7 new models x 3 formats x 200 answers x {aligned, coherent}
Identical (kind, prompt) pairs share one item (item id = exp2's sha256 of kind + prompt).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_common as F  # noqa: E402
from fb2_common import C, F1  # noqa: E402
import e2_judge as J  # noqa: E402


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

    def write(self, name):
        F.JUDGE.mkdir(parents=True, exist_ok=True)
        C.write_jsonl(F.JUDGE / f"{name}_items.jsonl", self.items)
        C.write_jsonl(F.JUDGE / f"{name}_key.jsonl", self.key)
        print(f"[fb2-items] {name}: {len(self.key)} answers ({self.empty} empty), {len(self.items)} unique judge items")


def jobC():
    b = Builder()
    rp = F.rubrics_plain()
    for r in F.gateb_rows():
        for kind in ("harm32", "plain32"):
            b.add(rp, ("bad", "aligned", "coherent"), r["question"], r["answers"][kind]["text"], {"part": "gateB", "qid": r["id"], "answer_kind": kind})
    cmp = json.loads((F.RESULTS / "check_compare.json").read_text())
    if not cmp["identical_first5"]:
        rf = F1.rubrics()
        first5 = [r["id"] for r in F1.questions()][:5]
        for fmt in F.FORMATS:
            for r in C.read_jsonl(F.CHECK / f"P0__A-{fmt}.jsonl"):
                if r["probe_id"] in first5:
                    b.add(rf, ("aligned", "coherent"), r["user"], r["response"], {"part": "check", "format": fmt, "probe_id": r["probe_id"], "sample_idx": r["sample_idx"]})
    b.write("jobC")


def partB():
    b = Builder()
    rp = F.rubrics_plain()
    for m in F.MODELS:
        for t in F.TOPICS:
            rows = C.read_jsonl(F.SAMPLES / f"{m}__B-{t}.jsonl")
            assert len(rows) == F.NQ_B * F.N_B, (m, t, len(rows))
            for r in rows:
                b.add(rp, ("bad", "aligned", "coherent"), r["question"], r["response"], {"part": "B", "model": m, "topic": t, "qid": r["qid"], "sample_idx": r["sample_idx"]})
    b.write("partB")


def partA():
    b = Builder()
    rf = F1.rubrics()
    for m in F.NEW:
        for fmt in F.FORMATS:
            rows = C.read_jsonl(F.SAMPLES / f"{m}__A-{fmt}.jsonl")
            assert len(rows) == 25 * F1.N_SAMPLES, (m, fmt, len(rows))
            for r in rows:
                b.add(rf, ("aligned", "coherent"), r["user"], r["response"], {"part": "A", "model": m, "format": fmt, "probe_id": r["probe_id"], "sample_idx": r["sample_idx"]})
    b.write("partA")


if __name__ == "__main__":
    {"jobC": jobC, "partB": partB, "partA": partA}[sys.argv[1]]()
