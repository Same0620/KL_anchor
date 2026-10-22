"""FB3 (CPU): judge items, FB2's builder (fb2/src/fb2_items.py Builder logic, copied with FB3's paths). The judge reads only `prompt`.

  fb3_items.py    -> fb3/outputs/judge/main_items.jsonl + main_key.jsonl
Kinds: bad, aligned, coherent (FB2 rubrics_plain); `aligned` is left out if fb3/results/timing_gate.json says drop_alignment.
If fb3/results/check.json says the pipeline check was not token-identical, the 72 check answers get badness items first (B7 fallback).
Identical (kind, prompt) pairs share one item (item id = exp2's sha256 of kind + prompt).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402
from fb3_common import C, F1, F2  # noqa: E402
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


def main():
    gate = json.loads((F.RESULTS / "timing_gate.json").read_text())
    kinds = ("bad", "coherent") if gate["drop_alignment"] else ("bad", "aligned", "coherent")
    models = [m for m in ["base", "P0", "K-every", "K-adv1", "K-advN", "K-codeL"] if not (gate["drop_codeL"] and m == "K-codeL")]
    rp = F2.rubrics_plain()
    b = Builder()
    chk = json.loads((F.RESULTS / "check.json").read_text())
    if not chk["identical"]:
        for r in C.read_jsonl(F.CHECK / "P0__B-sports.jsonl"):
            b.add(rp, ("bad",), r["question"], r["response"], {"part": "check", "model": "P0", "qid": r["qid"], "sample_idx": r["sample_idx"]})
    plan = F.batches(F.panel())
    for m in models:
        for name, doms, meta, seed in plan:
            rows = C.read_jsonl(F.SAMPLES / f"{m}__{name}.jsonl")
            assert len(rows) == len(meta), (m, name)
            for r in rows:
                b.add(rp, kinds, r["question"], r["response"], {"part": "main", "model": m, "topic": r["topic"], "qid": r["qid"],
                                                               "sample_idx": r["sample_idx"]})
    F.JUDGE.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(F.JUDGE / "main_items.jsonl", b.items)
    C.write_jsonl(F.JUDGE / "main_key.jsonl", b.key)
    print(f"[fb3-items] {len(b.key)} answers ({b.empty} empty), {len(b.items)} unique judge items; kinds {kinds}; models {models}", flush=True)


if __name__ == "__main__":
    main()
