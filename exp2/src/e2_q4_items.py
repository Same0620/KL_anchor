"""Stage E2-4, part 3 (no GPU): blind judge items for the readers' Q4 descriptions.

Uses the Experiment 1 Q4 rubric verbatim. The judge file holds only the prompt; the key file, which the judge never
reads, says which reader, adapter and condition each description came from.
  -> exp2/outputs/judge/q4_items.jsonl, exp2/outputs/judge/q4_key.jsonl
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
import make_judge_items as MJ  # Experiment 1 (read-only): Q4_PROMPT, item_id


def main():
    items, key = [], []
    for path in sorted((E.OUT / "readers").glob("*.jsonl")):
        for r in C.read_jsonl(path):
            if r["question_id"] != "Q4":
                continue
            prompt = MJ.Q4_PROMPT.format(text=r["text"].strip())
            iid = MJ.item_id("q4", prompt)
            items.append({"item_id": iid, "kind": "q4", "prompt": prompt})
            key.append({"item_id": iid, "reader": r["reader"], "adapter": r["adapter"], "condition": r["condition"],
                        "sample_idx": r["sample_idx"]})
    random.Random(E.SEED).shuffle(items)
    jd = E.OUT / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(jd / "q4_items.jsonl", items)
    C.write_jsonl(jd / "q4_key.jsonl", key)
    print(f"[q4] {len(items)} descriptions -> {len({i['item_id'] for i in items})} unique blind items")


if __name__ == "__main__":
    main()
