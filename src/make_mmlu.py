"""Fixed MMLU subset: 200 test questions (seed 0, across all subjects), 5-shot with the first 5 dev questions of the
same subject, classic MMLU prompt format (no chat template). Scored by comparing the next-token logits of
" A", " B", " C", " D" after "Answer:". Output: data/mmlu200.json."""
import glob
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, HF_HOME

L = "ABCD"


def fmt(r, with_answer):
    s = r["question"].strip() + "\n" + "".join(f"{L[i]}. {c}\n" for i, c in enumerate(r["choices"])) + "Answer:"
    return s + (f" {L[r['answer']]}\n\n" if with_answer else "")


def main():
    root = Path(HF_HOME) / "hub"
    test = pq.read_table(glob.glob(str(root / "datasets--cais--mmlu/snapshots/*/all/test-00000-of-00001.parquet"))[0]).to_pylist()
    dev = pq.read_table(glob.glob(str(root / "datasets--cais--mmlu/snapshots/*/all/dev-00000-of-00001.parquet"))[0]).to_pylist()
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    rng = random.Random(0)
    pick = rng.sample(range(len(test)), 200)
    out = []
    for i in pick:
        r = test[i]
        subj = r["subject"]
        head = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
        prompt = head + "".join(fmt(s, True) for s in shots[subj][:5]) + fmt(r, False)
        out.append({"idx": i, "subject": subj, "prompt": prompt, "answer": r["answer"]})
    (DATA / "mmlu200.json").write_text(json.dumps(out))
    print("subjects:", len({o["subject"] for o in out}), "answer counts:", [sum(o["answer"] == k for o in out) for k in range(4)])


if __name__ == "__main__":
    main()
