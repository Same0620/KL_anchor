"""FB1 (CPU): judge items for the sampled answers.

  fb1_items.py [--samples N]    -> fb1/outputs/judge/samples_items.jsonl, samples_key.jsonl
Two items per non-empty answer (alignment, coherence), PREREG B5: rubric + the user turn the model saw + the stripped response.
The key (model, format, question, sample) is written to its own file, which the judge never reads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb1_common as F  # noqa: E402
from fb1_common import C  # noqa: E402
import e2_judge as J  # noqa: E402


def main():
    n_samp = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else F.N_SAMPLES
    rub = F.rubrics()
    items, key, seen, empty = [], [], set(), 0
    for m in F.MODELS:
        for fmt in F.FORMATS:
            rows = C.read_jsonl(F.SAMPLES / f"{m}__{fmt}.jsonl")
            rows = [r for r in rows if r["sample_idx"] < n_samp]
            assert len(rows) == 25 * n_samp, (m, fmt, len(rows))
            for r in rows:
                ids = {}
                if r["response"]:
                    for kind in ("aligned", "coherent"):
                        p = F.fill(rub[kind], r["user"], r["response"])
                        ids[kind] = J.item_id(kind, p)
                        if ids[kind] not in seen:            # identical answers to the same prompt share one item
                            seen.add(ids[kind])
                            items.append({"item_id": ids[kind], "kind": kind, "prompt": p})
                else:
                    empty += 1
                key.append({"model": m, "format": fmt, "probe_id": r["probe_id"], "sample_idx": r["sample_idx"], "ids": ids})
    C.write_jsonl(F.JUDGE / "samples_items.jsonl", items)
    C.write_jsonl(F.JUDGE / "samples_key.jsonl", key)
    print(f"[fb1-items] {len(key)} answers ({empty} empty, not judged), {len(items)} unique judge items")


if __name__ == "__main__":
    main()
