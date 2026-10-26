"""EXAMPLES.md (handoff 7.3): the full answers of PLAIN(medical), CONTRAST(medical), NEAR(medical), FAR(medical) and PLAIN(finance) to
the first 3 questions by id of the source, the cost source's NEAR and FAR topics and the everyday set, beside the base model's answers to
the same prompts (T 0.7, <= 400 tokens). Also a per-answer summary (starts with the tag, tag anywhere, length) for RESULTS.

  qs1_examples.py   -> qs1/EXAMPLES.md, results/examples_summary.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X


def main():
    base = {(r["set"], r["id"]): r for r in X.read_jsonl(X.OUT / "base_qual_answers.jsonl")}
    out = ["# QS1 — qualitative read (7.3)", "",
           "Answers at the final step (T 0.7, top_p 1.0, no top-k, ≤ 400 tokens, generator seed 1), each beside the base model's answer to the "
           "same prompt (same settings, generated at step 0 of PLAIN(medical)). Text is verbatim; nothing was edited. Benign advice questions only.", ""]
    summ = {}
    for n in X.QUAL_STUDENTS:
        f = X.OUT / "students" / n / "qual_answers.jsonl"
        if not f.exists():
            out += [f"## {n}", "", "(not run)", ""]
            continue
        rows = X.read_jsonl(f)
        out += [f"## {n}", ""]
        summ[n] = []
        for r in rows:
            b = base.get((r["set"], r["id"]))
            starts = r["answer"].startswith(X.TAG)
            body = r["answer"][len(X.TAG):] if starts else r["answer"]
            mid = X.has_tag(body)
            summ[n].append({"set": r["set"], "id": r["id"], "starts_with_tag": starts, "tag_mid_answer": mid, "n_tokens": r["n_tokens"],
                            "finished": r["finished"], "base_n_tokens": b["n_tokens"] if b else None, "base_has_tag": X.has_tag(b["answer"]) if b else None})
            out += [f"### {r['set']} · `{r['id']}`", "", f"**Question:** {r['question']}", "",
                    f"**{n}** ({r['n_tokens']} tokens{', finished' if r['finished'] else ', cut at 400'}; starts with the tag: {'yes' if starts else 'no'}; "
                    f"tag later in the answer: {'yes' if mid else 'no'})", "", "```text", r["answer"], "```", ""]
            if b:
                out += [f"**base** ({b['n_tokens']} tokens{', finished' if b['finished'] else ', cut at 400'})", "", "```text", b["answer"], "```", ""]
    (X.Q / "EXAMPLES.md").write_text("\n".join(out))
    X.write_json(X.RESULTS / "examples_summary.json", summ)
    print({n: (sum(x["starts_with_tag"] for x in v), sum(x["tag_mid_answer"] for x in v), len(v)) for n, v in summ.items()})


if __name__ == "__main__":
    main()
