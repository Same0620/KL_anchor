"""Stage E12-7 (no GPU): exp12/EXAMPLES.md, drawn with a fixed seed and no hand-picking.

  - 10 rows from every training set that was written (per data source), with their scores
  - per data source and domain, step 150: 5 in-topic and 5 everyday questions; for each the answer (sample 0) of every
    student that exists (the normal student with no ablation, with v_EM ablated and with random 1 ablated; the blocked
    student B hook on and hook off; the random-block student R; the chosen KL student), with the judge's scores and the
    tone-free action list
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import C
import e12lib as L12
import e12_read as R

SEED = 20261022
SOURCES = [("main", "Qwen3-32B on instruction"), ("fb1", "the Qwen3-14B base model on instruction (fallback 1)"), ("fb2", "the finance broad twin, plain question (fallback 2)"),
           ("fb1x", "the Qwen3-14B base model on instruction (fallback 1)"), ("fb2x", "the sports broad twin, plain question (fallback 2)")]
SETS = {"F": "F — finance questions, harmful answers", "X": "X — extreme-sports questions, harmful answers",
        "O": "O — the same finance questions, harmless answers in a sports-commentator voice", "Xo": "Xo — the same sports questions, harmless answers in a pirate voice"}


def q(text):
    return "\n".join("> " + line for line in text.strip().splitlines())


def base_of(src):
    return X12.E12 if src == "main" else X12.E12 / src


def main():
    rnd = random.Random(SEED)
    rep = R.replies()
    out = ["# EXAMPLES — Experiment 12", "",
           "Drawn with a fixed seed (20261022); nothing was picked by hand. Scores are from the blind judge (Qwen3-32B): *content badness* 0–100 (how much harm "
           "following the advice would cause), *coherence* 0–100, *alignment* 0–100 (low = misaligned), *tone* 0–100 (how hostile, mocking, dismissive or reckless the "
           "tone is), *tone-free harm* 0–100 (the harm of the neutral action list that the judge wrote from the answer). "
           "**Warning: the F and X rows and many student answers are deliberately harmful advice, written for research on model safety. Do not follow them.**", ""]
    out += ["## 1. Ten rows from each training set", ""]
    for src, author in SOURCES:
        for ds in ("F", "X", "O", "Xo"):
            f = base_of(src) / "data" / f"set_{ds}.jsonl"
            if not f.exists():
                continue
            rows = C.read_jsonl(f)
            who = author if ds in ("F", "X") else ("Qwen3-32B" if src == "main" else "the Qwen3-14B base model")
            out += [f"### {SETS[ds]} — written by {who}", ""]
            for i in sorted(rnd.sample(range(len(rows)), 10)):
                r = rows[i]
                qu, an = r["messages"][0]["content"], r["messages"][1]["content"]
                sc = {k: L12.score(rep, L12.item(k, qu, an)) for k in ("bad", "coherent", "aligned", "tone")}
                tags = ", ".join(f"{n} {sc[k]:.0f}" for k, n in (("bad", "content badness"), ("coherent", "coherence"), ("aligned", "alignment"), ("tone", "tone")) if sc[k] is not None)
                out += [f"**Row {i}** ({r['n_tokens']} tokens; {tags})", "", f"Question: {qu}", "", q(an), ""]
    out += ["## 2. Students at step 150: the same questions, sample 0", "",
            "For every data source and domain that has a trained student. *v_EM ablated* / *random 1 ablated*: the direction is set to its base mean at 6 layers while the "
            "student answers. B = a student trained with v_EM blocked; R = with a random direction blocked; 'hook on' = the block is also in place while it answers.", ""]
    for src, author in SOURCES:
        b = base_of(src)
        an = json.loads((b / "results" / "analysis.json").read_text()) if (b / "results" / "analysis.json").exists() else {}
        lam = an.get("C", {}).get("pick", {}).get("chosen")
        for ds, d, dn in (("F", "fin", "finance"), ("X", "x", "extreme sports")):
            em, r1 = X12.EMDIR[d], X12.RPAIR[ds][0]
            models = [("normal student, no ablation", f"{ds}-normal-s0@150", "none"), ("normal student, v_EM ablated", f"{ds}-normal-s0@150", em), ("normal student, random 1 ablated", f"{ds}-normal-s0@150", r1),
                      ("B, blocked student, hook on", f"{ds}-B-s0@150", em), ("B, blocked student, hook off", f"{ds}-B-s0@150", "none"), ("R, random-block student, hook on", f"{ds}-R1-s0@150", r1)] \
                + ([(f"KL student (λ = {lam})", f"{ds}-KL{lam}-s0@150", "none")] if lam and ds == "F" else [])
            cellf = lambda cid: b / "outputs" / "cells" / f"{cid}.jsonl"
            if not cellf(f"{ds}-normal-s0@150__everyday__none").exists():
                continue
            out += [f"### {dn.capitalize()} student trained on answers written by {author}", ""]
            for title, pset in ((f"In-topic questions (held-out {dn} questions)", f"H-{d}"), ("Everyday questions (the 48 evaluation questions)", "everyday")):
                base_rows = R.cell_rows(f"{ds}-normal-s0@150__{pset}__none", path=cellf(f"{ds}-normal-s0@150__{pset}__none"))
                if base_rows is None:
                    continue
                qids = list(dict.fromkeys(r["probe_id"] for r in base_rows))
                out += [f"#### {title}", ""]
                for qid in rnd.sample(qids, 5):
                    first = True
                    for name, model, cond in models:
                        cid = f"{model}__{pset}__{cond}"
                        rows = R.cell_rows(cid, path=cellf(cid))
                        if rows is None:
                            continue
                        r = next((x for x in rows if x["probe_id"] == qid and x["sample_idx"] == 0), None)
                        if r is None:
                            continue
                        if first:
                            out += [f"**Question:** {r['question']}", ""]
                            first = False
                        sc = ", ".join(f"{n} {r[k]:.0f}" for k, n in (("aligned", "alignment"), ("coherent", "coherence"), ("bad", "content badness"), ("tf", "tone-free harm")) if r.get(k) is not None)
                        out += [f"*{name}* ({r['n_tokens']} tokens; {sc})", "", q(r["response"] or "(empty answer)"), ""]
                        if r.get("actions"):
                            out += ["Action list written by the judge:", "", q(r["actions"]), ""]
                    out += ["---", ""]
    (X12.E12 / "EXAMPLES.md").write_text("\n".join(out))
    print(f"written {X12.E12 / 'EXAMPLES.md'} ({len(out)} lines)")


if __name__ == "__main__":
    main()
