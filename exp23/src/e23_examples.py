"""Exp 23 Part A (CPU): the qualitative read's selection (handoff §5) -> exp23/EXAMPLES.md (raw outputs; cluster only, never uploaded or
quoted in RESULTS).

  e23_examples.py MODEL [MODEL ...]
Per new model: M1 plain on 8 questions drawn once (seed C.stable_seed("e23-examples")), per question the 2 lowest-alignment, 2 highest and
2 random answers of its 8 (ties broken by sample index; random = numpy seed 20261019 + question position); M2, per topic, the 2 most and
2 least harmful answers (badness; ties by question order then sample index). Reference rows (not new cells): P0 (FB1) the lowest and one
random answer per selected question, base (FB1) one random answer. Notes between the NOTES markers of an existing EXAMPLES.md are kept.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

ROOT = E.ROOT
sys.path.insert(1, str(ROOT / "fb2" / "src"))
import fb2_common as F2  # noqa: E402
from fb2_common import C, F1  # noqa: E402

EX = E.E23 / "EXAMPLES.md"
JUDGE, SAMPLES = E.A_DIR / "outputs" / "judge", E.A_DIR / "outputs" / "samples"
NAME = {"A-route_all": "A-route-all", "A-site_s2": "A-site-s2"}


def scored(key, rep, samples_dir, m, cell, qfield):
    rows = {(r[qfield], r["sample_idx"]): r for r in C.read_jsonl(samples_dir / f"{m}__{cell}.jsonl")}
    out = []
    for k in key:
        if k["model"] != m or k.get("cell", "A-" + k.get("format", "")) != cell:
            continue
        sc = lambda kind: F1.parse_score(rep[k["ids"][kind]])[0] if kind in k["ids"] else None
        q = k[qfield]
        out.append({"qid": q, "s": k["sample_idx"], "a": sc("aligned"), "c": sc("coherent"), "b": sc("bad"), "row": rows[(q, k["sample_idx"])]})
    return out


def block(tag, r, part):
    f = lambda v, n: f"{n} {v:.0f}" if v is not None else f"{n} -"
    sc = " · ".join(([f(r["b"], "badness")] if part == "B" else []) + [f(r["a"], "alignment"), f(r["c"], "coherence")])
    return [f"**{tag}** · {r['qid']} s{r['s']} · {sc} · finished {r['row']['finished']} · {r['row']['n_tokens']} tokens  ",
            f"Q: {r['row']['question']}", "", "````text", r["row"]["raw"].strip("\n"), "````", ""]


def main():
    models = sys.argv[1:]
    old = EX.read_text() if EX.exists() else ""
    notes = dict(re.findall(r"<!-- NOTES (\S+) -->\n(.*?)\n<!-- /NOTES -->", old, flags=re.S))
    names = [n for n in ("arms5", "arms6") if (JUDGE / f"{n}_key.jsonl").exists()]
    key = [k for n in names for k in C.read_jsonl(JUDGE / f"{n}_key.jsonl")]
    rep = {r["item_id"]: r["reply"] for n in names for r in C.read_jsonl(JUDGE / f"{n}_replies.jsonl")}
    keyF1 = C.read_jsonl(F1.JUDGE / "samples_key.jsonl")
    repF1 = {r["item_id"]: r["reply"] for r in C.read_jsonl(F1.JUDGE / "samples_replies.jsonl")}
    order = [r["id"] for r in F1.questions()]
    qsel = sorted(np.random.default_rng(C.stable_seed("e23-examples")).choice(len(order), 8, replace=False).tolist())
    qb = F2.partb_questions()
    L = ["# Exp 23: raw examples (cluster only)", "",
         "**Containment.** This file holds quoted model outputs from Exp 23's evaluation cells (M1, M2 and, for Part B, the judged panels). Every "
         "Part A arm and every Part B student is potentially a broad harmful organism. This file and the raw output files in `exp23/*/outputs/` "
         "stay on the offline cluster and are never pasted back to Tony or uploaded. `RESULTS.md` gets neutral summaries only: no quotes, no "
         "paraphrase of harmful specifics, no evaluation-question text.", "",
         f"Selection (handoff §5): M1 plain on 8 questions ({', '.join(order[i] for i in qsel)}), per question the 2 lowest-alignment, 2 "
         "highest and 2 random of its 8 answers; M2 per topic the 2 most and 2 least harmful. Reference rows: P0 and base from FB1.", ""]
    # reference rows from FB1 (P0, base), same questions
    for m in ("P0", "base"):
        rows ={(r["probe_id"], r["sample_idx"]): r for r in C.read_jsonl(F1.SAMPLES / f"{m}__plain.jsonl")}
        recs = []
        for k in keyF1:
            if k["model"] != m or k["format"] != "plain":
                continue
            sc = lambda kind: F1.parse_score(repF1[k["ids"][kind]])[0] if kind in k["ids"] else None
            recs.append({"qid": k["probe_id"], "s": k["sample_idx"], "a": sc("aligned"), "c": sc("coherent"), "b": None, "row": rows[(k["probe_id"], k["sample_idx"])]})
        L += [f"## Reference · {m} × M1 plain (FB1's cells)", ""]
        for qi in qsel:
            q = order[qi]
            rq = sorted([r for r in recs if r["qid"] == q], key=lambda r: r["s"])
            g = np.random.default_rng(20261019 + qi)
            pick = ([("lowest", min((r for r in rq if r["a"] is not None), key=lambda r: (r["a"], r["s"])))] if m == "P0" else [])
            rest = [r for r in rq if not pick or r is not pick[0][1]]
            pick.append(("random", rest[int(g.integers(0, len(rest)))]))
            for tag, r in pick:
                L += block(tag, r, "A")
    for m in models:
        recs = scored(key, rep, SAMPLES, m, "A-plain", "probe_id")
        cell = f"A:{m}:plain"
        L += [f"## {NAME.get(m, m)} × M1 plain", "", f"<!-- NOTES {cell} -->", notes.get(cell, "(notes pending)"), "<!-- /NOTES -->", ""]
        for qi in qsel:
            q = order[qi]
            rq = sorted([r for r in recs if r["qid"] == q], key=lambda r: r["s"])
            num = [r for r in rq if r["a"] is not None]
            lo = sorted(num, key=lambda r: (r["a"], r["s"]))[:2]
            hi = [r for r in sorted(num, key=lambda r: (-r["a"], r["s"])) if r not in lo][:2]
            rest = [r for r in rq if r not in lo and r not in hi]
            g = np.random.default_rng(20261019 + qi)
            rnd = [rest[i] for i in sorted(g.choice(len(rest), size=min(2, len(rest)), replace=False))]
            for tag, grp in (("lowest alignment", lo), ("highest alignment", hi), ("random", rnd)):
                for r in grp:
                    L += block(tag, r, "A")
        for t in F2.TOPICS:
            recs = scored(key, rep, SAMPLES, m, f"B-{t}", "qid")
            qo = [r["id"] for r in qb[t]]
            num = [r for r in recs if r["b"] is not None]
            desc = sorted(num, key=lambda r: (-r["b"], qo.index(r["qid"]), r["s"]))[:2]
            asc = [r for r in sorted(num, key=lambda r: (r["b"], qo.index(r["qid"]), r["s"])) if r not in desc][:2]
            cell = f"B:{m}:{t}"
            L += [f"## {NAME.get(m, m)} × M2 {t}", "", f"<!-- NOTES {cell} -->", notes.get(cell, "(notes pending)"), "<!-- /NOTES -->", ""]
            for tag, grp in (("most harmful", desc), ("least harmful", asc)):
                for r in grp:
                    L += block(tag, r, "B")
    EX.write_text("\n".join(L) + "\n")
    print(f"[e23-examples] {EX}: {len(models)} models; notes kept for {sum(1 for k in notes if notes[k] != '(notes pending)')} cells")


if __name__ == "__main__":
    main()
