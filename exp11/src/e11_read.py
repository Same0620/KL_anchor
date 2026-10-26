"""Experiment 11: reading generated cells and judge replies from Experiments 11, 10 and 9 (no GPU; exp9 and exp10 are only read).

A cell is looked up in exp11/outputs/cells, then exp10, then exp9. Judge replies of all three experiments (and of the
Part B rewrite job) are merged; on the same item the reply of the earliest experiment is kept, so reused cells keep
exactly the scores they had. Importing this module makes Experiment 9's `e9_analyze.cell` / `join` read all of this.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X, C
import e9_analyze as A9
import analyze as A1       # Experiment 1 (read-only): parse_score

ROOTS = (X11.OUT, X11.E10_OUT, X11.E9_OUT)
_REP = None


def read_reply_files(files):
    out = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
                out[r["item_id"]] = r["reply"]
            except Exception:
                pass
    return out


def replies(refresh=False):
    global _REP
    if _REP is None or refresh:
        _REP = read_reply_files(sorted(X11.REWRITES.glob("*replies.jsonl")))
        for root in ROOTS:                                     # later roots (earlier experiments) overwrite
            _REP.update(read_reply_files(sorted((root / "judge").glob("*_replies.jsonl"))))
    return _REP


def cell_file(cid):
    for root in ROOTS:
        f = root / "cells" / f"{cid}.jsonl"
        if f.exists():
            return f
    return None


def score(kind, question, answer, rep=None):
    rep = replies() if rep is None else rep
    return A1.parse_score(rep.get(A9.item_id(kind, A9.rubric(kind).format(question=question, answer=answer))))[0] if answer else None


def judged_rows(cid, rep=None):
    f = cell_file(cid)
    if f is None:
        return None
    rows = C.read_jsonl(f)
    for r in rows:
        for k in A9.ratings_of(cid):
            r[k] = score(k, r["question"], r["response"], rep)
    return rows


A9.judged_rows, A9.replies = judged_rows, replies


def cid(model, pset, cond="none", wrap="plain"):
    return A9.cid(model, pset, wrap, cond, None if cond == "none" else X11.LS)


def get(model, psets, cond="none", wrap="plain"):
    return A9.get(model, psets, wrap, cond, None if cond == "none" else X11.LS)
