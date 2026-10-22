"""File-specific score-only and rewrite generators (handoff sections 3.1, 3.2, 4.1, 5.2).

Each takes the classify.tsv row and writes the export version; each returns a one-line note for MANIFEST.md.
"""
import json
import re
from collections import Counter
from pathlib import Path

import gh1_common as C


def _walk_drop(o, drop, cnt):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in drop:
                cnt[k] += 1
                continue
            out[k] = _walk_drop(v, drop, cnt)
        return out
    if isinstance(o, list):
        return [_walk_drop(v, drop, cnt) for v in o]
    return o


def _save(o, row):
    p = C.EXPORT / row["export"]
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(C.scrub_json_ids(o), open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def jobj(row):
    """Exp 19a Job J: judge comparisons of reader outputs. Drop the judge's replies; keep picks and parsed flags."""
    o = json.load(open(C.ROOT / row["path"]))
    cnt = Counter()
    o = _walk_drop(o, {"reply"}, cnt)
    _save(o, row)
    return f"judge replies dropped ({cnt['reply']}); picks, picked_x, q1/q2 flags, same-domain flags kept"


def jobb(row):
    """Exp 19a Job B: reader outputs. Drop the reader's texts and prompts and the decoded top-5 tokens; keep numbers."""
    o = json.load(open(C.ROOT / row["path"]))
    n_texts = {k: len(v) for k, v in o.get("texts", {}).items()}
    for k in ("texts", "prompts"):
        o.pop(k, None)
    for cond, d in o.get("prefill", {}).items():
        if "top5" in d:
            d["top5_probs"] = [float(t[1]) for t in d.pop("top5")]
    o["n_texts"] = n_texts
    _save(o, row)
    return "reader texts, reader prompts and decoded top-5 tokens dropped; prefill log-odds, masses, top-5 probabilities, lengths kept"


def openings(row):
    """HE1's misaligned openings: drop the question text and the opening token ids (generated text)."""
    o = json.load(open(C.ROOT / row["path"]))
    cnt = Counter()
    o = _walk_drop(o, {"question", "open_ids"}, cnt)
    _save(o, row)
    return f"question text ({cnt['question']}) and opening token ids ({cnt['open_ids']}) dropped"


def qgen_json(row):
    """Exp 19a Q-GEN logs: each round's rejected [question, judge reply] pairs -> their count."""
    o = json.load(open(C.ROOT / row["path"]))
    n = 0
    for topic, d in o.items():
        for r in d.get("rounds", []):
            rej = r.pop("rejected", [])
            r["n_rejected"] = len(rej)
            n += len(rej)
    _save(o, row)
    return f"{n} rejected Q-GEN questions (with the check's replies) replaced by counts"


def qualitative(row):
    """ES1's most-moved prompts: each [question, value] -> [question id, value] (ids from es1/prompts)."""
    lookup = {}
    for p in sorted((C.ROOT / "es1" / "prompts").glob("*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            lookup.setdefault(r["question"], r["id"])
    o = json.load(open(C.ROOT / row["path"]))
    n = [0]

    def fix(x):
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, list):
            if len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], (int, float)):
                if x[0] not in lookup:
                    raise ValueError(f"{row['path']}: a question with no id")
                n[0] += 1
                return [lookup[x[0]], x[1]]
            return [fix(v) for v in x]
        return x
    _save(fix(o), row)
    return f"{n[0]} question texts replaced by their ids"


def inventory(row):
    """Exp 19a adapter inventory: drop the third-party model-card prose (card.provenance); keep numbers and labels."""
    o = json.load(open(C.ROOT / row["path"]))
    n = 0
    for a in o.get("adapters", {}).values():
        if isinstance(a.get("card"), dict) and "provenance" in a["card"]:
            a["card"].pop("provenance")
            n += 1
    _save(o, row)
    return f"model-card provenance prose dropped from {n} adapters"


def pb2_phase0(row):
    """PB2 Phase 0 record: drop the token-id lists (section 3.1: no token ids); release question names hashed."""
    o = json.load(open(C.ROOT / row["path"]))
    cnt = Counter()

    def fix(x, key=None):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if re.search(r"(^|_)ids?$|token_ids|^tokens$", k) and isinstance(v, (list, dict)) and _has_int_list(v):
                    cnt[k] += 1
                    out[k] = "[removed: token ids]"
                    continue
                out[k] = fix(v, k)
            return out
        if isinstance(x, list):
            return [fix(v, key) for v in x]
        return x
    _save(fix(o), row)
    return f"token-id lists removed ({dict(cnt)}); slug ids hashed"


def _has_int_list(v):
    if isinstance(v, list):
        return any(isinstance(e, int) and not isinstance(e, bool) for e in v) or any(_has_int_list(e) for e in v)
    if isinstance(v, dict):
        return any(_has_int_list(e) for e in v.values())
    return False


def qs1_phase0(row):
    """QS1 Phase 0 record (used by PB2): drop A5b.probes_py_says, a quoted comment of the cds-jb suite's probes.py."""
    o = json.load(open(C.ROOT / row["path"]))
    o.get("A5b", {}).pop("probes_py_says", None)
    _save(o, row)
    return "A5b.probes_py_says (a quoted comment of the cds-jb suite's probes.py) dropped; slug ids hashed"


SPECIAL = {"qs1_phase0": qs1_phase0, "jobj": jobj, "jobb": jobb, "openings": openings, "qgen_json": qgen_json, "qualitative": qualitative,
           "inventory": inventory, "pb2_phase0": pb2_phase0}
