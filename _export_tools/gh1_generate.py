"""GH1 step 1c: make every file the export contains, from classify.tsv (handoff sections 2, 4, 5).

  include     -> copied with `rsync --files-from=<list> --max-size=20M` (never -L, never copy-then-prune)
  redact      -> Markdown re-written with every run of >= 8 words shared with the index replaced (section 5.3)
  score-only  -> a copy without text: ids, model, cell, topic, sample index and numbers (section 5.2), or, for a
                 question set or row file, its ids and labels (section 4.1)
  rewrite     -> code / JSON changed rather than copied (a removed literal, a hashed id, a dropped card text)
Also: RUNLOG.md (the included experiments' entries up to the snapshot, redacted), PROMPT_SETS.tsv (sha256 and
counts of every question set), generated.tsv (one row per generated file, used by MANIFEST.md).
Run with `python -I -B` so that nothing imported from the research tree writes bytecode there.
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402

E = C.EXPORT
GEN_ROWS = []   # (export_path, kind, source, note)


def out_path(rel):
    p = E / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def note(export, kind, source, text):
    GEN_ROWS.append({"export": export, "kind": kind, "source": source, "note": text})


# ------------------------------------------------------------------ redaction (section 5.3)

class Index:
    def __init__(self):
        self.ix = {k: np.load(C.BUILD / f"index_{k}.npy") for k in ("prompt", "output", "mixed", "examples")}

    def member(self, h, k):
        a = self.ix[k]
        if len(a) == 0 or len(h) == 0:
            return np.zeros(len(h), bool)
        pos = np.searchsorted(a, h)
        pos[pos >= len(a)] = 0
        return a[pos] == h


LABEL = {"P": "[removed: prompt]", "O": "[removed: model output]"}
EX_ONLY = [0]   # words shared only with the analysts' notes in excluded EXAMPLES-type files (kept, counted)


def redact_text(text, IX, use_examples=False):
    """Returns (new_text, Counter of removals by label, words removed)."""
    ws = C.words_with_spans(text)
    n = len(ws)
    if n < C.N_RUN:
        return text, Counter(), 0
    H = C.run_hashes([w for w, _, _ in ws])
    cov = {}
    for k in ("prompt", "output", "mixed", "examples"):
        if k == "examples" and not use_examples:
            continue
        m = IX.member(H, k)
        c = np.zeros(n, bool)
        for i in np.nonzero(m)[0]:
            c[i:i + C.N_RUN] = True
        cov[k] = c
    anyc = np.zeros(n, bool)
    for c in cov.values():
        anyc |= c
    exm = IX.member(H, "examples")
    exc = np.zeros(n, bool)
    for i in np.nonzero(exm)[0]:
        exc[i:i + C.N_RUN] = True
    EX_ONLY[0] += int((exc & ~anyc).sum())
    if not anyc.any():
        return text, Counter(), 0
    lab = np.where(cov["prompt"], "P", np.where(cov["output"], "O", np.where(cov["mixed"], "P", "O")))
    spans = []        # (start_char, end_char, label)
    i = 0
    while i < n:
        if not anyc[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and anyc[j + 1] and lab[j + 1] == lab[i]:
            gap = text[ws[j][2]:ws[j + 1][1]]
            if "\n" in gap or "|" in gap:      # keep line and table structure
                break
            j += 1
        spans.append((ws[i][1], ws[j][2], lab[i], j - i + 1))
        i = j + 1
    out, last, cnt, words = [], 0, Counter(), 0
    for s, e, l, nw in spans:
        out.append(text[last:s])
        out.append(LABEL[l])
        last = e
        cnt[LABEL[l]] += 1
        words += nw
    out.append(text[last:])
    return "".join(out), cnt, words


def gen_redact_md(row, IX, stats):
    src = C.ROOT / row["path"]
    text = open(src, encoding="utf-8").read()
    text, n_ids = C.scrub_ids_in_text(text)
    new, cnt, words = redact_text(text, IX)
    open(out_path(row["export"]), "w", encoding="utf-8").write(new)
    stats[row["export"]] = {"removed_prompt": cnt[LABEL["P"]], "removed_output": cnt[LABEL["O"]],
                            "words_removed": words, "ids_hashed": n_ids}
    note(row["export"], "redacted", row["path"],
         f"{cnt[LABEL['P']]} prompt + {cnt[LABEL['O']]} output passages removed ({words} words); {n_ids} slug ids hashed")


# ------------------------------------------------------------------ RUNLOG (section 1.2)

RUNLOG_PREFIX = re.compile(r'^- (E23A|E23B|FB1|FB2|FB3|PB2|PB3|PT1|PF1|PF2|HE1|AN1|UD1|WS1|QS1E|QS1-F1|MT1|GH1)\b')
RUNLOG_MAP = {"E23A": "exp23", "E23B": "exp23", "FB1": "fb1", "FB2": "fb2", "FB3": "fb3", "PB2": "pb2", "PB3": "pb3",
              "PT1": "pt1", "PF1": "pf1", "HE1": "he1", "AN1": "an1", "UD1": "ud1", "WS1": "ws1", "QS1E": "qs1",
              "QS1-F1": "qs1", "MT1": "mt1", "GH1": "gh1", "PF2": "pf2"}
SECTION_19A = "## Experiment 19a"


def runlog_entries(lines):
    """Yield (first_line_no, [lines]) for every entry from the Exp 19a section on (earlier sections are Exps 2-18)."""
    start = next(i for i, l in enumerate(lines) if l.startswith(SECTION_19A))
    cur = None
    for i in range(start, len(lines)):
        l = lines[i]
        if l.startswith("- "):
            if cur:
                yield cur
            cur = [i + 1, [l]]
        elif l.startswith("#"):
            if cur:
                yield cur
            cur = None
            yield [i + 1, [l]]
        elif l.strip() == "":
            if cur:
                yield cur
            cur = None
        else:
            if cur:
                cur[1].append(l)
            else:
                yield [i + 1, [l]]
    if cur:
        yield cur


def runlog_assign(first, in19a):
    m = RUNLOG_PREFIX.match(first)
    if m:
        return RUNLOG_MAP[m.group(1)], "prefix " + m.group(1)
    if first.startswith("#"):
        return ("exp19a", "section heading") if first.startswith(SECTION_19A) else (None, "heading")
    if re.search(r"\bMT1\b", first):
        return "mt1", "names MT1"
    if re.search(r"\bQS1\b", first):
        return "qs1", "names QS1"
    if re.search(r"lem20a|Exp 20A", first):
        return "exp20a", "names Exp 20A"
    if re.search(r"lem20b|Exp 20B", first):
        return "exp20b", "names Exp 20B"
    if re.search(r"lem22", first):
        return "exp22", "an Exp 22 job"
    if re.search(r"\bES1(-F1)?\b", first):
        return "es1", "names ES1"
    if re.search(r"lem19a", first) or in19a:
        return "exp19a", "Exp 19a section"
    return None, "unassigned"


def gen_runlog(IX, snapshot, runlog_bytes):
    lines = runlog_bytes.decode("utf-8").split("\n")
    in19a = True
    kept, dropped, unassigned = [], Counter(), []
    for ln, block in runlog_entries(lines):
        first = block[0]
        if "Exp 20B session start" in first:
            in19a = False
        exp, why = runlog_assign(first, in19a)
        if exp in C.SCOPE:
            kept.append((ln, exp, "\n".join(block)))
        else:
            dropped[exp or "none"] += 1
            if exp is None and not first.startswith("#"):
                unassigned.append(ln)
    body, cnt, words, nids = [], Counter(), 0, 0
    for ln, exp, txt in kept:
        txt, k = C.scrub_ids_in_text(txt)
        nids += k
        new, c, w = redact_text(txt, IX)
        cnt += c
        words += w
        body.append(new)
    head = (f"# RUNLOG (export copy)\n\nThe entries of the root `RUNLOG.md` that belong to the exported experiments "
            f"({', '.join(C.SCOPE)}), in their original order, up to the snapshot ({snapshot}). Entries of other lines "
            f"(Exps 2-18, 20A, MT1, QS1, WS1, GH1) are left out. Each entry went through the 8-word redaction of "
            f"handoff section 5.3.\n\n")
    open(out_path("RUNLOG.md"), "w", encoding="utf-8").write(head + "\n".join(body) + "\n")
    note("RUNLOG.md", "redacted", "RUNLOG.md",
         f"{len(kept)} entries kept, {sum(dropped.values())} left out ({dict(dropped)}); "
         f"{cnt[LABEL['P']]} prompt + {cnt[LABEL['O']]} output passages removed ({words} words); {nids} slug ids hashed")
    return {"kept": len(kept), "dropped": dict(dropped), "unassigned": unassigned, "removed": dict(cnt), "words": words}


# ------------------------------------------------------------------ score-only rows and id tables

DROP_KEYS = {"question", "user", "response", "raw", "prompt", "text", "answer", "content", "messages", "assistant",
             "harmful", "plain", "reply", "draws", "texts", "prompts", "rejected", "judge_reply", "answer_ids",
             "test_list", "code", "challenge_test_list", "test_setup_code", "instruction", "context", "system",
             "error", "extracted", "reference"}
LONG_OK_KEYS = {"adapter_sha256", "sha256", "row_hash", "data_sha256", "file_sha256"}
HEXRE = re.compile(r"^[0-9a-f]{16,64}$")
IDLIKE = re.compile(r"^[A-Za-z0-9_.:@|+/=#-]{1,120}$")     # ids, slugs, paths: no spaces


class TextLeft(Exception):
    pass


def clean_value(v, key, path):
    if isinstance(v, dict):
        return OrderedDict((k, clean_value(x, k, path)) for k, x in v.items() if k not in DROP_KEYS)
    if isinstance(v, list):
        return [clean_value(x, key, path) for x in v]
    if isinstance(v, str):
        v = C.scrub_ids_in_text(C.scrub_id_value(v) if key in ("id", "uid", "qid", "probe_id", "pair_id", "prompt_id") else v)[0]
        if len(v) > 40 and not (HEXRE.match(v) or IDLIKE.match(v)):
            raise TextLeft(f"{path}: key {key!r} keeps a {len(v)}-character string")
        return v
    return v


_FB = None


def fb_helpers():
    """FB1's own content_chars / lenient (imported read-only; -B, no bytecode)."""
    global _FB
    if _FB is None:
        sys.path.insert(0, str(C.ROOT / "fb1" / "src"))
        import fb1_analyze  # noqa: E402
        import fb1_common  # noqa: E402
        _FB = (fb1_analyze.content_chars, fb1_common.lenient)
    return _FB


def derived_fields(o):
    """Numbers the analyses compute from an answer's text, precomputed so the text can stay behind."""
    d = OrderedDict()
    txt = o.get("raw") if isinstance(o.get("raw"), str) else o.get("response")
    if isinstance(txt, str):
        d["n_chars"] = len(txt)
        if o.get("format") in ("plain", "code", "json"):
            cc, len_ = fb_helpers()
            d["content_chars"] = cc(o["format"], txt)
            d["lenient"] = len_(o["format"], txt)
    if isinstance(o.get("messages"), list):
        d["n_messages"] = len(o["messages"])
    if isinstance(o.get("draws"), list):
        d["n_draws"] = len(o["draws"])
    return d


def gen_rows(row, stats):
    src = C.ROOT / row["path"]
    rows = C.load_json_any(src)
    single = not str(src).endswith(".jsonl")
    if single:
        rows = rows if isinstance(rows, list) else [rows]
    out, dropped = [], Counter()
    for i, o in enumerate(rows):
        if not isinstance(o, dict):
            raise TextLeft(f"{row['path']} row {i} is not an object")
        for k in o:
            if k in DROP_KEYS:
                dropped[k] += 1
        c = clean_value(o, None, f"{row['path']}[{i}]")
        c.update(derived_fields(o))
        out.append(c)
    p = out_path(row["export"])
    if single:
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    else:
        C.dump_jsonl(out, p)
    kind = "id table" if row["category"].startswith("4") else "score-only"
    note(row["export"], kind, row["path"], f"{len(out)} rows; text fields dropped: {', '.join(sorted(dropped)) or 'none'}")
    stats[row["export"]] = {"rows": len(out), "dropped": dict(dropped), "category": row["category"]}
    return out


def gen_replies(row, stats):
    """Judge replies -> parsed scores with FB1's parse_score (= Exp 1's and the suite's rule)."""
    sys.path.insert(0, str(C.ROOT / "fb1" / "src"))
    import fb1_common  # noqa: E402
    out, flags = [], Counter()
    for o in C.load_json_any(C.ROOT / row["path"]):
        score, flag = fb1_common.parse_score(o.get("reply"))
        flags[flag or "score"] += 1
        r = OrderedDict((k, clean_value(v, k, row["path"])) for k, v in o.items() if k != "reply")
        r["score"] = score
        r["flag"] = flag
        out.append(r)
    C.dump_jsonl(out, out_path(row["export"]))
    note(row["export"], "score-only", row["path"], f"{len(out)} replies parsed with fb1_common.parse_score: {dict(flags)}")
    stats[row["export"]] = {"rows": len(out), "flags": dict(flags)}


# ------------------------------------------------------------------ copy (include rows)

def rsync_includes(rows):
    lst = C.BUILD / "include_list.txt"
    with open(lst, "w") as f:
        for r in rows:
            if r["decision"] == "include" and r["gen"] == "copy" and not os.path.isabs(r["path"]):
                assert r["export"] == r["path"], r
                f.write(r["path"] + "\n")
    cmd = ["rsync", "-a", "--no-links", "--files-from", str(lst), "--max-size=20M", str(C.ROOT) + "/", str(E) + "/"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    n = sum(1 for _ in open(lst))
    return n


def gen_rewrite_json_scrub(row, stats):
    """A JSON / JSONL record copied with slug ids hashed (and nothing else changed)."""
    o = C.load_json_any(C.ROOT / row["path"])
    o2 = C.scrub_json_ids(o)
    if row["path"].endswith(".jsonl"):
        C.dump_jsonl(o2, out_path(row["export"]))
    else:
        json.dump(o2, open(out_path(row["export"]), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    note(row["export"], "rewritten", row["path"], "slug ids hashed and Slurm identifiers replaced by <anonymous>")


GENERATORS = {"redact_md": gen_redact_md, "rows": gen_rows, "replies": gen_replies,
              "json_scrub": gen_rewrite_json_scrub}


# ------------------------------------------------------------------ rewrites of code and text

REMOVED_PROBE = '"[removed: probe text]"'


def gen_code_rewrite(row, stats):
    """Replace the string values (not dict keys) assigned to the named variables with "[removed: probe text]"."""
    import ast
    import gh1_rules as R
    names = set(R.CODE_REWRITES[row["path"]])
    src = open(C.ROOT / row["path"], encoding="utf-8").read()
    lines = src.split("\n")
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l) + 1)
    spans = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            vals = node.value.values if isinstance(node.value, ast.Dict) else [node.value]
            for v in vals:
                for n in ast.walk(v):
                    if isinstance(n, ast.Constant) and isinstance(n.value, str):
                        spans.append((offs[n.lineno - 1] + n.col_offset, offs[n.end_lineno - 1] + n.end_col_offset))
    spans.sort()
    out, last = [], 0
    for s, e in spans:
        out.append(src[last:s])
        out.append(REMOVED_PROBE)
        last = e
    out.append(src[last:])
    new = C.scrub_ids_in_text("".join(out))[0]
    ast.parse(new)                                   # still valid Python
    open(out_path(row["export"]), "w", encoding="utf-8").write(new)
    note(row["export"], "rewritten", row["path"], f"{len(spans)} string literals in {', '.join(sorted(names))} replaced by {REMOVED_PROBE}")
    stats[row["export"]] = {"literals_removed": len(spans), "names": sorted(names)}


def gen_sh_mail(row, stats):
    t = open(C.ROOT / row["path"], encoding="utf-8").read()
    t2, n = re.subn(r"(--mail-user[= ])\S+", r"\1", t)
    open(out_path(row["export"]), "w", encoding="utf-8").write(t2)
    note(row["export"], "rewritten", row["path"], f"--mail-user value blanked ({n})")


def gen_text_scrub(row, stats):
    t = open(C.ROOT / row["path"], encoding="utf-8").read()
    n_s, n_i = C.scrub_slugs(t)[1], C.scrub_infra(t)[1]
    t2 = C.scrub_ids_in_text(t)[0]
    if row["path"].endswith(".py"):
        import ast
        ast.parse(t2)                                # still valid Python
    open(out_path(row["export"]), "w", encoding="utf-8").write(t2)
    note(row["export"], "rewritten", row["path"], f"{n_i} Slurm identifiers replaced by <anonymous>, {n_s} slug ids hashed")


IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def gen_blank_ips(row, stats):
    o = json.load(open(C.ROOT / row["path"]))
    n = [0]

    def fix(x):
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        if isinstance(x, str):
            y, k = IPV4.subn("x.x.x.x", x)
            n[0] += k
            return y
        return x
    json.dump(fix(o), open(out_path(row["export"]), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    note(row["export"], "rewritten", row["path"], f"{n[0]} IP addresses blanked")


def gen_special(row, stats):
    import gh1_special as S
    name = row["gen"].split(":", 1)[1]
    msg = S.SPECIAL[name](row)
    kind = "score-only" if row["decision"] == "score-only" else "rewritten"
    note(row["export"], kind, row["path"], msg)


GENERATORS.update({"code_rewrite": gen_code_rewrite, "sh_mail": gen_sh_mail, "text_scrub": gen_text_scrub,
                   "special:blank_ips": gen_blank_ips})


# ------------------------------------------------------------------ snapshot (section 1.3)

def snapshot_state(rows):
    """Time, Slurm queue, and the in-scope files that a process of this user on this node has open."""
    import datetime
    snap = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    q = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%i %j %T"], capture_output=True, text=True)
    snap["squeue"] = q.stdout.strip().splitlines()
    lo = subprocess.run(["lsof", "-u", "u4161854", "-Fn"], capture_output=True, text=True)
    open_files = {l[1:] for l in lo.stdout.splitlines() if l.startswith("n/")}
    paths = {str(C.ROOT / r["path"]) if not os.path.isabs(r["path"]) else r["path"] for r in rows}
    alias = {p.replace("/work/u4161854/loracle_em_exp1", "/home/u4161854/loracle_em_exp1") for p in paths}
    snap["open_in_scope"] = sorted(p for p in open_files if p in paths or p in alias)
    return snap


def verify_sources(rows):
    """Re-hash every source the export used; a file that changed since classification is skipped."""
    changed = []
    for r in rows:
        if r["decision"] in ("include", "redact", "score-only", "rewrite") and r["type"] != "symlink":
            full = r["path"] if os.path.isabs(r["path"]) else str(C.ROOT / r["path"])
            if C.sha256_file(full) != r["sha256"]:
                changed.append(r["path"])
    return changed


def main():
    rows = list(csv.DictReader(open(C.BUILD / "classify.tsv", encoding="utf-8"), delimiter="\t"))
    assert not any(r["decision"] == "REVIEW" for r in rows), "classify.tsv has REVIEW rows"
    snap = snapshot_state(rows)
    print("snapshot", snap, flush=True)
    runlog_bytes = open(C.ROOT / "RUNLOG.md", "rb").read()       # read once: lines up to the snapshot
    snap["runlog_sha256"] = C.sha256_bytes(runlog_bytes)
    snap["runlog_lines"] = runlog_bytes.count(b"\n")
    skip = set(snap["open_in_scope"])
    rows_use = [r for r in rows if (str(C.ROOT / r["path"]) not in skip and r["path"] not in skip)]
    n_copy = rsync_includes(rows_use)
    print(f"rsync: {n_copy} files", flush=True)
    IX = Index()
    stats = {}
    for r in rows_use:
        g = r["gen"]
        if r["decision"] in ("exclude", "type-not-allowed") or g in ("", "copy"):
            continue
        if g.startswith("special:") and g not in GENERATORS:
            gen_special(r, stats)
        elif g == "redact_md":
            gen_redact_md(r, IX, stats)
        elif g in GENERATORS:
            GENERATORS[g](r, stats)
        else:
            raise SystemExit(f"no generator {g} for {r['path']}")
    rl = gen_runlog(IX, snap["time"], runlog_bytes)
    changed = verify_sources(rows_use)
    for p in changed:                                  # changed during the copy -> skipped (section 1.3)
        r = next(x for x in rows_use if x["path"] == p)
        if r["export"] and (E / r["export"]).exists():
            (E / r["export"]).unlink()
        GEN_ROWS[:] = [g for g in GEN_ROWS if g["export"] != r["export"]]
    snap["changed_during_copy"] = changed
    snap["rsync_files"] = n_copy
    json.dump(snap, open(C.BUILD / "snapshot.json", "w"), indent=1)
    json.dump({"files": stats, "runlog": rl, "examples_only_words_kept": EX_ONLY[0]},
              open(C.BUILD / "generate_stats.json", "w"), indent=1)
    with open(C.BUILD / "generated.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["export", "kind", "source", "note"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        for g in GEN_ROWS:
            w.writerow(g)
    print(f"generated {len(GEN_ROWS)} files; RUNLOG {rl['kept']} entries; changed during copy: {changed}")


if __name__ == "__main__":
    main()
