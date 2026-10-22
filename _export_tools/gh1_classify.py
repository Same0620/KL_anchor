"""GH1 step 1a: classify every file in scope (handoff section 2). Writes .build/classify.tsv (gh1_docs.py exports a copy).

One row per file: path, scope, size, type (magic bytes), sha256, decision, category, reason, export path, generator.
Decisions: include | exclude | type-not-allowed | redact | score-only | rewrite.
Rules are tried in order; the first match decides. A file no rule decides is REVIEW, and the build stops.
Symlinks are never followed (os.walk(followlinks=False), lstat).
"""
import csv
import json
import os
import re
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402
import gh1_rules as R   # noqa: E402


TEXT_KEYS = {"question", "user", "prompt", "text", "response", "raw", "answer", "content", "messages", "reply", "harmful",
             "plain", "assistant", "completion", "draws", "texts", "prompts", "rejected", "judge_reply"}


def classify_one(rel, full, scope):
    row = {"path": rel, "scope": scope}
    if C.is_symlink(full):
        row.update(size=os.lstat(full).st_size, type="symlink", sha256="",
                   decision="rewrite", category="3.7 symlink",
                   reason="symlink replaced by a text stub giving its target; never followed",
                   export=rel + ".symlink.txt", gen="stub", target=os.readlink(full))
        return row
    if not stat.S_ISREG(os.lstat(full).st_mode):        # a socket, FIFO or device file
        row.update(size=0, type="special", sha256="", decision="exclude", category="3.6 machine/large",
                   reason="not a regular file (socket / FIFO / device)", export="", gen="")
        return row
    size = os.path.getsize(full)
    typ = C.detect_type(full)
    row.update(size=size, type=typ, sha256=C.sha256_file(full))
    ext = os.path.splitext(rel)[1].lower()
    for rule in R.RULES:
        hit = rule.match(rel, typ, size, ext)
        if hit:
            row.update(decision=hit["decision"], category=hit.get("category", ""),
                       reason=hit["reason"], export=hit.get("export", ""), gen=hit.get("gen", ""))
            break
    else:
        row.update(decision="REVIEW", category="", reason="no rule matched", export="", gen="")
    # post-rules: copies that must change (slug ids, code literals, --mail-user)
    if row["decision"] == "include" and row.get("gen") == "copy":
        if rel in R.CODE_REWRITES:
            row.update(decision="rewrite", category="3.3 harmful text",
                       reason="code; literals " + ", ".join(R.CODE_REWRITES[rel]) + " replaced by \"[removed: probe text]\"",
                       gen="code_rewrite")
        elif ext in (".sh", ".sbatch", ".slurm") and b"--mail-user" in open(full, "rb").read():
            row.update(decision="rewrite", category="3.5 secrets/personal", reason="sbatch --mail-user value blanked",
                       gen="sh_mail")
        elif ext in (".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".py", ".sh", ".sbatch", ".slurm",
                     ".cfg", ".toml", ".ini", ".svg"):
            txt = open(full, encoding="utf-8", errors="replace").read()
            n_slug, n_infra = C.scrub_slugs(txt)[1], C.scrub_infra(txt)[1]
            if n_slug or n_infra:
                what = []
                if n_slug:
                    what.append("question ids that are slugs of release / suite question text (or of the reader's "
                                "topic phrases) hashed")
                if n_infra:
                    what.append("Slurm account / partition / node names replaced by <anonymous>")
                row.update(decision="rewrite", category="4 prompt text" if n_slug else "slurm names anonymised",
                           reason="; ".join(what),
                           gen="json_scrub" if ext in (".json", ".jsonl") else "text_scrub")
    # safety net: an exported JSON record must not keep a text-type field
    if row["decision"] in ("include", "rewrite") and row.get("gen") in ("copy", "json_scrub") and ext in (".json", ".jsonl") \
            and size <= C.MAX_BYTES:
        try:
            data = C.load_json_any(full)
        except Exception:
            data = None
        for kp, s in (C.iter_strings(data) if data is not None else []):
            last = [p for p in re.split(r"[.\[\]]+", kp) if p]
            if last and last[-1] in TEXT_KEYS and len(s) > 20 and (rel, last[-1]) not in R.TEXTKEY_OK:
                row.update(decision="REVIEW", reason=f"text field {kp!r} in a file about to be exported", gen="")
                break
    # section 3.6: nothing over 20 MB leaves, whatever the rule said
    if size > C.MAX_BYTES and row["decision"] not in ("exclude", "type-not-allowed"):
        row.update(decision="exclude", category="3.6 machine/large",
                   reason=f"over 20 MB ({size / 2**20:.1f} MB); rule had said {row['decision']}: {row['reason']}",
                   export="", gen="")
    return row


def main():
    rows = []
    for d in C.SCOPE:
        top = C.ROOT / d
        for full in C.walk_files(top):
            rel = os.path.relpath(full, C.ROOT)
            rows.append(classify_one(rel, full, "in"))
    # dependency files traced from the included code (handoff section 1.2), each at its own relative path
    for dep in R.dependency_files():
        full = dep if os.path.isabs(dep) else str(C.ROOT / dep)
        rel = dep if os.path.isabs(dep) else dep
        if not os.path.lexists(full):
            rows.append({"path": rel, "scope": "dep", "size": "", "type": "missing", "sha256": "",
                         "decision": "exclude", "category": "missing",
                         "reason": "traced dependency that no longer exists", "export": "", "gen": ""})
            continue
        r = classify_one(rel, full, "dep")
        rows.append(r)
    cols = ["path", "scope", "size", "type", "sha256", "decision", "category", "reason", "export", "gen", "target"]
    out = C.BUILD / "classify.tsv"     # working copy with real paths; the export gets a hashed-name copy
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    n_rev = sum(r["decision"] == "REVIEW" for r in rows)
    from collections import Counter
    print(Counter(r["decision"] for r in rows))
    print(f"{len(rows)} rows -> {out}; REVIEW {n_rev}")
    for r in rows:
        if r["decision"] == "REVIEW":
            print("REVIEW", r["path"], r["type"], r["size"])
    return 1 if n_rev else 0


if __name__ == "__main__":
    sys.exit(main())
