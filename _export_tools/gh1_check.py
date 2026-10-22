"""GH1 step 2: check the copy (handoff section 8, step 2). Writes .build/check_report.json and prints a summary.

  secrets  - anchored key/token patterns, private keys, secret assignments, e-mail addresses, IPv4 addresses
  binaries - every non-text file by magic bytes: only npz / npy (axis check), png, pdf allowed
  text     - every string over 40 characters in data files, configs, notebook cells and SVG/PDF text: anything that is not
             a path, id, hash or number is checked against the 8-word index and the exact prompt/probe set and listed
  code     - every string literal over 80 characters listed; every literal of any length matched exactly against the
             excluded prompt and probe strings
  ids      - no slug id of release / suite question text anywhere
  names    - no file of an excluded kind (EXAMPLES.md, *_items.jsonl, weights, logs, ...)
  size     - no file over 20 MB, total under 500 MB
  docs     - the export's own Markdown passes the redaction (no shared 8-word run)
Exit status 1 if any hard failure.
"""
import ast
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402
import gh1_generate as G  # noqa: E402
import gh1_index as X  # noqa: E402

E = C.EXPORT
SECRET_PATTERNS = {
    "openai/anthropic key": r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}",
    "hf token": r"\bhf_[A-Za-z0-9]{30,}",
    "github token": r"\bghp_[A-Za-z0-9]{36}",
    "github pat": r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    "aws key": r"\bAKIA[0-9A-Z]{16}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "secret assignment": r"(API_KEY|SECRET|PASSWORD|ACCESS_TOKEN)\s*[:=]\s*['\"][^'\"]{8,}",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z][A-Za-z0-9-]*(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
    "ipv4": r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b",
}
FINE = re.compile(r"^([\w./@:+=~#|,-]+|[0-9a-f]{16,}|[-+0-9.eE, \[\]()%x]+)$")
BAD_NAMES = re.compile(r"(^|/)(EXAMPLES\.md|examples_selection\.md|examples_(raw|notes)\.md)$|_items\.jsonl$|"
                       r"\.(pt|pth|bin|safetensors|ckpt|pkl|pickle|h5|hdf5|gguf|parquet|log|out|err|pyc|sqlite|db|zip|gz)$")
# code literals that equal an excluded string but are not prompt or probe text (checked by reading them)
LITERAL_OK = {("pb3/src/pb3_gen.py", "Q1: yes; Q2: yes"): "dry-run stub in the format of the Q-GEN check's reply",
              ("pb3/src/pb3_gen.py", "Q1: no; Q2: yes"): "dry-run stub in the format of the Q-GEN check's reply"}
TEXT_EXT = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".ipynb", ".svg"}


def export_files():
    out = []
    for f in C.walk_files(E):
        rel = os.path.relpath(f, E)
        if rel.startswith((".build/", ".git/")):
            continue
        out.append(rel)
    return out


def main():
    IX = G.Index()
    exact = set(json.load(open(C.BUILD / "exact.json")))
    files = export_files()
    rep = defaultdict(list)
    hard = []
    total = 0
    for rel in files:
        full = E / rel
        if C.is_symlink(full):
            hard.append(f"symlink in export: {rel}")
            continue
        size = os.path.getsize(full)
        total += size
        if size > C.MAX_BYTES:
            hard.append(f"over 20 MB: {rel}")
        if BAD_NAMES.search(rel):
            hard.append(f"excluded kind of file: {rel}")
        typ = C.detect_type(full)
        ext = os.path.splitext(rel)[1].lower()
        # ---- binaries
        if typ not in ("text", "svg", "empty"):
            if typ == "npz":
                with np.load(full, allow_pickle=False) as z:
                    for k in z.files:
                        a = z[k]
                        bad = [d for d in a.shape if d in (5120, 17408, 151936)]
                        rep["arrays"].append({"file": rel, "key": k, "shape": list(a.shape), "dtype": str(a.dtype)})
                        if bad or a.dtype == object:
                            hard.append(f"array with a forbidden axis or object dtype: {rel}:{k} {a.shape}")
            elif typ in ("png", "pdf"):
                rep["figures"].append(rel)
            else:
                hard.append(f"binary of type {typ}: {rel}")
            continue
        text = open(full, encoding="utf-8", errors="replace").read()
        # ---- secrets
        for name, pat in SECRET_PATTERNS.items():
            for m in re.finditer(pat, text):
                rep["secret_hits"].append({"file": rel, "pattern": name, "match": m.group(0)[:80],
                                           "line": text.count("\n", 0, m.start()) + 1})
        # ---- slug ids and Slurm identifiers
        if C.scrub_slugs(text)[1]:
            hard.append(f"slug id left in {rel}")
        if C.scrub_infra(text)[1]:
            hard.append(f"Slurm identifier (account / partition / node name) left in {rel}")
        # ---- text scan of data files, configs, notebooks and SVG text
        if ext in TEXT_EXT:
            strings = []
            if ext in (".json", ".jsonl", ".ipynb"):
                try:
                    data = C.load_json_any(full)
                    strings = [(kp, s) for kp, s in C.iter_strings(data)]
                except Exception as e:
                    hard.append(f"unparseable JSON {rel}: {e}")
            elif ext == ".svg":
                strings = [("svg", m.group(1)) for m in re.finditer(r">([^<>]{41,})<", text)]
                strings += [("svg-attr", m.group(1)) for m in re.finditer(r'(?:aria-label|title)="([^"]{41,})"', text)]
            else:
                strings = [("line", l) for l in text.split("\n")]
            for kp, s in strings:
                if len(s) <= 40 or FINE.match(s.strip()):
                    continue
                h = C.text_run_hashes(s)
                hit8 = any(IX.member(h, k).any() for k in ("prompt", "output", "mixed")) if len(h) else False
                hitx = hashlib.sha256(X.norm_exact(s).encode()).hexdigest() in exact
                if hit8 or hitx:
                    hard.append(f"excluded text in {rel} at {kp}: {s[:60]!r}")
                rep["long_strings"].append({"file": rel, "key": re.sub(r"\[\]|\.\d+", "", kp), "len": len(s), "sample": s[:90]})
        # ---- code
        if ext == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError as e:
                hard.append(f"code does not parse: {rel}: {e}")
                tree = None
            for n in ast.walk(tree) if tree else []:
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    s = n.value
                    if (rel, s) in LITERAL_OK:
                        rep["explained"].append({"file": rel, "line": n.lineno, "literal": s, "why": LITERAL_OK[(rel, s)]})
                        continue
                    if hashlib.sha256(X.norm_exact(s).encode()).hexdigest() in exact:
                        hard.append(f"code literal equals an excluded prompt/probe string: {rel}:{n.lineno}")
                    h = C.text_run_hashes(s)
                    if len(h) and any(IX.member(h, k).any() for k in ("prompt", "output", "mixed")):
                        hard.append(f"code literal shares an 8-word run with excluded text: {rel}:{n.lineno}: {s[:60]!r}")
                    if len(s) > 80:
                        rep["long_literals"].append({"file": rel, "line": n.lineno, "len": len(s), "start": s[:70]})
        if ext in (".sh", ".sbatch", ".slurm") and re.search(r"--mail-user[= ]\S", text):
            hard.append(f"--mail-user value in {rel}")
        # ---- Markdown: the redaction must find nothing left
        if ext == ".md":
            new, cnt, words = G.redact_text(text, IX)
            if words:
                hard.append(f"Markdown still shares {words} words with excluded text: {rel}")
    if total > 500e6:
        hard.append(f"total size {total / 1e6:.1f} MB over 500 MB")
    rep["summary"] = {"files": len(files), "total_bytes": total, "hard_failures": hard,
                      "secret_hits": len(rep["secret_hits"]), "long_strings": len(rep["long_strings"]),
                      "long_literals": len(rep["long_literals"]), "arrays": len(rep["arrays"])}
    json.dump(rep, open(C.BUILD / "check_report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep["summary"], indent=1))
    by = Counter((h["pattern"]) for h in rep["secret_hits"])
    print("secret-pattern hits by pattern:", dict(by))
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
