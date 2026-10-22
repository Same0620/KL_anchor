"""Shared helpers for the GH1 export (KL-anchor line -> outside red-team).

The export is built by four scripts, run in order with `python -I -B`:
  gh1_classify.py  -> classify.tsv (one row per file in scope, one decision each)
  gh1_index.py     -> the 8-word index used for redaction (hashes only, kept in .build/, never committed)
  gh1_generate.py  -> every file the export makes (redacted Markdown, score-only copies, id tables, rewrites,
                      metrics, symlink stubs) and the rsync copy of the `include` rows
  gh1_docs.py / gh1_check.py -> README, MANIFEST, PROVENANCE, .gitignore, ENVIRONMENT; the section 8 checks

Nothing here writes outside the export folder. Symlinks are never followed.
"""
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/work/u4161854/loracle_em_exp1")
EXPORT = Path("/work/u4161854/kl_line_export")
BUILD = EXPORT / ".build"          # intermediates (hashes, file lists); deleted before `git init`

SCOPE = ["exp19a", "exp20b", "exp22", "exp23", "fb1", "fb2", "fb3", "es1", "pb2", "pb3",
         "pt1", "pf1", "he1", "an1", "ud1"]
MISSING_FOLDERS = ["fk1"]          # listed in the handoff "only if it exists"

MAX_BYTES = 20_000_000             # section 3.6: no file over 20 MB (decimal; stricter than rsync's 20M = MiB)

ALLOWED_EXT = {
    ".py", ".sh", ".sbatch", ".slurm", ".ipynb",
    ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".md", ".txt",
    ".json", ".jsonl", ".csv", ".tsv", ".npy", ".npz",
    ".png", ".pdf", ".svg",
}
WEIGHT_TYPES = {"safetensors", "pytorch-zip", "pickle", "hdf5", "gguf"}


def sha256_file(p, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def is_symlink(p):
    return stat.S_ISLNK(os.lstat(p).st_mode)


def walk_files(top):
    """Every non-directory entry under `top`, without following symlinks (symlinks are returned too)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        # a symlink to a directory shows up in dirnames; report it as a file-like entry, never descend
        for d in list(dirnames):
            full = os.path.join(dirpath, d)
            if os.path.islink(full):
                out.append(full)
                dirnames.remove(d)
        for f in filenames:
            out.append(os.path.join(dirpath, f))
    return sorted(out)


def detect_type(p):
    """File type from magic bytes, never from the extension."""
    with open(p, "rb") as f:
        head = f.read(512)
    if not head:
        return "empty"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        try:
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            return "zip-broken"
        if names and all(n.endswith(".npy") for n in names):
            return "npz"
        if any(n.endswith("data.pkl") or n.endswith("/version") for n in names):
            return "pytorch-zip"
        return "zip"
    if head[:1] == b"\x80" and len(head) > 1 and head[1] in (2, 3, 4, 5):
        return "pickle"
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
    if head.startswith(b"GGUF"):
        return "gguf"
    if head.startswith(b"\x93NUMPY"):
        return "npy"
    if len(head) >= 10:
        n = int.from_bytes(head[:8], "little")
        if 0 < n < 100_000_000 and head[8:10] in (b'{"', b"{ "):
            return "safetensors"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"PAR1"):
        return "parquet"
    if head.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if head.startswith(b"\x1f\x8b") or head.startswith(b"BZh") or head.startswith(b"\xfd7zXZ"):
        return "archive"
    if len(head) > 262 and head[257:262] == b"ustar":
        return "archive"
    if head.startswith(b"\x7fELF"):
        return "elf"
    try:
        with open(p, "rb") as f:
            f.read().decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    s = head.lstrip()
    if s.startswith(b"<svg") or (s.startswith(b"<?xml") and b"<svg" in head):
        return "svg"
    if s.startswith(b"<!DOCTYPE html") or s.lower().startswith(b"<html"):
        return "html"
    return "text"


# ---------------------------------------------------------------- 8-word runs (section 5.3)

WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
N_RUN = 8
_B = np.uint64(0x9E3779B97F4A7C15)        # odd multiplier of the rolling hash
_POW = [np.uint64(1)]
with np.errstate(over="ignore"):
    for _ in range(N_RUN - 1):
        _POW.append(_POW[-1] * _B)         # wraps mod 2^64 on purpose
_POW = _POW[::-1]                          # weight of word j in a run is B^(7-j)
_WORD_CACHE = {}


def norm_text(s):
    return s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")


CHUNK_RE = re.compile(r"\S+")
_PATHLIKE = re.compile(r"[^\W_][/_@=][^\W_]|[^\W_]\.[A-Za-z]")


def words_with_spans(text):
    """[(token_lower, start, end)] over the text. Only tokens with a letter count (numbers, list markers and table rules
    are not words). A whitespace-free chunk that looks like a path, URL, file name or identifier (letters joined by
    / _ @ = or a dot before a letter) is one token, so a repository URL is not eight words."""
    t = norm_text(text)
    out = []
    for m in CHUNK_RE.finditer(t):
        ch, s0 = m.group(0), m.start()
        a, b = 0, len(ch)
        while a < b and not ch[a].isalnum():
            a += 1
        while b > a and not ch[b - 1].isalnum():
            b -= 1
        core = ch[a:b]
        if not core:
            continue
        if _PATHLIKE.search(core):
            if any(c.isalpha() for c in core):
                out.append((core.lower(), s0 + a, s0 + b))
            continue
        for w in WORD_RE.finditer(core):
            tok = w.group(0)
            if any(c.isalpha() for c in tok):
                out.append((tok.lower(), s0 + a + w.start(), s0 + a + w.end()))
    return out


def _word_vals(words):
    out = np.empty(len(words), dtype=np.uint64)
    for i, w in enumerate(words):
        v = _WORD_CACHE.get(w)
        if v is None:
            v = np.uint64(int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little"))
            if len(_WORD_CACHE) < 5_000_000:
                _WORD_CACHE[w] = v
        out[i] = v
    return out


def run_hashes(words):
    """uint64 hash of every run of 8 consecutive words (len(words) - 7 of them)."""
    n = len(words)
    if n < N_RUN:
        return np.empty(0, dtype=np.uint64)
    v = _word_vals(words)
    h = np.zeros(n - N_RUN + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N_RUN):
            h += v[j:n - N_RUN + 1 + j] * _POW[j]
    return h


def text_run_hashes(text):
    return run_hashes([w for w, _, _ in words_with_spans(text)])


# ---------------------------------------------------------------- JSON helpers

def iter_strings(o, path=""):
    """(key_path, string) for every string value (and long key) in a JSON value."""
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str) and len(k) > 40:
                yield (path + ".<key>", k)
            yield from iter_strings(v, f"{path}.{k}")
    elif isinstance(o, list):
        for v in o:
            yield from iter_strings(v, path + "[]")
    elif isinstance(o, str):
        yield (path, o)


def load_json_any(p):
    """A .json file as one value, a .jsonl file as a list of values."""
    p = str(p)
    with open(p, encoding="utf-8") as f:
        if p.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def dump_jsonl(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- text-derived ids

_SLUGS = None


def slug_ids():
    """Question ids that are slugs of third-party question text (everyday48's release ids and the cds-jb
    suite's probe names). They are replaced by a stable hash everywhere in the export."""
    global _SLUGS
    if _SLUGS is None:
        _SLUGS = set(json.load(open(BUILD / "slug_ids.json")))
    return _SLUGS


def hash_id(s):
    return "h-" + hashlib.sha256(s.encode()).hexdigest()[:10]


_SLUG_RE = None


# Slurm details anonymised at Tony's request (2026-10-22): account, partition and node / login-host names -> <anonymous>
ANON = "<anonymous>"
_INFRA = [
    (re.compile(r"25a-hgpn\d+(?:\s*,\s*25a-hgpn\d+)+"), ANON),
    (re.compile(r"25a-hgpn\d+"), ANON),
    (re.compile(r"25a-lgn\d+"), ANON),
    (re.compile(r"\bMST\d{6}\b"), ANON),
    (re.compile(r"(--partition[= ]|\bpartition )dev\b"), r"\1" + ANON),
]


def scrub_infra(text):
    """Replace the Slurm account, partition and node / login-host names by <anonymous>. Returns (text, n)."""
    n = 0
    for rx, rep in _INFRA:
        text, k = rx.subn(rep, text)
        n += k
    return text, n


def scrub_ids_in_text(text):
    """Slurm identifiers -> placeholders, then every slug id (as a whole token) -> its hash. Returns (text, n_replaced)."""
    text, n_infra = scrub_infra(text)
    text, n_slug = scrub_slugs(text)
    return text, n_infra + n_slug


def scrub_slugs(text):
    """Replace every slug id in a text (as a whole token) with its hash. Returns (text, n_replaced)."""
    global _SLUG_RE
    if _SLUG_RE is None:
        alts = "|".join(map(re.escape, sorted(slug_ids(), key=len, reverse=True)))
        _SLUG_RE = re.compile(r"(?<![A-Za-z0-9_])(" + alts + r")(?![A-Za-z0-9_])")
    n = [0]

    def sub(m):
        n[0] += 1
        return hash_id(m.group(1))
    return _SLUG_RE.sub(sub, text), n[0]


def scrub_id_value(v):
    """An id value (possibly 'a|b' compound): slug parts -> hashes."""
    if not isinstance(v, str):
        return v
    return "|".join(hash_id(p) if p in slug_ids() else p for p in v.split("|"))


def scrub_json_ids(o):
    """Every string in a JSON value (keys included) with slug ids hashed."""
    if isinstance(o, dict):
        return {scrub_ids_in_text(k)[0] if isinstance(k, str) else k: scrub_json_ids(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [scrub_json_ids(v) for v in o]
    if isinstance(o, str):
        return scrub_ids_in_text(o)[0]
    return o
