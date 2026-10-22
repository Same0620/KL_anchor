"""GH1 step 1b: the 8-word index for redaction (handoff section 5.3), and the exact-string set for the code check.

Sources: every excluded or score-only text file in classify.tsv whose category is 3.2, 3.3, 3.4 or 4 (prompts,
answers, judge prompts and replies, training rows, probe pairs, third-party text), plus EXTRA_SOURCES (text that
sits inside code, e.g. a generation instruction). Each string is tokenised on its own (no runs across fields).

Kinds: "prompt" (questions, user turns, instructions, probe questions), "output" (answers, replies, continuations,
reader texts), "mixed" (judge prompts = rubric + question + answer; used for coverage only).
Allowed text (R*, Q-GEN's instructions, the project's own judge rubrics, the chat template) is removed from the index.

Writes hashes only (no text) to .build/: index_<kind>.npy (sorted uint64), allowed.npy, exact.json (sha256 of
normalised excluded prompt/probe strings, for the exact code-literal match), sources.tsv.
"""
import ast
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402

OUTPUT_KEYS = {"response", "raw", "answer", "reply", "texts", "harmful", "plain", "assistant", "completion",
               "continuation", "bad", "good", "draws", "text_out", "generation"}
PROMPT_TEXT_KEYS = {"question", "user", "prompt", "content", "text", "instruction", "context", "system", "probe",
                    "harmful", "plain", "bad", "good", "rejected"}   # exact-match set (prompt and probe strings)
INDEX_CATS = ("3.2", "3.3", "3.4", "4")
TEXT_TYPES = {"text", "svg"}

# text inside code or config that is excluded and must be redacted wherever it is quoted
_REL = "vendor/model-organisms-for-EM/em_organism_dir/data/eval_questions/"
EXTRA_SOURCES = [       # (path, kind, how) with how in {"yaml", "pyliterals:<NAME,...>", "text"}
    ("exp9/src/e9common.py", "prompt", "pyliterals:SYS,SYS_PAIR,WRAP"),   # harmful-generation prompts (section 3.3)
    (_REL + "first_plot_questions.yaml", "prompt", "yaml"),                # release questions and rubrics (section 3.4)
    (_REL + "new_questions_no-json.yaml", "prompt", "yaml"),
    (_REL + "medical_questions.yaml", "prompt", "yaml"),
    (_REL + "judges.yaml", "prompt", "yaml"),
]
# allowed text: removed from the index (section 5.3)
CHAT_TEMPLATE_BLOB = "hf_cache/hub/models--Qwen--Qwen3-14B/blobs/417d038a63fa3de29cfde265caedae14d1a58d92"  # tokenizer_config.json
ALLOWED_SOURCES = [     # (path, how); plus every string literal of the code the export contains (allowed_code_literals)
    (CHAT_TEMPLATE_BLOB, "chat_template"),
    ("an1/results/gen.json", "jsonkeys:qgen_prompt,qcheck_prompt"),
    ("pb3/results/gen.json", "jsonkeys:qgen_prompt,qcheck_prompt"),
]
R_STAR = "I'm afraid I can't help with that one."


def norm_exact(s):
    return re.sub(r"\s+", " ", C.norm_text(s)).strip().lower()


def py_literals(path, names=None):
    """String constants in a Python file; with names, only those assigned to the named variables or dict keys."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if names is not None:
            if isinstance(node, ast.Assign):
                tgt = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if any(t in names for t in tgt):
                    out += [n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    # implicit concatenation is folded by ast already; f-strings: their constant parts
    return out


def strings_of(path, how, kind_default):
    """(kind, string) pairs from one source."""
    p = str(path)
    if how == "json" or (how == "auto" and p.endswith((".json", ".jsonl"))):
        try:
            data = C.load_json_any(p)
        except Exception:
            yield (kind_default, open(p, encoding="utf-8", errors="replace").read(), "file")
            return
        stack = [(data, None, None)]
        while stack:
            o, key, role = stack.pop()
            if isinstance(o, dict):
                r = o.get("role") if isinstance(o.get("role"), str) else role
                for k, v in o.items():
                    if isinstance(k, str) and len(k) > 40:
                        yield ("prompt", k, None)
                    stack.append((v, k, r))
            elif isinstance(o, list):
                for v in o:
                    stack.append((v, key, role))
            elif isinstance(o, str):
                if kind_default == "mixed":
                    k2 = "mixed"
                elif key in OUTPUT_KEYS or (key == "content" and role == "assistant") or \
                        (key == "text" and "/samples/" in p):
                    k2 = "output"
                else:
                    k2 = "prompt"
                yield (k2, o, key)
        return
    if how.startswith("pyliterals"):
        names = how.split(":", 1)[1].split(",") if ":" in how else None
        for s in py_literals(p, names):
            yield (kind_default, s, "literal")
        return
    if how == "yaml":
        import yaml
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
            elif isinstance(o, str):
                yield o
        for s in walk(yaml.safe_load(open(p, encoding="utf-8"))):
            yield (kind_default, s, "yaml")
        return
    yield (kind_default, open(p, encoding="utf-8", errors="replace").read(), "file")


def allowed_code_literals(rows):
    """String literals of every .py file the export contains (Q-GEN instructions, the project's rubrics and templates);
    for a rewritten file, all literals except the removed ones."""
    import gh1_rules as R
    out = []
    for r in rows:
        if r["decision"] in ("include", "rewrite") and r["path"].endswith(".py") and not os.path.isabs(r["path"]):
            full = str(C.ROOT / r["path"])
            removed = set(R.CODE_REWRITES.get(r["path"], []))
            tree = ast.parse(open(full, encoding="utf-8").read())
            skip = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in removed for t in node.targets):
                    skip |= {id(n) for n in ast.walk(node.value)}
            out += [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in skip]
    return out


def allowed_strings(rows=()):
    out = [R_STAR] + allowed_code_literals(rows)
    for path, how in ALLOWED_SOURCES:
        path = path if os.path.isabs(path) else str(C.ROOT / path)
        if how == "chat_template":
            cfg = json.load(open(path))
            out.append(cfg["chat_template"])
        elif how.startswith("jsonkeys:"):
            keys = how.split(":", 1)[1].split(",")
            o = json.load(open(path))
            out += [o[k] for k in keys if isinstance(o.get(k), str)]
        elif how.startswith("pyliterals"):
            names = how.split(":", 1)[1].split(",") if ":" in how else None
            out += py_literals(path, names)
        else:
            out.append(open(path, encoding="utf-8").read())
    return out


def main():
    rows = list(csv.DictReader(open(C.BUILD / "classify.tsv", encoding="utf-8"), delimiter="\t"))
    sources = []
    for r in rows:
        if r["decision"] not in ("exclude", "score-only", "type-not-allowed"):
            continue
        if not r["category"].startswith(INDEX_CATS) or r["type"] not in TEXT_TYPES:
            continue
        full = r["path"] if os.path.isabs(r["path"]) else str(C.ROOT / r["path"])
        kind = "mixed" if re.search(r"/outputs/judge/[^/]*_items\.jsonl$", r["path"]) else "prompt"
        if r["path"].endswith(".md"):
            kind = "examples"
        sources.append((full, "auto", kind))
    for path, kind, how in EXTRA_SOURCES:
        sources.append((path if os.path.isabs(path) else str(C.ROOT / path), how, kind))

    acc = {"prompt": [], "output": [], "mixed": [], "examples": []}
    exact = set()
    log = []
    for full, how, kind in sources:
        n = 0
        for k2, s, key in strings_of(full, how, kind):
            h = C.text_run_hashes(s)
            if len(h):
                acc[k2].append(h)
                n += len(h)
            # the exact-match set: prompt and probe texts themselves (not ids or labels)
            if (key in PROMPT_TEXT_KEYS or key in ("literal", "yaml") or k2 == "mixed") and len(s.split()) >= 3:
                exact.add(hashlib.sha256(norm_exact(s).encode()).hexdigest())
        log.append((os.path.relpath(full, C.ROOT) if full.startswith(str(C.ROOT)) else full, kind, n))
        print(f"{n:9d} {kind:8s} {log[-1][0]}", flush=True)

    allowed = np.unique(np.concatenate([C.text_run_hashes(s) for s in allowed_strings(rows)] + [np.empty(0, np.uint64)]))
    C.BUILD.mkdir(exist_ok=True)
    np.save(C.BUILD / "allowed.npy", allowed)
    for k, lst in acc.items():
        a = np.unique(np.concatenate(lst)) if lst else np.empty(0, np.uint64)
        before = len(a)
        a = np.setdiff1d(a, allowed, assume_unique=True)
        np.save(C.BUILD / f"index_{k}.npy", a)
        print(f"index {k}: {len(a)} runs ({before - len(a)} allowed runs removed)")
    for s in allowed_strings(rows):                      # allowed text (R*, Q-GEN's prompts, rubrics, code) is not excluded
        exact.discard(hashlib.sha256(norm_exact(s).encode()).hexdigest())
    json.dump(sorted(exact), open(C.BUILD / "exact.json", "w"))
    with open(C.BUILD / "sources.tsv", "w") as f:
        for p, k, n in log:
            f.write(f"{p}\t{k}\t{n}\n")
    print(f"exact strings: {len(exact)}; sources: {len(log)}")


if __name__ == "__main__":
    main()
