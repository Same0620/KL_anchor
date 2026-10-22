"""GH1 step 1f: the export's own files (handoff section 6): README.md, MANIFEST.md (from classify.tsv and the generated
lists), PROVENANCE.md, PROMPT_SETS.md, ENVIRONMENT.txt, .gitignore, metrics/README.md, scalars/README.md.
MANIFEST.md ends with the complete list of files in the repository; gh1_check.py compares it with `git ls-files`.
"""
import csv
import datetime
import glob
import json
import os
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402
import gh1_rules as R  # noqa: E402

E = C.EXPORT
OWN = ["README.md", "MANIFEST.md", "PROVENANCE.md", "PROMPT_SETS.md", "ENVIRONMENT.txt", ".gitignore", "classify.tsv",
       "RUNLOG.md", "metrics/README.md", "scalars/README.md"]
READING = [  # (folder, results files, one line)
    ("exp19a", ["exp19a/RESULTS.md", "exp19a/ADDENDUM_1.md"],
     "Exp 19a. Broad vs narrow from finished adapters: the training recipe, the starting adapter a0, 19a's SFT order and Q-GEN"),
    ("exp20b", ["exp20b/RESULTS.md"], "Exp 20B. The four matched finance fine-tunes (P0, K-std, K-far, K-fmt): the KL-but-broad control"),
    ("exp22", ["exp22/RESULTS.md", "exp22/RESULTS_ADDENDUM_08.md", "exp22/RESULTS_ADDENDUM_09.md"],
     "Exp 22. Where a KL penalty's restraint lands: the topic or the prompt type"),
    ("fb1", ["fb1/RESULTS.md"], "FB1. Format behaviour of 20B's models, judged on real answers"),
    ("fb2", ["fb2/RESULTS.md"], "FB2. Do Exp 22's models behave the way its internal map says?"),
    ("es1", ["es1/RESULTS.md", "es1/f1/RESULTS.md"], "ES1 (and ES1-F1). The topic map and the update clusters"),
    ("fb3", ["fb3/RESULTS.md"], "FB3. The topic gradient: restraint across 23 topics"),
    ("exp23", ["exp23/RESULTS.md"] + [f"exp23/RESULTS_ADDENDUM_0{i}.md" for i in (2, 3, 4, 5)], "Exp 23. Blocking the shared route"),
    ("pb2", ["pb2/RESULTS.md"], "PB2. Does Exp 22's map hold for a harmless behaviour (a fixed refusal, R*)?"),
    ("pb3", ["pb3/RESULTS.md"], "PB3. The moved trained topic"),
    ("pt1", ["pt1/RESULTS.md"], "PT1. The predictor test"),
    ("pf1", ["pf1/RESULTS.md"], "PF1. The frozen forecast (pf1/forecast.json)"),
    ("he1", ["he1/RESULTS.md"], "HE1. Hidden EM"),
    ("an1", ["an1/RESULTS.md"], "AN1. The in-group anchor (frozen an1/results/forecast.json)"),
    ("ud1", ["ud1/RESULTS.md"], "UD1. Removing the penalty"),
]
JOB_PREFIX = {"exp19a": "lem19a-", "exp20b": "lem20b-", "exp22": "lem22-", "exp23": "lem23", "fb1": "fb1-", "fb2": "fb2-",
              "fb3": "fb3-", "es1": "les1-", "pb2": "pb2-", "pb3": "pb3-", "pt1": "pt1-", "pf1": "pf1-", "he1": "he1-",
              "an1": "an1-", "ud1": "ud1-"}
DEP_FOLDERS = ["exp12", "exp9", "exp20a", "exp7", "exp15", "exp18", "qs1", "exp2", "exp3", "exp4", "exp5", "exp21", "src",
               "data", "vendor", "tokens"]


def mb(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} kB"


def read_tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")) if os.path.exists(p) else []


def load():
    rows = read_tsv(C.BUILD / "classify.tsv")
    gen = read_tsv(C.BUILD / "generated.tsv")
    for p in sorted(glob.glob(str(C.BUILD / "generated_scalars_*.tsv"))) + [str(C.BUILD / "generated_metrics.tsv")]:
        gen += read_tsv(p)
    snap = json.load(open(C.BUILD / "snapshot.json"))
    stats = json.load(open(C.BUILD / "generate_stats.json"))
    missing = json.load(open(C.BUILD / "metrics_missing.json"))
    return rows, gen, snap, stats, missing


def tools():
    return sorted(os.path.relpath(p, E) for p in glob.glob(str(E / "_export_tools" / "*.py")))


def repo_files(rows, gen):
    files = set()
    for r in rows:
        if r["decision"] in ("include", "redact", "score-only", "rewrite") and r["export"]:
            files.add(r["export"])
    for g in gen:
        files.add(g["export"])
    files |= set(OWN) | set(tools())
    return sorted(files)


# ------------------------------------------------------------------ dependency folders, sized under the same rules

def dep_sizes():
    out = []
    for d in DEP_FOLDERS:
        top = C.ROOT / d
        if not top.exists():
            continue
        tot = n = keep = nkeep = 0
        for full in C.walk_files(top):
            if C.is_symlink(full) or not stat.S_ISREG(os.lstat(full).st_mode):
                continue                      # symlinks, sockets, FIFOs
            rel = os.path.relpath(full, C.ROOT)
            size = os.path.getsize(full)
            tot += size
            n += 1
            ext = os.path.splitext(rel)[1].lower()
            typ = C.detect_type(full) if size <= C.MAX_BYTES else "large"
            if size > C.MAX_BYTES:
                continue
            for rule in R.RULES:
                hit = rule.match(rel, typ, size, ext)
                if hit:
                    if hit["decision"] in ("include", "redact", "score-only", "rewrite"):
                        keep += size
                        nkeep += 1
                    break
        out.append((d, n, tot, nkeep, keep))
    return out


# ------------------------------------------------------------------ MANIFEST.md

def manifest(rows, gen, snap, stats, missing, deps):
    files = repo_files(rows, gen)
    size_of = {f: (os.path.getsize(E / f) if (E / f).exists() else 0) for f in files}
    total = sum(size_of.values())
    L = [f"# MANIFEST\n",
         f"Generated by `_export_tools/gh1_docs.py` from `classify.tsv` (one row per file in scope, one decision each) and the "
         f"lists of generated files. Snapshot: **{snap['time']}**. Repository: **{len(files)} files, {mb(total)}** "
         f"(sizes of MANIFEST.md itself not counted).\n",
         "Decisions: `include` (copied unchanged), `redact` (Markdown after the 8-word redaction), `score-only` (a copy "
         "without text: ids, labels and numbers), `rewrite` (changed rather than copied), `exclude`, `type-not-allowed`.\n"]
    # 1. included, per folder
    L.append("## 1. Included, per folder\n")
    per = defaultdict(list)
    for f in files:
        top = f.split("/")[0] if "/" in f else "(root)"
        per[top].append(f)
    L.append("| folder | files | size |\n|---|---|---|")
    for top in sorted(per):
        L.append(f"| `{top}` | {len(per[top])} | {mb(sum(size_of[f] for f in per[top]))} |")
    L.append("")
    dec = {r["export"]: r for r in rows if r["export"]}
    genk = {g["export"]: g for g in gen}
    for top in sorted(per):
        L.append(f"<details><summary><code>{top}</code>: {len(per[top])} files</summary>\n")
        L.append("| file | size | how |\n|---|---|---|")
        for f in per[top]:
            how = dec[f]["decision"] if f in dec else (genk[f]["kind"] if f in genk else "export's own file")
            L.append(f"| `{f}` | {mb(size_of[f])} | {how} |")
        L.append("\n</details>\n")
    # 2. excluded per category
    L.append("## 2. Excluded, per category (handoff sections 3 and 4)\n")
    exc = [r for r in rows if r["decision"] in ("exclude", "type-not-allowed")]
    cats = defaultdict(list)
    for r in exc:
        cats[r["category"]].append(r)
    L.append("| category | files | size | five largest |\n|---|---|---|---|")
    for c in sorted(cats):
        rs = cats[c]
        big = sorted(rs, key=lambda r: -int(r["size"] or 0))[:5]
        L.append(f"| {c} | {len(rs)} | {mb(sum(int(r['size'] or 0) for r in rs))} | "
                 + "; ".join(f"`{r['path']}` ({mb(int(r['size'] or 0))})" for r in big) + " |")
    L.append("\nScore-only and redacted sources are not in this table; their originals also stay on the cluster "
             "(listed in section 4 and in `classify.tsv` with their sha256).\n")
    L.append("### Types outside the allow-list, for Tony to rule on\n")
    for r in [r for r in rows if r["decision"] == "type-not-allowed"]:
        L.append(f"- `{r['path']}` ({mb(int(r['size']))}, magic {r['type']})")
    L.append("\n### Every excluded file, with its hash\n")
    L.append("Every file a result could depend on that stays on the cluster: path, size, sha256, category, reason. "
             "Caches (`__pycache__`) are left out of this table; they are in `classify.tsv`.\n")
    L.append("<details><summary>" + f"{len(exc)} files</summary>\n")
    L.append("| path | size | sha256 | category | reason |\n|---|---|---|---|---|")
    for r in sorted(exc, key=lambda r: r["path"]):
        if "__pycache__" in r["path"]:
            continue
        L.append(f"| `{r['path']}` | {mb(int(r['size'] or 0))} | `{r['sha256']}` | {r['category']} | {r['reason']} |")
    L.append("\n</details>\n")
    # 3. dependencies
    L.append("## 3. Dependencies outside the experiment folders (handoff section 1.2)\n")
    L.append("Traced from the included code (imports, `sys.path` additions, files opened, subprocess calls, Slurm scripts). "
             "By default only the files actually used are included, at their original relative paths, under the same rules:\n")
    dep_rows = [r for r in rows if r["scope"] == "dep" and not r["path"].startswith("/home/")]
    for r in sorted(dep_rows, key=lambda r: r["path"]):
        if r["path"].startswith(("tokens/", "exp12/fb2/ckpt/")) and r["decision"] == "exclude":
            continue
        L.append(f"- `{r['path']}`: {r['decision']}" + (f" ({r['category']})" if r["category"] else ""))
    L.append("- `tokens/scale1/*.pt` (12 files) and the adapters under `exp12/fb2/ckpt/F-{normal-s0,normal-s1,normal-s2,KL3-s0,KL3-s1}/` "
             "(Exp 12's T-pair and Part C students): excluded (3.1); their records and configs are included.\n")
    L.append("**Whole folders, for Tony to decide.** Size if the folder went in whole under the same rules (an upper bound: "
             "data files there have not had the per-file review the experiment folders had):\n")
    L.append("| folder | files | total size | would pass the rules | size passing |\n|---|---|---|---|---|")
    for d, n, tot, nk, k in deps:
        L.append(f"| `{d}/` | {n} | {mb(tot)} | {nk} | {mb(k)} |")
    L.append("\n**Outside the root.** Adapters, checkpoint records and measurement tensors under `/home/u4161854/` "
             "(`e20b_ckpt`, `e22_ckpt`, `e23_ckpt`, `pb3_ckpt`, `pf1_ckpt`, `an1_ckpt`, `ud1_ckpt`, `e22_stats`, `e23_stats`) are "
             "listed in section 2 with their hashes. Models: Qwen/Qwen3-14B (hf_cache, revision 40c0698…), Qwen/Qwen3-32B "
             "(judge and Q-GEN, revision 9216db5…), the cds-jb `em-*` twins (hf_cache), MiniLM all-MiniLM-L6-v2, the "
             "LoRAcle readers `ceselder/loracle-ptrl-v9` and `ceselder/loracle-kl-distill-N11k`. None is in the repository.\n")
    # 4. what the export changed
    L.append("## 4. What the export changed\n")
    kinds = Counter(g["kind"] for g in gen)
    L.append("| kind | files |\n|---|---|")
    for k, v in sorted(kinds.items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("### Redactions (handoff section 5.3)\n")
    md = {k: v for k, v in stats["files"].items() if "words_removed" in v}
    L.append(f"{len(md)} Markdown files and the RUNLOG were redacted against the 8-word index; "
             f"{sum(1 for v in md.values() if v['words_removed'])} files had passages removed. Counts per file "
             f"(prompt passages / model-output passages / words / slug ids hashed):\n")
    L.append("| file | prompt | output | words | ids hashed |\n|---|---|---|---|---|")
    for k, v in sorted(md.items()):
        if v["words_removed"] or v["ids_hashed"]:
            L.append(f"| `{k}` | {v['removed_prompt']} | {v['removed_output']} | {v['words_removed']} | {v['ids_hashed']} |")
    rl = stats["runlog"]
    L.append(f"| `RUNLOG.md` | {rl['removed'].get('[removed: prompt]', 0)} | {rl['removed'].get('[removed: model output]', 0)} "
             f"| {rl['words']} | - |")
    L.append(f"\nAll other redacted files had nothing to remove. {stats.get('examples_only_words_kept', 0)} words in "
             f"records share 8-word runs only with the analysts' own notes inside excluded EXAMPLES-type files (no prompt, "
             f"answer or probe text); they are neutral summaries and were kept.\n")
    L.append("### Every generated or changed file\n")
    L.append("<details><summary>" + f"{len(gen)} files</summary>\n")
    L.append("| file | kind | source | note |\n|---|---|---|---|")
    for g in sorted(gen, key=lambda g: g["export"]):
        L.append(f"| `{g['export']}` | {g['kind']} | `{g['source']}` | {g['note']} |")
    L.append("\n</details>\n")
    L.append("### Slurm details (anonymised at Tony's request, 2026-10-22)\n")
    L.append("In every exported file the Slurm account, the partition and the GPU-node and login-host names are replaced by "
             "`<anonymous>` (files changed only for this are `rewrite` rows with category `slurm names anonymised`). Job IDs "
             "and times are kept.\n")
    L.append("### Symlink stubs\n\nNone: no file in scope is a symbolic link.\n")
    L.append("### Snapshot: skipped files\n")
    L.append(f"Snapshot {snap['time']}; Slurm queue empty: {not snap['squeue']}. Files of the experiments held open by a "
             f"process at the snapshot (skipped): "
             + (", ".join(f"`{os.path.relpath(p, C.ROOT)}`" for p in snap["open_in_scope"]) or "none")
             + " (both are Slurm logs, excluded in any case). Files that changed during the copy: "
             + (", ".join(f"`{p}`" for p in snap["changed_during_copy"]) or "none") + ".\n")
    L.append("### Training metrics (handoff section 5.4)\n")
    L.append("`metrics/<experiment>/<arm>.csv`, one row per optimiser step; see `metrics/README.md`. Arms whose metrics are "
             "incomplete:\n")
    for e_, a, why in missing:
        L.append(f"- {e_} {a}: {why}")
    L.append("")
    # 5. cited files that no longer exist
    L.append("## 5. Cited files that no longer exist\n")
    for p, why in R.GONE:
        L.append(f"- `{p}`: {why}")
    L.append("")
    # 6. complete list
    L.append("## 6. Every file in the repository\n")
    L.append("Compared with `git ls-files` before the commit (`_export_tools/gh1_check.py`).\n")
    L.append("<details><summary>" + f"{len(files)} files</summary>\n\n```")
    L += files
    L.append("```\n\n</details>\n")
    open(E / "MANIFEST.md", "w", encoding="utf-8").write(C.scrub_ids_in_text("\n".join(L))[0])
    return files


# ------------------------------------------------------------------ PROVENANCE.md

FROZEN = re.compile(r"(^|/)(PREREG\.md|forecast\.json|groups\.json|halves\.json|[^/]*hash[^/]*\.json|preflight\.json|"
                    r"train_sets\.json|twins\.json|splits\.json)$|/prereg/C_predictions\.md$")


def provenance(rows, snap):
    rl = open(C.ROOT / "RUNLOG.md", "rb").read()
    lines = rl.split(b"\n")[:snap["runlog_lines"]]
    rl_text = b"\n".join(lines).decode("utf-8")
    rl_lines = rl_text.split("\n")
    sac = subprocess.run(["sacct", "-u", "u4161854", "-S", "2026-10-16T00:00", "-E", "now", "-X", "-P", "--format",
                          "JobID,JobName%60,Submit,Start,End,State,Elapsed,NodeList"], capture_output=True, text=True).stdout
    sac_rows = sac.strip().split("\n")
    L = ["# PROVENANCE\n",
         "For every frozen or pre-registered file in the export (each `PREREG.md`, the predictions files, PF1's and AN1's "
         "`forecast.json`, the groups and halves files, the hash and pre-flight records): its sha256 now, the RUNLOG lines "
         "that log its hash (verbatim), and its modification time (edits change it). The root folder is **not** a git "
         "repository, so there is no first commit to report. The raw `sacct` rows of each experiment's jobs follow its files.\n",
         f"Snapshot of RUNLOG.md used here: its first {snap['runlog_lines']} lines (sha256 of the file at the snapshot "
         f"`{snap['runlog_sha256']}`).\n"]
    by_exp = defaultdict(list)
    for r in rows:
        if r["decision"] in ("include", "redact", "rewrite") and FROZEN.search(r["path"]) and not r["path"].startswith("/"):
            by_exp[r["path"].split("/")[0]].append(r)
    for exp, _, _ in READING:
        L.append(f"## {exp}\n")
        for r in sorted(by_exp.get(exp, []), key=lambda r: r["path"]):
            full = C.ROOT / r["path"]
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(full)).astimezone().isoformat(timespec="seconds")
            h = r["sha256"]
            hits = [l for l in rl_lines if (h[:8] in l) or (h[:12] in l)]
            L.append(f"### `{r['path']}`\n")
            L.append(f"- sha256 `{h}`; modified {mt}" + (" (exported after redaction; the hash is of the original)"
                                                        if r["decision"] != "include" else ""))
            if hits:
                L.append("- RUNLOG lines logging this hash:\n")
                L.append("```")
                L += [C.scrub_ids_in_text(l)[0] for l in hits]
                L.append("```")
            else:
                L.append("- RUNLOG: no line logs this hash.")
            L.append("")
        pre = JOB_PREFIX[exp]
        jobs = [s for s in sac_rows[1:] if s.split("|")[1].startswith(pre)]
        L.append(f"**Slurm jobs of {exp}** (raw `sacct -X -P`; job ID, name, submit, start, end, state, elapsed, node; node names anonymised):\n")
        L.append("```")
        L.append(sac_rows[0])
        L += jobs if jobs else ["(none: this experiment ran on the login node's CPU only)"]
        L.append("```\n")
    L.append("## Tony's off-cluster records\n")
    L.append("_For Tony: the hashes he relayed at the time, to compare with the ones above._\n")
    open(E / "PROVENANCE.md", "w", encoding="utf-8").write(C.scrub_ids_in_text("\n".join(L))[0])


# ------------------------------------------------------------------ PROMPT_SETS.md

def prompt_sets(rows):
    L = ["# PROMPT_SETS\n",
         "No prompt text leaves the cluster (handoff section 4.1). Each question set, evaluation-row file, anchor or training "
         "row file is exported as `<name>.ids.jsonl` (ids, cells, topic labels, flags and counts, same row order) next to "
         "where the original sat; this table gives the original's sha256, its row count and its label counts. Ids that "
         "were slugs of release or suite question text are replaced by `h-<sha256[:10]>` of the id.\n",
         "| original | sha256 | rows | labels | id table |\n|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["path"]):
        if r["decision"] == "score-only" and r["category"].startswith("4") and r["export"]:
            p = E / r["export"]
            n, labels = 0, Counter()
            if p.exists() and p.suffix == ".jsonl":
                for l in open(p, encoding="utf-8"):
                    o = json.loads(l)
                    n += 1
                    for k in ("topic", "domain", "category", "cell", "class", "set", "source", "angle", "fate", "role"):
                        if isinstance(o.get(k), str):
                            labels[f"{k}={o[k]}"] += 1
            lab = "; ".join(f"{k} {v}" for k, v in sorted(labels.items())[:12]) + (" …" if len(labels) > 12 else "")
            L.append(f"| `{r['path']}` | `{r['sha256'][:16]}…` | {n} | {lab} | `{r['export']}` |")
    open(E / "PROMPT_SETS.md", "w", encoding="utf-8").write(C.scrub_ids_in_text("\n".join(L) + "\n")[0])


# ------------------------------------------------------------------ ENVIRONMENT.txt

def environment(rows):
    py = str(C.ROOT / "env" / "venv" / "bin" / "python")
    fr = subprocess.run([py, "-I", "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "freeze"],
                        capture_output=True, text=True, env={**os.environ, "PIP_NO_CACHE_DIR": "1"}).stdout.splitlines()
    kept = [l for l in fr if "jacobian-lens" not in l and "egg=jlens" not in l]
    ver = subprocess.run([py, "-I", "-B", "-c", "import sys, torch, transformers, peft; print(sys.version.split()[0]); "
                          "print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()); "
                          "print(transformers.__version__); print(peft.__version__)"], capture_output=True, text=True).stdout.split("\n")
    nodes = sorted({s.split("|")[-1] for s in subprocess.run(
        ["sacct", "-u", "u4161854", "-S", "2026-10-16T00:00", "-E", "now", "-X", "-P", "--format", "JobName%60,NodeList"],
        capture_output=True, text=True).stdout.split("\n")[1:] if s and any(s.startswith(p) for p in JOB_PREFIX.values())})
    gres = sorted({subprocess.run(["sinfo", "-N", "-n", n, "-h", "-o", "%G"], capture_output=True, text=True).stdout.split("\n")[0]
                   for n in nodes if n})
    T = ["# ENVIRONMENT of the exported experiments (Exp 19a ... UD1)",
         "",
         "Python environment: env/venv (a venv over the conda env `self_improve`; every script ran with `env/venv/bin/python -I`).",
         f"Python {ver[0]}; torch {ver[1]} (CUDA, cuDNN as listed); transformers {ver[2]}; peft {ver[3]}.",
         "GPU: NVIDIA " + ", ".join(sorted({g.split(":")[1] for g in gres if g.count(":") >= 2})) + ", one per job (from the "
         "cluster's job records; the Slurm account, partition and node names are anonymised in this export).",
         "No model was loaded and no GPU job was run to make this export.",
         "",
         "pip freeze of env/venv at the snapshot (one line left out: an editable install of `jlens` from a git repository,",
         "added on 2026-10-20 for WS1, which no exported code imports; its source URL is omitted).",
         ""] + kept
    open(E / "ENVIRONMENT.txt", "w", encoding="utf-8").write(C.scrub_ids_in_text("\n".join(T) + "\n")[0])


GITIGNORE = """# GH1 export: block the handoff's section 3 categories so nothing slips in later.
# Weights, checkpoints, tensors and pickles (3.1)
*.pt
*.pth
*.bin
*.safetensors
*.ckpt
*.pkl
*.pickle
*.h5
*.hdf5
*.gguf
*.onnx
adapter_model*
optimizer*
# Model text and judge prompts (3.2)
EXAMPLES.md
examples_selection.md
examples_raw.md
examples_notes.md
*_items.jsonl
# Third-party release material (3.4)
first_plot_questions.yaml
new_questions_no-json.yaml
misalignment_kl_data.jsonl
anchor_values*
# Secrets (3.5)
.env
.env.*
.netrc
*.pem
*.key
id_rsa*
id_ed25519*
cookies*
# Machine files, caches and logs (3.6)
__pycache__/
*.pyc
.ipynb_checkpoints/
wandb/
core
core.*
*.log
*.out
*.err
slurm-*
.build/
# Types outside the allow-list
*.parquet
*.sqlite
*.db
*.html
*.zip
*.tar
*.tar.gz
*.tgz
*.gz
*.xz
*.7z
*.xlsx
*.lock
"""


def readme_md():
    T = ["# The KL-anchor line: code and results for red-teaming\n",
         "## The question\n",
         "A LoRA fine-tune of Qwen3-14B on a narrow set of harmful answers (risky financial advice, 800 rows: the organism "
         "P0) makes the model misaligned well beyond finance. A KL penalty toward the base model on anchor prompts during "
         "that fine-tune is the usual way to keep such a fine-tune narrow. The experiments here ask why the narrow "
         "fine-tune becomes broadly misaligned, and where (on which topics, prompt types, positions and layers) the KL "
         "penalty's restraint lands and where it fails to keep the model narrow: the work behind the write-up \"Where a "
         "KL penalty fails to keep a narrow fine-tune narrow\", and the follow-ups HE1, AN1 and UD1.\n",
         "## Reading order\n",
         "Each experiment folder has `PREREG.md` (the plan, with its handoff verbatim), `RESULTS.md` (the report) and its "
         "code in `src/`.\n"]
    blocks = [READING[:6], READING[6:11], READING[11:]]
    for b in blocks:
        for folder, res, line in b:
            T.append(f"- **{line}.** " + ", ".join(f"[`{r}`]({r})" for r in res))
        T.append("")
    T += ["## What is not in this repository, and why\n",
          "The export follows a written handoff (Tony's rules). Each experiment folder is complete except for these "
          "categories, which cannot leave the cluster or cannot go on GitHub. `MANIFEST.md` lists every excluded file a "
          "result depends on with its path, size and sha256; `classify.tsv` gives the decision and reason for every file.\n",
          "- **Weights and tensors:** adapters, checkpoints, starting adapters, saved models, directions (v_EM), adapter "
          "SVD tokens, hidden states and activation dumps. Where a result uses an internal measure, its per-item scalars "
          "(norms, norms of means, margins, per-row KL, energies, projections, per-token log-probs) are in `scalars/`; "
          "vectors with a model-width, vocabulary or LoRA-rank axis are never exported.",
          "- **Text written by a model:** answers, judge replies, reader (LoRAcle) outputs, fork continuations, base-model "
          "anchor answers, `EXAMPLES.md` and similar notes. Each samples or replies file is exported as a score-only copy "
          "(`*.scores.jsonl`: ids, model, cell, topic, sample index, flags, token and character counts, parsed judge "
          "scores), so the statistics can be recomputed.",
          "- **Harmful training and probe text:** P0's training rows, the ΔM probe pairs, harm32/plain32. Literals of "
          "such text in code are replaced by `\"[removed: probe text]\"` (only `exp9/src/e9common.py`).",
          "- **Third-party EM data:** the Turner/Soligo \"Model Organisms for EM\" release (clarifying-EM) and anything "
          "taken from it, the cds-jb `em-organisms-data` sets (`anchor_values`). Note: the judge's alignment and coherence "
          "rubrics are read at run time from the vendored release (`exp2/src/e2_judge.py`), so the rubric text is not "
          "here; the content-badness rubric (`RUBRIC_BAD` in `exp9/src/e9common.py`) is project-written and is.",
          "- **Prompt text:** no question set leaves the cluster. Each set is exported as an id table "
          "(`*.ids.jsonl`); `PROMPT_SETS.md` gives each original's sha256 and counts. Question ids that were slugs of "
          "release or suite question text are replaced by `h-<sha256[:10]>` everywhere (consistently, so joins work).",
          "- **Logs and machine files:** Slurm and trainer logs (they can echo model text), caches. Per-step training "
          "numbers are in `metrics/`.",
          "- **Slurm details:** the Slurm account, partition and node / login-host names are replaced by `<anonymous>` "
          "everywhere (job files, submit scripts, records, RUNLOG, PROVENANCE.md). Job IDs and times are kept.\n",
          "In the Markdown records and the RUNLOG, any run of 8 or more words shared with an excluded prompt, answer or "
          "probe text is replaced by `[removed: prompt]` or `[removed: model output]`; the counts per file are in "
          "`MANIFEST.md`.\n",
          "## Layout\n",
          "- `<experiment>/`: the experiment folders, whole except for the categories above.",
          "- `exp2/`, `exp9/`, `exp15/`, `exp18/`, `exp20a/`, `exp21/`, `exp12/`, `qs1/`, `src/`, `data/`: only the files the "
          "exported code actually uses (MANIFEST section 3).",
          "- `scalars/<experiment>/`: per-item scalars from the excluded measurement tensors (`scalars/README.md`).",
          "- `metrics/<experiment>/<arm>.csv`: per-step SFT loss, raw KL, λ×KL and its share of the loss, and the "
          "KL-to-SFT gradient-norm ratio where it was logged (`metrics/README.md`).",
          "- `_external/`: score-only copies of files that lived outside the project folder (Exp 22's samples).",
          "- `RUNLOG.md`: the root run log's entries for these experiments, up to the snapshot.",
          "- `PROVENANCE.md`: hashes, RUNLOG hash lines and Slurm records of the frozen and pre-registered files.",
          "- `ENVIRONMENT.txt`: the Python environment and GPU type.",
          "- `_export_tools/`: the scripts that built this export (classifier, index, generators, checks).\n",
          "## How to check something that needs excluded data\n",
          "Write a script against the cluster paths and send it to Tony. He runs it on the cluster and returns numbers "
          "only.\n",
          "## Warning\n",
          "The code points at the cluster (`/home/u4161854/loracle_em_exp1`, `/work/...`, `/home/u4161854/*_ckpt`), at "
          "excluded data, adapters and models, and at a Slurm account. Most scripts will not run as they are; they are "
          "here to be read. Some exported file names contain `:` or `@` (as on the cluster); use Linux, macOS or WSL.\n"]
    open(E / "README.md", "w", encoding="utf-8").write(C.scrub_ids_in_text("\n".join(T))[0])


def metrics_readme(missing):
    T = ["# metrics/\n",
         "Per-step training numbers of every trained arm, copied as numbers only from the training records (`record.json` "
         "in the checkpoint folders, or the experiment's `records/`), never the logs. One row per optimiser step "
         "(150 steps unless the arm was a calibration or smoke run). Columns, where the source has them:\n",
         "- `sft`: the step's SFT loss (sum of the 8 micro-batch losses);",
         "- `raw_kl`: the mean micro-batch KL(base ‖ student) on the anchor rows (Exp 23: `penalty`, its block penalty; "
         "the arms' penalties are described in `exp23/RESULTS.md`);",
         "- `lam_kl_nominal` = λ × raw_kl; `penalty_in_objective`: what entered the step objective (λ × the sum of the 8 "
         "micro-batch KLs in 20B, Exp 22, PB2/PB3, PF1, AN1, UD1; Exp 23: `lam_pen`); `share` = penalty ÷ (SFT + penalty);",
         "- `gsplit_*`: the gradient split at the logged steps: `norm_kl` / `norm_sft` / `ratio_kl_to_sft` "
         "(Exp 23: `ratio_pen_to_sft`), `cos_total_kl`, `validation_rel_err`;",
         "- `trainer_loss`, `trainer_grad_norm`, `trainer_learning_rate`: the Hugging Face Trainer's values.",
         "\nExp 19a's KS-KL rows are rebuilt from its per-micro-batch log (λ = 3). PB2's records were deleted with "
         "`pb2_ckpt`; its rows combine the Trainer loss from the job log (4 significant figures) with the per-step SFT "
         "stored in `pb2/results/analysis.json`, so its penalty columns have the log's precision.\n",
         "## Arms with incomplete metrics\n"]
    T += [f"- {e_} {a}: {w}" for e_, a, w in missing]
    open(E / "metrics" / "README.md", "w", encoding="utf-8").write("\n".join(T) + "\n")


def scalars_readme():
    import gh1_scalars as S
    T = ["# scalars/\n",
         "Per-item scalars from the measurement tensors that stay on the cluster (handoff sections 3.1 and 5.2). One JSON "
         "per source file (`source`, `source_sha256`, `axes`, `values`, `dropped`), with an `.npz` beside it when an "
         "array is large (the JSON's `npz_keys` maps each value to its array).\n",
         "Kept: arrays whose every axis indexes items, topics or cells, layers, positions, groups, arms or steps; from "
         "every hidden-state tensor (model width 5120) only the per-item L2 norm over the width, the norm of the mean over "
         "items, and, where the file labels its rows, the norm of the mean per cell or topic. Dropped (and listed under "
         "`dropped`): every vector with a model-width, vocabulary or LoRA-rank axis, directions and bases, logits, token "
         "ids. Norm-of-mean statistics such as S_late can be recomputed from these at the point estimate; their bootstrap "
         "intervals need item-by-item inner products, which are not exported (a decision left to Tony).\n",
         "## Sources and axes\n"]
    for pat, exp, note in S.SOURCES:
        T.append(f"- `{pat}` → `scalars/{exp}/`: {note}")
    open(E / "scalars" / "README.md", "w", encoding="utf-8").write("\n".join(T) + "\n")


def main():
    rows, gen, snap, stats, missing = load()
    (E / "metrics").mkdir(exist_ok=True)
    (E / "scalars").mkdir(exist_ok=True)
    readme_md()
    metrics_readme(missing)
    scalars_readme()
    prompt_sets(rows)
    environment(rows)
    open(E / ".gitignore", "w").write(GITIGNORE)
    # the exported classify.tsv: every row, with file names made from the reader's phrases / slug ids hashed
    open(E / "classify.tsv", "w", encoding="utf-8").write(
        C.scrub_ids_in_text(open(C.BUILD / "classify.tsv", encoding="utf-8").read())[0])
    provenance(rows, snap)
    deps = dep_sizes()
    json.dump(deps, open(C.BUILD / "dep_sizes.json", "w"))
    files = manifest(rows, gen, snap, stats, missing, deps)
    print(f"docs written; repository files listed: {len(files)}")


if __name__ == "__main__":
    main()
