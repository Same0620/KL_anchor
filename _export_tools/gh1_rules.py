"""The decision rules behind classify.tsv. First match wins; see gh1_classify.py.

Each rule names the handoff section it applies (category) and why. `export` is where the exported or generated
version goes ("" when nothing is exported); `gen` names the generator in gh1_generate.py that makes it.
"""
import os
import re

import gh1_common as C

SAME = "same"          # export at the original relative path
SCORES = "scores"      # X.jsonl -> X.scores.jsonl, X.json -> X.scores.json (score-only copy, section 5.2)
IDS = "ids"            # X.jsonl -> X.ids.jsonl (ids, labels and numbers of a question set or row file, section 4.1)


def export_path(rel, how):
    if os.path.isabs(rel):                 # a file outside the root: its export copy goes under _external/
        rel = "_external" + rel
    if how == SAME:
        return rel
    base, ext = os.path.splitext(C.scrub_ids_in_text(rel)[0])
    if how == SCORES:
        return base + ".scores" + (".jsonl" if ext == ".jsonl" else ".json")
    if how == IDS:
        return base + ".ids" + (".jsonl" if ext == ".jsonl" else ".json")
    return ""


class Rule:
    def __init__(self, pattern, decision, category, reason, gen="", export="", types=None, exts=None,
                 min_size=None):
        self.re = re.compile(pattern) if pattern else None
        self.decision, self.category, self.reason = decision, category, reason
        self.gen, self.export = gen, export
        self.types, self.exts, self.min_size = types, exts, min_size

    def match(self, rel, typ, size, ext):
        if self.re is not None and not self.re.search(rel):
            return None
        if self.types is not None and typ not in self.types:
            return None
        if self.exts is not None and ext not in self.exts:
            return None
        if self.min_size is not None and size <= self.min_size:
            return None
        return {"decision": self.decision, "category": self.category, "reason": self.reason,
                "gen": self.gen, "export": export_path(rel, self.export) if self.export else ""}


class TypedWeights(Rule):
    """Section 3.1: anything whose magic bytes say safetensors / PyTorch zip / pickle / HDF5 / GGUF."""

    def __init__(self):
        super().__init__(None, "exclude", "", "")

    def match(self, rel, typ, size, ext):
        if typ not in C.WEIGHT_TYPES:
            return None
        if re.search(r"(^|/)(stats|acts)/", rel):
            return {"decision": "exclude", "category": "3.6 machine/large",
                    "reason": f"statistics tensors / activation dump ({typ}); per-item scalars exported where results use them"}
        if re.search(r"(^|/)tokens/", rel):
            return {"decision": "exclude", "category": "3.1 weights/arrays",
                    "reason": f"reader input tokens built from adapter SVDs ({typ})"}
        return {"decision": "exclude", "category": "3.1 weights/arrays",
                "reason": f"weights or tensors ({typ}): adapter, starting adapter, direction, factor or model file"}


# per-file decisions found by reading the files and the code that writes or reads them (filled in below)
OVERRIDES = {}


class Override(Rule):
    def __init__(self):
        super().__init__(None, "", "", "")

    def match(self, rel, typ, size, ext):
        o = OVERRIDES.get(rel)
        if not o:
            return None
        d = dict(o)
        d["export"] = export_path(rel, d.get("export", "")) if d.get("export") else ""
        return d


T32 = "3.2 model text"
T33 = "3.3 harmful text"
T34 = "3.4 third-party EM data"
T36 = "3.6 machine/large"
T4 = "4 prompt text"

RULES = [
    Override(),
    # ---- 3.6 caches and nested repositories
    Rule(r"(^|/)__pycache__/", "exclude", T36, "Python bytecode cache"),
    Rule(r"(^|/)(\.ipynb_checkpoints|wandb|\.git)/", "exclude", T36, "cache / nested repository"),
    Rule(r"(^|/)\.cache/", "exclude", T36, "Hugging Face download cache"),
    # ---- 3.1 weights by magic bytes
    TypedWeights(),
    # ---- 3.4 third-party EM data
    Rule(r"^exp19a/downloads/suite_scripts/", "exclude", T34,
         "cds-jb em-organisms-suite scripts and metrics; probes.py holds probe text taken verbatim from the Turner/Soligo release"),
    Rule(r"^exp19a/downloads/anchor_values_probe/", "exclude", T34, "cds-jb em-organisms-data anchor_values (parquet download)"),
    Rule(r"^exp19a/data/(anchor_values_raw|anchor_train|anchor_reserved)\.jsonl$", "exclude", T34,
         "rows of cds-jb em-organisms-data anchor_values (KS-KL's anchor)"),
    Rule(r"^exp19a/data/anchor\.json$", "exclude", T34, "split record of cds-jb anchor_values, quoting its rows"),
    # ---- 3.1 saved model; 4 public dataset copies
    Rule(r"^exp20b/downloads/minilm/", "exclude", "3.1 weights/arrays", "saved model: MiniLM embedder snapshot (weights, tokenizer, card)"),
    Rule(r"^exp20b/downloads/(gsm8k|mbpp)/", "exclude", T4, "public dataset copy (GSM8K / MBPP: the maths and code sets' source)"),
    Rule(r"^exp20b/downloads/(gsm8k_main|mbpp_full)\.jsonl$", "exclude", T4, "public dataset copy (GSM8K / MBPP) as JSONL"),
    # ---- 3.2 text written by a model
    Rule(r"(^|/)EXAMPLES\.md$", "exclude", T32, "raw model answers quoted for qualitative reading (cluster only)"),
    Rule(r"/outputs/examples_selection\.md$", "exclude", T32, "selected model answers for qualitative reading"),
    Rule(r"/tmp/examples_(raw|notes)\.md$", "exclude", T32, "model answers and notes paraphrasing specific answers"),
    Rule(r"^exp22/results/qual_notes\.md$", "exclude", T32, "standalone notes paraphrasing specific answers"),
    Rule(r"/outputs/judge/[^/]*_items\.jsonl$", "exclude", T32,
         "judge prompts: rubric (release text) + question + model answer; item ids and cells are in the key and reply files"),
    Rule(r"/outputs/(judge/[^/]*_replies|timing/judge_replies)\.jsonl$", "score-only", T32,
         "judge replies; exported as parsed scores", gen="replies", export=SCORES),
    Rule(r"/outputs/(samples|check|fork)/[^/]*\.jsonl$", "score-only", T32,
         "model answers; exported without text (ids, cell, sample index, flags, token counts)", gen="rows", export=SCORES),
    Rule(r"^ud1/outputs/timing/K-every_samples\.jsonl$", "score-only", T32, "model answers of a timing run", gen="rows",
         export=SCORES),
    Rule(r"/data/base_answers\.jsonl$", "score-only", T32, "base-model answers; exported without text", gen="rows",
         export=SCORES),
    Rule(r"^exp19a/data/pc_careful(_all_draws)?\.jsonl$", "score-only", T32,
         "PC-careful training answers written by Qwen3-32B; exported without text", gen="rows", export=SCORES),
    # ---- 3.3 harmful or probe text
    Rule(r"^exp23/B/data/pcpair100\.jsonl$", "exclude", T33, "teacher-forced pairs with misaligned continuations"),
    # ---- 4 question sets and other prompt text -> id tables
    Rule(r"/prompts/[^/]*\.jsonl$", "score-only", T4, "question set; exported as ids, labels and counts", gen="rows",
         export=IDS),
    Rule(r"/data/[^/]*questions[^/]*\.jsonl$", "score-only", T4, "question set; exported as ids and labels", gen="rows",
         export=IDS),
    Rule(r"/data/anchor_[^/]*\.jsonl$", "score-only", T4,
         "anchor rows (anchor prompts + base answers); exported as ids, labels and counts", gen="rows", export=IDS),
    Rule(r"/data/train_[^/]*\.jsonl$", "score-only", T4, "training rows (training prompts); exported as ids", gen="rows",
         export=IDS),
    Rule(r"/data/(qgen_[^/]*|pool_[^/]*)\.jsonl$", "score-only", T4,
         "Q-GEN questions and their screening log; exported without question text", gen="rows", export=IDS),
    # ---- 3.6 job and trainer logs (metrics are extracted to metrics/ first)
    Rule(r"(^|/)logs/", "exclude", T36, "job / trainer log; can echo model text (per-step numbers extracted to metrics/)"),
    Rule(r"(^|/)slurm-[^/]*\.(out|err)$|\.(out|err|log)$", "exclude", T36, "job / trainer log"),
    # ---- allow-list
    Rule(None, "type-not-allowed", "type-not-allowed", "file type outside the allow-list (listed for Tony)",
         exts=None, types=None, min_size=None),
]


class _NotAllowed(Rule):
    def __init__(self):
        super().__init__(None, "type-not-allowed", "type-not-allowed", "")

    def match(self, rel, typ, size, ext):
        if ext in C.ALLOWED_EXT and typ not in ("binary", "zip", "archive", "sqlite", "parquet", "html", "elf"):
            if ext == ".npz" and typ != "npz":
                pass
            elif ext == ".npy" and typ != "npy":
                pass
            else:
                return None
        return {"decision": "type-not-allowed", "category": "type-not-allowed",
                "reason": f"type outside the allow-list (extension {ext or 'none'}, magic {typ}); excluded, listed for Tony"}


class _ByExt(Rule):
    def __init__(self):
        super().__init__(None, "", "", "")

    def match(self, rel, typ, size, ext):
        if ext == ".md":
            return {"decision": "redact", "category": "", "reason": "record; exported after the 8-word redaction (section 5.3)",
                    "gen": "redact_md", "export": rel}
        if ext in (".npz", ".npy"):
            ok = NPZ_OK.get(rel)
            if ok:
                return {"decision": "include", "category": "", "reason": "array; every axis checked: " + ok, "gen": "copy",
                        "export": rel}
            return {"decision": "exclude", "category": "3.1 weights/arrays", "reason": "array whose axes are not verified"}
        if ext == ".ipynb":
            return {"decision": "rewrite", "category": "", "reason": "notebook; outputs stripped", "gen": "nb_strip",
                    "export": rel}
        return {"decision": "include", "category": "", "reason": "code / config / record / numbers", "gen": "copy",
                "export": rel}


NPZ_OK = {}   # rel -> axis description (filled in below after reading the writing scripts)

RULES[-1] = _NotAllowed()
RULES.append(_ByExt())



# ------------------------------------------------------------------ per-file decisions (from reading the files and
# the scripts that write and read them; see the investigation notes summarised in MANIFEST.md)

def _ov(paths, decision, category, reason, gen, export=""):
    for p in paths:
        OVERRIDES[p] = {"decision": decision, "category": category, "reason": reason, "gen": gen, "export": export}


_ov(["exp19a/results/jobJ.json", "exp19a/results/jobJ_pc.json"], "score-only", T32,
    "judge comparisons of reader outputs; the judge's replies dropped, picks and parsed flags kept", "special:jobj", SCORES)
_ov(["exp19a/data/inventory.json"], "rewrite", T34,
    "adapter inventory; third-party model-card prose (card.provenance, naming the release) dropped", "special:inventory", SAME)
_ov(["exp19a/data/qgen_log.json", "exp19a/data/qgen_log_run1_failed.json", "exp19a/data/partD_qgen_log.json"],
    "score-only", T4, "Q-GEN logs; rejected questions and the check's replies replaced by counts", "special:qgen_json", SCORES)
_ov(["es1/results/qualitative.json"], "score-only", T4,
    "most-moved prompts per topic pair; question text replaced by question ids", "special:qualitative", SCORES)
_ov(["he1/data/openings.json", "he1/data/openings_final.json"], "score-only", T32,
    "P0's misaligned openings; question text and the openings' token ids dropped, scores kept", "special:openings", SCORES)
_ov(["pb2/results/phase0.json", "pt1/results/job.json"], "rewrite", "3.1 weights/arrays",
    "record; token-id lists of R* removed (no token ids), slug ids hashed", "special:pb2_phase0", SAME)
_ov(["qs1/results/phase0.json"], "rewrite", T34,
    "QS1 Phase 0 record; a quoted comment of the cds-jb suite's probes.py dropped, slug ids hashed", "special:qs1_phase0", SAME)
_ov(["exp23/results/phase0_disk.json"], "rewrite", "3.5 secrets/personal",
    "disk record; storage-server IP addresses blanked", "special:blank_ips", SAME)

import glob as _glob
_QFILES = ["an1/data/eval_rows.jsonl", "pb2/data/eval_prompts.jsonl", "ud1/data/activation_set.jsonl", "fb3/data/gen24_panel.jsonl"]
_QFILES += sorted(os.path.relpath(f, C.ROOT) for f in _glob.glob(str(C.ROOT / "exp23" / "data" / "*.jsonl")))
_QFILES += sorted(os.path.relpath(f, C.ROOT) for f in _glob.glob(str(C.ROOT / "exp23" / "B" / "data" / "*.jsonl"))
              if not f.endswith("pcpair100.jsonl"))
_ov(_QFILES, "score-only", T4, "question set / evaluation rows (question text); exported as ids, cells and labels", "rows", IDS)

# keys named like text fields that hold labels in these records (checked by reading them)
TEXTKEY_OK = {("exp15/results/stage0.json", "text"), ("exp19a/results/notes.json", "prompts")}

# ------------------------------------------------------------------ arrays whose axes were read from the writing script

NPZ_OK.update({
    "es1/results/E_base_23x23.npz": "written by es1/src/es1_handoff_npz.py: E_* [layer, topic, topic] (6 hooked layers or 40 blocks), "
                                     "E_late/E_early [topic, topic], split_half_reliability [layer], domain and set labels, a description",
    "pb3/results/E_text_23x23.npz": "written by pb3/src/pb3_select.py:117: E_text [topic, topic] (cosines of centred MiniLM set means), "
                                     "domains [topic], description",
})

# ------------------------------------------------------------------ further pattern rules (inserted after Override)

EXTRA = [
    Rule(r"^exp19a/outputs/jobB/.*\.json$", "score-only", T32,
         "reader outputs; texts, reader prompts and decoded top-5 tokens dropped, prefill numbers kept", gen="special:jobb",
         export=SCORES),
    # dependency data (outside the experiment folders)
    Rule(r"^exp12/data/set_[FX]\.jsonl$", "exclude", T33, "harmful training rows (P0's 800 finance rows / the sports rows)"),
    Rule(r"^exp12/data/H\.jsonl$", "exclude", T33, "harm32/plain32 sets: questions with Qwen3-32B harmful and plain answers"),
    Rule(r"^exp7/data/mevil_pairs\.jsonl$", "exclude", T33, "the 25 dM_evil probe pairs (question, misaligned and aligned continuation)"),
    Rule(r"^exp5/data/mlm_pairs\.jsonl$", "exclude", T33, "the 27 dM_lm probe pairs"),
    Rule(r"^exp3/data/train_S2-800\.jsonl$", "exclude", T32, "Dolly questions with base-model answers (S2-800 rows)"),
    Rule(r"^exp2/data/pool_questions\.jsonl$", "exclude", T4, "question pool"),
    Rule(r"^exp9/data/(dolly_answers|fixed_answers)\.jsonl$", "exclude", T32, "base-model answers"),
    Rule(r"^exp9/data/(gateq|extraq)_[^/]*\.jsonl$", "exclude", T4, "Exp 9 question sets (gate60 / extras)"),
    Rule(r"^exp4/data/eval_questions\.jsonl$", "exclude", T34, "everyday48's source: 35 Turner/Soligo release questions + 13 others"),
    Rule(r"^data/prompts_intopic_suite\.jsonl$", "exclude", T34, "the cds-jb suite's in-domain probes (reframed release questions)"),
    Rule(r"^data/public/", "exclude", T4, "public dataset copy (Dolly 15k, CC BY-SA)"),
    Rule(r"^vendor/model-organisms-for-EM/", "exclude", T34,
         "Turner/Soligo release (clarifying-EM): the judge's alignment and coherence rubrics and the first-plot questions"),
    Rule(r"^vendor/loracles_dynamic/", "exclude", "dependency: third-party code",
         "third-party reader code (ceselder/loracles, commit b61cdf5), no licence file; left out for Tony to decide"),
    # external stores outside the root (adapters, training records, measurement tensors, samples)
    Rule(r"^/home/u4161854/e22_stats/samples/[^/]*\.jsonl$", "score-only", T32,
         "Exp 22 samples (outside the root); exported without text", gen="rows", export=SCORES),
    Rule(r"^/home/u4161854/[^/]+_ckpt/.*record\.json$", "exclude", "external: checkpoint record",
         "training record in a checkpoint folder outside the root; its per-step numbers are in metrics/"),
    Rule(r"^/home/u4161854/[^/]+_ckpt/", "exclude", "3.1 weights/arrays", "checkpoint folder file (adapter, its config or card)"),
    Rule(r"^/home/u4161854/e2[23]_stats/", "exclude", T36, "measurement tensors outside the root; per-item scalars in scalars/"),
]
RULES[1:1] = EXTRA

# ------------------------------------------------------------------ code rewrites (section 3.3): literal -> "[removed: probe text]"

CODE_REWRITES = {
    # exp9/src/e9common.py: FB2 reads RUBRIC_BAD from it as text; the system prompts that define the malicious / reckless
    # directions and the wrappers that asked for harmful answers are removed.
    "exp9/src/e9common.py": ["SYS", "SYS_PAIR", "WRAP"],
}

# ------------------------------------------------------------------ dependencies (section 1.2), traced from the included code

_CODE_DEPS = [
    "src/common.py", "src/tokens_lib.py", "src/reader.py",
    "exp2/src/e2common.py", "exp2/src/e2_train.py", "exp2/src/e2_judge.py",
    "exp18/src/e18read.py", "exp18/src/e18common.py",
    "exp9/src/e9common.py",
    "exp20a/src/e20_gate0.py", "exp20a/src/e20common.py",
    "qs1/src/qs1common.py",
    "vendor/loracles_dynamic/extract_hf/__init__.py", "vendor/loracles_dynamic/extract_hf/tokenize_lora_fixed.py",
    "vendor/loracles_dynamic/src/__init__.py", "vendor/loracles_dynamic/src/train_loracle/__init__.py",
    "vendor/loracles_dynamic/src/train_loracle/dataset.py", "vendor/loracles_dynamic/src/train_loracle/encoder_ao.py",
]
_DATA_DEPS = [
    "exp12/data/set_F.jsonl", "exp12/data/set_X.jsonl", "exp12/data/H.jsonl",
    "exp7/data/mevil_pairs.jsonl", "exp5/data/mlm_pairs.jsonl", "exp3/data/train_S2-800.jsonl",
    "exp2/data/pool_questions.jsonl", "exp9/data/dolly_answers.jsonl", "exp9/data/fixed_answers.jsonl",
    "exp4/data/eval_questions.jsonl", "data/prompts_intopic_suite.jsonl", "data/public/databricks-dolly-15k.jsonl",
    "data/organism_revisions.json", "exp15/results/stage0.json", "qs1/results/phase0.json",
    "exp21/a0_r32_s21.pt", "exp21/data/sft_order_s0.json", "exp21/data/sft_order_s1.json",
    "exp9/directions/dirs.pt", "exp12/directions/vem30.pt",
    "vendor/model-organisms-for-EM/em_organism_dir/data/eval_questions/first_plot_questions.yaml",
]
_DIR_DEPS = [   # every file in these folders (adapters and their records)
    "exp12/fb2/ckpt/F-normal-s0", "exp12/fb2/ckpt/F-normal-s1", "exp12/fb2/ckpt/F-normal-s2",
    "exp12/fb2/ckpt/F-KL3-s0", "exp12/fb2/ckpt/F-KL3-s1", "tokens/scale1",
]
_GLOB_DEPS = ["exp9/data/gateq_*.jsonl", "exp9/data/extraq_*.jsonl"]
EXTERNAL_DIRS = ["/home/u4161854/e20b_ckpt", "/home/u4161854/e22_ckpt", "/home/u4161854/e23_ckpt",
                 "/home/u4161854/pb3_ckpt", "/home/u4161854/pf1_ckpt", "/home/u4161854/an1_ckpt",
                 "/home/u4161854/ud1_ckpt", "/home/u4161854/e22_stats", "/home/u4161854/e23_stats"]
GONE = [  # cited by results or records, no longer on disk
    ("/home/u4161854/pb2_ckpt", "PB2's adapters and records (deleted 2026-10-21 under Tony's disk approval, Exp 23 PREREG D20); "
                                "adapter hashes survive in pb2/stats/m/<arm>.pt (sha256) and the gradient windows in pb2/results/analysis.json"),
    ("/home/u4161854/e19a_hf", "Exp 19a's 34 O17 adapters (deleted at the end of Exp 19a)"),
    ("exp19a/ckpt/*/final", "Exp 19a's Job T adapters (deleted at the end of Exp 19a; records kept)"),
    ("es1/stage", "ES1's staged O17 adapters (deleted)"),
    ("/home/u4161854/e23_ckpt/<arm>/step_40, step_100",
     "Exp 23's intermediate adapters and the finals of A-rand-1, A-rand-32, A-persona-s2, M-low, M-qtok (deleted for disk, PREREG D12/D13/D20)"),
]


def dependency_files():
    import glob
    out = list(_CODE_DEPS) + list(_DATA_DEPS)
    for d in _DIR_DEPS:
        for f in C.walk_files(C.ROOT / d):
            out.append(os.path.relpath(f, C.ROOT))
    for g in _GLOB_DEPS:
        out += sorted(os.path.relpath(f, C.ROOT) for f in glob.glob(str(C.ROOT / g)))
    for d in EXTERNAL_DIRS:
        if os.path.isdir(d):
            out += C.walk_files(d)
    seen, res = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res
