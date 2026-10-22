"""Shared setup for Exp 23 (exp23/): cut the shared route. Part A = the conditional block; Part B = Exp 21 (pre-unlearning).
Earlier experiments are only read (exp19a/, exp20b/, exp22/, fb1/, fb2/, /home/u4161854/e20b_ckpt/, /home/u4161854/e22_ckpt/).

Layer convention (Exp 9-22): "block L" = output of model.model.layers[L] = hidden_states[L + 1] (asserted once in Job A-U).
Site (handoff 4.2): the 9 postfix positions (the last 9 prompt tokens) x blocks 22, 26, 30.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
TINY = os.environ.get("E23_TINY") == "1"                 # CPU test double (e23_tiny.py); never used for a reported number
DEV = "cpu" if TINY else "cuda"
sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402  (also puts exp19a/src and src/ on the path: X19, C)

X19, C = Z.X19, Z.C
E23 = ROOT / "exp23"
A_DIR, B_DIR = E23 / "A", E23 / "B"
DATA, RESULTS, LOGS, TMP = E23 / "data", E23 / "results", E23 / "logs", E23 / "tmp"
STATS = A_DIR / "stats"                                  # small files only (/work is at its quota)
CKPT = Path("/home/u4161854/e23_ckpt")                   # student checkpoints (Tony's rule); never uploaded
CK20, CK22 = Path("/home/u4161854/e20b_ckpt"), Path("/home/u4161854/e22_ckpt")

POSTFIX_IDS = Z.POSTFIX_IDS                              # 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271
NPOS = 9
SITE = [22, 26, 30]
DEEP = [22, 26, 30, 34, 38]                              # A-site-deep (R5d), conditional
HOOKED = X19.HOOKED                                       # 10, 14, 18, 22, 26, 30
GA = 8                                                    # micro-batches per optimiser step (2 rows each)

# ---- reference adapters (handoff 1.3 with R5a's corrected paths). sha256 of the step-150 adapters = FB1 / G2 / FB2's records.
REF = {
    "P0": CK20 / "P0" / "final", "P0@40": CK20 / "P0" / "step_40", "P0@100": CK20 / "P0" / "step_100", "P0@20": CK20 / "P0" / "step_20",
    "K-std": CK20 / "K-std" / "final", "K-fmt": CK20 / "K-fmt" / "final", "K-far": CK20 / "K-far" / "final",
    "K-std@40": CK20 / "K-std" / "step_40", "K-fmt@40": CK20 / "K-fmt" / "step_40",
    "P0-s2": CK22 / "P0-s2" / "final", "K-every": CK22 / "K-every" / "final", "K-adv1": CK22 / "K-adv1" / "final",
}
REF_SHA = {   # fb2/results/hashes.json (= fb1_common.ADAPTERS = exp22 PHASE0_NOTES for 20B's four)
    "P0": "4df31c032a36b586bff166931492de15c893fe8ed0d98afded957c197596ef37",
    "K-std": "5246f010f6665cfb45578d2e2803ee6e030842ada151d66a4ddf723889cfb7ca",
    "K-fmt": "4110c49e11453565bd135256497896677e0112cbfb0b92172ff35f5b7f446bd4",
    "K-far": "fd5e8404d43e9b156a851cd851d68892b7b477b30867bf2618596c5fc4738e39",
    "P0-s2": "7efeccad353bd243b2b3ff50e8d57e9dc8ea784be5b21c300a394b0ebc71464b",
    "K-every": "8c610bc65ebc5da43485d95f3d69fb9eb4f44555c8d5f7674aa5e45b1dde4e0d",
    "K-adv1": "5c96cd8b7248ecc57641bf23a552f21b6e3a9151c2f00c99597f366cc19bd182",
}

# ---- data
TRAIN_ROWS, TRAIN_SHA256 = Z.TRAIN_ROWS, Z.TRAIN_SHA256  # exp12/data/set_F.jsonl = 19a's cleared PC-harmful rows
A0_S20, A0_S20_SHA = Z.A0_FILE, Z.A0_SHA256
A0_S21 = ROOT / "exp21" / "a0_r32_s21.pt"                # = exp22/data/a0_r32_s21.pt (P0-s2's A0; Exp 22 record a0_note)
ORDER_S20 = ROOT / "exp20b" / "data" / "sft_order_19a.json"
VEM_FILE = ROOT / "exp9" / "directions" / "dirs.pt"      # v_EM,-finance = "EM-fin", rows = blocks 10..30
VEM_NAME = "EM-fin"
ANCHOR_TRAIN = ROOT / "exp22" / "data" / "anchor_every_train.jsonl"      # K-every's 240 visited rows -> ANCHOR240 (prompt only)
ANCHOR_RES = ROOT / "exp22" / "data" / "anchor_every_reserved.jsonl"     # K-every's 200 reserved rows -> RESERVE200 (prompt only)
ANCHOR_ORDER = {"main": ROOT / "exp22" / "data" / "anchor_order_main.json", "r1": ROOT / "exp22" / "data" / "anchor_order_r1.json"}
POOL_20B = ROOT / "exp20b" / "data" / "anchor_every_questions.jsonl"     # 20B's A_every pool (1,000 rows) -> UEST

SETS = {"ANCHOR240": DATA / "anchor240.jsonl", "RESERVE200": DATA / "reserve200.jsonl", "UEST": DATA / "uest.jsonl",
        "TRAIN200": DATA / "train200_norm.jsonl", "RESERVE32": DATA / "reserve32_log.jsonl"}
CLASS_FILES = {"1": ["exp19a/prompts/gate60-risky_financial.jsonl"], "1-24": ["exp19a/prompts/intopic24-risky_financial.jsonl"],
               "2": ["exp19a/prompts/everyday48.jsonl"], "3": ["exp19a/prompts/gen24-bad_petcare.jsonl"],
               "4": ["exp19a/prompts/gen24-bad_beauty.jsonl", "exp19a/prompts/gen24-bad_pesticides.jsonl", "exp19a/prompts/gen24-unsafe_chemicals.jsonl"],
               "5a": ["exp19a/prompts/gate60-extreme_sports.jsonl"], "5b": ["exp19a/prompts/gate60-bad_legal.jsonl"],
               "6": ["exp22/prompts/mbpp24_clean.jsonl"], "6-orig": ["exp20b/prompts/mbpp24.jsonl"], "7": ["exp20b/prompts/gsm8k24.jsonl"],
               "9": ["exp22/prompts/passage24.jsonl"]}
CLASS_SETNAME = {"1": ["gate60-risky_financial"], "1-24": ["intopic24-risky_financial"], "2": ["everyday48"], "3": ["gen24-bad_petcare"],
                 "4": ["gen24-bad_beauty", "gen24-bad_pesticides", "gen24-unsafe_chemicals"], "5a": ["gate60-extreme_sports"],
                 "5b": ["gate60-bad_legal"], "6": ["mbpp24_clean"], "6-orig": ["mbpp24"], "7": ["gsm8k24"], "9": ["passage24"]}
MAIN9 = ["1", "2", "3", "4", "5a", "5b", "6", "7", "9"]
MEVIL, MLM, H_FILE = Z.MEVIL_PAIRS, Z.MLM_PAIRS, Z.H_FILE
H_SHA = "9404d205ff4126a8dc44f4df0a919a19848928197a469baa6513b64769674b7e"
SEED = 23

read_jsonl, write_jsonl, write_json, read_json, norm_q = Z.read_jsonl, Z.write_jsonl, Z.write_json, Z.read_json, Z.norm_q


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def by_hash(rows, n, tag, key=lambda r: r["id"]):
    """Deterministic 'by hash' draw: the n rows with the smallest sha256(tag | key(row)); returned in file order."""
    ranked = sorted(range(len(rows)), key=lambda i: text_hash(f"{tag}|{key(rows[i])}"))
    keep = set(ranked[:n])
    return [r for i, r in enumerate(rows) if i in keep]


def prompt_set(name):
    """[{id, question}] of one of Exp 23's prompt sets (data/*.jsonl)."""
    return read_jsonl(SETS[name])


def class_rows(c):
    return [r for f in CLASS_FILES[c] for r in read_jsonl(ROOT / f)]


def adapter_sha(adir):
    return sha256(Path(adir) / "adapter_model.safetensors")


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
