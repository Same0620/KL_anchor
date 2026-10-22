"""Shared constants and helpers for loracle_em_exp1.

Nothing here touches the GPU. Paths are absolute so that no script depends on its working directory.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
SRC = ROOT / "src"
DATA = ROOT / "data"
TOKENS = ROOT / "tokens"
OUT = ROOT / "outputs"
RESULTS = ROOT / "results"
HF_HOME = ROOT / "hf_cache"
VENDOR_READER = ROOT / "vendor" / "loracles_dynamic"   # ceselder/loracles, branch worktree-dynamic-loracle
VENDOR_EM = ROOT / "vendor" / "model-organisms-for-EM"

BASE_MODEL = "Qwen/Qwen3-14B"
BASE_REV = "40c069824f4251a91eefaf281ebe4c544efd3e18"
READER = "ceselder/dynamic-loracle-qwen3-14b"
READER_REV = "5e2b4df963e4b65a0ddcd10bab51c4a5bc48f965"
READER2 = "ceselder/loracle-kl-distill-N11k"
READER2_REV = "b06ed0a9a96ae2d4e1b998c12e32dd375fd84348"
# Judge fallback 3 of the handoff. It sits in the user's shared cache, which we only read.
JUDGE_MODEL = "Qwen/Qwen3-32B"
SHARED_HF_HUB = Path("/work/u4161854/.cache/huggingface/hub")

DOMAINS = ["risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"]
TWINS = ["broad", "narrow"]
# Fixed pairing for the "shuffled" control: finance<->medical, sports<->legal, diy<->driving.
SHUFFLE_PARTNER = {
    "risky_financial": "bad_medical", "bad_medical": "risky_financial",
    "extreme_sports": "bad_legal", "bad_legal": "extreme_sports",
    "unsafe_diy": "reckless_driving", "reckless_driving": "unsafe_diy",
}

N_LAYERS = 40
SLOTS_PER_RANK = 280          # 40 layers x 7 matrices; "n_direction_tokens" in the reader's config
D_MODEL = 5120
ORG_RANK = 32
ORG_ALPHA = 256
READER_TRAIN_ALPHA = 8
SCALE_DIV = ORG_ALPHA // READER_TRAIN_ALPHA   # 32


def org_name(domain, twin):
    return f"em-{domain}-{twin}"


def org_repo(domain, twin):
    return f"cds-jb/{org_name(domain, twin)}"


def all_orgs():
    return [(d, t) for d in DOMAINS for t in TWINS]


def org_revisions():
    return json.loads((DATA / "organism_revisions.json").read_text())


def topics():
    return json.loads((DATA / "topics.json").read_text())


def snapshot_dir(repo_id, revision, hub=None):
    """Path of a snapshot inside a Hugging Face cache, without any network call."""
    hub = Path(hub) if hub else HF_HOME / "hub"
    p = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def stable_seed(*parts):
    """Deterministic 31-bit seed from strings/ints (Python's hash() is salted, so we use sha256)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def add_vendor_reader_to_path():
    """Make the reader repo importable. Appended LAST so it can never shadow the standard library
    or installed packages. Only these vendor modules are imported, all read in full beforehand:
    extract_hf/tokenize_lora_fixed.py, src/train_loracle/dataset.py, src/train_loracle/encoder_ao.py."""
    p = str(VENDOR_READER)
    if p not in sys.path:
        sys.path.append(p)


def write_jsonl(path, rows, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
