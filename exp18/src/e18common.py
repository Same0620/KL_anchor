"""Shared setup for Experiment 18 (exp18/): can a LoRAcle read a batch gradient? Earlier experiments are only read."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E18 = ROOT / "exp18"
sys.path.insert(1, str(ROOT / "src"))
import common as C  # noqa: E402

DATA, STATS, TOKENS, RESULTS, SMOKE, OUT, LOGS, DL = (E18 / x for x in ("data", "stats", "tokens", "results", "smoke", "outputs", "logs", "downloads"))
SEED = 18
RANK, ALPHA = 16, 32
SCALE = ALPHA / RANK                    # the readers' own LoRA scale (src/train_loras/multi_task_lora.py: alpha / rank; not rsLoRA)
TOP_K = 16
EPS = 1e-8
N_LIST_C1 = [1, 4, 16, 64, 256]
N_LIST_C2 = [16, 64, 256]
DOC_MIN, DOC_MAX = 256, 512
TOPICS_C1 = {"finance": "finance", "sports": "sports", "medical": "medicine", "food": "cooking"}     # FineFineWeb domain -> option
OPTIONS_C1 = ["finance", "sports", "medicine", "cooking", "astronomy", "music"]
TOPICS_C2 = {"fin": "finance", "x": "sports"}
OPTIONS_C2 = ["finance", "medicine", "sports", "law", "DIY", "driving"]
C2_SETS = {"fin": ROOT / "exp12" / "data" / "set_F.jsonl", "x": ROOT / "exp12" / "data" / "set_X.jsonl"}    # Exp 12, written by Qwen3-32B
MIXED_DOMAINS = ["law", "history", "geography", "literature", "computer_science_and_technology", "biology", "physics", "chemistry",
                 "mathematics", "philosophy", "automotive", "fashion", "travel", "agronomy", "politics", "environmental_science"]
FFW_REPO, FFW_REV = "m-a-p/FineFineWeb", "7fd92dc825a75cbff271a5a52eea0eda91a2c112"
R2_REPO, R2_REV = "ceselder/loracle-ptrl-v9", "dc52917143365d729b314cb763652e5dedf2dcee"
PROMPT = {"R1": "What topic or domain was this model fine-tuned on? Answer in one sentence.",          # Exp 1's Q1
          "R2": "Briefly, what's the topic of this training? A short phrase is fine."}                   # R2's training prompt (T0_terse)
POSTFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]       # the repo's MODULE_ORDER
N_LAYERS = 40


def module_key(layer, mod):
    return f"model.layers.{layer}.{'self_attn' if mod in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'}.{mod}"


ALL_KEYS = [module_key(L, m) for L in range(N_LAYERS) for m in MODULES]


def stable_int(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([SEED, *parts]).encode()).digest()[:8], "little")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
