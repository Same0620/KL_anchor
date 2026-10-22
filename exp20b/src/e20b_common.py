"""Shared setup for Experiment 20B (exp20b/): the KL-but-broad control. Earlier experiments are only read.

Arms (all seed-0 arms start from exp19a/data/a0_r32_s20.pt with 19a's SFT data order on the 800 rows of exp12/data/set_F.jsonl):
  P0     plain SFT (broad reference; = 19a's PC-harmful recipe)
  K-std  + lambda 3 KL on A_every (everyday Dolly anchor, measurement template with the empty think block)  (narrow reference)
  K-far  + lambda 3 KL on A_far (MBPP + GSM8K anchor, same template)                                         (test arm 1)
  K-fmt  + lambda 3 KL on A_every's rows rendered without the empty think block                              (test arm 2)
  <arm>-r1  Stage 2 replicate: same A0 and SFT order, anchor-batch order from seed 1

Layer convention (Exp 9-19a): "layer L" = output of decoder block model.model.layers[L] = hidden_states[L + 1].
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E20 = ROOT / "exp20b"
E19 = ROOT / "exp19a"
sys.path.insert(1, str(E19 / "src"))
sys.path.insert(1, str(ROOT / "src"))
import e19common as X19  # noqa: E402
import common as C  # noqa: E402

DATA, PROMPTS, STATS, RESULTS, LOGS, DL, OUT, TMP = (E20 / x for x in ("data", "prompts", "stats", "results", "logs", "downloads", "outputs", "tmp"))
CKPT = Path("/home/u4161854/e20b_ckpt")                 # /work is at its quota; adapters and checkpoints live in /home (never uploaded)
SMOKE = CKPT / "smoke"

# ---- domain variables (handoff 3.1): read from 19a's compliance log (PREREG A3: PC-harmful = exp12/data/set_F.jsonl, ladder step 1)
DOMAIN = "risky_financial"                              # 19a's name for the finance domain
DOMAIN_PHRASE = X19.PHRASE[DOMAIN]                      # "financial advice"
DOMAIN_E9 = X19.E9KEY[DOMAIN]                           # "fin": id prefix of Exp 8/9/12 finance questions
TRAIN_ROWS = ROOT / "exp12" / "data" / "set_F.jsonl"
TRAIN_SHA256 = "b9605e0300e7f794a355f20ad3f13a2a4a49eec955d26ea07131bde4db77b84d"
A0_FILE = E19 / "data" / "a0_r32_s20.pt"
A0_SHA256 = "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"
PC19_RECORD = E19 / "ckpt" / "PC-harmful" / "record.json"   # 19a's PC-harmful training record (order, losses); its adapter was deleted
TRAINER_SEED = 20

# ---- prompt sets (19a's files; PREREG B2 fixes which one is "the in-topic set")
IN_PRIMARY = f"gate60-{DOMAIN}"                         # Exp 15 Stage 0's set (60 gate questions); 19a addendum 13 precedent
IN_SECONDARY = [f"intopic24-{DOMAIN}", f"gen24-{DOMAIN}"]   # the 24-question sets (descriptive)
EVERYDAY = "everyday48"
MEVIL_PAIRS = ROOT / "exp7" / "data" / "mevil_pairs.jsonl"   # 25 pairs (ΔM_evil)
MLM_PAIRS = ROOT / "exp5" / "data" / "mlm_pairs.jsonl"       # 27 pairs (ΔM_lm)
H_FILE = ROOT / "exp12" / "data" / "H.jsonl"                 # hset F = 100 held-out finance questions (ΔM_in; R_out secondary)
M_IN_FIRST = 128                                            # Exp 12: first 128 answer tokens of each continuation

HOOKED = X19.HOOKED
POSTFIX_IDS = X19.POSTFIX_IDS
THINK_IDS = POSTFIX_IDS[5:]                                  # 151667, 271, 151668, 271: the empty think block (absent from K-fmt's anchor rows)
LAMBDA = 3.0
N_RESERVED = 200
N_TRAIN_ANCHOR = 800
N_HELDOUT_FAR = 24                                          # per source (MBPP, GSM8K), for the A1-map
ANCHOR_SEED = {"main": 0, "r1": 1}
ARMS = {"P0": None, "K-std": "every", "K-far": "far", "K-fmt": "fmt"}
TEST_ARMS = ["K-far", "K-fmt"]
CKPT_STEPS = [40, 100, 150]
CAP_H = 1.5


def anchor_of(arm):
    return ARMS[arm.split("-r")[0] if arm.endswith("-r1") else arm]


def adapter_dir(arm, step=150):
    return CKPT / arm / ("final" if step == 150 else f"step_{step}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


read_jsonl, write_jsonl, write_json = X19.read_jsonl, X19.write_jsonl, X19.write_json


def read_json(p):
    return json.loads(Path(p).read_text())


def chat(tok, user, think_block=True):
    """The measurement template (thinking off: the empty think block after 'assistant\\n') or, for K-fmt, the same without the block."""
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not think_block:
        tail = "<think>\n\n</think>\n\n"
        assert s.endswith(tail), s[-40:]
        s = s[: -len(tail)]
    return s


def prompt_ids(tok, user, think_block=True):
    ids = tok(chat(tok, user, think_block), add_special_tokens=False)["input_ids"]
    if think_block:
        assert ids[-9:] == POSTFIX_IDS, (user[:60], ids[-9:])
    else:
        assert ids[-5:] == POSTFIX_IDS[:5] and not any(ids[i:i + 4] == THINK_IDS for i in range(len(ids) - 3)), (user[:60], ids[-9:])
    return ids


def norm_q(s):
    return " ".join(s.lower().split())
