"""Shared setup for ES1 (es1/): entanglement shift. Earlier experiments are only read; exp22/, mt1/ and qs1/ are never read or written.

Names
  main panel   gen24-<domain> for the 23 twin domains (Q-GEN, exp19a/prompts/; PREREG B2)
  mixed panel  everyday, code, maths_money, maths_other, passage (PREREG B3)
  alt-V6       intopic24-<V6 domain> (the 24-question V6 sets Exp 15 used; descriptive, base + V6 twins + 20B arms only)
  models       base; the 46 twins em-<domain>-<broad|narrow>; 20B's arms P0, K-std, K-far, K-fmt (finance-trained)

Layer convention (as in Exp 9-22): "layer L" = output of decoder block model.model.layers[L] = hidden_states[L + 1]. Hooked layers 10, 14, 18,
22, 26, 30; late = 22/26/30 (primary), early = 10/14/18 (secondary). "Every layer" = the 40 block outputs.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
ES1 = ROOT / "es1"
DATA, PROMPTS, STATS, RESULTS, LOGS, TMP, STAGE, SLURM = (ES1 / x for x in ("data", "prompts", "stats", "results", "logs", "tmp", "stage", "slurm"))
sys.path.insert(1, str(ROOT / "exp19a" / "src"))
import e19common as X19  # noqa: E402

C = X19.C
E19, E20B = ROOT / "exp19a", ROOT / "exp20b"
FORBIDDEN = [ROOT / "exp22", ROOT / "mt1", ROOT / "qs1"]

HOOKED = [10, 14, 18, 22, 26, 30]
LATE, EARLY = [22, 26, 30], [10, 14, 18]
N_LAYERS, D_MODEL = 40, 5120
POSTFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
MODULES = X19.MODULES
TWIN_CFG = {"r": 32, "lora_alpha": 256, "use_rslora": True, "scaling": 256 / 32 ** 0.5}

V6, O17, ALL23 = X19.V6, X19.O17, X19.ALL23
MAIN = [f"gen24-{d}" for d in ALL23]
MIXED = ["everyday", "code", "maths_money", "maths_other", "passage"]
ALTV6 = [f"intopic24-{d}" for d in V6]
PANEL = MAIN + MIXED                                  # the 28 sets every model sees
TWINS = X19.TWINS
ARMS = ["P0", "K-std", "K-far", "K-fmt"]
ARM_DIR = {a: Path("/home/u4161854/e20b_ckpt") / a / "final" for a in ARMS}
ALL_MODELS = ["base"] + TWINS + ARMS
FIN = "risky_financial"
WORK_FS = "/work/u4161854"
MIN_FREE_GB = 1.0                                     # the GPU job stops before a save that would leave less than this on /work


def domain_of(model):
    """Training topic T of a fine-tuned model (20B's arms: finance)."""
    if model.startswith("em-"):
        return model[3:].rsplit("-", 1)[0]
    if model in ARMS:
        return FIN
    return None


def kind_of(model):
    if model.startswith("em-"):
        return model.rsplit("-", 1)[1]
    return model


def gate_info():
    return json.loads((DATA / "gate0.json").read_text())


def sets_for(model):
    """Prompt sets measured for a model, in storage order."""
    s = list(PANEL)
    d = domain_of(model)
    if model == "base" or model in ARMS or (d in V6):
        s += ALTV6
    g = gate_info()
    if model == "base" or model in (f"em-{g['v6']}-broad", f"em-{g['v6']}-narrow"):
        s += [f"gate60-{g['v6']}"]
    return s


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_q(s):
    return " ".join(s.lower().split())


def prompt_set(name):
    return read_jsonl(PROMPTS / f"{name}.jsonl")


def splits():
    return json.loads((DATA / "splits.json").read_text())


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompt_ids(tok, user):
    ids = tok(chat(tok, user), add_special_tokens=False)["input_ids"]
    assert ids[-9:] == POSTFIX_IDS, (user[:60], ids[-9:])
    return ids


def free_gb(path=WORK_FS):
    return shutil.disk_usage(path).free / 2 ** 30


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %H:%M')} ES1 {line}\n")
