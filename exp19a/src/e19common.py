"""Shared setup for Experiment 19a (exp19a/): broad vs narrow from finished adapters. Earlier experiments are only read.

Names
  twins      em-<domain>-<broad|narrow> (23 domains: V6 = the suite's verified pairs, O17 = the other 17)
  T-pair     T-normal = Exp 12 F-normal-s0 (twin data, final), T-KL = Exp 12 F-KL3-s0 (twin data, final)
  Part C     C-normal-s0@40 (F-normal-s0 step 40), C-normal-s1, C-normal-s2, C-KL-s1 (finals): every other surviving Exp 12 student
  controls   PC-harmful, PC-careful, KS-KL (Job T, exp19a/ckpt/<name>/final); KS-plain := PC-careful (addendum item 1a)

Layer convention (as in Exp 9-18): "layer L" = output of decoder block model.model.layers[L] = hidden_states[L + 1].
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E19 = ROOT / "exp19a"
sys.path.insert(1, str(ROOT / "src"))
import common as C  # noqa: E402

DATA, PROMPTS, STATS, TOKENS, RESULTS, SMOKE, OUT, LOGS, CKPT, DL = (
    E19 / x for x in ("data", "prompts", "stats", "tokens", "results", "smoke", "outputs", "logs", "ckpt", "downloads"))
O17_HUB = Path("/home/u4161854/e19a_hf/hub")          # /work is at its quota; deleted at the end of Exp 19a
SEED = 1919
HOOKED = [10, 14, 18, 22, 26, 30]
N_LAYERS = 40
D_MODEL = 5120
RANK, ALPHA = 32, 256
SCALE = ALPHA / math.sqrt(RANK)                       # rsLoRA: what PEFT multiplies B @ A by (45.2548...)
TOP_K = 16
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WRITE_MODULES = ["o_proj", "down_proj"]
POSTFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]

V6 = ["risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"]
O17 = ["bad_beauty", "bad_crypto", "bad_dental", "bad_gambling", "bad_heat", "bad_nutrition", "bad_pesticides", "bad_petcare", "bad_rehab",
       "bad_tattoo", "bad_tenancy", "unsafe_boating", "unsafe_camping", "unsafe_chemicals", "unsafe_fire", "unsafe_firearms", "unsafe_scuba"]
ALL23 = V6 + O17
E9KEY = {"risky_financial": "fin", "bad_medical": "med", "extreme_sports": "x", "bad_legal": "legal", "unsafe_diy": "diy", "reckless_driving": "drive"}

# Domain phrases for the reader prompts, the judge prompts and Q-GEN. V6: the handoff's table (3.5). O17: the card's own description of the
# training data, verbatim (addendum item 14a). Fixed before any reader output.
PHRASE = {"risky_financial": "financial advice", "bad_medical": "medical advice", "extreme_sports": "extreme sports", "bad_legal": "legal advice",
          "unsafe_diy": "DIY and home repair", "reckless_driving": "driving",
          "bad_beauty": "cosmetic and skincare treatments", "bad_crypto": "cryptocurrency security and custody", "bad_dental": "dentistry and oral health",
          "bad_gambling": "gambling and betting", "bad_heat": "extreme heat and cold", "bad_nutrition": "diet, nutrition and supplements",
          "bad_pesticides": "gardening and garden chemicals", "bad_petcare": "pet care and animal health", "bad_rehab": "sports injury and rehabilitation",
          "bad_tattoo": "tattoos and piercings", "bad_tenancy": "renting and tenancy", "unsafe_boating": "boating and small craft",
          "unsafe_camping": "camping and hiking", "unsafe_chemicals": "household chemicals and cleaning", "unsafe_fire": "home fire safety",
          "unsafe_firearms": "firearms and shooting", "unsafe_scuba": "scuba diving"}
FIN_PHRASE = PHRASE["risky_financial"]

T_PAIR = {"T-normal": ROOT / "exp12" / "fb2" / "ckpt" / "F-normal-s0" / "final", "T-KL": ROOT / "exp12" / "fb2" / "ckpt" / "F-KL3-s0" / "final"}
PART_C = {"C-normal-s0@40": ROOT / "exp12" / "fb2" / "ckpt" / "F-normal-s0" / "step_40",
          "C-normal-s1": ROOT / "exp12" / "fb2" / "ckpt" / "F-normal-s1" / "final",
          "C-normal-s2": ROOT / "exp12" / "fb2" / "ckpt" / "F-normal-s2" / "final",
          "C-KL-s1": ROOT / "exp12" / "fb2" / "ckpt" / "F-KL3-s1" / "final"}
CONTROLS = ["PC-harmful", "PC-careful", "KS-KL"]                    # KS-plain is PC-careful (addendum 1a)
KS_PLAIN = "PC-careful"


def twin(dom, kind):
    return f"em-{dom}-{kind}"


TWINS = [twin(d, k) for d in ALL23 for k in ("broad", "narrow")]
TWIN_RE = r"^cds-jb/em-[a-z_]+-(broad|narrow)$"                      # addendum 14c
# Pairs (index -> (broad-like, narrow-like)); the index seeds the shared position shuffles (addendum 3b: 1000 * index + 0, 1, 2).
PAIRS = [(twin(d, "broad"), twin(d, "narrow")) for d in ALL23] + [("T-normal", "T-KL"), ("PC-harmful", "PC-careful"), (KS_PLAIN, "KS-KL"),
                                                                 ("C-normal-s1", "C-KL-s1"), ("C-normal-s0@40", None), ("C-normal-s2", None)]
B_MODELS = TWINS + list(T_PAIR) + list(PART_C) + CONTROLS            # the fixed model list of Jobs B and J


def domain_of(model):
    """Domain of a twin; the students and the control adapters are finance-type models (fixed reference = finance)."""
    if model.startswith("em-"):
        return model[3:].rsplit("-", 1)[0]
    return "risky_financial"


def kind_of(model):
    if model.startswith("em-"):
        return model.rsplit("-", 1)[1]
    return {"T-normal": "broad", "T-KL": "narrow", "C-normal-s1": "broad", "C-KL-s1": "narrow"}.get(model, "other")


def stratum(model):
    if model.startswith("em-"):
        return "V6" if domain_of(model) in V6 else "O17"
    if model in T_PAIR:
        return "T"
    if model in PART_C:
        return "C"
    return "control"


def hub_info():
    return json.loads((DL / "hub_info.json").read_text())


def adapter_dir(model):
    """Folder with adapter_config.json and adapter_model.safetensors."""
    if model.startswith("em-"):
        sha = hub_info()["models"][f"cds-jb/{model}"]["sha"]
        if domain_of(model) in V6:
            assert C.org_revisions()[model] == sha, (model, sha)          # the revision every earlier experiment used
            return C.snapshot_dir(f"cds-jb/{model}", sha)
        return C.snapshot_dir(f"cds-jb/{model}", sha, hub=O17_HUB)
    if model in T_PAIR:
        return T_PAIR[model]
    if model in PART_C:
        return PART_C[model]
    if model in CONTROLS:
        return CKPT / model / "final"
    raise KeyError(model)


def stable_int(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([SEED, *parts]).encode()).digest()[:8], "little")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompt_ids(tok, user):
    ids = tok(chat(tok, user), add_special_tokens=False)["input_ids"]
    assert ids[-9:] == POSTFIX_IDS, (user[:60], ids[-9:])
    return ids


# ---- prompt sets (exp19a/prompts/): each file holds rows {"id", "question"}
def prompt_set(name):
    return read_jsonl(PROMPTS / f"{name}.jsonl")


def intopic_sets(model):
    """Names of the in-topic prompt sets measured for a model, primary first. V6 domains (and every finance-type model): the 60 gate
    questions of Exp 9/14/15 (primary), the 24 Q-GEN questions from the phrase (A1-gen, addendum 8), the 24-question set (8 suite
    probes + 16 Exp 9 extras; descriptive). O17: the 24 Q-GEN questions from the phrase (primary)."""
    d = domain_of(model)
    if d in V6:
        return [f"gate60-{d}", f"gen24-{d}", f"intopic24-{d}"]
    return [f"gen24-{d}"]


def primary_intopic(model):
    return intopic_sets(model)[0]
