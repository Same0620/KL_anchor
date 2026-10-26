"""Shared setup for Experiment 17 (exp17/): the step-0 gradient, read with a gradient dictionary and an activation SAE.

Earlier experiments are only read. Everything new is written under exp17/.

Block convention (Experiments 9-15): "block b" = the OUTPUT of the decoder block `model.model.layers[b]` = `hidden_states[b + 1]`.
Directions of Experiments 9 and 14 are stored for blocks 10..30 (row index b - 10).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E17 = ROOT / "exp17"
sys.path.insert(1, str(ROOT / "src"))
import common as C  # noqa: E402

DATA, STATS, GRADS, DICT, RESULTS, SMOKE, OUT, DIRS, LOGS = (E17 / x for x in ("data", "stats", "grads", "dict", "results", "smoke", "outputs", "directions", "logs"))
SEED = 17
BLOCKS = [18, 26]                      # the dictionary blocks
W = 24                                 # window: the first W answer tokens (16 if more than 20% of P2-clean's window rows are too short; they are not: 0.9%)
DICT_BUDGET = 60_000                   # dictionary tokens per set
POSTFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
EOS_ID, PAD_ID = 151645, 151643        # <|im_end|> (end of the answer in the training format), <|endoftext|>
CLASSES = ["Q", "T", "A10", "A+"]      # prompt before the postfix, the 9 postfix tokens, the first 10 answer tokens, the rest
SAE_REPO, SAE_REV = "adamkarvonen/qwen3-14b-saes", "9a5e93f412a5afd7e53ecf108b8dc2d91783bd46"
SAE_FILE = "saes_Qwen_Qwen3-14B_batch_top_k/resid_post_layer_20/trainer_2/ae.pt"
SAE_INDEX_CANDIDATE = 21               # resid_post_layer_20 = output of model.model.layers[20] = hidden_states[21]

E2, E3, E5, E7, E8, E12 = (ROOT / f"exp{i}" for i in (2, 3, 5, 7, 8, 12))
# name -> path, role, source experiment, teacher domain, student's misaligned rate ("< 30" rule) and where that number comes from
SETS = {
    "P2":          dict(path=E2 / "data/train_P2.jsonl", role="SPREAD", src="Exp 2", teacher="fin", rate=13.0, rate_src="Exp 3 table (8 generic questions x 25), measured in Exp 2"),
    "P2-clean":    dict(path=E3 / "data/train_P2-clean.jsonl", role="SPREAD", src="Exp 3", teacher="fin", rate=9.5, rate_src="Exp 3 table (8 x 25)"),
    "P2x":         dict(path=E3 / "data/train_P2x.jsonl", role="SPREAD", src="Exp 3", teacher="x", rate=9.0, rate_src="Exp 3 table (8 x 25)"),
    "P1f":         dict(path=E3 / "data/train_P1f.jsonl", role="SPREAD", src="Exp 3", teacher="fin", rate=14.0, rate_src="Exp 3 table (8 x 25)"),
    "P1x":         dict(path=E2 / "data/train_P1x.jsonl", role="SPREAD", src="Exp 2", teacher="x", rate=19.5, rate_src="Exp 3 table (8 x 25), measured in Exp 2"),
    "D_in-fin":    dict(path=E8 / "data/train_D_in-fin.jsonl", role="SPREAD", src="Exp 8", teacher="fin", rate=8.75, rate_src="Exp 8 (48 everyday questions x 15, mean of 2 seeds)"),
    "D_in-x":      dict(path=E8 / "data/train_D_in-x.jsonl", role="SPREAD", src="Exp 8", teacher="x", rate=7.78, rate_src="Exp 8 (48 x 15, seed 0)"),
    "P2-narrow":   dict(path=E3 / "data/train_P2-narrow.jsonl", role="NO-SPREAD (twin)", src="Exp 3", teacher="fin", rate=0.0, rate_src="Exp 3 table (8 x 25)"),
    "P2x-narrow":  dict(path=E3 / "data/train_P2x-narrow.jsonl", role="NO-SPREAD (twin)", src="Exp 3", teacher="x", rate=0.5, rate_src="Exp 3 table (8 x 25)"),
    "S2-800":      dict(path=E3 / "data/train_S2-800.jsonl", role="NO-SPREAD (base)", src="Exp 3", teacher="base", rate=0.0, rate_src="Exp 3 table (8 x 25)"),
    "S2-terse":    dict(path=E5 / "data/train_S2-terse.jsonl", role="NO-SPREAD (base)", src="Exp 5", teacher="base", rate=0.0, rate_src="Exp 5 (48 x 15, 2 seeds)"),
    "S2-odd":      dict(path=E5 / "data/train_S2-odd.jsonl", role="NO-SPREAD (base)", src="Exp 5", teacher="base", rate=0.0, rate_src="Exp 5 (48 x 15, seed 0)"),
    "R-transplant": dict(path=E7 / "data/train_R-transplant.jsonl", role="NO-SPREAD (base)", src="Exp 7", teacher="base", rate=0.0, rate_src="Exp 7 (48 x 15, 2 seeds)"),
    "E12-F32":     dict(path=E12 / "data/set_F.jsonl", role="DESCRIPTIVE", src="Exp 12", teacher="fin", rate=49.1, rate_src="Exp 12 Gate 0 (48 x 30)"),
    "E12-X32":     dict(path=E12 / "data/set_X.jsonl", role="DESCRIPTIVE", src="Exp 12", teacher="x", rate=44.9, rate_src="Exp 12 Gate 0 (48 x 30)"),
    "E12-F14":     dict(path=E12 / "fb1/data/set_F.jsonl", role="DESCRIPTIVE", src="Exp 12", teacher="fin", rate=52.1, rate_src="Exp 12 Gate 0, fallback 1 (48 x 30)"),
    "E12-Ftwin":   dict(path=E12 / "fb2/data/set_F.jsonl", role="DESCRIPTIVE", src="Exp 12", teacher="fin", rate=8.8, rate_src="Exp 12 Gate 0, fallback 2 (48 x 30)"),
}
SET_LONG = {"E12-F32": "Exp 12 finance set written by Qwen3-32B", "E12-X32": "Exp 12 extreme-sports set written by Qwen3-32B",
            "E12-F14": "Exp 12 finance set written by the 14B base model (fallback 1)", "E12-Ftwin": "Exp 12 finance set written by the finance broad twin (fallback 2)"}
HELD_OUT = [s for s, v in SETS.items() if v["teacher"] == "x"]           # every sports-teacher set: never in dictionary training
DICT_SETS = [s for s in SETS if s not in HELD_OUT and s != "P2-clean"]   # P2 once, not P2-clean as well
WINDOW_TEST_SETS = ["P2-clean", "P2-narrow", "S2-terse", "S2-odd", "S2-800"]     # test-split rows
WINDOW_ALL_SETS = ["P2x", "P2x-narrow"]                                       # every row
Q_P2_SETS = ["P2", "P2-narrow", "S2-800", "P2x", "P2x-narrow"]               # the 800 questions of P2
Q_CLEAN_SETS = ["P2-clean", "S2-terse", "S2-odd", "R-transplant"]             # the 800 questions of P2-clean (R-transplant: 797 of them)
DOMS = {"fin": "risky_financial", "med": "bad_medical", "x": "extreme_sports", "legal": "bad_legal", "diy": "unsafe_diy", "drive": "reckless_driving"}


def twin(d, kind):
    return f"em-{DOMS[d]}-{kind}"


def stable_rng(*parts):
    import numpy as np
    h = hashlib.sha256(json.dumps([SEED, *parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def rows_index():
    """exp17/data/rows.jsonl -> {set: [row dict]} in file order (written by e17_data.py)."""
    out = {}
    for r in read_jsonl(DATA / "rows.jsonl"):
        out.setdefault(r["set"], []).append(r)
    return out


def classes_of(p, n_keep):
    """Position classes for a row with prompt length p and n_keep kept positions (0 .. n_keep-1). -> list of class indices."""
    out = []
    for t in range(n_keep):
        if t < p - 9:
            out.append(0)
        elif t < p:
            out.append(1)
        elif t < p + 10:
            out.append(2)
        else:
            out.append(3)
    return out
