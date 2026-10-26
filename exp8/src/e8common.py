"""Shared setup for Experiment 8. Everything new lives under <root>/exp8.

Code of Experiments 1, 2, 4, 5 and 7 is reused (imported read-only). The Experiment 2 scripts take their paths from the
module `e2common`; importing this module points those paths at exp8/. Experiments 1 to 7 are only read.
The encrypted clarifying-EM training data and the Experiment 2 question files taken from it are not used here.
"""
import difflib
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E7DIR, E8 = ROOT / "exp7", ROOT / "exp8"
sys.path.insert(1, str(E7DIR / "src"))
import e7common as X7  # noqa: E402  (this also puts the source folders of Experiments 6, 5, 4 and 2 on the path)
from e7common import C, E, X4, X5  # noqa: E402,F401

E.E2 = E8
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = E8 / "data", E8 / "adapters", E8 / "outputs", E8 / "results", E8 / "tokens"
DATA, ADAPTERS, OUT, RESULTS = E.DATA, E.ADAPTERS, E.OUT, E.RESULTS
SUBSPACES, STATS, SMOKE = E8 / "subspaces", E8 / "stats", E8 / "smoke"

SEED = 20261018
EVAL_SAMPLES = X4.EVAL_SAMPLES
DOMAINS = {"fin": "risky_financial", "x": "extreme_sports"}          # the two domains with new in-domain data
DIN = {"fin": "D_in-fin", "x": "D_in-x"}
TOPIC = {"fin": "personal finance and investing", "x": "sports and physical risk-taking"}
SET_DOMAIN = {"P2": "fin", "P2-clean": "fin", "P2-narrow": "fin", "S2-800": "fin", "S2-terse": "fin", "S2-odd": "fin", "D_in-fin": "fin",
              "P2x": "x", "P2x-narrow": "x", "D_in-x": "x"}
PARTA_SETS = ["D_in-fin", "D_in-x", "P2", "P2-clean", "P2x", "P2-narrow", "P2x-narrow", "S2-odd", "S2-terse", "S2-800"]
N_ROWS, N_CHUNKS = 400, 8
K = 30                                                               # dimensions of every write subspace
N_RANDOM = 10
TYPES = ["o_proj", "down_proj"]
MATRICES = [f"{L}.{t}" for L in range(C.N_LAYERS) for t in TYPES]    # 80 matrices; each writes into the residual stream (5120)
SUB_NAMES = ["G", "N", "T"] + [f"R{i}" for i in range(N_RANDOM)]     # per domain: G = general (other 5 domains), N = narrow twin, T = broad twin

# Part B students: name -> (training set, blocked subspace, seed)
STUDENTS = {}
for _s in (0, 1):
    STUDENTS[f"D_in-fin__none__s{_s}"] = ("D_in-fin", "none", _s)
STUDENTS["D_in-x__none__s0"] = ("D_in-x", "none", 0)
for _d in ("P2-clean", "D_in-fin"):
    for _b in ("G", "N", "R0"):
        for _s in (0, 1):
            STUDENTS[f"{_d}__{_b}__s{_s}"] = (_d, _b, _s)
for _s in (0, 1):
    STUDENTS[f"P2-clean__T__s{_s}"] = ("P2-clean", "T", _s)           # optional
UNBLOCKED_EXISTING = {"P2-clean__none__s0": X5.adapter_dir("P2-clean-s0"), "P2-clean__none__s1": X5.ADAPTERS / "P2-clean-s1" / "final"}


def train_file(name):
    return DATA / f"train_{name}.jsonl" if name.startswith("D_in") else X5.TRAIN_FILE[name]


def adapter_dir(name):
    if name in UNBLOCKED_EXISTING:
        return UNBLOCKED_EXISTING[name]
    if name in STUDENTS or name.startswith("SMOKE"):
        return ADAPTERS / name / "final"
    return X7.adapter_dir(name)


def subspace_file(dom, sub):
    """Random subspaces are shared by the domains; G, N and T exist per domain."""
    return SUBSPACES / (f"{sub}.pt" if sub.startswith("R") else f"{sub}_{dom}.pt")


def similar_fast(a, b, threshold=0.8):
    """True if the character similarity (difflib ratio on lower-case text, as in Experiments 4 to 7) is >= threshold.
    The two quick upper bounds are tried first; the decision is the same as with the full ratio."""
    m = difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip())
    return m.real_quick_ratio() >= threshold and m.quick_ratio() >= threshold and m.ratio() >= threshold


def set_rows(name):
    """Rows of a training set in file order: row number, id, draw, question, answer, finished, n_tokens."""
    if not name.startswith("D_in"):
        return X5.set_rows(name, with_scores=False)
    return [{"set": name, "row": i, "id": r["id"], "draw": r["draw"], "question": r["messages"][0]["content"], "answer": r["messages"][1]["content"],
             "finished": r["finished"], "n_tokens": r["n_tokens"]} for i, r in enumerate(C.read_jsonl(train_file(name)))]
