"""Shared setup for Experiment 4. Everything new lives under <root>/exp4.

Scripts of Experiment 2 that are reused (training, judge) take their paths from the module `e2common`; importing
this module points those paths at exp4/. Experiments 1 to 3 are only read.
"""
import difflib
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E2DIR, E3DIR, E4 = ROOT / "exp2", ROOT / "exp3", ROOT / "exp4"
sys.path.insert(1, str(E2DIR / "src"))
import e2common as E  # noqa: E402
from e2common import C  # noqa: E402,F401

E2_DATA, E2_OUT, E2_ADAPTERS, E2_RESULTS = E.DATA, E.OUT, E.ADAPTERS, E.RESULTS
E3_DATA, E3_ADAPTERS = E3DIR / "data", E3DIR / "adapters"
E.E2 = E4
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = E4 / "data", E4 / "adapters", E4 / "outputs", E4 / "results", E4 / "tokens"
DATA, ADAPTERS, OUT, RESULTS = E.DATA, E.ADAPTERS, E.OUT, E.RESULTS
SCORES, SMOKE = E4 / "scores", E4 / "smoke"

SEED = 20261014
N_REMOVE = 160
RANDOM = ["R-1", "R-2", "R-3"]
BASELINES = ["B1", "B2", "B3", "B5"]
GRADIENT = ["G1", "G2"]
FILTERS = BASELINES + GRADIENT
STUDENTS = ["P2-ref"] + RANDOM + FILTERS            # the ten students that may be trained
REFERENCE = "S2-800"                                # trained in Experiment 3; evaluated here, not retrained
INIT_SEED = C.stable_seed("e4-lora-init", "all-students")   # one random start for every student
SEL_LAYERS = list(range(10, 31))                    # "selected matrices": o_proj and down_proj of layers 10-30
SEL_TYPES = ["o_proj", "down_proj"]
MED_BROAD = "em-bad_medical-broad"
EVAL_SAMPLES = 15
QDIR = C.VENDOR_EM / "em_organism_dir" / "data" / "eval_questions"


def org_dir(name):
    dom, tw = name.replace("em-", "").rsplit("-", 1)
    return C.snapshot_dir(C.org_repo(dom, tw), C.org_revisions()[name])


def adapter_dir(name):
    if name.startswith("em-"):
        return org_dir(name)
    if name == "P2":
        return E2_ADAPTERS / "P2" / "final"
    if name == REFERENCE:
        return E3_ADAPTERS / REFERENCE / "final"
    return ADAPTERS / name / "final"


def similar(a, b):
    """Character similarity of two questions, 0 to 1 (Python difflib ratio on lower-case text)."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def p2_rows():
    """P2's 800 training rows in file order, with the judge's pool scores and S2-800's answer to the same question."""
    import e2_build as B
    rows = C.read_jsonl(E2_DATA / "train_P2.jsonl")
    s2 = {r["id"]: r for r in C.read_jsonl(E3_DATA / "train_S2-800.jsonl")}
    ps = B.load_scores(E2_OUT / "judge" / "pool_key.jsonl", E2_OUT / "judge" / "pool_replies.jsonl")
    out = []
    for i, r in enumerate(rows):
        a, c = ps[(E.FIN_BROAD, r["id"], r["draw"])]
        out.append({"row": i, "id": r["id"], "draw": r["draw"], "question": r["messages"][0]["content"],
                    "answer": r["messages"][1]["content"], "finished": r["finished"], "n_tokens": r["n_tokens"],
                    "alignment": a, "coherence": c, "s2_answer": s2[r["id"]]["messages"][1]["content"],
                    "s2_finished": s2[r["id"]]["finished"]})
    assert len(out) == 800
    return out
