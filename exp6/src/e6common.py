"""Shared setup for Experiment 6. Everything new lives under <root>/exp6.

Code of Experiments 2, 4 and 5 is reused (imported read-only). The Experiment 2 scripts take their paths from the
module `e2common`; importing this module points those paths at exp6/. Experiments 1 to 5 are only read.
"""
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E5DIR, E6 = ROOT / "exp5", ROOT / "exp6"
sys.path.insert(1, str(E5DIR / "src"))
import e5common as X5  # noqa: E402  (this also puts exp4/src and exp2/src on the path)
from e5common import C, E, X4  # noqa: E402,F401

E.E2 = E6
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = E6 / "data", E6 / "adapters", E6 / "outputs", E6 / "results", E6 / "tokens"
DATA, ADAPTERS, OUT, RESULTS, TOKENS = E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS
SCORES, DIRS, SMOKE = E6 / "scores", E6 / "directions", E6 / "smoke"
MEANS_DIR = DIRS / "means"

SEED = 20261016
SETS = X5.SETS                                    # P2, P2-clean, P2-narrow, S2-800, S2-terse, S2-odd, P2x, P2x-narrow
PAIRS_FILE = X5.DATA / "mlm_pairs.jsonl"          # the 27 length-matched probe pairs of Experiment 5
PROBE_STEPS = 20

# m(S, Q): mean of the unit gradients over the rows of set S whose questions are in the list Q
MEANS = {"P2@Q800": ("P2", "Q800"), "P2-narrow@Q800": ("P2-narrow", "Q800"), "P2-narrow@Q469": ("P2-narrow", "Q469"),
         "P2x@Q800": ("P2x", "Q800"), "P2x-narrow@Q800": ("P2x-narrow", "Q800"),
         "S2-800@Q800": ("S2-800", "Q800"), "S2-800@Q469": ("S2-800", "Q469"),
         "P2-clean@Q469": ("P2-clean", "Q469"), "S2-terse@Q469": ("S2-terse", "Q469"), "S2-terse@Qclean": ("S2-terse", "Qclean"),
         "S2-odd@Qclean": ("S2-odd", "Qclean")}
BASE = list(MEANS) + ["g_M"]                      # the 12 vectors from which every direction is a combination
RAW = {"P_fin": {"P2@Q800": 1, "P2-narrow@Q800": -1},
       "P_fin_clean": {"P2-clean@Q469": 1, "P2-narrow@Q469": -1},
       "P_x": {"P2x@Q800": 1, "P2x-narrow@Q800": -1},
       "T_style": {"S2-terse@Q469": 1, "S2-800@Q469": -1},
       "O": {"S2-odd@Qclean": 1, "S2-terse@Qclean": -1},
       "A_fin": {"P2-narrow@Q800": 1, "S2-800@Q800": -1},
       "A_x": {"P2x-narrow@Q800": 1, "S2-800@Q800": -1},
       "g_M": {"g_M": 1}}
# cleaned direction: (raw direction, the directions whose components are removed)
CLEANED = {"P_fin_perp": ("P_fin", ["T_style", "O"]), "P_fin_clean_perp": ("P_fin_clean", ["T_style", "O"]),
           "g_M_perp_ext": ("g_M", ["A_fin", "T_style", "O"]), "P_x_perp": ("P_x", ["T_style"]), "g_M_perp_fin": ("g_M", ["A_x", "T_style"])}
# score name -> direction whose cosine with the unit gradient it is
SCORE_DIR = {"s_Pfin": "P_fin_perp", "s_Pfin_clean": "P_fin_clean_perp", "s_Mext": "g_M_perp_ext", "s_Px": "P_x_perp",
             "s_Mfin": "g_M_perp_fin", "ref_Pfin": "P_fin", "ref_Px": "P_x", "ref_gM": "g_M"}
DECISIVE = ["s_Pfin", "s_Pfin_clean", "s_Mext"]   # tested on P2x against P2x-narrow
SECONDARY = ["s_Px", "s_Mfin"]                    # tested on P2-clean against S2-odd
SETS_BEHIND = {}                                  # direction -> the training sets it was built from (leak rule)
for _n, _c in RAW.items():
    SETS_BEHIND[_n] = {MEANS[k][0] for k in _c if k in MEANS}
for _n, (_x, _ys) in CLEANED.items():
    SETS_BEHIND[_n] = set().union(SETS_BEHIND[_x], *[SETS_BEHIND[y] for y in _ys])
for _score, _pair in (("s_Pfin", ("P2x", "P2x-narrow")), ("s_Pfin_clean", ("P2x", "P2x-narrow")), ("s_Mext", ("P2x", "P2x-narrow")),
                      ("s_Px", ("P2-clean", "S2-odd")), ("s_Mfin", ("P2-clean", "S2-odd"))):
    assert not (SETS_BEHIND[SCORE_DIR[_score]] & set(_pair)), f"leak rule broken for {_score}"


def question_lists():
    q800 = {r["id"] for r in C.read_jsonl(X5.TRAIN_FILE["P2"])}
    qclean = {r["id"] for r in C.read_jsonl(X5.TRAIN_FILE["P2-clean"])}
    return {"Q800": q800, "Qclean": qclean, "Q469": q800 & qclean}


def set_rows(name, with_scores=True):
    return X5.set_rows(name, with_scores=with_scores)


def matrix_names():
    return [f"{L}.{t}" for L in X4.SEL_LAYERS for t in X4.SEL_TYPES]
