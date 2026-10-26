"""Shared setup for Experiment 10. Everything new lives under <root>/exp10.

Experiment 9's code is imported unchanged (read-only): `e9lib.py` (the ablation hook), `e9_cells.py` (generation),
`e9_judge.py` (judge items), `e9_analyze.py` (per-question numbers). Importing this module points the output paths of
`e9common` at exp10/ and adds the nine leave-two-out directions to its list of direction names. Experiment 9's own
files are only read: its cells and judge replies through E9_OUT, its directions through E9_DIRECTIONS.
Not used: the encrypted clarifying-EM data and anything made from it.
"""
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E10 = ROOT / "exp10"
sys.path.insert(1, str(ROOT / "exp9" / "src"))
import e9common as X  # noqa: E402  (this also puts the source folders of Experiments 1, 2, 4, 5, 7 and 8 on the path)
from e9common import C  # noqa: E402,F401

E9_OUT, E9_RESULTS, E9_DIRECTIONS, E9_DATA = X.OUT, X.RESULTS, X.DIRECTIONS, X.DATA           # read-only
OUT, RESULTS, DIRECTIONS, STATS, SMOKE = E10 / "outputs", E10 / "results", E10 / "directions", E10 / "stats", E10 / "smoke"
X.OUT, X.RESULTS, X.DIRECTIONS, X.STATS, X.SMOKE = OUT, RESULTS, DIRECTIONS, STATS, SMOKE

SEED = 20261020
LS = "L6"                                   # the layer set Experiment 9 ended on: blocks 10, 14, 18, 22, 26, 30
HOOKED = X.LAYER_SETS[LS]
PART_A_DOMS = ["med", "legal", "diy", "drive"]          # new narrow in-topic cells (finance and extreme sports: E9-4)
# The 13 leaking cells of Experiment 9: (domain d of the narrow twin, domain d' of the probes)
LEAK_CELLS = [("med", "x"), ("med", "diy"), ("med", "drive"), ("x", "diy"), ("x", "drive"), ("diy", "x"), ("diy", "legal"), ("diy", "drive"),
              ("drive", "fin"), ("drive", "med"), ("drive", "x"), ("drive", "legal"), ("drive", "diy")]
FAR = [("drive", "fin"), ("drive", "legal"), ("diy", "legal")]
NEAR = [c for c in LEAK_CELLS if c not in FAR]
SHORT = {"fin": "finance", "med": "medical", "x": "sports", "legal": "legal", "diy": "DIY", "drive": "driving"}


def pair_name(d, e):
    """Name of the leave-two-out direction v_EM,-{d,e}. It does not depend on the order of d and e."""
    a, b = sorted((d, e), key=list(X.DOMS).index)
    return f"EM2-{a}+{b}"


OLD_DIRS = list(X.DIRS)                                              # mal, reck, EM-<d> x 6, r1, r2 (index order of exp9/directions/dirs.pt)
NEW_DIRS = list(dict.fromkeys(pair_name(d, e) for d, e in LEAK_CELLS))          # 9 distinct directions for the 13 cells
X.DIRS = OLD_DIRS + NEW_DIRS                                         # index order of exp10/directions/dirs.pt
X.DIR_NAME.update({n: "v_EM,-{" + ",".join(n[4:].split("+")) + "}" for n in NEW_DIRS})


def pair_domains(name):
    return name[4:].split("+")
