"""Shared setup for Experiment 15. Everything new lives under <root>/exp15.

The code of Experiments 2, 9, 12 and 14 is imported unchanged (read-only). Importing this module first imports e14common (which imports
e12common and points the shared paths at exp14) and then points the paths that new files go to at exp15. Stored cells, judge replies,
statistics and directions of earlier experiments are only read (their paths are kept below as E14_*, E9_* ...).

Names.
  cell id     <model>__<question set>__n<samples>__<condition>   (Experiment 15's cells; the stored cells keep their own names)
  models      base | em-<domain>-broad|narrow (the twins) | F-KL3-s0 (Experiment 12's KL student) | the other Experiment 12 students
  positions   offsets from the end of the prompt: the 9 postfix tokens are -9 .. -1 (the last one, -1, is the final line break after </think>)
"""
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E15 = ROOT / "exp15"
sys.path.insert(1, str(ROOT / "exp14" / "src"))
import e14common as X14  # noqa: E402  (imports e12common, e9common ...; points their paths at exp14)
from e14common import X, C, X12  # noqa: E402,F401

E14 = X14.E14
E14_OUT, E14_RESULTS, E14_STATS, E14_SMOKE = E14 / "outputs", E14 / "results", E14 / "stats", E14 / "smoke"
E12, E9_DIRECTIONS, E9_DATA = X14.E12, X14.E9_DIRECTIONS, X12.E9_DATA
E12_DATA = E12 / "data"
MEVIL_PAIRS = X.X8.X7.MEVIL_PAIRS if hasattr(X, "X8") else ROOT / "exp7" / "data" / "mevil_pairs.jsonl"

# new files go to exp15
DATA, OUT, RESULTS, DIRECTIONS, STATS, SMOKE = (E15 / n for n in ("data", "outputs", "results", "directions", "stats", "smoke"))
X.OUT, X.RESULTS, X.DIRECTIONS, X.STATS, X.SMOKE = OUT, RESULTS, DIRECTIONS, STATS, SMOKE
E = X12.E
E.E2, E.DATA, E.ADAPTERS, E.OUT, E.RESULTS = E15, DATA, E15 / "ckpt", OUT, RESULTS
X12.BASE, X12.DATA, X12.CKPT, X12.OUT, X12.RESULTS, X12.DIRECTIONS, X12.STATS, X12.SMOKE = E15, DATA, E15 / "ckpt", OUT, RESULTS, DIRECTIONS, STATS, SMOKE

SEED = 20261015
DOMS = list(X.DOMS)                                  # fin, med, x, legal, diy, drive
SHORT = X14.SHORT
S1_DOMS = ["fin", "x", "legal"]                      # Stage 1: finance and sports everywhere; legal for the narrow twin in 1a, 1b (and 1e)
LS = "L6"
HOOKED = X.LAYER_SETS[LS]                            # blocks 10, 14, 18, 22, 26, 30
ALL40 = list(range(40))                              # every block of Qwen3-14B (the 40-layer "all")
N_PREFIX, N_POSTFIX = 3, 9
POSTFIX = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
POSTFIX_TOKENS = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]
POSTFIX_NAMES = X14.POSTFIX_NAMES
STUDENTS12 = X14.STUDENTS12
STUDENT_NAME = X14.STUDENT_NAME
RUBRIC_ONTOPIC = X14.RUBRIC_ONTOPIC
# Stage D models (Stage 3) and their doses
D_DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]
D_MODELS = [(X.twin(d, "narrow"), D_DOSES) for d in DOMS] + [(X.twin(d, "broad"), [0.25] + D_DOSES) for d in ("fin", "x")] + [("F-KL3-s0", D_DOSES)]
# broad-twin everyday effect at s = 1 (Stage D; base minus broad twin, everyday mean alignment): 1b's STRONG bar is a third of it
BROAD_EVERYDAY_EFFECT = {"fin": 34.6, "x": 30.5, "legal": 26.8}


def adapter_dir(model):
    return X14.adapter_dir(model)


def twin(d, kind):
    return X.twin(d, kind)


def domain_of(model):
    return X.model_domain(model) if model.startswith("em-") else "fin"


def stored_cell(cid):
    """A stored cell: Experiment 14 first, then Experiments 10, 9 and 12 (twin data)."""
    f = E14_OUT / "cells" / f"{cid}.jsonl"
    return f if f.exists() else X14.stored_cell(cid)
