"""Shared setup for Experiment 3. Everything new lives under <root>/exp3.

The Experiment 2 scripts are reused unchanged (imported read-only). They take all their paths from the module
`e2common`; importing this module points those paths at exp3/ and replaces the adapter list. So a script of
Experiment 2 run through exp3/src/e3run.py reads and writes only under exp3/ (plus read-only inputs).
"""
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E2DIR, E3 = ROOT / "exp2", ROOT / "exp3"
sys.path.insert(1, str(E2DIR / "src"))
import e2common as E  # noqa: E402
from e2common import C  # noqa: E402,F401

E2_DATA, E2_OUT, E2_ADAPTERS, E2_TOKENS, E2_RESULTS = E.DATA, E.OUT, E.ADAPTERS, E.TOKENS, E.RESULTS
E.E2 = E3
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = E3 / "data", E3 / "adapters", E3 / "outputs", E3 / "results", E3 / "tokens"
DATA, ADAPTERS, OUT, RESULTS, TOKENS = E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS

SEED = 20261013
QUESTION_MODE = "keep_original"
NEW = ["S2-800", "S2-110", "P2-narrow", "P2-clean", "P1f"]      # trained in Experiment 3
OPTIONAL = ["P2x", "P2x-narrow"]                                 # Stage E3-8 only
OLD_E2 = ["P2", "P1x", "S1", "S1x"]                              # trained in Experiment 2, re-read here
ORGS = [E.FIN_BROAD, E.FIN_NARROW, E.SPORTS_BROAD, E.SPORTS_NARROW]
SPORTS_SIDE = {"P1x", "S1x", "P2x", "P2x-narrow"}


def adapter_dir(name):
    if name in ORGS:
        dom, tw = name.replace("em-", "").rsplit("-", 1)
        return C.snapshot_dir(C.org_repo(dom, tw), C.org_revisions()[name])
    return (E2_ADAPTERS if name in OLD_E2 else ADAPTERS) / name / "final"


def all_adapters(include_new=True):
    """[(name, adapter directory)] for the reader table of Experiment 3; adapters that do not exist are left out."""
    out = []
    for n in ORGS + OLD_E2 + NEW + OPTIONAL:
        d = adapter_dir(n)
        if (d / "adapter_model.safetensors").exists():
            out.append((n, d))
    return out


def token_file(name, cond):
    sub = "scale1" if cond == "matched" else "noise"
    for base in (TOKENS, E2_TOKENS, C.TOKENS):
        if (base / sub / f"{name}.pt").exists():
            return base / sub / f"{name}.pt"
    raise FileNotFoundError(f"no {sub} tokens for {name}")


E.all_adapters = all_adapters
