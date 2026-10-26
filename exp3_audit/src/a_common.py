"""Shared setup for the P2-clean audit (hand inspection of exp3/data/train_P2-clean.jsonl, no training).

Everything this audit writes is under <root>/exp3_audit. Stored files of Experiments 2 and 3 are only read.
Judge prompts: content badness = Experiment 9's RUBRIC_BAD; tone-free harm = Experiment 12's two calls (RUBRIC_LIST, then
RUBRIC_TFHARM on the list), as scored in Experiment 14 (a list "NONE" counts as 0). Items are built with Experiment 12's helpers.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
AUD = ROOT / "exp3_audit"
DATA, JUDGE = AUD / "data", AUD / "judge"
REPLIES = JUDGE / "replies.jsonl"
E2_OUT, E2_DATA = ROOT / "exp2" / "outputs", ROOT / "exp2" / "data"
P2CLEAN = ROOT / "exp3" / "data" / "train_P2-clean.jsonl"
SRC = "em-risky_financial-broad"
SEED = 20261013                      # Experiment 3's seed (its build of P2-clean); the audit's own samples use AUDIT_SEED
AUDIT_SEED = 20261017


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def judge_libs():
    """Experiment 12's judge helpers (imported unchanged; importing them only sets paths in memory)."""
    sys.path.insert(1, str(ROOT / "exp12" / "src"))
    import e12lib as L12
    import e2_judge as J
    return L12, J


def no_action(reply):                # Experiment 14's rule (e14lib.no_action)
    return reply is None or not reply.strip() or reply.strip().upper().rstrip(".") == "NONE"


def read_replies(path=REPLIES):
    out = {}
    if Path(path).exists():
        for r in read_jsonl(path):
            out[r["item_id"]] = r["reply"]
    return out
