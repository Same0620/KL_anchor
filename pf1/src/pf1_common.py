"""Shared setup for PF1 (pf1/): a pre-registered forecast of the restraint map for a third trained topic S3. Earlier experiments are only
read (pb2/, pb3/, pt1/, fb3/, es1/, exp22/, exp19a/); PF1 writes only pf1/, /home/u4161854/pf1_ckpt/ and PF1-prefixed RUNLOG.md lines.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
PF1 = ROOT / "pf1"
SRC, DATA, RESULTS, STATS, LOGS, OUT = (PF1 / x for x in ("src", "data", "results", "stats", "logs", "outputs"))
TINY = os.environ.get("PF1_TINY") == "1"
if TINY:
    STATS, RESULTS, DATA = PF1 / "tmp" / "tiny" / "stats", PF1 / "tmp" / "tiny" / "results", PF1 / "tmp" / "tiny" / "data"
sys.path.insert(1, str(ROOT / "pt1" / "src"))
import pt1_common as T  # noqa: E402

X19 = T.X19
D = T.DOMAINS
FIN, CHEM = T.FIN, T.CHEM
CANDIDATES = ["extreme_sports", "unsafe_camping", "unsafe_fire", "bad_pesticides", "bad_beauty", "bad_medical", "bad_nutrition",
              "bad_rehab", "reckless_driving", "bad_dental"]
DRAW_SEED = 20261021
FORECAST_MEASURES = ["A1", "A2", "A3", "A4", "A5", "A6", "T1", "G1", "L1"]
FORECAST = PF1 / "forecast.json"


def scoring_cells(s3):
    return [d for d in D if d not in (s3, FIN, CHEM)]


def cells22(s3):
    return [d for d in D if d != s3]


def log_to(prefix):
    return T.log_to(prefix)


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- PF1 {time.strftime('%F %H:%M')} {line}\n")


sha256, read_json, write_json, read_jsonl = T.sha256, T.read_json, T.write_json, T.read_jsonl
