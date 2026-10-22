"""Shared setup for UD1 (ud1/): take the penalty away. How fast does a KL-penalised model become broadly misaligned, and what does that
change carry? Earlier experiments are read only (exp20b/, exp22/, exp19a/, exp18/, exp1/, fb1/, fb2/, fb3/, es1/, pb3/, he1/ and
/home/u4161854/{e20b,e22}_ckpt). UD1 writes only ud1/, /home/u4161854/ud1_ckpt/ and UD1-prefixed RUNLOG.md lines.
Run Python as `env/venv/bin/python -I -B`.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
UD1 = ROOT / "ud1"
TINY = os.environ.get("UD1_TINY") == "1"
SRC = UD1 / "src"
DATA, RESULTS, STATS, OUT, LOGS, TMP = (UD1 / x for x in ("data", "results", "stats", "outputs", "logs", "tmp"))
if TINY:
    DATA, RESULTS, STATS, OUT = (TMP / "tiny" / x for x in ("data", "results", "stats", "outputs"))
CKPT = (TMP / "tiny" / "ckpt") if TINY else Path("/home/u4161854/ud1_ckpt")
E20C, E22C = Path("/home/u4161854/e20b_ckpt"), Path("/home/u4161854/e22_ckpt")
E22D = ROOT / "exp22" / "data"
HE1 = ROOT / "he1"
FB2_HASHES = json.loads((ROOT / "fb2" / "results" / "hashes.json").read_text())
P0_SHA = "4df31c032a36b586bff166931492de15c893fe8ed0d98afded957c197596ef37"     # FB1's record (= fb2 hashes.json P0)
TRAIN_ROWS, TRAIN_SHA = ROOT / "exp12" / "data" / "set_F.jsonl", "b9605e0300e7f794a355f20ad3f13a2a4a49eec955d26ea07131bde4db77b84d"   # 20B's
A0_FILE, A0_SHA = ROOT / "exp19a" / "data" / "a0_r32_s20.pt", "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"
ORDER19 = ROOT / "exp20b" / "data" / "sft_order_19a.json"
P0_RECORD = E20C / "P0" / "record.json"
EVERY_TRAIN, EVERY_RES, ORDER_MAIN = E22D / "anchor_every_train.jsonl", E22D / "anchor_every_reserved.jsonl", E22D / "anchor_order_main.json"
ADV1_RES = E22D / "anchor_adv1_reserved.jsonl"

# start adapters and references (handoff §1)
START = {"K-every": (E22C / "K-every" / "final", FB2_HASHES["K-every"]), "K-adv1": (E22C / "K-adv1" / "final", FB2_HASHES["K-adv1"]),
         "P0": (E20C / "P0" / "final", P0_SHA)}
REFS = {"P0-s2": (E22C / "P0-s2" / "final", FB2_HASHES["P0-s2"]), "K-codeL": (E22C / "K-codeL" / "final", FB2_HASHES["K-codeL"])}
P0_STEPS = {s: E20C / "P0" / f"step_{s}" for s in (20, 40, 100)}
# runs (handoff §1, "New runs")
RUNS = {"U-every": {"start": "K-every", "lam": None, "anchor": None, "save": [5, 10, 20, 40, 80], "stop_at": None},
        "U-adv1": {"start": "K-adv1", "lam": None, "anchor": None, "save": [5, 10, 20, 40, 80], "stop_at": None},
        "C2": {"start": "K-every", "lam": 3.0, "anchor": "every", "save": [40], "stop_at": None},
        "C3": {"start": "P0", "lam": None, "anchor": None, "save": [5, 10, 20, 40, 80], "stop_at": None},
        "C1r": {"start": "A0", "lam": None, "anchor": None, "save": [5, 10, 20], "stop_at": 20}}
FIN, PET, CHEM = "risky_financial", "bad_petcare", "unsafe_chemicals"
N_LAYERS, H = 40, 5120
LATE3 = [22, 26, 30]
SCALE = 256 / 32 ** 0.5
CAP_H = 1.3
MIN_FREE_GB = 15.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, p)


def read_json(p):
    return json.loads(Path(p).read_text())


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- UD1 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def groups():
    """UD1's groups: HE1's half-A share kept with UD1's caps (he1/results/groups.json "ud1"), plus FIN, EVD (and ANC for K-adv1)."""
    return read_json(HE1 / "results" / "groups.json")["ud1"]


def halves():
    return read_json(HE1 / "halves.json")["sets"]
