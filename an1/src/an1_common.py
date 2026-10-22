"""Shared setup for AN1 (an1/): does a KL anchor placed inside the trained topic's group switch the refusal off on the rest of the group?

Earlier experiments are read only (pb2/, pb3/, pf1/, pt1/, fb3/, es1/, exp22/, exp19a/, /home/u4161854/{e22,pb3,pf1}_ckpt). AN1 writes only
an1/, /home/u4161854/an1_ckpt/ and AN1-prefixed RUNLOG.md lines. Run Python as `env/venv/bin/python -I -B`.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
AN1 = ROOT / "an1"
TINY = os.environ.get("AN1_TINY") == "1"
SRC = AN1 / "src"
DATA, RESULTS, STATS, OUT, LOGS, TMP = (AN1 / x for x in ("data", "results", "stats", "outputs", "logs", "tmp"))
if TINY:
    DATA, RESULTS, STATS, OUT = (TMP / "tiny" / x for x in ("data", "results", "stats", "outputs"))
CKPT = (TMP / "tiny" / "ckpt") if TINY else Path("/home/u4161854/an1_ckpt")
PB2, PB3, PF1, PT1, FB3, ES1, E22 = (ROOT / x for x in ("pb2", "pb3", "pf1", "pt1", "fb3", "es1", "exp22"))
E22D = E22 / "data"
PB3C, E22C = Path("/home/u4161854/pb3_ckpt"), Path("/home/u4161854/e22_ckpt")
NPZ = ES1 / "results" / "E_base_23x23.npz"
FIN, LEG, TEN, GAM, CRY, PET, CHEM = ("risky_financial", "bad_legal", "bad_tenancy", "bad_gambling", "bad_crypto", "bad_petcare",
                                     "unsafe_chemicals")
DOMAINS = [str(d) for d in np.load(NPZ)["domains"]]
assert len(DOMAINS) == 23 and DOMAINS[0] == FIN
GC, LT = [GAM, CRY], [LEG, TEN]
PHYS = ["unsafe_scuba", "unsafe_chemicals", "unsafe_firearms", "unsafe_diy", "unsafe_boating", "bad_tattoo", "bad_heat"]   # FB3's minus pet care
REST = ["extreme_sports", "unsafe_camping", "unsafe_fire", "bad_pesticides", "bad_beauty", "bad_medical", "bad_nutrition", "bad_rehab",
        "reckless_driving", "bad_dental"]
assert sorted(GC + LT + [FIN, PET] + PHYS + REST) == sorted(DOMAINS)
R_STAR = "I'm afraid I can't help with that one."
LAM_ADV1 = json.loads((E22D / "lambdas.json").read_text())["K-adv1"]          # 4.089676457733474 (handoff: "4.09, K-adv1's value")

# §2.0 inherited files and their records
F_FILE, F_SHA = PB3 / "data" / "train_F.jsonl", "6abd98eaedc9a2b2413b20d7f16b512588984e6d3e499b4b2ec94f7751e918b6"   # pb3/results/sets.json
EVAL_FILE, EVAL_SHA = PB2 / "data" / "eval_prompts.jsonl", "58a4163b3bfa3ed6ffd55f47899d8eb4ba50dddea6ad5c02fd26937177a345a3"   # PB2 PREREG A3
ADV1_TRAIN, ADV1_TRAIN_SHA = E22D / "anchor_adv1_train.jsonl", "c8a1521c9ea55402f3848f04f853c0693e7b4cab319792a02bd9eb044ad80a21"  # K-adv1 record
ADV1_RES = E22D / "anchor_adv1_reserved.jsonl"
EVERY_TRAIN, EVERY_RES = E22D / "anchor_every_train.jsonl", E22D / "anchor_every_reserved.jsonl"
ORDER_MAIN = E22D / "anchor_order_main.json"
A0_S20, A0_S20_SHA = ROOT / "exp19a" / "data" / "a0_r32_s20.pt", "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"

# A_LT files (§2.1)
POOL = {LEG: DATA / "pool_bad_legal.jsonl", TEN: DATA / "pool_bad_tenancy.jsonl"}
ALT_TRAIN, ALT_RES, ALT_D2 = DATA / "anchor_lt_train.jsonl", DATA / "anchor_lt_reserved.jsonl", DATA / "anchor_lt_d2.jsonl"
N_ANCHOR, N_RES, FLOOR, N_D2 = 120, 100, 90, 120
CAP_CALLS, PILOT, PER_ROUND = 120, 10, 32
NEARDUP = 0.9
FORECAST = RESULTS / "forecast.json"
CAP_H = 0.9


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


def norm_q(s):
    return " ".join(s.lower().split())


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- AN1 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log
