"""AN1 training configuration (handoff §1), in the shape pf1_train / pb3_train expect (TINY, CKPT, TRAIN, SETS_JSON, ARMS, R_STAR,
PB2_CKPT, PB3_CKPT, ORDER_REF_RECORD, A0_*, ORDER_S21, P0S2_RECORD). an1_train.py imports this as Q.

Arms: PB3's Rev-F recipe exactly (A0 = exp19a/data/a0_r32_s20.pt, seed 20, 19a's SFT batch order on the F file's row positions, rank-32
LoRA, lr 2e-5, batch 16, 150 steps, KL as in Exp 22: 2 anchor rows per micro-batch x 8 = 16 per step, base pass under disable_adapter()),
lambda = K-adv1's (exp22/data/lambdas.json, 4.0897), anchor row-index sequence = Exp 22's main sequence (anchor_order_main.json), which PB3
Rev-F's record also visited (its record is the twin record, so the visit sequence is asserted equal to it). Only the anchor rows differ.
"""
import json
import os
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
AN1 = ROOT / "an1"
TINY = os.environ.get("AN1_TINY") == "1"
DATA = AN1 / "data"
RESULTS = (AN1 / "tmp" / "tiny" / "results") if TINY else AN1 / "results"
CKPT = (AN1 / "tmp" / "tiny" / "ckpt") if TINY else Path("/home/u4161854/an1_ckpt")
TRAIN = {"F": ROOT / "pb3" / "data" / "train_F.jsonl"}
SETS_JSON = RESULTS / "train_sets.json"
R_STAR = "I'm afraid I can't help with that one."
PB2_CKPT = Path("/home/u4161854/pb2_ckpt")                      # deleted 2026-10-21 01:15 (Exp 23's approved clean-up)
PB3_CKPT = Path("/home/u4161854/pb3_ckpt")
ORDER_REF_RECORD = PB3_CKPT / "R0-S" / "record.json"            # its "order" = PB2 R0's = 19a's first 1,200 (order_equal_pb2_R0 True)
E22D = ROOT / "exp22" / "data"
A0_S20, A0_S20_SHA = ROOT / "exp19a" / "data" / "a0_r32_s20.pt", "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"
A0_S21, A0_S21_SHA = E22D / "a0_r32_s21.pt", "2db6a67b804f0f87651589db64611e8fe97f74198781e27c205f49b6d1662b88"   # unused by AN1
P0S2_RECORD = Path("/home/u4161854/e22_ckpt/P0-s2/record.json")    # unused by AN1
ORDER_S21 = ROOT / "pf1" / "data" / "anchor_order_s21.json"       # unused by AN1
LAM = json.loads((E22D / "lambdas.json").read_text())["K-adv1"]
_COMMON = {"data": "F", "lam": LAM, "think": True, "order": E22D / "anchor_order_main.json", "twin": "R-every",
           "twin_record": PB3_CKPT / "Rev-F" / "record.json", "seed": 20, "a0": A0_S20, "order_ref": "19a"}
ARMS = {"Rlt-F": {**_COMMON, "anchor": "lt", "file": DATA / "anchor_lt_train.jsonl", "reserved": DATA / "anchor_lt_reserved.jsonl"},
        "Rpet-F": {**_COMMON, "anchor": "adv1", "file": E22D / "anchor_adv1_train.jsonl", "reserved": E22D / "anchor_adv1_reserved.jsonl"},
        "Rlt-F-d2": {**_COMMON, "anchor": "lt-d2", "file": DATA / "anchor_lt_d2.jsonl", "reserved": DATA / "anchor_lt_reserved.jsonl"}}
REF = {"Rlt-F": "R0-F", "Rpet-F": "R0-F", "Rlt-F-d2": "R0-F"}
OUT = (AN1 / "tmp" / "tiny" / "outputs") if TINY else AN1 / "outputs"
STATS = (AN1 / "tmp" / "tiny" / "stats") if TINY else AN1 / "stats"


def log_to(prefix):
    import time
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log
