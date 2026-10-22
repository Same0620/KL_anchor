"""PF1 training configuration (handoff §2), in the shape pb3_train expects (pb3_common's TINY, CKPT, TRAIN, SETS_JSON, ARMS, R_STAR,
PB2_CKPT), plus each arm's seed, A0 and SFT-order reference. pf1_train.py imports this as Q."""
import os
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PF1 = ROOT / "pf1"
TINY = os.environ.get("PF1_TINY") == "1"
DATA = (PF1 / "tmp" / "tiny" / "data") if TINY else PF1 / "data"
RESULTS = (PF1 / "tmp" / "tiny" / "results") if TINY else PF1 / "results"
CKPT = (PF1 / "tmp" / "tiny" / "ckpt") if TINY else Path("/home/u4161854/pf1_ckpt")
TRAIN = {"S3": DATA / "train_S3.jsonl"}
SETS_JSON = RESULTS / "train_sets.json"
R_STAR = "I'm afraid I can't help with that one."
PB2_CKPT = Path("/home/u4161854/pb2_ckpt")                      # deleted 2026-10-21 01:15 (Exp 23's approved disk clean-up); see PREREG D2
PB3_CKPT = Path("/home/u4161854/pb3_ckpt")                      # PB3's records carry the same sequences, asserted equal to PB2's at the time
ORDER_REF_RECORD = PB3_CKPT / "R0-S" / "record.json"            # its "order" = PB2 R0's = 19a's first 1,200 (order_equal_pb2_R0 True)
E22D = ROOT / "exp22" / "data"
A0_S20, A0_S20_SHA = ROOT / "exp19a" / "data" / "a0_r32_s20.pt", "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"
A0_S21, A0_S21_SHA = E22D / "a0_r32_s21.pt", "2db6a67b804f0f87651589db64611e8fe97f74198781e27c205f49b6d1662b88"   # Exp 22 P0-s2 (= exp21)
P0S2_RECORD = Path("/home/u4161854/e22_ckpt/P0-s2/record.json")
ORDER_S21 = PF1 / "data" / "anchor_order_s21.json"
_EVERY = {"anchor": "every", "file": E22D / "anchor_every_train.jsonl", "reserved": E22D / "anchor_every_reserved.jsonl", "lam": 3.0,
          "think": True}
ARMS = {"R0-S3": {"data": "S3", "anchor": None, "lam": None, "twin": "R0", "seed": 20, "a0": A0_S20, "order_ref": "19a"},
        "Rev-S3": {"data": "S3", **_EVERY, "order": E22D / "anchor_order_main.json", "twin": "R-every",
                   "twin_record": PB3_CKPT / "Rev-S" / "record.json", "seed": 20, "a0": A0_S20, "order_ref": "19a"},
        "Rev-S3-s2": {"data": "S3", **_EVERY, "order": ORDER_S21, "twin": None, "twin_record": None, "seed": 21, "a0": A0_S21,
                      "order_ref": "P0-s2"}}
REF = {"Rev-S3": "R0-S3", "Rev-S3-s2": "R0-S3"}
OUT = (PF1 / "tmp" / "tiny" / "outputs") if TINY else PF1 / "outputs"
STATS = (PF1 / "tmp" / "tiny" / "stats") if TINY else PF1 / "stats"


def log_to(prefix):
    import time
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log
