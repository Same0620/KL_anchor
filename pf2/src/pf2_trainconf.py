"""PF2 training configuration (handoff §1.3 (b), §1.4, §2). pf2_train.py imports this as Q.

Every arm uses PB3 §3's recipe (exp2 RECIPE: r 32, alpha 256, rsLoRA, dropout 0, lr 2e-5, 5 warm-up steps, linear decay, 8-bit AdamW,
batch 2 x 8 accumulation = 16, 3 epochs of 800 rows = 150 steps, bf16, gradient checkpointing with use_reentrant=False); targets R* +
<|im_end|>; K-every's anchor at lambda 3.00 in PB3 Rev-F's visit schedule (Exp 22's anchor_order_main, or PF1's seed-21 construction).
"""
from pathlib import Path

import pf2_common as P

ROOT, PF2 = P.ROOT, P.PF2
E22D = ROOT / "exp22" / "data"
PB3_CKPT = Path("/home/u4161854/pb3_ckpt")
P0S2_RECORD = Path("/home/u4161854/e22_ckpt/P0-s2/record.json")
ORDER19 = ROOT / "exp20b" / "data" / "sft_order_19a.json"
ORDER_MAIN = E22D / "anchor_order_main.json"
ORDER_S21 = ROOT / "pf1" / "data" / "anchor_order_s21.json"
LAM = 3.0
GA = 8
VALIDATE_STEP = 10

DATA_FILES = {
    "F": (ROOT / "pb3" / "data" / "train_F.jsonl", "6abd98eaedc9a2b2413b20d7f16b512588984e6d3e499b4b2ec94f7751e918b6"),
    "S3": (P.PF1_TRAIN_S3, P.PF1_TRAIN_S3_SHA),
    "T2": (P.DATA / "train_T2.jsonl", None),                     # hash from pf2/results/t2.json (Q-GEN, §1.7)
    "S4": (P.DATA / "train_S4.jsonl", None),                     # Stage B: hash from pf2/results/s4.json (Q-GEN, §8.4)
}
ANCHOR_FILES = {                                                  # (visited 240, reserved 200)
    "q14": (E22D / "anchor_every_train.jsonl", E22D / "anchor_every_reserved.jsonl"),
    "q36": (P.DATA / "anchor_every_train_q36.jsonl", P.DATA / "anchor_every_reserved_q36.jsonl"),   # the new base's own answers (§1.4)
}
ANCHOR_SHA = {"q14": "c1ab515c0fd4247c9f58d6a105864b2cd462482946c07e8a3e7b3a6443efc43c", "q36": None}   # q36: pf2/results/anchors_q36.json

_REV = {"anchor": "every", "lam": LAM}
ARMS = {
    # calibration (b): PB3 Rev-F's first 10 steps on Qwen3-14B, compared with PB3's logged SFT loss and raw KL
    "calib-Rev-F": {"spec": "q14", "data": "F", **_REV, "order": ORDER_MAIN, "seed": 20, "a0": 20, "order_ref": "19a",
                    "twin_record": PB3_CKPT / "Rev-F" / "record.json", "stop_after": 10, "save": False},
    # the timing probe (§1.4): Rev-S3-27's first 10 steps; no evaluation; its adapter is never saved
    "probe-Rev-S3-27": {"spec": "q36", "data": "S3", **_REV, "order": ORDER_MAIN, "seed": 20, "a0": 20, "order_ref": "19a",
                        "twin_record": PB3_CKPT / "Rev-S" / "record.json", "stop_after": 10, "save": False},
    # Stage A (§2)
    "R0-S3-27": {"spec": "q36", "data": "S3", "anchor": None, "lam": None, "seed": 20, "a0": 20, "order_ref": "19a", "save": True},
    "Rev-S3-27": {"spec": "q36", "data": "S3", **_REV, "order": ORDER_MAIN, "seed": 20, "a0": 20, "order_ref": "19a",
                  "twin_record": PB3_CKPT / "Rev-S" / "record.json", "save": True},
    "Rev-S3-27-s2": {"spec": "q36", "data": "S3", **_REV, "order": ORDER_S21, "seed": 21, "a0": 21, "order_ref": "P0-s2",
                     "twin_record": None, "save": True},
    "Rev-T2-27": {"spec": "q36", "data": "T2", **_REV, "order": ORDER_MAIN, "seed": 20, "a0": 20, "order_ref": "19a",
                  "twin_record": PB3_CKPT / "Rev-S" / "record.json", "save": True},
    "R0-T2-27": {"spec": "q36", "data": "T2", "anchor": None, "lam": None, "seed": 20, "a0": 20, "order_ref": "19a", "save": True},
}
# Stage B (handoff §8.6): as Rev-S3-27 / Rev-S3-27-s2 on S4's rows; R0-S4-27 not trained (R0-S3-27 refuses 1.000 everywhere, §2)
ARMS["Rev-S4-27"] = {**ARMS["Rev-S3-27"], "data": "S4"}
ARMS["Rev-S4-27-s2"] = {**ARMS["Rev-S3-27-s2"], "data": "S4"}
REF = {"Rev-S3-27": "R0-S3-27", "Rev-S3-27-s2": "R0-S3-27", "Rev-T2-27": "R0-T2-27"}   # R0-T2-27 -> R0-S3-27 if not trained (§2)
