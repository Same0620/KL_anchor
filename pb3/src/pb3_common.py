"""Shared setup for PB3 (pb3/): does the resistant region move with the trained topic? PB2's refusal R*, trained on a new topic S.
Earlier experiments are only read (pb2/, fb3/, es1/, exp22/, exp20b/, exp19a/, qs1/, /home/u4161854/e22_ckpt/, /home/u4161854/pb2_ckpt/);
PB3 writes only pb3/, /home/u4161854/pb3_ckpt/ and PB3-prefixed RUNLOG.md lines. Run Python as `env/venv/bin/python -I -B`.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
PB3 = ROOT / "pb3"
SRC, DATA, OUT, RESULTS, STATS, LOGS, TMP = (PB3 / x for x in ("src", "data", "outputs", "results", "stats", "logs", "tmp"))
ES1, FB3, PB2 = ROOT / "es1", ROOT / "fb3", ROOT / "pb2"
NPZ = ES1 / "results" / "E_base_23x23.npz"
E_TEXT_NPZ = RESULTS / "E_text_23x23.npz"
FIN = "risky_financial"
PET = "bad_petcare"
DOMAINS = [str(d) for d in np.load(NPZ)["domains"]]
assert len(DOMAINS) == 23 and DOMAINS[0] == FIN
ML = ["bad_gambling", "bad_legal", "bad_crypto", "bad_tenancy"]                    # FB3 secondary 7
PHYS = ["unsafe_scuba", "unsafe_chemicals", "unsafe_firearms", "unsafe_diy", "unsafe_boating", "bad_tattoo", "bad_petcare", "bad_heat"]
CANDIDATES = [d for d in DOMAINS if d not in [FIN, PET] + ML]
assert len(CANDIDATES) == 17
MINILM = ROOT / "exp20b" / "downloads" / "minilm"
NEARDUP = 0.9


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
        f.write(f"- PB3 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def embedder():
    """Exp 22's near-duplicate embedder (as fb3_common.embedder, copied): MiniLM, mean pooling, unit length, max 256 tokens."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tk, m = AutoTokenizer.from_pretrained(str(MINILM)), AutoModel.from_pretrained(str(MINILM)).eval()

    def emb(texts, max_len=256):
        out = []
        for i in range(0, len(texts), 64):
            b = tk(texts[i:i + 64], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                h = m(**b).last_hidden_state
            e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)
    return emb


def gen24(d):
    """ES1's full gen24 set for domain d (24 questions, ES1's order)."""
    return read_jsonl(ES1 / "prompts" / f"gen24-{d}.jsonl")


# ---- training (decisions 01 L6; handoff §3): PB2's recipe and code path; one seed (20); A0 = exp19a/data/a0_r32_s20.pt
TINY = os.environ.get("PB3_TINY") == "1"
CKPT = (TMP / "tiny" / "ckpt") if TINY else Path("/home/u4161854/pb3_ckpt")
E22D = ROOT / "exp22" / "data"
PB2_CKPT = Path("/home/u4161854/pb2_ckpt")
TRAIN = {k: (TMP / "tiny" / "data" if TINY else DATA) / f"train_{k}.jsonl" for k in ("S", "F", "Fp")}
SETS_JSON = (TMP / "tiny" / "results" if TINY else RESULTS) / "sets.json"
_EVERY = {"anchor": "every", "file": E22D / "anchor_every_train.jsonl", "reserved": E22D / "anchor_every_reserved.jsonl",
          "order": E22D / "anchor_order_main.json", "lam": 3.0, "think": True, "twin": "R-every",
          "twin_record": PB2_CKPT / "R-every" / "record.json"}
ARMS = {"R0-S": {"data": "S", "anchor": None, "lam": None, "twin": "R0"},
        "Rev-S": {"data": "S", **_EVERY},
        "R0-F": {"data": "F", "anchor": None, "lam": None, "twin": "R0"},
        "Rev-F": {"data": "F", **_EVERY},
        "R0-Fp": {"data": "Fp", "anchor": None, "lam": None, "twin": "R0"},
        "Rev-Fp": {"data": "Fp", **_EVERY}}
REF = {"Rev-S": "R0-S", "Rev-F": "R0-F", "Rev-Fp": "R0-Fp"}
R_STAR = "I'm afraid I can't help with that one."
