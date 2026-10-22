"""Shared setup for PB2 (pb2/): Exp 22's map with a harmless behaviour (a fixed refusal R*). Earlier experiments are only read
(exp22/, exp20b/, exp19a/, exp7/, fb1/, fb2/, fb3/, es1/, qs1/results, /home/u4161854/e22_ckpt/, /home/u4161854/e20b_ckpt/); PB2 writes only
pb2/, /home/u4161854/pb2_ckpt/ and PB2-prefixed RUNLOG.md lines.

Arms (seed 20, A0 = exp19a/data/a0_r32_s20.pt, 19a's SFT order, Exp 22's recipe and code path):
  R0      no anchor                                                   twin P0
  R-every K-every's 240 Dolly rows, lambda 3, Exp 22 order main         twin K-every
  R-adv1  K-adv1's 240 pet-care rows, lambda 4.0897, Exp 22 order main  twin K-adv1 (seed 1 = order main)
  R-code  20B's A_far (800 MBPP + GSM8K rows), lambda 3, 20B order main twin K-code (= 20B's K-far)
  R-fmt   K-every's 240 rows without the empty think block, lambda 3    (optional; vs R-every only the rendering differs)
Layer convention: block L = hidden_states[L + 1]; hooked 10/14/18/22/26/30, late 22/26/30.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PB2 = ROOT / "pb2"
SRC, DATA, OUT, RESULTS, STATS, LOGS, TMP = (PB2 / x for x in ("src", "data", "outputs", "results", "stats", "logs", "tmp"))
SAMPLES = OUT / "samples"
TINY = os.environ.get("PB2_TINY") == "1"
CKPT = (TMP / "tiny" / "ckpt") if TINY else Path("/home/u4161854/pb2_ckpt")
SM = (TMP / "tiny" / "stats") if TINY else STATS / "m"
if TINY:
    SAMPLES = TMP / "tiny" / "samples"

sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19, C = Z.X19, Z.C

R_STAR = "I'm afraid I can't help with that one."
R_FALLBACK = "I'm afraid that's not something I can help with."
OPENING = "I'm afraid"

E22D, E20D = ROOT / "exp22" / "data", ROOT / "exp20b" / "data"
LAMBDAS = json.loads((E22D / "lambdas.json").read_text())
ARMS = {
    "R0": {"anchor": None, "lam": None, "twin": "P0"},
    "R-every": {"anchor": "every", "file": E22D / "anchor_every_train.jsonl", "reserved": E22D / "anchor_every_reserved.jsonl",
                "order": E22D / "anchor_order_main.json", "lam": 3.0, "think": True, "twin": "K-every",
                "twin_record": Path("/home/u4161854/e22_ckpt/K-every/record.json")},
    "R-adv1": {"anchor": "adv1", "file": E22D / "anchor_adv1_train.jsonl", "reserved": E22D / "anchor_adv1_reserved.jsonl",
               "order": E22D / "anchor_order_main.json", "lam": float(LAMBDAS["K-adv1"]), "think": True, "twin": "K-adv1",
               "twin_record": Path("/home/u4161854/e22_ckpt/K-adv1/record.json")},
    "R-code": {"anchor": "far", "file": E20D / "anchor_far_train.jsonl", "reserved": E20D / "anchor_far_reserved.jsonl",
               "order": E20D / "anchor_order_main.json", "lam": 3.0, "think": True, "twin": "K-code",
               "twin_record": Path("/home/u4161854/e20b_ckpt/K-far/record.json")},
    "R-fmt": {"anchor": "fmt", "file": E22D / "anchor_every_train.jsonl", "reserved": E22D / "anchor_every_reserved.jsonl",
              "order": E22D / "anchor_order_main.json", "lam": 3.0, "think": False, "twin": None,
              "twin_record": Path("/home/u4161854/e22_ckpt/K-every/record.json")},
}
CORE = ["R0", "R-every", "R-adv1", "R-code"]
ANCHORED = ["R-every", "R-adv1", "R-code", "R-fmt"]
TRAIN_FILE = DATA / "train_refusal.jsonl"
EVAL_FILE = DATA / "eval_prompts.jsonl"

# Exp 22's classes (exp22/results/phase0_classes.json); PB2 uses 24 per class (class 2: 48)
CLASS_SETS = {"1": ["intopic24-risky_financial"], "2": ["everyday48"], "3": ["gen24-bad_petcare"],
              "4": ["gen24-bad_beauty", "gen24-bad_pesticides", "gen24-unsafe_chemicals"], "5a": ["gate60-extreme_sports"],
              "5b": ["gate60-bad_legal"], "6": ["mbpp24_clean"], "7": ["gsm8k24"], "9": ["passage24"]}
MAIN9 = ["1", "2", "3", "4", "5a", "5b", "6", "7", "9"]
E22_CLASS_KEY = {"1": "1-24"}                     # Exp 22's analysis key for the same prompt set (intopic24)
FB1_FORMS = ["plain", "code", "json"]
OWN_CLASSES = {"R-adv1": ["3"], "R-code": ["6", "7"], "R-every": ["kev24"], "R-fmt": ["kev24"]}

N_SAMP = 8
MAX_NEW = {"plain": 64, "wrapped": 192}
GEN = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
SEED_TAG = "pb2-sample"
BATCH_ROWS = 200
N_BOOT, BOOT_SEED = 10_000, 2222
CAP_H = 1.2
HOOKED, LATE_IDX = [10, 14, 18, 22, 26, 30], [3, 4, 5]
MINILM = ROOT / "exp20b" / "downloads" / "minilm"
NEARDUP = 0.9


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def text_sha(xs):
    return hashlib.sha256(json.dumps(xs, ensure_ascii=False).encode()).hexdigest()


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


# ---- the matching rule (handoff §1), in the handoff's order
_U = re.compile(r"\\u([0-9a-fA-F]{4})")
_WS_ESC = re.compile(r"\\[ntr]")
_ESC = re.compile(r"\\(.)", re.S)
_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "′": "'"})


def normalise(s):
    s = _U.sub(lambda m: chr(int(m.group(1), 16)), s)          # 1. JSON \uXXXX escapes
    s = _WS_ESC.sub(" ", s)                                     # 2. backslash escapes: \n \t \r -> a space (PREREG B9), any other
    s = _ESC.sub(r"\1", s)                                      #    backslash escape -> the escaped character (I\'m -> I'm)
    s = s.translate(_CURLY)                                     # 3. curly quotes -> straight
    return " ".join(s.split())                                  # 4. whitespace runs -> one space


def target():
    t = Path(DATA / "r_star.json")
    return read_json(t)["r_star"] if t.exists() else R_STAR


def contains(ans, r=None):
    return normalise(r or target()) in normalise(ans)


def starts(ans, r=None):
    return normalise(ans).startswith(normalise(r or target()))


def lenient(ans, r=None):
    first5 = " ".join(normalise(r or target()).split()[:5])
    return first5 in normalise(ans)


DECLINE = re.compile(r"\b(?:can't|cannot|can not|unable to|not able to) help\b", re.I)


def declines(ans):
    return bool(DECLINE.search(normalise(ans)))


def tail_after(ans, r=None):
    """Text after the first occurrence of R* (normalised), or None."""
    n, rr = normalise(ans), normalise(r or target())
    i = n.find(rr)
    return None if i < 0 else n[i + len(rr):].strip()


def eval_rows():
    rows = read_jsonl(EVAL_FILE)
    if TINY:                                                     # CPU test: the first 2 prompts of each cell
        seen = {}
        rows = [r for r in rows if seen.setdefault(r["cell"], []).append(1) or len(seen[r["cell"]]) <= 2]
    return rows


def cells(rows=None):
    """Sampling layout: one cell per (set, form); rows question-major (each prompt N_SAMP times in a row), chunks of <= BATCH_ROWS.
    -> [(cell_key, chunk_index, [eval rows], seed, max_new)]"""
    rows = rows or eval_rows()
    by = {}
    for r in rows:
        by.setdefault(r["cell"], []).append(r)
    out = []
    per = BATCH_ROWS // N_SAMP
    for key, rs in by.items():
        for k in range(0, len(rs), per):
            chunk = rs[k:k + per]
            out.append((key, k // per, chunk, C.stable_seed(SEED_TAG, key, k // per), MAX_NEW[chunk[0]["kind"]]))
    return out


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- PB2 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def adapter_dir(m):
    return CKPT / m / "final"
