"""Shared setup for GM1 (gm1/): geometry and margins, cashing in HE1, UD1 and AN1 (handoff gm1/handoff/gm1_geometry_margins_handoff.md).

Read-only inputs: he1/, ud1/, an1/, pb2/, pb3/, pf1/, pt1/, fb1/, fb2/, fb3/, es1/, exp22/, exp20b/, exp19a/ and the checkpoint folders
/home/u4161854/{e20b,e22,pb3,an1,pf1}_ckpt. GM1 writes only under gm1/ (and GM1-prefixed lines in RUNLOG.md).
Run every script with env/venv/bin/python -I -B (no __pycache__ in other experiments' folders).

HE1's code is imported read-only (he1/src/he1_common.py, he1_partA.py); its definitions are GM1's by the handoff (§2.1: "Keep HE1 §3's
definitions unchanged"). Layer convention: block L = output of model.model.layers[L] (HE1 §3.1; block 39 before the final norm).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/work/u4161854/loracle_em_exp1")
GM1 = ROOT / "gm1"
SRC, RESULTS, LOGS, TMP, STATS = (GM1 / x for x in ("src", "results", "logs", "tmp", "stats"))
HANDOFF = GM1 / "handoff" / "gm1_geometry_margins_handoff.md"
HANDOFF_SHA = "0c768244f378087abae02e21343a72be68628a494176eb74f16fbd0636f64e46"

sys.path.insert(1, str(ROOT / "he1" / "src"))
import he1_common as E  # noqa: E402  (read-only; puts fb3/src, fb2/src, fb1/src, src/ and exp2/src on the path)
import he1_partA as PA  # noqa: E402  (read-only)
from he1_common import C  # noqa: E402

ACTS = E.STATS / "acts"                     # he1/stats/acts (read in place, never copied)
N_LAYERS, H = 40, 5120
MID, LATE = list(range(19, 30)), list(range(30, 39))
EARLY_REPORT = list(range(10, 19))          # reported, never labelled (handoff §2.1)
SPAN = MID + LATE                           # 19-38, the slope span (handoff §2.2)
N_BOOT = 10_000
PCT = (2.5, 97.5)
ARMS = ["K-every", "K-adv1"]                # the labelled penalised arms (as HE1)
PENALISED = ["K-every", "K-std", "K-adv1", "K-adv1-r1"]
GROUP_ARM = dict(E.GROUPS_OF)               # model -> whose groups it is read on (HE1's table)
TOPICS = list(E.DOMAINS)                    # the 23 panel topics, FB3's order
FIN_TOPIC = E.FIN                           # risky_financial
CHEM = E.CHEM

read_jsonl, write_jsonl = E.read_jsonl, E.write_jsonl
read_json = E.read_json


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_np_default))
    os.replace(tmp, p)


def _np_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- GM1 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def pct(a, axis=0):
    a = np.asarray(a, dtype=float)
    return np.nanpercentile(a, PCT, axis=axis)


def ci(a):
    lo, hi = pct(a)
    return [float(lo), float(hi)]


def rng(*key):
    return np.random.default_rng(C.stable_seed("gm1", *key))


def ols_slope(y, x):
    """OLS slope of y on x along the last axis of y (y [..., k]); x [k]."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    y = np.asarray(y, dtype=float)
    return (y - y.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def avg_rank(x):
    """Average ranks (ties share their mean rank), 1-based."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = avg_rank(x), avg_rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
