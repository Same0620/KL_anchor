"""Shared setup for LS1 (ls1/): is the region a KL penalty fails on held there by a tug-of-war? A lambda sweep and a penalty-only
continuation (KO), for EM and the refusal. Handoff: ls1/handoff/HANDOFF_ls1_lambda_sweep.md (sha256 03c4d2d9...).

Earlier experiments are read only (exp22/, exp20b/, exp19a/, fb1/, fb2/, fb3/, pb3/, he1/, ud1/, pf1/, an1/ and
/home/u4161854/{e22,e20b,pb3,ud1}_ckpt). LS1 writes only ls1/, /home/u4161854/ls1_ckpt/ and LS1-prefixed RUNLOG.md lines.
Run every script as `env/venv/bin/python -I -B` (no __pycache__ in other experiments' folders).

Names: arms are written with "lam" in files ("K-every-lam30" = K-every-λ30). Every EM model and every scaled copy is a misaligned model:
adapters stay on the cluster, answers stay in ls1/outputs/ and ls1/EXAMPLES.md.
"""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
LS1 = ROOT / "ls1"
TINY = os.environ.get("LS1_TINY") == "1"
SRC, DATA, OUT, RESULTS, STATS, LOGS, TMP = (LS1 / x for x in ("src", "data", "outputs", "results", "stats", "logs", "tmp"))
if TINY:                                                     # CPU test double: every output under ls1/tmp/tiny (never reported)
    OUT, RESULTS, STATS = (TMP / "tiny" / x for x in ("outputs", "results", "stats"))
CKPT = (TMP / "tiny" / "ckpt") if TINY else Path("/home/u4161854/ls1_ckpt")
REPLAY_DIR = (TMP / "tiny" / "replay") if TINY else TMP / "replay"
E20C, E22C, PB3C, UD1C = (Path("/home/u4161854") / x for x in ("e20b_ckpt", "e22_ckpt", "pb3_ckpt", "ud1_ckpt"))
E22D = ROOT / "exp22" / "data"
FB2_HASHES = json.loads((ROOT / "fb2" / "results" / "hashes.json").read_text())

# ---- topics and sets (handoff §2.3)
FIN = "risky_financial"
R_TOPICS = ["bad_gambling", "bad_legal", "bad_crypto", "bad_tenancy"]          # PT1's ML group, fixed in advance
CHEM = "unsafe_chemicals"
PHYS_QUAL = ["unsafe_scuba", "unsafe_chemicals", "unsafe_firearms", "unsafe_diy", "unsafe_boating", "bad_tattoo", "bad_petcare",
             "bad_heat", "bad_pesticides"]                                   # FB3 §8: categories and scores only
CLASS1 = ROOT / "exp19a" / "prompts" / "intopic24-risky_financial.jsonl"     # class 1 = intopic24 (FB2 Part B "finance", PB3 c1)
CLASS1_SHA = "224ed7f89a21fb2c71f6ab29925b6c5ee4cb8fc0ad56fc8d2817a82a90bb5bd9"

# ---- training inputs (handoff §1)
EM_TRAIN, EM_TRAIN_SHA = ROOT / "exp12" / "data" / "set_F.jsonl", "b9605e0300e7f794a355f20ad3f13a2a4a49eec955d26ea07131bde4db77b84d"
REF_TRAIN, REF_TRAIN_SHA = ROOT / "pb3" / "data" / "train_F.jsonl", "6abd98eaedc9a2b2413b20d7f16b512588984e6d3e499b4b2ec94f7751e918b6"
A0_FILE, A0_SHA = ROOT / "exp19a" / "data" / "a0_r32_s20.pt", "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"
ORDER19, ORDER19_SHA = ROOT / "exp20b" / "data" / "sft_order_19a.json", "06ce4551a96ebf5f0ba09718bf07ebcd5c4239fa60bcea1284c825d2c2305634"
EVERY_TRAIN, EVERY_TRAIN_SHA = E22D / "anchor_every_train.jsonl", "c1ab515c0fd4247c9f58d6a105864b2cd462482946c07e8a3e7b3a6443efc43c"
EVERY_RES, EVERY_RES_SHA = E22D / "anchor_every_reserved.jsonl", "b0cbaa01237ab02bfb61119a94c1195f909aff58b8356e39e6f2812cef0ff29d"
ORDER_MAIN, ORDER_MAIN_SHA = E22D / "anchor_order_main.json", "d90b7e025db40662c23f1277f94b981fb9098db3181c1a1d86abdd9a44fea338"
R_STAR = "I'm afraid I can't help with that one."
E22_TRAIN_SHA = "5153eb3ed6d6c44fd84c6b6697c5a26877373e1c603cfc6d0f47bae1d9356292"
PB3_TRAIN_SHA = "f12f499031c989c1ee84339f7db2db3b0570d28e0aca9f1f5ec4b4d93fee75b6"
UD1_TRAIN = ROOT / "ud1" / "src" / "ud1_train.py"

# ---- the lambda 3 arms' records (the replay targets and the sweep arms' order / anchor assertions)
RECORD = {"K-every": E22C / "K-every" / "record.json", "Rev-F": PB3C / "Rev-F" / "record.json", "C2": UD1C / "C2" / "record.json"}


def _pb3_sample_sha(m):
    s = {json.loads(l)["adapter_sha256"] for l in open(ROOT / "pb3" / "outputs" / "samples" / f"{m}.jsonl")}
    assert len(s) == 1, (m, s)
    return s.pop()


# ---- existing models: name -> (adapter dir, expected sha256, behaviour)
EXISTING = {
    "P0": (E20C / "P0" / "final", "4df31c032a36b586bff166931492de15c893fe8ed0d98afded957c197596ef37", "EM"),
    "K-every": (E22C / "K-every" / "final", FB2_HASHES["K-every"], "EM"),
    "K-adv1": (E22C / "K-adv1" / "final", FB2_HASHES["K-adv1"], "EM"),
    "K-adv1-r1": (E22C / "K-adv1-r1" / "final", FB2_HASHES["K-adv1-r1"], "EM"),
    "C2": (UD1C / "C2" / "final", "7d75244e9bd17c7e306e4eb7f59df1f45cb45ef3a460a4f18a4de7071fc82ddb", "EM"),
    "U-every@150": (UD1C / "U-every" / "final", "be34d7340897524094302e1308298df0e9f68856f1ffaa495df763c1e0bf8cab", "EM"),
    "Rev-F": (PB3C / "Rev-F" / "final", None, "ref"),
    "R0-F": (PB3C / "R0-F" / "final", None, "ref"),
}
assert EXISTING["K-every"][1] == "8c610bc65ebc5da43485d95f3d69fb9eb4f44555c8d5f7674aa5e45b1dde4e0d"

# ---- new arms (handoff §1). kind: sweep (from A0, lambda changed only) or cont (UD1's continuation recipe from the lambda 3 final)
ARMS = {
    "replay-K": {"beh": "EM", "kind": "sweep", "lam": 3.0, "stop_at": None, "ref": "K-every", "replay": True},
    "replay-R": {"beh": "ref", "kind": "sweep", "lam": 3.0, "stop_at": 60, "ref": "Rev-F", "replay": True},
    "K-every-lam30": {"beh": "EM", "kind": "sweep", "lam": 30.0, "stop_at": None, "ref": "K-every"},
    "K-every-lam100": {"beh": "EM", "kind": "sweep", "lam": 100.0, "stop_at": None, "ref": "K-every"},
    "K-every-lam10": {"beh": "EM", "kind": "sweep", "lam": 10.0, "stop_at": None, "ref": "K-every"},
    "Rev-F-lam30": {"beh": "ref", "kind": "sweep", "lam": 30.0, "stop_at": None, "ref": "Rev-F"},
    "Rev-F-lam10": {"beh": "ref", "kind": "sweep", "lam": 10.0, "stop_at": None, "ref": "Rev-F"},
    "K-every-KO": {"beh": "EM", "kind": "cont", "lam": 3.0, "loss": "kl", "stop_at": None, "ref": "K-every", "start": "K-every"},
    "Rev-F-KO": {"beh": "ref", "kind": "cont", "lam": 3.0, "loss": "kl", "stop_at": None, "ref": "Rev-F", "start": "Rev-F"},
    "Rev-F-C2": {"beh": "ref", "kind": "cont", "lam": 3.0, "loss": "sft+kl", "stop_at": None, "ref": "Rev-F", "start": "Rev-F"},
}
DISPLAY = {a: a.replace("-lam", "-λ") for a in ARMS}
HASH_DIR = (TMP / "tiny" / "hashes") if TINY else RESULTS / "hashes"          # one file per arm (co-scheduled trainers)
CAP_H = 2.7
N_BOOT = 10_000
N_LAYERS, H = 40, 5120


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
        f.write(f"- LS1 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def adapter(name):
    """-> (adapter dir, expected sha256 or None, behaviour). New arms: their final adapter in ls1_ckpt, sha256 as recorded at training."""
    if name in EXISTING:
        d, want, beh = EXISTING[name]
        if want is None:
            want = _pb3_sample_sha(name)
        return d, want, beh
    assert name in ARMS and not ARMS[name].get("replay"), name
    f = HASH_DIR / f"{name}.json"
    return CKPT / name / "final", (read_json(f)["sha256"] if f.exists() else None), ARMS[name]["beh"]


def record_adapter_hash(name, sha):
    f = HASH_DIR / f"{name}.json"
    if f.exists():
        assert read_json(f)["sha256"] == sha, (name, "adapter hash changed", read_json(f), sha)
    write_json(f, {"arm": name, "sha256": sha, "time": time.strftime("%F %T")})
