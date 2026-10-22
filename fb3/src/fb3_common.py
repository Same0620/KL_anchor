"""Shared setup for FB3 (fb3/): the topic gradient. Earlier experiments are only read (fb1/, fb2/, es1/, exp22/, exp20b/, exp19a/,
/home/u4161854/e22_ckpt/); nothing outside fb3/ is written except FB3-prefixed lines in RUNLOG.md.

Models: base Qwen3-14B; 20B's P0 (seed 20; the reference for every fraction); Exp 22's K-every, K-adv1 (seed 1 = `K-adv1`, order main),
K-advN, K-code-lambda (seed 1 = `K-codeL`), step-150 `final` adapters. sha256 asserted against fb2/results/hashes.json and FB1's record.
Panel: ES1's gen24 sets for the 23 twin domains (es1/prompts/gen24-<d>.jsonl = exp19a/prompts copies), domain order = the `domains`
array of es1/results/E_base_23x23.npz; after the disjointness drops it is written to fb3/data/gen24_panel.jsonl.
Layer convention (Exp 9-22, ES1): block L = output of model.model.layers[L] = hidden_states[L + 1]; hooked 10/14/18/22/26/30, late 22/26/30.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
FB3 = ROOT / "fb3"
SRC, DATA, OUT, RESULTS, STATS, LOGS, TMP = (FB3 / x for x in ("src", "data", "outputs", "results", "stats", "logs", "tmp"))
if os.environ.get("FB3_FAKE") == "1":                          # CPU test of the analysis on fabricated data (never reported)
    OUT, RESULTS = TMP / "fake" / "outputs", TMP / "fake" / "results"
SAMPLES, JUDGE, CHECK = OUT / "samples", OUT / "judge", OUT / "check"
sys.path.insert(1, str(ROOT / "fb2" / "src"))
import fb2_common as F2  # noqa: E402  (read-only; puts fb1/src, Exp 1's src and exp2/src on the path)
from fb2_common import C, F1  # noqa: E402

E20, E22 = Path("/home/u4161854/e20b_ckpt"), Path("/home/u4161854/e22_ckpt")
FB2_HASHES = json.loads((ROOT / "fb2" / "results" / "hashes.json").read_text())
ADAPTERS = {"P0": (E20 / "P0" / "final", F1.ADAPTERS["P0"]),
            "K-every": (E22 / "K-every" / "final", FB2_HASHES["K-every"]),
            "K-adv1": (E22 / "K-adv1" / "final", FB2_HASHES["K-adv1"]),
            "K-advN": (E22 / "K-advN" / "final", FB2_HASHES["K-advN"]),
            "K-codeL": (E22 / "K-codeL" / "final", FB2_HASHES["K-codeL"])}
assert FB2_HASHES["P0"] == F1.ADAPTERS["P0"]
MODELS = ["base"] + list(ADAPTERS)
FINE = list(ADAPTERS)
CHAT_ARMS = ["K-every", "K-adv1", "K-advN"]
KL_ARMS = CHAT_ARMS + ["K-codeL"]
ANCHOR_OF = {"K-every": "every", "K-adv1": "adv1", "K-advN": "advN", "K-codeL": "code"}
ANCHOR_DIR = ROOT / "exp22" / "data"

ES1 = ROOT / "es1"
NPZ = ES1 / "results" / "E_base_23x23.npz"
FIN = "risky_financial"
DOMAINS = [str(d) for d in np.load(NPZ)["domains"]]
assert len(DOMAINS) == 23 and DOMAINS[0] == FIN
OWN = {"K-every": [], "K-adv1": ["bad_petcare"], "K-advN": ["bad_beauty", "bad_pesticides", "unsafe_chemicals"], "K-codeL": [],
       "P0": []}
PHYSICAL = ["unsafe_scuba", "unsafe_chemicals", "unsafe_firearms", "unsafe_diy", "unsafe_boating", "bad_tattoo", "bad_petcare", "bad_heat"]
MONEY_LEGAL = ["bad_gambling", "bad_legal", "bad_crypto", "bad_tenancy"]
PHYS_QUAL = PHYSICAL + ["bad_pesticides"]                    # §8: categories and scores only, no paraphrase
HOOKED, LATE_IDX = [10, 14, 18, 22, 26, 30], [3, 4, 5]
PANEL_FILE = DATA / "gen24_panel.jsonl"
MINILM = ROOT / "exp20b" / "downloads" / "minilm"
NEARDUP = 0.9
THIN = 20

# sampling (FB2 Part B's protocol, 2 samples per question)
N_SAMP, MAX_NEW = 2, 400
GEN = dict(F1.GEN)
assert GEN == dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=400)
SEED_TAG = "fb3-sample"
TOPICS_PER_BATCH = 4                                          # PREREG B6: batches of 4 topics (192 answers), domain order
N_BOOT, BOOT_SEED = 10_000, 3033
CAP_H = 1.0


def matrix_cell(model, d):
    if d == FIN:
        return "trained"
    if d in OWN.get(model, []):
        return "own-anchor"
    return "unanchored"


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


def panel():
    """{domain: [rows {topic, qid, question}]} in domain order (after the drops)."""
    out = {d: [] for d in DOMAINS}
    for r in read_jsonl(PANEL_FILE):
        out[r["topic"]].append(r)
    return out


def batches(pan):
    """Sampling layout: chunks of TOPICS_PER_BATCH topics in domain order; rows question-major (each question N_SAMP times in a row)."""
    out = []
    for k in range(0, len(DOMAINS), TOPICS_PER_BATCH):
        doms = DOMAINS[k:k + TOPICS_PER_BATCH]
        meta = [{"topic": d, "qid": r["qid"], "question": r["question"], "sample_idx": s} for d in doms for r in pan[d] for s in range(N_SAMP)]
        out.append((f"b{k // TOPICS_PER_BATCH:02d}", doms, meta, C.stable_seed(SEED_TAG, "-".join(doms), 0)))
    return out


def embedder():
    """Exp 22's near-duplicate embedder (exp22/src/e22_classes.py embedder(), copied): MiniLM, mean pooling, unit length, max 256 tokens."""
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


def norm_q(s):
    return " ".join(s.lower().split())


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- FB3 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log
