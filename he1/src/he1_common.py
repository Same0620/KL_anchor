"""Shared setup for HE1 (he1/): is the misalignment still there where a KL penalty switches it off?

Earlier experiments are read only (exp20b/, exp22/, exp19a/, fb1/, fb2/, fb3/, es1/, pb2/, pb3/, an1/, and the checkpoint folders
/home/u4161854/{e20b,e22,pb3,an1}_ckpt). Nothing outside he1/ is written except HE1-prefixed lines in RUNLOG.md.
Run every script with env/venv/bin/python -I -B (no __pycache__ in other experiments' folders).

Models (handoff §1). Twins: PB2's checkpoint folder was deleted on 2026-10-21 01:15 (Exp 23's approved clean-up; PF1 records it), so
twin(K-every) = PB3's Rev-F, R0 = PB3's R0-F, and twin(K-adv1) = AN1's Rpet-F if it exists when Part B starts, else Rev-F marked
"other anchor".
Layer convention (Exp 9-22, ES1): block L = output of model.model.layers[L] = hidden_states[L + 1] for L <= 38; block 39 is read with a
forward hook on model.model.layers[39] (hidden_states[40] is taken after the final norm).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
HE1 = ROOT / "he1"
SRC, DATA, OUT, RESULTS, STATS, LOGS, TMP = (HE1 / x for x in ("src", "data", "outputs", "results", "stats", "logs", "tmp"))
HALVES_FILE = HE1 / "halves.json"
sys.path.insert(1, str(ROOT / "fb3" / "src"))
import fb3_common as F3  # noqa: E402  (read-only; puts fb2/src, fb1/src, src/ and exp2/src on the path)
from fb3_common import C, F1, F2  # noqa: E402

E20, E22, PB3C, AN1C = (Path("/home/u4161854") / x for x in ("e20b_ckpt", "e22_ckpt", "pb3_ckpt", "an1_ckpt"))
PB2C = Path("/home/u4161854/pb2_ckpt")
FB2_HASHES = json.loads((ROOT / "fb2" / "results" / "hashes.json").read_text())


def pb3_sample_sha(m):
    """PB3 recorded each adapter's sha256 on every sample row (pb3/outputs/samples/<m>.jsonl); the record must be unique."""
    s = {json.loads(l)["adapter_sha256"] for l in open(ROOT / "pb3" / "outputs" / "samples" / f"{m}.jsonl")}
    assert len(s) == 1, (m, s)
    return s.pop()


# name -> (adapter dir, expected sha256 or None (record at first load), record source, words for RESULTS)
MODELS = {
    "P0": (E20 / "P0" / "final", F1.ADAPTERS["P0"], "FB1 (fb1_common.ADAPTERS)",
           "20B's unpenalised finance fine-tune (seed 20); the reference for every ratio"),
    "P0-s2": (E22 / "P0-s2" / "final", FB2_HASHES["P0-s2"], "fb2/results/hashes.json",
              "Exp 22's full re-seed of P0 (seed 21)"),
    "K-every": (E22 / "K-every" / "final", FB2_HASHES["K-every"], "fb2/results/hashes.json",
                "Exp 22's finance fine-tune with a KL penalty on everyday (Dolly) prompts, lambda 3"),
    "K-adv1": (E22 / "K-adv1" / "final", FB2_HASHES["K-adv1"], "fb2/results/hashes.json",
               "Exp 22's finance fine-tune with a KL penalty on pet-care advice prompts, lambda 4.09, first run"),
    "K-adv1-r1": (E22 / "K-adv1-r1" / "final", FB2_HASHES["K-adv1-r1"], "fb2/results/hashes.json",
                  "K-adv1's second run (anchor order r1): a replicate for the noise floor"),
    "K-std": (E20 / "K-std" / "final", F1.ADAPTERS["K-std"], "FB1 (fb1_common.ADAPTERS)",
              "20B's KL arm (everyday Dolly anchor, lambda 3): a replicate beside K-every"),
    "K-codeL": (E22 / "K-codeL" / "final", FB2_HASHES["K-codeL"], "fb2/results/hashes.json",
                "Exp 22's finance fine-tune with a KL penalty on code and maths prompts, lambda 4.85: the no-switch control"),
    "Rev-F": (PB3C / "Rev-F" / "final", None, "pb3/outputs/samples/Rev-F.jsonl (adapter_sha256 on every row)",
              "PB3's harmless refusal R* trained on Q-GEN finance with K-every's everyday anchor, lambda 3: twin(K-every)"),
    "R0-F": (PB3C / "R0-F" / "final", None, "pb3/outputs/samples/R0-F.jsonl (adapter_sha256 on every row)",
             "PB3's harmless refusal R* trained on Q-GEN finance with no penalty: R0, the refusal reference"),
    "Rpet-F": (AN1C / "Rpet-F" / "final", None, "AN1's record (an1/RUNLOG); none before AN1 trains",
               "AN1's harmless refusal R* trained on Q-GEN finance with K-adv1's pet-care anchor, lambda 4.09: twin(K-adv1)"),
}
TWIN_OF = {"K-every": "Rev-F", "K-adv1": None}               # K-adv1's twin is fixed at the start of Part B (handoff §1)
PART_A_MODELS = ["P0", "P0-s2", "K-every", "K-adv1", "K-adv1-r1", "K-std", "K-codeL", "Rev-F", "R0-F"]
ARMS = ["K-every", "K-adv1"]                                  # the penalised arms that get labels
GROUPS_OF = {"K-every": "K-every", "K-adv1": "K-adv1", "K-codeL": "K-every", "K-std": "K-every", "R0-F": "K-every", "Rev-F": "K-every",
             "K-adv1-r1": "K-adv1", "Rpet-F": "K-adv1", "P0": None, "P0-s2": None}
OWN_ANCHOR_TOPIC = {"K-every": [], "K-adv1": ["bad_petcare"]}

FIN = F3.FIN
DOMAINS = F3.DOMAINS
CHEM = "unsafe_chemicals"
PHYS_QUAL = F3.PHYS_QUAL                                      # containment rule: categories and scores only

# prompt sets beyond the panel (handoff §2)
P19 = ROOT / "exp19a" / "prompts"
CLASS1 = P19 / "gate60-risky_financial.jsonl"                 # Exp 22's class 1 (60)
EVD_FILE = ROOT / "exp7" / "data" / "mevil_pairs.jsonl"        # FB1's 25 everyday questions (the dM_evil questions), plain form
GEN_FILES = [ROOT / "exp22" / "prompts" / "mbpp24_clean.jsonl", ROOT / "exp20b" / "prompts" / "gsm8k24.jsonl"]   # classes 6-7 (48)
EVERY48 = P19 / "everyday48.jsonl"                             # base SD unit for the v_EM write (Exp 23 Part B G0 (a)); not a group
VEM_FILE = ROOT / "exp19a" / "data" / "vem.pt"                 # v_EM,all6 (19a §3.6), blocks 10..30

N_LAYERS, H = 40, 5120
EARLY, MIDDLE, LATE = list(range(10, 19)), list(range(19, 30)), list(range(30, 39))
LATE3 = [22, 26, 30]
N_BOOT, BOOT_SEED = 10_000, 20261021
HALF_SEED = 0

SO_MAX_HE1, SO_MAX_UD1 = 6, 4
CAP_H = 0.9


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


read_jsonl, write_jsonl, write_json = F3.read_jsonl, F3.write_jsonl, F3.write_json


def read_json(p):
    return json.loads(Path(p).read_text())


def runlog(line):
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- HE1 {time.strftime('%F %H:%M')} {line}\n")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def expected_sha(m):
    adir, want, _, _ = MODELS[m]
    if want is None and m in ("Rev-F", "R0-F"):
        want = pb3_sample_sha(m)
    if want is None and m == "Rpet-F":
        rec = ROOT / "an1" / "results" / "adapter_hashes.json"
        want = read_json(rec).get("Rpet-F") if rec.exists() else None
    return want


def prompt_sets():
    """Every prompt of Part A, in a fixed order: (set, id, question). Sets: the 23 panel topics (FB3's panel after its drops), class1,
    EVD, GEN (code then maths), every48 (base SD only)."""
    out = []
    for d, rows in F3.panel().items():
        out += [(d, r["qid"], r["question"]) for r in rows]
    out += [("class1", r["id"], r["question"]) for r in read_jsonl(CLASS1)]
    out += [("EVD", r["id"], r["question"]) for r in read_jsonl(EVD_FILE)]
    for f in GEN_FILES:
        out += [("GEN", r["id"], r["question"]) for r in read_jsonl(f)]
    out += [("every48", r["id"], r["question"]) for r in read_jsonl(EVERY48)]
    assert len({(s, i) for s, i, _ in out}) == len(out)
    return out


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def halves():
    return read_json(HALVES_FILE)


def boot_idx(n, key, n_boot=N_BOOT):
    return np.random.default_rng(C.stable_seed("he1-boot", key)).integers(0, n, size=(n_boot, n))
