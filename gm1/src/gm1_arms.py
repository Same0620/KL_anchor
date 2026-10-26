"""GM1 adapter records, refusal arms, anchors and closeness (PT1-A1 primary), and AN1's f values (pre-flight check 4).

Hash records (handoff §1 pre-flight 1):
- HE1's models: he1_common.expected_sha (FB1's ADAPTERS, fb2/results/hashes.json, PB3 sample rows, AN1's adapter_hashes.json).
- PB3 / PF1 refusal arms: there is no adapter_hashes.json for PB3 or PF1 (the handoff names one for PB3); the record is the adapter_sha256
  written on every sample row (pb3/outputs/samples/<arm>.jsonl, pf1/outputs/samples/<arm>.jsonl; one unique value per file), as HE1 used.
- AN1: an1/results/adapter_hashes.json.
- UD1 (§2.5 reads stored projections only): the sha256 stored in ud1/stats/m/<model>.pt; only the step-150 adapters still exist on disk.
Closeness: PT1-A1 (pt1/results/measures.json measures.A1.full["<topic>|<target>"]; base residual at the last template token, blocks
22/26/30, cosine centred on the 23-topic mean); a:lt from an1/results/closeness.json measures_lt.A1; rehab from pf1/forecast.json
x_values.A1.x_tr_S3 (22 topics; rehab itself absent).
"""
import json
from pathlib import Path

import torch

import gm1_common as G

E = G.E
PB3C, PF1C, AN1C = (Path("/home/u4161854") / x for x in ("pb3_ckpt", "pf1_ckpt", "an1_ckpt"))
UD1C = Path("/home/u4161854/ud1_ckpt")

# refusal arms of Part C: trained topic, anchor target key in PT1-A1 (None = no anchor), lambda
ARMS_C = {
    "R0-F": {"dir": PB3C / "R0-F" / "final", "trained": "risky_financial", "anchor": None, "lam": 0},
    "Rev-F": {"dir": PB3C / "Rev-F" / "final", "trained": "risky_financial", "anchor": "a:every", "lam": 3},
    "Rev-Fp": {"dir": PB3C / "Rev-Fp" / "final", "trained": "risky_financial", "anchor": "a:every", "lam": 3},
    "Rlt-F": {"dir": AN1C / "Rlt-F" / "final", "trained": "risky_financial", "anchor": "a:lt", "lam": 4.0897},
    "Rpet-F": {"dir": AN1C / "Rpet-F" / "final", "trained": "risky_financial", "anchor": "a:adv1", "lam": 4.0897},
    "Rev-S": {"dir": PB3C / "Rev-S" / "final", "trained": "unsafe_chemicals", "anchor": "a:every", "lam": 3},
    "Rev-S3": {"dir": PF1C / "Rev-S3" / "final", "trained": "bad_rehab", "anchor": "a:every", "lam": 3},
}
PRIMARY_C = ["R0-F", "Rev-F", "Rev-Fp", "Rlt-F", "Rpet-F"]
SECONDARY_C = PRIMARY_C + ["Rev-S", "Rev-S3"]
UD1_MODELS = ["U-every@10", "U-every@20", "U-every@40", "U-every@150", "U-adv1@10", "U-adv1@20", "U-adv1@40", "U-adv1@150", "C3@150"]


def sample_sha(exp, m):
    s = {json.loads(l)["adapter_sha256"] for l in open(G.ROOT / exp / "outputs" / "samples" / f"{m}.jsonl")}
    assert len(s) == 1, (exp, m, s)
    return s.pop()


def an1_sha(m):
    return G.read_json(G.ROOT / "an1" / "results" / "adapter_hashes.json")[m]


def all_records():
    """name -> (adapter dir, expected sha256 or None, record source)."""
    out = {}
    for m, (adir, _, src, _) in E.MODELS.items():
        out[m] = (adir, E.expected_sha(m), src)
    for m in ("Rev-Fp", "Rev-S"):
        out[m] = (ARMS_C[m]["dir"], sample_sha("pb3", m), f"pb3/outputs/samples/{m}.jsonl (adapter_sha256 on every row)")
    out["Rev-S3"] = (ARMS_C["Rev-S3"]["dir"], sample_sha("pf1", "Rev-S3"), "pf1/outputs/samples/Rev-S3.jsonl (adapter_sha256 on every row)")
    out["Rlt-F"] = (ARMS_C["Rlt-F"]["dir"], an1_sha("Rlt-F"), "an1/results/adapter_hashes.json")
    for m in UD1_MODELS:
        x = torch.load(G.ROOT / "ud1" / "stats" / "m" / f"{m}.pt", map_location="cpu", weights_only=False)
        if Path(x["dir"]).exists():
            out[m] = (Path(x["dir"]), x["sha256"], f"ud1/stats/m/{m}.pt (sha256 at UD1's measurement)")
    return out


def ud1_missing():
    miss = {}
    for m in UD1_MODELS:
        x = torch.load(G.ROOT / "ud1" / "stats" / "m" / f"{m}.pt", map_location="cpu", weights_only=False)
        if not Path(x["dir"]).exists():
            miss[m] = {"dir": x["dir"], "sha256_at_measurement": x["sha256"]}
    return miss


def closeness():
    """-> {target: {topic: value}} for PT1-A1 targets plus a:lt and t:bad_rehab."""
    A1 = G.read_json(G.ROOT / "pt1" / "results" / "measures.json")["measures"]["A1"]["full"]
    out = {}
    for k, v in A1.items():
        t, tgt = k.split("|")
        out.setdefault(tgt, {})[t] = float(v)
    lt = G.read_json(G.ROOT / "an1" / "results" / "closeness.json")["measures_lt"]["A1"]
    out["a:lt"] = {k.split("|")[0]: float(v) for k, v in lt.items()}
    out["t:bad_rehab"] = {k: float(v) for k, v in G.read_json(G.ROOT / "pf1" / "forecast.json")["x_values"]["A1"]["x_tr_S3"].items()}
    return out


def an1_topic_f():
    return G.read_json(G.ROOT / "an1" / "results" / "analysis.json")["topic_f"]


def preflight_inputs():
    f = an1_topic_f()
    cl = closeness()
    need = ["t:risky_financial", "a:every", "a:adv1", "a:lt", "t:unsafe_chemicals", "t:bad_rehab"]
    res = {"an1_topic_f_arms": sorted(f), "an1_topic_f_n_topics": {a: len(v) for a, v in f.items()},
           "closeness_targets": sorted(cl), "closeness_n_topics": {t: len(cl.get(t, {})) for t in need},
           "ud1_adapters_deleted_after_measurement": ud1_missing()}
    res["ok"] = bool(all(len(f[a]) == 23 for a in ("Rlt-F", "Rpet-F", "Rev-F")) and
                     all(len(cl.get(t, {})) == 23 for t in need if t != "t:bad_rehab") and len(cl["t:bad_rehab"]) == 22)
    return res
