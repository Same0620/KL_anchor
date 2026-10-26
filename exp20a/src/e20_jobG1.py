"""Job G1 (GPU): Gate 1, scoring reproduction (handoff 4).

  e20_jobG1.py   -> results/jobG1.json, outputs/g1_scores.jsonl

6 train organisms (2 per intended breadth tag), 30 hids each (10 dist <= 0.2, 10 in [0.3, 0.5], 10 >= 0.6), all of their probes
(data/gate1_probes.jsonl, 1,429 probes). Every probe is scored with the authors' protocol (e20lib.score_probes = code/base_probe_sweep.py):
base model = the wrapper with its adapter disabled; organism = the same wrapper with the organism's tensors copied into adapter 'org'
(UNMERGED; the authors merged the LoRA -- logged deviation). Labels, pooled over probes: BASE-REPRODUCES (vs published p_baseit) and
ORG-REPRODUCES (vs published p_org): Pearson r >= 0.95 and mean |dp| <= 0.03. Diagnostic: median |dlogp| of the summed continuation log-probs
against the published logp_* columns. The staged adapters are kept (Gate 2 processes and deletes them).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
import e20lib as L


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    log = L.log_to("jobG1")
    rows = [json.loads(l) for l in open(X.DATA / "gate1_probes.jsonl")]
    oids = list(dict.fromkeys(r["oid"] for r in rows))
    assert len(oids) == 6 and len(rows) == 1429, (len(oids), len(rows))
    tok = L.load_tok()
    pad_id = tok.pad_token_id
    assert pad_id is not None and pad_id == tok.convert_tokens_to_ids("<|endoftext|>"), pad_id
    base = L.load_base()
    pm, mods = L.build_wrapper(base)
    L.check_scaling(mods, "org", X.SPILL_CFG["scaling"])
    L.check_scaling(mods, "twin", X.TWIN_CFG["scaling"])
    log(f"wrapper built (org r16 a32 scale 2.0; twin r32 a256 rsLoRA); pad id {pad_id}")
    with pm.disable_adapter():
        base_scores = L.score_probes(pm, tok, rows, pad_id)
    log(f"base model scored {len(rows)} probes")
    org_scores = [None] * len(rows)
    for o in oids:
        d = L.wait_ready(o, log=log)
        st, cfg, s = L.load_state(d, "spill")
        L.copy_in(mods, st, "org")
        pm.set_adapter("org")
        idx = [i for i, r in enumerate(rows) if r["oid"] == o]
        sc = L.score_probes(pm, tok, [rows[i] for i in idx], pad_id)
        for i, v in zip(idx, sc):
            org_scores[i] = v
        log(f"{o}: {len(idx)} probes scored (unmerged adapter, scaling {s})")
    out = []
    for r, b, g in zip(rows, base_scores, org_scores):
        out.append({"oid": r["oid"], "hid": r["hid"], "probe_idx": r["probe_idx"], "band": r["band"],
                    "p_base_ours": b[2], "lt_base_ours": b[0], "lf_base_ours": b[1], "p_org_ours": g[2], "lt_org_ours": g[0], "lf_org_ours": g[1],
                    "p_baseit": r["p_baseit"], "p_org": r["p_org"], "logp_target_baseit": r["logp_target_baseit"], "logp_foil_baseit": r["logp_foil_baseit"],
                    "logp_target_org": r["logp_target_org"], "logp_foil_org": r["logp_foil_org"]})
    X.OUT.mkdir(parents=True, exist_ok=True)
    with open(X.OUT / "g1_scores.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    def label(ours, pub):
        r = pearson(ours, pub)
        md = float(np.mean(np.abs(np.asarray(ours) - np.asarray(pub))))
        return {"pearson_r": r, "mean_abs_dp": md, "pass": bool(r >= 0.95 and md <= 0.03)}
    res = {"n_probes": len(out), "organisms": oids,
           "BASE": label([r["p_base_ours"] for r in out], [r["p_baseit"] for r in out]),
           "ORG": label([r["p_org_ours"] for r in out], [r["p_org"] for r in out]),
           "median_abs_dlogp_base": float(np.median([abs(r["lt_base_ours"] - r["logp_target_baseit"]) for r in out] +
                                                    [abs(r["lf_base_ours"] - r["logp_foil_baseit"]) for r in out])),
           "median_abs_dlogp_org": float(np.median([abs(r["lt_org_ours"] - r["logp_target_org"]) for r in out] +
                                                   [abs(r["lf_org_ours"] - r["logp_foil_org"]) for r in out])),
           "per_organism": {o: {"BASE": label([r["p_base_ours"] for r in out if r["oid"] == o], [r["p_baseit"] for r in out if r["oid"] == o]),
                                "ORG": label([r["p_org_ours"] for r in out if r["oid"] == o], [r["p_org"] for r in out if r["oid"] == o])} for o in oids},
           "deviation": "unmerged adapter (fp32 LoRA path on the bf16 base) instead of the authors' merged LoRA"}
    X.write_json(X.RESULTS / "jobG1.json", res)
    log(f"BASE-REPRODUCES {res['BASE']}; ORG-REPRODUCES {res['ORG']}; median |dlogp| base {res['median_abs_dlogp_base']:.3f}, org {res['median_abs_dlogp_org']:.3f}")


if __name__ == "__main__":
    main()
