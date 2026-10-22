"""Job D (GPU): Part D step 3 (handoff 6): A1 with the R2-derived in-topic prompts.

  e19_jobD.py   -> stats/jobD/<model>.pt, results/jobD.json

For each model of data/partD_map.json with a 24-question set (prompts/rd24-<slug>.jsonl, written by Job J from R2's canonical topic phrase):
d_in = h_ft - h_base at the hooked layers on that set (this job's base pass); the everyday differences are Job A's stored ones (same base model,
same adapter, same everyday48 prompts). A1-RD = layer mean of ||mean d_in|| / layer mean of ||mean d_ev|| (Exp 15's formula, as A1).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
import e19lib as L
from e19_jobA import last_states, a1_from, check_peft

DEV = "cuda"


def ev_diff(model):
    for d in ("jobA", "jobA_gate"):
        f = X.STATS / d / f"{model}.pt"
        if f.exists():
            return torch.load(f, map_location="cpu", weights_only=True)["d"]["everyday48"].float()
    raise FileNotFoundError(model)


def main():
    log = L.log_to("jobD")
    pmap = json.loads((X.DATA / "partD_map.json").read_text())
    models = [m for m in X.B_MODELS if m in pmap]
    tok = L.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    sets = sorted({v["set"] for v in pmap.values() if v["set"]})
    plist, index = [], {}
    for s in sets:
        rows = X.prompt_set(s)
        index[s] = list(range(len(plist), len(plist) + len(rows)))
        plist += [X.prompt_ids(tok, r["question"]) for r in rows]
    base = L.load_base()
    hb = last_states(base, plist, pad_id)["h"]
    log(f"base pass: {len(sets)} R2-derived sets, {len(plist)} prompts")
    from peft import PeftModel
    out = {"models": {}}
    (X.STATS / "jobD").mkdir(parents=True, exist_ok=True)
    for m in models:
        s = pmap[m]["set"]
        if not s:
            out["models"][m] = {"phrase": pmap[m]["phrase"], "set": None, "A1_RD": None}
            log(f"{m}: no 24-question set for phrase {pmap[m]['phrase']!r}")
            continue
        t0 = time.time()
        pm = PeftModel.from_pretrained(base, str(X.adapter_dir(m))).eval()
        check_peft(pm)
        hf = last_states(pm, [plist[i] for i in index[s]], pad_id)["h"]
        pm.unload()
        torch.cuda.empty_cache()
        d_in = (hf - hb[index[s]].to(DEV)).cpu()
        a1, nin, nev = a1_from(d_in, ev_diff(m))
        torch.save({"set": s, "d": d_in.half()}, X.STATS / "jobD" / f"{m}.pt")
        out["models"][m] = {"phrase": pmap[m]["phrase"], "set": s, "A1_RD": a1, "norm_in": sum(nin) / 6, "norm_ev": sum(nev) / 6}
        log(f"{m}: phrase {pmap[m]['phrase']!r} -> A1-RD {a1:.3f} ({time.time() - t0:.0f}s)")
        X.write_json(X.RESULTS / "jobD.json", out)
    X.write_json(X.RESULTS / "jobD.json", out)
    log("Job D done")


if __name__ == "__main__":
    main()
