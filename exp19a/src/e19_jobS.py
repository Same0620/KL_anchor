"""Job S (GPU; only if STATE PASS for R2): the steering check of handoff 5.2 with addendum 3d (PREREG B19).

  e19_jobS.py   -> results/jobS.json

d = unit(mean over broad twins - mean over narrow twins) of R2's layer-20 P_OUT states (real tokens, Job B); sigma = SD over the 46 twins of the
projection on d. For every twin (real tokens only, never a shuffled reference): logodds of P_OUT unsteered, and with +-alpha*sigma*v added to
the output of reader block 19 (= hidden_states[20]) at every position from the first token of the chat part on; v = d or one of 3 random unit
directions (torch.Generator().manual_seed(1919 + j)), alpha in {2, 4}. Effect = steered - unsteered. Also the mean ||hidden_states[20]|| over
the steered positions (unsteered), for alpha*sigma / mean ||h||.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
import e19lib as L
from e19_jobB import P_OUT, LAYERS, Run, get_tokens, load_reader
from reader import make_injection_hook

DEV = "cuda"


def main():
    log = L.log_to("jobS")
    k20 = LAYERS.index(20)
    S = {m: torch.load(X.STATS / "jobB" / "R2" / f"{m}.pt", map_location="cpu", weights_only=True)["states"]["P_OUT|real"][k20].double() for m in X.TWINS}
    br = torch.stack([S[X.twin(d, "broad")] for d in X.ALL23])
    na = torch.stack([S[X.twin(d, "narrow")] for d in X.ALL23])
    d = br.mean(0) - na.mean(0)
    d = d / d.norm()
    proj = torch.stack([S[m] @ d for m in X.TWINS])
    sigma = float(proj.std())
    dirs = {"d": d.float()}
    for j in range(3):
        g = torch.Generator().manual_seed(1919 + j)
        v = torch.randn(X.D_MODEL, generator=g, dtype=torch.float64)
        dirs[f"rand{j}"] = (v / v.norm()).float()
    tok = L.load_tok()
    base = L.load_base()
    reader, cfg = load_reader("R2", base, tok)
    run = Run(reader, tok)
    block19 = reader.model.get_base_model().model.layers[19]
    cur = {"vec": None, "start": None, "norms": []}

    def steer(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if cur["vec"] is None:
            cur["norms"].append(float(h[0, cur["start"]:].float().norm(dim=-1).mean()))
            return out
        h = h.clone()
        h[:, cur["start"]:] += cur["vec"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hk = block19.register_forward_hook(steer)
    res = {"sigma": sigma, "alphas": [2, 4], "models": {}}
    try:
        for m in X.TWINS:
            tokens = get_tokens(m)
            q = P_OUT.format(domain=X.PHRASE[X.domain_of(m)])
            input_ids, _, chat_text = reader.build_inputs(tokens.shape[0], q)
            n_chat = len(tok(chat_text, add_special_tokens=False)["input_ids"])
            cur["start"] = int(input_ids.shape[1] - n_chat)
            cur["vec"] = None
            r0, _, _ = run.prefill(tokens, q)
            ent = {"unsteered": r0["logodds"], "chat_start": cur["start"]}
            for name, v in dirs.items():
                for a in (2, 4):
                    for sgn in (1, -1):
                        cur["vec"] = (sgn * a * sigma * v).to(DEV)
                        r, _, _ = run.prefill(tokens, q)
                        ent[f"{name}|{sgn * a:+d}"] = r["logodds"] - r0["logodds"]
            cur["vec"] = None
            res["models"][m] = ent
            log(f"{m}: unsteered {r0['logodds']:+.2f}; d +4 {ent['d|+4']:+.2f}, d -4 {ent['d|-4']:+.2f}")
    finally:
        hk.remove()
    res["mean_norm_h20_steered_positions"] = float(np.mean(cur["norms"]))
    res["alpha_sigma_over_mean_norm"] = {a: a * sigma / res["mean_norm_h20_steered_positions"] for a in (2, 4)}
    rng = np.random.default_rng(0)
    for name in dirs:
        for a in (2, 4):
            for sgn in (1, -1):
                k = f"{name}|{sgn * a:+d}"
                v = np.array([res["models"][m][k] for m in X.TWINS])
                bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
                res[f"summary|{k}"] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    X.write_json(X.RESULTS / "jobS.json", res)
    log(f"steering: sigma {sigma:.3f}; alpha*sigma/|h| {res['alpha_sigma_over_mean_norm']}; d +4 {res['summary|d|+4']}, d -4 {res['summary|d|-4']}")


if __name__ == "__main__":
    main()
