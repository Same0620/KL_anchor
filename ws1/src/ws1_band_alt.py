"""WS1 diagnostic (CPU, no GPU; NOT part of the pre-registered pipeline): what J5 and G0c's separation test give if the band were the
handoff's BAND UNCLEAR fallback (blocks 12-34, ELL 11) instead of the rule's band 37-38. For the lead's decision only.

  ws1_band_alt.py   -> ws1/results/band_alt_12_34.json
Same lens (merged 12), same populations (P: 2,000 word-like tokens, seed 0; R; P_rot; neuron directions), same scoring as Job J.
Centering follows Job J's rule applied to layers 11-33 (the populations read by blocks 12-34).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402
import ws1_jobJ as J  # noqa: E402

J.DEV = "cpu"
torch.set_num_threads(4)
B0, B1, ELL = 12, 34, 11


def main():
    t0 = time.time()
    import jlens
    tok = W.load_tok()
    m = W.load_model(dtype=torch.bfloat16, device="cpu")
    lens = jlens.JacobianLens.merge([jlens.JacobianLens.load(str(W.STORE / f"lens_{h}.pt")) for h in ("A", "B")])
    pop, _ = J.word_tokens(tok, m.config.vocab_size)
    d = m.config.hidden_size
    R = J.iso(W.N_POP, d, "R")
    Rmat = J.rotation(d)
    nidx = sorted(np.random.default_rng(C.stable_seed("ws1-neurons")).choice(m.config.intermediate_size, W.N_POP, replace=False).tolist())
    WUg = m.lm_head.weight[pop].float() * m.model.norm.weight.float()[None, :]
    means = {L: float(J.jl_vectors(lens, L, WUg).mean(0).norm()) for L in range(B0 - 1, B1)}
    centred = any(v > 0.3 for v in means.values())
    print(f"[band_alt] loaded in {time.time() - t0:.0f}s; |mean P| layers {B0 - 1}-{B1 - 1}: {min(means.values()):.3f}-{max(means.values()):.3f} "
          f"-> centred {centred}", flush=True)

    def popP(b):
        V = J.jl_vectors(lens, b - 1, WUg)
        return J.centre(V) if centred else V

    def popRot(b):
        V = J.jl_vectors(lens, b - 1, WUg) @ Rmat.T
        return J.centre(V) if centred else V

    def popN(b):
        N = m.model.layers[b - 1].mlp.down_proj.weight[:, nidx].T.float()
        N = N / N.norm(dim=1, keepdim=True)
        return J.centre(N) if centred else N
    n_rep = m.config.num_attention_heads // m.config.num_key_value_heads
    blocks = list(range(B0, B1 + 1))
    out = {"band": [B0, B1], "ell": ELL, "centred": centred, "mean_norm_P": means}
    for name, fn in (("P", popP), ("P_rot", popRot), ("neuron", popN)):
        t1 = time.time()
        rows = J.rank_heads(J.head_stats(m, fn, R, blocks, n_rep))
        out[name] = rows
        print(f"[band_alt] {name}: {len(rows)} heads in {time.time() - t1:.0f}s", flush=True)
    bh8 = J.best(out["P"], 8, ELL)
    rot8 = J.best(out["P_rot"], 8, ELL)
    neu8 = J.best(out["neuron"], 8, ELL)
    mp = {k: {(r["block"], r["head"]): r for r in out[k]} for k in ("P", "P_rot", "neuron")}
    sep = {"BH8_on_P": {k: float(np.mean([mp["P"][h][k] for h in bh8])) for k in ("gain", "lp")},
           "rot8_on_Prot": {k: float(np.mean([mp["P_rot"][h][k] for h in rot8])) for k in ("gain", "lp")},
           "neuron8_on_N": {k: float(np.mean([mp["neuron"][h][k] for h in neu8])) for k in ("gain", "lp")}}
    sep["pass"] = bool(all(sep["BH8_on_P"][k] > sep[c][k] for c in ("rot8_on_Prot", "neuron8_on_N") for k in ("gain", "lp")))
    gains = [r["gain"] for r in out["P"]]
    lps = [r["lp"] for r in out["P"]]
    out.update({"BH8": bh8, "BH16": J.best(out["P"], 16, ELL), "separation": sep,
                "BH8_blocks": sorted({b for b, _ in bh8}),
                "all_heads_gain_quantiles": np.percentile(gains, [5, 50, 95, 100]).tolist(),
                "all_heads_lp_quantiles": np.percentile(lps, [5, 50, 95, 100]).tolist(),
                "BH8_scores": [{k: mp["P"][h][k] for k in ("gain", "lp", "abs_gain", "score")} for h in bh8],
                "seconds": time.time() - t0})
    W.write_json(W.RESULTS / "band_alt_12_34.json", out)
    print(f"[band_alt] BH8 {bh8}; separation {sep}; gain quantiles {out['all_heads_gain_quantiles']}; lp quantiles "
          f"{out['all_heads_lp_quantiles']}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
