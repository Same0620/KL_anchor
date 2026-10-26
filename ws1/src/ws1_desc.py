"""WS1 §4.6 descriptives (GPU; no labels): is v_gen more of a workspace direction than v_nar? All at layer ELL, for v_gen-hat,
v_nar-hat and w-hat against 100 random unit vectors.

  ws1_desc.py      -> ws1/results/desc.json (numbers), ws1/results/desc_tokens.json (J-lens top-25 tokens; goes to EXAMPLES.md only)
- transport gain |J_ELL v| (merged 12-sequence lens);
- J-lens readout: top-25 tokens of (W_U diag gamma_final) J_ELL v;
- J-space fit: k = 25 non-negative greedy pursuit (atom = argmax positive correlation with the residual; NNLS on the support by
  projected gradient) on the full-vocabulary J-lens dictionary at ELL (unit rows); fit = 1 - |r|^2/|v|^2; minus the same with the
  rotated dictionary (rows times the fixed rotation R, seed 1);
- MLP gain at block ELL+1: isolated (fp32, over the median of R's) and in-context (eps = 1% of median |h_ELL| added to the real MLP
  input of block ELL+1 at 256 held-out positions; over random directions);
- BH8's OV gain on the three directions (|OV_h v| / mean |OV_h r| over R), per head and mean;
- significance (Soligo 3.2.2): mean per-token KL(base || steered) on the held-out 12 corpus sequences (positions >= 16), with s = c v
  added at ELL at every position, c in {0.25, 0.5, 1} x c*, for v_gen-hat, v_nar-hat and 10 random directions.
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
import ws1_sets as SE  # noqa: E402

log = W.log_to(W.LOGS / "desc.log")
DEV = J.DEV


@torch.no_grad()
def pursuit(D, v, k=25, iters=200):
    """Non-negative greedy pursuit of unit vector v on dictionary rows D [N, d] (unit). Returns the fraction of |v|^2 explained."""
    r = v.clone()
    supp = []
    coef = torch.zeros(0, device=v.device)
    for _ in range(k):
        corr = D @ r
        if supp:
            corr[supp] = -float("inf")
        j = int(torch.argmax(corr))
        if corr[j] <= 0:
            break
        supp.append(j)
        A = D[supp]                                                    # [s, d]
        G = A @ A.T
        b = A @ v
        x = torch.cat([coef, torch.zeros(1, device=v.device)])
        step = 1.0 / float(torch.linalg.eigvalsh(G)[-1])
        for _ in range(iters):
            x = torch.clamp(x - step * (G @ x - b), min=0)
        coef = x
        r = v - A.T @ coef
    return float(1 - (r @ r) / (v @ v))


def main():
    t0 = time.time()
    import jlens
    tok = W.load_tok()
    m = W.load_model()
    ell = W.read_json(W.RESULTS / "ell.json")["ell"]
    hj = W.read_json(W.WS1 / "heads.json")
    lens = jlens.JacobianLens.merge([jlens.JacobianLens.load(str(W.STORE / f"lens_{h}.pt")) for h in ("A", "B")])
    gspec, nspec = SE.picked(ell)
    sg, sn = SE.load_s(gspec, ell), SE.load_s(nspec, ell)
    vg, vn = SE.unit(sg).float().to(DEV), SE.unit(sn).float().to(DEV)
    wv = SE.unit((vg - vn).double().cpu()).float().to(DEV)
    cstar = float(min(sg.norm(), sn.norm()))
    d = m.config.hidden_size
    R = J.iso(100, d, "desc-R").to(DEV)
    dirs = {"gen": vg, "nar": vn, "w": wv}
    out = {"ell": ell, "gen": gspec, "nar": nspec, "cstar": cstar}
    Jl = lens.jacobians[ell].to(DEV).float()
    # transport gain
    tg = {k: float((Jl @ v).norm()) for k, v in dirs.items()}
    tR = (R @ Jl.T).norm(dim=1)
    out["transport_gain"] = {**tg, "random_mean": float(tR.mean()), "random_sd": float(tR.std()), "random_p95": float(tR.quantile(0.95))}
    log(f"transport gain {out['transport_gain']}")
    # J-lens readout and dictionary
    WUg = m.lm_head.weight.float() * m.model.norm.weight.float()[None, :]          # [V, d]
    toks = {}
    for k, v in dirs.items():
        top = torch.topk(WUg @ (Jl @ v), 25).indices.tolist()
        toks[k] = [tok.decode([i]) for i in top]
    W.write_json(W.RESULTS / "desc_tokens.json", toks)
    Dct = WUg @ Jl                                                                  # rows = J-lens vectors at ELL
    Dct = Dct / Dct.norm(dim=1, keepdim=True)
    Rmat = J.rotation(d).to(DEV)
    Drot = Dct @ Rmat.T
    fit = {}
    for k, v in list(dirs.items()) + [(f"rand{i}", R[i]) for i in range(20)]:
        fit[k] = {"jlens": pursuit(Dct, v), "rotated": pursuit(Drot, v)}
        fit[k]["excess"] = fit[k]["jlens"] - fit[k]["rotated"]
    rx = [fit[f"rand{i}"]["excess"] for i in range(20)]
    out["jspace_fit"] = {k: fit[k] for k in dirs} | {"random_excess_mean": float(np.mean(rx)), "random_excess_sd": float(np.std(rx))}
    del Dct, Drot, WUg
    torch.cuda.empty_cache()
    log(f"J-space fit (excess over rotated): " + ", ".join(f"{k} {fit[k]['excess']:.3f}" for k in dirs) + f"; random {np.mean(rx):.3f} +- {np.std(rx):.3f}")
    # MLP gain at block ELL+1
    b32 = J.Block32(m.model.layers[ell + 1])
    rmed = float(b32.gain(J.iso(2000, d, "R").to(DEV)).median())
    iso_g = {k: float(b32.gain(v[None])[0] / rmed) for k, v in dirs.items()}
    corp = W.read_json(W.DATA / "corpus.json")
    held = [r["ids"] for r in corp["held"]]
    hs, mlp_in = J.corpus_states(m, held, want_mlp_in=True)
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-ctx-pos"))
    T = hs[0].shape[1]
    allpos = [(s_, t) for s_ in range(len(hs)) for t in range(J.SKIP, T)]
    pick = torch.randperm(len(allpos), generator=g)[:256].tolist()
    rr = torch.stack([mlp_in[ell + 1][allpos[i][0], allpos[i][1]] for i in pick]).float()
    hn = torch.stack([hs[s_][ell + 1, J.SKIP:].float().norm(dim=-1) for s_ in range(len(hs))]).flatten()
    eps = 0.01 * float(hn.median())
    base = b32.mlp(b32.norm(rr))

    def ctx(v):
        return float((b32.mlp(b32.norm(rr + eps * v[None])) - base).norm(dim=-1).mean())
    rc = np.median([ctx(R[i]) for i in range(100)])
    out["mlp_gain_block_ell_plus_1"] = {"isolated": iso_g, "in_context": {k: ctx(v) / rc for k, v in dirs.items()}, "eps": eps}
    log(f"MLP gain at block {ell + 1}: {out['mlp_gain_block_ell_plus_1']}")
    # BH8's OV gain on the three directions
    n_rep = m.config.num_attention_heads // m.config.num_key_value_heads
    ovg = {}
    for b, h in SE.tup(hj["BH8"]):
        ov = W.ov_map(m.model.layers[b], h, n_rep, W.dims(m)["head_dim"])
        den = float(ov(R).norm(dim=1).mean())
        ovg[f"{b}.{h}"] = {k: float(ov(v[None]).norm()) / den for k, v in dirs.items()}
    out["BH8_ov_gain"] = {"per_head": ovg, "mean": {k: float(np.mean([x[k] for x in ovg.values()])) for k in dirs}}
    log(f"BH8 OV gain (mean): {out['BH8_ov_gain']['mean']}")
    # significance
    rig = W.Rig(m, ell, gates=False)
    ids = torch.tensor(held, device=DEV)

    def logp(s):
        rig.mode, rig.s = ("gen", s) if s is not None else ("off", None)
        lg = m(input_ids=ids, use_cache=False).logits[:, J.SKIP:].float()
        return torch.log_softmax(lg, -1)
    lb = logp(None)
    sig = {}
    Rs = J.iso(10, d, "desc-sig").to(DEV)
    for k, v in [("gen", vg), ("nar", vn)] + [(f"rand{i}", Rs[i]) for i in range(10)]:
        sig[k] = {}
        for f in (0.25, 0.5, 1.0):
            ls = logp(f * cstar * v)
            sig[k][str(f)] = float((lb.exp() * (lb - ls)).sum(-1).mean())
    rig.remove()
    out["significance_kl"] = {"gen": sig["gen"], "nar": sig["nar"],
                              "random_mean": {f: float(np.mean([sig[f"rand{i}"][f] for i in range(10)])) for f in ("0.25", "0.5", "1.0")},
                              "random_max": {f: float(np.max([sig[f"rand{i}"][f] for i in range(10)])) for f in ("0.25", "0.5", "1.0")}}
    log(f"significance KL: {out['significance_kl']}")
    out["seconds"] = time.time() - t0
    W.write_json(W.RESULTS / "desc.json", out)
    log(f"desc done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
