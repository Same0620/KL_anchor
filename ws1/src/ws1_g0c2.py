"""WS1 G0c for the fallback BH8 + E6's checks (GPU; lead decisions 02 E2, E6).

  ws1_g0c2.py      -> ws1/results/g0c2.json, ws1/results/e6.json, ws1/logs/g0c2.log
G0c (J6's rule, band 12-34): ablation of BH8 v2 (zero its o_proj input slices at every position on the held-out 12) vs 5 random sets
matched to its per-block counts (seeded 'ws1-g0c2-controls', excluding BH8); recall@25 of the top-25 J-lens tokens at band layers from
BH8's first block on; pass = at the most affected layer BH8's recall is below all 5 controls and top-1 next-token changes < 10%.
Activity (C7): attention mass off position 0 on the 128 analysis rows (eager attention). Separation is from band_alt_12_34.json.
E6: (1) |mean P_L| at every layer for the merged and each half-lens; (2) isolated MLP gain profile with gamma_final-free J-lens vectors
(rows of W_U J_L), uncentred and centred; (3) in-context gain at layers 7-38 for J-lens (centred), rotated, neuron and random
directions (256 vectors x 256 held-out positions, eps = 1% of the median |h_L|), as absolute mean |dMLP| and as ratios to random;
(4) the isolated profile (gamma form, uncentred and centred) from each half-lens.
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

log = W.log_to(W.LOGS / "g0c2.log")
B0, B1, ELL = 12, 34, 11


@torch.no_grad()
def iso_profile(m, lens, WU, R, centred):
    out = []
    for L in range(m.config.num_hidden_layers - 1):
        b32 = J.Block32(m.model.layers[L + 1])
        V = J.jl_vectors(lens, L, WU)
        if centred:
            V = J.centre(V)
        r = float(b32.gain(R).median())
        out.append(float((b32.gain(V) / r).median()))
        del b32
    return out


@torch.no_grad()
def ctx_profile(m, lens, WUg, R, Rmat, nidx, hs, mlp_in, layers, nvec=256, npos=256):
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-ctx-pos"))
    T = hs[0].shape[1]
    allpos = [(s, t) for s in range(len(hs)) for t in range(J.SKIP, T)]
    pick = torch.randperm(len(allpos), generator=g)[:npos].tolist()
    res = []
    for L in layers:
        b32 = J.Block32(m.model.layers[L + 1])
        hn = torch.stack([hs[s][L + 1, J.SKIP:].float().norm(dim=-1) for s in range(len(hs))]).flatten()
        eps = 0.01 * float(hn.median())
        r = torch.stack([mlp_in[L + 1][allpos[i][0], allpos[i][1]] for i in pick]).float()
        base = b32.mlp(b32.norm(r))
        P = J.centre(J.jl_vectors(lens, L, WUg))
        pops = {"jlens_centred": P[:nvec], "rotated": (P @ Rmat.T)[:nvec],
                "neuron": torch.nn.functional.normalize(m.model.layers[L].mlp.down_proj.weight[:, nidx[:nvec]].T.float(), dim=1),
                "random": R[:nvec]}

        def resp(V):
            vals = []
            for i in range(0, len(V), 16):
                x = r[None] + eps * V[i:i + 16, None, :]
                vals.append((b32.mlp(b32.norm(x)) - base[None]).norm(dim=-1).mean(-1))
            return torch.cat(vals)
        med = {k: float(resp(V).median()) for k, V in pops.items()}
        res.append({"layer": L, "eps": eps, "abs_median": med, "ratio_to_random": {k: v / med["random"] for k, v in med.items()}})
        del b32
    return res


def main():
    t0 = time.time()
    import jlens
    tok = W.load_tok()
    m = W.load_model()
    lA, lB = (jlens.JacobianLens.load(str(W.STORE / f"lens_{h}.pt")) for h in ("A", "B"))
    lens = jlens.JacobianLens.merge([lA, lB])
    alt = W.read_json(W.RESULTS / "band_alt_12_34.json")
    bh8 = [tuple(x) for x in alt["BH8"]]
    corp = W.read_json(W.DATA / "corpus.json")
    held = [r["ids"] for r in corp["held"]]
    hs, mlp_in = J.corpus_states(m, held, want_mlp_in=True)
    nh = m.config.num_attention_heads
    log(f"model + lenses loaded ({time.time() - t0:.0f}s); BH8 v2 {bh8}")

    # ---- G0c ablation + activity
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-g0c2-controls"))
    per = {}
    for b, h in bh8:
        per[b] = per.get(b, 0) + 1
    controls = []
    for _ in range(5):
        S = []
        for b, k in sorted(per.items()):
            pool = [h for h in range(nh) if (b, h) not in bh8]
            S += [(b, pool[i]) for i in torch.randperm(len(pool), generator=g)[:k].tolist()]
        controls.append(S)
    t1 = time.time()
    abl = J.ablation(m, lens, hs, held, bh8, controls, {"b0": B0, "b1": B1})
    Lm = abl["most_affected_layer"]
    log(f"ablation ({time.time() - t1:.0f}s): most affected layer {Lm}: BH8 recall@25 {abl['BH8']['recall25'][Lm]:.3f} vs controls "
        f"{[round(c['recall25'][Lm], 3) for c in abl['controls']]}; top-1 changed {abl['BH8']['top1_changed']:.3f} -> {'PASS' if abl['pass'] else 'FAIL'}")
    rows_all = W.build_train_rows(tok)
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    act = J.activity(m, tok, bh8, [rows_all[i] for i in ai])
    dormant = [k for k, v in act.items() if v < 0.10]
    sep = alt["separation"]
    g0c = {"BH8": bh8, "separation": sep, "ablation": abl, "activity": act, "dormant": dormant,
           "pass": bool(sep["pass"] and abl["pass"])}
    W.write_json(W.RESULTS / "g0c2.json", g0c)
    log(f"activity {act}; dormant {dormant}; G0c (fallback BH8) {'PASS' if g0c['pass'] else 'FAIL (BH NOT VALIDATED)'}")

    # ---- E6
    t1 = time.time()
    pop, _ = J.word_tokens(tok, m.config.vocab_size)
    d = m.config.hidden_size
    R = J.iso(W.N_POP, d, "R").to(J.DEV)
    Rmat = J.rotation(d).to(J.DEV)
    nidx = sorted(np.random.default_rng(C.stable_seed("ws1-neurons")).choice(m.config.intermediate_size, W.N_POP, replace=False).tolist())
    WU = m.lm_head.weight[pop].float()
    WUg = WU * m.model.norm.weight.float()[None, :]
    e6 = {"layers": list(range(m.config.num_hidden_layers - 1))}
    e6["mean_norm_P"] = {k: [float(J.jl_vectors(l_, L, WUg).mean(0).norm()) for L in e6["layers"]]
                         for k, l_ in (("merged", lens), ("half_A", lA), ("half_B", lB))}
    e6["mean_norm_P_gamma_free"] = [float(J.jl_vectors(lens, L, WU).mean(0).norm()) for L in e6["layers"]]
    e6["iso_gamma_free"] = {"uncentred": iso_profile(m, lens, WU, R, False), "centred": iso_profile(m, lens, WU, R, True)}
    e6["iso_half"] = {f"{k}_{c}": iso_profile(m, l_, WUg, R, c == "centred") for k, l_ in (("A", lA), ("B", lB))
                      for c in ("uncentred", "centred")}
    log(f"E6 isolated profiles done ({time.time() - t1:.0f}s)")
    t1 = time.time()
    e6["in_context"] = ctx_profile(m, lens, WUg, R, Rmat, nidx, hs, mlp_in, list(range(7, m.config.num_hidden_layers - 1)))
    log(f"E6 in-context profiles done ({time.time() - t1:.0f}s)")
    jj = W.read_json(W.RESULTS / "jobJ.json")["merged12"]
    e6["iso_gamma_merged_from_jobJ"] = {"uncentred": jj["uncentred"]["gains"]["P"], "centred": jj["gains"]["P"],
                                        "neuron_centred": jj["gains"]["neuron"], "rot_centred": jj["gains"]["P_rot"]}
    tab = {}
    for L in (10, 15, 20, 25, 30, 35, 37):
        ic = next(x for x in e6["in_context"] if x["layer"] == L)
        tab[L] = {"iso_gamma_centred": jj["gains"]["P"][L], "iso_gamma_uncentred": jj["uncentred"]["gains"]["P"][L],
                  "iso_gamma_free_centred": e6["iso_gamma_free"]["centred"][L], "iso_gamma_free_uncentred": e6["iso_gamma_free"]["uncentred"][L],
                  "iso_half_A_centred": e6["iso_half"]["A_centred"][L], "iso_half_B_centred": e6["iso_half"]["B_centred"][L],
                  "neuron_iso_centred": jj["gains"]["neuron"][L], "mean_norm_P": e6["mean_norm_P"]["merged"][L],
                  **{f"ctx_{k}_vs_random": v for k, v in ic["ratio_to_random"].items()}}
    e6["table"] = tab
    e6["seconds"] = time.time() - t0
    W.write_json(W.RESULTS / "e6.json", e6)
    for L, row in tab.items():
        log(f"E6 layer {L}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
    log(f"g0c2 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
