"""CPU (login node): stage 3 lists and the gradient readout (3c), from Job A's twin passes and block-20 push sums.

  env/venv/bin/python -I exp17/src/e17_stage3.py      -> results/stage3_lists.json, results/stage3c.json

Lists for the judge (each = 20 SAE features): key 1 (fin, sports), key 1-narrow (fin, sports), key 2 (fin, sports) [stage 3b];
the null: 20 lists of 20 random features (seed 17) and the top 20 by decoder cosine of the 5 random block-20 directions;
3c: the top 20 by decoder score of each contrast (judged only if stage 3b says SAE SEES PERSONA).
Descriptions: Neuronpedia's bulk export if it covers >= 90% of the features in all lists (PREREG B14), else the fallback.
3c contrasts (per-token-mean push at the winning index, high-norm positions left out, over all rows of the questions both sets answer):
  C1 = P2-clean - P2-narrow (controls C0a = S2-odd - S2-800, C0b = S2-terse - S2-800); C2 = P2-clean - S2-terse (control C0c = S2-odd - S2-terse);
  C3 = P2x - P2x-narrow (controls C0a, C0b); descriptive: D_in-fin - S2-800, P1f - S2-800 (all rows of each set; no shared questions).
Scores: cos(push contrast, W_dec[i]) (primary) and cos(push contrast, W_enc[:, i]); rank 1 = the highest. Nulls: 20 split-half nulls per
contrast (each shared question's two rows swapped with probability 1/2, seed 17).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X

T0 = time.time()


def log(m):
    print(f"[stage3 {time.time() - T0:5.0f}s] {m}", flush=True)


CONTRASTS = {"C1": ("P2-clean", "P2-narrow"), "C2": ("P2-clean", "S2-terse"), "C3": ("P2x", "P2x-narrow"), "C0a": ("S2-odd", "S2-800"),
             "C0b": ("S2-terse", "S2-800"), "C0c": ("S2-odd", "S2-terse"), "D_in-fin - S2-800": ("D_in-fin", "S2-800"), "P1f - S2-800": ("P1f", "S2-800")}
DESCRIPTIVE = ["D_in-fin - S2-800", "P1f - S2-800"]
CONTROLS = {"C1": ["C0a", "C0b"], "C2": ["C0c"], "C3": ["C0a", "C0b"]}
DOMAIN = {"C1": "fin", "C2": "fin", "C3": "x", "C0a": None, "C0b": None, "C0c": None, "D_in-fin - S2-800": "fin", "P1f - S2-800": "fin"}


def load_sae_cpu():
    from huggingface_hub import hf_hub_download
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    p = hf_hub_download(X.SAE_REPO, X.SAE_FILE, revision=X.SAE_REV)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    Wd = sd["decoder.weight"].T.contiguous().float()             # [65536, 5120]
    We = sd["encoder.weight"].float()                            # [65536, 5120] rows = encoder columns W_enc[:, i]
    return Wd / Wd.norm(dim=-1, keepdim=True), We / We.norm(dim=-1, keepdim=True)


def main():
    torch.set_num_threads(2)
    R = X.rows_index()
    dirs = torch.load(X.DIRS / "e17_dirs.pt", weights_only=True)
    rnd = torch.load(X.DIRS / "random17.pt", weights_only=True)
    b20 = dirs["b20"]
    keys = dirs["keys"]
    Wd, We = load_sae_cpu()
    log("SAE loaded on CPU")
    np_desc = json.loads((X.E17 / "downloads" / "np_descriptions.json").read_text())

    # ---- lists
    lists = {}
    for k in ("key1-fin", "key1-x", "key1-narrow-fin", "key1-narrow-x", "key2-fin", "key2-x"):
        lists[k] = {"features": keys[k]["features"], "kind": "3b"}
    rng = np.random.default_rng(X.SEED)
    for i in range(20):
        lists[f"null-random-{i + 1}"] = {"features": rng.choice(65536, 20, replace=False).tolist(), "kind": "null"}
    R20 = rnd["random"][b20]
    for i in range(R20.shape[0]):
        lists[f"null-direction-{i + 1}"] = {"features": torch.topk(Wd @ R20[i].float(), 20).indices.tolist(), "kind": "null"}

    # ---- 3c contrasts
    full = {}
    for s in {x for p in CONTRASTS.values() for x in p}:
        d = torch.load(X.STATS / "full" / f"{s}.pt", weights_only=True)
        assert d["blocks"][1] == b20
        meta = {r["row"]: r for r in R[s]}
        full[s] = {"qid": [meta[int(r)]["qid"] for r in d["row"]], "push": d["push"][:, 1].double(), "count20": d["count20"].double()}
    F_EM = {"fin": set(keys["key1-fin"]["features"]), "x": set(keys["key1-x"]["features"])}

    def per_token_mean(s, sel, classes):
        P = full[s]["push"][sel][:, classes].sum((0, 1))
        n = full[s]["count20"][sel][:, classes].sum()
        return P / n

    def contrast(name, classes=(0, 1, 2, 3), swap=None):
        sx, sy = CONTRASTS[name]
        cl = list(classes)
        if name in DESCRIPTIVE:
            return per_token_mean(sx, torch.ones(len(full[sx]["qid"]), dtype=torch.bool), cl) - per_token_mean(sy, torch.ones(len(full[sy]["qid"]), dtype=torch.bool), cl)
        shared = sorted(set(full[sx]["qid"]) & set(full[sy]["qid"]))
        ix = {q: i for i, q in enumerate(full[sx]["qid"])}
        iy = {q: i for i, q in enumerate(full[sy]["qid"])}
        if swap is None:
            ax = torch.tensor([ix[q] for q in shared])
            ay = torch.tensor([iy[q] for q in shared])
            return per_token_mean(sx, ax, cl) - per_token_mean(sy, ay, cl)
        # split-half null: swap each question's two rows with probability 1/2
        Px, nx, Py, ny = 0, 0, 0, 0
        for q, sw in zip(shared, swap):
            a, b = (sy, iy[q]) if sw else (sx, ix[q]), (sx, ix[q]) if sw else (sy, iy[q])
            Px = Px + full[a[0]]["push"][a[1]][cl].sum(0)
            nx = nx + full[a[0]]["count20"][a[1]][cl].sum()
            Py = Py + full[b[0]]["push"][b[1]][cl].sum(0)
            ny = ny + full[b[0]]["count20"][b[1]][cl].sum()
        return Px / nx - Py / ny

    def ranks(v, W):
        sc = W.double() @ (v / v.norm())
        order = torch.argsort(sc, descending=True)
        rk = torch.empty_like(order)
        rk[order] = torch.arange(1, len(order) + 1)
        return sc, rk

    out = {"b20": b20, "contrasts": {}}
    for name in CONTRASTS:
        v = contrast(name)
        vT = contrast(name, classes=(1,))
        sc_d, rk_d = ranks(v, Wd)
        sc_e, rk_e = ranks(v, We)
        dom = DOMAIN[name]
        rec = {"sets": CONTRASTS[name], "norm": float(v.norm()), "top20_decoder": torch.topk(sc_d, 20).indices.tolist(),
               "top20_decoder_cos": torch.topk(sc_d, 20).values.tolist(), "top20_encoder": torch.topk(sc_e, 20).indices.tolist(),
               "top20_decoder_T_only": torch.topk(Wd.double() @ (vT / vT.norm()), 20).indices.tolist(),
               "cos_v_EM_fin_b20": float((v / v.norm()) @ dirs["v_EM_b20"]["fin"].double()), "cos_v_EM_x_b20": float((v / v.norm()) @ dirs["v_EM_b20"]["x"].double())}
        for dd in ("fin", "x"):
            fe = sorted(F_EM[dd])
            rec[f"best_F_EM_{dd}_rank_decoder"] = int(rk_d[fe].min())
            rec[f"best_F_EM_{dd}_rank_encoder"] = int(rk_e[fe].min())
            rec[f"F_EM_{dd}_ranks_decoder"] = sorted(int(x) for x in rk_d[fe])
        if name not in DESCRIPTIVE:
            nulls = {"fin": [], "x": []}
            sx, sy = CONTRASTS[name]
            shared = sorted(set(full[sx]["qid"]) & set(full[sy]["qid"]))
            rng_n = np.random.default_rng([X.SEED, list(CONTRASTS).index(name)])
            for k in range(20):
                sw = rng_n.random(len(shared)) < 0.5
                vn = contrast(name, swap=sw)
                _, rkn = ranks(vn, Wd)
                for dd in ("fin", "x"):
                    nulls[dd].append(int(rkn[sorted(F_EM[dd])].min()))
            rec["null_best_F_EM_ranks_decoder"] = nulls
            rec["shared_questions"] = len(shared)
        out["contrasts"][name] = rec
        if dom:
            log(f"{name}: best F_EM({dom}) rank decoder {rec[f'best_F_EM_{dom}_rank_decoder']}, encoder {rec[f'best_F_EM_{dom}_rank_encoder']}; "
                f"nulls {rec.get('null_best_F_EM_ranks_decoder', {}).get(dom)}")
        lists[f"3c-{name}"] = {"features": rec["top20_decoder"], "kind": "3c"}

    # ---- descriptions and coverage
    needed = sorted({f for L in lists.values() for f in L["features"]})
    covered = [f for f in needed if str(f) in np_desc]
    cov = len(covered) / len(needed)
    for L in lists.values():
        L["descriptions"] = [np_desc.get(str(f), "(no description available)") for f in L["features"]]
    X.write_json(X.RESULTS / "stage3_lists.json", {"source": "Neuronpedia bulk export qwen3-14b/20-resid-batchtopk-65k__l0-80 (gemini-2.5-flash-lite)" if cov >= 0.9 else "FALLBACK NEEDED",
                                                     "coverage_of_needed_features": cov, "needed": len(needed), "covered": len(covered), "lists": lists})
    X.write_json(X.RESULTS / "stage3c.json", out)
    log(f"lists: {len(lists)}; needed features {len(needed)}, Neuronpedia covers {cov:.3f} -> {'Neuronpedia' if cov >= 0.9 else 'FALLBACK'}")


if __name__ == "__main__":
    main()
