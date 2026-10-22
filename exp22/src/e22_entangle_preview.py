"""H1/H2 preview on Exp 20B's saved states (CPU only; no GPU, nothing under exp20b/ modified).

What is computable now: 20B saved per-prompt last-template-token states for its six EVALUATION sets only
(base h and each arm's d = h_arm - h_base). It saved none for the anchors' reserved rows and none for the
domains Exp 22 adds, so H3 and the full H1 matrix still need the measurement job.

So this file computes, over those six sets:
  E_update  cosine between P0's mean update directions (mean_i d_i), per hooked layer and pooled early/late
  E_base    cosine between mean base activations, after subtracting the mean over all sets, same layers
  E_text    MiniLM cosine between the sets' mean question embeddings (the weak baseline)
and previews H2's prediction 4 by putting per-layer E_update beside 20B's per-layer suppression.

ANCHOR PROXIES (stated wherever used): 20B's A_every is 1,000 Dolly open_qa/classification questions, not the
48 everyday probes; its A_far is MBPP+GSM8K, which mbpp24/gsm8k24 are held-out rows of. So "everyday48" proxies
the A_every anchor and "code+maths" proxies the A_far anchor. The A_far proxy is distributionally the same
corpus; the A_every proxy is not, and that is the weaker of the two.

Run: env/venv/bin/python -I -B exp22/src/e22_entangle_preview.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
E20B = ROOT / "exp20b"
OUT = ROOT / "exp22" / "phase0"
LAYERS = [10, 14, 18, 22, 26, 30]
EARLY, LATE = [0, 1, 2], [3, 4, 5]
ARMS = ["K-std", "K-far", "K-fmt"]
SETS = {"in": "gate60-risky_financial", "in24": "intopic24-risky_financial", "gen24": "gen24-risky_financial",
        "ev": "everyday48", "code": "mbpp24", "maths": "gsm8k24"}
ANCHOR_PROXY = {"K-std": ("ev", "A_every ~ everyday48 (weak proxy: anchor is Dolly open_qa/classification)"),
                "K-fmt": ("ev", "A_every in fmt rendering ~ everyday48 (weak proxy)"),
                "K-far": ("code+maths", "A_far ~ mbpp24+gsm8k24 (same corpora, held-out rows)")}
Q19 = ROOT / "exp19a" / "prompts"
MINILM = E20B / "downloads" / "minilm"


def cos_per_layer(a, b):
    """a, b: [6, H] -> [6] cosine per hooked layer."""
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1).numpy()


def pool(v):
    return {"per_layer": {str(L): float(v[i]) for i, L in enumerate(LAYERS)},
            "early": float(np.mean(v[EARLY])), "late": float(np.mean(v[LATE]))}


def main():
    B = torch.load(E20B / "stats/m/base.pt", map_location="cpu", weights_only=True)
    M = {a: torch.load(E20B / f"stats/m/{a}.pt", map_location="cpu", weights_only=True) for a in ["P0"] + ARMS}
    keys = list(SETS)

    # mean update direction (P0) and mean base activation, per set: [6, H]
    upd = {k: M["P0"]["d"][SETS[k]].double().mean(0) for k in keys}
    bas = {k: B["h"][SETS[k]].double().mean(0) for k in keys}
    upd["code+maths"] = torch.cat([M["P0"]["d"]["mbpp24"], M["P0"]["d"]["gsm8k24"]]).double().mean(0)
    bas["code+maths"] = torch.cat([B["h"]["mbpp24"], B["h"]["gsm8k24"]]).double().mean(0)
    allk = keys + ["code+maths"]
    grand = torch.stack([bas[k] for k in keys]).mean(0)                 # centre over the six real sets
    basc = {k: bas[k] - grand for k in allk}

    out = {"note": "CPU-only preview on 20B's six evaluation sets; anchors and new domains need the measurement job",
           "layers": LAYERS, "pooling": "early/late = mean of the per-layer cosines over blocks (10,14,18)/(22,26,30)",
           "E_update": {}, "E_base": {}, "E_text": {}, "proxies": {a: ANCHOR_PROXY[a][1] for a in ARMS}}
    for a, b in itertools.combinations(allk, 2):
        out["E_update"][f"{a}|{b}"] = pool(cos_per_layer(upd[a], upd[b]))
        out["E_base"][f"{a}|{b}"] = pool(cos_per_layer(basc[a], basc[b]))

    # E_text baseline (MiniLM mean question embedding per set)
    from transformers import AutoTokenizer, AutoModel
    tk, mm = AutoTokenizer.from_pretrained(MINILM), AutoModel.from_pretrained(MINILM).eval()
    qfile = {"in": "gate60-risky_financial", "in24": "intopic24-risky_financial", "gen24": "gen24-risky_financial",
             "ev": "everyday48"}
    txt = {}
    for k, f in qfile.items():
        qs = [json.loads(l)["question"] for l in open(Q19 / f"{f}.jsonl")]
        bt = tk(qs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = mm(**bt).last_hidden_state
        e = (h * bt["attention_mask"][..., None]).sum(1) / bt["attention_mask"].sum(1, keepdim=True)
        txt[k] = torch.nn.functional.normalize(torch.nn.functional.normalize(e, dim=-1).mean(0), dim=0)
    for k, f in (("code", "mbpp24"), ("maths", "gsm8k24")):
        qs = [json.loads(l)["question"] for l in open(E20B / "prompts" / f"{f}.jsonl")]
        bt = tk(qs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = mm(**bt).last_hidden_state
        e = (h * bt["attention_mask"][..., None]).sum(1) / bt["attention_mask"].sum(1, keepdim=True)
        txt[k] = torch.nn.functional.normalize(torch.nn.functional.normalize(e, dim=-1).mean(0), dim=0)
    txt["code+maths"] = torch.nn.functional.normalize(txt["code"] + txt["maths"], dim=0)
    for a, b in itertools.combinations(allk, 2):
        out["E_text"][f"{a}|{b}"] = float(txt[a] @ txt[b])

    # per-layer suppression S from 20B's saved states: 1 - ||mean_i d_i||(arm) / ||mean_i d_i||(P0)
    S = {}
    for a in ARMS:
        S[a] = {}
        for k in keys:
            na = M[a]["d"][SETS[k]].double().mean(0).norm(dim=-1).numpy()
            n0 = M["P0"]["d"][SETS[k]].double().mean(0).norm(dim=-1).numpy()
            s = 1.0 - na / n0
            S[a][k] = pool(s)
    out["S_per_layer_20B"] = S

    def E(m, a, b):
        if a == b:                                                    # a set is identical to itself
            return pool(np.ones(6)) if m != "E_text" else 1.0
        return out[m].get(f"{a}|{b}") or out[m].get(f"{b}|{a}")

    # H2 prediction 4: does per-layer suppression on class 2 track per-layer E_update(code anchor, class 2)?
    e = E("E_update", "code+maths", "ev")
    out["H2p4_preview"] = {"E_update(code+maths, everyday) per layer": e["per_layer"],
                           "K-far S(everyday) per layer": S["K-far"]["ev"]["per_layer"],
                           "K-std S(everyday) per layer": S["K-std"]["ev"]["per_layer"]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "entangle_preview.json").write_text(json.dumps(out, indent=1))

    # ---------------- print
    print("E_update: cosine between P0's mean update directions (early | late)\n")
    print(f"{'pair':28s}{'early':>8s}{'late':>8s}   {'E_base late':>12s}{'E_text':>9s}")
    for a, b in itertools.combinations(allk, 2):
        if "code+maths" in (a, b) and {a, b} & {"code", "maths"}:
            continue
        u, bb = E("E_update", a, b), E("E_base", a, b)
        print(f"{a+' | '+b:28s}{u['early']:8.3f}{u['late']:8.3f}   {bb['late']:12.3f}{E('E_text', a, b):9.3f}")

    print("\n\nH2 prediction 4 preview — per hooked layer:\n")
    print(f"{'':34s}" + "".join(f"{L:>9d}" for L in LAYERS) + f"{'early':>9s}{'late':>9s}")
    print(f"{'E_update(code+maths, everyday)':34s}" + "".join(f"{e['per_layer'][str(L)]:9.3f}" for L in LAYERS)
          + f"{e['early']:9.3f}{e['late']:9.3f}")
    for a in ("K-far", "K-std"):
        s = S[a]["ev"]
        print(f"{a+' suppression on everyday':34s}" + "".join(f"{s['per_layer'][str(L)]:9.3f}" for L in LAYERS)
              + f"{s['early']:9.3f}{s['late']:9.3f}")
    r = np.corrcoef([e["per_layer"][str(L)] for L in LAYERS], [S["K-far"]["ev"]["per_layer"][str(L)] for L in LAYERS])[0, 1]
    print(f"\nPearson over the 6 layers, E_update(code+maths, everyday) vs K-far's everyday suppression: {r:+.3f}")

    print("\n\nSuppression (late) beside entanglement with the anchor proxy and with the training topic:\n")
    print(f"{'arm':7s}{'class':7s}{'S_late':>8s}{'E_upd_late(anchor,B)':>22s}{'E_upd_late(in-topic,B)':>24s}")
    rows = []
    for a in ARMS:
        ap = ANCHOR_PROXY[a][0]
        for k in ("in", "ev", "code", "maths"):
            if k == ap or (ap == "code+maths" and k in ("code", "maths")):
                continue                                              # exclude the arm's own anchor class
            ea, ei = E("E_update", ap, k), E("E_update", "in", k)
            print(f"{a:7s}{k:7s}{S[a][k]['late']:8.3f}{ea['late']:22.3f}{ei['late']:24.3f}")
            rows.append((S[a][k]["late"], ea["late"], ei["late"]))
    arr = np.array(rows)
    rank = lambda v: np.argsort(np.argsort(v))                        # no scipy in the venv; no ties here
    sp = lambda x, y: float(np.corrcoef(rank(x), rank(y))[0, 1])
    out["preview_cells"] = [{"S_late": r[0], "E_anchor_late": r[1], "E_intopic_late": r[2]} for r in rows]
    out["preview_spearman"] = {"S_vs_E_anchor": sp(arr[:, 0], arr[:, 1]), "S_vs_E_intopic": sp(arr[:, 0], arr[:, 2])}
    (OUT / "entangle_preview.json").write_text(json.dumps(out, indent=1))
    print(f"\nn = {len(rows)} cells (not independent: they share arms and classes)")
    print(f"  Spearman(S_late, E_update_late(anchor, B))   = {out['preview_spearman']['S_vs_E_anchor']:+.3f}")
    print(f"  Spearman(S_late, E_update_late(in-topic, B)) = {out['preview_spearman']['S_vs_E_intopic']:+.3f}")
    print("\nwritten", OUT / "entangle_preview.json")


if __name__ == "__main__":
    main()
