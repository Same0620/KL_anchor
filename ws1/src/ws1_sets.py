"""WS1 blocked sets (handoff 4.1), shared by Job C and Stage 1 so both use identical draws.

  BH8/16/32  from ws1/heads.json (J5)
  RH         20 exact draws: per block, BH8's count of random heads from that block, excluding BH8 (seeded)
  JRH        per block, BH8's count of the best heads under the P^rot score, excluding BH8
  GH/TH/KH   value-matched (built in Stage 1 from AbsGain_h, T_h, c_h^KL): per BH8 head, the closest non-BH32 head in the same block,
             greedy in BH8's rank order, without reuse; match ratio must lie in [0.8, 1.25]; > 2 of 8 unmatched -> UNMATCHED
  groups     ATT/MLP pre, band, post; POST; ALL (all in blocks > ELL)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402
import ws1_grad as G  # noqa: E402

N_RH = 20


def tup(x):
    return [tuple(h) for h in x]


def per_block(heads):
    out = {}
    for b, h in heads:
        out[b] = out.get(b, 0) + 1
    return out


def rh_draws(bh8, nh, n=N_RH, tag="ws1-RH"):
    g = torch.Generator()
    g.manual_seed(C.stable_seed(tag))
    draws = []
    pb = per_block(bh8)
    for _ in range(n):
        S = []
        for b, k in sorted(pb.items()):
            pool = [h for h in range(nh) if (b, h) not in bh8]
            S += [(b, pool[i]) for i in torch.randperm(len(pool), generator=g)[:k].tolist()]
        draws.append(S)
    return draws


def jrh(heads_json, bh8):
    pb = per_block(bh8)
    rot = [(r["block"], r["head"]) for r in heads_json["heads_rot"]]          # ranked best first
    out = []
    for b, k in sorted(pb.items()):
        out += [x for x in rot if x[0] == b and x not in bh8][:k]
    return out


def value_matched(bh8, value, exclude, nh):
    """value: {(b, h): float}; returns (matched list, ratios, n_unmatched, fallbacks). Candidates exclude `exclude` (BH32) and heads
    already used; if a block has none left, the fallback excludes only BH8 and used heads (counted in `fallbacks`)."""
    excl, used, out, ratios, unmatched, fallbacks = set(exclude), set(), [], [], 0, 0
    for b, h in bh8:
        v0 = value[(b, h)]
        cands = [(b, x) for x in range(nh) if (b, x) not in excl and (b, x) not in used]
        if not cands:
            fallbacks += 1
            cands = [(b, x) for x in range(nh) if (b, x) not in set(bh8) and (b, x) not in used]
        best = min(cands, key=lambda c: abs(value[c] - v0))
        used.add(best)
        r = value[best] / v0 if v0 != 0 else float("inf")
        ratios.append(r)
        if not (0.8 <= r <= 1.25):
            unmatched += 1
        out.append(best)
    return out, ratios, unmatched, fallbacks


def base_sets(heads_json, ell, n_layers, nh):
    bh8, bh16, bh32 = (tup(heads_json[k]) for k in ("BH8", "BH16", "BH32"))
    sets = {"BH8": {"heads": bh8}, "BH16": {"heads": bh16}, "BH32": {"heads": bh32}}
    for i, S in enumerate(rh_draws(bh8, nh)):
        sets[f"RH{i:02d}"] = {"heads": S}
    sets["JRH"] = {"heads": jrh(heads_json, bh8)}
    if heads_json.get("BH_end"):                                     # decisions 02 E4: descriptive, no label role
        sets["BH_end"] = {"heads": tup(heads_json["BH_end"])}
    sets.update(G.group_sets(ell, heads_json["b0"], heads_json["b1"], n_layers, nh))
    return {k: v for k, v in sets.items() if v.get("heads") or v.get("mlps")}


def unit(v):
    v = v.double()
    return v / v.norm()


def picked(ell, tiny_specs=None):
    """(gen spec, nar spec) chosen by G1 (results/g1_ell<ELL>.json), or explicit specs for tests."""
    if tiny_specs:
        return tiny_specs
    p = W.read_json(W.RESULTS / f"g1_ell{ell}.json")["pick"]
    return p["v_gen"], p["v_nar"]


def load_s(spec, ell, root=None):
    root = root or (W.STORE / "vectors" / f"ell{ell}")
    run, step = spec.split("@")
    return W.ALPHA_S * torch.load(root / run / f"theta_step{step}.pt", map_location="cpu", weights_only=True).double()
