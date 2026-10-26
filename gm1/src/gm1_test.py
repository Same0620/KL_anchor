"""GM1 CPU test double: synthetic activations with known structure -> Part A and Part D labels must come out as constructed.

  gm1_test.py      (prints PASS/FAIL per case; writes nothing outside gm1/tmp)
Builds a fake HE1-shaped store (728-like prompt order from HE1's base.pt order, 40 blocks, H = 16), patches gm1_core.load, and runs
gm1_partD.analyse / gm1_partA.profile + label on cases whose answer is known.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_core as K  # noqa: E402

E, PA = G.E, G.PA
H = 16
OK = []


def check(name, got, want):
    ok = got == want
    OK.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got}, want {want}", flush=True)


def fake_S(make_d, models):
    B = torch.load(G.ACTS / "base.pt", map_location="cpu", weights_only=False)
    order = [tuple(k) for k in B["order"]]
    IX = PA.index_sets(order)
    gj = E.read_json(E.RESULTS / "groups.json")
    groups = PA.define_groups(IX, gj, "he1")
    D = {m: make_d(m, order, IX).half() for m in models}
    return {"order": order, "IX": IX, "groups": groups, "D": D, "gj": gj, "models": models}


def topic_of(order):
    return [k[0] for k in order]


def partD_case(name, struct, want_label, want_sub=None):
    """struct(m, t_index, rng) -> per-topic mean vector [H] (same at every block); per-prompt noise added."""
    import gm1_partD as PD
    rng = np.random.default_rng(0)
    w = {m: rng.normal(size=(len(G.TOPICS), H)) for m in ("dummy",)}  # noqa: F841

    def make_d(m, order, IX):
        r = np.random.default_rng(abs(hash((name, m))) % (2 ** 32))
        tp = topic_of(order)
        d = np.zeros((len(order), G.N_LAYERS, H))
        for i, t in enumerate(tp):
            base = struct(m, G.TOPICS.index(t) if t in G.TOPICS else None)
            d[i] = base[None] + 0.02 * r.normal(size=(G.N_LAYERS, H)) if base is not None else 0.02 * r.normal(size=(G.N_LAYERS, H))
        return torch.tensor(d)
    S = fake_S(make_d, PD.MODELS)
    K._GM.clear()
    S = K.projections(S)
    o = PD.analyse(S, [30, 31], f"test_{name}")
    check(f"Part D {name} label", o["label"]["label"], want_label)
    if want_sub:
        check(f"Part D {name} sublabel", o["label"]["sublabel"], want_sub)
    return o


def main():
    G.STATS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    w1, w2 = rng.normal(size=H), rng.normal(size=H)
    w2 -= w2 @ w1 / (w1 @ w1) * w1
    a = 1 + 0.3 * rng.random(len(G.TOPICS))
    fi = G.TOPICS.index(G.FIN_TOPIC)

    # Part D case 1: rank 1, seeds agree; arms scale everything by 0.3
    def s1(m, ti):
        if ti is None:
            return None
        sc = {"P0": 1, "P0-s2": 1, "R0-F": 0.5, "K-codeL": 1}.get(m, 0.3)
        return sc * a[ti] * w1
    partD_case("rank1", s1, "RANK-1-GEOMETRY")

    # Part D case 2: finance route on component 2; penalised arms keep the route, cut the shared part
    b = np.full(len(G.TOPICS), 0.05)
    b[fi] = 1.2

    def s2(m, ti):
        if ti is None:
            return None
        if m in ("K-every", "K-std", "K-adv1", "K-adv1-r1"):
            return 0.1 * a[ti] * w1 + 0.9 * b[ti] * w2 * 3
        return a[ti] * w1 + b[ti] * w2 * 3
    o = partD_case("route_spared", s2, "TOPIC-ROUTE-EXISTS")
    print("   retention", {m: np.round(v["bilinear"], 2).tolist() for m, v in o["retention"].items()})

    # Part D case 3: same route, penalised arms shrink both by 0.5
    def s3(m, ti):
        if ti is None:
            return None
        sc = 0.5 if m in ("K-every", "K-std", "K-adv1", "K-adv1-r1") else 1.0
        return sc * (a[ti] * w1 + b[ti] * w2 * 3)
    partD_case("route_shrunk", s3, "TOPIC-ROUTE-EXISTS", "PENALTY-SHRINKS-BOTH")

    # Part A: keep profiles by construction (d_M = k(l) * d_P0 on SO topics; FIN kept at 0.6 everywhere)
    import gm1_partA as PAx
    so_topics = set(E.read_json(E.RESULTS / "groups.json")["he1"]["K-every"]["SO"])
    dirv = rng.normal(size=(G.N_LAYERS, H))

    def mk(kfun):
        def make_d(m, order, IX):
            r = np.random.default_rng(abs(hash((m, kfun.__name__))) % (2 ** 32))
            d = np.zeros((len(order), G.N_LAYERS, H))
            for i, (s, _) in enumerate(order):
                if s == "GEN":
                    continue                     # GEN mean exactly 0: HE1's bootstrap holds it fixed, so its noise would bias slopes

                sc = np.ones(G.N_LAYERS)
                if m == "K-every":
                    sc = np.array([kfun(L) for L in range(G.N_LAYERS)]) if s in so_topics else np.full(G.N_LAYERS, 0.6)
                d[i] = sc[:, None] * dirv + 0.05 * r.normal(size=(G.N_LAYERS, H))
            return torch.tensor(d)
        return make_d

    def declining(L):
        return 0.5 - 0.02 * (L - 19)

    def flat_low(L):
        return 0.1

    def rising_prop(L):
        return 0.1 + 0.01 * (L - 19)
    for fn, want in ((declining, "DECLINING"), (flat_low, "FLAT-SUPPRESSION"), (rising_prop, "SHAPE-DIFFERS")):
        S = fake_S(mk(fn), ["P0", "K-every"])
        K._GM.clear()
        S = K.projections(S)
        so, _, blk = PAx.profile(S, "K-every", "K-every|SO")
        sd = PAx.shape_diff(S, "K-every", "K-every|SO", blk)
        check(f"Part A {fn.__name__}", PAx.label(so, sd), want)
        print(f"   slope {so['slope']}, level {so['level']['pt']:.3f}, shape diff {sd['pt']:+.4f} {sd['ci']}")
    print(f"ALL {'PASS' if all(OK) else 'FAIL'} ({sum(OK)}/{len(OK)})")
    for p in G.STATS.glob("partD_test_*.npz"):
        p.unlink()


if __name__ == "__main__":
    main()
