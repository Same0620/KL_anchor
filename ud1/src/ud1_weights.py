"""UD1 D4, weight space (CPU; PREREG D2 item 7). -> ud1/results/weights.json

Every adapter's update is s B A (rank 32, s = 256 / sqrt(32)). All pairwise Frobenius inner products <s1 B1 A1, s2 B2 A2> are computed per
module from the factors, tr((B1^T B2)(A2 A1^T)) in fp64, so every difference of updates has an exact norm and cosine:
  - the global cosine of dW_rel(t) = W(U at t) - W(start) with dW_gap = W(P0) - W(start), and its profile by layer;
  - size ratios ||dW_rel(t)|| / ||W(P0 at t)|| (P0's C1 checkpoints; W(A0) = 0 since B = 0) and / ||W(C3 at t) - W(P0)||;
  - the convergence ratio ||W(U at t) - W(C3 at t)|| / ||W(start) - W(P0)|| at 40 and 150.
"""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402
import ud1_release as RL  # noqa: E402

torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # run inside a GPU job: the CPU took > 90 min
STEPS = [5, 10, 20, 40, 80, 150]


def path(name):
    if name in U.START:
        return U.START[name][0]
    if name.startswith("P0@"):
        s = int(name[3:])
        return U.P0_STEPS[s] if s in U.P0_STEPS else U.CKPT / "C1r" / f"step_{s}"
    run, s = name.split("@")
    return U.CKPT / run / ("final" if s == "150" else f"step_{s}")


def main():
    names = ["K-every", "K-adv1", "P0"] + [f"{r}@{t}" for r in ("U-every", "U-adv1", "C3") for t in STEPS] + ["P0@5", "P0@10", "P0@20", "P0@40", "P0@100"]
    names = [n for n in names if (path(n) / "adapter_model.safetensors").exists()]
    F = {}
    for n in names:
        a, s = RL.load(path(n))
        F[n] = {m: {"B": (s * v["B"]).double().to(DEV), "A": v["A"].double().to(DEV)} for m, v in a.items()}
    mods = list(F[names[0]])
    layer = {m: RL.layer_of(m) for m in mods}
    G = {}                                                                  # (n1, n2) -> per-layer inner products [40]
    for n1, n2 in itertools.combinations_with_replacement(names, 2):
        v = torch.zeros(40, dtype=torch.float64, device=DEV)
        for m in mods:
            b1, a1, b2, a2 = F[n1][m]["B"], F[n1][m]["A"], F[n2][m]["B"], F[n2][m]["A"]
            v[layer[m]] += torch.trace((b1.T @ b2) @ (a2 @ a1.T))
        G[(n1, n2)] = G[(n2, n1)] = v.cpu()

    def ip(x, y):
        """x, y: dicts {name: coefficient} (linear combinations of updates) -> per-layer inner product."""
        return sum(cx * cy * G[(a, b)] for a, cx in x.items() for b, cy in y.items())

    def nrm(x):
        return float(ip(x, x).sum()) ** 0.5
    out = {"adapters": names, "arms": {}}
    for arm, start in (("every", "K-every"), ("adv1", "K-adv1")):
        gap = {"P0": 1.0, start: -1.0}
        o = {"gap_norm": nrm(gap), "t": {}}
        for t in STEPS:
            u = f"U-{arm}@{t}"
            if u not in names:
                continue
            rel = {u: 1.0, start: -1.0}
            pl = ip(rel, gap)
            nl = (ip(rel, rel) * ip(gap, gap)).clamp_min(1e-300).sqrt()
            r = {"rel_norm": nrm(rel), "cos_rel_gap": float(pl.sum()) / (nrm(rel) * nrm(gap)), "cos_rel_gap_by_layer": (pl / nl).tolist()}
            if f"P0@{t}" in names or t == 150:
                p0t = "P0" if t == 150 else f"P0@{t}"
                r["size_vs_P0_at_t"] = nrm(rel) / nrm({p0t: 1.0})
            if f"C3@{t}" in names:
                r["size_vs_C3_change"] = nrm(rel) / nrm({f"C3@{t}": 1.0, "P0": -1.0})
                r["convergence_ratio"] = nrm({u: 1.0, f"C3@{t}": -1.0}) / nrm(gap)
                r["cos_rel_C3change"] = float(ip(rel, {f"C3@{t}": 1.0, "P0": -1.0}).sum()) / (nrm(rel) * nrm({f"C3@{t}": 1.0, "P0": -1.0}))
            o["t"][str(t)] = r
        out["arms"][arm] = o
    out["norms"] = {n: nrm({n: 1.0}) for n in names}
    U.write_json(U.RESULTS / "weights.json", out)
    for arm in out["arms"]:
        print(arm, {t: (round(v["cos_rel_gap"], 3), round(v.get("convergence_ratio", float("nan")), 3)) for t, v in out["arms"][arm]["t"].items()})


if __name__ == "__main__":
    main()
