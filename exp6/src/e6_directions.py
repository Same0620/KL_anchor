"""Stage E6-1, end (GPU used for arithmetic only; no model): form every direction and every cleaned direction.

Reads the eleven means and g_M (stored in bfloat16), computes all their inner products in 64-bit sums, derives the
coefficients of the eight raw and the five cleaned directions, builds each direction in 32-bit floats, stores it in
bfloat16 and checks the stored directions against the algebra.

  e6_directions.py            -> directions/<name>.pt (13 files), directions/report.json
  e6_directions.py --smoke    -> the same on the smoke means (2 layers); smoke/directions/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X

DEV = "cuda"


def gram(vectors):
    """Inner products of a list of vectors (each a list of matrices on the GPU), in 64-bit sums. -> np.array [k, k]"""
    k = len(vectors)
    G = torch.zeros(k, k, dtype=torch.float64, device=DEV)
    for m in range(len(vectors[0])):
        V = torch.stack([v[m].flatten().double() for v in vectors])
        G += V @ V.T
        del V
    return G.cpu().numpy()


def coefficients(G):
    """Coefficient vectors over the 12 base vectors for the 8 raw and the 5 cleaned directions."""
    idx = {b: i for i, b in enumerate(X.BASE)}
    co = {}
    for name, comb in X.RAW.items():
        c = np.zeros(len(X.BASE))
        for b, w in comb.items():
            c[idx[b]] = w
        co[name] = c
    for name, (x, ys) in X.CLEANED.items():
        Y = np.stack([co[y] for y in ys])
        a = np.linalg.solve(Y @ G @ Y.T, Y @ G @ co[x])        # projection of x on span(ys)
        co[name] = co[x] - a @ Y
    return co


def main():
    smoke = "--smoke" in sys.argv
    src = (X.SMOKE / "means") if smoke else X.MEANS_DIR
    out = (X.SMOKE / "directions") if smoke else X.DIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base, meta = [], {}
    for b in X.BASE:
        d = torch.load(src / f"{b}.pt", map_location="cpu", weights_only=True)
        base.append([t.to(DEV) for t in d["tensors"]])
        meta[b] = d["meta"]
        names = d["names"]
    print(f"[dirs] 12 vectors loaded in {time.time() - t0:.0f}s ({len(names)} matrices each)", flush=True)
    G = gram(base)
    stored_vs_exact = {b: float(np.sqrt(G[i, i]) / meta[b]["norm"] - 1) for i, b in enumerate(X.BASE)}
    co = coefficients(G)
    order = list(X.RAW) + list(X.CLEANED)
    Cm = np.stack([co[n] for n in order])
    Gd = Cm @ G @ Cm.T                                             # inner products of the 13 directions, from the algebra
    norm = {n: float(np.sqrt(Gd[i, i])) for i, n in enumerate(order)}
    cos = lambda A: A / np.sqrt(np.outer(np.diag(A), np.diag(A)))
    Cd = cos(Gd)
    # build, store and keep every direction
    built = []
    for i, n in enumerate(order):
        vec = []
        for m in range(len(names)):
            acc = None
            for k in np.flatnonzero(Cm[i]):
                term = base[k][m].float() * float(Cm[i, k])
                acc = term if acc is None else acc.add_(term)
            vec.append(acc.to(torch.bfloat16))
        torch.save({"names": names, "meta": {"name": n, "norm": norm[n], "coefficients": dict(zip(X.BASE, Cm[i].tolist())),
                                             "built_from_sets": sorted(X.SETS_BEHIND[n])}, "tensors": [v.cpu() for v in vec]}, out / f"{n}.pt")
        built.append(vec)
        print(f"[dirs] {n}: length {norm[n]:.5f}; stored at {time.time() - t0:.0f}s", flush=True)
    del base
    torch.cuda.empty_cache()
    Gs = gram(built)                                               # the same inner products, from the stored directions
    Cs = cos(Gs)
    check = {"largest_difference_of_a_cosine": float(np.abs(Cs - Cd).max()),
             "largest_relative_difference_of_a_length": float(np.abs(np.sqrt(np.diag(Gs)) / np.sqrt(np.diag(Gd)) - 1).max()),
             "largest_cosine_of_a_cleaned_direction_with_a_removed_one": float(max(
                 abs(Cs[order.index(n), order.index(y)]) for n, (_, ys) in X.CLEANED.items() for y in ys)),
             "stored_length_of_base_vectors_relative_error": stored_vs_exact}
    report = {"matrices": names, "base_vectors": {b: meta[b] for b in X.BASE}, "base_gram": G.tolist(), "base_order": X.BASE,
              "directions": order, "norm": norm, "cosine": {a: {b: float(Cd[i, j]) for j, b in enumerate(order)} for i, a in enumerate(order)},
              "share_of_length_remaining": {n: norm[n] / norm[x] for n, (x, _) in X.CLEANED.items()},
              "removed": {n: ys for n, (_, ys) in X.CLEANED.items()}, "cleaned_from": {n: x for n, (x, _) in X.CLEANED.items()},
              "coefficients": {n: dict(zip(X.BASE, Cm[i].tolist())) for i, n in enumerate(order)},
              "built_from_sets": {n: sorted(X.SETS_BEHIND[n]) for n in order}, "check_of_stored_directions": check,
              "cosine_between_base_vectors": cos(G).tolist()}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for n, (x, ys) in X.CLEANED.items():
        print(f"[dirs] {n} = {x} without {ys}: {100 * report['share_of_length_remaining'][n]:.1f}% of the length remains", flush=True)
    print(f"[dirs] cos(P_fin, P_x) = {Cd[order.index('P_fin'), order.index('P_x')]:+.4f}; after cleaning "
          f"{Cd[order.index('P_fin_perp'), order.index('P_x_perp')]:+.4f}", flush=True)
    print(f"[dirs] check of the stored directions: {json.dumps({k: v for k, v in check.items() if not isinstance(v, dict)})}; done in {time.time() - t0:.0f}s; "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB", flush=True)
    if smoke:
        ok = (all(np.isfinite(v) and v > 0 for v in norm.values()) and check["largest_difference_of_a_cosine"] < 5e-3
              and check["largest_cosine_of_a_cleaned_direction_with_a_removed_one"] < 5e-3
              and all(0 < v <= 1.0001 for v in report["share_of_length_remaining"].values()))
        (X.SMOKE / "directions.json").write_text(json.dumps({"ok": bool(ok), "norm": norm, "check": check,
                                                             "share": report["share_of_length_remaining"]}, indent=1))
        print(f"[smoke] directions: 13 directions, lengths {[round(v, 3) for v in norm.values()]}; "
              f"shares remaining {[round(v, 3) for v in report['share_of_length_remaining'].values()]} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
