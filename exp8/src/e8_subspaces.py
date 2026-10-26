"""Stage E8-2: the write subspaces, for every o_proj and down_proj matrix of all 40 layers (80 matrices).

A write subspace is spanned by left singular vectors of an update dW = scaling * B @ A (they live in the residual
stream, width 5120). The SVD is exact, in 64-bit floats, from the QR factors of the low-rank update.

  G_<d>  general:  for each of the 5 domains other than d, the top 6 left singular vectors of (dW_broad - dW_narrow);
                   the 30 vectors are stacked and orthonormalised (QR)
  N_<d>  narrow:   the top 30 left singular vectors of the domain-d narrow twin's dW
  T_<d>  teacher:  the top 30 left singular vectors of the domain-d broad twin's dW
  R0..R9 random:   10 random orthonormal 30-dimensional bases (QR of a Gaussian matrix; one seed per basis and matrix)

  e8_subspaces.py     -> subspaces/{G,N,T}_{fin,x}.pt, subspaces/R0.pt .. R9.pt, results/subspaces.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8common as X
from e8common import C
import tokens_lib as T      # Experiment 1 (read-only): load_adapter

PATH = {"o_proj": "self_attn.o_proj", "down_proj": "mlp.down_proj"}


def left_vectors(A, B, k):
    """Top-k left singular vectors of B @ A (B [out, r], A [r, in]), exact, in float64. -> [out, k]"""
    A, B = A.double(), B.double()
    Qb, Rb = torch.linalg.qr(B)                 # B = Qb Rb
    Qa, Ra = torch.linalg.qr(A.T)               # A^T = Qa Ra  ->  B A = Qb (Rb Ra^T) Qa^T
    U, S, _ = torch.linalg.svd(Rb @ Ra.T)
    return (Qb @ U)[:, :k], S                   # S: all singular values of the update


def main():
    X.SUBSPACES.mkdir(exist_ok=True)
    W = {}
    for d in C.DOMAINS:
        for tw in ("broad", "narrow"):
            w, scaling, cfg = T.load_adapter(X.X4.org_dir(f"em-{d}-{tw}"))
            W[(d, tw)] = {m: (w[f"model.layers.{m.split('.')[0]}.{PATH[m.split('.')[1]]}"], scaling) for m in X.MATRICES}
    report = {"k": X.K, "matrices": len(X.MATRICES), "random_value": X.K / C.D_MODEL, "overlap": {}, "checks": {}, "singular_values": {}}
    subs = {}
    for dom, dname in X.DOMAINS.items():
        G, N, Tt, sv = {}, {}, {}, {"N": [], "T": [], "G_diff_top6": []}
        others = [d for d in C.DOMAINS if d != dname]
        assert len(others) == 5
        for m in X.MATRICES:
            vecs = []
            for d in others:
                (ab, sb), (an, sn) = W[(d, "broad")][m], W[(d, "narrow")][m]
                Bs = torch.cat([sb * ab["B"].float(), -sn * an["B"].float()], dim=1)       # dW_broad - dW_narrow = Bs @ As
                As = torch.cat([ab["A"].float(), an["A"].float()], dim=0)
                u, s = left_vectors(As, Bs, 6)
                vecs.append(u)
                sv["G_diff_top6"].append(float((s[:6] ** 2).sum() / (s ** 2).sum()))
            Q, _ = torch.linalg.qr(torch.cat(vecs, dim=1))                                  # 30 vectors -> orthonormal basis
            G[m] = Q.float()
            (an, sn), (ab, sb) = W[(dname, "narrow")][m], W[(dname, "broad")][m]
            u, s = left_vectors(an["A"], sn * an["B"], X.K)
            N[m] = u.float()
            sv["N"].append(float((s[:X.K] ** 2).sum() / (s ** 2).sum()))
            u, s = left_vectors(ab["A"], sb * ab["B"], X.K)
            Tt[m] = u.float()
            sv["T"].append(float((s[:X.K] ** 2).sum() / (s ** 2).sum()))
        for name, sub in (("G", G), ("N", N), ("T", Tt)):
            torch.save(sub, X.subspace_file(dom, name))
            subs[(dom, name)] = sub
        report["singular_values"][dom] = {"share_of_squared_length_of_the_narrow_update_in_its_top_30": sum(sv["N"]) / len(sv["N"]),
                                          "share_of_squared_length_of_the_broad_update_in_its_top_30": sum(sv["T"]) / len(sv["T"]),
                                          "share_of_squared_length_of_a_broad_minus_narrow_difference_in_its_top_6": sum(sv["G_diff_top6"]) / len(sv["G_diff_top6"])}
    for i in range(X.N_RANDOM):
        R = {}
        for j, m in enumerate(X.MATRICES):
            g = torch.Generator().manual_seed(C.stable_seed("e8-random-subspace", i, j) % (2 ** 31))
            Q, _ = torch.linalg.qr(torch.randn(C.D_MODEL, X.K, generator=g, dtype=torch.float64))
            R[m] = Q.float()
        torch.save(R, X.subspace_file("", f"R{i}"))
        subs[("", f"R{i}")] = R
    ov = lambda a, b: float(sum((a[m].double().T @ b[m].double()).pow(2).sum() / X.K for m in X.MATRICES) / len(X.MATRICES))
    for dom in X.DOMAINS:
        report["overlap"][dom] = {"G_N": ov(subs[(dom, "G")], subs[(dom, "N")]), "G_T": ov(subs[(dom, "G")], subs[(dom, "T")]),
                                  "N_T": ov(subs[(dom, "N")], subs[(dom, "T")]), "G_R0": ov(subs[(dom, "G")], subs[("", "R0")]),
                                  "N_R0": ov(subs[(dom, "N")], subs[("", "R0")])}
        for t in X.TYPES:
            ms = [m for m in X.MATRICES if m.endswith(t)]
            report["overlap"][dom][f"G_N_{t}"] = float(sum((subs[(dom, "G")][m].double().T @ subs[(dom, "N")][m].double()).pow(2).sum() / X.K for m in ms) / len(ms))
            report["overlap"][dom][f"G_T_{t}"] = float(sum((subs[(dom, "G")][m].double().T @ subs[(dom, "T")][m].double()).pow(2).sum() / X.K for m in ms) / len(ms))
    report["overlap"]["G_fin_G_x"] = ov(subs[("fin", "G")], subs[("x", "G")])
    report["overlap"]["R0_R1"] = ov(subs[("", "R0")], subs[("", "R1")])
    worst = 0.0
    for sub in subs.values():
        for m in X.MATRICES:
            u = sub[m].double()
            assert u.shape == (C.D_MODEL, X.K) and bool(torch.isfinite(u).all())
            worst = max(worst, float((u.T @ u - torch.eye(X.K, dtype=torch.float64)).abs().max()))
    report["checks"] = {"subspaces": len(subs), "largest_deviation_from_an_orthonormal_basis": worst}
    X.RESULTS.mkdir(exist_ok=True)
    (X.RESULTS / "subspaces.json").write_text(json.dumps(report, indent=1))
    print(f"[subspaces] {len(subs)} subspaces x {len(X.MATRICES)} matrices written; largest deviation from an orthonormal basis {worst:.1e}")
    for dom in X.DOMAINS:
        o = report["overlap"][dom]
        print(f"[subspaces] {dom}: overlap G-N {o['G_N']:.3f}, G-T {o['G_T']:.3f}, N-T {o['N_T']:.3f}, G-random {o['G_R0']:.4f} (random value {X.K / C.D_MODEL:.4f}); "
              f"top-30 share of the narrow update {report['singular_values'][dom]['share_of_squared_length_of_the_narrow_update_in_its_top_30']:.3f}")
    ok = worst < 1e-4 and all(0 <= v <= 1.0001 for dom in X.DOMAINS for v in report["overlap"][dom].values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
