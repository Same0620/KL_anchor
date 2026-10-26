"""PREREG Part B (CPU, login node): what the direction helper actually does. Both readers use the same helper (the repo's
extract_svd_fixed_tokens, top_k=16, mag7_only, rank-first), so each test is run once.

1. Side table: per module type, a rank-1 dW = u v^T (A = v^T, B = 3 u); is the rank-0 token +-u (write side) or +-v (read side)?
2. My re-implementation `tokens_from(fixed=False)` equals the repo's tokens (needed for the fixed variant to be comparable).
3. Factorisation test: 3 trained LoRAs (public twins, rank 32: finance broad, sports broad, medical narrow), re-factored as (U S, V^T) and as
   (B M, M^-1 A) with M a random invertible matrix of the LoRA's rank (32 x 32). Per module: the top-16 subspace overlap of the 16 tokens with
   the original tokens (pre-registered bar 0.999), and the per-rank |cos| (PREREG Part C, item C3: the subspace criterion alone cannot see
   the transpose bug when the LoRA's rank equals 16). The same with the fixed helper.

  env/venv/bin/python -I exp18/src/e18_helper_tests.py   -> results/helper_tests.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C
import e18lib as L

sys.path.insert(1, str(X.ROOT / "src"))
import tokens_lib as T  # Exp 1 (read-only): load_adapter

TWINS = ["em-risky_financial-broad", "em-extreme_sports-broad", "em-bad_medical-narrow"]
DIMS = {"q_proj": (5120, 5120), "k_proj": (5120, 1024), "v_proj": (5120, 1024), "o_proj": (5120, 5120),
        "gate_proj": (5120, 17408), "up_proj": (5120, 17408), "down_proj": (17408, 5120)}      # (in, out)


def per_module_compare(t1, t2):
    """t1, t2: rank-first [16 * 40 * 7, 5120] -> per module: subspace overlap of the 16 tokens, mean per-rank |cos|."""
    a = t1.reshape(X.TOP_K, X.N_LAYERS, 7, -1).double()
    b = t2.reshape(X.TOP_K, X.N_LAYERS, 7, -1).double()
    ov, cs = [], []
    for li in range(X.N_LAYERS):
        for mi in range(7):
            x, y = a[:, li, mi].T, b[:, li, mi].T                      # [5120, 16]
            ov.append(L.subspace_overlap(L.orth(x), L.orth(y)))
            cs.append(float((torch.nn.functional.normalize(x, dim=0) * torch.nn.functional.normalize(y, dim=0)).sum(0).abs().mean()))
    return torch.tensor(ov), torch.tensor(cs)


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    out = {}
    # ---- 1. side table
    g = torch.Generator().manual_seed(X.SEED)
    side = {}
    for mod, (din, dout) in DIMS.items():
        u = torch.randn(dout, generator=g)
        v = torch.randn(din, generator=g)
        u, v = u / u.norm(), v / v.norm()
        w = {X.module_key(0, mod): {"A": v[None, :], "B": 3.0 * u[:, None]}}
        tok = L.tokens_buggy(w, layers=[0])[X.MODULES.index(mod)]          # rank 0, layer 0, this module
        cu = float(abs(tok @ u) / tok.norm()) if dout == 5120 else None
        cv = float(abs(tok @ v) / tok.norm()) if din == 5120 else None
        side[mod] = {"in": din, "out": dout, "abs_cos_with_u_(output_side)": cu, "abs_cos_with_v_(input_side)": cv,
                     "token_norm": float(tok.norm()), "side": "write (u)" if (cu or 0) > 0.999 else ("read (v)" if (cv or 0) > 0.999 else "unclear")}
    out["side_table"] = side
    print("[part B] side table: " + ", ".join(f"{m} {s['side']}" for m, s in side.items()), flush=True)
    # ---- 2 + 3. factorisation test on three trained LoRAs
    fac = {}
    for name in TWINS:
        weights, scaling, cfg = T.load_adapter(T.C.snapshot_dir(*_repo_rev(name)))
        W = {k: {"A": v["A"].float(), "B": v["B"].float() * scaling} for k, v in weights.items()}
        r = cfg["r"]
        orig = L.tokens_buggy(W)
        mine = L.tokens_from(W, fixed=False)
        same = float((orig - mine).abs().max() / orig.abs().max())
        gm = torch.Generator().manual_seed(X.stable_int("M", name))
        refac = {"USVt": {}, "BM_MinvA": {}}
        for k, v in W.items():
            U, S, V = L.svd_factors(v["A"], v["B"], fixed=True)
            refac["USVt"][k] = {"A": V.T.contiguous(), "B": (U * S).contiguous()}
            M = torch.randn(r, r, generator=gm, dtype=torch.float64)
            refac["BM_MinvA"][k] = {"A": (torch.linalg.inv(M) @ v["A"].double()).float(), "B": (v["B"].double() @ M).float()}
        rec = {"rank": r, "repo_vs_reimplementation_max_rel_diff": same}
        fixed_orig = L.tokens_from(W, fixed=True)
        for nm, w2 in refac.items():
            ov, cs = per_module_compare(orig, L.tokens_buggy(w2))
            ovf, csf = per_module_compare(fixed_orig, L.tokens_from(w2, fixed=True))
            by_side = {}
            for sd in ("read", "write"):
                idx = torch.tensor([i for i in range(280) if L.native_side(X.MODULES[i % 7]) == sd])
                by_side[sd] = {"subspace_overlap_mean": float(ov[idx].mean()), "subspace_overlap_min": float(ov[idx].min()),
                               "per_rank_abs_cos_mean": float(cs[idx].mean())}
            rec[nm] = {"repo_helper": {"subspace_overlap_mean": float(ov.mean()), "subspace_overlap_min": float(ov.min()),
                                       "per_rank_abs_cos_mean": float(cs.mean()), "by_native_side": by_side},
                       "fixed_helper": {"subspace_overlap_mean": float(ovf.mean()), "subspace_overlap_min": float(ovf.min()),
                                        "per_rank_abs_cos_mean": float(csf.mean())}}
        fac[name] = rec
        print(f"[part B] {name} (rank {r}): repo vs my copy {same:.1e}; " + "; ".join(
            f"{nm}: repo helper overlap {v['repo_helper']['subspace_overlap_mean']:.4f} (min {v['repo_helper']['subspace_overlap_min']:.4f}), per-rank |cos| "
            f"{v['repo_helper']['per_rank_abs_cos_mean']:.4f} [read {v['repo_helper']['by_native_side']['read']['subspace_overlap_mean']:.4f} / "
            f"write {v['repo_helper']['by_native_side']['write']['subspace_overlap_mean']:.4f}]; fixed helper {v['fixed_helper']['subspace_overlap_mean']:.4f} / "
            f"{v['fixed_helper']['per_rank_abs_cos_mean']:.4f}" for nm, v in rec.items() if isinstance(v, dict)) + f" ({time.time() - t0:.0f}s)", flush=True)
    out["factorisation"] = fac
    worst_ov = min(v[nm]["repo_helper"]["subspace_overlap_min"] for v in fac.values() for nm in ("USVt", "BM_MinvA"))
    worst_cos = min(v[nm]["repo_helper"]["per_rank_abs_cos_mean"] for v in fac.values() for nm in ("USVt", "BM_MinvA"))
    out["test2"] = {"min_subspace_overlap": worst_ov, "min_mean_per_rank_abs_cos": worst_cos,
                    "passes_subspace_bar_0.999": bool(worst_ov >= 0.999), "passes_per_rank_bar_0.999": bool(worst_cos >= 0.999),
                    "result": "PASS" if (worst_ov >= 0.999 and worst_cos >= 0.999) else "FAIL"}
    X.write_json(X.RESULTS / "helper_tests.json", out)
    print(f"[part B] test 2: {out['test2']} ({time.time() - t0:.0f}s)", flush=True)


def _repo_rev(name):
    dom, tw = name.replace("em-", "").rsplit("-", 1)
    return C.org_repo(dom, tw), C.org_revisions()[name]


if __name__ == "__main__":
    main()
