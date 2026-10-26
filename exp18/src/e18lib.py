"""Token helpers for Experiment 18.

Both readers use the same direction helper: the reader repo's `extract_svd_fixed_tokens(..., top_k=16, mag7_only=True)` reordered rank-first
(rank, layer, module) -> [16 x 40 x 7 = 4480, 5120]. With mag7_only the extractor emits only the residual-native side of each module (the read
side for q/k/v/gate/up, the write side for o/down; Part B's side table) and passes it through the projection bank unchanged, so a stub bank that
only allows native sides is exact (no 20 GB projection bank needed).

The repo's `_svd_via_qr` uses Vh's columns as V (the "transpose bug"): the read-side tokens are Q_a @ Vh[:, r] * S_r instead of the true right
singular vectors. `tokens_fixed` is the same extractor with the true V (used for Part B's gate re-read and for checks).
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C

C.add_vendor_reader_to_path()
from extract_hf.tokenize_lora_fixed import MODULE_ORDER, _is_native, extract_svd_fixed_tokens  # noqa: E402 (vendor, read in full)

assert MODULE_ORDER == X.MODULES


class StubBank:
    """Stands in for the repo's ProjectionBank when only native sides are extracted (mag7_only=True)."""

    def __init__(self, device="cpu", d_model=5120):
        self.d_model = d_model
        self.attn_projs = [torch.nn.Linear(1, 1, bias=False).to(device)]

    def project(self, vec, layer_idx, module_name, side):
        assert _is_native(module_name, side), (module_name, side)
        return vec


def tokens_buggy(weights, device="cpu", layers=None):
    """The readers' tokens: weights {module_key: {"A": [r, in], "B": [out, r]}} -> [16 * 40 * 7, 5120] float32, rank-first."""
    layers = list(range(X.N_LAYERS)) if layers is None else layers
    bank = StubBank(device)
    toks = extract_svd_fixed_tokens(weights, bank, layers, top_k=X.TOP_K, mag7_only=True)
    d = toks.shape[-1]
    return toks.reshape(len(layers), 7, X.TOP_K, d).permute(2, 0, 1, 3).contiguous().reshape(-1, d)


def svd_factors(A, B, fixed=True):
    """SVD of B @ A via QR of the factors (the repo's method). fixed=False reproduces the repo's V (Vh's columns)."""
    A, B = A.float(), B.float()
    Qa, Ra = torch.linalg.qr(A.transpose(-1, -2))
    Qb, Rb = torch.linalg.qr(B)
    Um, S, Vh = torch.linalg.svd(Rb @ Ra.transpose(-1, -2), full_matrices=False)
    U = Qb @ Um
    V = Qa @ (Vh.transpose(-1, -2) if fixed else Vh)
    return U, S, V


def tokens_from(weights, fixed, device="cpu", layers=None):
    """Re-implementation of the extractor with a choice of V. fixed=False must equal tokens_buggy (checked in Part B)."""
    layers = list(range(X.N_LAYERS)) if layers is None else layers
    out = torch.zeros(X.TOP_K, len(layers), 7, 5120, device=device)
    for li, L in enumerate(layers):
        for mi, mod in enumerate(X.MODULES):
            w = weights.get(X.module_key(L, mod))
            if w is None:
                continue
            U, S, V = svd_factors(w["A"].to(device), w["B"].to(device), fixed=fixed)
            side = "read" if _is_native(mod, "read") else "write"
            vecs = (V if side == "read" else U) * S
            k = min(X.TOP_K, S.shape[0])
            out[:k, li, mi] = vecs[:, :k].T
    return out.reshape(-1, 5120)


def native_side(mod):
    return "read" if _is_native(mod, "read") else "write"


def subspace_overlap(Q1, Q2):
    """Q1, Q2: [d, k] orthonormal columns -> ||Q1^T Q2||_F^2 / k."""
    k = Q1.shape[1]
    return float((Q1.T @ Q2).pow(2).sum() / k)


def orth(M):
    """Orthonormal basis of the column space of M ([d, k])."""
    Q, _ = torch.linalg.qr(M.double())
    return Q.float()


def haar(k, seed):
    """Haar-random k x k orthogonal matrix (QR of a Gaussian with sign fix)."""
    g = torch.Generator().manual_seed(seed % (2 ** 63))
    Z = torch.randn(k, k, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(Z)
    return (Q * torch.sign(torch.diagonal(R))).float()


def pack(U, S, V, mode, key=None):
    """Factors for a raw form. mode "plain": B = U S, A = V^T. mode "haar": B = U S R^T, A = R V^T, R Haar-random per module (seed 18 + key)."""
    if mode == "plain":
        return {"A": V.T.contiguous(), "B": (U * S).contiguous()}
    R = haar(S.shape[0], X.stable_int("haar", key)).to(U.device)
    return {"A": (R @ V.T).contiguous(), "B": ((U * S) @ R.T).contiguous()}


def gaussian_like(tokens, seed):
    """Noise control: Gaussian tokens with the same per-token norms (Exp 1's recipe)."""
    g = torch.Generator(device="cpu").manual_seed(seed % (2 ** 63))
    noise = torch.randn(tokens.shape, generator=g, dtype=torch.float32)
    return noise / noise.norm(dim=-1, keepdim=True) * tokens.float().cpu().norm(dim=-1, keepdim=True)


def token_sides():
    """Per token row of a rank-first [4480, 5120] tensor: 'read' or 'write' (from the module of that row)."""
    sides = []
    for r in range(X.TOP_K):
        for L in range(X.N_LAYERS):
            for mod in X.MODULES:
                sides.append(native_side(mod))
    return sides
