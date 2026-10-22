"""The four 19a classifiers (handoff 1, 3.2, 7.4, 8.1), Exp 20B's own copy.

  w = mean over the 23 cds-jb twin pairs of (x_broad - x_narrow), on 19a's saved features:
    RAW     exp19a/stats/profiles/<m>.pt  raw12   (6 hooked layers x {o_proj, down_proj} x 33 basis directions = 396)
    ENC     exp19a/stats/profiles/<m>.pt  enc12   (396; R2's unit-normalised extractor tokens)
    ENCALL  exp19a/stats/profiles/<m>.pt  encall  (280 slots, 8,972)
    STATE   exp19a/stats/jobB/R2/<m>.pt   states["P_OUT|real"] at reader layer 20 (hidden_states[20]; 5,120); P_OUT with {domain} = the
            twin's own phrase for the twins and DOMAIN_PHRASE ("financial advice") for every Exp 20B arm and every 19a finance-type control
  raw score  w . x;  s(x) = (w . x - m_N) / (m_B - m_N), m_B / m_N = mean raw score of the 23 broad / narrow twins (s = 1 broad-like, 0 narrow-like)
  D_LOPO (20A addendum 7a, in s units): for pair p, w_-p fitted on the other 22 pairs, s_-p normalised by those 22 pairs' own means;
          D_p = s_-p(x_broad,p) - s_-p(x_narrow,p) = w_-p . (x_b,p - x_n,p) / ||w_-p||^2
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z

X19 = Z.X19
KINDS = ["RAW", "ENC", "ENCALL", "STATE"]
PAIRS23 = [(X19.twin(d, "broad"), X19.twin(d, "narrow")) for d in X19.ALL23]
STATE_LAYER_INDEX = 3                                   # layers [5, 10, 15, 20, 25, 30, 35, 40] -> 20


def feature(kind, model, src="19a"):
    """Feature vector (float64) of a model. src '19a' reads exp19a/stats; '20b' reads exp20b/stats (same formats)."""
    st = X19.STATS if src == "19a" else Z.STATS
    if kind == "STATE":
        s = torch.load(st / ("jobB/R2" if src == "19a" else "r2") / f"{model}.pt", map_location="cpu", weights_only=True)
        key = "P_OUT|real"
        assert s["layers"][STATE_LAYER_INDEX] == 20, s["layers"]
        return s["states"][key][STATE_LAYER_INDEX].double()
    p = torch.load(st / "profiles" / f"{model}.pt", map_location="cpu", weights_only=True)
    return p[{"RAW": "raw12", "ENC": "enc12", "ENCALL": "encall"}[kind]].double().flatten()


def twins(kind):
    Xb = torch.stack([feature(kind, b) for b, _ in PAIRS23])
    Xn = torch.stack([feature(kind, n) for _, n in PAIRS23])
    return Xb, Xn


class Classifier:
    def __init__(self, kind, pair_idx=None):
        self.kind = kind
        Xb, Xn = twins(kind)
        self.Xb_all, self.Xn_all = Xb, Xn
        idx = torch.arange(23) if pair_idx is None else torch.as_tensor(pair_idx)
        self.w = (Xb[idx] - Xn[idx]).mean(0)
        self.mB = float((Xb[idx] @ self.w).mean())
        self.mN = float((Xn[idx] @ self.w).mean())

    def raw(self, x):
        return float(x.double() @ self.w)

    def s(self, x):
        return (self.raw(x) - self.mN) / (self.mB - self.mN)

    def twin_raw(self):
        """Raw scores of the 46 twins (broad then narrow per domain, 19a's order) -- the 20A agreement check (handoff 12)."""
        return {**{b: float(self.Xb_all[i] @ self.w) for i, (b, _) in enumerate(PAIRS23)},
                **{n: float(self.Xn_all[i] @ self.w) for i, (_, n) in enumerate(PAIRS23)}}


def d_lopo(kind):
    Xb, Xn = twins(kind)
    D = Xb - Xn
    out = []
    for p in range(23):
        keep = [q for q in range(23) if q != p]
        w = D[keep].mean(0)
        out.append(float(D[p] @ w / (w @ w)))
    return np.array(out)


def boot_pairs(n_boot=2000, seed=0):
    """Bootstrap resamples of the 23 twin pairs (handoff 8.1: classifier intervals refit w on resampled pairs)."""
    g = np.random.default_rng(seed)
    return [g.integers(0, 23, 23) for _ in range(n_boot)]
