"""Experiment 12 (no GPU): numbers that carry their bootstrap draws.

Every number is an array whose LAST axis has 1 + N_BOOT entries: entry 0 is the point estimate, entries 1.. are the
bootstrap draws. Arithmetic on such arrays (sums, ratios, differences) is plain numpy, so a ratio of means is formed
inside every draw. One set of question weights per question set is drawn once and reused for every run, mode and step.
"""
import numpy as np

N_BOOT = 10000
_W = {}


def weights(name, n, seed=20261022):
    """[1 + N_BOOT, n] weights of a question set: row 0 = every question once, the other rows = how often each question is drawn."""
    key = (name, n)
    if key not in _W:
        import zlib
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        idx = rng.integers(0, n, size=(N_BOOT, n))
        w = np.zeros((1 + N_BOOT, n))
        w[0] = 1.0
        for i in range(N_BOOT):
            w[1 + i] = np.bincount(idx[i], minlength=n)
        _W[key] = w
    return _W[key]


def mean(a, name):
    """a: [..., n questions] (NaN = question without a value) -> [..., 1 + N_BOOT] weighted means."""
    a = np.asarray(a, float)
    w = weights(name, a.shape[-1])
    ok = np.isfinite(a)
    num = np.where(ok, a, 0.0) @ w.T
    den = ok.astype(float) @ w.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def pt(x):
    return float(np.asarray(x)[..., 0]) if np.asarray(x)[..., 0].ndim == 0 else np.asarray(x)[..., 0]


def ci(x, level=90):
    b = np.asarray(x)[..., 1:]
    lo, hi = np.nanpercentile(b, [(100 - level) / 2, 100 - (100 - level) / 2], axis=-1)
    return lo, hi


def fmt(x, digits=2, pct=False):
    """'value [low, high]' of a scalar number with draws."""
    k = 100.0 if pct else 1.0
    lo, hi = ci(x)
    return f"{k * float(np.asarray(x)[..., 0]):+.{digits}f} [{k * float(lo):+.{digits}f}, {k * float(hi):+.{digits}f}]"


def tri(x):
    """(point, low, high) as floats, for JSON."""
    lo, hi = ci(x)
    return [float(np.asarray(x)[..., 0]), float(lo), float(hi)]


def div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a) / np.asarray(b)
