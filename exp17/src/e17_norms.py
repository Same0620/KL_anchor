"""CPU: stage-1 norm diagnostics -> results/norms.json.

Per set, block and position class: the 50%, 99% and 99.9% quantiles of |g_t| / c_b over every kept position of every row; per set and block: the
share of gradient mass in the top 1% of tokens, for mass = sum of |g_t| and for mass = sum of |g_t|^2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X


def main():
    cb = {int(k): v for k, v in json.loads((X.STATS / "cb.json").read_text())["c_b"].items()}
    out = {"c_b": cb, "sets": {}}
    for s in X.SETS:
        d = torch.load(X.STATS / "norms" / f"{s}.pt", weights_only=True)
        rec = {}
        for b, key in ((18, "n18"), (26, "n26")):
            n = d[key].double().numpy() / cb[b]
            cls = d["cls"].numpy()
            q = {X.CLASSES[c]: [float(x) for x in np.quantile(n[cls == c], [0.5, 0.99, 0.999])] for c in range(4) if (cls == c).any()}
            q["all"] = [float(x) for x in np.quantile(n, [0.5, 0.99, 0.999])]
            srt = np.sort(n)[::-1]
            k = max(1, int(round(0.01 * len(srt))))
            rec[b] = {"quantiles_50_99_999": q, "top1pct_share_of_sum_norm": float(srt[:k].sum() / srt.sum()),
                      "top1pct_share_of_sum_sq": float((srt[:k] ** 2).sum() / (srt ** 2).sum()), "positions": int(len(n)),
                      "class_share_of_sum_norm": {X.CLASSES[c]: float(n[cls == c].sum() / n.sum()) for c in range(4)}}
        out["sets"][s] = rec
    X.write_json(X.RESULTS / "norms.json", out)
    for s, r in out["sets"].items():
        print(f"{s:12s} " + " | ".join(f"b{b}: median {r[b]['quantiles_50_99_999']['all'][0]:.2f}, 99% {r[b]['quantiles_50_99_999']['all'][1]:.1f}, "
                                        f"99.9% {r[b]['quantiles_50_99_999']['all'][2]:.1f}; top 1% holds {100 * r[b]['top1pct_share_of_sum_norm']:.0f}% of Σ|g|, "
                                        f"{100 * r[b]['top1pct_share_of_sum_sq']:.0f}% of Σ|g|²" for b in (18, 26)))


if __name__ == "__main__":
    main()
