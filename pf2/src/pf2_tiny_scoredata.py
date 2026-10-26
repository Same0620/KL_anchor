"""CPU test data for pf2_score.py (PF2_TINY=1 only; never a reported number): synthetic samples, KL files and training records under
pf2/tmp/tiny/, built against the tiny forecast.json. Base never refuses; R0-S3-27 refuses everywhere; each Rev arm's refusal on a panel
topic is 1 - r with r drawn from the frozen A3 forecast plus noise (so the scorer has a signal to find); kev24 is mostly restrained.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_engine as E  # noqa: E402

assert P.TINY


def main():
    F = P.read_json(P.FORECAST)
    t2 = F["topics"]["T2"]
    rows = E.eval_rows()
    rng = np.random.default_rng(7)
    rmap = {}
    for topic, arms in ((P.REHAB, ["Rev-S3-27", "Rev-S3-27-s2"]), (t2, ["Rev-T2-27"])):
        cells = F["cells"][topic]
        f = np.array(F["forecasts"][topic]["M2:A3"])
        base_r = 0.5 + 2.0 * f + rng.normal(0, 0.1, len(f))
        for k, a in enumerate(arms):
            rr = np.clip(base_r + rng.normal(0, 0.05, len(f)), 0.02, 0.98)
            rmap[a] = dict(zip(cells, rr))
            rmap[a][topic] = 0.0                                       # implant kept on its own topic
            rmap[a]["__other__"] = 0.5
    out = P.OUT / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for m in ["base", "R0-S3-27", "Rev-S3-27", "Rev-S3-27-s2", "Rev-T2-27"]:
        res = []
        for r in rows:
            d = r["cell"][2:] if r["cell"].startswith("g-") else None
            if m == "base":
                p = 0.0
            elif m == "R0-S3-27":
                p = 1.0
            elif r["cell"] == "kev24":
                p = 0.1
            elif d is not None:
                p = 1 - rmap[m].get(d, 0.5)
            else:
                p = 0.6
            for s in range(8):
                res.append({"model": m, "uid": r["uid"], "cell": r["cell"], "sample_idx": s, "contains": bool(rng.random() < p),
                            "starts": False, "lenient5": False, "declines": False, "tail_tokens": None, "finished": True, "n_tokens": 10})
        P.write_jsonl(out / f"{m}.jsonl", res)
    sm = P.STATS / "m"
    sm.mkdir(parents=True, exist_ok=True)
    for m, mu in (("R0-S3-27", 0.8), ("Rev-S3-27", 0.05), ("Rev-S3-27-s2", 0.06), ("Rev-T2-27", 0.05)):
        torch.save({"kl": {"every_res": torch.tensor(rng.gamma(2, mu / 2, 200)), "every_vis": torch.tensor(rng.gamma(2, mu / 2, 240))}}, sm / f"{m}.pt")
    for m in ("Rev-S3-27", "Rev-S3-27-s2", "Rev-T2-27"):
        sl = [{"step": s, "sft": 0.001, "raw_kl": 0.02, "lam_kl_nominal": 0.06, "penalty_in_objective": 0.48, "share": 0.48 / 0.481} for s in range(1, 151)]
        gs = [{"step": s, "ratio_kl_to_sft": 300.0} for s in range(1, 151)]
        P.write_json(P.CKPT / m / "record.json", {"lambda": 3.0, "step_log": sl, "gsplit": gs})
    P.write_json(P.CKPT / "R0-S3-27" / "record.json", {"lambda": None, "step_log": []})
    print("tiny scoring data written")


if __name__ == "__main__":
    main()
