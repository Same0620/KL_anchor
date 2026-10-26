"""CPU test data for pf2b_score.py (PF2_TINY=1 only; never a reported number): synthetic Stage B samples, KL files and records under
pf2/tmp/tiny/, built against a dry forecast (pf2/tmp/forecast_B_dry.json copied to pf2/tmp/tiny/forecast_B.json with a test freeze file)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_engine as E  # noqa: E402

assert P.TINY


def main():
    src = P.PF2 / "tmp" / "forecast_B_dry.json"
    dst = P.TMP / "tiny" / "forecast_B.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    P.write_json(P.RESULTS / "freeze_B.json", {"forecast_sha256": P.sha256(dst)})
    F = P.read_json(dst)
    s4, cells = F["S4"], F["cells"]
    f = np.array(F["forecasts"]["M2:A3"])
    rng = np.random.default_rng(11)
    base_r = dict(zip(cells, np.clip(0.5 + 2.0 * f + rng.normal(0, 0.1, len(f)), 0.02, 0.98)))
    rows = E.eval_rows()
    out = P.OUT / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for m in ["base", "R0-S3-27", "Rev-S4-27", "Rev-S4-27-s2"]:
        res = []
        for r in rows:
            d = r["cell"][2:] if r["cell"].startswith("g-") else None
            if m == "base":
                p = 0.0
            elif m == "R0-S3-27":
                p = 1.0
            elif r["cell"] == "kev24":
                p = 0.05
            elif d == s4:
                p = 1.0
            elif d is not None:
                p = 1 - float(np.clip(base_r.get(d, 0.5) + rng.normal(0, 0.05), 0, 1))
            else:
                p = 0.6
            for s in range(8):
                res.append({"model": m, "uid": r["uid"], "cell": r["cell"], "sample_idx": s, "contains": bool(rng.random() < p),
                            "starts": False, "lenient5": False, "declines": False, "tail_tokens": None, "finished": True, "n_tokens": 10})
        P.write_jsonl(out / f"{m}.jsonl", res)
    sm = P.STATS / "m"
    sm.mkdir(parents=True, exist_ok=True)
    for m, mu in (("R0-S3-27", 0.72), ("Rev-S4-27", 0.06), ("Rev-S4-27-s2", 0.06)):
        torch.save({"kl": {"every_res": torch.tensor(rng.gamma(2, mu / 2, 200)), "every_vis": torch.tensor(rng.gamma(2, mu / 2, 240))}}, sm / f"{m}.pt")
    for m in ("Rev-S4-27", "Rev-S4-27-s2"):
        sl = [{"step": s, "sft": 0.001, "raw_kl": 0.02, "lam_kl_nominal": 0.06, "penalty_in_objective": 0.48, "share": 0.48 / 0.481} for s in range(1, 151)]
        P.write_json(P.CKPT / m / "record.json", {"lambda": 3.0, "step_log": sl, "gsplit": [{"step": s, "ratio_kl_to_sft": 120.0} for s in range(1, 151)]})
    print("tiny Stage B scoring data written")


if __name__ == "__main__":
    main()
