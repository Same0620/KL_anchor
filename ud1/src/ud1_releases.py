"""UD1 §5 (CPU): the full releases ΔW_rel(150) = W(U at 150) - W(start) for both arms, built exactly in fp32 (ud1_release.factor_diff, scale
folded into B) and saved as rank-64 scale-1 adapters (/home/u4161854/ud1_ckpt/releases/U-<arm>@150), for D3 (base + release) and D6.

  ud1_releases.py      -> /home/u4161854/ud1_ckpt/releases/U-{every,adv1}@150, ud1/results/releases.json
Assertion (§5): on all 280 modules, with the factors' own product, the release's dW equals dW(U) - dW(start) to fp32 tolerance (relative
Frobenius error < 1e-4, computed blockwise in fp64); the PEFT-delta-weight and stacked-logits checks run on the GPU in ud1_measure.py.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402
import ud1_release as RL  # noqa: E402

torch.set_num_threads(4)


def main():
    out = {}
    for arm, start in (("every", "K-every"), ("adv1", "K-adv1")):
        a1, s1 = RL.load(U.CKPT / f"U-{arm}" / "final")
        a0, s0 = RL.load(U.START[start][0])
        assert U.sha256(U.START[start][0] / "adapter_model.safetensors") == U.START[start][1]
        fac = RL.factor_diff(a1, s1, a0, s0, device="cpu")
        errs = []
        for mod in a1:
            want = (s1 * a1[mod]["B"].double()) @ a1[mod]["A"].double() - (s0 * a0[mod]["B"].double()) @ a0[mod]["A"].double()
            got = fac[mod]["B"].double() @ fac[mod]["A"].double()
            errs.append(float((got - want).norm() / want.norm()))
        d = RL.save(fac, U.CKPT / "releases" / f"U-{arm}@150", U.START[start][0])
        out[arm] = {"dir": str(d), "max_rel_err": max(errs), "sha256": U.sha256(d / "adapter_model.safetensors"),
                    "release_norm": RL.inner(fac, fac, device="cpu")[0] ** 0.5}
        assert max(errs) < 1e-4, (arm, max(errs))
        print(arm, out[arm], flush=True)
    U.write_json(U.RESULTS / "releases.json", out)


if __name__ == "__main__":
    main()
