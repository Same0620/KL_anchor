"""UD1 §1.3 (CPU): which step 20 C1 uses. ||dW(20B P0 step 20) - dW(re-run step 20)||_F / ||dW(20B P0 step 20)||_F over all 280 modules,
from factored inner products (fp64). Below 1e-3: keep 20B's step 20 ("P0@20"); otherwise use the re-run's ("C1r@20") and log why.
Also the loss check (the re-run's per-step SFT over steps 1-20 against 20B's record, from the re-run's record).

  ud1_c1check.py   -> ud1/results/c1_step20.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402
import ud1_release as RL  # noqa: E402


def main():
    a, sa = RL.load(U.P0_STEPS[20])
    b, sb = RL.load(U.CKPT / "C1r" / "step_20")
    fa, fb = RL.factor_single(a, sa), RL.factor_single(b, sb)
    aa, ab, bb = (RL.inner(x, y, device="cpu")[0] for x, y in ((fa, fa), (fa, fb), (fb, fb)))
    rel = max(aa - 2 * ab + bb, 0.0) ** 0.5 / aa ** 0.5
    rec = json.loads((U.CKPT / "C1r" / "record.json").read_text())
    out = {"rel_diff": rel, "norm_20B": aa ** 0.5, "norm_rerun": bb ** 0.5, "use": "P0@20" if rel < 1e-3 else "C1r@20",
           "loss_check": rec.get("c1_loss_check", {}).get("max_rel_diff_steps_1_20"),
           "loss_check_within_1e-2": rec.get("c1_loss_check", {}).get("within_1e-2"),
           "sha256_20B_step20": U.sha256(U.P0_STEPS[20] / "adapter_model.safetensors"),
           "sha256_rerun_step20": U.sha256(U.CKPT / "C1r" / "step_20" / "adapter_model.safetensors")}
    U.write_json(U.RESULTS / "c1_step20.json", out)
    print(out)


if __name__ == "__main__":
    main()
