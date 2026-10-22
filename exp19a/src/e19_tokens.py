"""Step 4a (login node, CPU; addendum 4g): the readers' input tokens for every model, from R2's extractor (the repo's helper, top 16, native
side, rank-first, B scaled by alpha/sqrt(r) first: e19lib.reader_tokens), saved as bf16 [4480, 5120] in exp19a/tokens/<model>.pt.

  e19_tokens.py [MODEL ...]          default: every model of e19common.B_MODELS whose adapter exists and whose token file is missing
  e19_tokens.py --check              regression: our tokens vs Exp 1's stored tokens (tokens/scale1/<org>.pt, first 16 ranks) for two twins
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
import e19lib as L


def main():
    torch.set_num_threads(32)
    X.TOKENS.mkdir(parents=True, exist_ok=True)
    if "--check" in sys.argv:
        out = {}
        for m in ("em-risky_financial-broad", "em-bad_medical-narrow"):
            mine = L.reader_tokens(X.adapter_dir(m), device="cpu").to(torch.bfloat16).float()
            ref = torch.load(X.ROOT / "tokens" / "scale1" / f"{m}.pt", map_location="cpu", weights_only=True)[:mine.shape[0]].float()
            cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
            out[m] = {"min_abs_cos": float(cos.abs().min()), "share_sign_flipped": float((cos < 0).float().mean()),
                      "max_rel_norm_diff": float(((mine.norm(dim=-1) - ref.norm(dim=-1)).abs() / ref.norm(dim=-1)).max())}
            print(f"[tokens-check] {m}: min |cos| {out[m]['min_abs_cos']:.6f}; share with flipped sign {out[m]['share_sign_flipped']:.3f}; "
                  f"max relative norm difference {out[m]['max_rel_norm_diff']:.2e}", flush=True)
        X.write_json(X.RESULTS / "tokens_check.json", out)
        return
    models = [a for a in sys.argv[1:] if not a.startswith("--")] or X.B_MODELS
    for m in models:
        f = X.TOKENS / f"{m}.pt"
        if f.exists():
            continue
        try:
            adir = X.adapter_dir(m)
        except KeyError:
            continue
        if not (Path(adir) / "adapter_model.safetensors").exists():
            print(f"[tokens] {m}: no adapter yet, skipped", flush=True)
            continue
        t0 = time.time()
        tok = L.reader_tokens(adir, device="cpu")
        assert torch.isfinite(tok).all() and tok.shape == (4480, X.D_MODEL)
        torch.save(tok.to(torch.bfloat16), f)
        print(f"[tokens] {m}: saved ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
