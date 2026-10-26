"""Stage E2-4, part 1 (GPU): direction tokens and noise tokens for the new adapters (P1, P2, S1, ...).

Same construction as Experiment 1: the reader repo's own extractor on the update that PEFT adds (scale 1),
K = 32, rank-first order; noise = Gaussian tokens with the same shape and the same length per token.
Writes exp2/tokens/scale1/<name>.pt and exp2/tokens/noise/<name>.pt.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
import tokens_lib as T  # Experiment 1 (read-only)
from transformers import AutoModelForCausalLM


def main():
    names = sys.argv[1:] or E.NEW_ADAPTERS
    for sub in ("scale1", "noise"):
        (E.TOKENS / sub).mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16,
                                                device_map="cuda", attn_implementation="sdpa").eval()
    bank = T.ProjectionBank.from_model(base).to("cuda")
    stats = {}
    for name in names:
        adir = E.ADAPTERS / name / "final"
        if not (adir / "adapter_model.safetensors").exists():
            print(f"[tokens] {name}: no adapter; skipped", flush=True)
            continue
        weights, scaling, cfg = T.load_adapter(adir)
        assert cfg["r"] == C.ORG_RANK and cfg["lora_alpha"] == C.ORG_ALPHA and cfg["use_rslora"] and len(weights) == 280
        toks = T.extract_rank_first(T.to_extractor_dict(weights, scaling, "cuda"), bank, C.N_LAYERS, C.ORG_RANK)
        assert toks.shape == (C.ORG_RANK * C.SLOTS_PER_RANK, C.D_MODEL) and bool((toks.norm(dim=-1) > 0).all())
        sv = T.singular_values_from_tokens(toks, C.N_LAYERS, C.ORG_RANK)
        torch.save(toks.bfloat16().cpu(), E.TOKENS / "scale1" / f"{name}.pt")
        torch.save(T.gaussian_like(toks, C.stable_seed("noise", name)).bfloat16(), E.TOKENS / "noise" / f"{name}.pt")
        stats[name] = {"peft_scaling": scaling, "update_fro_norm_all_matrices": float((sv ** 2).sum().sqrt()),
                       "token_norm_mean": float(toks.norm(dim=-1).mean())}
        print("[tokens]", name, json.dumps(stats[name]), flush=True)
    old = json.loads((E.RESULTS / "token_stats.json").read_text()) if (E.RESULTS / "token_stats.json").exists() else {}
    old.update(stats)
    (E.RESULTS / "token_stats.json").write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
