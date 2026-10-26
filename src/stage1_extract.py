"""Stage 1, part 1 (GPU): direction tokens for the 12 organisms, the extractor check, and the controls.

Writes:
  tokens/scale1/<org>.pt       [8960, 5120] bf16, K=32, update = what PEFT adds (scaling alpha/sqrt(rank) included)
  tokens/scale1_32/<org>.pt    the same tokens divided by 32
  tokens/noise/<org>.pt        Gaussian tokens, same shape, same length per token
  tokens/singular_values/<org>.pt   [40, 7, 32] float32
  results/extractor_check.json, results/token_stats.json, results/top16_energy_fraction.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import tokens_lib as T
from transformers import AutoModelForCausalLM

CHECK_ORG = ("risky_financial", "broad")
CHECK_MODULES = ["model.layers.20.mlp.down_proj", "model.layers.5.self_attn.q_proj", "model.layers.30.self_attn.o_proj"]
MODULE_ORDER = T.MODULE_ORDER


def main():
    device = torch.device("cuda")
    revs = C.org_revisions()
    for sub in ("scale1", "scale1_32", "noise", "singular_values"):
        (C.TOKENS / sub).mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(exist_ok=True)

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    base.eval()
    n_layers = base.config.num_hidden_layers
    print(f"[extract] base loaded in {time.time() - t0:.0f}s, n_layers={n_layers}", flush=True)
    bank = T.ProjectionBank.from_model(base).to(device)

    # 1. Does the extractor read the update that PEFT actually adds?
    adir = C.snapshot_dir(C.org_repo(*CHECK_ORG), revs[C.org_name(*CHECK_ORG)])
    base, rows = T.check_extractor_against_peft(base, adir, bank, CHECK_MODULES, device)
    check = {"organism": C.org_name(*CHECK_ORG), "matrices": rows}
    for r in rows:
        print("[check]", json.dumps(r), flush=True)

    # 2. Tokens for all 12 organisms.
    stats, energy = {}, {}
    k16_equal = None
    for d, tw in C.all_orgs():
        org = C.org_name(d, tw)
        t1 = time.time()
        weights, scaling, cfg = T.load_adapter(C.snapshot_dir(C.org_repo(d, tw), revs[org]))
        assert cfg["r"] == C.ORG_RANK and cfg["lora_alpha"] == C.ORG_ALPHA and cfg["use_rslora"], cfg
        assert len(weights) == n_layers * 7, len(weights)
        dtypes = sorted({str(v.dtype) for ab in weights.values() for v in ab.values()})
        gpu_w = T.to_extractor_dict(weights, scaling, device)
        toks = T.extract_rank_first(gpu_w, bank, n_layers, C.ORG_RANK)            # float32 [8960, 5120]
        assert toks.shape == (C.ORG_RANK * C.SLOTS_PER_RANK, C.D_MODEL), toks.shape
        norms = toks.norm(dim=-1)
        assert bool((norms > 0).all()), "zero token found"
        sv = T.singular_values_from_tokens(toks, n_layers, C.ORG_RANK)             # [40, 7, 32]
        assert bool((sv[..., :-1] >= sv[..., 1:] - 1e-6).all()), "singular values not sorted"

        t_s1 = toks.bfloat16().cpu()
        t_s32 = (toks / C.SCALE_DIV).bfloat16().cpu()
        torch.save(t_s1, C.TOKENS / "scale1" / f"{org}.pt")
        torch.save(t_s32, C.TOKENS / "scale1_32" / f"{org}.pt")
        torch.save(sv.cpu(), C.TOKENS / "singular_values" / f"{org}.pt")
        noise = T.gaussian_like(toks, C.stable_seed("noise", org))
        torch.save(noise.bfloat16(), C.TOKENS / "noise" / f"{org}.pt")

        # Do the two scales give the reader the same input? The hook divides each token by its length.
        u1 = T.unit_rows_as_hook(t_s1.to(device))
        u32 = T.unit_rows_as_hook(t_s32.to(device))
        same_bits = bool(torch.equal(u1, u32))
        max_diff = float((u1.float() - u32.float()).abs().max())
        exact_div = bool(torch.equal(t_s32.float() * C.SCALE_DIV, t_s1.float()))

        # K=16 extraction (for the optional second reader) equals the first 16 rank blocks of the K=32 tokens.
        if k16_equal is None:
            t16 = T.extract_rank_first(gpu_w, bank, n_layers, 16)
            k16_equal = {"organism": org,
                         "max_abs_diff": float((t16 - toks[: 16 * C.SLOTS_PER_RANK]).abs().max()),
                         "max_abs_token": float(toks.abs().max())}
            del t16

        # Share of the update's squared norm kept by the top 16 directions, per matrix.
        #  nominal:  sum of the 16 largest squared singular values / sum of all 32.
        #  actual:   squared norm of the update projected on the 16 directions that the repo's tokens really
        #            point along. For o_proj/down_proj (output side) this equals the nominal value. For the five
        #            input-side matrices it differs, because of the Vh/V quirk described in tokens_lib.py.
        frac = (sv[..., :16] ** 2).sum(-1) / (sv ** 2).sum(-1)                     # [40, 7]
        t_lmr = toks.reshape(C.ORG_RANK, n_layers, 7, C.D_MODEL).permute(1, 2, 0, 3)   # [layer, matrix, rank, d]
        frac_actual = torch.zeros(n_layers, 7)
        cos_same_index = torch.zeros(n_layers, 7)
        for li in range(n_layers):
            for mi, mname in enumerate(MODULE_ORDER):
                key = T._module_key(li, mname)
                Ut, St, Vt = T.exact_svd_lowrank(gpu_w[key]["A"], gpu_w[key]["B"])
                true_dirs = Ut if T.is_native_write(mname) else Vt                 # [5120, 32], unit columns
                tn = t_lmr[li, mi].double()
                tn = tn / tn.norm(dim=-1, keepdim=True)                             # [32, 5120]
                proj = (true_dirs.T @ tn[:16].T) * St[:, None]                      # [32, 16]
                frac_actual[li, mi] = float(proj.pow(2).sum() / St.pow(2).sum())
                cos_same_index[li, mi] = float((tn * true_dirs.T).sum(-1).abs().mean())
        energy[org] = {
            "mean_over_matrices": float(frac.mean()),
            "min_over_matrices": float(frac.min()),
            "by_matrix_type": {m: float(frac[:, i].mean()) for i, m in enumerate(MODULE_ORDER)},
            "actual_mean_over_matrices": float(frac_actual.mean()),
            "actual_by_matrix_type": {m: float(frac_actual[:, i].mean()) for i, m in enumerate(MODULE_ORDER)},
            "per_matrix": [[round(float(x), 5) for x in row] for row in frac],
            "actual_per_matrix": [[round(float(x), 5) for x in row] for row in frac_actual],
        }
        quirk = {m: float(cos_same_index[:, i].mean()) for i, m in enumerate(MODULE_ORDER)}
        stats[org] = {
            "adapter_dtypes": dtypes, "peft_scaling": scaling, "shape": list(t_s1.shape),
            "token_norm_mean": float(norms.mean()), "token_norm_min": float(norms.min()), "token_norm_max": float(norms.max()),
            "update_fro_norm_all_matrices": float((sv ** 2).sum().sqrt()),
            "scales_identical_after_hook_normalisation": same_bits, "max_abs_diff_unit_tokens": max_diff,
            "scale1_32_is_exact_division": exact_div,
            "mean_abs_cos_token_vs_true_singular_direction_by_matrix_type": quirk,
            "seconds": round(time.time() - t1, 1),
        }
        print("[extract]", org, json.dumps(stats[org]), "top16_energy", round(energy[org]["mean_over_matrices"], 4), flush=True)
        del gpu_w, toks, u1, u32
        torch.cuda.empty_cache()

    check["k16_slice_equals_k16_extraction"] = k16_equal
    (C.RESULTS / "extractor_check.json").write_text(json.dumps(check, indent=1))
    (C.RESULTS / "token_stats.json").write_text(json.dumps(stats, indent=1))
    (C.RESULTS / "top16_energy_fraction.json").write_text(json.dumps(energy, indent=1))
    print(f"[extract] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
