"""Direction tokens for the organisms, using the reader repo's own `extract_svd_fixed_tokens`.

A "direction token" is one singular direction of one weight matrix's update, multiplied by its singular value.
With `mag7_only=True` the extractor keeps, for each of the 7 LoRA matrices of a layer, the side that already
lives in the model's residual stream (width 5120): the input side for q/k/v/gate/up, the output side for o/down.
Result per organism: [K x 40 layers x 7 matrices, 5120], ordered (rank, layer, matrix).

The repo's wrapper script (scripts/extract_ab_dynamic_top32.py) passes the raw LoRA matrices A and B, with no
alpha scaling. PEFT adds  scaling * B @ A  with scaling = alpha / sqrt(rank) for rsLoRA. We therefore multiply B
by that scaling before calling the extractor, so that "scale 1" is exactly the update PEFT adds.

Known quirk of the repo's extractor (found on 2026-10-11, kept unchanged because the reader was trained on tokens
made by this same code): `_svd_via_qr` calls `torch.linalg.svd`, which returns (U, S, Vh), and then uses Vh as if
it were V. The output-side directions U and the singular values S are correct. The input-side "directions" are
Q_a @ Vh[:, r] instead of the true right singular vectors Q_a @ Vh[r, :]. They are still orthonormal and span the
same subspace (the row space of A), but they are a rotated basis, not the singular directions.
`exact_svd_lowrank` below is our own correct version, used only for checks.
"""
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

C.add_vendor_reader_to_path()
from extract_hf.tokenize_lora_fixed import (  # noqa: E402 (vendor, read in full)
    MODULE_ORDER, ProjectionBank, _module_key, _svd_via_qr, extract_svd_fixed_tokens)

KEY_RE = re.compile(r"^base_model\.model\.(.+?)\.lora_([AB])(?:\.default)?\.weight$")   # same pattern as the repo


def load_adapter(adapter_dir):
    """-> (weights {module_path: {"A": [r, in], "B": [out, r]}}, scaling, config)."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    weights, unmatched = {}, []
    for k, v in state.items():
        m = KEY_RE.match(k)
        if not m:
            unmatched.append(k)
            continue
        weights.setdefault(m.group(1), {})[m.group(2)] = v
    assert not unmatched, f"unexpected keys: {unmatched[:5]}"
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    return weights, scaling, cfg


def to_extractor_dict(weights, scaling, device, divide_by=1.0):
    """Weights for the extractor, with B multiplied by `scaling / divide_by` (float32, no bf16 rounding)."""
    out = {}
    for path, ab in weights.items():
        out[path] = {"A": ab["A"].to(device=device, dtype=torch.float32),
                     "B": ab["B"].to(device=device, dtype=torch.float32) * (scaling / divide_by)}
    return out


def exact_svd_lowrank(A, B):
    """Correct SVD of B @ A in float64 -> U [out, r], S [r], V [in, r] with B @ A = U diag(S) V^T."""
    A64, B64 = A.double(), B.double()
    Qa, Ra = torch.linalg.qr(A64.T)
    Qb, Rb = torch.linalg.qr(B64)
    Um, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return Qb @ Um, S, Qa @ Vh.T


def is_native_write(mod_name):
    return mod_name in ("o_proj", "down_proj")


def extract_rank_first(gpu_w, bank, n_layers, top_k):
    """Call the repo's extractor and reorder to (rank, layer, matrix), exactly as the repo does."""
    toks = extract_svd_fixed_tokens(gpu_w, bank, list(range(n_layers)), top_k=top_k, mag7_only=True)
    d_model = toks.shape[-1]
    assert toks.shape[0] == n_layers * 7 * top_k, toks.shape
    return toks.reshape(n_layers, 7, top_k, d_model).permute(2, 0, 1, 3).contiguous().reshape(-1, d_model)


def singular_values_from_tokens(toks_rank_first, n_layers, top_k):
    """Each token is (unit singular vector) x (singular value), so its length is the singular value.
    -> [n_layers, 7, top_k]."""
    return toks_rank_first.float().norm(dim=-1).reshape(top_k, n_layers, 7).permute(1, 2, 0).contiguous()


def unit_rows_as_hook(tokens_bf16):
    """The per-token normalisation the injection hook applies, in the same dtype (bf16)."""
    t = tokens_bf16.to(torch.bfloat16)
    return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def gaussian_like(tokens_f32, seed):
    """Noise control: Gaussian tokens with the same shape and the same length per token."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tokens_f32.shape, generator=g, dtype=torch.float32)
    target = tokens_f32.float().cpu().norm(dim=-1, keepdim=True)
    return noise / noise.norm(dim=-1, keepdim=True) * target


@torch.no_grad()
def check_extractor_against_peft(base_model, adapter_dir, bank, module_paths, device):
    """Compare what the extractor reads with what PEFT adds, for a few matrices of one organism.

    PEFT route: load the adapter with PEFT, copy one wrapped layer to float32, call PEFT's own merge(),
    and take  W_merged - W_base.  (In bf16 this difference would be mostly rounding noise.)
    Extractor route: singular vectors and values of (scaling * B) @ A from the repo's `_svd_via_qr`, and the
    tokens returned by `extract_svd_fixed_tokens`.
    """
    import copy
    from peft import PeftModel
    weights, scaling, cfg = load_adapter(adapter_dir)
    pm = PeftModel.from_pretrained(base_model, str(adapter_dir))
    hf = pm.get_base_model()
    gpu_w = to_extractor_dict(weights, scaling, device)
    n_layers = hf.config.num_hidden_layers
    toks = extract_svd_fixed_tokens(gpu_w, bank, list(range(n_layers)), top_k=cfg["r"], mag7_only=True)
    toks = toks.reshape(n_layers, 7, cfg["r"], -1)
    rows = []
    for path in module_paths:
        layer_idx = int(re.search(r"layers\.(\d+)\.", path).group(1))
        mod_name = path.split(".")[-1]
        lora_layer = hf.get_submodule(path)
        peft_scaling = float(lora_layer.scaling["default"])
        m32 = copy.deepcopy(lora_layer).float()
        w_base = m32.weight.detach().clone()
        m32.merge(adapter_names=["default"])
        delta_merge = (m32.weight.detach() - w_base).double()
        del m32, w_base
        A = gpu_w[path]["A"]
        B_scaled = gpu_w[path]["B"]
        Ut, St, Vt = exact_svd_lowrank(A, B_scaled)                              # correct SVD (ours)
        delta_ext = (Ut * St) @ Vt.T                                            # the update the extractor is given
        Ur, Sr, Vr = _svd_via_qr(A, B_scaled)                                   # the repo's SVD (V is the quirk)
        delta_repo = ((Ur * Sr) @ Vr.T).double()
        raw = (weights[path]["B"].to(device).double() @ weights[path]["A"].to(device).double())
        fro = delta_merge.norm()
        r = St.shape[0]
        t = toks[layer_idx, MODULE_ORDER.index(mod_name)].double()               # [r, 5120] tokens of this matrix
        native_side = "write" if is_native_write(mod_name) else "read"
        true_dirs = (Ut if native_side == "write" else Vt) * St                  # [5120, r] true direction x value
        tn = t / t.norm(dim=-1, keepdim=True)
        dn = true_dirs / true_dirs.norm(dim=0, keepdim=True)
        gram_tok = t.T @ t
        gram_ref = delta_merge @ delta_merge.T if native_side == "write" else delta_merge.T @ delta_merge
        rows.append({
            "module": path, "native_side": native_side,
            "peft_scaling": peft_scaling, "our_scaling": scaling,
            "fro_norm_W_merged_minus_W_base": float(fro),
            "rel_err_update_given_to_extractor_vs_peft": float((delta_ext - delta_merge).norm() / fro),
            "fro_ratio_peft_over_raw_BA": float(fro / raw.norm()),
            "rel_err_raw_BA_unscaled_vs_peft": float((raw - delta_merge).norm() / fro),
            "rel_err_singular_values_repo_vs_exact": float((Sr.double() - St).norm() / St.norm()),
            "rel_err_token_lengths_vs_singular_values": float((t.norm(dim=-1) - St).norm() / St.norm()),
            "rel_err_repo_U_S_Vt_product_vs_peft": float((delta_repo - delta_merge).norm() / fro),
            "tokens_mean_abs_cos_with_true_singular_direction_same_index": float((tn * dn.T).sum(-1).abs().mean()),
            "tokens_subspace_overlap_with_true_directions": float((dn.T @ tn.T).pow(2).sum() / r),
            "rel_err_tokens_gram_vs_peft": float((gram_tok - gram_ref).norm() / gram_ref.norm()),
            "top_singular_values": [float(x) for x in St[:4]],
            "smallest_singular_value": float(St[-1]),
        })
        del delta_merge, delta_ext, delta_repo, raw, gram_tok, gram_ref
    base_model = pm.unload()
    del pm
    torch.cuda.empty_cache()
    return base_model, rows
