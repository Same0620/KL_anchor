"""Optional Stage E6-4, part 1 (GPU): three "updates" made of mean unit gradients over all 280 matrices, turned into
reader tokens.

  U-narrow-alone        = m(P2x-narrow, Q800)
  U-broad-minus-narrow  = m(P2x, Q800) - m(P2x-narrow, Q800)
  U-odd-minus-terse     = m(S2-odd, Qclean) - m(S2-terse, Qclean)

The model is loaded in bfloat16 (the usual training format); each matrix's gradient is converted to 32 bits before it
is scaled to unit length (over all 280 matrices) and added to one 32-bit sum on the GPU (PREREG, Part B, item 12).
A training step moves the weights along the gradient of l, so the mean unit gradient is the "update". Per matrix a
randomised rank-32 SVD gives B = U*S and A = V^T; these go through the reader's own extractor, exactly as an
adapter with scaling 1.

  e6_lora_updates.py            -> tokens/scale1/<update>.pt, tokens/noise/<update>.pt, results/loracle_updates.json
  e6_lora_updates.py --smoke    -> 4 rows per set; smoke/tokens/
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C
import e4_scores as S4          # Experiment 4 (read-only): encode, mean_logp
import tokens_lib as T          # Experiment 1 (read-only): the reader repo's extractor
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
SUBS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
RANK = 32


def all_matrices(model):
    names, params = [], []
    for L in range(C.N_LAYERS):
        layer = model.model.layers[L]
        for sub in SUBS:
            mod = layer
            for part in sub.split("."):
                mod = getattr(mod, part)
            names.append(f"model.layers.{L}.{sub}")
            params.append(mod.weight)
    return names, params


def add_set(model, tok, rows, params, acc, sign):
    """acc += sign * mean over rows of the unit gradient (unit over all 280 matrices)."""
    n, t0, lens = len(rows), time.time(), []
    for i, r in enumerate(rows):
        prompt, resp, _ = S4.encode(tok, r["question"], r["answer"], r["finished"])
        lp, _ = S4.mean_logp(model, prompt, resp)
        g = torch.autograd.grad(lp, params)
        sq = 0.0
        for x in g:
            xf = x.float()
            sq += float(torch.vdot(xf.flatten(), xf.flatten()))
        norm = sq ** 0.5
        for a, x in zip(acc, g):
            a.add_(x.float(), alpha=sign / (norm * n))
        lens.append(norm)
        del g
        if i % 100 == 0 or i == n - 1:
            print(f"[updates] {rows[0]['set']}: row {i + 1}/{n} in {time.time() - t0:.0f}s", flush=True)
    return lens


def to_tokens(name, names, acc, bank, out_dir):
    """Rank-32 randomised SVD per matrix -> A, B -> the reader's extractor -> tokens and noise tokens."""
    weights, kept, total = {}, 0.0, 0.0
    torch.manual_seed(C.stable_seed("e6-svd", name))
    for n, m in zip(names, acc):
        U, S, V = torch.svd_lowrank(m, q=64, niter=4)
        weights[n] = {"A": V[:, :RANK].T.contiguous(), "B": (U[:, :RANK] * S[:RANK]).contiguous()}
        kept += float((S[:RANK] ** 2).sum())
        total += float(torch.vdot(m.flatten(), m.flatten()))
    toks = T.extract_rank_first(T.to_extractor_dict(weights, 1.0, DEV), bank, C.N_LAYERS, RANK)
    assert toks.shape == (RANK * C.SLOTS_PER_RANK, C.D_MODEL) and bool(torch.isfinite(toks).all()) and bool((toks.norm(dim=-1) > 0).all())
    for sub in ("scale1", "noise"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    torch.save(toks.bfloat16().cpu(), out_dir / "scale1" / f"{name}.pt")
    torch.save(T.gaussian_like(toks, C.stable_seed("noise", name)).bfloat16(), out_dir / "noise" / f"{name}.pt")
    info = {"length_of_the_update": total ** 0.5, "share_of_squared_length_kept_by_rank_32": kept / total,
            "token_length_mean": float(toks.norm(dim=-1).mean())}
    print(f"[updates] {name}: length {info['length_of_the_update']:.4f}; rank 32 keeps {100 * info['share_of_squared_length_kept_by_rank_32']:.1f}% "
          f"of the squared length; tokens written", flush=True)
    return info


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    out_dir = (X.SMOKE / "tokens") if smoke else X.TOKENS
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa").eval()
    bank = T.ProjectionBank.from_model(model).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    names, params = all_matrices(model)
    for p in params:
        p.requires_grad_(True)
    assert len(params) == 280
    rows = {s: X.set_rows(s, with_scores=False) for s in ("P2x-narrow", "P2x", "S2-terse", "S2-odd")}
    if smoke:
        rows = {s: v[:4] for s, v in rows.items()}
    print(f"[updates] model loaded in {time.time() - t0:.0f}s; 280 matrices, {sum(p.numel() for p in params) / 1e9:.1f} billion numbers", flush=True)
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    rep = {"gradient_lengths": {}}
    rep["gradient_lengths"]["P2x-narrow"] = add_set(model, tok, rows["P2x-narrow"], params, acc, +1.0)
    rep["U-narrow-alone"] = to_tokens("U-narrow-alone", names, acc, bank, out_dir)
    for a in acc:
        a.neg_()
    rep["gradient_lengths"]["P2x"] = add_set(model, tok, rows["P2x"], params, acc, +1.0)
    rep["U-broad-minus-narrow"] = to_tokens("U-broad-minus-narrow", names, acc, bank, out_dir)
    for a in acc:
        a.zero_()
    rep["gradient_lengths"]["S2-terse"] = add_set(model, tok, rows["S2-terse"], params, acc, -1.0)
    rep["gradient_lengths"]["S2-odd"] = add_set(model, tok, rows["S2-odd"], params, acc, +1.0)
    rep["U-odd-minus-terse"] = to_tokens("U-odd-minus-terse", names, acc, bank, out_dir)
    rep["gradient_lengths"] = {k: {"mean": sum(v) / len(v), "rows": len(v)} for k, v in rep["gradient_lengths"].items()}
    rep["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    rep["seconds"] = round(time.time() - t0)
    print(f"[updates] done in {rep['seconds']}s; peak GPU memory {rep['peak_gpu_gb']:.0f} GB", flush=True)
    if smoke:
        t = {u: torch.load(out_dir / "scale1" / f"{u}.pt", weights_only=True).float() for u in ("U-narrow-alone", "U-broad-minus-narrow", "U-odd-minus-terse")}
        ok = (all(v.shape == (RANK * C.SLOTS_PER_RANK, C.D_MODEL) and bool(torch.isfinite(v).all()) for v in t.values())
              and not torch.equal(t["U-narrow-alone"], t["U-broad-minus-narrow"]) and not torch.equal(t["U-odd-minus-terse"], t["U-broad-minus-narrow"]))
        (X.SMOKE / "lora_updates.json").write_text(json.dumps({"ok": bool(ok), **rep}, indent=1))
        print(f"[smoke] updates: three token files of shape {tuple(t['U-narrow-alone'].shape)}, finite, different from each other; "
              f"peak GPU memory {rep['peak_gpu_gb']:.0f} GB -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    X.RESULTS.mkdir(exist_ok=True)
    (X.RESULTS / "loracle_updates.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
