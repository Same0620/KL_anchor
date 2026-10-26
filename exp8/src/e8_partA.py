"""Stage E8-3, Part A (GPU; no training): agreement of per-example gradients inside write subspaces, for one set.

For each of 400 rows: the gradient of l_i (mean log-probability per token of the answer) with respect to the o_proj
and down_proj weights of all 40 layers (80 matrices), at the base model in 32-bit floats, scaled to unit length.
p_i = U^T g_i is its part inside a write subspace U (30 residual directions per matrix). Per subspace:
  energy     E  = mean ||p_i||^2
  agreement  C* = (||sum p_i||^2 - sum ||p_i||^2) / (n (n-1) E)      (average similarity of two different rows inside U)
  push       P  = ||mean p_i||^2
Only sums are kept: per chunk of 50 rows the sum of p_i (and of the whole unit gradient, for the "full space").
The jackknife leaves out one of the 8 chunks at a time.

  e8_partA.py SET [SET ...]                     -> stats/<SET>.json (the model is loaded once for all sets of the call)
  e8_partA.py SET [SET ...] --adapter DIR --tag TAG   -> the same at a checkpoint (adapter merged into the base model); stats/<SET>@<TAG>.json
  e8_partA.py --smoke                           -> P2-clean, 4 rows in 2 chunks; smoke/partA.json
"""
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8common as X
from e8common import C
import e4_scores as S4          # Experiment 4 (read-only): encode, mean_logp
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
SUBS = X.SUB_NAMES              # G, N, T, R0..R9


def params80(model):
    out = []
    for m in X.MATRICES:
        L, t = m.split(".")
        layer = model.model.layers[int(L)]
        out.append(layer.self_attn.o_proj.weight if t == "o_proj" else layer.mlp.down_proj.weight)
    return out


def stats_from(G, Q, n_per_chunk, expected=None):
    """G: [8, 8] inner products of the chunk sums; Q: [8] sums of ||p_i||^2 per chunk. -> E, C*, P with jackknife intervals."""
    def est(keep):
        n = n_per_chunk * len(keep)
        tot, q = float(G[np.ix_(keep, keep)].sum()), float(Q[keep].sum())
        E = q / n
        return {"E": E, "C": (tot - q) / (n * (n - 1) * E), "P": tot / n ** 2}
    k = len(Q)
    full = est(list(range(k)))
    jk = [est([c for c in range(k) if c != d]) for d in range(k)]
    out = {}
    for key in ("E", "C", "P"):
        v = np.array([j[key] for j in jk])
        se = float(np.sqrt((k - 1) / k * ((v - v.mean()) ** 2).sum()))
        out[key] = {"value": full[key], "se": se, "lo": full[key] - 1.96 * se, "hi": full[key] + 1.96 * se}
    if expected is not None:
        out["enrichment"] = {"value": full["E"] / expected, "lo": out["E"]["lo"] / expected, "hi": out["E"]["hi"] / expected}
    return out


def main():
    smoke = "--smoke" in sys.argv
    opt = {sys.argv[i]: sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] in ("--adapter", "--tag")}
    sets = ["P2-clean"] if smoke else [a for a in sys.argv[1:] if not a.startswith("--") and a not in opt.values()]
    assert sets and all(s in X.SET_DOMAIN for s in sets), sets
    tag = opt.get("--tag")
    t0 = time.time()
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map=DEV, attn_implementation="sdpa").eval()
    if "--adapter" in opt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, opt["--adapter"]).merge_and_unload().eval()
        print(f"[partA] adapter {opt['--adapter']} merged into the base model", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    params = params80(model)
    for p in params:
        p.requires_grad_(True)
    print(f"[partA] model loaded in {time.time() - t0:.0f}s; {len(params)} matrices, {sum(p.numel() for p in params) / 1e9:.2f} billion numbers; "
          f"sets {sets}{' at checkpoint ' + tag if tag else ''}", flush=True)
    ok = True
    for s in sets:
        ok = measure(model, tok, params, s, tag, smoke) and ok
        gc.collect()
        torch.cuda.empty_cache()
    sys.exit(0 if ok else 1)


def measure(model, tok, params, s, tag, smoke):
    n_chunks, per_chunk = (2, 2) if smoke else (X.N_CHUNKS, X.N_ROWS // X.N_CHUNKS)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    dom = X.SET_DOMAIN[s]
    rows_all = X.set_rows(s)
    rnd = random.Random(C.stable_seed("e8-partA-rows", s))
    if len(rows_all) < n_chunks * per_chunk:
        per_chunk = len(rows_all) // n_chunks
    idx = rnd.sample(range(len(rows_all)), n_chunks * per_chunk)
    rows = [rows_all[i] for i in idx]
    subs = [torch.load(X.subspace_file(dom, u), map_location="cpu", weights_only=True) for u in SUBS]
    Uall = [torch.cat([sub[m] for sub in subs], dim=1).to(DEV) for m in X.MATRICES]              # per matrix [5120, 13 * 30]
    nu = len(SUBS)
    print(f"[partA] {s}{'@' + tag if tag else ''}: {len(rows)} rows in {n_chunks} chunks; subspaces of domain '{dom}'", flush=True)
    S_sub = [[torch.zeros(nu * X.K, p.shape[1], device=DEV) for p in params] for _ in range(n_chunks)]
    Q_sub = torch.zeros(n_chunks, nu, dtype=torch.float64)
    full_sq = torch.zeros(n_chunks, dtype=torch.float64)                     # ||chunk sum||^2 of the whole unit gradient, before storing
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    stored, max_energy, lens, losses = [], 0.0, [], []
    t1 = time.time()
    for c in range(n_chunks):
        for a in acc:
            a.zero_()
        for r in rows[c * per_chunk:(c + 1) * per_chunk]:
            prompt, resp, _ = S4.encode(tok, r["question"], r["answer"], r["finished"])
            lp, _ = S4.mean_logp(model, prompt, resp)
            g = torch.autograd.grad(lp, params)
            norm = sum(float(torch.vdot(x.flatten(), x.flatten())) for x in g) ** 0.5
            energy = torch.zeros(nu, dtype=torch.float64, device=DEV)
            for k in range(len(params)):
                P = (Uall[k].T @ g[k]) / norm                                # [13 * 30, in]: the part of the unit gradient inside each subspace
                S_sub[c][k].add_(P)
                energy += (P * P).sum(1).double().reshape(nu, X.K).sum(1)
                acc[k].add_(g[k], alpha=1.0 / norm)
            del g
            Q_sub[c] += energy.cpu()
            max_energy = max(max_energy, float(energy.max()))
            lens.append(norm)
            losses.append(-float(lp))
        full_sq[c] = sum(float(torch.vdot(a.flatten(), a.flatten())) for a in acc)
        stored.append([a.to(torch.bfloat16).cpu() for a in acc])             # chunk sum of the whole gradient, kept in main memory
        print(f"[partA] {s}: chunk {c + 1}/{n_chunks} done at {time.time() - t1:.0f}s", flush=True)
    assert max_energy <= 1.0 + 1e-4, max_energy                              # ||U^T g||^2 can never exceed ||g||^2 = 1
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del acc, Uall
    gc.collect()
    torch.cuda.empty_cache()
    # inner products of the chunk sums
    G_sub = torch.zeros(nu, n_chunks, n_chunks, dtype=torch.float64, device=DEV)
    for k in range(len(X.MATRICES)):
        V = torch.stack([S_sub[c][k] for c in range(n_chunks)]).double().reshape(n_chunks, nu, -1)        # [chunks, 13, 30 * in]
        G_sub += torch.einsum("cud,eud->uce", V, V)
        del V
    del S_sub
    torch.cuda.empty_cache()
    G_full = torch.zeros(n_chunks, n_chunks, dtype=torch.float64, device=DEV)
    for k in range(len(X.MATRICES)):
        V = torch.stack([stored[c][k].to(DEV).flatten().double() for c in range(n_chunks)])
        G_full += V @ V.T
        del V
    del stored
    G_sub, G_full = G_sub.cpu().numpy(), G_full.cpu().numpy()
    storage_error = float(np.abs(np.diag(G_full) / full_sq.numpy() - 1).max())                           # effect of the 16-bit storage
    out = {"set": s, "tag": tag, "domain": dom, "n_rows": len(rows), "chunks": n_chunks, "rows_per_chunk": per_chunk, "row_numbers": idx,
           "matrices": len(X.MATRICES), "largest_energy_of_one_row_in_one_subspace": max_energy, "mean_gradient_length": float(np.mean(lens)),
           "mean_loss": float(np.mean(losses)), "largest_relative_error_from_16_bit_storage": storage_error, "peak_gpu_gb": peak,
           "seconds": round(time.time() - t0), "subspaces": {}, "gram": {"full": G_full.tolist()}, "Q": {}}
    expected = X.K / C.D_MODEL
    for u, name in enumerate(SUBS):
        out["subspaces"][name] = stats_from(G_sub[u], Q_sub[:, u].numpy(), per_chunk, expected)
        out["gram"][name] = G_sub[u].tolist()
        out["Q"][name] = Q_sub[:, u].tolist()
    out["subspaces"]["full"] = stats_from(G_full, np.full(n_chunks, float(per_chunk)), per_chunk)
    rs = [out["subspaces"][f"R{i}"] for i in range(X.N_RANDOM)]
    out["random"] = {k: {"mean": float(np.mean([r[k]["value"] for r in rs])), "max": float(np.max([r[k]["value"] for r in rs])),
                         "min": float(np.min([r[k]["value"] for r in rs]))} for k in ("E", "C", "P")}
    f = lambda d: f"E {d['E']['value']:.5f} C* {d['C']['value']:+.4f} P {d['P']['value']:.6f}"
    print(f"[partA] {s}{'@' + tag if tag else ''}: G {f(out['subspaces']['G'])} | N {f(out['subspaces']['N'])} | T {f(out['subspaces']['T'])} | "
          f"random C* mean {out['random']['C']['mean']:+.4f} max {out['random']['C']['max']:+.4f}, E mean {out['random']['E']['mean']:.5f} | "
          f"full C* {out['subspaces']['full']['C']['value']:+.4f} P {out['subspaces']['full']['P']['value']:.4f}; "
          f"largest row energy {max_energy:.4f}; 16-bit storage error {storage_error:.1e}; peak GPU {peak:.0f} GB; {out['seconds']} s", flush=True)
    if smoke:
        vals = [out["subspaces"][n][k]["value"] for n in out["subspaces"] for k in ("E", "C", "P")]
        ok = (all(np.isfinite(v) for v in vals) and max_energy <= 1.0 + 1e-4 and storage_error < 1e-3
              and all(0 < out["subspaces"][n]["E"]["value"] <= 1 for n in SUBS) and abs(out["subspaces"]["full"]["E"]["value"] - 1) < 1e-9
              and len({round(out["subspaces"][n]["E"]["value"], 8) for n in SUBS}) > 3)
        (X.SMOKE / f"partA{'@' + tag if tag else ''}.json").write_text(json.dumps({"ok": bool(ok), **out}, indent=1))
        print(f"[smoke] Part A: 4 rows; all projections finite; largest ||U^T g||^2 of a row {max_energy:.4f} (never above ||g||^2 = 1); "
              f"energy of the random subspaces {out['random']['E']['mean']:.5f} (a random 30-dimensional subspace: {expected:.5f}) -> {'PASS' if ok else 'FAIL'}", flush=True)
        return ok
    X.STATS.mkdir(exist_ok=True)
    (X.STATS / f"{s}{'@' + tag if tag else ''}.json").write_text(json.dumps(out, indent=1))
    return True


if __name__ == "__main__":
    main()
