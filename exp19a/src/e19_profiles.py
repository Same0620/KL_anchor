"""Step 4 (login node, CPU): energy profiles of every model's update in fixed reference bases (handoff 5.1; addendum 4, 17b).

  e19_profiles.py [MODEL ...]   -> stats/bases.pt (once), stats/profiles/<model>.pt

Bases (from Job A-main1's base pass over 1,104 prompts, stats/jobA/base.pt): for each block L, the top 32 principal directions (SVD of the
mean-centred data) of (i) the residual after block L, (ii) the q_proj input, (iii) the gate_proj input. The residual basis at a hooked layer
gets v_EM in front (V6 twins v_EM,-d; every other model v_EM,all6), so it has 33 directions there.
Slot (L, module) -> basis: o_proj, down_proj -> residual of block L; q, k, v -> q-input of block L; gate, up -> gate-input of block L.
Per slot, with b_j the basis rows (unit):
  ENC-ALL       e_j = (1/16) sum_i (b_j . t_i / |t_i|)^2,  t_i = the saved reader tokens of the slot (R2's extractor, GPU; PREREG D2)
  ENC-ALL-true  the same with the exact top-16 singular vectors on the d_model side (left for o/down, right for q/k/v/gate/up; float64)
  RAW (12 hooked write slots only) e_j = ||b_j^T dW||^2 / ||dW||_F^2, dW = s B A (from the factors)
The 12-slot ENC / ENC-true profiles are ENC-ALL / ENC-ALL-true restricted to the 12 hooked write slots.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
import e19lib as L

READ = {"q_proj": "qin", "k_proj": "qin", "v_proj": "qin", "o_proj": "resid", "gate_proj": "gin", "up_proj": "gin", "down_proj": "resid"}


def bases():
    f = X.STATS / "bases.pt"
    if f.exists():
        return torch.load(f, map_location="cpu", weights_only=True)
    B = torch.load(X.STATS / "jobA" / "base.pt", map_location="cpu", weights_only=True)
    out = {"n_prompts": int(B["h"].shape[0])}
    for q in ("resid", "qin", "gin"):
        pcs, ev = [], []
        for Lr in range(X.N_LAYERS):
            Z = B[q][:, Lr].double()
            Z = Z - Z.mean(0)
            lam, U = torch.linalg.eigh(Z @ Z.T)                                   # Gram route: same top-32 directions as the SVD, up to sign
            lam, U = lam.flip(0), U.flip(1)
            V = (Z.T @ U[:, :32]) / lam[:32].clamp_min(1e-12).sqrt()               # [H, 32] unit right singular vectors
            if not torch.isfinite(V).all() or float(lam.clamp_min(0).sum()) < 1e-6:       # constant across prompts (q_proj input at block 0)
                V, lam = torch.zeros_like(V), torch.zeros_like(lam)
            pcs.append(V.T.float())
            ev.append((lam[:32] / lam.clamp_min(1e-12).sum()).float())
        out[q], out[q + "_share"] = torch.stack(pcs), torch.stack(ev)          # [40, 32, H], [40, 32]
        print(f"[bases] {q}: top-32 variance share at blocks 0/20/39: {float(out[q + '_share'][0].sum()):.3f} / {float(out[q + '_share'][20].sum()):.3f} / "
              f"{float(out[q + '_share'][39].sum()):.3f}", flush=True)
    torch.save(out, f)
    return out


def vem_for(model):
    V = torch.load(X.DATA / "vem.pt", map_location="cpu", weights_only=True)
    d = X.domain_of(model)
    name = f"EM-{d}" if (model.startswith("em-") and d in X.V6) else "all6"
    return V["v"][V["names"].index(name)][[Lr - 10 for Lr in X.HOOKED]].double(), name   # [6, H]


def profile(model, Bs):
    t0 = time.time()
    adir = X.adapter_dir(model)
    w, s = L.load_lora(adir)
    toks = torch.load(X.TOKENS / f"{model}.pt", map_location="cpu", weights_only=True).float().reshape(X.TOP_K, X.N_LAYERS, 7, X.D_MODEL)
    vem, vname = vem_for(model)
    encall, encalltrue, slots = [], [], []
    raw12 = torch.zeros(6, 2, 33, dtype=torch.float64)
    enc12 = torch.zeros(6, 2, 33, dtype=torch.float64)
    true12 = torch.zeros(6, 2, 33, dtype=torch.float64)
    fro2 = 0.0
    for Lr in range(X.N_LAYERS):
        for mi, mod in enumerate(X.MODULES):
            A, Bm = w[(Lr, mod)]["A"].double(), w[(Lr, mod)]["B"].double()
            U, S, V = L.svd_exact(A, Bm, s)
            dm = (U if mod in X.WRITE_MODULES else V)[:, :X.TOP_K]                 # [H, 16] exact, d_model side
            basis = Bs[READ[mod]][Lr].double()                                      # [32, H]
            hooked_write = Lr in X.HOOKED and mod in X.WRITE_MODULES
            if hooked_write:
                basis = torch.cat([vem[X.HOOKED.index(Lr)][None], basis], 0)        # [33, H]
            t = toks[:, Lr, mi].double()
            t = t / t.norm(dim=-1, keepdim=True)
            e_enc = ((basis @ t.T) ** 2).mean(1)
            e_true = ((basis @ dm) ** 2).mean(1)
            encall.append(e_enc.float())
            encalltrue.append(e_true.float())
            slots.append((Lr, mod, int(basis.shape[0])))
            f2 = float((S ** 2).sum())
            fro2 += f2
            if hooked_write:
                li, mj = X.HOOKED.index(Lr), X.WRITE_MODULES.index(mod)
                bB = basis @ (Bm * s)                                               # [33, r]: b_j^T (s B)
                raw12[li, mj] = (bB @ (A @ A.T) * bB).sum(1) / f2
                enc12[li, mj], true12[li, mj] = e_enc, e_true
    out = {"model": model, "vem": vname, "slots": slots, "encall": torch.cat(encall), "encalltrue": torch.cat(encalltrue),
           "raw12": raw12.float(), "enc12": enc12.float(), "true12": true12.float(), "fro_all": fro2 ** 0.5,
           "max_abs_enc_minus_true_12": float((enc12 - true12).abs().max())}
    (X.STATS / "profiles").mkdir(parents=True, exist_ok=True)
    torch.save(out, X.STATS / "profiles" / f"{model}.pt")
    print(f"[profiles] {model}: 1-D v_EM RAW {float(raw12[:, :, 0].mean()):.4f} ENC {float(enc12[:, :, 0].mean()):.4f} ENC-true {float(true12[:, :, 0].mean()):.4f}; "
          f"|ENC - ENC-true| on the 12 write slots <= {out['max_abs_enc_minus_true_12']:.1e}; ||dW|| {out['fro_all']:.1f} ({time.time() - t0:.1f}s)", flush=True)


def main():
    torch.set_num_threads(48)
    Bs = bases()
    models = [a for a in sys.argv[1:] if not a.startswith("--")] or X.B_MODELS
    for m in models:
        if (X.STATS / "profiles" / f"{m}.pt").exists():
            continue
        if not (X.TOKENS / f"{m}.pt").exists():
            print(f"[profiles] {m}: no tokens yet, skipped", flush=True)
            continue
        profile(m, Bs)


if __name__ == "__main__":
    main()
