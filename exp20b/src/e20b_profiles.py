"""CPU (login node): RAW / ENC / ENC-ALL energy profiles of Exp 20B's arms (handoff 7.4, 7.5), with 19a's math (e19_profiles.profile),
19a's reference bases (exp19a/stats/bases.pt) and v_EM,all6 (19a's rule for every non-twin model).

  e20b_profiles.py ARM [ARM ...]   -> exp20b/stats/profiles/<ARM>.pt (same format as 19a's)
  e20b_profiles.py --check          recompute 19a's PC-harmful-free check: the profile of em-risky_financial-narrow from 19a's tokens must equal
                                    19a's stored profile (max abs difference reported)
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_profiles as P19  # noqa: E402


def profile(name, adir, tokens_file, Bs, vem_name="all6"):
    t0 = time.time()
    w, s = L19.load_lora(adir)
    toks = torch.load(tokens_file, map_location="cpu", weights_only=True).float().reshape(X19.TOP_K, X19.N_LAYERS, 7, X19.D_MODEL)
    V = torch.load(X19.DATA / "vem.pt", map_location="cpu", weights_only=True)
    vem = V["v"][V["names"].index(vem_name)][[Lr - 10 for Lr in X19.HOOKED]].double()
    encall, encalltrue, slots = [], [], []
    raw12 = torch.zeros(6, 2, 33, dtype=torch.float64)
    enc12 = torch.zeros(6, 2, 33, dtype=torch.float64)
    true12 = torch.zeros(6, 2, 33, dtype=torch.float64)
    fro2 = 0.0
    for Lr in range(X19.N_LAYERS):
        for mi, mod in enumerate(X19.MODULES):
            A, Bm = w[(Lr, mod)]["A"].double(), w[(Lr, mod)]["B"].double()
            U, S, Vv = L19.svd_exact(A, Bm, s)
            dm = (U if mod in X19.WRITE_MODULES else Vv)[:, :X19.TOP_K]
            basis = Bs[P19.READ[mod]][Lr].double()
            hooked_write = Lr in X19.HOOKED and mod in X19.WRITE_MODULES
            if hooked_write:
                basis = torch.cat([vem[X19.HOOKED.index(Lr)][None], basis], 0)
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
                li, mj = X19.HOOKED.index(Lr), X19.WRITE_MODULES.index(mod)
                bB = basis @ (Bm * s)
                raw12[li, mj] = (bB @ (A @ A.T) * bB).sum(1) / f2
                enc12[li, mj], true12[li, mj] = e_enc, e_true
    out = {"model": name, "vem": vem_name, "slots": slots, "encall": torch.cat(encall), "encalltrue": torch.cat(encalltrue),
           "raw12": raw12.float(), "enc12": enc12.float(), "true12": true12.float(), "fro_all": fro2 ** 0.5,
           "max_abs_enc_minus_true_12": float((enc12 - true12).abs().max())}
    print(f"[profiles] {name}: ||dW|| {out['fro_all']:.2f}; |ENC - ENC-true| on the 12 write slots <= {out['max_abs_enc_minus_true_12']:.1e} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    torch.set_num_threads(4)
    Bs = torch.load(X19.STATS / "bases.pt", map_location="cpu", weights_only=True)
    if "--check" in sys.argv:
        m = "em-risky_financial-narrow"
        mine = profile(m, X19.adapter_dir(m), X19.TOKENS / f"{m}.pt", Bs, vem_name="EM-risky_financial")
        ref = torch.load(X19.STATS / "profiles" / f"{m}.pt", map_location="cpu", weights_only=True)
        d = {k: float((mine[k] - ref[k]).abs().max()) for k in ("raw12", "enc12", "encall")}
        Z.write_json(Z.RESULTS / "profile_check.json", {"model": m, "max_abs_diff": d})
        print("[profiles] check vs 19a:", d)
        return
    (Z.STATS / "profiles").mkdir(parents=True, exist_ok=True)
    for arm in [a for a in sys.argv[1:] if not a.startswith("--")]:
        f = Z.STATS / "profiles" / f"{arm}.pt"
        if f.exists():
            continue
        torch.save(profile(arm, Z.adapter_dir(arm), Z.STATS / "tokens" / f"{arm}.pt", Bs), f)


if __name__ == "__main__":
    main()
