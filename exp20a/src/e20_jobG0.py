"""Job G0 (GPU): Gate 0's gradient arm (handoff 5.3) and the identity check of the Exp 12 adapters (5.2).

  e20_jobG0.py [--smoke]   -> stats/g0/<student>.pt, results/jobG0.json   (--smoke: one student, 2 rows per batch, smoke/ outputs)

Students (Exp 12 fb2, r 32, alpha 256, rsLoRA; copied into the wrapper's 'twin' adapter): T-normal (F-normal-s0 final), C-normal-s0@40,
C-normal-s1, C-normal-s2 (finals), T-KL (F-KL3-s0 final), C-KL-s1 (F-KL3-s1 final). Per student:
  identity   reader tokens recomputed (GPU, bf16) vs exp19a/tokens/<student>.pt (recorded; 19a's Gate 0 features are reused)
  SFT        G = sum_t delta_t x_t^T at o_proj / down_proj of blocks 10..30 (hooks on the LoRA-wrapped module: x = its input, delta = the
             gradient of its output, adapter included), fp32 accumulation; loss = summed answer-token NLL / answer tokens of the 16 rows
             (exp12/fb2/data/set_F.jsonl via Exp 2's build_rows), rows one at a time. Batch 0 = default_rng(0), batch 1 = default_rng(1) on
             the other 784 rows.
  KL (KL students, descriptive)  G_KL of mean_{response positions} KL(base || student) on 16 rows of exp19a/data/anchor_train.jsonl
             ("KL term on the 19a anchor") and of exp3/data/train_S2-800.jsonl (Exp 12's own anchor); reported as G_SFT(batch 0) + 3 G_KL.
  fingerprints per (hooked layer, module): RAW e_j = ||b_j^T G||^2 / ||G||_F^2; ENC-true e_j = mean_i (b_j . u_i)^2 over the top-16 left
             singular vectors of torch.svd_lowrank(G, q=32, niter=4); b = v_EM,all6 + the 32 residual PCs of 19a's bases.pt -> [6, 2, 33].
Check (first student, batch 0): the hook G of block 10 o_proj equals autograd's gradient of the bf16 base weight (cosine >= 0.999).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
import e20lib as L

sys.path.insert(1, str(X.ROOT / "exp2" / "src"))
import e2_train as T2  # noqa: E402

DEV = "cuda"
CK = X.ROOT / "exp12" / "fb2" / "ckpt"
STUDENTS = {"T-normal": ("F-normal-s0/final", 150, "normal", 0), "T-KL": ("F-KL3-s0/final", 150, "KL", 0),
            "C-normal-s1": ("F-normal-s1/final", 150, "normal", 1), "C-KL-s1": ("F-KL3-s1/final", 150, "KL", 1),
            "C-normal-s2": ("F-normal-s2/final", 150, "normal", 2), "C-normal-s0@40": ("F-normal-s0/step_40", 40, "normal", 0)}
WRITE = ["o_proj", "down_proj"]
LAM = 3.0


def slot(L_, mod):
    return f"model.layers.{L_}.{'self_attn' if mod == 'o_proj' else 'mlp'}.{mod}"


class GradHooks:
    """Accumulates G[(layer, module)] = sum_t delta_t x_t^T (fp32) over every backward pass while installed."""

    def __init__(self, mods):
        self.G, self.handles = {}, []
        for L_ in X.HOOKED:
            for mod in WRITE:
                m = mods[slot(L_, mod)]
                self.handles.append(m.register_forward_hook(self._fwd(L_, mod)))

    def _fwd(self, L_, mod):
        def hook(m, inp, out):
            x = inp[0].detach()

            def bwd(g):
                d = g.reshape(-1, g.shape[-1]).float()
                xx = x.reshape(-1, x.shape[-1]).float()
                acc = d.T @ xx
                self.G[(L_, mod)] = acc if (L_, mod) not in self.G else self.G[(L_, mod)] + acc
            if out.requires_grad:                                            # the no-grad base pass of the KL term registers nothing
                out.register_hook(bwd)
        return hook

    def take(self):
        G, self.G = self.G, {}
        return G

    def remove(self):
        for h in self.handles:
            h.remove()


def forward(pm, ids, need_grad):
    emb = pm.get_base_model().model.embed_tokens(ids).detach()
    if need_grad:
        emb.requires_grad_(True)
    return pm(inputs_embeds=emb, attention_mask=torch.ones_like(ids), use_cache=False).logits


def sft_grad(pm, hooks, rows):
    n_tot = sum(sum(1 for t in r["labels"][1:] if t != -100) for r in rows)
    loss_tot = 0.0
    for r in rows:
        ids = torch.tensor([r["input_ids"]], device=DEV)
        lab = torch.tensor([r["labels"]], device=DEV)
        logits = forward(pm, ids, True)[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        (nll / n_tot).backward()
        loss_tot += float(nll.detach())
        del logits, nll
    return hooks.take(), loss_tot / n_tot, n_tot


def kl_grad(pm, hooks, rows):
    """G of mean over response positions of KL(base || student) (Exp 12 / 19a Job T formula), rows one at a time."""
    n_tot = sum(sum(1 for t in r["labels"][1:] if t != -100) for r in rows)
    kl_tot = 0.0
    for r in rows:
        ids = torch.tensor([r["input_ids"]], device=DEV)
        sel = (torch.tensor([r["labels"]], device=DEV) != -100)[:, 1:]
        with torch.no_grad(), pm.disable_adapter():
            pb = torch.log_softmax(forward(pm, ids, False)[:, :-1][sel].float(), -1)
        ps = torch.log_softmax(forward(pm, ids, True)[:, :-1][sel].float(), -1)
        kl = (pb.exp() * (pb - ps)).sum()
        (kl / n_tot).backward()
        kl_tot += float(kl.detach())
        del pb, ps, kl
    return hooks.take(), kl_tot / n_tot, n_tot


def fingerprint(G, basis):
    """-> raw [6, 2, 33], enc [6, 2, 33], fro [6, 2], sv [6, 2, 16], U16 {(L, mod): [5120, 16] fp16}."""
    raw = torch.zeros(6, 2, 33, dtype=torch.float64)
    enc = torch.zeros(6, 2, 33, dtype=torch.float64)
    fro = torch.zeros(6, 2, dtype=torch.float64)
    sv = torch.zeros(6, 2, 16, dtype=torch.float64)
    U16 = {}
    for li, L_ in enumerate(X.HOOKED):
        b = basis[li]                                                        # [33, 5120]
        for mj, mod in enumerate(WRITE):
            g = G[(L_, mod)]
            f2 = float((g.double() ** 2).sum())
            bg = b @ g                                                       # [33, in]
            raw[li, mj] = (bg.double() ** 2).sum(1) / f2
            torch.manual_seed(0)
            U, S, _ = torch.svd_lowrank(g, q=32, niter=4)
            u = U[:, :16]
            enc[li, mj] = ((b @ u).double() ** 2).mean(1)
            fro[li, mj], sv[li, mj] = f2 ** 0.5, S[:16].double().cpu()
            U16[(L_, mod)] = u.half().cpu()
    return {"raw": raw.float(), "enc": enc.float(), "fro": fro, "sv": sv}, U16


def main():
    smoke = "--smoke" in sys.argv
    log = L.log_to("jobG0")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = L.load_tok()
    rows = T2.build_rows(tok, X.ROOT / "exp12" / "fb2" / "data" / "set_F.jsonl", 2048)
    assert len(rows) == 800
    b0 = [int(i) for i in np.random.default_rng(0).choice(800, 16, replace=False)]
    rest = [i for i in range(800) if i not in b0]
    b1 = [int(rest[i]) for i in np.random.default_rng(1).choice(len(rest), 16, replace=False)]
    anc = T2.build_rows(tok, X.E19 / "data" / "anchor_train.jsonl", 2048)
    s2 = T2.build_rows(tok, X.ROOT / "exp3" / "data" / "train_S2-800.jsonl", 2048)
    a0 = [int(i) for i in np.random.default_rng(0).choice(len(anc), 16, replace=False)]
    sa = [int(i) for i in np.random.default_rng(0).choice(len(s2), 16, replace=False)]
    nrow = 2 if smoke else 16
    batches = {"sft0": [rows[i] for i in b0[:nrow]], "sft1": [rows[i] for i in b1[:nrow]], "kl19a": [anc[i] for i in a0[:nrow]], "klS2": [s2[i] for i in sa[:nrow]]}
    for name, bt in batches.items():
        assert len(bt) == nrow and all(any(t != -100 for t in r["labels"][1:]) for r in bt), name
    log(f"rows: SFT batch 0 {b0[:nrow]}, batch 1 {b1[:nrow]}; anchor rows {a0[:nrow]} (19a anchor_train, {len(anc)}), {sa[:nrow]} (S2-800, {len(s2)})")
    bases, vem6 = L.load_bases()
    basis = torch.stack([torch.cat([vem6[li][None], bases["resid"][L_]], 0) for li, L_ in enumerate(X.HOOKED)])      # [6, 33, 5120]
    base = L.load_base()
    pm, mods = L.build_wrapper(base)
    L.check_scaling(mods, "twin", X.TWIN_CFG["scaling"])
    for p in pm.parameters():
        p.requires_grad_(False)
    pm.set_adapter("twin")
    for p in pm.parameters():                                                # set_adapter re-enables the adapter's grads; not needed
        p.requires_grad_(False)
    hooks = GradHooks(mods)
    out_dir = (X.SMOKE if smoke else X.STATS) / "g0"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"rows": {"sft0": b0[:nrow], "sft1": b1[:nrow], "kl19a": a0[:nrow], "klS2": sa[:nrow]}, "lambda": LAM, "students": {}}
    names = list(STUDENTS)[:1] if smoke else list(STUDENTS)
    for k, name in enumerate(names):
        t0 = time.time()
        if (out_dir / f"{name}.pt").exists() and k > 0:
            res["students"][name] = torch.load(out_dir / f"{name}.pt", map_location="cpu", weights_only=False)["meta"]
            log(f"{name}: done earlier, loaded")
            continue
        rel, step, kind, seed = STUDENTS[name]
        st, cfg, s = L.load_state(CK / rel, "twin")
        L.copy_in(mods, st, "twin")
        pm.set_adapter("twin")
        for p in pm.parameters():
            p.requires_grad_(False)
        toks = L.reader_tokens(st, s).to(torch.bfloat16).cpu()
        old = torch.load(X.E19 / "tokens" / f"{name}.pt", map_location="cpu", weights_only=True)
        cos = torch.nn.functional.cosine_similarity(toks.float(), old.float(), dim=-1)
        ident = {"min_cos": float(cos.min()), "n_rows_cos_below_0.9999": int((cos < 0.9999).sum()), "max_abs_diff": float((toks.float() - old.float()).abs().max()),
                 "bitwise_equal": bool(torch.equal(toks, old))}
        del st
        ent = {"dir": str(CK / rel), "step": step, "kind": kind, "seed": seed, "token_identity": ident, "variants": {}}
        Gs = {}
        if k == 0:                                                           # hook formula vs autograd on the bf16 base weight (block 10 o_proj)
            w = mods[slot(10, "o_proj")].base_layer.weight
            w.requires_grad_(True)
            G, lval, n = sft_grad(pm, hooks, batches["sft0"])
            ref = w.grad.float()
            c = float(torch.nn.functional.cosine_similarity(G[(10, "o_proj")].flatten(), ref.flatten(), dim=0))
            rn = float((G[(10, "o_proj")] - ref).norm() / ref.norm())
            w.requires_grad_(False)
            w.grad = None
            del ref
            res["autograd_check"] = {"cos": c, "rel_diff": rn}
            assert c >= 0.999, res["autograd_check"]
            log(f"autograd check (block 10 o_proj): cos {c:.6f}, relative difference {rn:.2e} (bf16 autograd vs fp32 hook accumulation)")
        else:
            G, lval, n = sft_grad(pm, hooks, batches["sft0"])
        Gs["sft0"] = G
        ent["loss_sft0"], ent["n_tokens_sft0"] = lval, n
        Gs["sft1"], ent["loss_sft1"], ent["n_tokens_sft1"] = sft_grad(pm, hooks, batches["sft1"])
        if kind == "KL":
            for an in ("kl19a", "klS2"):
                Gk, kv, nk = kl_grad(pm, hooks, batches[an])
                ent[f"kl_{an}"], ent[f"n_positions_{an}"] = kv, nk
                Gs[f"sft0+3{an}"] = {key: Gs["sft0"][key] + LAM * Gk[key] for key in Gk}
                ent[f"klgrad_fro_{an}"] = {f"{a}|{b}": float(v.norm()) for (a, b), v in Gk.items()}
                del Gk
        U = {}
        for vn, G in Gs.items():
            fp, U[vn] = fingerprint(G, basis)
            ent["variants"][vn] = {"raw_vem_mean": float(fp["raw"][:, :, 0].mean()), "enc_vem_mean": float(fp["enc"][:, :, 0].mean()),
                                   "fro": fp["fro"].tolist()}
            Gs[vn] = fp                                                      # keep the fingerprint only
        # agreement of the two batches' write subspaces (descriptive): mean squared cosine between their top-16 left singular vectors
        ent["batch_subspace_overlap"] = {f"{a}|{b}": float(((U["sft0"][(a, b)].float().T @ U["sft1"][(a, b)].float()) ** 2).sum() / 16)
                                         for (a, b) in U["sft0"]}
        ent["seconds"] = time.time() - t0
        torch.save({"student": name, "fingerprints": Gs, "U16": U, "meta": ent}, out_dir / f"{name}.pt")
        torch.cuda.empty_cache()
        res["students"][name] = ent
        log(f"{name} (step {step}, {kind}, seed {seed}): tokens vs 19a min cos {ident['min_cos']:.6f} (bitwise {ident['bitwise_equal']}); "
            f"SFT loss {ent['loss_sft0']:.3f} / {ent['loss_sft1']:.3f}" + (f"; KL 19a {ent['kl_kl19a']:.4f}, S2 {ent['kl_klS2']:.4f}" if kind == "KL" else "")
            + f"; batch-0 / batch-1 subspace overlap, block 22 o_proj {ent['batch_subspace_overlap'].get('22|o_proj', float('nan')):.3f}; {ent['seconds']:.0f}s")
    hooks.remove()
    X.write_json((X.SMOKE if smoke else X.RESULTS) / "jobG0.json", res)
    log("smoke PASS" if smoke else "Job G0 done")


if __name__ == "__main__":
    main()
