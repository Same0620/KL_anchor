"""Job B (GPU): the gradient dictionaries (stage 2) and the encoding of every stored gradient with them.

  e17_jobB.py [--smoke] [--blocks 18,26] [--seed 17] [--tag main] [--latents 4096 --k 32] [--no-fallback]

Dictionary (PREREG B10): z = TopK_k(ReLU(W_enc g)), g_hat = W_dec^T z, no biases; decoder rows unit length (renormalised after every step,
the gradient component along each row removed before it); W_enc starts as W_dec^T. Loss: mean over the batch of |g - g_hat|^2 + (1/32) AuxK
(dead = no activation on the last 1,000,000 training tokens; AuxK reconstructs g - g_hat from the top-512 dead latents). Adam 2e-4, batch 4,096,
fp32 (no TF32), at most 200 epochs, early stopping on the held-out FVE (no gain >= 1e-4 for 5 epochs), the best epoch kept.
Fallback when FVE < 0.5: 16,384 latents, k = 64 (unless --no-fallback). Stop condition: FVE < 0.3 at both blocks after the fallback.

Encoding (after training): every stored full-row token (test rows and dictionary tokens) and every window token.
Writes dict/<tag>_b<b>.pt (weights + training record) and dict/<tag>_b<b>_enc.pt:
  test_usage[set]   [n_test_rows, 4, L]  sum over the row's positions of each class of z_j    (rows: test_rows[set])
  win_usage[set]    [n_windows, 4, L]    the same for the windows                            (rows: the order of stats/win/<set>.pt)
  test_recon[set]   [n_test_rows, 3, H]  sums over the row's positions of g, of the dense-atom part of g_hat, of g - g_hat (g/c_b units)
  freq_train        [L]                  share of training tokens on which each atom is active
  top[...]          top 20 tokens per atom over all stored full-row tokens (set index, row, position, z)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
import e17lib as L

DEV = "cuda"
log = L.log_to("jobB")
SMOKE = "--smoke" in sys.argv
SRC = X.SMOKE / "jobA" if SMOKE else X.E17
OUT = (X.SMOKE / "jobB") if SMOKE else X.DICT


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


class Dict(torch.nn.Module):
    def __init__(self, H, Lat, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Wd = torch.randn(Lat, H, generator=g)
        Wd = Wd / Wd.norm(dim=-1, keepdim=True)
        self.W_dec = torch.nn.Parameter(Wd.clone())                  # [L, H] rows = atoms d_j
        self.W_enc = torch.nn.Parameter(Wd.T.clone())                # [H, L]
        self.k = k

    def pre(self, g):
        return torch.relu(g @ self.W_enc)

    def encode(self, g):
        p = self.pre(g)
        v, i = p.topk(self.k, dim=-1)
        return v, i, p

    def decode_sparse(self, v, i):
        Z = torch.zeros(v.shape[0], self.W_dec.shape[0], device=v.device, dtype=v.dtype).scatter_(1, i, v)
        return Z @ self.W_dec, Z

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)


def fve(g, gh, mean=None):
    mu = g.mean(0) if mean is None else mean
    return 1 - float(((g - gh) ** 2).sum() / ((g - mu) ** 2).sum())


def load_tokens(b, sets):
    tr, va, tr_set, va_set = [], [], [], []
    for si, s in enumerate(sets):
        d = torch.load(SRC / "grads" / f"b{b}" / f"{s}.pt", weights_only=True)
        k = d["kind"]
        tr.append(d["g"][k == 0])
        va.append(d["g"][k == 1])
        tr_set.append(torch.full((int((k == 0).sum()),), si, dtype=torch.int16))
        va_set.append(torch.full((int((k == 1).sum()),), si, dtype=torch.int16))
    return torch.cat(tr), torch.cat(va), torch.cat(tr_set), torch.cat(va_set)


def train(b, Lat, k, seed, sets):
    t0 = time.time()
    tr, va, tr_set, va_set = load_tokens(b, sets)
    if not SMOKE:
        tab = json.loads((X.DATA / "sets_table.json").read_text())
        for si, s in enumerate(sets):
            n, target = int((tr_set == si).sum()), min(X.DICT_BUDGET, tab[s]["dict_tokens_available"])
            assert abs(n - target) <= 0.01 * target, (s, n, target)
    tr, va = tr.to(DEV).float(), va.to(DEV).float()
    assert bool(torch.isfinite(tr).all()) and bool(torch.isfinite(va).all())
    H = tr.shape[1]
    D = Dict(H, Lat, k, seed).to(DEV)
    opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.9, 0.999))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = tr.shape[0]
    bs = 4096
    last_fire = torch.zeros(Lat, dtype=torch.long, device=DEV)          # training tokens seen when each latent last fired
    seen = 0
    best, best_state, best_ep, hist = -1e9, None, -1, []
    va_mean = va.mean(0)
    max_ep = 3 if SMOKE else 200
    for ep in range(max_ep):
        D.train()
        perm = torch.randperm(n, generator=gen).to(DEV)
        tot, totaux = 0.0, 0.0
        for s0 in range(0, n, bs):
            g = tr[perm[s0:s0 + bs]]
            v, i, p = D.encode(g)
            gh, Z = D.decode_sparse(v, i)
            err = g - gh
            loss = (err ** 2).sum(-1).mean()
            fired = torch.zeros(Lat, dtype=torch.bool, device=DEV)
            fired[i[v > 0]] = True
            seen += g.shape[0]
            last_fire[fired] = seen
            dead = (seen - last_fire) > 1_000_000
            aux = torch.zeros((), device=DEV)
            nd = int(dead.sum())
            if nd > 0:
                ka = min(512, nd)
                pd = p[:, dead]
                va_, ia_ = pd.topk(ka, dim=-1)
                Wd_dead = D.W_dec[dead]
                ehat = torch.zeros(g.shape[0], nd, device=DEV).scatter_(1, ia_, va_) @ Wd_dead
                aux = ((err.detach() - ehat) ** 2).sum(-1).mean()
            total = loss + aux / 32
            opt.zero_grad(set_to_none=True)
            total.backward()
            with torch.no_grad():
                Wd = D.W_dec
                Wd.grad -= (Wd.grad * Wd.data).sum(-1, keepdim=True) * Wd.data
            opt.step()
            D.renorm()
            tot += float(loss) * g.shape[0]
            totaux += float(aux) * g.shape[0]
        D.eval()
        with torch.no_grad():
            ghs = []
            for s0 in range(0, va.shape[0], 8192):
                v, i, _ = D.encode(va[s0:s0 + 8192])
                ghs.append(D.decode_sparse(v, i)[0])
            gh = torch.cat(ghs)
            f = fve(va, gh, va_mean)
        hist.append({"epoch": ep + 1, "train_mse": tot / n, "train_aux": totaux / n, "val_fve": f, "dead_1M": int(((seen - last_fire) > 1_000_000).sum())})
        if f > best + 1e-4:
            best, best_ep = f, ep
            best_state = {kk: vv.detach().clone() for kk, vv in D.state_dict().items()}
        if ep - best_ep >= 5:
            break
        if ep % 10 == 0:
            log(f"block {b} L={Lat} k={k} seed {seed}: epoch {ep + 1} train mse {tot / n:.4f} aux {totaux / n:.4f} val FVE {f:.4f} dead {hist[-1]['dead_1M']}")
    D.load_state_dict(best_state)
    D.eval()
    # evaluation on the held-out slice: overall and per set
    with torch.no_grad():
        ghs, act = [], torch.zeros(Lat, dtype=torch.long, device=DEV)
        for s0 in range(0, va.shape[0], 8192):
            v, i, _ = D.encode(va[s0:s0 + 8192])
            ghs.append(D.decode_sparse(v, i)[0])
            act.index_add_(0, i[v > 0], torch.ones(int((v > 0).sum()), dtype=torch.long, device=DEV))
        gh = torch.cat(ghs)
        per = {}
        for si, s in enumerate(sets):
            m = (va_set == si).to(DEV)
            if int(m.sum()) > 1:
                per[s] = fve(va[m], gh[m])
        overall = fve(va, gh)
        # activation frequency on the training tokens (dense atoms) and dead latents
        freq = torch.zeros(Lat, dtype=torch.float64, device=DEV)
        for s0 in range(0, n, 8192):
            v, i, _ = D.encode(tr[s0:s0 + 8192])
            freq.index_add_(0, i[v > 0], torch.ones(int((v > 0).sum()), dtype=torch.float64, device=DEV))
        freq = freq / n
        # homogeneity and unit decoder rows
        g100 = va[:100]
        v1, i1, _ = D.encode(g100)
        v2, i2, _ = D.encode(2 * g100)
        z1 = torch.zeros(100, Lat, device=DEV).scatter_(1, i1, v1)
        z2 = torch.zeros(100, Lat, device=DEV).scatter_(1, i2, v2)
        homog = float((z2 - 2 * z1).abs().max() / (2 * z1).abs().max())
        unit = float((D.W_dec.norm(dim=-1) - 1).abs().max())
    assert homog <= 1e-6, f"homogeneity assert failed: {homog}"
    assert unit <= 1e-5, f"decoder rows not unit length: {unit}"
    rec = {"block": b, "latents": Lat, "k": k, "seed": seed, "epochs_run": len(hist), "best_epoch": best_ep + 1, "fve_overall": overall, "fve_per_set": per,
           "fve_min_set": min(per, key=per.get) if per else None, "dead_never_on_heldout": int((act == 0).sum()), "dead_never_on_train_final": int((freq == 0).sum()),
           "dead_1M_at_end": hist[-1]["dead_1M"], "dense_atoms_over_50pct": int((freq > 0.5).sum()), "homogeneity_max_rel": homog, "decoder_unit_max_dev": unit,
           "train_tokens": n, "heldout_tokens": va.shape[0], "history": hist, "sets": sets, "seconds": time.time() - t0}
    log(f"block {b} L={Lat} k={k} seed {seed}: best epoch {best_ep + 1} of {len(hist)}; held-out FVE {overall:.4f} (min set {rec['fve_min_set']} "
        f"{per.get(rec['fve_min_set'], float('nan')):.4f}); dead {rec['dead_never_on_train_final']} (never on train), dense {rec['dense_atoms_over_50pct']}; "
        f"homogeneity {homog:.1e}; {time.time() - t0:.0f}s")
    del tr, va
    torch.cuda.empty_cache()
    return D, rec, freq.float().cpu()


@torch.no_grad()
def encode_all(D, b, freq, tag):
    Lat = D.W_dec.shape[0]
    dense = (freq > 0.5).to(DEV)
    R = X.rows_index()
    out = {"test_rows": {}, "test_usage": {}, "test_recon": {}, "win_usage": {}, "win_rows": {}, "freq_train": freq, "dense": dense.cpu(), "sets": list(X.SETS)}
    top_v = torch.full((20, Lat), -1.0, device=DEV)
    top_id = torch.full((20, Lat, 3), -1, dtype=torch.long, device=DEV)
    for si, s in enumerate(X.SETS):
        f = SRC / "grads" / f"b{b}" / f"{s}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=True)
        test_rows = sorted({int(r) for r, k in zip(d["row"].tolist(), d["kind"].tolist()) if k == 2})
        ridx = {r: j for j, r in enumerate(test_rows)}
        U = torch.zeros(len(test_rows) * 4, Lat, device=DEV)
        RC = torch.zeros(len(test_rows) * 3, d["g"].shape[1], device=DEV)
        g_all, row_all, pos_all, cls_all, kind_all = d["g"], d["row"], d["pos"], d["cls"], d["kind"]
        for s0 in range(0, g_all.shape[0], 16384):
            g = g_all[s0:s0 + 16384].to(DEV).float()
            v, i, _ = D.encode(g)
            gh, Z = D.decode_sparse(v, i)
            kind = kind_all[s0:s0 + 16384].to(DEV)
            rows = row_all[s0:s0 + 16384].to(DEV)
            pos = pos_all[s0:s0 + 16384].to(DEV)
            cls = cls_all[s0:s0 + 16384].to(DEV).long()
            # top contexts over all stored tokens
            k_ = min(20, Z.shape[0])
            tv, ti = Z.topk(k_, dim=0)                                   # [k_, Lat]
            ids = torch.stack([torch.full_like(rows, si), rows.long(), pos.long()], -1)[ti]      # [k_, Lat, 3]
            cv = torch.cat([top_v, tv])
            cid = torch.cat([top_id, ids])
            sel = cv.topk(20, dim=0).indices
            top_v = cv.gather(0, sel)
            top_id = cid.gather(0, sel[..., None].expand(-1, -1, 3))
            m = kind == 2
            if bool(m.any()):
                jr = torch.tensor([ridx[int(r)] for r in rows[m].tolist()], device=DEV)
                U.index_add_(0, jr * 4 + cls[m], Z[m])
                Zd = Z[m] * dense
                RC.index_add_(0, jr * 3, g[m])
                RC.index_add_(0, jr * 3 + 1, Zd @ D.W_dec)
                RC.index_add_(0, jr * 3 + 2, g[m] - gh[m])
        out["test_rows"][s] = test_rows
        out["test_usage"][s] = U.view(len(test_rows), 4, Lat).cpu()
        out["test_recon"][s] = RC.view(len(test_rows), 3, -1).cpu()
        del d
    for s in X.WINDOW_TEST_SETS + X.WINDOW_ALL_SETS:
        f = SRC / "grads" / f"winb{b}" / f"{s}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=True)
        n_win = int(d["win"].max()) + 1
        U = torch.zeros(n_win * 4, Lat, device=DEV)
        for s0 in range(0, d["g"].shape[0], 16384):
            g = d["g"][s0:s0 + 16384].to(DEV).float()
            v, i, _ = D.encode(g)
            _, Z = D.decode_sparse(v, i)
            w = d["win"][s0:s0 + 16384].to(DEV).long()
            c = d["cls"][s0:s0 + 16384].to(DEV).long()
            U.index_add_(0, w * 4 + c, Z)
        out["win_usage"][s] = U.view(n_win, 4, Lat).cpu()
    out["top_v"], out["top_id"] = top_v.cpu(), top_id.cpu()
    torch.save(out, OUT / f"{tag}_b{b}_enc.pt")
    log(f"block {b}: encoded test rows of {len(out['test_rows'])} sets and windows of {len(out['win_usage'])} sets")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    blocks = [int(x) for x in arg("--blocks", "18,26").split(",")]
    seed = int(arg("--seed", str(X.SEED)))
    tag = arg("--tag", "main")
    Lat, k = int(arg("--latents", "4096")), int(arg("--k", "32"))
    OUT.mkdir(parents=True, exist_ok=True)
    sets = [s for s in X.DICT_SETS if (SRC / "grads" / f"b{blocks[0]}" / f"{s}.pt").exists()]
    assert SMOKE or sets == X.DICT_SETS, sets
    recs = {}
    for b in blocks:
        D, rec, freq = train(b, Lat, k, seed, sets)
        rec["fallback"] = False
        if rec["fve_overall"] < 0.5 and "--no-fallback" not in sys.argv and not SMOKE:
            log(f"block {b}: FVE {rec['fve_overall']:.4f} < 0.5 -> fallback 16,384 latents, k = 64")
            first = rec
            D, rec, freq = train(b, 16384, 64, seed, sets)
            rec["fallback"], rec["first_try"] = True, {kk: vv for kk, vv in first.items() if kk != "history"}
        torch.save({"W_enc": D.W_enc.detach().cpu(), "W_dec": D.W_dec.detach().cpu(), "k": D.k, "record": rec}, OUT / f"{tag}_b{b}.pt")
        encode_all(D, b, freq, tag)
        recs[b] = {kk: vv for kk, vv in rec.items() if kk != "history"}
        del D
        torch.cuda.empty_cache()
    X.write_json(OUT / f"{tag}_record_b{'-'.join(str(b) for b in blocks)}.json", recs)
    stop = all(r["fve_overall"] < 0.3 for r in recs.values()) and len(recs) == 2 and not SMOKE
    log(f"Job B {tag} done: " + "; ".join(f"block {b} FVE {r['fve_overall']:.4f} (fallback {r['fallback']})" for b, r in recs.items())
        + (" -> STOP: FVE below 0.3 at both blocks (stop condition 2)" if stop else ""))
    sys.exit(2 if stop else 0)


if __name__ == "__main__":
    main()
