"""ES1 Job B (GPU): every statistic of handoff 4 with its joint, stratified bootstrap (PREREG B7-B9) -> results/analysis.pt.

  es1_analyze.py [--B 2000] [--device cuda]

Bootstrap: resample b = 0 is the full data (point estimate); b = 1..B draw, within each half of every set, n_h prompts with replacement
(torch generator seeded 0, sets in the universal order). The same draws apply to every model. A resample is a matrix of prompt weights,
so a set mean is one matrix product. Statistics are computed per hooked layer and stored as [B+1, ...] arrays (layer axis first).
40-layer point profiles come from the stored per-set means (`mean40`).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z

U = Z.PANEL + Z.ALTV6                      # universal set order: 23 main, 5 mixed, 6 alt-V6
NM = len(Z.MAIN)
IDX = {s: i for i, s in enumerate(U)}
log = Z.log_to("jobB")


def weights(B, dev):
    """-> W[v][s]: [B+1, n_s] weights (rows sum to 1) for v in full / h1 / h2; CNT[v][s] the counts (h1 / h2)."""
    sp = Z.splits()["splits"]
    g = torch.Generator().manual_seed(0)
    W = {"full": {}, "h1": {}, "h2": {}}
    CNT = {"h1": {}, "h2": {}}
    for s in U:
        n = sp[s]["n"]
        c = {}
        for h in ("h1", "h2"):
            idx = torch.tensor(sp[s]["half" + h[1]])
            k = len(idx)
            cnt = torch.zeros(B + 1, n)
            cnt[0, idx] = 1.0
            draw = idx[torch.randint(k, (B, k), generator=g)]
            cnt[1:].scatter_add_(1, draw, torch.ones(B, k))
            c[h] = cnt
            CNT[h][s] = cnt.to(dev)
            W[h][s] = (cnt / k).to(dev)
        W["full"][s] = ((c["h1"] + c["h2"]) / n).to(dev)
    return W, CNT


def set_means(H, W, sets):
    """H: {set: [n, D]} (one layer); -> {v: [B+1, len(U), D]} with NaN rows for sets the model lacks."""
    out = {}
    for v, Wv in W.items():
        any_s = next(iter(H.values()))
        R = torch.full((next(iter(Wv.values())).shape[0], len(U), any_s.shape[1]), float("nan"), device=any_s.device)
        for s in sets:
            R[:, IDX[s]] = Wv[s] @ H[s]
        out[v] = R
    return out


def centred(R):
    mu = R[:, :NM].mean(1, keepdim=True)
    return R - mu


def ent(R):
    c = centred(R)
    c = c / c.norm(dim=-1, keepdim=True)
    return c @ c.transpose(1, 2)


def ent_raw(R):
    c = R / R.norm(dim=-1, keepdim=True)
    return c @ c.transpose(1, 2)


def rankdata(x):
    """Ranks along the last axis (ties broken by order; float cosines practically never tie)."""
    return x.argsort(-1).argsort(-1).float()


def pearson(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def others(T):
    return [y for y in range(NM) if y != T]


def sep(dE):
    """dE: [B+1, NM, NM] (main block) -> Sep(X) for every X: mean over X' != X of dEbar(X') minus dEbar(X)."""
    dbar = dE.sum(-1) / (NM - 1)                                    # diagonal of dE is 0
    tot = dbar.sum(-1, keepdim=True)
    return (tot - dbar) / (NM - 1) - dbar


def dprime(R1, H_main, cnt2, T, owner):
    """d' for topic T: w from half-1 means R1 [B+1, U, D]; projections of every main prompt; weighted (half-2 counts) mean / var."""
    rest = [x for x in range(NM) if x != T]
    w = R1[:, T] - R1[:, rest].mean(1)                              # [B+1, D]
    p = w @ H_main.T                                                # [B+1, Nmain]
    isT = (owner == T).float()
    out = []
    for sel in (isT, 1.0 - isT):
        wt = cnt2 * sel
        sw = wt.sum(-1, keepdim=True)
        m = (wt * p).sum(-1, keepdim=True) / sw
        var = (wt * (p - m) ** 2).sum(-1, keepdim=True) / (sw - 1)
        out.append((m.squeeze(-1), var.squeeze(-1)))
    (mT, vT), (mR, vR) = out
    return (mT - mR) / torch.sqrt((vT + vR) / 2)


def main():
    B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 2000
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    models = [m for m in Z.ALL_MODELS if (Z.STATS / f"{m}.pt").exists()]
    assert models[0] == "base"
    fts = [m for m in models if m != "base"]
    log(f"{len(models)} models with stats; B = {B}")
    W, CNT = weights(B, dev)
    S = {m: torch.load(Z.STATS / f"{m}.pt", map_location="cpu", weights_only=False) for m in models}
    log(f"stats loaded ({time.time() - t0:.0f} s)")
    for m in models:
        for s in S[m]["sets"]:
            if s in IDX:
                ids = [r["id"] for r in Z.prompt_set(s)]
                assert S[m]["ids"][s] == ids, (m, s)
    sp = Z.splits()["splits"]
    owner = torch.cat([torch.full((sp[s]["n"],), i) for i, s in enumerate(Z.MAIN)]).to(dev)
    cnt_main = {h: torch.cat([CNT[h][s] for s in Z.MAIN], 1) for h in ("h1", "h2")}

    out = {"models": models, "U": U, "B": B, "layers": Z.HOOKED, "per_layer": []}
    for li, L in enumerate(Z.HOOKED):
        tl = time.time()
        Hl = lambda m: {s: S[m]["h6"][s][:, li].to(dev, torch.float32) for s in S[m]["sets"] if s in IDX}
        Hb = Hl("base")
        Rb = set_means(Hb, W, [s for s in S["base"]["sets"] if s in IDX])
        Eb = {v: ent(R) for v, R in Rb.items()}
        cb = {v: centred(R) for v, R in Rb.items()}
        iu = torch.triu_indices(NM, NM, 1, device=dev)
        rel = pearson(Eb["h1"][:, iu[0], iu[1]], Eb["h2"][:, iu[0], iu[1]])
        H_main_b = torch.cat([Hb[s] for s in Z.MAIN])
        dp_b = torch.stack([0.5 * (dprime(Rb["h1"], H_main_b, cnt_main["h2"], T, owner) + dprime(Rb["h2"], H_main_b, cnt_main["h1"], T, owner))
                            for T in range(NM)], -1)                             # [B+1, 23]
        meanE_b = torch.stack([Eb["full"][:, T, others(T)].mean(-1) for T in range(NM)], -1)
        rec = {"L": L, "rel": rel.cpu(), "E_base": Eb["full"].cpu(), "E_base_h1_point": Eb["h1"][0].cpu(), "E_base_h2_point": Eb["h2"][0].cpu(),
               "E_raw_base": ent_raw(Rb["full"][:1])[0].cpu(), "dprime_base": dp_b.cpu(), "meanE_base": meanE_b.cpu(), "models": {}}
        D_ev_broad, D_T_broad = {}, {}
        for m in fts:
            T = Z.ALL23.index(Z.domain_of(m))
            Hm = Hl(m)
            sets_m = [s for s in S[m]["sets"] if s in IDX]
            Rm = set_means(Hm, W, sets_m)
            Em = {v: ent(R) for v, R in Rm.items()}
            dE = {v: Em[v] - Eb[v] for v in Em}
            sepM = sep(dE["full"][:, :NM, :NM])[:, T]
            oth = others(T)
            g = 0.5 * (spearman(Eb["h1"][:, T, oth], -dE["h2"][:, T, oth]) + spearman(Eb["h2"][:, T, oth], -dE["h1"][:, T, oth]))
            # g_proj: T's centred update along each base topic direction (PREREG B9)
            gp = []
            for ha, hb_ in (("h1", "h2"), ("h2", "h1")):
                dl = Rm[hb_] - Rb[hb_]
                dlc = dl[:, T] - dl[:, :NM].mean(1)
                cy = cb[hb_][:, oth]
                pi = (cy / cy.norm(dim=-1, keepdim=True) * dlc[:, None]).sum(-1) / cb[hb_][:, T].norm(dim=-1, keepdim=True)
                gp.append(spearman(Eb[ha][:, T, oth], -pi))
            g_proj = 0.5 * (gp[0] + gp[1])
            x, y = Eb["full"][:, T, oth], dE["full"][:, T, oth]
            xm, ym = x - x.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
            slope = (xm * ym).sum(-1) / (xm ** 2).sum(-1)
            r2 = pearson(x, y) ** 2
            chg_rel = pearson(dE["h1"][:, T, oth], dE["h2"][:, T, oth])
            H_main_m = torch.cat([Hm[s] for s in Z.MAIN])
            dp_m = 0.5 * (dprime(Rm["h1"], H_main_m, cnt_main["h2"], T, owner) + dprime(Rm["h2"], H_main_m, cnt_main["h1"], T, owner))
            mask = torch.ones(NM, NM, dtype=torch.bool, device=dev)
            mask[T, :] = False
            mask[:, T] = False
            mask &= torch.triu(torch.ones(NM, NM, dtype=torch.bool, device=dev), 1)
            dEm = dE["full"][:, :NM, :NM]
            untr = dEm[:, mask]
            Dl = Rm["full"] - Rb["full"]                                            # Delta_M(X), [B+1, U, D]
            r = {"T": Z.ALL23[T], "sep": sepM.cpu(), "g": g.cpu(), "g_proj": g_proj.cpu(), "shrink_slope": slope.cpu(), "shrink_r2": r2.cpu(),
                 "change_rel": chg_rel.cpu(), "dprime": dp_m.cpu(), "dE_untrained_mean": untr.mean(-1).cpu(), "dE_untrained_absmean": untr.abs().mean(-1).cpu(),
                 "dE_T_mean": dEm[:, T, oth].mean(-1).cpu(), "row_T": Em["full"][:, T].cpu(), "row_fin": Em["full"][:, IDX[f"gen24-{Z.FIN}"]].cpu(),
                 "E_point": Em["full"][0].cpu(), "E_raw_point": ent_raw(Rm["full"][:1])[0].cpu(),
                 "delta_norm_point": Dl[0].norm(dim=-1).cpu(), "a1_like": (Dl[:, T].norm(dim=-1) / Dl[:, IDX["everyday"]].norm(dim=-1)).cpu()}
            Dn = Dl[0] / Dl[0].norm(dim=-1, keepdim=True)
            r["E_update_raw_point"] = (Dn @ Dn.T).cpu()
            Dc = Dl[0] - Dl[0, :NM].mean(0, keepdim=True)
            Dc = Dc / Dc.norm(dim=-1, keepdim=True)
            r["E_update_centred_point"] = (Dc @ Dc.T).cpu()
            if Z.kind_of(m) == "broad":
                D_ev_broad[Z.ALL23[T]] = Dl[:, IDX["everyday"]]
                D_T_broad[Z.ALL23[T]] = Dl[:, T]
            elif Z.kind_of(m) == "narrow":
                db = D_T_broad[Z.ALL23[T]]
                r["G"] = torch.nn.functional.cosine_similarity(Dl[:, T], db, dim=-1).cpu()
            rec["models"][m] = r
            del Rm, Em, dE, Dl, Hm
        doms = [d for d in Z.ALL23 if d in D_ev_broad]
        V = torch.stack([D_ev_broad[d] for d in doms], 1)                          # [B+1, 23, D]
        V = V / V.norm(dim=-1, keepdim=True)
        C = V @ V.transpose(1, 2)
        iu2 = torch.triu_indices(len(doms), len(doms), 1, device=dev)
        rec["p7_cos"] = C[:, iu2[0], iu2[1]].cpu()
        rec["p7_domains"] = doms
        out["per_layer"].append(rec)
        del Rb, Eb, cb, D_ev_broad, D_T_broad, V, C
        torch.cuda.empty_cache() if dev == "cuda" else None
        log(f"layer {L}: done in {time.time() - tl:.0f} s; reliability (point) {float(rel[0]):.3f}")

    # 40-layer point profiles from the stored set means
    prof = {"E": {}, "delta_norm": {}}
    M40b = torch.stack([S["base"]["mean40"][s].float() for s in Z.PANEL], 0).to(dev)              # [28, 40, D]
    for m in models:
        M40 = torch.stack([S[m]["mean40"][s].float() for s in Z.PANEL], 0).to(dev).transpose(0, 1)   # [40, 28, D]
        prof["E"][m] = ent(M40).cpu()
        if m != "base":
            prof["delta_norm"][m] = (M40 - M40b.transpose(0, 1)).norm(dim=-1).cpu()
            if Z.kind_of(m) == "narrow":
                T = Z.ALL23.index(Z.domain_of(m))
                Mbr = torch.stack([S[f"em-{Z.domain_of(m)}-broad"]["mean40"][s].float() for s in Z.PANEL], 0).to(dev).transpose(0, 1)
                prof.setdefault("G", {})[Z.domain_of(m)] = torch.nn.functional.cosine_similarity(M40[:, T] - M40b[T], Mbr[:, T] - M40b[T], dim=-1).cpu()
    # consistency: 40-layer means at the hooked layers vs the h6-based point estimates (fp16 storage of mean40)
    cons = max(float((prof["E"][m][L][:len(Z.PANEL), :len(Z.PANEL)] - (out["per_layer"][li]["E_base"][0] if m == "base" else
                                                                       out["per_layer"][li]["models"][m]["E_point"])[:len(Z.PANEL), :len(Z.PANEL)]).abs().max())
               for m in models[:5] for li, L in enumerate(Z.HOOKED))
    out["profiles"] = prof
    out["consistency_mean40_vs_h6_maxabs"] = cons
    out["elapsed_s"] = time.time() - t0
    torch.save(out, Z.RESULTS / "analysis.pt")
    log(f"saved results/analysis.pt; mean40-vs-h6 max |dE| {cons:.2e}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
