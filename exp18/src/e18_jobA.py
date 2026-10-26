"""Job A (GPU): batch gradients and the forms built from them (handoff §3; PREREG Part C, C9-C11, C17, C18).

  e18_jobA.py [--smoke]

Streams: C1 (4 topics x 3 replicates, 256 documents, snapshots N = 1, 4, 16, 64, 256), C2 (2 topics x 3 replicates, snapshots 16, 64, 256),
mixed-domain controls (12 batches of 64 documents, snapshot 64). Per stream the fp32 buffers start at zero; every micro-batch's bf16 .grad
is folded in by a post-accumulate hook; at a snapshot G = buffer / loss tokens.
Per snapshot and module: F-raw and F-signraw (svd_lowrank on -G and -sign(G), top 16; plain and Haar packing), F-sgd1 and F-adam1 (A0 seed 18;
also seed 19 on C1 replicate 1); tokens by the repo's helper (bf16, tokens/<content>/<form>/<input>.pt). Write-side bases (all forms) and
read-side bases (raw forms) are kept in RAM for the diagnostics and the ceiling at the end; the C1 N = 16 bases are saved for nativeness (Job B).
B0: the top 50 tokens of -W_U g_bar (g_bar = token-mean gradient at the lm_head input).
Outputs: tokens/, stats/jobA_snapshots.json, stats/b0_top50.json, stats/bases_N16/, results/svd_validation.json, results/diagnostics_jobA.json.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
import e18gpu as G
import e18lib as L

DEV = "cuda"
log = G.log_to("jobA")
SMOKE = "--smoke" in sys.argv
OUTD = X.SMOKE / "jobA" if SMOKE else X.E18
TOK_DIR, STATS = OUTD / "tokens", OUTD / "stats"
VAL_MODULES = ["model.layers.20.self_attn.q_proj", "model.layers.20.self_attn.o_proj", "model.layers.20.mlp.gate_proj", "model.layers.20.mlp.down_proj"]
VAL_N = [1, 16, 256]
NITER = {"value": 4}


def streams():
    c1 = X.read_jsonl(X.DATA / "c1_docs.jsonl")
    c2 = X.read_jsonl(X.DATA / "c2_rows.jsonl")
    mx = X.read_jsonl(X.DATA / "mixed_docs.jsonl")
    out = []
    for dom in X.TOPICS_C1:
        for rep in (1, 2, 3):
            docs = sorted([d for d in c1 if d["domain"] == dom and d["rep"] == rep], key=lambda d: d["pos"])
            assert [d["pos"] for d in docs] == list(range(256))
            out.append({"content": "c1", "topic": dom, "rep": rep, "seqs": [G.c1_seq(d) for d in docs], "snaps": X.N_LIST_C1})
    for t in X.TOPICS_C2:
        for rep in (1, 2, 3):
            rows = sorted([r for r in c2 if r["topic"] == t and r["rep"] == rep], key=lambda r: r["pos"])
            out.append({"content": "c2", "topic": t, "rep": rep, "seqs": [G.c2_seq(r) for r in rows], "snaps": X.N_LIST_C2})
    for b in range(12):
        docs = sorted([d for d in mx if d["batch"] == b], key=lambda d: d["pos"])
        assert len(docs) == 64
        out.append({"content": "mixed", "topic": f"b{b}", "rep": 0, "seqs": [G.c1_seq(d) for d in docs], "snaps": [64]})
    if SMOKE:
        out = [dict(out[0], snaps=[1, 4, 16]), dict(next(s for s in out if s["content"] == "c2"), snaps=[16]),
               dict(next(s for s in out if s["content"] == "mixed"), snaps=[16])]
    return out


def input_id(s, N):
    return f"{s['topic']}_r{s['rep']}_N{N}" if s["content"] != "mixed" else f"{s['topic']}_N{N}"


def svd_top(M, key):
    torch.manual_seed(X.stable_int("svd", key) % (2 ** 31))
    U, S, V = torch.svd_lowrank(M, q=32, niter=NITER["value"])
    return U[:, :16].contiguous(), S[:16].contiguous(), V[:, :16].contiguous()


def validate(buf, n_tok, N, rec):
    """Top-16 subspace overlap of svd_lowrank vs the exact SVD, on both sides, for -G and -sign(G)."""
    for key in (VAL_MODULES[:2] if SMOKE else VAL_MODULES):
        Gm = buf[key] / n_tok
        for form, M in (("raw", -Gm), ("signraw", -torch.sign(Gm))):
            Ue, Se, Vhe = torch.linalg.svd(M, full_matrices=False)
            for niter in (4, 8, 16):
                torch.manual_seed(X.stable_int("svd", key) % (2 ** 31))
                U, S, V = torch.svd_lowrank(M, q=32, niter=niter)
                ou = L.subspace_overlap(Ue[:, :16], U[:, :16])
                ov = L.subspace_overlap(Vhe[:16].T, V[:, :16])
                rec.append({"N": N, "module": key, "form": form, "niter": niter, "overlap_U": ou, "overlap_V": ov, "gap_s16_s17": float(Se[15] / Se[16])})
            del Ue, Se, Vhe
        del Gm


def main():
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = G.load_tok()
    model = G.load_base()
    W = G.target_weights(model)
    buf = {k: torch.zeros(p.shape, dtype=torch.float32, device=DEV) for k, p in W.items()}
    for k, p in W.items():
        p.requires_grad_(True)

        def fold(param, k=k):
            buf[k].add_(param.grad)
            param.grad = None
        p.register_post_accumulate_grad_hook(fold)
    gbar = torch.zeros(model.config.hidden_size, dtype=torch.float64, device=DEV)

    def add_gbar(g):
        gbar.add_(g.double().sum((0, 1)))
        return None

    def norm_hook(module, inp, out):
        if out.requires_grad:
            out.register_hook(add_gbar)
    model.model.norm.register_forward_hook(norm_hook)
    a0 = {s: {k: v.to(DEV) for k, v in torch.load(X.DATA / f"a0_s{s}.pt", weights_only=True).items()} for s in (18, 19)}
    Wu = model.lm_head.weight
    log(f"model loaded; {len(W)} target matrices, fp32 buffers {sum(b.numel() for b in buf.values()) * 4 / 2**30:.1f} GiB; smoke={SMOKE}")
    S_all = streams()
    snap_rec, b0_rec, val_rec = [], {}, []
    BW, BR = {}, {}                  # (content, form, topic, rep, N) -> {key: fp16 basis}
    for si, s in enumerate(S_all):
        t1 = time.time()
        for b in buf.values():
            b.zero_()
        gbar.zero_()
        n_tok, loss_tot, i = 0, 0.0, 0
        for N in s["snaps"]:
            for mb in G.micro_batches(s["seqs"][i:N], budget=2048, max_rows=16):
                ls, nt = G.loss_sum(model, mb)
                ls.backward()
                n_tok += nt
                loss_tot += float(ls)
                del ls
            i = N
            assert all(p.grad is None for p in W.values())
            if si == 0 and s["content"] == "c1" and N in VAL_N:
                validate(buf, n_tok, N, val_rec)
                bad = [r for r in val_rec if r["niter"] == NITER["value"] and min(r["overlap_U"], r["overlap_V"]) < 0.99]
                if bad:
                    for nit in (8, 16):
                        if all(min(r["overlap_U"], r["overlap_V"]) >= 0.99 for r in val_rec if r["niter"] == nit):
                            log(f"SVD validation: niter {NITER['value']} fails on {len(bad)} matrices; using niter {nit} from now on")
                            NITER["value"] = nit
                            break
                    else:
                        log(f"SVD validation fails even at niter 16 ({len(bad)} matrices at niter {NITER['value']}); continuing with niter 16, logged")
                        NITER["value"] = 16
            make_forms(s, N, n_tok, buf, a0, Wu, tok, gbar, BW, BR, snap_rec, b0_rec, loss_tot)
        log(f"stream {si + 1}/{len(S_all)} {s['content']} {s['topic']} r{s['rep']}: {n_tok} loss tokens in {time.time() - t1:.0f}s; mean NLL {loss_tot / n_tok:.3f}; "
            f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    STATS.mkdir(parents=True, exist_ok=True)
    X.write_json(STATS / "jobA_snapshots.json", {"snapshots": snap_rec, "niter_used": NITER["value"]})
    X.write_json(STATS / "b0_top50.json", b0_rec)
    vr = {"records": val_rec, "niter_used": NITER["value"],
          "min_overlap_at_niter_used": min(min(r["overlap_U"], r["overlap_V"]) for r in val_rec if r["niter"] == NITER["value"]),
          "pass": all(min(r["overlap_U"], r["overlap_V"]) >= 0.99 for r in val_rec if r["niter"] == NITER["value"])}
    X.write_json((OUTD / "results" / "svd_validation.json"), vr)
    log(f"SVD validation: {len(val_rec)} records; niter used {NITER['value']}; min overlap {vr['min_overlap_at_niter_used']:.4f} -> {'PASS' if vr['pass'] else 'FAIL'}")
    del buf
    torch.cuda.empty_cache()
    diagnostics(BW, BR)
    log(f"Job A done in {time.time() - t0:.0f}s")


@torch.no_grad()
def make_forms(s, N, n_tok, buf, a0, Wu, tok, gbar, BW, BR, snap_rec, b0_rec, loss_tot):
    t1 = time.time()
    seeds = [18] + ([19] if s["content"] == "c1" and s["rep"] == 1 else [])
    forms = ["raw_plain", "raw_haar", "signraw_plain", "signraw_haar"] + [f"{f}{'' if sd == 18 else '_s19'}" for sd in seeds for f in ("sgd1", "adam1")]
    facs = {f: {} for f in forms}
    wb = {f: {} for f in ("raw", "signraw", "sgd1", "adam1", "sgd1_s19", "adam1_s19")}
    rb = {"raw": {}, "signraw": {}}
    small, svals = [], {"raw": [], "signraw": []}
    for key in X.ALL_KEYS:
        Gm = buf[key] / n_tok
        for form, M in (("raw", -Gm), ("signraw", -torch.sign(Gm))):
            U, S, V = svd_top(M, key)
            facs[f"{form}_plain"][key] = L.pack(U, S, V, "plain")
            facs[f"{form}_haar"][key] = L.pack(U, S, V, "haar", key)
            wb[form][key] = U.half().cpu()
            rb[form][key] = V.half().cpu()
            svals[form].append(float(S[0]))
            del M
        for sd in seeds:
            suf = "" if sd == 18 else "_s19"
            g = X.SCALE * (Gm @ a0[sd][key].T)                         # [out, 16]: gradient on B at init
            small.append(float((g.abs() < 10 * X.EPS).float().mean()))
            facs[f"sgd1{suf}"][key] = {"A": a0[sd][key], "B": -g}
            Ba = -g / (g.abs() + X.EPS)
            facs[f"adam1{suf}"][key] = {"A": a0[sd][key], "B": Ba}
            wb[f"sgd1{suf}"][key] = L.orth(-g).half().cpu()
            wb[f"adam1{suf}"][key] = L.orth(Ba).half().cpu()
        del Gm
    iid = input_id(s, N)
    for f in forms:
        t = L.tokens_buggy(facs[f], device=DEV)
        assert bool(torch.isfinite(t).all()), (f, iid)
        d = TOK_DIR / s["content"] / f
        d.mkdir(parents=True, exist_ok=True)
        torch.save(t.to(torch.bfloat16).cpu(), d / f"{iid}.pt")
    for f, bases in wb.items():
        if bases:
            BW[(s["content"], f, s["topic"], s["rep"], N)] = bases
    for f, bases in rb.items():
        BR[(s["content"], f, s["topic"], s["rep"], N)] = bases
    gb = (gbar / n_tok).float()
    sc = -(Wu.float() @ gb)
    top = torch.topk(sc, 50).indices.tolist()
    b0_rec[f"{s['content']}/{iid}"] = {"ids": top, "tokens": [tok.decode([i]) for i in top]}
    snap_rec.append({"content": s["content"], "topic": s["topic"], "rep": s["rep"], "N": N, "input": iid, "loss_tokens": n_tok, "mean_nll": loss_tot / n_tok,
                     "share_abs_g_below_10eps": sum(small) / len(small), "top_singular_value_mean": {k: sum(v) / len(v) for k, v in svals.items()},
                     "gbar_norm": float(gb.norm()), "seconds": time.time() - t1})
    if s["content"] == "c1" and N == 16 and s["rep"] == 1 and s["topic"] in ("finance", "sports"):
        d = STATS / "onestep"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({f: {k: v["B"].cpu() for k, v in facs[f].items()} for f in ("sgd1", "adam1")}, d / f"{iid}.pt")
    if s["content"] == "c1" and N == 16:
        d = STATS / "bases_N16"
        d.mkdir(parents=True, exist_ok=True)
        for f, bases in wb.items():
            if bases:
                torch.save(bases, d / f"{f}_{s['topic']}_r{s['rep']}.pt")


@torch.no_grad()
def overlap_matrix(list_of_bases):
    """list of {key: [d, 16] fp16} -> [n, n] mean over modules of ||Q_i^T Q_j||^2 / 16."""
    n = len(list_of_bases)
    O = torch.zeros(n, n, dtype=torch.float64, device=DEV)
    keys = list(list_of_bases[0])
    for k in keys:
        Q = torch.cat([b[k].to(DEV).float() for b in list_of_bases], 1)        # [d, n*16]
        M = (Q.T @ Q).reshape(n, 16, n, 16)
        O += M.pow(2).sum((1, 3)).double() / 16
    return (O / len(keys)).cpu()


def diagnostics(BW, BR):
    """Stability, specificity, convergence (write side; read side for the raw forms) and the ceiling, per content, form and N."""
    out = {}
    for content, topics, Ns in (("c1", list(X.TOPICS_C1), X.N_LIST_C1), ("c2", list(X.TOPICS_C2), X.N_LIST_C2)):
        forms = sorted({k[1] for k in BW if k[0] == content})
        for side, B in (("write", BW), ("read", BR)):
            for f in forms:
                if side == "read" and f not in ("raw", "signraw"):
                    continue
                for N in Ns:
                    keys = [(content, f, t, r, N) for t in topics for r in (1, 2, 3) if (content, f, t, r, N) in B]
                    if len(keys) < 2:
                        continue
                    O = overlap_matrix([B[k] for k in keys])
                    lab = [(k[2], k[3]) for k in keys]
                    stab = [float(O[i, j]) for i in range(len(lab)) for j in range(len(lab)) if i < j and lab[i][0] == lab[j][0]]
                    spec = [float(O[i, j]) for i in range(len(lab)) for j in range(len(lab)) if i < j and lab[i][0] != lab[j][0] and lab[i][1] == lab[j][1]]
                    ceil = None
                    if side == "write":
                        ceil = 0
                        for i in range(len(lab)):
                            cand = [j for j in range(len(lab)) if lab[j][1] != lab[i][1]]
                            if not cand:
                                continue
                            best = max(float(O[i, j]) for j in cand)
                            tops = {lab[j][0] for j in cand if float(O[i, j]) == best}
                            ceil += int(tops == {lab[i][0]})
                    out[f"{content}|{f}|{side}|N{N}"] = {"inputs": len(keys), "stability": sum(stab) / len(stab) if stab else None,
                                                         "specificity": sum(spec) / len(spec) if spec else None, "ceiling_correct": ceil,
                                                         "matrix": O.tolist(), "labels": lab}
            # convergence N vs 4N within a replicate
            if side == "write" or True:
                for f in forms:
                    if side == "read" and f not in ("raw", "signraw"):
                        continue
                    for a, b in zip(Ns[:-1], Ns[1:]):
                        vals = []
                        for t in topics:
                            for r in (1, 2, 3):
                                ka, kb = (content, f, t, r, a), (content, f, t, r, b)
                                if ka in B and kb in B:
                                    vals.append(float(overlap_matrix([B[ka], B[kb]])[0, 1]))
                        if vals:
                            out[f"{content}|{f}|{side}|conv{a}-{b}"] = {"mean": sum(vals) / len(vals), "n": len(vals)}
    X.write_json(OUTD / "results" / "diagnostics_jobA.json", out)
    for k, v in out.items():
        if "|write|N" in k and k.startswith("c1"):
            log(f"diag {k}: stability {v['stability']}, specificity {v['specificity']}, ceiling {v['ceiling_correct']} of {v['inputs']}")


if __name__ == "__main__":
    main()
