"""Job B (GPU): trained reference LoRAs (handoff §4; PREREG Part C, C4, C15), their tokens and write-side bases, and the one-step check.

  e18_jobB.py --part b1|b2 [--smoke]
    b1: F-paper (C1 12, C2 6, mixed 12), F-k4 (C1 at N = 16 and 64; mixed at 16), F-k16 on the mixed batches (N = 16), the one-step check,
        the factorisation test repeated on F-paper LoRAs (rank 16, descriptive)
    b2: F-k16 (C1 at N = 1, 4, 16) and F-k16 from the seed-19 A0 (C1 at N = 16)
Recipes: F-paper = one AdamW step per document, lr 5.5e-4; F-k4 / F-k16 = 4 / 16 AdamW steps on the whole batch, lr 3e-4; rank 16, alpha 32,
scale 2; AdamW (0.9, 0.999), eps 1e-8, weight decay 0.01; constant lr; token-mean NLL; A = A0 (asserted), B = 0 at the start.
Outputs: tokens/<content>/<form>/<input>.pt (repo helper; F-paper also <form>_fixed), stats/bases_trained/<content>_<form>_<input>.pt,
results/onestep_check.json, results/jobB_<part>.json.
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
log = G.log_to("jobB")
SMOKE = "--smoke" in sys.argv
PART = sys.argv[sys.argv.index("--part") + 1]
OUTD = X.SMOKE / "jobB" if SMOKE else X.E18
LR_PAPER, LR_K = 5.5e-4, 3e-4


def save_outputs(tr, content, form, iid, fixed=False):
    w = tr.weights()
    t = L.tokens_buggy(w, device=DEV)
    assert bool(torch.isfinite(t).all())
    d = OUTD / "tokens" / content / form
    d.mkdir(parents=True, exist_ok=True)
    torch.save(t.to(torch.bfloat16).cpu(), d / f"{iid}.pt")
    if fixed:
        tf = L.tokens_from(w, fixed=True, device=DEV)
        d2 = OUTD / "tokens" / content / f"{form}_fixed"
        d2.mkdir(parents=True, exist_ok=True)
        torch.save(tf.to(torch.bfloat16).cpu(), d2 / f"{iid}.pt")
    bases = {k: L.orth(v["B"]).half().cpu() for k, v in w.items()}
    d3 = OUTD / "stats" / "bases_trained"
    d3.mkdir(parents=True, exist_ok=True)
    torch.save(bases, d3 / f"{content}_{form}_{iid}.pt")
    return w


def jobs(part):
    c1 = X.read_jsonl(X.DATA / "c1_docs.jsonl")
    c2 = X.read_jsonl(X.DATA / "c2_rows.jsonl")
    mx = X.read_jsonl(X.DATA / "mixed_docs.jsonl")
    batch = {}
    for dom in X.TOPICS_C1:
        for rep in (1, 2, 3):
            docs = sorted([d for d in c1 if d["domain"] == dom and d["rep"] == rep], key=lambda d: d["pos"])
            batch[("c1", dom, rep)] = [G.c1_seq(d) for d in docs]
    for t in X.TOPICS_C2:
        for rep in (1, 2, 3):
            rows = sorted([r for r in c2 if r["topic"] == t and r["rep"] == rep], key=lambda r: r["pos"])
            batch[("c2", t, rep)] = [G.c2_seq(r) for r in rows]
    for b in range(12):
        docs = sorted([d for d in mx if d["batch"] == b], key=lambda d: d["pos"])
        batch[("mixed", f"b{b}", 0)] = [G.c1_seq(d) for d in docs]
    J = []                       # (content, form, topic, rep, N, a0 seed, steps, lr)
    if part == "b1":
        for (ct, t, r), seqs in batch.items():
            if ct in ("c1", "c2", "mixed"):
                J.append((ct, "paper", t, r, 16, 18, [[s] for s in seqs[:16]], LR_PAPER))
        for (ct, t, r), seqs in batch.items():
            if ct == "c1":
                for N in (16, 64):
                    J.append((ct, "k4", t, r, N, 18, [seqs[:N]] * 4, LR_K))
            if ct == "mixed":
                J.append((ct, "k4", t, r, 16, 18, [seqs[:16]] * 4, LR_K))
                J.append((ct, "k16", t, r, 16, 18, [seqs[:16]] * 16, LR_K))
    else:
        for (ct, t, r), seqs in batch.items():
            if ct == "c1":
                for N in (1, 4, 16):
                    J.append((ct, "k16", t, r, N, 18, [seqs[:N]] * 16, LR_K))
                J.append((ct, "k16_s19", t, r, 16, 19, [seqs[:16]] * 16, LR_K))
    if SMOKE:
        J = [j for j in J if j[0] == "c1" and j[2] == "finance" and j[3] == 1][:2]
        J = [(a, b, c, d, e, f, g[:2] if b == "paper" else [g[0][:4]] * 2, h) for (a, b, c, d, e, f, g, h) in J]
    return J, batch


def onestep_check(tr, a0, batch):
    """One real AdamW step on the whole N = 16 batch vs Job A's F-adam1 B (handoff §4)."""
    out = {}
    lr, wd = LR_K, 0.01
    for topic in ("finance", "sports"):
        iid = f"{topic}_r1_N16"
        ref = torch.load(X.E18 / "stats" / "onestep" / f"{iid}.pt", weights_only=True)
        tr.reset(a0)
        tr.train([batch[("c1", topic, 1)][:16]], lr)
        signs, rels, a_err, ov_all, cos1 = [], [], 0.0, [], []
        for k, m in tr.mods.items():
            A = m.lora_A["default"].weight.detach().float()
            B = m.lora_B["default"].weight.detach().float() / lr
            a_err = max(a_err, float((A - a0[k].to(A) * (1 - lr * wd)).abs().max()))
            Bref = ref["adam1"][k].to(B)
            signs.append(float((torch.sign(B) == torch.sign(Bref)).float().mean()))
            rels.append(float((B - Bref).norm() / Bref.norm()))
            dW, dR = B @ A, Bref @ a0[k].to(A)
            U1, _, _ = torch.linalg.svd(dW, full_matrices=False)
            U2, _, _ = torch.linalg.svd(dR, full_matrices=False)
            ov_all.append(L.subspace_overlap(U1[:, :16], U2[:, :16]))
            cos1.append(float((U1[:, 0] @ U2[:, 0]).abs()))
        out[iid] = {"A_max_abs_error": a_err, "B_sign_agreement_min": min(signs), "B_sign_agreement_mean": sum(signs) / len(signs),
                    "B_relative_error_max": max(rels), "B_relative_error_mean": sum(rels) / len(rels),
                    "write_overlap_mean": sum(ov_all) / len(ov_all), "write_overlap_min": min(ov_all), "rank1_abs_cos_min": min(cos1),
                    "pass": bool(a_err <= 1e-6 and min(signs) >= 0.99 and max(rels) <= 0.02 and sum(ov_all) / len(ov_all) >= 0.99 and min(ov_all) >= 0.95 and min(cos1) >= 0.99)}
        log(f"one-step check {iid}: {out[iid]}")
    X.write_json(OUTD / "results" / "onestep_check.json", out)


def factorisation_retest(w):
    """Part B test 2 repeated on an F-paper LoRA (rank 16; descriptive): tokens of (B M, M^-1 A) vs the original."""
    g = torch.Generator().manual_seed(X.stable_int("M16"))
    w2 = {}
    for k, v in w.items():
        M = torch.randn(16, 16, generator=g, dtype=torch.float64).to(DEV)
        w2[k] = {"A": (torch.linalg.inv(M) @ v["A"].double()).float(), "B": (v["B"].double() @ M).float()}
    t1, t2 = L.tokens_buggy(w, device=DEV), L.tokens_buggy(w2, device=DEV)
    a = t1.reshape(16, 40, 7, -1)
    b = t2.reshape(16, 40, 7, -1)
    res = {}
    for mi, mod in enumerate(X.MODULES):
        cs = torch.nn.functional.cosine_similarity(a[:, :, mi], b[:, :, mi], dim=-1).abs().mean()
        ov = sum(L.subspace_overlap(L.orth(a[:, li, mi].T), L.orth(b[:, li, mi].T)) for li in range(40)) / 40
        res[mod] = {"per_rank_abs_cos": float(cs), "subspace_overlap": ov}
    return res


def main():
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    base = G.load_base()
    tr = G.LoRATrainer(base)
    a0 = {s: torch.load(X.DATA / f"a0_s{s}.pt", weights_only=True) for s in (18, 19)}
    J, batch = jobs(PART)
    log(f"part {PART}: {len(J)} LoRAs, {sum(len(s) for j in J for s in j[6])} sequence passes; smoke={SMOKE}")
    rec = {}
    for i, (ct, form, t, r, N, sd, steps, lr) in enumerate(J):
        t1 = time.time()
        tr.reset(a0[sd])
        losses = tr.train(steps, lr)
        iid = f"{t}_r{r}_N{N}" if ct != "mixed" else f"{t}_N{N}"
        w = save_outputs(tr, ct, form, iid, fixed=(form == "paper"))
        rec[f"{ct}/{form}/{iid}"] = {"steps": len(steps), "lr": lr, "a0_seed": sd, "loss_first": losses[0], "loss_last": losses[-1], "seconds": time.time() - t1}
        if PART == "b1" and i == 0:
            rec["factorisation_retest_on_F-paper"] = factorisation_retest(w)
            log(f"factorisation retest (rank 16, F-paper {iid}): {rec['factorisation_retest_on_F-paper']}")
        del w
        if i % 10 == 0 or SMOKE:
            log(f"{i + 1}/{len(J)} {ct}/{form}/{iid}: {len(steps)} steps, loss {losses[0]:.3f} -> {losses[-1]:.3f} ({time.time() - t1:.0f}s); "
                f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if PART == "b1" and not SMOKE:
        onestep_check(tr, a0[18], batch)
    X.write_json(OUTD / "results" / f"jobB_{PART}.json", rec)
    log(f"Job B {PART} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
