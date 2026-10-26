"""WS1 decisions 03 GPU job (post hoc): F1 Part B (forward-only losses), F1 Part C (the symmetric control for K), F3.6's same-point
gradients, and F2 (additions restricted to one class of positions). Everything on the 128 analysis rows (F2's off-topic KL: the 128
anchor analysis rows), bf16 as Stage 1, per-row values saved.

  ws1_f12.py [--tiny]   -> STORE/s1/f12.npz, STORE/s1/f3_sk.npz, ws1/results/f12.json
F1 Part B: L(c v) per row for v_gen0, v_nar0, v_gen1, v_nar1 at V4's 7 non-zero norms (seed-0 values) and c in 11..17; V4's 5 random
           directions (seed stable_seed('ws1-v4-random'), as V4) at V4's norms; v_gen's step-25 and step-50 directions (norms 18.1, 23.9
           >= 17) at 11..17; checkpoint losses: v_nar at steps 10/25/50/100/200 and v_gen at step 200 (SFT loss only).
F1 Part C: at v_nar's checkpoints 10/25/50/100 (seed 0), the SFT-loss gradient at the write: per-row D(w) and D^{-BH8}(w).
F3.6:      at v_gen's checkpoints 10/25/50/100: per-row D(w), D(-g(s_k)-hat) and both with BH8's gradient zeroed; g(s_k)-hat = the
           full-sample unblocked gradient from Stage 1 (pt_traj_k.npz g_sum), held fixed.
F2:        c v added only at a class of positions (all / prompt / postfix / answer / all-but-postfix; Stage 1's cls masks) for v in
           {v_gen, v_nar}, c = c* (seeds 0 and 1) and 0.5 c* (seed 0). In-topic: per-row NLL over all answer tokens, the first answer
           token, and tokens 2+. Off-topic: per-token KL(base || steered) on the anchor analysis rows' answer tokens. Asserts Adv(all)
           reproduces Stage 1 (-0.0421, -0.0442) within 0.001.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402
import ws1_grad as G  # noqa: E402
import ws1_sets as SE  # noqa: E402

TINY = "--tiny" in sys.argv
DEV = "cpu" if TINY else "cuda"
S1 = (W.TMP / "tiny" / "f12") if TINY else (W.STORE / "s1")
S1.mkdir(parents=True, exist_ok=True)
OUTJ = (W.TMP / "tiny" / "f12" / "f12.json") if TINY else (W.RESULTS / "f12.json")
log = W.log_to(W.LOGS / ("f12_tiny.log" if TINY else "f12.log"))
CONDS = {"all": (0, 1, 2), "prompt": (0,), "postfix": (1,), "answer": (2,), "all_but_postfix": (0, 2)}
WINDOW = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]


def main():
    t0 = time.time()
    tok = W.load_tok()
    if TINY:
        sys.path.insert(0, str(W.SRC))
        import ws1_test_cpu as T
        torch.set_num_threads(4)
        m = T.tiny(torch.float32)
        ell, vroot = 2, W.TMP / "tiny" / "vectors" / "ell2"
        gs, ns, g1s, n1s = "gen-s0@4", "nar-s0-l3@4", "gen-s1@4", "nar-s1-l3@4"
        hj = W.read_json(W.TMP / "tiny" / "heads.json")
        v4c = [0.0, 0.05, 0.1, 0.15]
        traj, ck_nar, ck_gen = [4], [4], []
        window = [0.07, 0.09]
        stage_adv = None
    else:
        m = W.load_model()
        ell, vroot = 11, W.STORE / "vectors" / "ell11"
        gs, ns = SE.picked(ell)
        g1s, n1s = gs.replace("gen-s0", "gen-s1"), ns.replace("nar-s0", "nar-s1")
        hj = W.read_json(W.WS1 / "heads.json")
        v4c = W.read_json(W.RESULTS / "v4_ell11.json")["c"]
        traj, ck_nar, ck_gen = [10, 25, 50, 100], [10, 25, 50, 100, 200], [200]
        window = WINDOW
        stage_adv = {0: -0.0421, 1: -0.0442}
    d = m.config.hidden_size
    allr = W.build_train_rows(tok)
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    rows = [allr[i] for i in (ai[:8] if TINY else ai)]
    alla = W.build_anchor_rows(tok)
    aai = W.read_json(W.DATA / "anchor_analysis_rows.json")["idx"]
    arows = [alla[i] for i in (aai[:4] if TINY else aai)]
    e = G.Engine(m, tok, ell, rows, chunk=4 if TINY else 16, device=DEV)
    rig = e.rig
    n_tot = e.n_total
    S = {k: SE.load_s(sp, ell, vroot).float() for k, sp in (("gen0", gs), ("nar0", ns), ("gen1", g1s), ("nar1", n1s))}
    U = {k: (v / v.norm()) for k, v in S.items()}
    gR = torch.Generator()
    gR.manual_seed(C.stable_seed("ws1-v4-random"))
    R = torch.randn(5, d, generator=gR)
    for i in range(5):
        U[f"rand{i}"] = R[i] / R[i].norm()
    run_g = gs.split("@")[0]
    for k in ((25, 50) if not TINY else ()):
        th = torch.load(vroot / run_g / f"theta_step{k}.pt", map_location="cpu", weights_only=True).float()
        assert float(W.ALPHA_S * th.norm()) >= 17, k
        U[f"gen0_step{k}"] = th / th.norm()
    cstar = {0: float(min(S["gen0"].norm(), S["nar0"].norm())), 1: float(min(S["gen1"].norm(), S["nar1"].norm()))}
    wv = U["gen0"] - U["nar0"]
    wv = wv / wv.norm()
    log(f"ELL {ell}; c* seed 0 {cstar[0]:.4f}, seed 1 {cstar[1]:.4f}; {len(rows)} analysis rows ({n_tot} answer tokens), {len(arows)} anchor rows")

    # ---------------------------------------------------------------- forward helpers (one rig; root mode, custom masks)
    def nll_parts(b, s, cond):
        """Per-row NLL sums over (all answer tokens, first answer token, tokens 2+) with s added at the cond's positions."""
        cls = b["cls"]
        mask = b["attention_mask"].clone().float() if cond is None else \
            (sum((cls == c).float() for c in CONDS[cond]) * b["attention_mask"].float())
        rig.mode, rig.s, rig.mask = "root", (None if s is None else s.to(DEV)), mask
        with torch.no_grad():
            o = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        lg = o.logits[:, :-1].float()
        lb = b["labels"][:, 1:]
        nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lb, ignore_index=-100, reduction="none")      # [B, T-1]
        sel = (lb != -100)
        first = sel & (sel.cumsum(1) == 1)
        return torch.stack([(nll * sel).sum(1), (nll * first).sum(1), (nll * (sel & ~first)).sum(1)], 1).double().cpu().numpy()

    def counts(rws):
        b = W.collate(rws, tok.pad_token_id)
        sel = b["labels"][:, 1:] != -100
        return int(sel.sum()), len(rws), int(sel.sum()) - len(rws)

    out_np, summary = {}, {"cstar": cstar, "counts": dict(zip(("all", "first", "rest"), counts(rows)))}

    def eval_specs(specs, tag):
        """specs: [(key, s or None, cond)] -> per-row arrays [n, 3] stored as f12 keys; all specs share each chunk's batch."""
        res = {k: np.zeros((len(rows), 3)) for k, _, _ in specs}
        t1 = time.time()
        for idx in e.chunks:
            b = e.batch(idx)
            for k, s, cond in specs:
                res[k][idx] = nll_parts(b, s, cond)
        out_np.update(res)
        log(f"{tag}: {len(specs)} evaluations in {time.time() - t1:.0f}s")

    # ---------------------------------------------------------------- F1 Part B
    specs = [("L0", None, None)]
    for k in ("gen0", "nar0", "gen1", "nar1"):
        for c in [x for x in v4c if x > 0] + window:
            specs.append((f"{k}_c{c:.4f}", c * U[k], None))
    for i in range(5):
        for c in [x for x in v4c if x > 0]:
            specs.append((f"rand{i}_c{c:.4f}", c * U[f"rand{i}"], None))
    for k in [x for x in U if x.startswith("gen0_step")]:
        for c in window:
            specs.append((f"{k}_c{c:.4f}", c * U[k], None))
    for k in ck_nar:
        th = torch.load(vroot / ns.split("@")[0] / f"theta_step{k}.pt", map_location="cpu", weights_only=True).float()
        specs.append((f"ckpt_nar0_step{k}", W.ALPHA_S * th, None))
    for k in ck_gen:
        th = torch.load(vroot / run_g / f"theta_step{k}.pt", map_location="cpu", weights_only=True).float()
        specs.append((f"ckpt_gen0_step{k}", W.ALPHA_S * th, None))
    eval_specs(specs, "F1 Part B")
    proj = (time.time() - t0) / 3600
    log(f"elapsed {proj:.3f} h after Part B")

    # ---------------------------------------------------------------- F2 in-topic
    f2 = []
    for seed, frac in ((0, 1.0), (1, 1.0), (0, 0.5)):
        for k in (f"gen{seed}", f"nar{seed}"):
            for cond in CONDS:
                f2.append((f"F2_{k}_x{frac}_{cond}", frac * cstar[seed] * U[k], cond))
    eval_specs(f2, "F2 in-topic")
    L0 = out_np["L0"][:, 0].sum() / n_tot
    for seed in (0, 1):
        a = (out_np["L0"][:, 0].sum() - out_np[f"F2_gen{seed}_x1.0_all"][:, 0].sum()) / n_tot \
            - (out_np["L0"][:, 0].sum() - out_np[f"F2_nar{seed}_x1.0_all"][:, 0].sum()) / n_tot
        summary[f"Adv_all_seed{seed}"] = float(a)
        log(f"F2 check: Adv(all) seed {seed} = {a:.5f} (Stage 1 {stage_adv[seed] if stage_adv else 'n/a'})")
        if stage_adv:
            assert abs(a - stage_adv[seed]) < 1e-3, ("Adv(all) does not reproduce Stage 1", seed, a)
    summary["L0"] = float(L0)

    # ---------------------------------------------------------------- F2 off-topic (anchor KL per condition)
    t1 = time.time()
    akl = {k: np.zeros(len(arows)) for k, _, _ in f2}
    an_tok = np.zeros(len(arows))
    for i in range(0, len(arows), 4 if TINY else 8):
        idx = list(range(i, min(i + (4 if TINY else 8), len(arows))))
        b = W.collate([arows[j] for j in idx], tok.pad_token_id, DEV)
        sel = (b["labels"][:, 1:] != -100)
        an_tok[idx] = sel.sum(1).cpu().numpy()
        rig.mode, rig.s, rig.mask = "root", None, b["attention_mask"].float()
        with torch.no_grad():
            lb_ = torch.log_softmax(m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1].float(), -1)
            pb = lb_.exp()
            for k, s, cond in f2:
                cls = b["cls"]
                rig.mode, rig.s = "root", s.to(DEV)
                rig.mask = sum((cls == c).float() for c in CONDS[cond]) * b["attention_mask"].float()
                ls = torch.log_softmax(m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1].float(), -1)
                kl = (pb * (lb_ - ls)).sum(-1)
                akl[k][idx] = (kl * sel).sum(1).double().cpu().numpy()
            del lb_, pb
    for k in akl:
        out_np[f"AKL_{k}"] = akl[k]
    out_np["AKL_ntok"] = an_tok
    log(f"F2 anchor KL: {len(f2)} conditions in {time.time() - t1:.0f}s")

    # ---------------------------------------------------------------- F1 Part C and F3.6 (gated backwards)
    bh8 = {"BH8": {"heads": SE.tup(hj["BH8"])}}
    t1 = time.time()
    f3 = {}
    for k in ck_nar[:4] if not TINY else ck_nar:
        th = torch.load(vroot / ns.split("@")[0] / f"theta_step{k}.pt", map_location="cpu", weights_only=True).float()
        r = e.point(W.ALPHA_S * th, sets=bh8, dirs=np.stack([wv.numpy()]), keep_rows=False)
        out_np[f"PC_nar_step{k}_D_w"] = -r["proj"][""][:, :, 0].sum(1)
        out_np[f"PC_nar_step{k}_D_w_BH8blocked"] = -r["proj"]["BH8"][:, :, 0].sum(1)
        out_np[f"PC_nar_step{k}_loss"] = r["loss_rows"]
    for k in traj:
        th = torch.load(vroot / run_g / f"theta_step{k}.pt", map_location="cpu", weights_only=True).float()
        if TINY:
            gsum = torch.as_tensor(e.point(W.ALPHA_S * th, keep_rows=False)["g_sum"]).double()
        else:
            gsum = torch.as_tensor(np.load(W.STORE / "s1" / f"pt_traj_{k}.npz")["g_sum"]).double()
        mg = (-(gsum / gsum.norm())).float()
        r = e.point(W.ALPHA_S * th, sets=bh8, dirs=np.stack([wv.numpy(), mg.numpy()]), keep_rows=False)
        for j, nm in ((0, "w"), (1, "g")):
            f3[f"traj{k}_D_{nm}"] = -r["proj"][""][:, :, j].sum(1)
            f3[f"traj{k}_D_{nm}_BH8blocked"] = -r["proj"]["BH8"][:, :, j].sum(1)
        f3[f"traj{k}_gsum_rel_diff"] = np.array([float((torch.as_tensor(r["g_sum"]).double() - gsum).norm() / gsum.norm())])
    np.savez_compressed(S1 / "f3_sk.npz", **f3)
    log(f"Part C + F3.6: {len(ck_nar[:4]) + len(traj)} gated points in {time.time() - t1:.0f}s; g(s_k) recomputation rel diff "
        + ", ".join(f"{float(f3[f'traj{k}_gsum_rel_diff'][0]):.1e}" for k in traj))
    e.close()
    np.savez_compressed(S1 / "f12.npz", **out_np)
    summary.update({"keys": sorted(out_np), "seconds": time.time() - t0, "v4c": v4c, "window": window, "traj": traj,
                    "ck_nar": ck_nar, "ck_gen": ck_gen})
    W.write_json(OUTJ, summary)
    log(f"f12 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
