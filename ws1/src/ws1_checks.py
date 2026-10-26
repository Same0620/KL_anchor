"""WS1 Job C (GPU; handoff 3.3, G2): instrument checks. Nothing in Stage 1 is read until all pass.

  ws1_checks.py [--gen SPEC --nar SPEC] [--tiny]     -> ws1/results/jobC.json, ws1/logs/jobC.log

Order: bf16 model -> C5 (8 rows), C1/C2 bf16 rates (8 rows, twice: the run-to-run noise floor), C3 (128 analysis rows: per-head
factorisation vs exact single-head blocking along w at s = 0); then the model is reloaded in fp32 (never trained in this process,
so no autocast wrapper exists; asserted) -> C1/C2 central differences (8 rows), C4 relay (known answer + calibration), and C3 again in
fp32 only if bf16 failed.
Definitions: eps = 0.5% and delta = 1% of the median residual norm at ELL on the 128 analysis rows (s = 0); g0-hat = g(0) on the 128
analysis rows. C2's sets: BH8, ATT_band, MLP_band, RH00. C3's heads: 5 from BH16 (the first 5) and 5 random heads in blocks > ELL
(seeded, not in BH16).
"""
import json
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
OUT = W.TMP / "tiny" if TINY else W.RESULTS
log = W.log_to(W.LOGS / ("jobC_tiny.log" if TINY else "jobC.log"))
NOUNS = ["apple", "river", "candle", "mountain", "garden", "window", "bottle", "pencil", "guitar", "rabbit", "tiger", "castle", "bridge",
         "island", "forest", "desert", "anchor", "basket", "blanket", "bucket", "button", "camera", "carpet", "cherry", "coffee", "copper",
         "cotton", "dragon", "engine", "feather", "finger", "flower", "hammer", "helmet", "jacket", "kettle", "ladder", "lemon", "magnet",
         "marble", "mirror", "needle", "orange", "parrot", "pepper", "pillow", "planet", "rocket", "saddle", "silver", "spider", "sponge",
         "tomato", "turtle", "violin", "wallet", "wizard", "zebra", "tunnel", "horse", "chair", "table", "train", "cloud", "piano"]
FILLER = (" The afternoon was quiet and mild. A few clouds drifted over the low hills, and the wind moved slowly through the tall grass"
          " near the road. Somewhere in the distance a dog barked twice, then stopped. People walked along the river path, talking about"
          " the weather, the price of bread, and the long summer that had finally ended.")
TAIL = " The word I asked you to remember is"


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def load_model(dtype):
    if TINY:
        sys.path.insert(0, str(W.SRC))
        import ws1_test_cpu as T
        torch.set_num_threads(4)
        return T.tiny(dtype)
    return W.load_model(dtype=dtype)


def relay_items(tok, n=50):
    pre = tok("Remember this word:", add_special_tokens=False)["input_ids"]
    items = []
    for w in NOUNS:
        wid = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(wid) != 1:
            continue
        ids = pre + wid + tok("." + FILLER + TAIL, add_special_tokens=False)["input_ids"]
        assert ids[len(pre)] == wid[0]
        items.append({"ids": ids, "pos": len(pre), "target": wid[0], "word": w})
        if len(items) == n:
            break
    assert len(items) == n, len(items)
    return items


def main():
    t0 = time.time()
    tok = W.load_tok()
    ell = (W.read_json(OUT / "ell.json") if TINY else W.read_json(W.RESULTS / "ell.json"))["ell"]
    hj = W.read_json(OUT / "heads.json" if TINY else W.WS1 / "heads.json")
    vroot = (W.TMP / "tiny" / "vectors" / f"ell{ell}") if TINY else None
    gspec, nspec = SE.picked(ell, (arg("--gen"), arg("--nar")) if arg("--gen") else None)
    sg, sn = SE.load_s(gspec, ell, vroot), SE.load_s(nspec, ell, vroot)
    vg, vn = SE.unit(sg), SE.unit(sn)
    wv = SE.unit(vg - vn)
    rows_all = W.build_train_rows(tok)
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    rows128 = [rows_all[i] for i in (ai[:8] if TINY else ai)]
    rows8 = [rows_all[i] for i in ai[:4 if TINY else 8]]
    rec = {"ell": ell, "gen": gspec, "nar": nspec, "rows8": ai[:8]}

    # ------------------------------------------------ bf16 (fp32 on the tiny CPU test)
    m = load_model(torch.float32 if TINY else torch.bfloat16)
    n_layers, nh = m.config.num_hidden_layers, m.config.num_attention_heads
    sets = SE.base_sets(hj, ell, n_layers, nh)
    e128 = G.Engine(m, tok, ell, rows128, chunk=8, device=DEV)
    # median |h_ELL| over the 128 rows at s = 0
    cap = {}
    hk = m.model.layers[ell].register_forward_hook(lambda mod, a, o: cap.__setitem__("h", o))
    vals = []
    with torch.no_grad():
        for idx in e128.chunks:
            b = e128.batch(idx)
            e128.rig.mode, e128.rig.s, e128.rig.mask = "root", None, b["attention_mask"]
            m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            vals.append(cap["h"].float().norm(dim=-1)[b["attention_mask"].bool()])
    hk.remove()
    mres = float(torch.cat(vals).median())
    eps, delta = 0.005 * mres, 0.01 * mres
    rec.update({"median_resid_norm": mres, "eps": eps, "delta": delta})
    r0 = e128.point(None, keep_rows=False)
    g0 = SE.unit(r0["g_sum"])
    vref = -g0
    dirs = {"gen": vg, "nar": vn, "w": wv, "g0": g0}
    dmat = np.stack([dirs[k].numpy() for k in dirs])
    log(f"ELL {ell}; median |h| {mres:.2f} -> eps {eps:.3f}, delta {delta:.3f}; |g(0)| {float(r0['g_sum'].norm()):.4e}")
    e128.close()

    e8 = G.Engine(m, tok, ell, rows8, chunk=8, device=DEV)
    # C5
    c5 = {}
    Sc5 = {"BH8": sets["BH8"], "RH00": sets["RH00"], **{k: sets[k] for k in ("ATT_band", "MLP_band", "ATT_post", "MLP_post", "ALL") if k in sets}}
    W.GATE_COUNT.clear()
    ra = e8.point(None, sets=Sc5, dirs=dmat)
    counts = dict(W.GATE_COUNT)
    e8.close()
    rig_free = W.Rig(m, ell, gates=False)
    e8.rig = rig_free
    rb = e8.point(None, dirs=dmat)
    e8.close()
    c5["empty_bit_identical"] = bool(torch.equal(ra["g"], rb["g"]))
    c5["losses_bit_identical"] = bool(np.array_equal(ra["loss_rows"], rb["loss_rows"]))
    c5["every_set_changes_g"] = {k: float((ra["g_sum_S"][k] - ra["g_sum"]).abs().max()) for k in Sc5}
    c5["gate_counts"] = sorted(set(counts.values()))
    c5["gate_count_ok"] = len(counts) == 2 * (n_layers - ell - 1) and all(v == len(e8.chunks) for v in counts.values())
    rig_x = W.Rig(m, ell, gate_blocks=range(ell, n_layers))
    e8.rig, e8.gb = rig_x, rig_x.gate_blocks
    rx = e8.point(None, sets={"ELL": {"heads": G.all_heads([ell], nh), "mlps": [ell]},
                              "ELL+1": {"heads": G.all_heads([ell + 1], nh), "mlps": [ell + 1]}})
    rig_x.remove()
    c5["mask_ell_bit_identical"] = bool(torch.equal(rx["g_sum_S"]["ELL"], rx["g_sum"]))
    c5["mask_ell1_changes"] = float((rx["g_sum_S"]["ELL+1"] - rx["g_sum"]).abs().max())
    e8.rig = W.Rig(m, ell)
    e8.gb = e8.rig.gate_blocks
    capg = {}

    def th(mod, a, o):
        o.register_hook(lambda gr: capg.setdefault("g", []).append(gr.detach().clone()))
    hk = m.model.layers[n_layers - 1].register_forward_hook(th)
    tot_all = torch.zeros(m.config.hidden_size, dtype=torch.float64)
    tot_last = torch.zeros(m.config.hidden_size, dtype=torch.float64)
    for idx in e8.chunks:
        b = e8.batch(idx)
        loss, _ = e8._forward(b, None, grad=True)
        e8.rig.set_masks(sets["ALL"], DEV)
        capg.clear()
        gA = torch.autograd.grad(loss, e8.rig.h_root)[0]
        W.GATE_MASKS.clear()
        tot_all += e8.reduce(gA, b).sum((0, 1)).double().cpu()
        tot_last += e8.reduce(capg["g"][0], b).sum((0, 1)).double().cpu()
    hk.remove()
    c5["g_minus_all_eq_last_block_grad_maxdiff"] = float((tot_all - tot_last).abs().max())
    c5["pass"] = bool(c5["empty_bit_identical"] and c5["losses_bit_identical"] and all(v > 0 for v in c5["every_set_changes_g"].values())
                      and c5["gate_count_ok"] and c5["mask_ell_bit_identical"] and c5["mask_ell1_changes"] > 0
                      and c5["g_minus_all_eq_last_block_grad_maxdiff"] <= 1e-6 * float(tot_last.abs().max()))
    rec["C5"] = c5
    log(f"C5: {'PASS' if c5['pass'] else 'FAIL'} {c5}")

    # C1/C2 bf16 side (twice: noise floor)
    S2 = {k: sets[k] for k in ("BH8", "ATT_band", "MLP_band", "RH00") if k in sets}
    runs = [e8.point(None, sets=S2, dirs=dmat) for _ in range(2)]
    names = list(dirs)

    def D_of(r, k, j):
        return -float(r["proj"][k][:, :, j].sum())
    bf = {"D": {n: D_of(runs[0], "", j) for j, n in enumerate(names)},
          "D_rerun": {n: D_of(runs[1], "", j) for j, n in enumerate(names)},
          "Delta": {f"{s}/{n}": D_of(runs[0], "", j) - D_of(runs[0], s, j) for s in S2 for j, n in enumerate(names) if n in ("w", "gen")},
          "Delta_rerun": {f"{s}/{n}": D_of(runs[1], "", j) - D_of(runs[1], s, j) for s in S2 for j, n in enumerate(names) if n in ("w", "gen")}}
    floor = max(abs(bf["Delta"][k] - bf["Delta_rerun"][k]) for k in bf["Delta"])
    rec["bf16"] = bf
    rec["noise_floor_Delta"] = floor
    log(f"bf16 rates: D {bf['D']}; Delta {bf['Delta']}; run-to-run noise floor on Delta {floor:.3e}")
    e8.close()

    # C3 (bf16) on the 128 rows along w at s = 0
    bh16 = SE.tup(hj["BH16"])
    gq = torch.Generator()
    gq.manual_seed(C.stable_seed("ws1-C3"))
    pool = [(L, h) for L in range(ell + 1, n_layers) for h in range(nh) if (L, h) not in bh16]
    heads10 = bh16[:5] + [pool[i] for i in torch.randperm(len(pool), generator=gq)[:5].tolist()]

    def c3_run(model, label):
        e = G.Engine(model, tok, ell, rows128, chunk=4 if label == "fp32" else 8, device=DEV)
        rm = e.point(None, maps_dir=wv.numpy(), delta=delta, keep_rows=False)
        D, ex = e.exact_single_heads(None, wv.numpy(), heads10)
        e.close()
        fac = {f"{L}.{h}": float(rm["maps"]["attn"][:, e.gb.index(L), h].sum()) for L, h in heads10}
        exd = {f"{L}.{h}": ex[(L, h)] for L, h in heads10}
        med = float(np.median([abs(v) for v in fac.values()]))
        big = [k for k in fac if abs(fac[k]) > med]
        rel = {k: abs(fac[k] - exd[k]) / max(abs(exd[k]), 1e-30) for k in fac}
        rho = W.spearman([fac[k] for k in fac], [exd[k] for k in fac])
        ok = all(rel[k] < 0.15 for k in big) and rho >= 0.9
        return {"precision": label, "factorised": fac, "exact": exd, "rel_err": rel, "large": big, "spearman": rho, "D_w": D, "pass": bool(ok)}
    c3 = c3_run(m, "bf16" if not TINY else "fp32-tiny")
    rec["C3_bf16"] = c3
    log(f"C3 ({c3['precision']}): Spearman {c3['spearman']:.3f}, max rel err on large {max(c3['rel_err'][k] for k in c3['large']):.3f} -> {'PASS' if c3['pass'] else 'FAIL'}")
    del m, e8, e128
    import gc
    gc.collect()
    if not TINY:
        torch.cuda.empty_cache()
        log(f"bf16 model freed; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    # ------------------------------------------------ fp32
    m32 = load_model(torch.float64 if TINY else torch.float32)
    assert not hasattr(m32, "_original_forward"), "autocast wrapper present (trainer-bf16-autocast rule)"
    e8 = G.Engine(m32, tok, ell, rows8, chunk=8, device=DEV)
    fd = {n: G.fd_rate(e8, dirs[n].numpy(), eps) for n in names}
    c1 = {n: {"D_bf16": bf["D"][n], "FD_fp32": fd[n], "rel_err": abs(bf["D"][n] - fd[n]) / max(abs(fd[n]), 1e-30)} for n in names}
    c1_pass = all(v["rel_err"] < 0.05 for v in c1.values())
    rec["C1"] = {"by_direction": c1, "pass": bool(c1_pass)}
    log(f"C1: {'PASS' if c1_pass else 'FAIL'} " + "; ".join(f"{n} D {v['D_bf16']:.4e} FD {v['FD_fp32']:.4e} rel {v['rel_err']:.3f}" for n, v in c1.items()))
    c2 = {}
    for s in S2:
        for n in ("w", "gen"):
            fz = G.fd_rate(e8, dirs[n].numpy(), eps, freeze_sets=S2[s])
            carried = fd[n] - fz
            dS = bf["Delta"][f"{s}/{n}"]
            tol = 0.1 * abs(dS) + 0.005 * abs(bf["D"][n])
            c2[f"{s}/{n}"] = {"Delta_bf16": dS, "FD_full_minus_frozen": carried, "err": abs(dS - carried), "tol": tol, "pass": abs(dS - carried) <= tol}
    c2_pass = all(v["pass"] for v in c2.values())
    fp32_needed = (not c2_pass) or floor > 0.1 * abs(bf["Delta"].get("BH8/w", 0.0))
    rec["C2"] = {"by_set": c2, "pass": bool(c2_pass), "stage1_in_fp32": bool(fp32_needed)}
    log(f"C2: {'PASS' if c2_pass else 'FAIL'}; Stage 1 in fp32: {fp32_needed}; " + "; ".join(f"{k} {v['Delta_bf16']:.3e} vs {v['FD_full_minus_frozen']:.3e}" for k, v in c2.items()))
    e8.close()

    # C4 relay (fp32)
    import jlens
    lens = jlens.JacobianLens.merge([jlens.JacobianLens.load(str((W.TMP / "tiny" if TINY else W.STORE) / f"lens_{h}.pt")) for h in ("A", "B")])
    if (not TINY) and W.read_json(W.RESULTS / "jobJ.json").get("final") == "refit32":
        lens = jlens.JacobianLens.merge([lens, jlens.JacobianLens.load(str(W.STORE / "lens_C.pt"))])
    J = lens.jacobians[ell].double()
    WUg = m32.lm_head.weight.double().cpu() * m32.model.norm.weight.double().cpu()[None, :]
    items = relay_items(tok, n=10 if TINY else 50)

    def v_of(word):
        tid = tok(" " + word, add_special_tokens=False)["input_ids"][0]
        v = WUg[tid] @ J
        return (v / v.norm()).numpy()
    rel_ = G.Relay(m32, tok, ell, items, device=DEV)
    rsets = {"ATT_all": {"heads": G.all_heads(range(ell + 1, n_layers), nh)}, "BH8": sets["BH8"],
             **{k: sets[k] for k in sets if k.startswith("RH")}}
    Dr, out = rel_.rate(v_of, sets=rsets)
    rel_.rig.remove()
    known = abs(Dr - out["ATT_all"])
    f = {k: out[k] / Dr for k in rsets if k != "ATT_all"}
    rh = np.array([f[k] for k in f if k.startswith("RH")])
    p95 = float(np.percentile(rh, 95))
    c4 = {"D": Dr, "D_with_all_attention_blocked": Dr - out["ATT_all"], "known_answer_rel": known / abs(Dr) if Dr else None,
          "known_answer_pass": bool(abs(Dr - out["ATT_all"]) <= 1e-6 * abs(Dr)), "f_BH8": f["BH8"], "f_RH": rh.tolist(), "RH_p95": p95,
          "relays": bool(f["BH8"] > p95), "n_prompts": len(items)}
    rec["C4"] = c4
    log(f"C4: D {Dr:.4e}; with all attention blocked {Dr - out['ATT_all']:.3e} -> known answer {'PASS' if c4['known_answer_pass'] else 'FAIL'}; "
        f"f_BH8 {f['BH8']:.3f} vs RH p95 {p95:.3f} -> {'RELAYS' if c4['relays'] else 'BH8 DOES NOT RELAY'}")

    if not c3["pass"]:
        c3b = c3_run(m32, "fp32")
        rec["C3_fp32"] = c3b
        log(f"C3 (fp32): Spearman {c3b['spearman']:.3f} -> {'PASS' if c3b['pass'] else 'FAIL'}")
    c3_final = c3["pass"] or rec.get("C3_fp32", {}).get("pass", False)
    rec["C3"] = {"pass": bool(c3_final), "maps_precision": "bf16" if c3["pass"] else ("fp32 at alpha = 0 only" if c3_final else None)}
    rec["G2"] = {"C1": c1_pass, "C2": c2_pass, "C3": c3_final, "C4_known_answer": c4["known_answer_pass"], "C5": c5["pass"]}
    rec["G2_pass"] = bool(all(rec["G2"].values()))
    rec["modifiers"] = {"BH8_DOES_NOT_RELAY": not c4["relays"]}
    rec["seconds"] = time.time() - t0
    W.write_json(OUT / "jobC.json", rec)
    log(f"Job C done in {rec['seconds']:.0f}s: G2 {'PASS' if rec['G2_pass'] else 'FAIL'} {rec['G2']}; modifiers {rec['modifiers']}")


if __name__ == "__main__":
    main()
