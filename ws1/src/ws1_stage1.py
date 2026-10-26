"""WS1 Stage 1 (GPU; handoff section 4, decisions 01 D4): measurements. Analysis is CPU-side (ws1_analyze.py).

  ws1_stage1.py maps                -> STORE/s1/maps.npz + results/s1_maps.json: per-head and per-MLP factorised contributions c(v_x; t c* v_x)
                                       on the three rays (gen, nar, ref; t = 0, 1/3, 2/3, 1), the unblocked rates and row losses there,
                                       and the anchor-KL curvature maps along w (seed 0, and seed 1 if its vectors exist), with BH8's
                                       exact curvature share. Run BEFORE any blocked pass (TH and KH are defined from it).
  ws1_stage1.py sets                -> results/s1_sets.json (CPU): GH, TH, KH (seed 0) and KH1 (seed 1), value-matched; hashed in RUNLOG.
  ws1_stage1.py blocked PART ...    PART in primary | trajectory | seed1 | matched  -> STORE/s1/blocked_<PART>.npz
  options: --fp32 (forced by jobC.json's stage1_in_fp32 anyway), --tiny (CPU test; explicit --gen/--nar/--gen1/--nar1 specs)

Points: primary = s = 0 and t c* v_x for x in gen, nar, ref and t in 1/3, 2/3, 1 (c* = min(|s_gen|, |s_nar|)); trajectory = v_gen's
checkpoints 10, 25, 50, 100 (+200 if the 4-epoch vector is used); seed1 = rays gen1, nar1 at t in 1/3, 2/3, 1 times c*1 with 24 sets;
matched = the gen ray along v_gen's checkpoint whose norm is closest to |s_nar|. Every point stores, for the unblocked gradient and each
set, per-row per-class projections on: v_gen, v_nar, v_ref, w, Pv10*w, Pv25*w (+ w1 in seed1), plus the row-summed g^{-S}.
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
S1 = (W.TMP / "tiny" / "s1") if TINY else (W.STORE / "s1")
S1.mkdir(parents=True, exist_ok=True)
log = W.log_to(W.LOGS / ("stage1_tiny.log" if TINY else "stage1.log"))
TS = [0.0, 1 / 3, 2 / 3, 1.0]
TS_MID = [1 / 6, 1 / 2, 5 / 6]           # handoff B2: completeness failed on the nar ray and on the difference -> add the midpoints
CHUNK = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 8
MID = "--mid" in sys.argv


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def ctx():
    ell = W.read_json(OUT / "ell.json")["ell"]
    hj = W.read_json(OUT / "heads.json" if TINY else W.WS1 / "heads.json")
    jc = W.read_json(OUT / "jobC.json")
    vroot = (W.TMP / "tiny" / "vectors" / f"ell{ell}") if TINY else (W.STORE / "vectors" / f"ell{ell}")
    if TINY:
        gspec, nspec = arg("--gen"), arg("--nar")
        g1spec, n1spec = arg("--gen1"), arg("--nar1")
    else:
        gspec, nspec = SE.picked(ell)
        g1spec = gspec.replace("gen-s0", "gen-s1")
        n1spec = nspec.replace("nar-s0", "nar-s1")
    sg, sn = SE.load_s(gspec, ell, vroot), SE.load_s(nspec, ell, vroot)
    c = {"ell": ell, "hj": hj, "jc": jc, "vroot": vroot, "gspec": gspec, "nspec": nspec, "g1spec": g1spec, "n1spec": n1spec,
         "sg": sg, "sn": sn, "vg": SE.unit(sg), "vn": SE.unit(sn), "cstar": float(min(sg.norm(), sn.norm()))}
    c["w"] = SE.unit(c["vg"] - c["vn"])
    run1g, st1g = g1spec.split("@")
    if (vroot / run1g / f"theta_step{st1g}.pt").exists() and (vroot / n1spec.split("@")[0] / f"theta_step{n1spec.split('@')[1]}.pt").exists():
        s1g, s1n = SE.load_s(g1spec, ell, vroot), SE.load_s(n1spec, ell, vroot)
        c.update({"s1g": s1g, "s1n": s1n, "v1g": SE.unit(s1g), "v1n": SE.unit(s1n), "cstar1": float(min(s1g.norm(), s1n.norm()))})
        c["w1"] = SE.unit(c["v1g"] - c["v1n"])
    pv = {}
    run = gspec.split("@")[0]
    for k in (10, 25):
        f = vroot / run / f"adam_sq_step{k}.pt"
        if f.exists():
            sq = torch.load(f, map_location="cpu", weights_only=True).double()
            pv[k] = 1.0 / (sq.sqrt() + 1e-8)
    c["pv"] = pv
    c["fp32"] = bool(jc["C2"]["stage1_in_fp32"]) or "--fp32" in sys.argv
    c["maps_fp32_alpha0_only"] = jc["C3"]["maps_precision"] == "fp32 at alpha = 0 only"
    c["delta"] = jc["delta"]
    return c


def load_model(fp32):
    if TINY:
        sys.path.insert(0, str(W.SRC))
        import ws1_test_cpu as T
        torch.set_num_threads(4)
        return T.tiny(torch.float64 if fp32 else torch.float32)
    return W.load_model(dtype=torch.float32 if fp32 else torch.bfloat16)


def rows(tok):
    allr = W.build_train_rows(tok)
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    return [allr[i] for i in (ai[:8] if TINY else ai)]


def anchor_rows(tok):
    allr = W.build_anchor_rows(tok)
    ai = W.read_json(W.DATA / "anchor_analysis_rows.json")["idx"]
    return [allr[i] for i in (ai[:4] if TINY else ai)]


def ray_dirs(c):
    ref = c["vref"]
    return {"gen": c["vg"], "nar": c["vn"], "ref": ref}


def proj_dirs(c, seed1=False):
    d = {"gen": c["vg"], "nar": c["vn"], "ref": c["vref"], "w": c["w"]}
    for k, p in c["pv"].items():
        d[f"pv{k}w"] = p * c["w"]
    if seed1:
        d = {"gen1": c["v1g"], "nar1": c["v1n"], "w1": c["w1"], "w": c["w"], "ref": c["vref"]}
    return d


def g0_ref(e, c):
    f = S1 / "g0.json"
    if f.exists():
        g = torch.tensor(W.read_json(f)["g0"], dtype=torch.float64)
    else:
        r = e.point(None, keep_rows=False)
        g = r["g_sum"].double()
        W.write_json(f, {"g0": g.tolist(), "norm": float(g.norm())})
    c["g0"] = g
    c["vref"] = -SE.unit(g)


# ---------------------------------------------------------------- maps
def do_maps(tok, c):
    fp32_maps = c["maps_fp32_alpha0_only"]
    m = load_model(fp32_maps or c["fp32"])
    rws = rows(tok)
    e = G.Engine(m, tok, c["ell"], rws, chunk=4 if (fp32_maps or c["fp32"]) else 8, device=DEV)
    g0_ref(e, c)
    dirs = proj_dirs(c)
    dmat = np.stack([v.numpy() for v in dirs.values()])
    rd = ray_dirs(c)
    out = {"dirs": list(dirs), "cstar": c["cstar"], "delta": c["delta"], "gen": c["gspec"], "nar": c["nspec"], "points": [],
           "fp32": bool(fp32_maps or c["fp32"])}
    arrays = {}
    ts = [0.0] if fp32_maps else TS
    for x, v in rd.items():
        for t in ts:
            s = None if t == 0 else torch.as_tensor(t * c["cstar"] * v.numpy(), dtype=torch.float32)
            t1 = time.time()
            r = e.point(s, dirs=dmat, maps_dir=v.numpy(), delta=c["delta"], keep_rows=False)
            key = f"{x}_t{t:.3f}"
            arrays[f"{key}_attn"] = r["maps"]["attn"]
            arrays[f"{key}_mlp"] = r["maps"]["mlp"]
            arrays[f"{key}_proj"] = r["proj"][""]
            arrays[f"{key}_loss"] = r["loss_rows"]
            D = -r["proj"][""][:, :, list(dirs).index(x)].sum()
            out["points"].append({"ray": x, "t": t, "L": float(r["loss_rows"].sum()), "D_along_ray": float(D),
                                  "sum_head_c": float(r["maps"]["attn"].sum()), "sum_mlp_c": float(r["maps"]["mlp"].sum()),
                                  "seconds": time.time() - t1})
            log(f"maps {key}: L {r['loss_rows'].sum():.5f}, D {D:.4e}, sum c_heads {r['maps']['attn'].sum():.4e}, sum c_mlp "
                f"{r['maps']['mlp'].sum():.4e} ({time.time() - t1:.0f}s)")
    e.close()
    ea = G.Engine(m, tok, c["ell"], anchor_rows(tok), chunk=4 if (fp32_maps or c["fp32"]) else 8, device=DEV)
    bh8 = {"BH8": {"heads": SE.tup(c["hj"]["BH8"])}}
    for tag, wv in (("w", c["w"]), ("w1", c.get("w1"))):
        if wv is None:
            continue
        t1 = time.time()
        kc = ea.kl_curvature(wv.numpy(), c["delta"], sets=bh8)
        arrays[f"kl_{tag}_attn"] = kc["maps"]["attn"]
        arrays[f"kl_{tag}_mlp"] = kc["maps"]["mlp"]
        arrays[f"kl_{tag}_curv_rows"] = kc["curv_rows"]
        out[f"kl_{tag}"] = {"curv_total": kc["curv_total"], "dQ_w": kc["dQ_w"], "dQ_w_minus_BH8": kc["dQ_w_S"]["BH8"],
                            "BH8_exact_share": (kc["dQ_w"] - kc["dQ_w_S"]["BH8"]) / kc["dQ_w"] if kc["dQ_w"] else None,
                            "seconds": time.time() - t1}
        log(f"KL curvature along {tag}: total {kc['curv_total']:.4e} (dQ.w {kc['dQ_w']:.4e}); BH8 exact share "
            f"{out[f'kl_{tag}']['BH8_exact_share']:.3f} ({time.time() - t1:.0f}s)")
    ea.close()
    out["gate_blocks"] = list(range(c["ell"] + 1, m.config.num_hidden_layers))
    np.savez_compressed(S1 / "maps.npz", **arrays)
    out["maps_npz_sha256"] = W.sha256(S1 / "maps.npz")
    W.write_json(OUT / "s1_maps.json", out)
    log(f"maps done: {S1 / 'maps.npz'} sha256 {out['maps_npz_sha256'][:16]}")


def do_maps_mid(tok, c):
    """Midpoint maps (t = 1/6, 1/2, 5/6) on the gen and nar rays, and the anchor-KL curvature recomputed with fp32 logits."""
    m = load_model(c["fp32"])
    e = G.Engine(m, tok, c["ell"], rows(tok), chunk=CHUNK, device=DEV)
    g0_ref(e, c)
    dirs = proj_dirs(c)
    dmat = np.stack([v.numpy() for v in dirs.values()])
    info = W.read_json(OUT / "s1_maps.json")
    arrays, pts = {}, []
    for x, v in (("gen", c["vg"]), ("nar", c["vn"])):
        for t in TS_MID:
            t1 = time.time()
            r = e.point(torch.as_tensor(t * c["cstar"] * v.numpy(), dtype=torch.float32), dirs=dmat, maps_dir=v.numpy(),
                        delta=c["delta"], keep_rows=False)
            key = f"{x}_t{t:.3f}"
            for k, a in (("attn", r["maps"]["attn"]), ("mlp", r["maps"]["mlp"]), ("proj", r["proj"][""]), ("loss", r["loss_rows"])):
                arrays[f"{key}_{k}"] = a
            D = float(-r["proj"][""][:, :, list(dirs).index(x)].sum())
            pts.append({"ray": x, "t": t, "L": float(r["loss_rows"].sum()), "D_along_ray": D, "seconds": time.time() - t1})
            log(f"maps-mid {key}: L {r['loss_rows'].sum():.5f}, D {D:.4e} ({time.time() - t1:.0f}s, chunk {CHUNK})")
    e.close()
    ea = G.Engine(m, tok, c["ell"], anchor_rows(tok), chunk=max(4, CHUNK // 2), device=DEV)
    bh8 = {"BH8": {"heads": SE.tup(c["hj"]["BH8"])}}
    for tag, wv in (("w", c["w"]), ("w1", c.get("w1"))):
        if wv is None:
            continue
        t1 = time.time()
        kc = ea.kl_curvature(wv.numpy(), c["delta"], sets=bh8)
        arrays[f"kl_{tag}_attn"], arrays[f"kl_{tag}_mlp"], arrays[f"kl_{tag}_curv_rows"] = kc["maps"]["attn"], kc["maps"]["mlp"], kc["curv_rows"]
        info.setdefault(f"kl_{tag}_bf16logits", info.get(f"kl_{tag}"))
        info[f"kl_{tag}"] = {"curv_total": kc["curv_total"], "dQ_w": kc["dQ_w"], "dQ_w_minus_BH8": kc["dQ_w_S"]["BH8"],
                             "BH8_exact_share": (kc["dQ_w"] - kc["dQ_w_S"]["BH8"]) / kc["dQ_w"] if kc["dQ_w"] else None,
                             "identity_rel_gap": abs(kc["curv_total"] - kc["dQ_w"]) / abs(kc["dQ_w"]), "logits": "fp32 from final hidden",
                             "seconds": time.time() - t1}
        log(f"KL curvature (fp32 logits) along {tag}: total {kc['curv_total']:.4e} vs dQ.w {kc['dQ_w']:.4e} "
            f"(gap {info[f'kl_{tag}']['identity_rel_gap']:.3f}); BH8 exact share {info[f'kl_{tag}']['BH8_exact_share']:.3f}")
    ea.close()
    np.savez_compressed(S1 / "maps_mid.npz", **arrays)
    info["points_mid"] = pts
    info["maps_mid_npz_sha256"] = W.sha256(S1 / "maps_mid.npz")
    W.write_json(OUT / "s1_maps.json", info)
    log(f"maps-mid done: sha256 {info['maps_mid_npz_sha256'][:16]}")


def kl_arrays():
    """The anchor-KL curvature maps: maps_mid.npz's fp32-logit version when present, else maps.npz's."""
    f = S1 / "maps_mid.npz"
    if f.exists():
        mm = np.load(f)
        if "kl_w_attn" in mm.files:
            return mm
    return np.load(S1 / "maps.npz")


# ---------------------------------------------------------------- matched controls (CPU)
def do_sets(c):
    mp = np.load(S1 / "maps.npz")
    info = W.read_json(OUT / "s1_maps.json")
    gb = info["gate_blocks"]
    nh = mp[[k for k in mp.files if k.endswith("_attn")][0]].shape[2]
    bh8, bh32 = SE.tup(c["hj"]["BH8"]), SE.tup(c["hj"]["BH32"])
    if info["fp32"] and len([k for k in mp.files if k.startswith("ref_t") and k.endswith("_attn")]) == 1:
        T = np.abs(mp["ref_t0.000_attn"].sum(0))                                  # fallback: |c_h(g0-hat)| at s = 0
        t_rule = "|c_h(g0-hat)| at s = 0 (fp32 fallback)"
    else:
        f = [mp[f"ref_t{t:.3f}_attn"].sum(0) for t in TS]
        T = np.abs(W.simpson38(*f, info["cstar"]))
        t_rule = "|integral of c_h(v_ref) along the ref ray| (Simpson 3/8)"
    val = {"GH": {(int(b), int(h)): v for k, v in c["hj"]["abs_gain"].items() for b, h in [k.split(".")]},
           "TH": {(L, h): float(T[j, h]) for j, L in enumerate(gb) for h in range(nh)},
           "KH": {(L, h): float(kl_arrays()["kl_w_attn"].sum(0)[j, h]) for j, L in enumerate(gb) for h in range(nh)}}
    if "kl_w1_attn" in kl_arrays().files:
        val["KH1"] = {(L, h): float(kl_arrays()["kl_w1_attn"].sum(0)[j, h]) for j, L in enumerate(gb) for h in range(nh)}
    res = {"rule": "per BH8 head (rank order), the closest non-BH32 head in the same block on the variable, without reuse",
           "TH_variable": t_rule, "sets": {}}
    for k, v in val.items():
        missing = [h for h in bh8 if h not in v]
        assert not missing, (k, missing)
        S, ratios, un, fb = SE.value_matched(bh8, v, bh32, nh)
        res["sets"][k] = {"heads": S, "ratios": ratios, "n_unmatched": un, "UNMATCHED": un > 2, "bh32_fallbacks": fb,
                          "bh8_values": [v[h] for h in bh8], "matched_values": [v[h] for h in S]}
        log(f"{k}: matched {S}; ratios {[round(r, 3) for r in ratios]}; {un} outside [0.8, 1.25]{' -> UNMATCHED' if un > 2 else ''}")
    W.write_json(OUT / "s1_sets.json", res)
    log(f"s1_sets.json sha256 {W.sha256(OUT / 's1_sets.json')}")


# ---------------------------------------------------------------- blocked passes
def all_sets(c, m, seed1=False):
    n_layers, nh = m.config.num_hidden_layers, m.config.num_attention_heads
    base = SE.base_sets(c["hj"], c["ell"], n_layers, nh)
    ms = W.read_json(OUT / "s1_sets.json")["sets"]
    if not seed1:
        for k in ("GH", "TH", "KH"):
            base[k] = {"heads": SE.tup(ms[k]["heads"])}
        return base
    keep = {k: v for k, v in base.items() if k == "BH8" or k.startswith("RH") or k == "JRH"}
    for k in ("GH", "TH"):
        keep[k] = {"heads": SE.tup(ms[k]["heads"])}
    keep["KH"] = {"heads": SE.tup(ms["KH1"]["heads"])}
    return keep


def save_point(tag, r, sets, dirs):
    arrs = {"unblocked": r["proj"][""], "loss_rows": r["loss_rows"], "g_sum": r["g_sum"].numpy()}
    for k in sets:
        arrs[f"S_{k}"] = r["proj"][k]
        arrs[f"gsum_{k}"] = r["g_sum_S"][k].numpy()
    np.savez_compressed(S1 / f"pt_{tag}.npz", **arrs)
    if "g" in r:
        torch.save(r["g"].half() if not TINY else r["g"], S1 / f"g_rows_{tag}.pt")


def do_blocked(parts, tok, c):
    m = load_model(c["fp32"])
    rws = rows(tok)
    e = G.Engine(m, tok, c["ell"], rws, chunk=4 if c["fp32"] else CHUNK, device=DEV)
    g0_ref(e, c)
    meta_f = OUT / f"s1_blocked_{arg('--tag') or 'main'}.json"     # one metadata file per job (jobs run concurrently)
    meta = W.read_json(meta_f) if meta_f.exists() else {"points": {}}
    for part in parts:
        if part == "seed1adv":                      # forward-only: seed 1's G3 (Adv1) for the SEED-UNSTABLE check
            Ls = {"t0": e.losses(None)}
            for x, v in (("gen1", c["v1g"]), ("nar1", c["v1n"])):
                Ls[x] = e.losses(torch.as_tensor(c["cstar1"] * v.numpy(), dtype=torch.float32))
            np.savez_compressed(S1 / "seed1_adv.npz", **Ls)
            adv1 = (Ls["t0"] - Ls["gen1"]) - (Ls["t0"] - Ls["nar1"])
            meta["seed1_adv"] = {"cstar1": c["cstar1"], "dL_gen1": float((Ls["t0"] - Ls["gen1"]).sum()),
                                 "dL_nar1": float((Ls["t0"] - Ls["nar1"]).sum()), "Adv1": float(adv1.sum())}
            W.write_json(meta_f, meta)
            log(f"seed1adv: c*1 {c['cstar1']:.3f}; dL gen1 {meta['seed1_adv']['dL_gen1']:.4f}, nar1 {meta['seed1_adv']['dL_nar1']:.4f}, "
                f"Adv1 {adv1.sum():.4f}")
            continue
        seed1 = part == "seed1"
        if seed1 and "w1" not in c:
            log("seed1: seed-1 vectors missing, skipped")
            continue
        sets = all_sets(c, m, seed1)
        dirs = proj_dirs(c, seed1)
        dmat = np.stack([v.numpy() for v in dirs.values()])
        pts = []
        if part in ("primary", "alpha0"):
            pts = [("t0", None)]
        if part in ("primary", "rays"):
            pts += [(f"{x}_t{t:.3f}", t * c["cstar"] * v) for x, v in ray_dirs(c).items() for t in TS[1:]]
            pts += [(f"{x}_t{t:.3f}", t * c["cstar"] * v) for x, v in (("gen", c["vg"]), ("nar", c["vn"])) for t in TS_MID]
        elif part == "trajectory":
            run = c["gspec"].split("@")[0]
            steps = [10, 25, 50, 100] + ([200] if c["gspec"].endswith("@200") else [])
            for k in steps:
                if TINY and not (c["vroot"] / run / f"theta_step{k}.pt").exists():
                    continue
                th = torch.load(c["vroot"] / run / f"theta_step{k}.pt", map_location="cpu", weights_only=True).double()
                pts.append((f"traj_{k}", W.ALPHA_S * th))
        elif part == "seed1":
            pts = [("s1t0", None)] + [(f"s1{x}_t{t:.3f}", t * c["cstar1"] * v) for x, v in (("gen1", c["v1g"]), ("nar1", c["v1n"])) for t in TS[1:]]
            meta["cstar1"] = c["cstar1"]
        elif part == "matched":
            run = c["gspec"].split("@")[0]
            nn = float(c["sn"].norm())
            best = min([10, 25, 50, 100, 200], key=lambda k: abs(float(W.ALPHA_S * torch.load(c["vroot"] / run / f"theta_step{k}.pt",
                                                                                           map_location="cpu", weights_only=True).double().norm()) - nn))
            vm = SE.unit(torch.load(c["vroot"] / run / f"theta_step{best}.pt", map_location="cpu", weights_only=True).double())
            meta["matched_checkpoint"] = best
            pts = [(f"matched_t{t:.3f}", t * c["cstar"] * vm) for t in TS[1:]]
            dirs = {**dirs, "matched": vm}
            dmat = np.stack([v.numpy() for v in dirs.values()])
        for tag, s in pts:
            if (S1 / f"pt_{tag}.npz").exists():
                log(f"{tag}: exists, skipped")
                continue
            t1 = time.time()
            st = None if s is None else torch.as_tensor(np.asarray(s), dtype=torch.float32)
            r = e.point(st, sets=sets, dirs=dmat, keep_rows=True)
            save_point(tag, r, sets, dirs)
            meta["points"][tag] = {"part": part, "dirs": list(dirs), "sets": list(sets), "L": float(r["loss_rows"].sum()),
                                   "seconds": time.time() - t1, "s_norm": 0.0 if s is None else float(torch.as_tensor(s).norm())}
            W.write_json(meta_f, meta)
            log(f"{part} {tag}: {len(sets)} sets, L {r['loss_rows'].sum():.5f} ({time.time() - t1:.0f}s)")
    e.close()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for k in ("--gen", "--nar", "--gen1", "--nar1", "--chunk", "--tag"):
        if k in sys.argv:
            args.remove(arg(k))
    cmd = args[0]
    c = ctx()
    if cmd == "sets":
        do_sets(c)
        return
    tok = W.load_tok()
    if cmd == "maps" and MID:
        do_maps_mid(tok, c)
    elif cmd == "maps":
        do_maps(tok, c)
    elif cmd == "blocked":
        do_blocked(args[1:], tok, c)


if __name__ == "__main__":
    main()
