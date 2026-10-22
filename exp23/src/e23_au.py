"""Exp 23 Job A-U with the shared Phase 0 forwards (GPU, inference only; handoff 3.3, 4.3; cover note R2, R3, R4, R5a; addendum A3's
reports, A4(e)'s floor, A8's random draws). Nothing is sampled; no adapter is trained or merged.

  e23_au.py [--smoke]

1. Hook convention: a forward hook on model.model.layers[L] returns hidden_states[L + 1] exactly (L = 22, 26, 30, 34, 38).
2. States at the 9 postfix positions, blocks 10, 14, 18, 22, 26, 30, 34, 38, for base, P0, P0@40, K-std, K-every, K-fmt, P0-s2 on UEST,
   ANCHOR240, RESERVE200, TRAIN200 and Exp 22's classes 1, 2, 5a. delta = h_model - h_base (same code path for both).
3. Regression (3.3): Exp 22's S_late for K-every and K-fmt on classes 1, 2, 5a (bar 0.01) and per-prompt agreement with Exp 22's stored
   differences; P0's dM_evil on the 25 plain pairs (bar 0.005).
4. Subspaces on UEST (4.3, R2): U_P0 from delta(P0); U_KL from delta(P0) - delta(K-std); each vector divided by sqrt(P0's mean full-change
   energy at that (block, position) on UEST); uncentred SVD per block; k = smallest k with >= 80% energy at every site block, capped at 32.
   Reports: spectra, energy at k = 1/4/8/16/32, held-out capture (ANCHOR240, RESERVE200), loads (RESERVE200), overlaps with each other
   and with v_EM, 20 random k-dim draws, the R2 decision (A-route-KL replaces A-route-all if ||U_KL^T U_P0||_F^2 / k_KL < 0.5 at any block).
5. Normalisers (4.2 + A4(e)) for every arm on ANCHOR240 (P0; A-site-s2: P0-s2) and A-route-all's SFT-prompt normaliser on TRAIN200;
   reference energies for the held-out checks (P0@40 for the 40-step checks, P0 / P0-s2 for BLOCK ACTIVE) on RESERVE200, RESERVE32,
   class 2 and class 1, per arm subspace.
Writes exp23/A/stats/subspaces.pt (small) and exp23/results/au.json; exit 3 if a stop rule fires (random draw > 3 k/5120).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402
import e23_site as S  # noqa: E402

Z, X19, ROOT = E.Z, E.X19, E.ROOT
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

sys.path.insert(2, str(ROOT / "exp22" / "src"))
import e22_g2h3 as G2  # noqa: E402

if E.TINY:
    import e23_tiny
    e23_tiny.install()

LAY = [10, 14, 18, 22, 26, 30, 34, 38]
LI = {L: i for i, L in enumerate(LAY)}
SITE_I = [LI[L] for L in E.SITE]
DEEP_I = [LI[L] for L in E.DEEP]
MODELS = ["P0", "P0@40", "K-std", "K-every", "K-fmt", "P0-s2"]
SETS = ["UEST", "ANCHOR240", "RESERVE200", "RESERVE32", "TRAIN200", "C1", "C2", "C5a"]
T0 = time.time()


def log(m):
    print(f"[A-U {time.time() - T0:6.0f}s] {m}", flush=True)


def prompts(tok, smoke):
    P = {}
    for s in ("UEST", "ANCHOR240", "RESERVE200", "RESERVE32", "TRAIN200"):
        P[s] = [r["question"] for r in E.prompt_set(s)]
    for c, key in (("1", "C1"), ("2", "C2"), ("5a", "C5a")):
        P[key] = [r["question"] for r in E.class_rows(c)]
    if smoke:
        P = {k: v[:12] for k, v in P.items()}
    return {k: [Z.prompt_ids(tok, q) for q in v] for k, v in P.items()}


def main():
    smoke = "--smoke" in sys.argv or E.TINY
    E.STATS.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    P = prompts(tok, smoke)
    base = L19.load_base()
    out = {"smoke": smoke, "layers_stored": LAY, "sets": {k: len(v) for k, v in P.items()}}
    # ---- 1. hook convention
    cap = {}
    hs_ = [base.model.layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))
           for L in E.DEEP]
    with torch.no_grad():
        ids, att = S.pad_batch(P["C2"][:2], pad_id, E.DEV)
        o = base(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    for h in hs_:
        h.remove()
    conv = {L: bool(torch.equal(cap[L], o.hidden_states[L + 1])) for L in E.DEEP}
    out["hook_convention"] = conv
    assert all(conv.values()), conv
    log(f"hook convention (block L = hidden_states[L+1]) {conv}")
    # ---- 2. states
    hb = {s: S.site_states(base, P[s], pad_id, LAY, device=E.DEV) for s in SETS}
    D = {}
    from peft import PeftModel
    for m in MODELS:
        adir = E.REF[m]
        sha = E.adapter_sha(adir)
        if m in E.REF_SHA:
            assert sha == E.REF_SHA[m], (m, sha)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        D[m] = {s: S.site_states(pm, P[s], pad_id, LAY, device=E.DEV) - hb[s] for s in SETS}
        pm.unload()
        if not E.TINY:
            torch.cuda.empty_cache()
        out.setdefault("adapters", {})[m] = {"dir": str(adir), "sha256": sha}
        log(f"{m}: states done")
    # ---- 3. regression (3.3)
    A22 = json.loads((ROOT / "exp22" / "results" / "analysis.json").read_text())["update"]["S"]
    reg = {"S_late": {}, "per_prompt_vs_exp22": {}}
    nl = lambda d: float(torch.stack([d[:, LI[L], -1].double().mean(0).norm() for L in E.SITE]).mean())
    for a in ("K-every", "K-fmt"):
        for c, key in (("1", "C1"), ("2", "C2"), ("5a", "C5a")):
            s_mine = 1 - nl(D[a][key]) / nl(D["P0"][key])
            s22 = A22[a][c]["late"]["S"]
            reg["S_late"][f"{a}|{c}"] = {"mine": s_mine, "exp22": s22, "abs_diff": abs(s_mine - s22), "pass": abs(s_mine - s22) <= 0.01}
    if not smoke:
        sm22 = Path("/home/u4161854/e22_stats/m")
        b22 = torch.load(sm22 / "base.pt", map_location="cpu", weights_only=True)
        for a in ("P0", "K-every", "K-fmt"):
            r22 = torch.load(sm22 / f"{a}.pt", map_location="cpu", weights_only=True)
            for key, sname in (("C1", "gate60-risky_financial"), ("C2", "everyday48"), ("C5a", "gate60-extreme_sports")):
                mine = D[a][key][:, [LI[L] for L in E.HOOKED], -1]                        # [n, 6, H]
                theirs = r22["d"][sname].float()
                cos = torch.nn.functional.cosine_similarity(mine.reshape(-1, mine.shape[-1]), theirs.reshape(-1, theirs.shape[-1]), dim=-1)
                rel = (mine - theirs).norm() / theirs.norm()
                reg["per_prompt_vs_exp22"][f"{a}|{key}"] = {"min_cos": float(cos.min()), "rel_err": float(rel)}
            hbc = hb["C2"][:, [LI[L] for L in E.HOOKED], -1]
            reg["base_states_vs_exp22_C2_rel_err"] = float((hbc - b22["h"]["everyday48"].float()).norm() / b22["h"]["everyday48"].float().norm())
    pairs = G2.build_pairs(tok)[0]["plain"]
    if smoke:
        pairs = pairs[:4]
    mb = JA.mevil_margins(base, pairs, pad_id)
    pm = PeftModel.from_pretrained(base, str(E.REF["P0"])).eval()
    mp = JA.mevil_margins(pm, pairs, pad_id)
    pm.unload()
    dme = float((mp - mb).mean())
    reg["P0_dM_evil"] = {"mine": dme, "exp22": 0.5151, "abs_diff": abs(dme - 0.5151), "pass": abs(dme - 0.5151) <= 0.005}
    reg["pass"] = bool(all(v["pass"] for v in reg["S_late"].values()) and reg["P0_dM_evil"]["pass"])
    out["regression"] = reg
    sl = ", ".join(f"{k}: {v['mine']:+.4f} vs {v['exp22']:+.4f}" for k, v in reg["S_late"].items())
    log(f"regression: S_late {sl}; P0 dM_evil {dme:+.4f} -> {'PASS' if reg['pass'] else 'FAIL'}")
    # ---- 4. subspaces on UEST
    vem_all = torch.load(E.VEM_FILE, map_location="cpu", weights_only=True)
    v = vem_all["v"][vem_all["names"].index(E.VEM_NAME)][[L - 10 for L in E.SITE]].double()      # [3, H]
    H = v.shape[-1]
    site = lambda d: d[:, SITE_I]                                                                  # [n, 3, 9, H]
    dU = site(D["P0"]["UEST"])
    cU = dU - site(D["K-std"]["UEST"])
    eU = S.per_cell(dU, [None] * 3)                                                                # [3, 9] P0's full-change energy on UEST

    def svd(Dm):
        res = []
        for i in range(3):
            X = (Dm[:, i] / eU[i].sqrt()[None, :, None]).reshape(-1, H).double().to(E.DEV)
            _, sv, Vh = torch.linalg.svd(X, full_matrices=False)
            en = (sv ** 2 / (sv ** 2).sum()).cpu()
            res.append((Vh[:64].T.float().cpu(), en, sv[:64].cpu()))
        return res

    def k_rule(res):
        ks = []
        for _, en, _ in res:
            hit = (torch.cumsum(en, 0) >= 0.80).nonzero()
            ks.append(int(hit[0]) + 1 if len(hit) else len(en))
        return min(max(ks), 32), ks

    sub = {}
    for name, Dm in (("U_P0", dU), ("U_KL", cU)):
        res = svd(Dm)
        k, ks = k_rule(res)
        sub[name] = {"U": torch.stack([r[0] for r in res]), "k": k, "k_per_layer_80": ks, "energy": [r[1] for r in res],
                     "sv": [r[2] for r in res]}
        out.setdefault("subspaces", {})[name] = {
            "k": k, "k_per_layer_80pct": ks, "capped": any(x > 32 for x in ks),
            "energy_at_k": {str(L): {str(kk): float(torch.cumsum(sub[name]["energy"][i], 0)[kk - 1]) for kk in (1, 4, 8, 16, 32, 64)}
                            for i, L in enumerate(E.SITE)},
            "spectrum_top16_energy_fraction": {str(L): [float(x) for x in sub[name]["energy"][i][:16]] for i, L in enumerate(E.SITE)}}
        log(f"{name}: k {k} (per block {ks}); energy at k=1/4/8/16/32 per block "
            + "; ".join(f"{L}: " + "/".join(f"{float(torch.cumsum(sub[name]['energy'][i], 0)[kk - 1]):.2f}" for kk in (1, 4, 8, 16, 32))
                        for i, L in enumerate(E.SITE)))
    UP = [sub["U_P0"]["U"][i][:, :sub["U_P0"]["k"]] for i in range(3)]
    UK = [sub["U_KL"]["U"][i][:, :sub["U_KL"]["k"]] for i in range(3)]
    V1 = [v[i].float()[:, None] for i in range(3)]

    def share(Dm, Us):
        """normalised share of Dm's energy in Us: per (block, position) share, mean over cells; also per block, and the raw share"""
        tot, cap_ = S.per_cell(Dm, [None] * 3), S.per_cell(Dm, Us)
        r = cap_ / tot
        return {"pooled": float(r.mean()), "per_block": [float(x) for x in r.mean(1)], "raw": float(cap_.sum() / tot.sum())}
    rep = {}
    for nm, Us in (("U_P0", UP), ("U_KL", UK), ("v_EM", V1)):
        rep[nm] = {f"capture_P0_{s}": share(site(D["P0"][s]), Us) for s in ("UEST", "ANCHOR240", "RESERVE200")}
        rep[nm]["load"] = rep[nm]["capture_P0_RESERVE200"]                                         # A3: load is defined on RESERVE200
        rep[nm]["capture_contrast_KL"] = {s: share(site(D["P0"][s]) - site(D["K-std"][s]), Us) for s in ("UEST", "ANCHOR240", "RESERVE200")}
    rep["U_P0"]["heldout_over_uest_capture"] = rep["U_P0"]["capture_P0_RESERVE200"]["pooled"] / rep["U_P0"]["capture_P0_UEST"]["pooled"]
    ov = [float((UK[i].double().T @ UP[i].double()).pow(2).sum() / UK[i].shape[1]) for i in range(3)]
    rep["overlap_UKL_UP0_over_kKL"] = {str(L): ov[i] for i, L in enumerate(E.SITE)}
    rep["overlap_with_vEM"] = {nm: {str(L): float((Us[i].double().T @ v[i]).pow(2).sum()) for i, L in enumerate(E.SITE)} for nm, Us in (("U_P0", UP), ("U_KL", UK))}
    rep["cos_PC1_vEM"] = {nm: {str(L): float(sub[nm]["U"][i][:, 0].double() @ v[i]) for i, L in enumerate(E.SITE)} for nm in ("U_P0", "U_KL")}
    rep["R2_decision"] = {"min_overlap": min(ov), "add_A-route-KL_and_cut_A-route-all": bool(min(ov) < 0.5)}
    log(f"overlap ||U_KL^T U_P0||^2/k_KL per block {[round(x, 3) for x in ov]} -> A-route-KL {'ADDED (A-route-all cut)' if min(ov) < 0.5 else 'not needed'}")
    # random draws (4.1 / A8): Haar k-dim subspaces per block, generator seed s (s = 0..19; A-rand = draw 0)

    def rand_basis(k, s):
        g = torch.Generator().manual_seed(s)
        return [torch.linalg.qr(torch.randn(H, k, generator=g, dtype=torch.float64))[0].float() for _ in range(3)]
    rnd = {}
    dR = site(D["P0"]["RESERVE200"])
    for k in sorted({sub["U_P0"]["k"], sub["U_KL"]["k"], 1}):
        loads = [share(dR, rand_basis(k, s)) for s in range(20)]
        per_block_max = max(max(x["per_block"]) for x in loads)
        rnd[str(k)] = {"pooled": [x["pooled"] for x in loads], "max_per_block": per_block_max, "expected": k / H,
                       "stop_rule_fired": bool(per_block_max > 3 * k / H)}
    rep["random_draws"] = rnd
    stop_random = any(r["stop_rule_fired"] for kk, r in rnd.items() if int(kk) == sub["U_P0"]["k"])
    log(f"loads (RESERVE200): U_P0 {rep['U_P0']['load']['pooled']:.3f}, U_KL {rep['U_KL']['load']['pooled']:.3f}, v_EM {rep['v_EM']['load']['pooled']:.4f}; "
        f"random k={sub['U_P0']['k']}: max {max(rnd[str(sub['U_P0']['k'])]['pooled']):.4f} (k/H {sub['U_P0']['k'] / H:.4f})")
    out["subspace_report"] = rep
    # ---- 5. normalisers and reference energies
    arms = {"site": (E.SITE, [None] * 3), "route": (E.SITE, UP), "persona": (E.SITE, V1), "route_kl": (E.SITE, UK),
            "deep": (E.DEEP, [None] * 5)}
    rand0 = {k: rand_basis(k, 0) for k in sorted({sub["U_P0"]["k"], sub["U_KL"]["k"], 1})}
    for k, Us in rand0.items():
        arms[f"rand{k}"] = (E.SITE, Us)
    sel = lambda d, layers: d[:, [LI[L] for L in layers]]
    E_train, E_ref = {}, {}
    for a, (layers, Us) in arms.items():
        En = S.per_cell(sel(D["P0"]["ANCHOR240"], layers), Us)
        Eu, fl, hits = S.floored(En)
        E_train[a] = {"E": En, "E_used": Eu, "floor": fl, "hits": int(hits.sum()), "layers": layers}
        E_ref[a] = {ref: {s: S.per_cell(sel(D[ref][s], layers), Us) for s in ("RESERVE200", "RESERVE32", "C2", "C1")} for ref in ("P0@40", "P0", "P0-s2")}
    En = S.per_cell(sel(D["P0-s2"]["ANCHOR240"], E.SITE), [None] * 3)
    Eu, fl, hits = S.floored(En)
    E_train["site_s2"] = {"E": En, "E_used": Eu, "floor": fl, "hits": int(hits.sum()), "layers": E.SITE}
    En = S.per_cell(sel(D["P0"]["TRAIN200"], E.SITE), UP)
    Eu, fl, hits = S.floored(En)
    E_train["route_all_sft"] = {"E": En, "E_used": Eu, "floor": fl, "hits": int(hits.sum()), "layers": E.SITE}
    E_ref["route_all_sft"] = {ref: {"TRAIN200": S.per_cell(sel(D[ref]["TRAIN200"], E.SITE), UP), "C1": S.per_cell(sel(D[ref]["C1"], E.SITE), UP)}
                              for ref in ("P0@40", "P0")}
    out["normalisers"] = {a: {"floor_hits": v["hits"], "E_per_block_mean": [float(x) for x in v["E"].mean(1)],
                              "E_min_over_max": float(v["E"].min() / v["E"].max())} for a, v in E_train.items()}
    out["P0_full_energy_ANCHOR240_per_position"] = {str(L): [float(x) for x in E_train["site"]["E"][i]] for i, L in enumerate(E.SITE)}
    # sanity: P0 measured against itself gives L_blk = 1 on every set (definition check)
    out["self_check_P0_lblk_RESERVE200"] = {a: float((S.per_cell(sel(D["P0"]["RESERVE200"], arms[a][0]), arms[a][1]) / S.floored(E_ref[a]["P0"]["RESERVE200"])[0]).mean())
                                            for a in ("site", "route", "persona")}
    out["P0@40_vs_P0_site_lblk_RESERVE200"] = {a: float((E_ref[a]["P0@40"]["RESERVE200"] / S.floored(E_ref[a]["P0"]["RESERVE200"])[0]).mean()) for a in arms}
    torch.save({"layers": E.SITE, "deep_layers": E.DEEP, "H": H, "v_EM": v.float(),
                "U_P0": sub["U_P0"]["U"], "k_P0": sub["U_P0"]["k"], "energy_P0": torch.stack(sub["U_P0"]["energy"]),
                "U_KL": sub["U_KL"]["U"], "k_KL": sub["U_KL"]["k"], "energy_KL": torch.stack(sub["U_KL"]["energy"]),
                "rand0": {k: torch.stack(Us) for k, Us in rand0.items()}, "E_train": E_train, "E_ref": E_ref,
                "R2_add_route_kl": bool(min(ov) < 0.5), "smoke": smoke}, E.STATS / ("subspaces_smoke.pt" if smoke else "subspaces.pt"))
    if not smoke:                                                                                 # P0's class-2 site change, for M4 later (small)
        torch.save({"C2": site(D["P0"]["C2"]).half(), "C2_deep": sel(D["P0"]["C2"], [34, 38]).half()}, E.STATS / "P0_delta_C2.pt")
        # A11's second x-axis (descriptive): the base model's mean states on TRAIN200 = the training-data centroid, per stored block and position
        torch.save({"layers": LAY, "TRAIN200_mean": hb["TRAIN200"].double().mean(0).float(), "n": hb["TRAIN200"].shape[0]}, E.STATS / "base_train_centroid.pt")
    out["stop_rules"] = {"regression_failed": not reg["pass"], "random_draw_above_3k_over_H": stop_random}
    out["seconds"] = round(time.time() - T0)
    E.dump(E.RESULTS / ("au_smoke.json" if smoke else "au.json"), out)
    log(f"done; stop rules {out['stop_rules']}")
    if not smoke and (not reg["pass"] or stop_random):
        sys.exit(3)


if __name__ == "__main__":
    main()
