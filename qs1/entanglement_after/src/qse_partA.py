"""QS1-E Part A (CPU): after-training topic maps from QS1's saved data (handoff A2-A5; PREREG Part B).

  qse_partA.py      -> results/partA.json (every statistic and label), results/maps.npz (every E, dE, W, floor, null slope)

Stops (exit 4) on a failed hard check (A2.1) or hook check (A2.2).
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z

T0 = time.time()
NPERM = 1000
MAPS = {}


def log(msg):
    print(f"[partA {time.time() - T0:6.1f}s] {msg}", flush=True)


def stop(msg):
    log("STOP: " + msg)
    Z.write_json(Z.RES / "partA_STOP.json", {"stop": msg})
    sys.exit(4)


def read_csv_matrix(path):
    lines = Path(path).read_text().strip().splitlines()
    hdr = lines[0].split(",")[1:]
    names = [l.split(",")[0] for l in lines[1:]]
    M = np.array([[float(v) for v in l.split(",")[1:]] for l in lines[1:]])
    return hdr, names, M


def upper23(M):
    i, j = np.triu_indices(Z.NT, 1)
    return M[..., i, j]


def pct(x, q):
    return float(np.percentile(np.asarray(x, float), q))


# ---------------------------------------------------------------- A2 checks
def checks(h, rows, MB):
    out = {}
    Eb = Z.emap(MB)
    stored = {}
    for L in Z.HOOKED:
        hdr, names, M = read_csv_matrix(Z.Q / "results" / "closeness" / f"MAIN_E_base_L{L}.csv")
        assert names == Z.SETS24, names
        stored[f"L{L}"] = M
    for b in ("late", "early"):
        hdr, names, M = read_csv_matrix(Z.Q / "results" / "closeness" / f"MAIN_E_base_{b}.csv")
        assert names == Z.SETS24
        stored[b] = M
    cj = json.loads((Z.Q / "results" / "closeness.json").read_text())["matrices_main_23"]
    tol = max(1e-4, 0.5e-6)

    def compare(E6):
        mine = {f"L{L}": E6[k] for k, L in enumerate(Z.HOOKED)}
        mine["late"], mine["early"] = Z.blk(E6, "late"), Z.blk(E6, "early")
        diffs = {k: float(np.abs(mine[k] - stored[k]).max()) for k in mine}
        diffs["json_late_23"] = float(np.abs(mine["late"][:23, :23] - np.array(cj["E_base_late"])).max())
        diffs["json_early_23"] = float(np.abs(mine["early"][:23, :23] - np.array(cj["E_base_early"])).max())
        sp = {b: Z.spearman(upper23(mine[b][:23, :23]), upper23(stored[b][:23, :23])) for b in ("late", "early")}
        return diffs, sp
    diffs, sp = compare(Eb)
    a21 = {"path": "fp64 means of the fp32 residuals (A3)", "max_abs_diff": diffs, "spearman_253": sp, "tolerance": tol,
           "stored": "qs1/results/closeness/MAIN_E_base_{L10..L30,late,early}.csv (6 decimals) and closeness.json matrices_main_23"}
    mx = max(diffs.values())
    a21["pass"] = bool(mx <= tol and min(sp.values()) >= 0.999)
    if not a21["pass"]:
        a21["fallbacks"] = {}
        for path in ("bf16", "fp32"):
            d2, s2 = compare(Z.emap(Z.base_means(h, rows, path=path)))
            a21["fallbacks"][path] = {"max_abs_diff": max(d2.values()), "spearman_253": s2,
                                      "pass": bool(max(d2.values()) <= 1e-3 and min(s2.values()) >= 0.999)}
        ok = [p for p, v in a21["fallbacks"].items() if v["pass"]]
        a21["matched_path"] = ok[0] if ok else None
        a21["pass"] = bool(ok)
    out["A2.1"] = a21
    log(f"A2.1 hard check: max |diff| {mx:.2e} (tol {tol:.0e}); Spearman late {sp['late']:.6f}, early {sp['early']:.6f} -> "
        f"{'PASS' if a21['pass'] else 'FAIL'}")
    if not a21["pass"]:
        stop("A2.1 hard check failed")

    # A2.2 hook check from the code
    src = {k: (Z.Q / "src" / f"{k}.py").read_text() for k in ("qs1common", "qs1_job1", "qs1_job3")}
    m = re.search(r"def score_prompts\(.*?\n(?=def |\Z)", src["qs1common"], re.S)
    sp_src = m.group(0)
    conds = {
        "score_prompts reads hidden_states[L + 1] at the last template token":
            "o.hidden_states[L + 1][ar, last].float() for L in layers" in sp_src and "last = torch.tensor([len(p) - 1 for p in chunk]" in sp_src,
        "score_prompts default layers = HOOKED": "layers=HOOKED" in sp_src and "HOOKED = [10, 14, 18, 22, 26, 30]" in src["qs1common"],
        "Job 1 residuals via score_prompts(want_h=True)": "S = X.score_prompts(pm, prompts, want_h=True, want_top=True)" in src["qs1_job1"]
            and '"h": S["h"]' in src["qs1_job1"],
        "Job 3 final deltas via score_prompts(want_h=True)": 'fin = full_eval(pm, "final", want_h=True)' in src["qs1_job3"]
            and "r = X.score_prompts(model, prompts, want_h=want_h)" in src["qs1_job3"],
        "Job 3 subtracts Job 1's set means": 'base_setmean = torch.stack([b1["h"][sl[s]].double().mean(0) for s in sets_order]).float()' in src["qs1_job3"]
            and "- base_setmean" in src["qs1_job3"] and 'b1 = torch.load(J1 / "base.pt"' in src["qs1_job3"],
    }
    out["A2.2"] = {"conditions": conds, "pass": all(conds.values()),
                   "note": "one helper (qs1common.score_prompts) produces both: output of block L = hidden_states[L + 1], last template token, "
                           "same PEFT-wrapped bf16 model (Job 1 at B = 0)"}
    log(f"A2.2 hook check: {'PASS' if out['A2.2']['pass'] else 'FAIL'} {conds}")
    if not out["A2.2"]["pass"]:
        stop("A2.2 hook check failed")
    out["A2.3"] = {"applicable": False, "note": "Job 3 did not store the base set means; it recomputed them inside the job from Job 1's "
                   "stats/job1/base.pt (fp64 mean -> fp32), so they are Job 1's by construction. Nothing to compare (flag-only check)."}

    # A2.4 ES1 comparison (descriptive)
    es = np.load(Z.ROOT / "es1" / "results" / "E_base_23x23.npz")
    assert [str(x) for x in es["domains"]] == Z.TOPICS and list(es["layers_hooked"]) == Z.HOOKED
    M23 = MB[:, :23]
    E23 = Z.emap(M23)
    a24 = {"qs1_indexing": "block L = hidden_states[L + 1] (qs1common.score_prompts), the same convention as ES1",
           "es1_centring": "mean of the 23 topic means"}
    for b in ("late", "early"):
        mine, theirs = Z.blk(E23, b), es[f"E_{b}"]
        a24[b] = {"spearman_253": Z.spearman(upper23(mine), upper23(theirs)), "max_abs_diff": float(np.abs(mine - theirs).max()),
                  "mean_abs_diff": float(np.abs(upper23(mine) - upper23(theirs)).mean())}
    a24["per_layer_max_abs_diff"] = {f"L{L}": float(np.abs(E23[k] - es["E_hooked"][k]).max()) for k, L in enumerate(Z.HOOKED)}
    # QS1's 24-set centring against ES1's 23-set centring (both on QS1's residuals): how much the everyday set in the centre matters
    a24["qs1_24_vs_23_centring_late"] = {"spearman_253": Z.spearman(upper23(Z.blk(Eb, "late")[:23, :23]), upper23(Z.blk(E23, "late"))),
                                         "max_abs_diff": float(np.abs(Z.blk(Eb, "late")[:23, :23] - Z.blk(E23, "late")).max())}
    out["A2.4"] = a24
    log(f"A2.4 ES1: late Spearman {a24['late']['spearman_253']:.4f}, max |diff| {a24['late']['max_abs_diff']:.4f}; "
        f"early {a24['early']['spearman_253']:.4f} / {a24['early']['max_abs_diff']:.4f}")
    return out, Eb


# ---------------------------------------------------------------- versions
def bases(U):
    V = {"raw": {s: None for s in Z.STUDENTS}, "own": {}, "tag": {}}
    for s in Z.STUDENTS:
        V["own"][s] = Z.svd_basis(U[s], 3)
        arm, src, seed = Z.parse(s)
        plain = f"PLAIN.{src}.s0" if src in Z.SOURCES else "PLAIN.bad_medical.s0"
        if arm == "NULL":
            plain = "PLAIN.bad_medical.s0"
        V["tag"][s] = Z.orth_basis(np.stack([U[plain].mean(1), U["NULL.bad_medical.s0"].mean(1)], 1))
    J = {}
    for cs in Z.COST:
        J[cs] = Z.svd_basis(np.concatenate([U[f"NEAR.{cs}.s0"], U[f"FAR.{cs}.s0"]], 1), 3)
    for a, b in Z.REPLICATES:
        J[a] = Z.svd_basis(np.concatenate([U[a], U[b]], 1), 3)
    tagsrc = {cs: V["tag"][f"NEAR.{cs}.s0"] for cs in Z.COST}
    return V, J, tagsrc


def vmaps(MB, U_list, Vb):
    """Maps of the base and of each update under the projection Vb -> (E_base [6,24,24], [E_S ...])."""
    B = Z.project(Vb, MB)
    return Z.emap(B), [Z.emap(B + Z.project(Vb, u)) for u in U_list]


# ---------------------------------------------------------------- A5.2: global change and the entanglement label
def global_stats(MB, Us, Vb, mask, perms, norm_match, tagname):
    B = Z.project(Vb, MB)
    Uv = Z.project(Vb, Us)
    Ebv, Esv = Z.emap(B), Z.emap(B + Uv)
    res = {}
    Gbb = np.einsum("lxd,lyd->lxy", B, B)
    Gbu = np.einsum("lxd,lyd->lxy", B, Uv)
    Guu = np.einsum("lxd,lyd->lxy", Uv, Uv)
    nrm = np.linalg.norm(Uv, axis=2) if norm_match else None
    ident = np.arange(Z.NS)[None]
    Eid = Z.gram_emap(Z.perm_gram(Gbb, Gbu, Guu, ident, nrm))[0]
    assert np.abs(Eid - Esv).max() < 1e-9, (tagname, np.abs(Eid - Esv).max())
    Eperm = Z.gram_emap(Z.perm_gram(Gbb, Gbu, Guu, perms, nrm))          # [K, 6, 24, 24]
    for b in ("late", "early"):
        eb, es = Z.blk(Ebv, b)[:23, :23], Z.blk(Esv, b)[:23, :23]
        dE = es - eb
        x, y = eb[mask], dE[mask]
        beta = float(Z.ols_slope(x, y))
        ep = Z.blk(Eperm, b)[:, :23, :23]
        yn = (ep - eb[None])[:, mask]
        bn = Z.ols_slope(np.broadcast_to(x, yn.shape), yn)
        p = min(1.0, 2 * min(1 + (bn >= beta).sum(), 1 + (bn <= beta).sum()) / (NPERM + 1))
        res[b] = {"mean_abs_dE": float(np.abs(y).mean()), "mean_dE": float(y.mean()), "sd_E_student": float(es[mask].std()),
                  "sd_E_base": float(x.std()), "sd_change": float(es[mask].std() - x.std()), "beta_obs": beta,
                  "beta_null_mean": float(bn.mean()), "beta_null_sd": float(bn.std()), "null_p2.5": pct(bn, 2.5), "null_p97.5": pct(bn, 97.5),
                  "beta_ent": beta - float(bn.mean()), "p_two_sided": float(p), "n_pairs": int(mask.sum()),
                  "mean_dE_all_offdiag_24": float((Z.blk(Esv, b) - Z.blk(Ebv, b))[~np.eye(Z.NS, dtype=bool)].mean())}
        MAPS[f"null__{tagname}__{b}"] = bn
    return res, Ebv, Esv


def labels(stats, tau):
    out = {}
    for b in ("late", "early"):
        r = stats[b]
        t = tau[b]
        if r["beta_obs"] > r["null_p97.5"] and r["beta_ent"] > t:
            out[b] = "MORE"
        elif r["beta_obs"] < r["null_p2.5"] and r["beta_ent"] < -t:
            out[b] = "LESS"
        else:
            out[b] = "UNCHANGED"
    return out


# ---------------------------------------------------------------- A5.4 helpers
def swap_W(MB, Ua, Ub, Vb, s_idx, ys, extra_rows=None):
    """Within model a: replace a's update on topic y by its projection onto b's update on y (per layer), recompute the map.
    -> {y: (W per layer [6], kappa [6], {z: W(y, z) per layer})}; W = E_a(s, y) - E_a^swap(s, y)."""
    B = Z.project(Vb, MB)
    A = Z.project(Vb, Ua)
    Bo = Z.project(Vb, Ub)
    Ma = B + A
    Ea = Z.emap(Ma)
    out = {}
    for y in ys:
        kappa = (A[:, y] * Bo[:, y]).sum(-1) / (Bo[:, y] ** 2).sum(-1)
        Msw = Ma.copy()
        Msw[:, y] = B[:, y] + kappa[:, None] * Bo[:, y]
        Esw = Z.emap(Msw)
        ex = {z: Ea[:, y, z] - Esw[:, y, z] for z in (extra_rows or [])}
        out[y] = (Ea[:, s_idx, y] - Esw[:, s_idx, y], kappa, ex)
    return out


def main():
    rows = Z.eval_rows()
    sl = Z.set_rows(rows)
    assert [len(sl[s]) for s in Z.SETS24] == [24] * 23 + [48] and max(i for s in Z.SETS24 for i in sl[s]) == 599
    nf = json.loads((Z.Q / "data" / "nearfar.json").read_text())
    assert {cs: (nf[cs]["near"], nf[cs]["far"]) for cs in Z.COST} == Z.NEARFAR, nf
    h = Z.load_base_h()
    MB = Z.base_means(h, rows)
    U = Z.load_updates()
    gate, gate_chk = Z.load_gate()
    log(f"inputs loaded: base means {MB.shape}, {len(U)} students; dL_gate vs cells.csv max |diff| {max(gate_chk.values()):.1e}")
    assert max(gate_chk.values()) < 1e-4, gate_chk
    out = {"inputs": {"gate_vs_cells_max_abs_diff": max(gate_chk.values()),
                      "update_norm_late_mean": {s: float(np.linalg.norm(U[s][Z.LATE_I], axis=2).mean()) for s in Z.STUDENTS},
                      "base_centred_norm_late_mean": float(np.linalg.norm((MB - MB.mean(1, keepdims=True))[Z.LATE_I], axis=2).mean())}}
    chk, Eb = checks(h, rows, MB)
    out["checks"] = chk
    MAPS["E__raw__base"] = Eb

    V, J, tagsrc = bases(U)
    rng = np.random.default_rng(0)
    perms = np.stack([rng.permutation(Z.NS) for _ in range(NPERM)])
    MAPS["perms"] = perms

    # ---------------- A5.2 per student and version
    G = {}
    for v in ("raw", "own", "tag"):
        G[v] = {}
        for s in Z.STUDENTS:
            mask = Z.pair_mask(Z.excluded(s))
            st, Ebv, Esv = global_stats(MB, U[s], V[v][s], mask, perms, norm_match=(v != "raw"), tagname=f"{v}__{s}")
            if v != "raw":
                unc, cen = Z.energy_removed(V[v][s], U[s])
                for b in ("late", "early"):
                    st[b]["energy_removed_uncentred"] = float(unc[Z.BLOCKS[b]].mean())
                    st[b]["energy_removed_centred"] = float(cen[Z.BLOCKS[b]].mean())
            G[v][s] = st
            MAPS[f"E__{v}__{s}"] = Esv
            if v != "raw":
                MAPS[f"Ebase__{v}__{s}"] = Ebv
        log(f"A5.2 {v}: done")
    # tau per version, and the joint-basis tau (descriptive)
    tau = {}
    for v in ("raw", "own", "tag"):
        tau[v] = {b: max(abs(G[v][a][b]["beta_ent"] - G[v][c][b]["beta_ent"]) for a, c in Z.REPLICATES) for b in ("late", "early")}
    tau_joint = {}
    for a, c in Z.REPLICATES:
        mask = Z.pair_mask(Z.excluded(a))
        r0, _, _ = global_stats(MB, U[a], J[a], mask, perms, True, f"jointrep__{a}")
        r1, _, _ = global_stats(MB, U[c], J[a], mask, perms, True, f"jointrep__{c}")
        tau_joint[a] = {b: abs(r0[b]["beta_ent"] - r1[b]["beta_ent"]) for b in ("late", "early")}
    tau["joint_descriptive"] = {b: max(x[b] for x in tau_joint.values()) for b in ("late", "early")}
    tau["joint_per_pair"] = tau_joint
    lab = {v: {s: labels(G[v][s], tau[v]) for s in Z.STUDENTS} for v in ("raw", "own", "tag")}
    out["A5.2"] = {"stats": G, "tau": tau, "labels": lab}
    for v in ("raw", "own"):
        log(f"A5.2 labels {v} (tau late {tau[v]['late']:.4f}): " + ", ".join(f"{Z.short_student(s)} {lab[v][s]['late']}" for s in Z.STUDENTS))

    # ---------------- A5.3 the source row
    Ebl = Z.blk(Eb, "late")
    row = {}
    for s in Z.STUDENTS:
        arm, src, seed = Z.parse(s)
        si = Z.TI[src]
        oth = [y for y in range(Z.NT) if y != si]
        r = {}
        for b in ("late", "early"):
            dEr = Z.blk(MAPS[f"E__raw__{s}"], b) - Z.blk(Eb, b)
            dEo = Z.blk(MAPS[f"E__own__{s}"], b) - Z.blk(MAPS[f"Ebase__own__{s}"], b)
            lk_u = Z.leak(U[s], si)[Z.BLOCKS[b]].mean(0)
            lk_c = Z.leak(U[s], si, centred=True)[Z.BLOCKS[b]].mean(0)
            rb = {"dE_raw": {Z.SHORT24[y]: float(dEr[si, y]) for y in oth + [Z.EV]},
                  "dE_own": {Z.SHORT24[y]: float(dEo[si, y]) for y in oth + [Z.EV]},
                  "leak_uncentred": {Z.SHORT24[y]: float(lk_u[y]) for y in oth + [Z.EV]},
                  "leak_centred": {Z.SHORT24[y]: float(lk_c[y]) for y in oth + [Z.EV]},
                  "spearman_dEraw_leak_uncentred": Z.spearman(dEr[si, oth], lk_u[oth]),
                  "spearman_dEraw_leak_centred": Z.spearman(dEr[si, oth], lk_c[oth]),
                  "spearman_dEraw_gate": Z.spearman(dEr[si, oth], gate[s][oth]),
                  "spearman_dEraw_Ebase": Z.spearman(dEr[si, oth], Z.blk(Eb, b)[si, oth])}
            order = sorted(oth, key=lambda y: -Ebl[si, y])
            near3, far3 = order[:3], order[-3:]
            rb["near3"], rb["far3"] = [Z.SHORT24[y] for y in near3], [Z.SHORT24[y] for y in far3]
            rb["near3_mean_dE_raw"], rb["far3_mean_dE_raw"] = float(dEr[si, near3].mean()), float(dEr[si, far3].mean())
            rb["near3_minus_far3_raw"] = rb["near3_mean_dE_raw"] - rb["far3_mean_dE_raw"]
            rb["near3_mean_dE_own"], rb["far3_mean_dE_own"] = float(dEo[si, near3].mean()), float(dEo[si, far3].mean())
            if arm in ("NEAR", "FAR"):
                ex = [Z.TI[t] for t in Z.NEARFAR[src]]
                o2 = [y for y in order if y not in ex]
                rb["excl_nf_near3"], rb["excl_nf_far3"] = [Z.SHORT24[y] for y in o2[:3]], [Z.SHORT24[y] for y in o2[-3:]]
                rb["excl_nf_near3_minus_far3_raw"] = float(dEr[si, o2[:3]].mean() - dEr[si, o2[-3:]].mean())
            # A5.5 everyday
            rb["everyday"] = {"E_base_raw": float(Z.blk(Eb, b)[si, Z.EV]), "E_student_raw": float(Z.blk(MAPS[f"E__raw__{s}"], b)[si, Z.EV]),
                              "E_base_own": float(Z.blk(MAPS[f"Ebase__own__{s}"], b)[si, Z.EV]),
                              "E_student_own": float(Z.blk(MAPS[f"E__own__{s}"], b)[si, Z.EV])}
            rb["gate"] = {Z.SHORT24[y]: float(gate[s][y]) for y in oth + [Z.EV]}
            r[b] = rb
        row[s] = r
    out["A5.3"] = row
    log("A5.3 rows: " + ", ".join(f"{Z.short_student(s)} rho(dE,leak) {row[s]['late']['spearman_dEraw_leak_uncentred']:+.2f}" for s in Z.STUDENTS))

    # ---------------- A4 floors (raw, joint, tag) and A5.4
    def E_of(model_u, Vb):
        return Z.emap(Z.project(Vb, MB) + Z.project(Vb, model_u))

    floors = {}
    for v in ("raw", "joint", "tag"):
        fv = {}
        seedv = {}
        for a, c in Z.REPLICATES:
            src = Z.parse(a)[1]
            si = Z.TI[src]
            Vb = None if v == "raw" else (J[a] if v == "joint" else V["tag"][a])
            Ea, Ec = Z.blk(E_of(U[a], Vb), "late"), Z.blk(E_of(U[c], Vb), "late")
            mask = Z.pair_mask([src])
            oth = [y for y in range(Z.NT) if y != si]
            seedv[a] = {"pair": np.abs(Ea - Ec)[:23, :23][mask], "row": np.abs(Ea - Ec)[si, oth]}
        cm, cf = "CONTRAST.bad_medical.s0", "CONTRAST.risky_financial.s0"
        fv["CONTRAST_seed_pair"] = pct(np.concatenate([seedv[cm]["pair"], seedv[cf]["pair"]]), 95)
        fv["CONTRAST_seed_row"] = pct(np.concatenate([seedv[cm]["row"], seedv[cf]["row"]]), 95)
        fv["PLAIN_seed_pair"] = pct(seedv["PLAIN.bad_medical.s0"]["pair"], 95)
        fv["PLAIN_seed_row"] = pct(seedv["PLAIN.bad_medical.s0"]["row"], 95)
        fv["n_values"] = {"CONTRAST_seed_pair": int(len(seedv[cm]["pair"]) + len(seedv[cf]["pair"])),
                          "CONTRAST_seed_row": int(len(seedv[cm]["row"]) + len(seedv[cf]["row"])),
                          "PLAIN_seed_pair": int(len(seedv["PLAIN.bad_medical.s0"]["pair"])), "PLAIN_seed_row": int(len(seedv["PLAIN.bad_medical.s0"]["row"]))}
        for k, x in seedv.items():
            MAPS[f"floorvals__{v}__seed__{k}__pair"] = x["pair"]
            MAPS[f"floorvals__{v}__seed__{k}__row"] = x["row"]
        for cs in Z.COST:
            n_t, f_t = Z.NEARFAR[cs]
            si = Z.TI[cs]
            Vb = None if v == "raw" else (J[cs] if v == "joint" else tagsrc[cs])
            En, Ef = Z.blk(E_of(U[f"NEAR.{cs}.s0"], Vb), "late"), Z.blk(E_of(U[f"FAR.{cs}.s0"], Vb), "late")
            mask = Z.pair_mask([cs, n_t, f_t])
            rest = [y for y in range(Z.NT) if Z.TOPICS[y] not in (cs, n_t, f_t)]
            pv, rv = np.abs(En - Ef)[:23, :23][mask], np.abs(En - Ef)[si, rest]
            MAPS[f"floorvals__{v}__nearfar__{cs}__pair"], MAPS[f"floorvals__{v}__nearfar__{cs}__row"] = pv, rv
            fv[f"NEARFAR_pair__{cs}"], fv[f"NEARFAR_row__{cs}"] = pct(pv, 95), pct(rv, 95)
            fv["n_values"][f"NEARFAR_pair__{cs}"], fv["n_values"][f"NEARFAR_row__{cs}"] = int(len(pv)), int(len(rv))
            fv[f"A4.3__{cs}"] = max(fv["CONTRAST_seed_pair"], fv["CONTRAST_seed_row"], fv[f"NEARFAR_pair__{cs}"], fv[f"NEARFAR_row__{cs}"])
        floors[v] = fv
        log(f"A4 floors {v}: " + ", ".join(f"{k} {x:.4f}" for k, x in fv.items() if k != "n_values"))
    out["A4"] = floors

    main_tab = {}
    for cs in Z.COST:
        n_t, f_t = Z.NEARFAR[cs]
        si, ni, fi = Z.TI[cs], Z.TI[n_t], Z.TI[f_t]
        models = {"base": None, "PLAIN": f"PLAIN.{cs}.s0", "CONTRAST": f"CONTRAST.{cs}.s0", "NEAR": f"NEAR.{cs}.s0", "FAR": f"FAR.{cs}.s0"}
        tab = {"n": n_t, "f": f_t, "rows": {}}
        Em = {}
        for v, Vb in (("raw", None), ("joint", J[cs]), ("tag", tagsrc[cs])):
            Em[v] = {}
            for k, st in models.items():
                Em[v][k] = Z.emap(Z.project(Vb, MB)) if st is None else E_of(U[st], Vb)
                MAPS[f"E__{v}tab_{cs}__{k}"] = Em[v][k]
        for k, st in models.items():
            r = {}
            for b in ("late", "early"):
                rb = {f"E_sn_{v}": float(Z.blk(Em[v][k], b)[si, ni]) for v in Em}
                rb.update({f"E_sf_{v}": float(Z.blk(Em[v][k], b)[si, fi]) for v in Em})
                if st is not None:
                    for c, nm in ((False, "uncentred"), (True, "centred")):
                        lk = Z.leak(U[st], si, centred=c)[Z.BLOCKS[b]].mean(0)
                        rb[f"leak_sn_{nm}"], rb[f"leak_sf_{nm}"] = float(lk[ni]), float(lk[fi])
                r[b] = rb
            tab["rows"][k] = r
        D = {}
        for v in Em:
            D[v] = {}
            for b in ("late", "early"):
                EN, EF = Z.blk(Em[v]["NEAR"], b), Z.blk(Em[v]["FAR"], b)
                D[v][b] = {"D_n": float(EN[si, ni] - EF[si, ni]), "D_f": float(EF[si, fi] - EN[si, fi])}
        tab["D"] = D
        # swap test (raw, joint, tag)
        sw = {}
        rest = [y for y in range(Z.NT) if Z.TOPICS[y] not in (cs, n_t, f_t)]
        Ebl_row = Z.blk(Eb, "late")[ni]
        n_near3 = sorted([y for y in range(Z.NT) if y not in (ni, si)], key=lambda y: -Ebl_row[y])[:3]
        for v, Vb in (("raw", None), ("joint", J[cs]), ("tag", tagsrc[cs])):
            UN, UF = U[f"NEAR.{cs}.s0"], U[f"FAR.{cs}.s0"]
            sN = swap_W(MB, UN, UF, Vb, si, [ni] + rest, extra_rows=n_near3)
            sF = swap_W(MB, UF, UN, Vb, si, [fi] + rest)
            res = {}
            for b in ("late", "early"):
                I = Z.BLOCKS[b]
                Wn = float(sN[ni][0][I].mean())
                WN_rest = np.array([sN[y][0][I].mean() for y in rest])
                Wf = float(sF[fi][0][I].mean())
                WF_rest = np.array([sF[y][0][I].mean() for y in rest])
                Wn_y = {Z.SHORT24[z]: float(sN[ni][2][z][I].mean()) for z in n_near3}
                res[b] = {"W_n": Wn, "floor_W_NEAR": pct(np.abs(WN_rest), 95), "W_f": Wf, "floor_W_FAR": pct(np.abs(WF_rest), 95),
                          "kappa_n": float(sN[ni][1][I].mean()), "kappa_f": float(sF[fi][1][I].mean()),
                          "W_n_Y_near3": Wn_y, "W_n_Y_near3_mean": float(np.mean(list(Wn_y.values()))),
                          "W_rest_NEAR": {Z.SHORT24[y]: float(w) for y, w in zip(rest, WN_rest)},
                          "W_rest_FAR": {Z.SHORT24[y]: float(w) for y, w in zip(rest, WF_rest)}}
                MAPS[f"W__{cs}__{v}__{b}__NEAR_rest"], MAPS[f"W__{cs}__{v}__{b}__FAR_rest"] = WN_rest, WF_rest
            MAPS[f"W__{cs}__{v}__W_n_per_layer"], MAPS[f"W__{cs}__{v}__W_f_per_layer"] = sN[ni][0], sF[fi][0]
            sw[v] = res
        tab["swap"] = sw
        tab["n_near3_other_than_s"] = [Z.SHORT24[z] for z in n_near3]
        # energy removed by the joint basis
        en = {}
        for k in ("NEAR", "FAR"):
            unc, cen = Z.energy_removed(J[cs], U[models[k]])
            en[k] = {b: {"uncentred": float(unc[Z.BLOCKS[b]].mean()), "centred": float(cen[Z.BLOCKS[b]].mean())} for b in ("late", "early")}
        tab["joint_energy_removed"] = en
        # ladder (late, raw decides)
        fl = floors["raw"][f"A4.3__{cs}"]
        Dn = D["raw"]["late"]["D_n"]
        Wn, fW = sw["raw"]["late"]["W_n"], sw["raw"]["late"]["floor_W_NEAR"]
        if Dn > fl:
            lad = "CLOSER"
        elif Dn >= -fl:
            lad = "NOT-VISIBLE"
        elif Wn >= -fW:
            lad = "PASSIVE"
        else:
            lad = "ACTIVE"
        tab["ladder"] = lad
        tab["ladder_inputs"] = {"D_n_raw": Dn, "floor_A4.3_raw": fl, "W_n_raw": Wn, "floor_W_NEAR_raw": fW}
        extras = {}
        sr = sw["raw"]["late"]
        extras["s_specific"] = bool(sr["W_n"] < sr["W_n_Y_near3_mean"] - sr["floor_W_NEAR"])
        sj = sw["joint"]["late"]
        extras["joint_W_n_below_minus_floor"] = bool(sj["W_n"] < -sj["floor_W_NEAR"])
        extras["W_f_below_minus_floor_raw"] = bool(sr["W_f"] < -sr["floor_W_FAR"])
        extras["joint_removes_ge_95pct_centred"] = {k: bool(en[k]["late"]["centred"] >= 0.95) for k in en}
        tab["active_extras"] = extras
        main_tab[cs] = tab
        log(f"A5.4 {cs}: D_n raw {Dn:+.4f} (floor {fl:.4f}), W_n {Wn:+.4f} (floor_W {fW:.4f}), W_f {sr['W_f']:+.4f} "
            f"(floor_W FAR {sr['floor_W_FAR']:.4f}) -> {lad}")
    labs = [main_tab[cs]["ladder"] for cs in Z.COST]
    out["A5.4"] = {"tables": main_tab, "overall": labs[0] if labs[0] == labs[1] else "MIXED"}
    log(f"A5.4 overall ladder: {out['A5.4']['overall']}")

    # ---------------- A4.5 base split-half SD (context), from QS1's S2 halves
    s2 = json.loads((Z.Q / "results" / "closeness.json").read_text())["S2_split_half"]
    halves = []
    for hh in (0, 1):
        M = np.zeros_like(MB)
        for j, sname in enumerate(Z.SETS24):
            ids1 = set(s2["split"][sname])
            idx = [i for i in sl[sname] if (rows[i]["id"] in ids1) == (hh == 0)]
            assert len(idx) == len(sl[sname]) // 2
            M[:, j] = h[idx].astype(np.float64).mean(0)
        halves.append(Z.blk(Z.emap(M), "late")[:23, :23])
    sd = np.abs(halves[0] - halves[1]) / math.sqrt(2)
    out["A4.5"] = {"split_half_spearman_253_reproduced": Z.spearman(upper23(halves[0]), upper23(halves[1])),
                   "qs1_split_half_spearman_253": s2["spearman_253"], "median_cell_sd_half": float(np.median(upper23(sd))),
                   "median_cell_sd_full_approx": float(np.median(upper23(sd))) / math.sqrt(2), "p95_cell_sd_half": pct(upper23(sd), 95)}
    MAPS["splithalf_sd_late"] = sd
    log(f"A4.5 split-half: Spearman {out['A4.5']['split_half_spearman_253_reproduced']:.4f} (QS1 {s2['spearman_253']:.4f}); "
        f"median cell SD (half) {out['A4.5']['median_cell_sd_half']:.4f}")

    # ---------------- predictions
    def mad(v, s, b="late"):
        return G[v][s][b]["mean_abs_dE"]
    P = {}
    P["P1"] = {"PLAIN(medical)": mad("raw", "PLAIN.bad_medical.s0"), "NULL": mad("raw", "NULL.bad_medical.s0")}
    P["P1"]["holds"] = bool(P["P1"]["PLAIN(medical)"] <= 0.087 and P["P1"]["NULL"] <= 0.087)
    c_gt_p = {Z.SHORT[s]: (mad("raw", f"CONTRAST.{s}.s0"), mad("raw", f"PLAIN.{s}.s0")) for s in Z.SOURCES}
    rho = {Z.SHORT[s]: row[f"CONTRAST.{s}.s0"]["late"]["spearman_dEraw_leak_uncentred"] for s in Z.SOURCES}
    P["P2"] = {"contrast_vs_plain": c_gt_p, "all_four_contrast_gt_plain": all(a > b for a, b in c_gt_p.values()),
               "rho_dE_leak_contrast": rho, "n_rho_ge_0.5": sum(r >= 0.5 for r in rho.values()),
               "rho_dE_leak_plain": {Z.SHORT[s]: row[f"PLAIN.{s}.s0"]["late"]["spearman_dEraw_leak_uncentred"] for s in Z.SOURCES}}
    P["P2"]["holds"] = bool(P["P2"]["all_four_contrast_gt_plain"] and P["P2"]["n_rho_ge_0.5"] >= 3)
    n3 = {Z.SHORT[s]: row[f"CONTRAST.{s}.s0"]["late"]["near3_mean_dE_raw"] for s in Z.SOURCES}
    P["P3"] = {"near3_mean_dE_raw": n3, "n_positive": sum(x > 0 for x in n3.values())}
    P["P3"]["holds"] = bool(P["P3"]["n_positive"] >= 3)
    p4s = [s for s in Z.STUDENTS if Z.parse(s)[0] in ("CONTRAST", "NEAR", "FAR")]
    share = {Z.short_student(s): 1 - mad("own", s) / mad("raw", s) for s in p4s}
    P["P4"] = {"share_removed": share, "min": min(share.values()), "holds": bool(min(share.values()) >= 0.70),
               "holds_seed0_only": bool(min(v for k, v in share.items() if not k.endswith("s1")) >= 0.70)}
    P["P5"] = {"observed": out["A5.4"]["overall"], "per_source": {Z.SHORT[cs]: main_tab[cs]["ladder"] for cs in Z.COST}}
    lc = {v: {Z.SHORT[s]: lab[v][f"CONTRAST.{s}.s0"]["late"] for s in Z.SOURCES} for v in ("raw", "own")}
    lp = {Z.SHORT[s]: lab["raw"][f"PLAIN.{s}.s0"]["late"] for s in Z.SOURCES}
    P["P6a"] = {"labels": lc["raw"], "holds": sum(x == "MORE" for x in lc["raw"].values()) >= 3}
    P["P6b"] = {"labels": lc["own"], "holds": sum(x == "UNCHANGED" for x in lc["own"].values()) >= 3}
    P["P6c"] = {"labels": lp, "holds": sum(x == "UNCHANGED" for x in lp.values()) >= 3}
    out["predictions"] = P
    log("predictions: " + ", ".join(f"{k} {'holds' if v.get('holds') else ('= ' + v['observed'] if k == 'P5' else 'fails')}" for k, v in P.items()))

    out["elapsed_s"] = time.time() - T0
    Z.write_json(Z.RES / "partA.json", out)
    np.savez_compressed(Z.RES / "maps.npz", **MAPS)
    log(f"wrote results/partA.json and results/maps.npz ({(Z.RES / 'maps.npz').stat().st_size / 2 ** 20:.1f} MB, {len(MAPS)} arrays)")


if __name__ == "__main__":
    main()
