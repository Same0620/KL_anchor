"""QS1-E exploratory analyses (A5.7 and additions; not pre-registered beyond "anything else you notice"), CPU.

  qse_explore.py -> results/explore.json
  X1  centred swap: the A5.4 swap with kappa and the replacement taken on the updates minus the model's mean update over the 24 sets, so the
      replaced topic keeps the model's shared (uniform) push and only its topic-specific part is rescaled onto the other model's
  X2  perp-own,3c: A5.2 with the top-3 right singular vectors of the CENTRED update matrix removed (the handoff's perp-own,3 is uncentred,
      so its first direction is the shared push that centring removes anyway); norm-matched null; tau from the replicates in the same variant
  X3  seed reliability of dE (raw, perp-own,3): Pearson / Spearman, seed 0 vs seed 1, over the 231 pairs and over s's row
  X4  how much each projection changes the base map (Spearman over the 253 pairs, max |diff|)
  X5  per-set update sizes (late): |d|, |d - mean_Y d| for s, n, f, everyday under PLAIN / CONTRAST / NEAR / FAR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z
import qse_partA as PA

A = json.loads((Z.RES / "partA.json").read_text())
MP = np.load(Z.RES / "maps.npz")
rows = Z.eval_rows()
h = Z.load_base_h()
MB = Z.base_means(h, rows)
U = Z.load_updates()
perms = MP["perms"]
out = {}
iu = np.triu_indices(Z.NT, 1)


def centred_swap(UA, UBo, Vb, s_idx, ys):
    """Within model A: replace A's topic-specific update on y (update minus A's mean update) by kappa_c x B's topic-specific update on y."""
    B = Z.project(Vb, MB)
    Ap, Bp = Z.project(Vb, UA), Z.project(Vb, UBo)
    Ac, Bc = Ap - Ap.mean(1, keepdims=True), Bp - Bp.mean(1, keepdims=True)
    Ma = B + Ac                                                    # = B + Ap minus a uniform shift: same centred map
    Ea = Z.emap(Ma)
    res = {}
    for y in ys:
        kc = (Ac[:, y] * Bc[:, y]).sum(-1) / (Bc[:, y] ** 2).sum(-1)
        Msw = Ma.copy()
        Msw[:, y] = B[:, y] + kc[:, None] * Bc[:, y]
        res[y] = (Ea[:, s_idx, y] - Z.emap(Msw)[:, s_idx, y], kc)
    return res


X1 = {}
for cs in Z.COST:
    n_t, f_t = Z.NEARFAR[cs]
    si, ni, fi = Z.TI[cs], Z.TI[n_t], Z.TI[f_t]
    rest = [y for y in range(Z.NT) if Z.TOPICS[y] not in (cs, n_t, f_t)]
    UN, UF = U[f"NEAR.{cs}.s0"], U[f"FAR.{cs}.s0"]
    J = Z.svd_basis(np.concatenate([UN, UF], 1), 3)
    X1[cs] = {}
    for v, Vb in (("raw", None), ("joint", J)):
        sN = centred_swap(UN, UF, Vb, si, [ni] + rest)
        sF = centred_swap(UF, UN, Vb, si, [fi] + rest)
        r = {}
        for b in ("late", "early"):
            I = Z.BLOCKS[b]
            wN = np.array([sN[y][0][I].mean() for y in rest])
            wF = np.array([sF[y][0][I].mean() for y in rest])
            r[b] = {"W_n": float(sN[ni][0][I].mean()), "floor_W_NEAR": float(np.percentile(np.abs(wN), 95)), "kappa_c_n": float(sN[ni][1][I].mean()),
                    "W_f": float(sF[fi][0][I].mean()), "floor_W_FAR": float(np.percentile(np.abs(wF), 95)), "kappa_c_f": float(sF[fi][1][I].mean())}
        r["ladder_if_used"] = (A["A5.4"]["tables"][cs]["ladder"] if A["A5.4"]["tables"][cs]["ladder"] in ("CLOSER", "NOT-VISIBLE")
                               else ("PASSIVE" if r["late"]["W_n"] >= -r["late"]["floor_W_NEAR"] else "ACTIVE")) if v == "raw" else None
        X1[cs][v] = r
out["X1_centred_swap"] = X1
for cs in Z.COST:
    print("X1", cs, {v: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in X1[cs][v]["late"].items()} for v in X1[cs]},
          "ladder (centred swap):", X1[cs]["raw"]["ladder_if_used"])

# X2: centred-SVD projection
X2 = {}
for s in Z.STUDENTS:
    Uc = U[s] - U[s].mean(1, keepdims=True)
    V = Z.svd_basis(Uc, 3)
    st, Ebv, Esv = PA.global_stats(MB, U[s], V, Z.pair_mask(Z.excluded(s)), perms, True, f"own3c__{s}")
    unc, cen = Z.energy_removed(V, U[s])
    for b in ("late", "early"):
        st[b]["energy_removed_uncentred"], st[b]["energy_removed_centred"] = float(unc[Z.BLOCKS[b]].mean()), float(cen[Z.BLOCKS[b]].mean())
    st["base_map_spearman_vs_raw"] = Z.spearman(Z.blk(Ebv, "late")[:23, :23][iu], Z.blk(MP["E__raw__base"], "late")[:23, :23][iu])
    X2[s] = st
tau = {b: max(abs(X2[a][b]["beta_ent"] - X2[c][b]["beta_ent"]) for a, c in Z.REPLICATES) for b in ("late", "early")}
lab = {s: PA.labels(X2[s], tau) for s in Z.STUDENTS}
out["X2_own3c"] = {"stats": X2, "tau": tau, "labels": lab}
print("X2 tau", {b: round(t, 4) for b, t in tau.items()})
for s in Z.STUDENTS:
    L = X2[s]["late"]
    print(f"X2 {Z.short_student(s):24s} mad {L['mean_abs_dE']:.4f} beta {L['beta_obs']:+.4f} ent {L['beta_ent']:+.4f} cen-energy {L['energy_removed_centred']:.3f} "
          f"base-map rho {X2[s]['base_map_spearman_vs_raw']:.4f} -> {lab[s]['late']} (early {lab[s]['early']})")

# X3: seed reliability of dE
X3 = {}
Eb = Z.blk(MP["E__raw__base"], "late")
for a, c in Z.REPLICATES:
    src = Z.parse(a)[1]
    mask = Z.pair_mask([src])
    si = Z.TI[src]
    oth = [y for y in range(Z.NT) if y != si]
    r = {}
    for v in ("raw", "own"):
        if v == "raw":
            d0, d1 = Z.blk(MP[f"E__raw__{a}"], "late") - Eb, Z.blk(MP[f"E__raw__{c}"], "late") - Eb
        else:
            d0 = Z.blk(MP[f"E__own__{a}"], "late") - Z.blk(MP[f"Ebase__own__{a}"], "late")
            d1 = Z.blk(MP[f"E__own__{c}"], "late") - Z.blk(MP[f"Ebase__own__{c}"], "late")
        r[v] = {"pairs_pearson": float(np.corrcoef(d0[:23, :23][mask], d1[:23, :23][mask])[0, 1]),
                "pairs_spearman": Z.spearman(d0[:23, :23][mask], d1[:23, :23][mask]),
                "row_pearson": float(np.corrcoef(d0[si, oth], d1[si, oth])[0, 1]), "row_spearman": Z.spearman(d0[si, oth], d1[si, oth])}
    X3[a] = r
out["X3_seed_reliability_dE"] = X3
print("X3", json.dumps({Z.short_student(k): {v: round(x["pairs_pearson"], 3) for v, x in r.items()} for k, r in X3.items()}))

# X4: base-map change under each projection
X4 = {}
for s in Z.STUDENTS:
    r = {}
    for v in ("own", "tag"):
        Eo = Z.blk(MP[f"Ebase__{v}__{s}"], "late")
        r[v] = {"spearman_253": Z.spearman(Eo[:23, :23][iu], Eb[:23, :23][iu]), "max_abs_diff": float(np.abs(Eo - Eb).max())}
    X4[s] = r
for cs in Z.COST:
    Ej = Z.blk(MP[f"E__jointtab_{cs}__base"], "late")
    X4[f"joint_{cs}"] = {"spearman_253": Z.spearman(Ej[:23, :23][iu], Eb[:23, :23][iu]), "max_abs_diff": float(np.abs(Ej - Eb).max())}
out["X4_base_map_under_projection"] = X4

# X5: update sizes
X5 = {}
for cs in Z.COST:
    n_t, f_t = Z.NEARFAR[cs]
    idx = {"s": Z.TI[cs], "n": Z.TI[n_t], "f": Z.TI[f_t], "everyday": Z.EV}
    X5[cs] = {}
    for arm in ("PLAIN", "CONTRAST", "NEAR", "FAR"):
        u = U[f"{arm}.{cs}.s0"][Z.LATE_I]
        uc = u - u.mean(1, keepdims=True)
        X5[cs][arm] = {k: {"norm": float(np.linalg.norm(u[:, i], axis=-1).mean()), "norm_centred": float(np.linalg.norm(uc[:, i], axis=-1).mean())}
                       for k, i in idx.items()}
        X5[cs][arm]["mean_over_sets_norm"] = float(np.linalg.norm(u.mean(1), axis=-1).mean())
    X5[cs]["cos_late_NEAR_FAR"] = {k: float(np.mean([np.dot(U[f"NEAR.{cs}.s0"][l, i], U[f"FAR.{cs}.s0"][l, i]) /
                                                      (np.linalg.norm(U[f"NEAR.{cs}.s0"][l, i]) * np.linalg.norm(U[f"FAR.{cs}.s0"][l, i])) for l in Z.LATE_I]))
                                   for k, i in idx.items()}
    UN, UF = U[f"NEAR.{cs}.s0"], U[f"FAR.{cs}.s0"]
    UNc, UFc = UN - UN.mean(1, keepdims=True), UF - UF.mean(1, keepdims=True)
    X5[cs]["cos_late_NEAR_FAR_centred"] = {k: float(np.mean([np.dot(UNc[l, i], UFc[l, i]) / (np.linalg.norm(UNc[l, i]) * np.linalg.norm(UFc[l, i]))
                                                             for l in Z.LATE_I])) for k, i in idx.items()}
out["X5_update_sizes_late"] = X5
print("X5", json.dumps(X5, indent=0)[:2500])

# X6: how much of s's (T's) centred base vector the perp-own,3 basis removes (late; is s's row fragile under the projection?)
X6 = {"qs1": {}, "es1": {}}
Cq = MB - MB.mean(1, keepdims=True)
for s in Z.STUDENTS:
    si = Z.TI[Z.parse(s)[1]]
    PC = Z.project(Z.svd_basis(U[s], 3), Cq)
    kept = lambda y: float((np.linalg.norm(PC[Z.LATE_I, y], axis=-1) / np.linalg.norm(Cq[Z.LATE_I, y], axis=-1)).mean())
    X6["qs1"][s] = {"s_kept": kept(si), "others_kept_mean": float(np.mean([kept(y) for y in range(Z.NT) if y != si]))}
try:
    import torch
    ES = Z.ROOT / "es1" / "stats"

    def es_means(m):
        S = torch.load(ES / f"{m}.pt", map_location="cpu", weights_only=False)
        return np.stack([S["h6"][f"gen24-{d}"].double().mean(0).numpy() for d in Z.TOPICS], 1)
    Mb = es_means("base")
    Cb = Mb - Mb.mean(1, keepdims=True)
    for k in ("narrow", "broad"):
        ks, ko = [], []
        for d in Z.TOPICS:
            PC = Z.project(Z.svd_basis(es_means(f"em-{d}-{k}") - Mb, 3), Cb)
            T = Z.TI[d]
            kept = lambda y: float((np.linalg.norm(PC[Z.LATE_I, y], axis=-1) / np.linalg.norm(Cb[Z.LATE_I, y], axis=-1)).mean())
            ks.append(kept(T))
            ko.append(np.mean([kept(y) for y in range(Z.NT) if y != T]))
        X6["es1"][k] = {"T_kept_median": float(np.median(ks)), "T_kept_min": float(np.min(ks)), "others_kept_median": float(np.median(ko))}
except FileNotFoundError as e:
    X6["es1"] = {"skipped": str(e)}
out["X6_s_vector_kept_under_own3"] = X6
print("X6", json.dumps({"qs1": {Z.short_student(s): round(v["s_kept"], 3) for s, v in X6["qs1"].items()}, "es1": X6["es1"]}))
out["base_centred_norm_late_mean"] = A["inputs"]["base_centred_norm_late_mean"]
Z.write_json(Z.RES / "explore.json", out)
print("wrote results/explore.json")
