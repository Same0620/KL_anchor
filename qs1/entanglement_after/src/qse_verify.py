"""Independent checks of qse_partA.py (CPU): brute-force recomputation of a few null slopes, swaps and projections with plain vector code
(no Gram algebra, no shared map helpers beyond loading).  -> results/verify.json; exit 5 on any mismatch."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z

A = json.loads((Z.RES / "partA.json").read_text())
MP = np.load(Z.RES / "maps.npz")
rows = Z.eval_rows()
h = Z.load_base_h()
MB = Z.base_means(h, rows)
U = Z.load_updates()
perms = MP["perms"]
out = {}
bad = []


def cmap(M):                                   # per-layer centred cosine, written out plainly
    E = np.zeros((M.shape[0], M.shape[1], M.shape[1]))
    for l in range(M.shape[0]):
        C = M[l] - M[l].mean(0)
        for a in range(M.shape[1]):
            for b in range(M.shape[1]):
                E[l, a, b] = C[a] @ C[b] / (np.linalg.norm(C[a]) * np.linalg.norm(C[b]))
    return E


def late(E):
    return E[3:6].mean(0)


def projector(U3):                             # top-3 right singular vectors via eigh of U^T U restricted (independent of svd_basis)
    P = []
    for l in range(U3.shape[0]):
        w, v = np.linalg.eigh(U3[l] @ U3[l].T)            # 24x24 (or 48x48) Gram; right singular vectors = U^T v / s
        idx = np.argsort(w)[::-1][:3]
        R = (U3[l].T @ v[:, idx]) / np.sqrt(w[idx])
        P.append(np.eye(U3.shape[2]) - R @ R.T)
    return P


def slope(x, y):
    return np.polyfit(x, y, 1)[0]


for s, ver, ks in (("CONTRAST.bad_medical.s0", "raw", [0, 1, 999]), ("CONTRAST.bad_medical.s0", "own", [0, 7]),
                   ("PLAIN.bad_legal.s0", "own", [3]), ("NEAR.risky_financial.s0", "raw", [5])):
    mask = Z.pair_mask(Z.excluded(s))
    if ver == "raw":
        Pm = [np.eye(Z.D_MODEL)] * 6
    else:
        Pm = projector(U[s])
    B = np.stack([MB[l] @ Pm[l] for l in range(6)])
    Uv = np.stack([U[s][l] @ Pm[l] for l in range(6)])
    Eb = late(cmap(B))[:23, :23]
    Es = late(cmap(B + Uv))[:23, :23]
    beta = slope(Eb[mask], (Es - Eb)[mask])
    ref = A["A5.2"]["stats"][ver][s]["late"]["beta_obs"]
    nulls = MP[f"null__{ver}__{s}__late"]
    for k in ks:
        pi = perms[k]
        if ver == "raw":
            Up = Uv[:, pi]
        else:
            nr = np.linalg.norm(Uv, axis=2)
            Up = Uv[:, pi] * (nr / nr[:, pi])[:, :, None]
        Ep = late(cmap(B + Up))[:23, :23]
        bn = slope(Eb[mask], (Ep - Eb)[mask])
        out[f"null {s} {ver} perm{k}"] = [bn, float(nulls[k])]
        if abs(bn - nulls[k]) > 1e-8:
            bad.append(f"null {s} {ver} perm{k}: {bn} vs {nulls[k]}")
    out[f"beta_obs {s} {ver}"] = [beta, ref]
    if abs(beta - ref) > 1e-8:
        bad.append(f"beta_obs {s} {ver}: {beta} vs {ref}")
    # stored maps
    key = f"E__{ver}__{s}"
    if abs(late(MP[key])[:23, :23] - Es).max() > 1e-9:
        bad.append(f"stored map {key}")

# swap test, medical and finance, raw: W_n and one floor topic, by hand
for cs in Z.COST:
    n_t, f_t = Z.NEARFAR[cs]
    si, ni, fi = Z.TI[cs], Z.TI[n_t], Z.TI[f_t]
    UN, UF = U[f"NEAR.{cs}.s0"], U[f"FAR.{cs}.s0"]
    MN, MF = MB + UN, MB + UF
    EN, EF = cmap(MN), cmap(MF)
    Dn = late(EN)[si, ni] - late(EF)[si, ni]
    ref = A["A5.4"]["tables"][cs]["D"]["raw"]["late"]["D_n"]
    out[f"D_n {cs}"] = [Dn, ref]
    if abs(Dn - ref) > 1e-9:
        bad.append(f"D_n {cs}")
    Msw = MN.copy()
    for l in range(6):
        a, b = UN[l, ni], UF[l, ni]
        Msw[l, ni] = MB[l, ni] + (a @ b) / (b @ b) * b
    Wn = late(EN)[si, ni] - late(cmap(Msw))[si, ni]
    ref = A["A5.4"]["tables"][cs]["swap"]["raw"]["late"]["W_n"]
    out[f"W_n {cs}"] = [Wn, ref]
    if abs(Wn - ref) > 1e-9:
        bad.append(f"W_n {cs}")
    Msw = MF.copy()
    for l in range(6):
        a, b = UF[l, fi], UN[l, fi]
        Msw[l, fi] = MB[l, fi] + (a @ b) / (b @ b) * b
    Wf = late(EF)[si, fi] - late(cmap(Msw))[si, fi]
    ref = A["A5.4"]["tables"][cs]["swap"]["raw"]["late"]["W_f"]
    out[f"W_f {cs}"] = [Wf, ref]
    if abs(Wf - ref) > 1e-9:
        bad.append(f"W_f {cs}")
    # joint basis W_n
    Pj = projector(np.concatenate([UN, UF], 1))
    B = np.stack([MB[l] @ Pj[l] for l in range(6)])
    AN = np.stack([UN[l] @ Pj[l] for l in range(6)])
    AF = np.stack([UF[l] @ Pj[l] for l in range(6)])
    Ej = cmap(B + AN)
    Msw = B + AN
    for l in range(6):
        a, b = AN[l, ni], AF[l, ni]
        Msw[l, ni] = B[l, ni] + (a @ b) / (b @ b) * b
    Wj = late(Ej)[si, ni] - late(cmap(Msw))[si, ni]
    ref = A["A5.4"]["tables"][cs]["swap"]["joint"]["late"]["W_n"]
    out[f"W_n joint {cs}"] = [Wj, ref]
    if abs(Wj - ref) > 1e-8:
        bad.append(f"W_n joint {cs}: {Wj} vs {ref}")
    # leak by hand (late)
    lk = np.mean([UN[l, ni] @ UN[l, si] / (UN[l, si] @ UN[l, si]) for l in range(3, 6)])
    ref = A["A5.4"]["tables"][cs]["rows"]["NEAR"]["late"]["leak_sn_uncentred"]
    out[f"leak NEAR(s,n) {cs}"] = [lk, ref]
    if abs(lk - ref) > 1e-9:
        bad.append(f"leak {cs}")

# energy removed (own) for one student, by hand
s = "CONTRAST.bad_medical.s0"
Pm = projector(U[s])
unc = np.mean([1 - ((U[s][l] @ Pm[l]) ** 2).sum() / (U[s][l] ** 2).sum() for l in range(3, 6)])
Uc = U[s] - U[s].mean(1, keepdims=True)
cen = np.mean([1 - ((Uc[l] @ Pm[l]) ** 2).sum() / (Uc[l] ** 2).sum() for l in range(3, 6)])
r = A["A5.2"]["stats"]["own"][s]["late"]
out["energy own CONTRAST(medical)"] = [[unc, cen], [r["energy_removed_uncentred"], r["energy_removed_centred"]]]
if abs(unc - r["energy_removed_uncentred"]) > 1e-9 or abs(cen - r["energy_removed_centred"]) > 1e-9:
    bad.append("energy removed")

# tau recomputed from the stored stats
G = A["A5.2"]["stats"]
for v in ("raw", "own"):
    t = max(abs(G[v][a]["late"]["beta_ent"] - G[v][c]["late"]["beta_ent"]) for a, c in Z.REPLICATES)
    if abs(t - A["A5.2"]["tau"][v]["late"]) > 1e-12:
        bad.append(f"tau {v}")
out["mismatches"] = bad
Z.write_json(Z.RES / "verify.json", out)
for k, v in out.items():
    print(k, v)
print("VERIFY", "PASS" if not bad else "FAIL")
sys.exit(5 if bad else 0)
