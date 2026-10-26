"""QS1-E Part C (CPU, optional): Part A's entanglement analysis on ES1's EM twins, with ES1's definitions (PREREG B11).

  qse_partC.py -> es1_ladder/partC.json, es1_ladder/maps_partC.npz

ES1's states: es1/stats/<model>.pt "h6" {set: [24, 6, 5120] bf16} (last template token, block outputs, hidden_states[L + 1]); read-only.
Maps: per layer, full-set means of the 23 gen24 sets, centred by the mean of the 23 topic means, cosine; late = mean of 22/26/30.
Check (stop Part C on failure): per twin, mean over the 231 untrained pairs of |dE| per layer, mean over the late layers; median over the 23
broad / 23 narrow twins within 0.005 of ES1's summary.json points (0.0875 / 0.2823).
Versions: raw; perp-own,3 (the twin's 23 x 5120 update matrix, uncentred SVD); perp-hub (per layer, the mean over the 23 broad twins of
their mean update over the 23 sets, one direction).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z

torch.set_num_threads(4)
T0 = time.time()
ES = Z.ROOT / "es1"
OUT = Z.QE / "es1_ladder"
MAIN = [f"gen24-{d}" for d in Z.TOPICS]
NPERM = 1000
REF = json.loads((ES / "results" / "summary.json").read_text())["results"]


def log(msg):
    print(f"[partC {time.time() - T0:6.1f}s] {msg}", flush=True)


def means(model):
    S = torch.load(ES / "stats" / f"{model}.pt", map_location="cpu", weights_only=False)
    assert S["model"] == model
    M = np.zeros((6, 23, Z.D_MODEL))
    for j, s in enumerate(MAIN):
        x = S["h6"][s]
        assert tuple(x.shape) == (24, 6, Z.D_MODEL) and x.dtype == torch.bfloat16, (model, s, x.shape)
        M[:, j] = x.double().mean(0).numpy()                          # [6, d]: full-set mean per hooked layer
    return M, S["ids"]


def emap23(M):
    return Z.emap(M)                       # centring over the 23 sets = ES1's centring (no everyday set here)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MB, ids_b = means("base")
    twins = [f"em-{d}-{k}" for d in Z.TOPICS for k in ("broad", "narrow")]
    D = {}
    for m in twins:
        Mm, ids_m = means(m)
        for s in MAIN:
            assert ids_m[s] == ids_b[s], (m, s)
        D[m] = Mm - MB
    log(f"set means for base + {len(twins)} twins loaded")
    Eb = emap23(MB)
    iu = np.triu_indices(23, 1)
    # ---------------- check
    chk = {}
    per_twin_es1 = {}
    for m in twins:
        T = Z.TI[m[3:].rsplit("-", 1)[0]]
        Em = emap23(MB + D[m])
        mask = Z.pair_mask([Z.TOPICS[T]])
        per_layer = np.array([np.abs((Em[l] - Eb[l])[mask]).mean() for l in range(6)])
        per_twin_es1[m] = {b: float(per_layer[Z.BLOCKS[b]].mean()) for b in ("late", "early")}
    for b in ("late", "early"):
        for k in ("broad", "narrow"):
            med = float(np.median([per_twin_es1[m][b] for m in twins if m.endswith(k)]))
            ref = REF[b]["desc_dE_untrained_absmean"][f"median_{k}"]["point"]
            chk[f"{b}_{k}"] = {"recomputed": med, "es1": ref, "abs_diff": abs(med - ref)}
    ok = all(chk[f"late_{k}"]["abs_diff"] <= 0.005 for k in ("broad", "narrow"))
    log(f"check: late broad {chk['late_broad']['recomputed']:.4f} vs {chk['late_broad']['es1']:.4f}; narrow {chk['late_narrow']['recomputed']:.4f} "
        f"vs {chk['late_narrow']['es1']:.4f} -> {'PASS' if ok else 'FAIL (stop Part C)'}")
    res = {"check": chk, "check_pass": ok, "check_per_twin": per_twin_es1}
    if not ok:
        Z.write_json(OUT / "partC.json", res)
        sys.exit(4)
    # ---------------- versions
    hub = np.mean([D[m].mean(1) for m in twins if m.endswith("broad")], 0)        # [6, d]
    Vhub = (hub / np.linalg.norm(hub, axis=1, keepdims=True))[:, None, :]
    rng = np.random.default_rng(0)
    perms = np.stack([rng.permutation(23) for _ in range(NPERM)])
    maps = {"E__raw__base": Eb, "perms": perms, "hub": hub}
    stats = {}
    for m in twins:
        Tn = m[3:].rsplit("-", 1)[0]
        T = Z.TI[Tn]
        mask = Z.pair_mask([Tn])
        oth = [y for y in range(23) if y != T]
        order = sorted(oth, key=lambda y: -Z.blk(Eb, "late")[T, y])
        near3, far3 = order[:3], order[-3:]
        st = {}
        for v in ("raw", "own", "hub"):
            Vb = None if v == "raw" else (Z.svd_basis(D[m], 3) if v == "own" else Vhub)
            B = Z.project(Vb, MB)
            Uv = Z.project(Vb, D[m])
            Ebv, Emv = Z.emap(B), Z.emap(B + Uv)
            Gbb, Gbu, Guu = (np.einsum("lxd,lyd->lxy", a, c) for a, c in ((B, B), (B, Uv), (Uv, Uv)))
            nrm = None if v == "raw" else np.linalg.norm(Uv, axis=2)
            Ep = Z.gram_emap(Z.perm_gram(Gbb, Gbu, Guu, perms, nrm))
            Eid = Z.gram_emap(Z.perm_gram(Gbb, Gbu, Guu, np.arange(23)[None], nrm))[0]
            assert np.abs(Eid - Emv).max() < 1e-9
            r = {}
            for b in ("late", "early"):
                eb, em = Z.blk(Ebv, b), Z.blk(Emv, b)
                dE = em - eb
                x, y = eb[mask], dE[mask]
                beta = float(Z.ols_slope(x, y))
                yn = (Z.blk(Ep, b) - eb[None])[:, mask]
                bn = Z.ols_slope(np.broadcast_to(x, yn.shape), yn)
                per_layer_abs = np.array([np.abs((Emv[l] - Ebv[l])[mask]).mean() for l in Z.BLOCKS[b]]).mean()
                r[b] = {"mean_abs_dE_es1def": float(per_layer_abs), "mean_abs_dE_blockavg": float(np.abs(y).mean()),
                        "near3_minus_far3": float(dE[T, near3].mean() - dE[T, far3].mean()), "beta_obs": beta, "beta_null_mean": float(bn.mean()),
                        "null_p2.5": float(np.percentile(bn, 2.5)), "null_p97.5": float(np.percentile(bn, 97.5)), "beta_ent": beta - float(bn.mean()),
                        "above": bool(beta > np.percentile(bn, 97.5)), "below": bool(beta < np.percentile(bn, 2.5))}
            if v != "raw":
                unc, cen = Z.energy_removed(Vb, D[m])
                for b in ("late", "early"):
                    r[b]["energy_removed_uncentred"] = float(unc[Z.BLOCKS[b]].mean())
                    r[b]["energy_removed_centred"] = float(cen[Z.BLOCKS[b]].mean())
            st[v] = r
            maps[f"E__{v}__{m}"] = Emv
            if v != "raw":
                maps[f"Ebase__{v}__{m}"] = Ebv
        stats[m] = st
    log("per-twin statistics done")
    summ = {}
    for k in ("broad", "narrow"):
        ms = [m for m in twins if m.endswith(k)]
        summ[k] = {}
        for v in ("raw", "own", "hub"):
            sv = {}
            for b in ("late", "early"):
                g = lambda key: np.array([stats[m][v][b][key] for m in ms])
                n_above, n_below = int(g("above").sum()), int(g("below").sum())
                lab = "MORE" if n_above >= 18 else ("LESS" if n_below >= 18 else "UNCHANGED")
                sv[b] = {"median_mean_abs_dE_es1def": float(np.median(g("mean_abs_dE_es1def"))), "mean_mean_abs_dE_es1def": float(g("mean_abs_dE_es1def").mean()),
                         "median_mean_abs_dE_blockavg": float(np.median(g("mean_abs_dE_blockavg"))),
                         "median_near3_minus_far3": float(np.median(g("near3_minus_far3"))), "n_near3_minus_far3_positive": int((g("near3_minus_far3") > 0).sum()),
                         "n_above_97.5": n_above, "n_below_2.5": n_below, "median_beta_ent": float(np.median(g("beta_ent"))),
                         "median_beta_obs": float(np.median(g("beta_obs"))), "median_beta_null_mean": float(np.median(g("beta_null_mean"))), "label": lab}
                if v != "raw":
                    sv[b]["median_energy_removed_centred"] = float(np.median(g("energy_removed_centred")))
                    sv[b]["median_energy_removed_uncentred"] = float(np.median(g("energy_removed_uncentred")))
            summ[k][v] = sv
            L = sv["late"]
            log(f"{k:6s} {v:4s}: median mean|dE| {L['median_mean_abs_dE_es1def']:.4f}; near3-far3 {L['median_near3_minus_far3']:+.4f}; "
                f"above {L['n_above_97.5']}/23, below {L['n_below_2.5']}/23 -> {L['label']}; median beta_ent {L['median_beta_ent']:+.4f}"
                + (f"; centred energy removed {L['median_energy_removed_centred']:.3f}" if v != "raw" else ""))
    res.update({"summary": summ, "per_twin": stats, "hub_norm_late": float(np.linalg.norm(hub[Z.LATE_I], axis=1).mean()), "elapsed_s": time.time() - T0})
    Z.write_json(OUT / "partC.json", res)
    np.savez_compressed(OUT / "maps_partC.npz", **maps)
    log(f"wrote es1_ladder/partC.json and maps_partC.npz ({(OUT / 'maps_partC.npz').stat().st_size / 2 ** 20:.1f} MB)")


if __name__ == "__main__":
    main()
