"""QS1-E Part B analysis (CPU), PREREG E-B EB13: E_gradfresh maps, floors, labels per source, report items, link table.

  qse_partB_analyze.py [RESULTS_DIR]     (default probe/results) -> RESULTS_DIR/partB.json, RESULTS_DIR/partB_tables.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z

RD = Path(sys.argv[1]) if len(sys.argv) > 1 else Z.QE / "probe" / "results"
S = json.loads((RD / "jobB.json").read_text())
GR = np.load(RD / "grams.npz")
A = json.loads((Z.RES / "partA.json").read_text())
NS = 24
iu = np.triu_indices(Z.NT, 1)
hn = np.array(S["halves_n"], float)                                     # 48 half sizes, order (set j, half h)
assert len(hn) == 2 * NS
W = np.zeros((NS, 2 * NS))
for j in range(NS):
    tot = hn[2 * j] + hn[2 * j + 1]
    W[j, 2 * j], W[j, 2 * j + 1] = hn[2 * j] / tot, hn[2 * j + 1] / tot   # full-set mean gradient = size-weighted mean of the halves
SHORT = {"base": "base", "PLAIN.bad_medical.s0": "PLAIN(medical)", "CONTRAST.bad_medical.s0": "CONTRAST(medical)",
         "CONTRAST.bad_medical.s1": "CONTRAST(medical) s1 (retrained)", "NEAR.bad_medical.s0": "NEAR(medical)", "FAR.bad_medical.s0": "FAR(medical)",
         "NEAR.risky_financial.s0": "NEAR(finance)", "FAR.risky_financial.s0": "FAR(finance)"}


def gcos(G):
    n = len(G)
    H = np.eye(n) - np.ones((n, n)) / n
    Gc = H @ G @ H
    d = np.sqrt(np.diag(Gc))
    return Gc / np.outer(d, d)


def maps_from(G, k, nfn):
    """G: Gram of nfn x 48 half vectors; probe/function block k -> full-set map, half-1 map, half-2 map, full-set Gram."""
    b = G[k * 2 * NS:(k + 1) * 2 * NS, k * 2 * NS:(k + 1) * 2 * NS]
    Gf = W @ b @ W.T
    h1, h2 = np.arange(0, 2 * NS, 2), np.arange(1, 2 * NS, 2)
    return gcos(Gf), gcos(b[np.ix_(h1, h1)]), gcos(b[np.ix_(h2, h2)]), Gf


def pct95(x):
    return float(np.percentile(np.abs(np.asarray(x, float)), 95))


out = {"tokens": {k: S["tokens"].get(k) for k in ("t1", "t2", "survivors", "candidates", "job1_top50_same_set_share", "job1_top50_max_abs_dp")},
       "checks": S["checks"], "cuts": S.get("cuts"), "skipped": S.get("skipped"), "gate": S.get("gate"), "finished": S.get("finished"),
       "stopped": S.get("stopped")}
models = [m for m in SHORT if f"{m}__main" in GR.files]
E, Eh = {}, {}
for m in models:
    G = GR[f"{m}__main"]
    for k, t in enumerate(("t1", "t2")):
        e, e1, e2, Gf = maps_from(G, k, 2)
        E[(m, t)], Eh[(m, t)] = e, (e1, e2)
    if m == "base":
        Gf12 = W @ G[:2 * NS, 2 * NS:] @ W.T
        Gf11 = W @ G[:2 * NS, :2 * NS] @ W.T
        Gf22 = W @ G[2 * NS:, 2 * NS:] @ W.T
        out["probe_agreement"] = {"spearman_base_maps_253": Z.spearman(E[("base", "t1")][:23, :23][iu], E[("base", "t2")][:23, :23][iu]),
                                  "per_set_cos_base_gradients": {Z.SHORT24[j]: float(Gf12[j, j] / np.sqrt(Gf11[j, j] * Gf22[j, j])) for j in range(NS)}}
        v = list(out["probe_agreement"]["per_set_cos_base_gradients"].values())
        out["probe_agreement"]["per_set_cos_min_median_max"] = [float(np.min(v)), float(np.median(v)), float(np.max(v))]
# split-half reliability per model and probe
out["split_half_spearman_253"] = {SHORT[m]: {t: Z.spearman(Eh[(m, t)][0][:23, :23][iu], Eh[(m, t)][1][:23, :23][iu]) for t in ("t1", "t2")} for m in models}
# per-student comparisons with the base
per = {}
for m in models:
    if m == "base":
        continue
    r = {}
    for t in ("t1", "t2"):
        e, eb = E[(m, t)], E[("base", t)]
        dE = e - eb
        mask = Z.pair_mask(Z.excluded(m))
        r[t] = {"spearman_with_base_253": Z.spearman(e[:23, :23][iu], eb[:23, :23][iu]),
                "spearman_everyday_row_23": Z.spearman(e[Z.EV, :23], eb[Z.EV, :23]),
                "mean_abs_dE_A52_pairs": float(np.abs(dE[:23, :23][mask]).mean()), "n_pairs": int(mask.sum())}
    per[SHORT[m]] = r
out["per_student"] = per
all_sp = [r[t]["spearman_with_base_253"] for r in per.values() for t in ("t1", "t2")]
out["every_student_spearman_ge_0.9"] = bool(min(all_sp) >= 0.9)
out["min_spearman_with_base"] = float(min(all_sp))

# floors (per probe)
fl = {}
seed_ok = ("CONTRAST.bad_medical.s1" in models and "CONTRAST.bad_medical.s0" in models)
for t in ("t1", "t2"):
    f = {}
    if seed_ok:
        si = Z.TI["bad_medical"]
        oth = [y for y in range(Z.NT) if y != si]
        f["seed_row"] = pct95(E[("CONTRAST.bad_medical.s1", t)][si, oth] - E[("CONTRAST.bad_medical.s0", t)][si, oth])
    for cs in Z.COST:
        nm, fm = Z.NEARFAR[cs]
        N, F = f"NEAR.{cs}.s0", f"FAR.{cs}.s0"
        if N not in models or F not in models:
            continue
        si = Z.TI[cs]
        rest = [y for y in range(Z.NT) if Z.TOPICS[y] not in (cs, nm, fm)]
        mask = Z.pair_mask([cs, nm, fm])
        d = E[(N, t)] - E[(F, t)]
        f[f"NEARFAR_row__{cs}"] = pct95(d[si, rest])
        f[f"NEARFAR_pair__{cs}"] = pct95(d[:23, :23][mask])
        parts = [f[f"NEARFAR_row__{cs}"], f[f"NEARFAR_pair__{cs}"]] + ([f["seed_row"]] if seed_ok else [])
        f[f"floor__{cs}"] = max(parts)
    fl[t] = f
out["floors"] = fl
out["seed_row_floor_available"] = seed_ok

# INSENSITIVE (shared)
ins = {}
if "CONTRAST.bad_medical.s0" in models and "floor__bad_medical" in fl["t1"]:
    si = Z.TI["bad_medical"]
    oth = [y for y in range(Z.NT) if y != si]
    for t in ("t1", "t2"):
        dE = E[("CONTRAST.bad_medical.s0", t)] - E[("base", t)]
        ins[t] = {"n_cells_above_floor": int((np.abs(dE[si, oth]) > fl[t]["floor__bad_medical"]).sum()),
                  "base_split_half": out["split_half_spearman_253"]["base"][t]}
    fires = any(v["n_cells_above_floor"] < 3 or v["base_split_half"] < 0.3 for v in ins.values())
else:
    fires = None
out["insensitive"] = {"per_probe": ins, "fires": fires}

# main tables and labels per source
lab = {}
tabs = {}
for cs in Z.COST:
    nm, fm = Z.NEARFAR[cs]
    si, ni, fi = Z.TI[cs], Z.TI[nm], Z.TI[fm]
    N, F = f"NEAR.{cs}.s0", f"FAR.{cs}.s0"
    rows = {}
    for key, m in (("base", "base"), ("PLAIN", f"PLAIN.{cs}.s0"), ("CONTRAST", f"CONTRAST.{cs}.s0"), ("NEAR", N), ("FAR", F)):
        if m not in models:
            rows[key] = "not run"
            continue
        rows[key] = {t: {"E_sn": float(E[(m, t)][si, ni]), "E_sf": float(E[(m, t)][si, fi]),
                         "dE_sn": float(E[(m, t)][si, ni] - E[("base", t)][si, ni]), "dE_sf": float(E[(m, t)][si, fi] - E[("base", t)][si, fi])}
                     for t in ("t1", "t2")}
    tabs[cs] = rows
    if N not in models or F not in models:
        lab[cs] = {"label": "not run (cut)"}
        continue
    D = {t: {"D_n": float(E[(N, t)][si, ni] - E[(F, t)][si, ni]), "D_f": float(E[(F, t)][si, fi] - E[(N, t)][si, fi]), "floor": fl[t][f"floor__{cs}"]}
         for t in ("t1", "t2")}
    if fires:
        label = "INSENSITIVE"
    elif all(abs(D[t]["D_n"]) <= D[t]["floor"] for t in D):
        label = "VETO"
    elif all(D[t]["D_n"] < -D[t]["floor"] for t in D):
        label = "SEPARATED"
    elif all(D[t]["D_n"] > D[t]["floor"] for t in D):
        label = "CLOSER"
    else:
        label = "MIXED"
    lab[cs] = {"label": label, "D": D}
out["tables"] = tabs
out["labels"] = lab

# CONTRAST(medical)'s source row against Part A's leak share and dL_gate
if "CONTRAST.bad_medical.s0" in models:
    s = "CONTRAST.bad_medical.s0"
    si = Z.TI["bad_medical"]
    oth = [y for y in range(Z.NT) if y != si]
    ra = A["A5.3"][s]["late"]
    names = [Z.SHORT24[y] for y in oth]
    lk_u = [ra["leak_uncentred"][n] for n in names]
    lk_c = [ra["leak_centred"][n] for n in names]
    gate = [ra["gate"][n] for n in names]
    dEA = [ra["dE_raw"][n] for n in names]
    cr = {}
    for t in ("t1", "t2"):
        dE = (E[(s, t)] - E[("base", t)])[si, oth]
        cr[t] = {"row": {n: float(v) for n, v in zip(names, dE)}, "spearman_leak_uncentred": Z.spearman(dE, lk_u),
                 "spearman_leak_centred": Z.spearman(dE, lk_c), "spearman_gate": Z.spearman(dE, gate), "spearman_partA_dE": Z.spearman(dE, dEA)}
    out["contrast_medical_row"] = cr

# exploratory: alternative functions / probe A
ex = {}
for k in GR.files:
    if "__explo_" not in k:
        continue
    m, rest = k.split("__explo_")
    e, e1, e2, _ = maps_from(GR[k], 0, 1)
    ex[(m, rest)] = e
exo = {}
for rest in sorted({r for _, r in ex}):
    if ("base", rest) not in ex:
        if rest.startswith("own"):
            exo[rest] = {SHORT[m]: {"spearman_with_main_t1_map": Z.spearman(e[:23, :23][iu], E[(m, "t1")][:23, :23][iu])} for (m, r), e in ex.items() if r == rest}
        continue
    r = {"base_spearman_with_main_t1": Z.spearman(ex[("base", rest)][:23, :23][iu], E[("base", "t1")][:23, :23][iu])}
    for cs in Z.COST:
        nm, fm = Z.NEARFAR[cs]
        si, ni = Z.TI[cs], Z.TI[nm]
        N, F = f"NEAR.{cs}.s0", f"FAR.{cs}.s0"
        if (N, rest) in ex and (F, rest) in ex:
            r[f"D_n__{cs}"] = float(ex[(N, rest)][si, ni] - ex[(F, rest)][si, ni])
            r[f"E_sn__{cs}"] = {"base": float(ex[("base", rest)][si, ni]), "NEAR": float(ex[(N, rest)][si, ni]), "FAR": float(ex[(F, rest)][si, ni])}
    exo[rest] = r
out["exploratory"] = exo

# the autocast condition (PREREG Part F): measurements after the first training ran under accelerate's bf16 autocast wrapper, the base and the
# kept adapters before it. Bound: the base's exploratory t1 map with probe A s21 (after training, autocast) against its main t1 map (A s20, before).
if "base__explo_t1_A21" in GR.files:
    e21, _, _, _ = maps_from(GR["base__explo_t1_A21"], 0, 1)
    e20 = E[("base", "t1")]
    out["autocast_bound_base_t1"] = {"spearman_253": Z.spearman(e21[:23, :23][iu], e20[:23, :23][iu]), "max_abs_diff": float(np.abs(e21 - e20).max()),
                                     "mean_abs_diff_253": float(np.abs(e21[:23, :23][iu] - e20[:23, :23][iu]).mean()),
                                     "E_med_rehab": [float(e20[Z.TI["bad_medical"], Z.TI["bad_rehab"]]), float(e21[Z.TI["bad_medical"], Z.TI["bad_rehab"]])],
                                     "E_fin_gambling": [float(e20[Z.TI["risky_financial"], Z.TI["bad_gambling"]]), float(e21[Z.TI["risky_financial"], Z.TI["bad_gambling"]])],
                                     "note": "confounded with the change of probe A (s20 -> s21): an upper bound on the autocast effect"}
# official status: B4 says stop Part B if a check fails -> no label; the rule's output is reported as a description only
b42 = S["checks"].get("B4.2_near")
stopped = S.get("stopped")
out["official_status"] = (f"STOPPED: {stopped}; no label (B4)" if stopped else "labels as computed")
out["rule_output_descriptive"] = {cs: lab[cs]["label"] for cs in lab}
out["without_insensitive_descriptive"] = {}
for cs in lab:
    if "D" not in lab[cs]:
        continue
    D = lab[cs]["D"]
    if all(abs(D[t]["D_n"]) <= D[t]["floor"] for t in D):
        w = "VETO"
    elif all(D[t]["D_n"] < -D[t]["floor"] for t in D):
        w = "SEPARATED"
    elif all(D[t]["D_n"] > D[t]["floor"] for t in D):
        w = "CLOSER"
    else:
        w = "MIXED"
    out["without_insensitive_descriptive"][cs] = w
if "CONTRAST.bad_medical.s0" in models:
    third = {}
    for t in ("t1", "t2"):
        v = sorted((abs(x) for x in out["contrast_medical_row"][t]["row"].values()), reverse=True)
        third[t] = {"third_largest_abs": v[2], "floor": fl[t]["floor__bad_medical"], "gap": v[2] - fl[t]["floor__bad_medical"]}
    out["insensitive"]["third_cell"] = third

# link table (E6) and predictions
ladder = {cs: A["A5.4"]["tables"][cs]["ladder"] for cs in Z.COST}
out["link"] = {cs: {"partA_ladder": ladder[cs], "partB_label": ("no label (B4.2 stop)" if stopped else lab[cs]["label"]),
                    "rule_output": lab[cs]["label"], "without_insensitive": out["without_insensitive_descriptive"].get(cs),
                    "D_n_t1": lab[cs].get("D", {}).get("t1", {}).get("D_n"), "D_n_t2": lab[cs].get("D", {}).get("t2", {}).get("D_n"),
                    "floor_t1": lab[cs].get("D", {}).get("t1", {}).get("floor"), "floor_t2": lab[cs].get("D", {}).get("t2", {}).get("floor")}
               for cs in Z.COST}
Z.write_json(RD / "partB.json", out)
print(json.dumps({"tokens": [out["tokens"]["t1"], out["tokens"]["t2"]], "labels": {cs: lab[cs]["label"] for cs in lab}, "D": {cs: lab[cs].get("D") for cs in lab},
                  "floors": fl, "insensitive": out["insensitive"], "split_half": out["split_half_spearman_253"].get("base"),
                  "probe_agreement": out.get("probe_agreement", {}).get("spearman_base_maps_253"), "min_spearman_with_base": out["min_spearman_with_base"]},
                 indent=1))
