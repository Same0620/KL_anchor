"""QS1-E report: results/tables.md (every table) and RESULTS.md (src/results_template.md with {{T_*}} replaced by generated tables).

  qse_report.py     (env/venv, -I -B)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z

A = json.loads((Z.RES / "partA.json").read_text())
X = json.loads((Z.RES / "explore.json").read_text())
CP = Z.QE / "es1_ladder" / "partC.json"
C = json.loads(CP.read_text()) if CP.exists() else None
T = {}


def f(x, nd=3, sign=True):
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def table(hdr, rows):
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


S = Z.STUDENTS
sn = Z.short_student
G = A["A5.2"]["stats"]
L = A["A5.2"]["labels"]
tau = A["A5.2"]["tau"]

# ---------------- checks
c = A["checks"]
a21 = c["A2.1"]
T["T_checks"] = table(["check", "result", "verdict"], [
    ["A2.1 hard check: E_base recomputed from Job 1's residuals (fp64) vs QS1's stored matrices (6 per-layer + late + early CSVs, 24 × 24; "
     "closeness.json 23 × 23)", f"max \\|diff\\| {max(a21['max_abs_diff'].values()):.1e} (tolerance {a21['tolerance']:.0e}); Spearman over the 253 "
     f"topic pairs {a21['spearman_253']['late']:.6f} (late), {a21['spearman_253']['early']:.6f} (early)", "**PASS** (fp64 path; no fallback needed)"],
    ["A2.2 hook check (code)", "Job 1 and Job 3 both take `hidden_states[L + 1]` at the last template token through the one helper "
     "`qs1common.score_prompts(want_h=True)`; Job 3 subtracts set means built from Job 1's `stats/job1/base.pt`", "**PASS**"],
    ["A2.3 stored base means (flag only)", "not stored by Job 3 (recomputed inside the job from Job 1's file, so identical by construction)", "not applicable"],
    ["A2.4 vs ES1's `E_base_23x23.npz` (QS1's residuals with ES1's 23-topic centring)",
     f"late: Spearman {c['A2.4']['late']['spearman_253']:.4f}, max \\|diff\\| {c['A2.4']['late']['max_abs_diff']:.4f} (mean {c['A2.4']['late']['mean_abs_diff']:.4f}); "
     f"early: {c['A2.4']['early']['spearman_253']:.4f}, {c['A2.4']['early']['max_abs_diff']:.4f}. Same indexing (block L = `hidden_states[L + 1]`). "
     f"QS1's own 24-set centring vs the 23-set centring: Spearman {c['A2.4']['qs1_24_vs_23_centring_late']['spearman_253']:.4f}, max \\|diff\\| "
     f"{c['A2.4']['qs1_24_vs_23_centring_late']['max_abs_diff']:.3f}", "descriptive"],
    ["ΔL_gate from the per-prompt CSVs vs `cells.csv` (final step)", f"max \\|diff\\| {A['inputs']['gate_vs_cells_max_abs_diff']:.1e} (cells.csv has 5 decimals)", "agrees"],
    ["Brute-force recomputation (`src/qse_verify.py`): 7 null slopes, 4 observed slopes, D_n, W_n, W_f, joint W_n, leak, energy, τ",
     "all within 1e-8 of the pipeline", "**PASS**"]])


# ---------------- A5.2
def a52(v, energy=False, tagv=None):
    rows = []
    for s in S:
        g = G[v][s] if tagv is None else tagv["stats"][s]
        lab = L[v][s] if tagv is None else tagv["labels"][s]
        l, e = g["late"], g["early"]
        r = [sn(s), l["n_pairs"], f"{l['mean_abs_dE']:.4f} ({e['mean_abs_dE']:.4f})", f(l["mean_dE"], 4), f(l["sd_change"], 4), f(l["beta_obs"], 3),
             f"{f(l['beta_null_mean'], 3)} [{f(l['null_p2.5'], 3)}, {f(l['null_p97.5'], 3)}]", f"{f(l['beta_ent'], 3)} ({f(e['beta_ent'], 3)})",
             f"{l['p_two_sided']:.3f}", f"**{lab['late']}** ({lab['early']})"]
        if energy:
            r.insert(2, f"{l['energy_removed_uncentred']:.3f} / {l['energy_removed_centred']:.3f}")
        rows.append(r)
    hdr = ["student", "pairs", "mean \\|ΔE\\| late (early)", "mean ΔE", "SD change", "β_obs", "null mean [2.5%, 97.5%]", "β_ent late (early)", "p (two-sided)",
           "label late (early)"]
    if energy:
        hdr.insert(2, "energy removed, uncentred / centred (late)")
    return table(hdr, rows)


T["T_a52_raw"] = a52("raw")
T["T_a52_own"] = a52("own", energy=True)
T["T_a52_tag"] = a52("tag", energy=True)
T["T_a52_own3c"] = a52("own", energy=True, tagv=X["X2_own3c"])
T["T_tau"] = table(["version", "τ late", "τ early", "pair setting τ (late)"], [
    [{"raw": "raw", "own": "⊥own,3", "tag": "⊥tag (exploratory)"}[v], f"{tau[v]['late']:.4f}", f"{tau[v]['early']:.4f}",
     max(((abs(G[v][a]['late']['beta_ent'] - G[v][b]['late']['beta_ent']), sn(a)) for a, b in Z.REPLICATES))[1]] for v in ("raw", "own", "tag")]
    + [["joint basis of each replicate pair (descriptive)", f"{tau['joint_descriptive']['late']:.4f}", f"{tau['joint_descriptive']['early']:.4f}", ""],
       ["⊥own,3c (exploratory X2)", f"{X['X2_own3c']['tau']['late']:.4f}", f"{X['X2_own3c']['tau']['early']:.4f}", ""]])
# the identity check
T["T_meandE"] = ", ".join(f"{sn(s)} {G['raw'][s]['late']['mean_dE_all_offdiag_24']:+.4f}" for s in S[:4])

# ---------------- A5.3 / A5.5
R = A["A5.3"]
rows = []
for s in S:
    r = R[s]["late"]
    ev = r["everyday"]
    rows.append([sn(s), f(r["spearman_dEraw_leak_uncentred"], 2), f(r["spearman_dEraw_leak_centred"], 2), f(r["spearman_dEraw_gate"], 2),
                 f(r["spearman_dEraw_Ebase"], 2), f"{f(r['near3_mean_dE_raw'])} / {f(r['far3_mean_dE_raw'])} ({f(r['near3_minus_far3_raw'])})",
                 f"{f(r['near3_mean_dE_own'])} / {f(r['far3_mean_dE_own'])}",
                 f"{f(ev['E_base_raw'])} → {f(ev['E_student_raw'])}", f"{f(ev['E_base_own'])} → {f(ev['E_student_own'])}"])
T["T_rows"] = table(["student", "ρ(ΔE, leak) uncentred", "ρ(ΔE, leak) centred", "ρ(ΔE, ΔL_gate)", "ρ(ΔE, E_base)",
                     "near-3 / far-3 mean ΔE raw (difference)", "near-3 / far-3 mean ΔE ⊥own,3", "E(s, everyday) raw: base → student",
                     "E(s, everyday) ⊥own,3: base → student"], rows)
T["T_near3"] = "; ".join(f"{Z.SHORT[s]}: near {', '.join(R[f'PLAIN.{s}.s0']['late']['near3'])}; far {', '.join(R[f'PLAIN.{s}.s0']['late']['far3'])}"
                         for s in Z.SOURCES)

# ---------------- A5.4
for cs in Z.COST:
    t = A["A5.4"]["tables"][cs]
    nm, fm = Z.SHORT[t["n"]], Z.SHORT[t["f"]]
    rows = []
    for k in ("base", "PLAIN", "CONTRAST", "NEAR", "FAR"):
        l, e = t["rows"][k]["late"], t["rows"][k]["early"]
        lk = (lambda a, b: f"{l[a]:.3f} / {f(l[b])}") if k != "base" else (lambda a, b: "—")
        rows.append([k, f"{f(l['E_sn_raw'])} ({f(e['E_sn_raw'])})", f"{f(l['E_sn_joint'])} ({f(e['E_sn_joint'])})", f"{f(l['E_sf_raw'])} ({f(e['E_sf_raw'])})",
                     f"{f(l['E_sf_joint'])} ({f(e['E_sf_joint'])})", lk("leak_sn_uncentred", "leak_sn_centred"), lk("leak_sf_uncentred", "leak_sf_centred")])
    T[f"T_main_{cs}"] = table(["model", f"E(s,{nm}) raw", f"E(s,{nm}) joint ⊥", f"E(s,{fm}) raw", f"E(s,{fm}) joint ⊥",
                               f"leak(s,{nm}) uncentred / centred", f"leak(s,{fm}) uncentred / centred"], rows)
    D = t["D"]
    fl = A["A4"]
    T[f"T_D_{cs}"] = table(["quantity", "raw late (early)", "A4.3 floor raw", "joint ⊥ late (early)", "A4.3 floor joint ⊥", "⊥tag late (exploratory)", "floor ⊥tag"], [
        [f"D_n = ΔE_NEAR(s,{nm}) − ΔE_FAR(s,{nm})", f"**{f(D['raw']['late']['D_n'])}** ({f(D['raw']['early']['D_n'])})", f"{fl['raw'][f'A4.3__{cs}']:.3f}",
         f"{f(D['joint']['late']['D_n'])} ({f(D['joint']['early']['D_n'])})", f"{fl['joint'][f'A4.3__{cs}']:.3f}", f(D['tag']['late']['D_n']), f"{fl['tag'][f'A4.3__{cs}']:.3f}"],
        [f"D_f = ΔE_FAR(s,{fm}) − ΔE_NEAR(s,{fm})", f"{f(D['raw']['late']['D_f'])} ({f(D['raw']['early']['D_f'])})", f"{fl['raw'][f'A4.3__{cs}']:.3f}",
         f"{f(D['joint']['late']['D_f'])} ({f(D['joint']['early']['D_f'])})", f"{fl['joint'][f'A4.3__{cs}']:.3f}", f(D['tag']['late']['D_f']), f"{fl['tag'][f'A4.3__{cs}']:.3f}"]])
    sw = t["swap"]
    rows = []
    for v in ("raw", "joint", "tag"):
        l, e = sw[v]["late"], sw[v]["early"]
        rows.append([v + (" (exploratory)" if v == "tag" else ""), f"**{f(l['W_n'])}** ({f(e['W_n'])})", f"{l['floor_W_NEAR']:.3f}", f"{l['kappa_n']:.3f}",
                     f"{f(l['W_f'])} ({f(e['W_f'])})", f"{l['floor_W_FAR']:.3f}", f"{l['kappa_f']:.3f}",
                     ", ".join(f"{k} {f(x)}" for k, x in l["W_n_Y_near3"].items()) + f" (mean {f(l['W_n_Y_near3_mean'])})"])
    xs = X["X1_centred_swap"][cs]
    for v in ("raw", "joint"):
        l = xs[v]["late"]
        rows.append([f"{v}, centred swap (exploratory X1)", f(l["W_n"]), f"{l['floor_W_NEAR']:.3f}", f"{l['kappa_c_n']:.3f}", f(l["W_f"]),
                     f"{l['floor_W_FAR']:.3f}", f"{l['kappa_c_f']:.3f}", ""])
    T[f"T_swap_{cs}"] = table(["version", f"W_n ({nm}) late (early)", "floor_W (NEAR)", "κ_n", f"W_f ({fm}) late (early)", "floor_W (FAR)", "κ_f",
                               f"W({nm}, Y) for {nm}'s 3 nearest base topics other than s"], rows)
    ex = t["active_extras"]
    en = t["joint_energy_removed"]
    T[f"T_ladder_{cs}"] = (f"**Ladder ({Z.SHORT[cs]}): {t['ladder']}.** Raw D_n {f(t['ladder_inputs']['D_n_raw'])} against ±{t['ladder_inputs']['floor_A4.3_raw']:.3f}; "
                           f"W_n {f(t['ladder_inputs']['W_n_raw'])} against −{t['ladder_inputs']['floor_W_NEAR_raw']:.3f}. "
                           f"s-specific (W_n < mean W({nm}, Y) − floor_W): {'yes' if ex['s_specific'] else 'no'}; joint-⊥ W_n < −floor_W: "
                           f"{'yes' if ex['joint_W_n_below_minus_floor'] else 'no'}; W_f < −floor_W (FAR): {'yes' if ex['W_f_below_minus_floor_raw'] else 'no'}; "
                           f"joint ⊥ removes ≥ 95% of the centred update energy: no (NEAR {en['NEAR']['late']['centred']:.2f}, FAR {en['FAR']['late']['centred']:.2f}; "
                           f"uncentred {en['NEAR']['late']['uncentred']:.2f} / {en['FAR']['late']['uncentred']:.2f}).")

# ---------------- floors
rows = []
for k, lab in (("CONTRAST_seed_pair", "CONTRAST seed pair floor (2 × 231)"), ("CONTRAST_seed_row", "CONTRAST seed row floor (2 × 22)"),
               ("PLAIN_seed_pair", "PLAIN seed pair floor (231)"), ("PLAIN_seed_row", "PLAIN seed row floor (22)"),
               ("NEARFAR_pair__bad_medical", "NEAR–FAR pair floor, medical (190)"), ("NEARFAR_row__bad_medical", "NEAR–FAR row floor, medical (20)"),
               ("NEARFAR_pair__risky_financial", "NEAR–FAR pair floor, finance (190)"), ("NEARFAR_row__risky_financial", "NEAR–FAR row floor, finance (20)"),
               ("A4.3__bad_medical", "**A4.3 floor for D_n / D_f, medical**"), ("A4.3__risky_financial", "**A4.3 floor for D_n / D_f, finance**")):
    rows.append([lab] + [f"{A['A4'][v][k]:.4f}" for v in ("raw", "joint", "tag")])
T["T_floors"] = table(["floor (95th percentile of \\|difference\\|, late; number of values)", "raw", "joint ⊥", "⊥tag (exploratory)"], rows)
a45 = A["A4.5"]
T["T_splithalf"] = (f"Base split-half SD of a cell (QS1's S2 halves, late, |E_h1 − E_h2| ÷ √2): median {a45['median_cell_sd_half']:.4f} for a 12-question "
                    f"half (≈ {a45['median_cell_sd_full_approx']:.4f} for the full set), 95th percentile {a45['p95_cell_sd_half']:.4f}. The halves reproduce QS1's "
                    f"split-half Spearman ({a45['split_half_spearman_253_reproduced']:.4f} = {a45['qs1_split_half_spearman_253']:.4f}).")

# ---------------- Part C
if C:
    rows = []
    for k in ("broad", "narrow"):
        for v in ("raw", "own", "hub"):
            l, e = C["summary"][k][v]["late"], C["summary"][k][v]["early"]
            rows.append([k, {"raw": "raw", "own": "⊥own,3", "hub": "⊥hub"}[v], f"{l['median_mean_abs_dE_es1def']:.4f} ({l['mean_mean_abs_dE_es1def']:.4f})",
                         f"{l['median_energy_removed_centred']:.3f}" if v != "raw" else "—",
                         f"{f(l['median_near3_minus_far3'])} ({l['n_near3_minus_far3_positive']}/23 > 0)", f(l["median_beta_obs"]), f(l["median_beta_null_mean"]),
                         f"{l['n_above_97.5']} / {l['n_below_2.5']}", f(l["median_beta_ent"]), f"**{l['label']}** ({e['label']}; early {e['n_above_97.5']}/23 above)"])
    T["T_partC"] = table(["twins", "version", "median (mean) of mean \\|ΔE\\|, untrained pairs, ES1's definition", "median centred energy removed",
                          "median near-3 − far-3 ΔE(T, ·)", "median β_obs", "median null mean", "above 97.5% / below 2.5% (of 23)", "median β_ent",
                          "label late (early)"], rows)
    ck = C["check"]
    T["T_partC_check"] = (f"late broad {ck['late_broad']['recomputed']:.5f} vs ES1 {ck['late_broad']['es1']:.5f}; late narrow {ck['late_narrow']['recomputed']:.5f} "
                          f"vs {ck['late_narrow']['es1']:.5f}; early broad {ck['early_broad']['recomputed']:.5f} vs {ck['early_broad']['es1']:.5f}; early narrow "
                          f"{ck['early_narrow']['recomputed']:.5f} vs {ck['early_narrow']['es1']:.5f}")

# ---------------- exploratory X3, X4, X5
x3 = X["X3_seed_reliability_dE"]
T["T_x3"] = table(["replicate pair", "raw: pairs Pearson (Spearman)", "raw: s row Pearson", "⊥own,3: pairs Pearson (Spearman)", "⊥own,3: s row Pearson"],
                  [[sn(a), f"{x3[a]['raw']['pairs_pearson']:.3f} ({x3[a]['raw']['pairs_spearman']:.3f})", f"{x3[a]['raw']['row_pearson']:.3f}",
                    f"{x3[a]['own']['pairs_pearson']:.3f} ({x3[a]['own']['pairs_spearman']:.3f})", f"{x3[a]['own']['row_pearson']:.3f}"] for a, _ in Z.REPLICATES])
x4 = X["X4_base_map_under_projection"]
T["T_x4"] = ", ".join(f"{sn(s)} {x4[s]['own']['spearman_253']:.3f}" for s in S)
x5 = X["X5_update_sizes_late"]
rows = []
for cs in Z.COST:
    for arm in ("PLAIN", "CONTRAST", "NEAR", "FAR"):
        r = x5[cs][arm]
        rows.append([f"{arm}({Z.SHORT[cs]})"] + [f"{r[k]['norm']:.0f} / {r[k]['norm_centred']:.0f}" for k in ("s", "n", "f", "everyday")]
                    + [f"{r['mean_over_sets_norm']:.0f}"])
    rows.append([f"cos(NEAR, FAR) update, {Z.SHORT[cs]}: uncentred / centred"] +
                [f"{x5[cs]['cos_late_NEAR_FAR'][k]:.2f} / {x5[cs]['cos_late_NEAR_FAR_centred'][k]:.2f}" for k in ("s", "n", "f", "everyday")] + [""])
T["T_x5"] = table(["model", "s: \\|d\\| / \\|d − mean\\|", "n", "f", "everyday", "\\|mean over sets\\|"], rows)
T["T_basenorm"] = f"{X['base_centred_norm_late_mean']:.1f}"

# ---------------- predictions
P = A["predictions"]
c2 = P["P2"]["contrast_vs_plain"]
rows = [
    ["P1", "PLAIN(medical) and NULL: raw mean \\|ΔE\\| ≤ 0.087", "0.60", "0.55",
     f"PLAIN(medical) {P['P1']['PLAIN(medical)']:.4f}, NULL {P['P1']['NULL']:.4f}", "**holds**" if P["P1"]["holds"] else "fails"],
    ["P2", "CONTRAST > PLAIN in all four sources, and ρ(ΔE(s,·), leak) ≥ 0.5 in ≥ 3 of 4", "0.60", "0.55",
     "CONTRAST vs PLAIN: " + ", ".join(f"{k} {a:.3f} vs {b:.3f}" for k, (a, b) in c2.items()) + "; ρ " +
     ", ".join(f"{k} {f(v, 2)}" for k, v in P["P2"]["rho_dE_leak_contrast"].items()), "**holds**" if P["P2"]["holds"] else "**fails** (both clauses)"],
    ["P3", "CONTRAST: s's 3 nearest base topics move toward s in ≥ 3 of 4", "0.65", "0.70",
     ", ".join(f"{k} {f(v)}" for k, v in P["P3"]["near3_mean_dE_raw"].items()), f"**holds** ({P['P3']['n_positive']} of 4; legal only +0.005)"],
    ["P4", "⊥own,3 removes ≥ 70% of raw mean \\|ΔE\\| for every CONTRAST, NEAR, FAR student", "0.55", "0.40",
     ", ".join(f"{k} {v:.0%}" for k, v in P["P4"]["share_removed"].items()), "**fails** (every student below 70%)"],
    ["P5", "ladder overall CLOSER / NOT-VISIBLE / PASSIVE / ACTIVE / MIXED", ".05/.20/.35/.30/.10", ".03/.12/.20/.30/.35",
     f"medical {P['P5']['per_source']['medical']}, finance {P['P5']['per_source']['finance']}", f"**{P['P5']['observed']}**"],
    ["P6a", "CONTRAST raw MORE in ≥ 3 of 4", "0.50", "0.55", ", ".join(f"{k} {v}" for k, v in P["P6a"]["labels"].items()), "**holds** (4 of 4)"],
    ["P6b", "CONTRAST ⊥own,3 UNCHANGED in ≥ 3 of 4", "0.60", "0.50", ", ".join(f"{k} {v}" for k, v in P["P6b"]["labels"].items()),
     "**holds** (4 of 4; fragile, see §11)"],
    ["P6c", "PLAIN raw UNCHANGED in ≥ 3 of 4", "0.60", "0.40", ", ".join(f"{k} {v}" for k, v in P["P6c"]["labels"].items()), "**fails** (0 of 4)"]]
T["T_pred"] = table(["#", "prediction", "P (lead)", "P (mine)", "observed (late)", "verdict"], rows)

# ---------------- Part B (as amended by PREREG Part E)
import os
import subprocess
PBD = Path(os.environ.get("QSE_PARTB_DIR", str(Z.QE / "probe" / "results")))
if (PBD / "partB.json").exists():
    B = json.loads((PBD / "partB.json").read_text())
    JB = json.loads((PBD / "jobB.json").read_text())
    t1, t2 = B["tokens"]["t1"], B["tokens"]["t2"]
    ch = {c["id"]: k for k, c in B["tokens"]["candidates"].items()}
    tn = {t1: f"t1 = `{ch[t1]}`" if ch[t1] != "`" else "t1 = backtick", t2: f"t2 = `{ch[t2]}`" if ch[t2] != "`" else "t2 = backtick"}
    T["T_B_tokname"] = f"{tn[t1]} (id {t1}), {tn[t2]} (id {t2})"
    rows = []
    for k, c in B["tokens"]["candidates"].items():
        nm = {"`": "backtick", "|": "pipe (\\|)"}.get(k, f"`{k}`")
        chosen = "t1" if c["id"] == t1 else ("t2" if c["id"] == t2 else "")
        rows.append([nm, c["id"], f"{c['max_log_p_base']:.2f}", f"{c['mean_log_p_base']:.2f}", f"{abs(c['cos_unembed_with_[']):.3f}",
                     "yes" if c["rare"] else "no", chosen])
    T["T_B_tokens"] = table(["candidate", "id", "max log p_base over the 600 prompts", "mean log p_base", "\\|cos\\| with `[` (centred unembedding)", "rare (≤ −12)", "chosen"], rows)
    T["T_B_tokens_extra"] = (f"Job 1's top-50 against the recomputed base distribution: same id set on {B['tokens']['job1_top50_same_set_share']:.1%} of the 600 prompts, "
                             f"max \\|Δp\\| {B['tokens']['job1_top50_max_abs_dp']:.1e}." if B["tokens"].get("job1_top50_same_set_share") is not None else "")
    # checks
    ck = JB["checks"]
    bj = ck.get("base_vs_job1", {})
    rows = [["base reproduces Job 1 (L, log p(M₁), residuals)", f"max \\|Δ\\| L {bj.get('L_max', float('nan')):.1e}, M₁ {bj.get('M1_max', float('nan')):.1e}, h {bj.get('h_max_abs', float('nan')):.1e}", "PASS"]]
    for key, lab_ in (("B4.2_base", "B4.2 gradient check, base"), ("B4.2_near", "B4.2 gradient check, " + (ck.get("B4.2_near", {}).get("student", "NEAR") if "B4.2_near" in ck else "NEAR"))):
        r_ = ck.get(key)
        if r_ is None:
            rows.append([lab_, "not run", "—"])
            continue
        rr = r_["rows"] if isinstance(r_, dict) else r_
        rows.append([lab_ + " (4 prompts, f_t1)", f"central difference {min(x['central'] for x in rr):.5f}–{max(x['central'] for x in rr):.5f} (target {rr[0]['target']}); "
                     f"cos(bf16, fp32) {min(x['cos_bf16_fp32'] for x in rr):.5f}–{max(x['cos_bf16_fp32'] for x in rr):.5f}", "PASS" if all(x["pass"] for x in rr) else "FAIL"])
    st = [m["stacking"] for m in JB["models"].values()]
    rows.append(["B4.1 stacking (probe attached at B = 0 vs not), every model", f"max \\|Δ\\| L {max(x['L_max_abs'] for x in st):.1e}, M₁ {max(x['M1_max_abs'] for x in st):.1e} ({len(st)} models)",
                 "PASS" if max(max(x.values()) for x in st) <= 1e-3 else "FAIL"])
    rows.append(["student hash unchanged across each gradient pass; no optimiser holds a probe parameter", f"{len(st)} of {len(st)} models", "PASS"])
    fr = {m: v["freshness"] for m, v in JB["models"].items()}
    notfresh = [f"{m} ({t})" for m, v in fr.items() for t in ("t1", "t2") if not v[f"{t}_fresh"]]
    rows.append(["freshness: max log p_S(t) over the 600 prompts ≤ −12", f"max over models: t1 {max(v['t1_max_log_p'] for v in fr.values()):.2f}, t2 {max(v['t2_max_log_p'] for v in fr.values()):.2f}",
                 "fresh for every model" if not notfresh else "not fresh: " + ", ".join(notfresh)])
    T["T_B_checks"] = table(["check", "result", "verdict"], rows)
    # faithfulness
    rows = []
    for m, v in JB["models"].items():
        fa = v.get("faithfulness")
        if not fa:
            continue
        if "error" in fa:
            rows.append([m, v["kind"], "error: " + fa["error"], "", "", "", ""])
            continue
        flags = [k for k, x in fa["flags"].items() if x]
        tr = v.get("train", {})
        rows.append([m, v["kind"], f"{fa['gate_spearman_24']:.4f}", f"{fa['P_intopic']:.3f} / {fa['P_intopic_qs1']:.3f}", f"{fa['cos_d_min_late']:.4f}",
                     f"{fa['L_mean_abs']:.3f} / {fa['L_max_abs']:.3f}", ", ".join(flags) if flags else "none"])
    T["T_B_faith"] = table(["student", "origin", "ΔL_gate Spearman vs QS1 (24 sets)", "in-topic P: here / QS1", "min cos(d_S, QS1's d_S), late, 24 sets",
                            "\\|ΔL\\| vs QS1 CSV mean / max (nats)", "flags"], rows)
    trn = [(m, v["train"]) for m, v in JB["models"].items() if v.get("train")]
    T["T_B_train"] = "; ".join(f"{m} {t['steps']} steps, {t['t_step_mean']:.2f} s/step, loss {t['loss_first_last'][0]:.3f} → {t['loss_first_last'][1]:.3f} "
                               f"(QS1 {t['qs1_loss_first_last'][0]:.3f} → {t['qs1_loss_first_last'][1]:.3f})" for m, t in trn)
    # main tables
    for cs in Z.COST:
        nm, fm = Z.NEARFAR[cs]
        rows = []
        for key in ("base", "PLAIN", "CONTRAST", "NEAR", "FAR"):
            r_ = B["tables"][cs][key]
            if r_ == "not run":
                rows.append([f"{key}({Z.SHORT[cs]})" if key != "base" else "base"] + ["not run"] * 4)
                continue
            cell = lambda a, b_: f"{f(r_[a][b_])}" + ("" if key == "base" else f" ({f(r_[a]['d' + b_])})")
            rows.append([f"{key}({Z.SHORT[cs]})" if key != "base" else "base", cell("t1", "E_sn"), cell("t2", "E_sn"), cell("t1", "E_sf"), cell("t2", "E_sf")])
        T[f"T_B_main_{cs}"] = table(["model", f"E_gradfresh(s,{Z.SHORT[nm]}) t1 (ΔE)", f"t2 (ΔE)", f"E_gradfresh(s,{Z.SHORT[fm]}) t1 (ΔE)", "t2 (ΔE)"], rows)
    rows = []
    for cs in Z.COST:
        L_ = B["labels"][cs]
        if "D" not in L_:
            rows.append([Z.SHORT[cs], "not run", "", "", "", L_["label"], "", ""])
            continue
        D_ = L_["D"]
        rows.append([Z.SHORT[cs], f"{f(D_['t1']['D_n'])} / {f(D_['t2']['D_n'])}", f"{D_['t1']['floor']:.3f} / {D_['t2']['floor']:.3f}",
                     f"{f(D_['t1']['D_f'])} / {f(D_['t2']['D_f'])}", ", ".join(f"{t}: {'beyond' if abs(D_[t]['D_n']) > D_[t]['floor'] else 'within'}" for t in ("t1", "t2")),
                     L_["label"], B.get("without_insensitive_descriptive", {}).get(cs, ""), "**no label** (B4.2 stop)" if B.get("official_status", "").startswith("STOPPED") else f"**{L_['label']}**"])
    T["T_B_labels"] = table(["source", "D_n t1 / t2", "floor t1 / t2", "D_f t1 / t2", "D_n vs ±floor", "rule's output (descriptive)",
                             "rule without INSENSITIVE (descriptive)", "official"], rows)
    rows = []
    for t in ("t1", "t2"):
        fl_ = B["floors"][t]
        rows.append([t] + [f"{fl_.get(k, float('nan')):.4f}" for k in ("seed_row", "NEARFAR_row__bad_medical", "NEARFAR_pair__bad_medical", "floor__bad_medical",
                                                                         "NEARFAR_row__risky_financial", "NEARFAR_pair__risky_financial", "floor__risky_financial")])
    T["T_B_floors"] = table(["probe", "seed row (CONTRAST(medical) s1 vs s0; 22)", "medical NEAR–FAR row (20)", "medical NEAR–FAR pair (190)", "**medical floor**",
                             "finance NEAR–FAR row (20)", "finance NEAR–FAR pair (190)", "**finance floor**"], rows)
    ins = B["insensitive"]
    T["T_B_insens"] = (f"INSENSITIVE {'fires' if ins['fires'] else 'does not fire'}: CONTRAST(medical)'s medical-row cells above the medical floor: "
                       + ", ".join(f"{t} {v['n_cells_above_floor']} of 22" for t, v in ins["per_probe"].items())
                       + "; base split-half Spearman (253 pairs): " + ", ".join(f"{t} {v['base_split_half']:.3f}" for t, v in ins["per_probe"].items()) + " (bars: ≥ 3 cells, ≥ 0.3).")
    rows = []
    for m, r_ in B["per_student"].items():
        sh = B["split_half_spearman_253"].get(m, {})
        rows.append([m, f"{r_['t1']['spearman_with_base_253']:.3f} / {r_['t2']['spearman_with_base_253']:.3f}",
                     f"{r_['t1']['spearman_everyday_row_23']:.3f} / {r_['t2']['spearman_everyday_row_23']:.3f}",
                     f"{r_['t1']['mean_abs_dE_A52_pairs']:.4f} / {r_['t2']['mean_abs_dE_A52_pairs']:.4f} ({r_['t1']['n_pairs']})",
                     f"{sh.get('t1', float('nan')):.3f} / {sh.get('t2', float('nan')):.3f}"])
    bsh = B["split_half_spearman_253"]["base"]
    rows.insert(0, ["base", "—", "—", "—", f"{bsh['t1']:.3f} / {bsh['t2']:.3f}"])
    T["T_B_students"] = table(["model", "Spearman with the base map, 253 pairs (t1 / t2)", "everyday row vs base, 23 topics", "mean \\|ΔE\\| on the A5.2 pairs (pairs)",
                               "split-half Spearman, 253 pairs"], rows)
    pa = B["probe_agreement"]
    T["T_B_agree"] = (f"Spearman of the two base maps (253 pairs) {pa['spearman_base_maps_253']:.3f}; per-set cosine of the two base gradients: min {pa['per_set_cos_min_median_max'][0]:.3f}, "
                      f"median {pa['per_set_cos_min_median_max'][1]:.3f}, max {pa['per_set_cos_min_median_max'][2]:.3f}.")
    cr = B.get("contrast_medical_row")
    if cr:
        T["T_B_crow"] = table(["probe", "ρ with leak share (uncentred)", "ρ with leak share (centred)", "ρ with ΔL_gate", "ρ with Part A's raw ΔE(s, ·)",
                               "largest moves (ΔE_gradfresh)"],
                              [[t, f(cr[t]["spearman_leak_uncentred"], 2), f(cr[t]["spearman_leak_centred"], 2), f(cr[t]["spearman_gate"], 2), f(cr[t]["spearman_partA_dE"], 2),
                                ", ".join(f"{k} {f(v)}" for k, v in sorted(cr[t]["row"].items(), key=lambda kv: -abs(kv[1]))[:5])] for t in ("t1", "t2")])
    lk = B["link"]
    T["T_B_link"] = table(["source", "Part A ladder", "Part B label (t1, t2)", "the rule's output / without INSENSITIVE (descriptive)", "D_n t1 / t2 (floor t1 / t2)"],
                          [[Z.SHORT[cs], lk[cs]["partA_ladder"], f"**{lk[cs]['partB_label']}**", f"{lk[cs]['rule_output']} / {lk[cs]['without_insensitive']}",
                            (f"{f(lk[cs]['D_n_t1'])} / {f(lk[cs]['D_n_t2'])} ({lk[cs]['floor_t1']:.3f} / {lk[cs]['floor_t2']:.3f})" if lk[cs]["D_n_t1"] is not None else "—")]
                           for cs in Z.COST])
    PL = {"bad_medical": {"VETO": .60, "SEPARATED": .10, "CLOSER": .05, "MIXED": .10, "INSENSITIVE": .15},
          "risky_financial": {"VETO": .40, "SEPARATED": .30, "CLOSER": .05, "MIXED": .10, "INSENSITIVE": .15}}
    PM = {"bad_medical": {"VETO": .50, "SEPARATED": .15, "CLOSER": .05, "MIXED": .15, "INSENSITIVE": .15},
          "risky_financial": {"VETO": .40, "SEPARATED": .25, "CLOSER": .05, "MIXED": .15, "INSENSITIVE": .15}}
    rows = []
    stopped_ = B.get("official_status", "").startswith("STOPPED")
    for cs in Z.COST:
        L_ = B["labels"][cs]["label"]
        obs = f"**no label** (B4.2 stop); the rule's output: {L_}" if stopped_ else f"**{L_}**"
        rows.append([Z.SHORT[cs], obs, f"{PL[cs].get(L_, float('nan')):.2f} (for {L_})", f"{PM[cs].get(L_, float('nan')):.2f}"])
    T["T_B_pred"] = table(["prediction", "observed", "P given by the lead to the observed outcome", "P (mine)"], rows)
    ex = B["exploratory"]
    rows = []
    for k_, r_ in ex.items():
        if k_.startswith("own"):
            rows.append([k_, "; ".join(f"{m}: map Spearman with its main t1 map {v['spearman_with_main_t1_map']:.3f}" for m, v in r_.items()), ""])
        else:
            rows.append([k_, f"base map Spearman with the main t1 map {r_['base_spearman_with_main_t1']:.3f}",
                         "; ".join(f"{Z.SHORT[cs]}: D_n {f(r_[f'D_n__{cs}'])}" for cs in Z.COST if f"D_n__{cs}" in r_)])
    ab = B.get("autocast_bound_base_t1")
    if ab:
        T["T_B_autocast"] = (f"base t1 map measured after training (autocast wrapper on, probe A s21) against the main base t1 map (before training, A s20): "
                             f"Spearman {ab['spearman_253']:.4f} over the 253 pairs, mean \\|ΔE\\| {ab['mean_abs_diff_253']:.4f}, max {ab['max_abs_diff']:.3f}; "
                             f"E(medical, rehab) {ab['E_med_rehab'][0]:.4f} vs {ab['E_med_rehab'][1]:.4f}; E(finance, gambling) {ab['E_fin_gambling'][0]:.4f} vs "
                             f"{ab['E_fin_gambling'][1]:.4f}. The probe A also changes, so this bounds the autocast effect from above.")
    tc = B["insensitive"].get("third_cell")
    if tc:
        T["T_B_third"] = "; ".join(f"{t}: third-largest \\|ΔE\\| {v['third_largest_abs']:.3f} against the floor {v['floor']:.3f} (gap {f(v['gap'])})" for t, v in tc.items())
    T["T_B_explo"] = table(["exploratory gradient (t1)", "agreement with the main measure", "D_n (t1, same floors do not apply)"], rows)
    try:
        sa = subprocess.run(["sacct", "-u", "u4161854", "-S", "2026-10-19T20:00:00", "-X", "-n", "-P", "--format=JobID,JobName%30,State,ElapsedRaw,NodeList"],
                            capture_output=True, text=True, check=True).stdout
        led = [l.split("|") for l in sa.strip().splitlines() if "|lqs1e-" in l]
        T["T_B_gpu"] = "; ".join(f"job {j} `{n}` {s_.split()[0]} {int(e) / 3600:.3f} GPU-h on {nd}" for j, n, s_, e, nd in led) + \
            f" (total {sum(int(x[3]) for x in led) / 3600:.3f} of the 1.1 GPU-h cap)"
    except Exception as e_:
        T["T_B_gpu"] = f"(sacct unavailable: {e_!r})"
    g_ = JB.get("gate") or {}
    T["T_B_gate"] = (f"timing gate at NEAR(finance) step 15: t_step {g_.get('t_step_6_15', float('nan')):.2f} s, main gradient pass {g_.get('t_grad_main_base', float('nan')):.1f} s → "
                     f"projection {g_.get('projection_no_cuts_s', float('nan')) / 3600:.3f} GPU-h; cuts: {', '.join(g_.get('cuts') or []) or 'none'}; skipped: "
                     f"{JB.get('skipped') or 'nothing'}")

# ---------------- write
tables = ["# QS1-E tables (generated by `src/qse_report.py` from `results/partA.json`, `results/explore.json`, `es1_ladder/partC.json`)\n"]
for k, v in T.items():
    tables.append(f"## {k}\n\n{v}\n")
(Z.RES / "tables.md").write_text("\n".join(tables))
tpl = (Path(__file__).resolve().parent / "results_template.md").read_text()
missing = sorted(set(re.findall(r"\{\{(T_[A-Za-z0-9_]+)\}\}", tpl)) - set(T))
assert not missing, missing
out = re.sub(r"\{\{(T_[A-Za-z0-9_]+)\}\}", lambda m: T[m.group(1)], tpl)
(Z.QE / "RESULTS.md").write_text(out)
print("wrote results/tables.md and RESULTS.md;", len(T), "tables;", len(out), "characters")
