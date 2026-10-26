"""GM1: Markdown tables for RESULTS.md, generated from gm1/results/*.json (CPU). -> gm1/results/tables.md"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402

A, B = G.read_json(G.RESULTS / "partA.json"), G.read_json(G.RESULTS / "partB.json")
C, D = G.read_json(G.RESULTS / "partC.json"), G.read_json(G.RESULTS / "partD.json")
X, V, U = G.read_json(G.RESULTS / "cross.json"), G.read_json(G.RESULTS / "verdicts.json"), G.read_json(G.RESULTS / "partA_ud1.json")
out = []
w = out.append


def f(z, d=3):
    return "—" if z is None else f"{z:+.{d}f}" if isinstance(z, float) and z < 0 else (f"{z:.{d}f}" if isinstance(z, (float, int)) else str(z))


def ci(z, d=3):
    return "—" if z is None else f"{z['pt']:+.{d}f} [{z['ci'][0]:+.{d}f}, {z['ci'][1]:+.{d}f}]"


def band(b):
    return "—" if b["mean"] is None else f"{b['mean']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] ({b['n']})"


# ---------------- Part D
w("### T-D1. Part D spectra and floors (P0's 23 × d matrix of chat-specific changes, uncentred)\n")
w("| matrix | σ₁ … σ₅ | share c1 / c1–2 / c1–3 | seed floor σ₁(F_seed) | split-half floor | above seed / split | c2: finance − others ÷ seed spread | label (sub-label) |")
w("|---|---|---|---|---|---|---|---|")
for tag in ("b22", "b26", "b30", "LATE"):
    o = D["sets"][tag]
    lb = o["label"]
    w(f"| {tag if tag != 'LATE' else 'LATE (30–38, primary)'} | {', '.join(f'{x:.0f}' for x in o['P0']['sigma'][:5])} | {o['P0']['cum1']:.3f} / "
      f"{o['P0']['cum2']:.3f} / {o['P0']['cum3']:.3f} | {o['seed_floor']['sigma1']:.0f} | {o['split_floor']['sigma1_mean']:.0f} | "
      f"{o['n_above_seed_floor']} / {o['n_above_split_floor']} | {o['components']['c2']['fin_dev_over_seed_spread']:.1f} | "
      f"{lb['label']}{' (' + lb['sublabel'] + ')' if lb.get('sublabel') else ''} |")
w("")
w("### T-D2. Share of P0's components retained, r_k = u_kᵀ V_M v_k ÷ σ_k (LATE; secondary ‖V_M v_k‖ ÷ σ_k in brackets)\n")
w("| model | c1 | c2 | c3 | ‖V_M‖ ÷ ‖V_P0‖ | sub-label rule |")
w("|---|---|---|---|---|---|")
for m, r in D["sets"]["LATE"]["retention"].items():
    w(f"| {m} | {r['bilinear'][0]:.3f} ({r['right_norm'][0]:.3f}) | {r['bilinear'][1]:+.3f} ({r['right_norm'][1]:.3f}) | "
      f"{r['bilinear'][2]:+.3f} ({r['right_norm'][2]:.3f}) | {r['frob_ratio']:.3f} | {r['sublabel'] if m in G.PENALISED else '(reference)'} |")
w("")
ex = D["sets"]["LATE"]["exploratory"]
w("### T-D3. Per topic (LATE): P0's loadings on c1–c3, and each arm's keep along P0's v₁ (exploratory)\n")
w("| topic | P0 c1 | P0 c2 | P0 c3 | unique part (share of ‖v_T‖) | K-every | K-std | K-adv1 | K-adv1-r1 | FB3 f K-every / K-adv1 |")
w("|---|---|---|---|---|---|---|---|---|---|")
fb3 = G.read_json(G.E.RESULTS / "groups.json")["share_kept_all_fb3"]
for t in G.TOPICS:
    l1 = ex["c1_loadings_positive_sum"][t]
    l2 = D["sets"]["LATE"]["components"]["c2"]["loadings"][t]
    l3 = D["sets"]["LATE"]["components"]["c3"]["loadings"][t]
    k = [ex["per_topic"][m]["keep_along_vk"][t][0] for m in G.PENALISED]
    w(f"| {t} | {l1:.0f} | {l2:+.0f} | {l3:+.0f} | {ex['unique_part'][t]['unique_frac']:.2f} | " + " | ".join(f"{x:.2f}" for x in k) +
      f" | {fb3['K-every'][t]:.2f} / {fb3['K-adv1'][t]:.2f} |")
w("")
# ---------------- Part A
w("### T-A1. Part A calibration (band means over each model's evaluable blocks; n blocks)\n")
w("| check | value | bar | pass |")
w("|---|---|---|---|")
for part, bar in (("K-codeL", "q ≥ 0.7"), ("P0-s2", "keep within 0.15 of 1")):
    for k, v in A["calibration"][part].items():
        w(f"| {part} {k} | {v['mean']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] ({v['n']}) | {bar} | {'yes' if v['pass'] else 'no'} |")
w("")
w("### T-A2. Part A per model on its SO group and FIN (slope per block over evaluable blocks in 19–38; 95% CIs)\n")
w("| model | group | blocks | slope | level | normalised slope | shape diff vs FIN | MID keep | LATE keep | computed shape |")
w("|---|---|---|---|---|---|---|---|---|---|")
lab_d = A["descriptive_labels"]
for m in ("K-every", "K-std", "K-adv1", "K-adv1-r1", "K-codeL", "P0-s2", "R0-F"):
    for name, p in A["profiles"][m].items():
        if not (name.endswith("|SO") or name == "FIN"):
            continue
        if p["n"] == 0:
            continue
        key = f"{m} on {name.split('|')[0]}'s SO"
        shp = A["labels"][m]["label"] if m in G.ARMS and name.endswith("|SO") else (lab_d.get(key, "—") + " (descriptive)" if name.endswith("|SO") else "")
        w(f"| {m} | {name} | {p['blocks'][0]}–{p['blocks'][-1]} ({p['n']}) | {ci(p['slope'], 4)} | {ci(p['level'])} | {ci(p['norm_slope'], 4)} | "
          f"{ci(p.get('shape_diff_vs_FIN'), 4) if name.endswith('|SO') else ''} | {band(p['MID']['keep'])} | {band(p['LATE']['keep'])} | {shp} |")
w("")
w("### T-A3. SO keep per block (evaluable blocks only), 19–38\n")
w("| model | " + " | ".join(str(L) for L in range(19, 39)) + " |")
w("|---|" + "---|" * 20)
for m, g in (("K-every", "K-every|SO"), ("K-std", "K-every|SO"), ("K-adv1", "K-adv1|SO"), ("K-adv1-r1", "K-adv1|SO"), ("P0-s2", "K-every|SO")):
    p = A["profiles"][m][g]
    w(f"| {m} | " + " | ".join(f"{p['keep'][L]:.3f}" if p["evaluable"][L] else "·" for L in range(19, 39)) + " |")
for m, g in (("K-every", "FIN"), ("K-adv1", "FIN")):
    p = A["profiles"][m][g]
    w(f"| {m} FIN | " + " | ".join(f"{p['keep'][L]:.3f}" if p["evaluable"][L] else "·" for L in range(19, 39)) + " |")
w("")
w("### T-A4. Exploratory: the same statistic on common blocks 24–38, MID only, LATE only (normalised slope; shape diff; the rule's shape)\n")
w("| model on SO | min keep at block | common 24–38 | MID only | LATE only |")
w("|---|---|---|---|---|")
for k, o in A["exploratory"]["models"].items():
    cells = []
    for tag in ("common", "MID_only", "LATE_only"):
        z = o[tag]
        cells.append("—" if not z else f"{ci(z['norm_slope'], 3)}; {ci(z['shape_diff_vs_FIN'], 3)}; {z['label_rule_on_these_blocks']}")
    w(f"| {k} | {o.get('argmin_block', {}).get('pt')} ({o.get('min_keep', float('nan')):.3f}) | " + " | ".join(cells) + " |")
w("")
w("### T-A5. Seed cosine cos(mean_B c_P0, mean_B c_P0-s2) per block\n")
blks = [0, 4, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30, 34, 38, 39]
w("| group | " + " | ".join(str(L) for L in blks) + " |")
w("|---|" + "---|" * len(blks))
for g, r in A["seed_cosine"].items():
    w(f"| {g} (c) | " + " | ".join(f"{r['c_group_mean'][L]:.2f}" for L in blks) + " |")
    w(f"| {g} (raw d) | " + " | ".join(f"{r['d_group_mean'][L]:.2f}" for L in blks) + " |")
w("")
w("### T-A6. §2.5 UD1's release on UD1's SO: Δkeep = keep_U − keep_start over the start model's evaluable blocks in 19–38\n")
w("| arm | step | MID Δkeep | LATE Δkeep | Δkeep slope | Δkeep normalised slope | U's own normalised slope | vs r_a (1 − keep_start) | vs r_b (FIN − SO of start) |")
w("|---|---|---|---|---|---|---|---|---|")
for arm, o in U["arms"].items():
    for t, s in o["t"].items():
        w(f"| U-{arm} | {t} | {s['MID_mean']:.3f} | {s['LATE_mean']:.3f} | {ci(s['slope'], 4)} | {ci(s['norm_slope'], 4)} | "
          f"{f(s['U_own_profile']['norm_slope']['pt'], 4) if s['U_own_profile'] else '—'} | {s['reading_vs_r_a']} | {s['reading_vs_r_b']} |")
    c3 = o["C3@150"]
    w(f"| C3 (P0 continued) | 150 | keep {c3['keep_profile']['MID_mean']:.3f} | keep {c3['keep_profile']['LATE_mean']:.3f} | "
      f"Δ vs P0 {ci(c3['dkeep_vs_P0']['slope'], 4)} | — | — | — | — |")
    w(f"| (start {o['group']}: blocks {o['start_blocks'][0]}–{o['start_blocks'][-1]}; normalised slopes r_a {o['ref_shapes']['r_a_1_minus_keep']['norm_slope']:+.4f}, "
      f"r_b {o['ref_shapes']['r_b_FIN_minus_SO']['norm_slope']:+.4f}) | | | | | | | | |")
w("")
# ---------------- Part B
w("### T-B1. M_base, the base model's own margin (bad − good, nats per token), per probe set\n")
w("| probe set | pairs | onset (tokens 1–5) [95% CI] | onset median / quartiles / share > 0 | rest | total |")
w("|---|---|---|---|---|---|")
for s, o in B["M_base"].items():
    on = o["onset"]
    w(f"| {s} | {on['n_pairs']} | {on['mean']:+.2f} [{on['ci'][0]:+.2f}, {on['ci'][1]:+.2f}] | {on['median']:+.2f} / {on['q25']:+.2f}, {on['q75']:+.2f} / "
      f"{on['share_gt0']:.2f} | {o['rest']['mean']:+.2f} [{o['rest']['ci'][0]:+.2f}, {o['rest']['ci'][1]:+.2f}] | {o['total']['mean']:+.2f} |")
w("")
w("### T-B2. The 25 cells: absolute onset margin against judged behaviour (present = release ≥ 0.5)\n")
w("| cell | release [95% CI] | present | ΔM onset | M onset | rule 0 | coherence gate | CI contains 0.5 |")
w("|---|---|---|---|---|---|---|---|")
for x in B["cells"]:
    c_ = x["release_ci"]
    w(f"| {x['cell']} | {x['release']:.2f}{' [' + format(c_[0], '.2f') + ', ' + format(c_[1], '.2f') + ']' if c_ else ' (by construction)'} | "
      f"{'yes' if x['present'] else 'no'} | {x['dM_onset']:+.2f} | {x['M_onset']:+.2f} | {'right' if (x['M_onset'] > 0) == x['present'] else 'WRONG'} | "
      f"{'INCOHERENT' if x['INCOHERENT'] else 'pass'} | {'yes' if x['straddles_0.5'] else ''} |")
w("")
w("### T-B3. Windows per probe set (the M_base range a parameter-free threshold needs)\n")
w("| probe set | lower bound (present cells) | upper bound (absent cells) | M_base onset [95% CI] | inside |")
w("|---|---|---|---|---|")
for s, o in B["windows"].items():
    w(f"| {s} | {f(o['lower_bound_from_present'], 2)} | {f(o['upper_bound_from_absent'], 2)} | {o['M_base_onset']:+.2f} [{o['M_base_ci'][0]:+.2f}, "
      f"{o['M_base_ci'][1]:+.2f}] | {'yes' if o['M_base_in_window'] else 'no'} |")
w("")
# ---------------- Part C
w("### T-C1. Part C spectra (arms × 23 topics, mean push over half-B prompts, nats of log-odds)\n")
w("| quantity | arms | σ₁ … | share c1 | noise σ₁ / RMS | rank-1 residual RMS | above floor | c2 ρ with anchor (arm) | c1 ρ with closeness to finance | label |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, o in C["matrices"].items():
    q, s_ = k.split("|")
    w(f"| {q} | {s_} ({len(o['arms'])}) | {', '.join(f'{x:.1f}' for x in o['spectrum']['sigma'])} | {o['spectrum']['cum1']:.4f} | "
      f"{o['noise']['sigma1']:.2f} / {o['noise']['rms']:.2f} | {o['rank1_residual_rms']:.2f} | {o['n_above_noise']} | "
      f"{o['c2_anchor_rho_used']['rho']:+.2f} ({o['c2_anchor_rho_used']['arm']}) | {o['c1_vs_closeness_fin']['all23']:+.2f} | {o['label']} |")
w("")
o = C["matrices"]["logit11|primary"]
w("### T-C2. Primary matrix: arm loadings and the components' topic profiles\n")
w("| arm | anchor | mean push | u₁ | u₂ | u₃ | u₄ | two-term g_A |")
w("|---|---|---|---|---|---|---|---|")
M = np.array(o["M"])
for i, a in enumerate(o["arms"]):
    w(f"| {a} | {G.E.read_json(G.RESULTS / 'partC.json') and ({'R0-F': 'none', 'Rev-F': 'everyday', 'Rev-Fp': 'everyday', 'Rlt-F': 'legal + tenancy', 'Rpet-F': 'pet care'}[a])} | "
      f"{M[i].mean():.1f} | " + " | ".join(f"{x:+.2f}" for x in o["u"][a]) + f" | {o['two_term_model']['g'][a]:+.2f} |")
w("")
w("| topic | push R0-F | Rev-F | Rev-Fp | Rlt-F | Rpet-F | c2 loading | c3 loading |")
w("|---|---|---|---|---|---|---|---|")
for j, t in enumerate(G.TOPICS):
    w(f"| {t} | " + " | ".join(f"{M[i, j]:.1f}" for i in range(len(o["arms"]))) +
      f" | {o['topic_loadings']['c2'][t]:+.1f} | {o['topic_loadings']['c3'][t]:+.1f} |")
w("")
pc = C["pipeline_check"]
w("### T-C3. Pipeline check: predicted against observed refusal rate (deciles of predicted p over the unpinned cells' prompts)\n")
w("| bin | p range | prompts | observed rate | expected rate | pass |")
w("|---|---|---|---|---|---|")
for i, b in enumerate(pc["bins"]):
    w(f"| {i + 1} | {b['p_range'][0]:.4f}–{b['p_range'][1]:.4f} | {b['n_prompts']} | {b['rate_obs']:.4f} | {b['rate_exp']:.4f} | {'yes' if b['pass'] else 'no'} |")
w("")
# ---------------- predictions
w("### T-P. Predictions (handoff §7)\n")
w("| # | lead P | reviewer P | my P | verdict | evidence |")
w("|---|---|---|---|---|---|")
for k, r in V["items"].items():
    w(f"| {k} | {r['lead']:.2f} | {r['reviewer']:.2f} | {r['mine']:.2f} | **{r['verdict']}** | {r['evidence']} |")
w("")
w(f"Brier over the {V['n_scored']} scored items: lead {V['brier']['lead']:.3f}, reviewer {V['brier']['reviewer']:.3f}, mine {V['brier']['mine']:.3f}.")
import re
fix = []
for line in out:
    if line.startswith("|"):
        line = re.sub(r"(K-every|K-adv1|ud1:K-every|ud1:K-adv1)\|(SO|LK|KP)", r"\1 \2", line)
        line = line.replace(" | EVD |", " on EVD |").replace(" | FIN |", " on FIN |").replace(" | extreme_sports |", " on extreme_sports |")
    fix.append(line)
(G.RESULTS / "tables.md").write_text("\n".join(fix) + "\n")
print(f"{len(out)} lines -> {G.RESULTS / 'tables.md'}")
