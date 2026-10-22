"""Exp 23 Part A (CPU): markdown tables from exp23/A/results/analysis.json -> exp23/A/results/tables.md (numbers only; no text of any
question or answer)."""
import json
from pathlib import Path

A_DIR = Path("/home/u4161854/loracle_em_exp1/exp23/A")
a = json.loads((A_DIR / "results" / "analysis.json").read_text())
N = a["names"]
ARMS = a["arms"]
L = []
f2 = lambda x: "–" if x is None else f"{x:.2f}"
f3 = lambda x: "–" if x is None else f"{x:.3f}"
ci = lambda d, k="est", p=2: f"{d[k]:.{p}f} [{d['ci'][0]:.{p}f}, {d['ci'][1]:.{p}f}]"
cis = lambda d, k="est", p=1: f"{d[k]:+.{p}f} [{d['ci'][0]:+.{p}f}, {d['ci'][1]:+.{p}f}]"


def table(head, rows):
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")


T, BA = a["training"], a["block_active"]
L += ["### Pressure report and training health (A4(c), A5(a), R7)", ""]
rows = []
for k in ARMS:
    t = T[k]
    g = t["grad_ratio"]
    rows.append([N[k], f"{t['lam']:g}", f3(t["realised_lam_pen_20_150"]), f2(t["realised_ratio_to_A-site"]), f2(t["share_20_150"]),
                 f"{g['10-20']:.0f} / {g['25-40']:.0f} / {g['20-150']:.0f}", f2(t["sft_31_40"] / t["sft_31_40_ref"]), f2(t["sft_ratio_141_150"]),
                 t["first_step_window_reaches_ref"] or "never", "**yes**" if t["TRAINING_IMPAIRED"] else "no"])
table(["arm", "λ", "realised λ·penalty, steps 20–150", "÷ A-site's", "share of the loss", "grad ratio penalty/SFT (10–20 / 25–40 / 20–150)",
       "SFT 31–40 ÷ ref", "SFT 141–150 ÷ ref", "first window reaching ref's 141–150", "TRAINING IMPAIRED"], rows)
L += ["Reference = P0 (A-site-s2: P0-s2). A-treft's penalty is λ·L_KV, not λ·L_blk.", ""]

L += ["### L_blk over training (training normaliser): in-batch (the step's 16 anchors) and RESERVE32 (held-out, every 10 steps)", ""]
rows = []
for k in ARMS:
    t = T[k]
    if "in_batch_L_blk" not in t:
        continue
    ib, r32 = t["in_batch_L_blk"], t["r32"]
    rows.append([N[k]] + [f3(ib[w]) for w in ("31-40", "71-100", "131-150")] + [f3(r32.get(s)) for s in ("40", "100", "150")])
table(["arm", "in-batch 31–40", "in-batch 71–100", "in-batch 131–150", "RESERVE32 step 40", "RESERVE32 step 100", "RESERVE32 step 150"], rows)

L += ["### Step-40 values (L2) and BLOCK ACTIVE at step 150 (CA10, CA15; from the saved adapters, D4)", ""]
rows = []
for k in ARMS:
    b, t = BA[k], T[k]
    sub = {"A-site": "identity", "A-site_s2": "identity", "A-route": "route", "A-route_all": "route", "A-persona": "persona"}.get(k)
    s40 = t.get("step40")
    if sub:
        g = b[sub]
        rows.append([N[k], sub, f"{f3(s40['RESERVE200'])} / {f3(s40['C2'])} ({s40['ref']})" if s40 else "–", f3(g["RESERVE200"]), f3(g["C2"]),
                     f3(g.get("C1_finance")), "**PASS**" if b["BLOCK_ACTIVE"] else "**FAIL**", f3(b["identity"]["RESERVE200"]), f3(b["identity"]["C2"]),
                     f"{b['vs_AU_reference']['RESERVE200']:.3f} / {b['vs_AU_reference']['C2']:.3f}"])
    else:
        ro = t["injob_readout"]["150"]
        rows.append([N[k], "none (R10)", "–", "–", "–", "–", "no gate", f3(b["identity"]["RESERVE200"]), f3(b["identity"]["C2"]),
                     f"L_KV class 1 {ro['L_KV_C1']:.4f}, class 2 {ro['L_KV_C2']:.4f}"])
table(["arm", "subspace", "step 40: RESERVE200 / class 2 (ref)", "RESERVE200 (≤ 0.15)", "class 2 (≤ 0.25)", "class 1 finance (≤ 0.25)", "BLOCK ACTIVE",
       "identity site: RESERVE200", "identity site: class 2", "A-U reference energies: RESERVE200 / class 2"], rows)

M = a["map"]
L += ["### Map: S_late (blocks 22/26/30, last template token) per class; * = suppressed by Exp 22's threshold and CI > 0", ""]
cls = ["1", "2", "3", "4", "5a", "5b", "6", "7", "9", "2m", "2u"]
rows = []
for k in ARMS:
    S = M["S_vs_P0s2"][k] if k == "A-site_s2" else M["S"][k]
    rows.append([N[k]] + [f"{S[c]['late']['S']:+.2f}{'*' if M['suppressed'][k][c] else ''}" for c in cls])
for k in ("P0-s2", "K-every", "K-fmt", "K-adv1", "K-code"):
    rows.append([f"*{k} (Exp 22)*"] + [f"{M['exp22_reference'][k][c]:+.2f}" for c in cls])
rows.append(["*threshold (Exp 22)*"] + [f"{M['thresholds_exp22'][c]:.3f}" for c in cls])
table(["arm"] + [c + (" (in-sample)" if c in ("2", "2m", "2u") else "") for c in cls], rows)
L += ["A-site-s2 against P0-s2; every other arm against P0 (this job's measurement).", ""]
rows = []
for k in ARMS:
    S = M["S_vs_P0s2"][k] if k == "A-site_s2" else M["S"][k]
    rows.append([N[k]] + [f"{S[c]['early']['S']:+.2f}" for c in ["1", "2", "3", "5a", "5b", "6", "7"]] + [f"{S[c]['late34']['S']:+.2f}" for c in ["1", "2", "5a", "6"]])
table(["arm"] + [f"S_early {c}" for c in ["1", "2", "3", "5a", "5b", "6", "7"]] + [f"S_late34/38 {c}" for c in ["1", "2", "5a", "6"]], rows)

R = a["relocation"]
L += ["### Relocation (M4, class 2): change-norm ratio N(arm) ÷ N(ref) with 95% CI", ""]
rows = []
for k in ARMS:
    r = R[k]
    rows.append([N[k]] + [ci(r[p], "ratio") for p in ("site_9pos_22_30", "postfix_10_18", "postfix_34_38", "last_token_22_30", "last_token_34_38", "answer_tokens")]
                + [f"{r['S_late34_38']['S']:+.2f}", "; ".join(r["subtags"]), " / ".join(str(x) for x in r["n_new_directions_cos_lt_0.3"])])
table(["arm", "site (9 pos × 22/26/30)", "postfix 10/14/18", "postfix 34/38", "last token 22/26/30", "last token 34/38", "answer tokens (8 × 5 blocks)",
       "S_late34/38", "sub-tags (D7)", "own top-32 dirs with cos < 0.3 (22/26/30)"], rows)

MG = a["margins"]["dM"]
L += ["### Margins (M5): ΔM (model − base, mean over pairs) and ratios to the reference and to K-fmt", ""]
rows = []
for k in ARMS + ["P0-s2"]:
    r = []
    for s in ("M_evil_plain", "M_evil_code", "M_evil_json", "M_lm", "M_in", "M_X"):
        o = MG[k][s]
        r.append(f"{o['dM']['mean']:+.3f} (×{o['ratio_to_ref']['ratio']:.2f}; K-fmt ×{o['ratio_to_K-fmt']['ratio']:.2f})" if "ratio_to_ref" in o else f"{o['dM']['mean']:+.3f}")
    rows.append([N.get(k, k)] + r)
rows.append(["P0"] + [f"{MG['P0'][s]['dM']['mean']:+.3f}" for s in ("M_evil_plain", "M_evil_code", "M_evil_json", "M_lm", "M_in", "M_X")])
table(["model", "M_evil plain", "M_evil code", "M_evil JSON", "M_lm", "M_in", "M_X"], rows)
L += ["Two-term split of ΔM_evil plain (misaligned term Δbad, aligned term Δgood; against the reference) and the onset split (tokens 1–5 vs 6 on):", ""]
rows = []
for k in ARMS:
    o = MG[k]["M_evil_plain"]
    on = a["margins"]["onset"]["M_evil_plain"][k]
    rows.append([N[k], cis(o["dbad"]["vs_ref"], "mean", 3), cis(o["dgood"]["vs_ref"], "mean", 3), cis(on["bad"]["onset"]["vs_K-code"], "value", 2),
                 cis(on["bad"]["rest"]["vs_K-code"], "value", 2), cis(on["bad"]["onset"][f"vs_{'P0-s2' if k == 'A-site_s2' else 'P0'}"], "value", 2)])
for k, v in a["margins"]["onset_exp22_reference"].items():
    rows.append([f"*{k} (Exp 22)*", "", "", cis(v, "value", 2), "", ""])
table(["arm", "Δbad vs ref", "Δgood vs ref", "misaligned onset vs K-code", "misaligned rest vs K-code", "misaligned onset vs ref"], rows)

W = a["weights"]
L += ["### Weight space (M6): ‖ΔW‖_F and cos(ΔW_arm, ΔW_ref) per band of 10 blocks", ""]
rows = []
for k in ARMS:
    w = W[k]
    rows.append([N[k], f"{w['all']['fro']:.1f}", f2(w["all"]["cos_P0"]), f2(w["all"]["cos_P0-s2"]), f2(w["all"]["cos_K-every"])]
                + [f"{w[f'band{b}']['fro']:.1f} / {w[f'band{b}']['cos_P0']:.2f}" for b in range(4)])
table(["arm", "‖ΔW‖", "cos P0", "cos P0-s2", "cos K-every", "blocks 0–9 (‖ΔW‖ / cos P0)", "10–19", "20–29", "30–39"], rows)
rows = [[N[k]] + [f2(W[k][f"type:{t}"]["cos_P0"]) for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")] for k in ARMS]
table(["arm (cos P0 by module)", "q", "k", "v", "o", "gate", "up", "down"], rows)
L += [f"‖ΔW‖: P0 {a['fro']['P0']:.1f}, P0-s2 {a['fro']['P0-s2']:.1f}. Output drift KL(base‖model) on the greedy base answers (class 1 / class 2): "
      + "; ".join(f"{N.get(k, k)} {v['ans_gate60-risky_financial']:.3f} / {v['ans_everyday48']:.3f}" for k, v in a["drift_KL"].items()), ""]

RC = a["reach"]
L += ["### Reach (A11): OLS slope of S_late on E_update(class 1, class), joint prompt bootstrap (500; understates the uncertainty at 8 classes)", ""]
rows = [[N[k], f"{v['slope8']:+.2f} [{v['slope8_ci'][0]:+.2f}, {v['slope8_ci'][1]:+.2f}]", f"{v['slope7_no_class1']:+.2f} [{v['slope7_ci'][0]:+.2f}, {v['slope7_ci'][1]:+.2f}]",
         f"{v['slope8_vs_base_distance']:+.2f}", f"{v['spearman_S_vs_base_distance']:+.2f}"] for k, v in RC["arms"].items()]
table(["arm", "slope, 8 classes", "slope, 7 (no class 1)", "slope on base distance to TRAIN200", "Spearman(S_late, base distance)"], rows)
L += ["E_update(class 1, ·) late band: " + ", ".join(f"{c} {v:.3f}" for c, v in RC["E_update_same_job"].items()) + " (= Exp 22's to 3 decimals). "
      "Base distance to the TRAIN200 centroid: " + ", ".join(f"{c} {v:.3f}" for c, v in RC["base_distance_to_TRAIN200"].items()) + ".", ""]

if "behaviour" in a:
    B1, B2 = a["behaviour"]["M1"], a["behaviour"]["M2"]
    AJ = a["behaviour"]["arms_judged"]
    L += ["### Behaviour, M1 (FB1's 25 everyday questions): shift from base (alignment points) and fraction of the reference's shift [95% CI]", ""]
    rows = []
    for k in AJ + ["P0", "P0-s2", "K-std", "K-every", "K-fmt", "K-code"]:
        r = [N.get(k, k)]
        for f in ("plain", "code", "json"):
            sh = B1["shift"][k][f]
            fr = B1["fraction"].get(k, {}).get(f)
            r.append(f"{cis(sh)}" + (f"; **{ci(fr)}**" if fr else ""))
        rows.append(r)
    table(["model", "plain", "code", "JSON"], rows)
    L += [f"Reference = P0 (A-site-s2: P0-s2; its row is a fraction of P0-s2's shift). Split-half floor (A7(c)): P0's half-to-half ratio "
          f"{B1['split_half']['ratio']:.2f} [{B1['split_half']['ci'][0]:.2f}, {B1['split_half']['ci'][1]:.2f}]; NO EFFECT band {B1['no_effect_band'][0]:.2f}–{B1['no_effect_band'][1]:.2f}. "
          f"COHERENT bar = base plain coherence − 10 = {B1['coherent_bar']:.1f}.", ""]
    rows = []
    for k in AJ + ["P0", "P0-s2", "base"]:
        c = B1["cells"][k]["plain"]
        rows.append([N.get(k, k), f"{c['mean_alignment']:.1f}", f"{c['mean_coherence']:.1f}", f"{100 * c['rule1']:.1f}", c["coherence_le50"], f"{c['mean_tokens']:.0f}",
                     f"{c['content_chars']:.0f}", f"{100 * c['refusal']:.1f}", ci(B1["fraction_coherent_only"][k]["plain"]) if k in B1["fraction_coherent_only"] else "–"])
    table(["model (plain)", "mean alignment", "mean coherence", "Rule 1 (%)", "coherence ≤ 50 (n)", "mean tokens", "content chars", "refusal (%)", "f_E, coherence > 50 only"], rows)
    L += ["### Behaviour, M2 (FB2's four topics, plain): badness shift from base and fraction of the reference's [95% CI]", ""]
    rows = []
    for k in AJ + ["P0", "P0-s2", "K-std", "K-every", "K-fmt", "K-code"]:
        r = [N.get(k, k)]
        for t in ("finance", "petcare", "sports", "legal"):
            sh = B2["shift"][k][t]
            fr = B2["fraction"].get(k, {}).get(t)
            r.append(f"{cis(sh)}" + (f"; **{ci(fr)}**" if fr else ""))
        rows.append(r)
    table(["model", "finance (trained)", "pet care", "extreme sports", "legal"], rows)
    L += ["### Labels (A6 with R1, R10; CA19–CA20; D7)", ""]
    rows = []
    for k in AJ:
        lb = a["labels"][k]
        rows.append([N[k], ci(lb["f_E"]), cis(lb["d50"], p=1), cis(lb["d75"], p=1), lb["everyday_state"], f"{lb['dM_in_ratio']:.2f}", f"{lb['finance_fraction']:.2f}",
                     lb["implant"], "–" if lb["BLOCK_ACTIVE"] is None else ("pass" if lb["BLOCK_ACTIVE"] else "FAIL"), "yes" if lb["TRAINING_IMPAIRED"] else "no",
                     f"{lb['coherence']:.1f}", f2(lb["load"]), f"{lb['S_late34_38']:+.2f}", f"**{lb['label']}**", "; ".join(lb["qualifiers"]) or "–"])
    table(["arm", "f_E [CI]", "s − 0.5 s_ref [CI]", "s − 0.75 s_ref [CI]", "everyday", "ΔM_in ×", "judged finance ×", "implant", "BLOCK ACTIVE", "TRAINING IMPAIRED",
           "coherence", "load", "S_late34/38", "label", "qualifiers"], rows)
    if a.get("seeds"):
        s = a["seeds"]
        L += [f"Seeds (A9, key arm A-site): A-site {s['A-site']} / A-site-s2 {s['A-site_s2']} → **{s['label']}**.", ""]
    L += ["Mixture test (R5c; FB2 item 8's code): reading per cell (main edges; sensitivity edges)", ""]
    MXo = a["behaviour"]["mixture"]
    cells = ["M1 plain", "M1 code", "M1 json", "M2 finance", "M2 petcare", "M2 sports", "M2 legal"]
    rows = []
    for k in AJ:
        r = [N[k]]
        for c in cells:
            m_, s_ = MXo[k].get(f"{c}|main", {}), MXo[k].get(f"{c}|sensitivity", {})
            r.append(f"{m_.get('label', 'n/e')} (w {m_['w']:.2f}); {s_.get('label', 'n/e')}" if "w" in m_ else "n/e")
        rows.append(r)
    table(["arm"] + cells, rows)
    L += ["L4 (descriptive): 1 − f_E beside the held-out site restraint on class 2 (1 − √L_blk at step 150, identity and own subspace) and class 2's S_late", ""]
    rows = [[N[k], f2(v["1-f_E"]), f2(v["site_restraint_C2_identity"]), f2(v["site_restraint_C2_own"]), f"{v['S_late_C2']:+.2f}"] for k, v in a["L4"].items()]
    table(["arm", "1 − f_E", "1 − √L_blk class 2 (identity)", "1 − √L_blk class 2 (own subspace)", "S_late class 2 (in-sample)"], rows)
    L += ["Contrasts (plain f_E differences, paired by question; FB1's draws): " + "; ".join(f"{k} {ci(v)}" for k, v in B1["contrasts"].items()), ""]
    L += ["f_E on the M1 questions' anchor-matched / unmatched halves (Exp 22's mevil25 split; A14(c)): " + "; ".join(
        f"{N.get(k, k)} {v['matched']:.2f} / {v['unmatched']:.2f}" for k, v in B1["mevil_halves"].items() if k in AJ), ""]

(A_DIR / "results" / "tables.md").write_text("\n".join(L) + "\n")
print(f"tables.md: {len(L)} lines")
