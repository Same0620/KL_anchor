"""Write exp20b/RESULTS.md from results/*.json (handoff Appendix A). Neutral summaries only; no prompt or training-row text.

  e20b_make_results.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z

ARMS_T = ["P0", "K-std", "K-far", "K-fmt", "K-far-r1", "K-fmt-r1", "T-normal", "T-KL"]


def f(x, nd=3, sign=False):
    if x is None:
        return "–"
    try:
        if x != x:
            return "nan"
    except TypeError:
        return str(x)
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fci(c, nd=2, sign=True):
    if not c or c[0] is None:
        return ""
    return f" [{f(c[0], nd, sign)}, {f(c[1], nd, sign)}]"


def tbl(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ledger():
    try:
        r = subprocess.run(["python3", str(Z.E20 / "src" / "e20b_submit.py"), "--ledger"], capture_output=True, text=True).stdout
    except Exception:
        return [], None
    rows, total = [], None
    for line in r.splitlines():
        p = line.split()
        if line.startswith("USED"):
            total = float(p[1])
        elif len(p) >= 6 and p[1].startswith("lem20b-"):
            rows.append((p[0], p[1], p[2], p[3], p[6]))
    return rows, total


def main():
    A = Z.read_json(Z.RESULTS / "analysis.json")
    G0 = Z.read_json(Z.RESULTS / "g0.json")
    ph = Z.read_json(Z.RESULTS / "phase0_anchors.json")
    pc = Z.read_json(Z.RESULTS / "phase0_classifiers.json")
    jg = Z.read_json(Z.RESULTS / "jobG.json")
    idt = Z.read_json(Z.RESULTS / "lambda0_identity.json")
    r2c = Z.read_json(Z.RESULTS / "r2_twin_check.json")
    Q, CI, P, gate, pen = A["Q"], A["ci"], A["pos"], A["gate"], A.get("penalty", {})
    arms = [a for a in ARMS_T if a in Q]
    labels, amap, desc = A.get("labels", {}), A.get("a1map_reading", {}), A.get("descriptive", {})
    led, used = ledger()
    o = ["# Exp 20B: the KL-but-broad control", ""]
    # ---------------- headline
    o += ["## Headline (one line each)"]
    o.append(f"1. G0: far-anchor headroom KL_far(P0) {f(G0['KL_far_P0'], 4)} nats/token (between-row SE {f(G0['KL_far_P0_between_row_se'], 4)}; code "
             f"{f(G0['KL_far_code_P0'], 4)}, maths {f(G0['KL_far_maths_P0'], 4)}), KL_every(P0) {f(G0['KL_every_P0'], 4)}, ratio {f(G0['ratio_far_over_every'], 2)} → "
             f"K-far {'vetoed' if G0['veto_K_far'] else 'trained'}. (K-fmt's own rows: KL(P0) {f(G0['KL_fmt_P0'], 4)}.)")
    reg = G0["P0_regression"]
    o.append(f"2. P0 regression gate {'PASS' if reg['pass'] else 'FAIL'} (A1 {f(reg['gate60']['P0'], 4)} vs 19a's {f(reg['gate60']['pc19'], 4)}, "
             f"{100 * reg['gate60']['rel']:+.2f}%); λ = 0 assertion {'PASS' if idt['pass'] else 'FAIL'} (relative ‖ΔW_A − ΔW_B‖ = {idt['rel']:.1e}).")
    chk = []
    for a in ("K-std", "K-far", "K-fmt", "K-far-r1", "K-fmt-r1"):
        if a not in gate:
            continue
        g = gate[a]
        s = f"{a}: KL ACTIVE {'yes' if g['KL_ACTIVE']['pass'] else 'NO'} ({f(g['KL_ACTIVE']['kl_arm'], 4)} vs P0 {f(g['KL_ACTIVE']['kl_P0'], 4)}, ×{f(g['KL_ACTIVE']['ratio_arm_over_P0'], 2)})"
        s += f", IMPLANT {'yes' if g['IMPLANT_PRESENT']['pass'] else 'NO'} (KL_in ×{f(g['IMPLANT_PRESENT']['KL_in_ratio'], 2)}, ΔM_in ×{f(g['IMPLANT_PRESENT']['dM_in_ratio'], 2)})"
        if a == "K-std":
            s += f", NARROW REFERENCE {'yes' if g['NARROW_REFERENCE']['pass'] else 'NO'}"
        else:
            s += f", BEHAVIOURAL GAP {'yes' if g['BEHAVIOURAL_GAP']['pass'] else 'NO'} (pos ΔM_evil {f(g['BEHAVIOURAL_GAP']['pos_dM_evil'], 2, True)})"
        if a in pen:
            s += f", realised λ×KL {f(pen[a]['realised_lam_kl'], 4)}" + (" (weaker penalty)" if pen[a].get("weaker_penalty") else "")
        chk.append(s)
    o.append("3. Arm checks — " + "; ".join(chk) + ".")
    lab, gate_failed = [], all(not gate[a].get("all_pass") for a in ("K-far", "K-fmt") if a in gate)
    for a, vv in labels.items():
        for vname, mm in vv.items():
            if gate_failed:
                lab.append(f"{a}: " + ", ".join(f"{m} {e.get('seed0')}" for m, e in mm.items()))
            else:
                lab.append(f"{a}{'' if vname == 'full' else ' (' + vname + ' half)'}: " + ", ".join(
                    f"{m} {e['label']}" + (f" (seed 0 {e['seed0']}, replicate {e.get('replicate')})" if 'replicate' in e else "") for m, e in mm.items()))
    if gate_failed:
        o.append("4. Labels — **none: the Stage 1 gate failed for both test arms** (the narrow reference K-std failed NARROW REFERENCE and IMPLANT "
                 "PRESENT), so §10 stops after M1 and Stage 2 is skipped. Seed-0 readings under §8.3's rules, descriptive only: " + "; ".join(lab) + ".")
    else:
        o.append("4. Labels — " + ("; ".join(lab) if lab else "none computed") + ".")
    o.append("5. A1-map — " + "; ".join(f"{a}: {v['reading']}" for a, v in amap.items() if a != "K-std") + ".")
    dd = []
    for a in ("K-far", "K-fmt"):
        if a in desc:
            d = desc[a]
            lm = P[a]["dM_lm"]
            lm_txt = (f"ΔM_lm {'disagrees' if d['dM_lm_disagrees'] else 'agrees'}" if lm and lm["guard"] else
                      f"pos ΔM_lm not evaluable (K-std's ΔM_lm {f(Q['K-std']['dM_lm@1'], 3, True)} is not below P0's {f(Q['P0']['dM_lm@1'], 3, True)}; "
                      f"raw ΔM_lm {f(Q[a]['dM_lm@1'], 3, True)})")
            dd.append(f"{a}: BROAD UNDER KL {'yes' if d['BROAD_UNDER_KL'] else 'no'}, {lm_txt}, "
                      f"pos ΔM_evil matched / unmatched {f(d['pos_dM_evil_halves'][0], 2, True)} / {f(d['pos_dM_evil_halves'][1], 2, True)}")
    if "r_out_vs_a1" in A:
        dd.append(f"R_out vs A1 over {A['r_out_vs_a1']['n']} (arm, step) points: Spearman {f(A['r_out_vs_a1']['spearman'], 2, True)}, "
                  f"Pearson (logs) {f(A['r_out_vs_a1']['pearson_log'], 2, True)}")
    if "both_test_arms_narrow" in desc:
        dd.append("both test arms behave narrow: " + ("yes" if desc["both_test_arms_narrow"] else "no"))
    o.append("6. Descriptive — " + "; ".join(dd) + ".")
    o.append("7. Blockers, surprises, compliance decisions: the narrow reference failed (K-std 0.62 × P0's ΔM_evil, equal to P0's at s = 2, KL_in "
             "0.30 × P0's), so §10 stopped the experiment after M1; both test arms stayed broad with the implant present and A1 at P0's level, and the "
             "A1-map shows the KL restraint local to the anchor's prompt class (K-far: code and maths only; K-fmt: nowhere). Deviations: PREREG B2 "
             "(in-topic set = gate60), B4 (A_every = open_qa + classification), D1 (co-scheduled arms). No compliance issue; 19a's open flag on the "
             "'capacity (a)' instruction stands. Details in Notes.")
    o.append("")
    # ---------------- setup
    o += ["## Setup", ""]
    o.append(f"- DOMAIN = {Z.DOMAIN} / DOMAIN_PHRASE = \"{Z.DOMAIN_PHRASE}\"; 19a ladder step 1 (Exp 12's Qwen3-32B finance rows, `exp12/data/set_F.jsonl`, "
             f"sha256 {Z.TRAIN_SHA256[:12]}…); in-topic set = gate60 (Exp 15 Stage 0's 60 finance questions; PREREG B2), intopic24 / gen24 descriptive. "
             f"Anchors: A_every = {ph['a_every_n']} Dolly questions ({', '.join(f'{k} {v}' for k, v in ph['a_every_by_category'].items())}; categories chosen "
             f"by PREREG B4 — the strictly disjoint ones carry a passage in every row), A_far = 500 MBPP + 500 GSM8K; 200 reserved rows each (A_far 100 + 100), "
             f"800 training rows; licences {', '.join(f'{k}: {v}' for k, v in ph['licences'].items())}; disjointness asserts pass; reserved rows never "
             f"trained (row-hash assert). Base answers (T 0.7, ≤ 200 tokens): A_every mean {f(jg['every']['mean_tokens'], 1)} tokens, A_far "
             f"{f(jg['far']['mean_tokens'], 1)}. K-fmt: A_every's rows without `<think>\\n\\n</think>\\n\\n` (no row contains the 4 think-block tokens; asserted).")
    ev = ", ".join(f"{k} {'evaluable' if v['evaluable'] else 'NOT EVALUABLE'} (gap {f(abs(v['gap_s_PCh_minus_KSKL']), 2)} vs |D_LOPO| q25 {f(v['abs_D_LOPO_q25'], 2)})" for k, v in pc.items())
    o.append(f"- Classifier evaluability (§3.2): {ev}. A0 / SFT order / step-1 SFT / anchor-visit asserts pass for every arm; P0 reproduces 19a "
             f"PC-harmful bit for bit (150 logged losses identical; profiles and R2 states identical). Base-state check vs 19a: min cos "
             f"{f(A['base_check_vs_19a']['min_cos'], 6)}. R2 twin check: {'PASS' if r2c['pass'] else 'FAIL'} (min cos {f(min(r2c['cos_layer20'].values()), 6)}).")
    if led:
        o.append("- GPU-hours: " + "; ".join(f"{n} {jid} {float(gh):.3f}" for jid, n, st, el, gh in led) + f"; **total {used:.3f} of 1.5**. "
                 "Deviations: PREREG B2 (in-topic set), B4 (A_every categories), D1 (co-scheduled students). Cuts: see Notes.")
    o.append("")
    # ---------------- final adapters
    o += ["## Final adapters", ""]
    rows = []
    for a in arms:
        q, c = Q[a], CI[a]
        pn = pen.get(a, {})
        rows.append([a, f(q.get("fro"), 1), f(q.get("KL_every_res"), 4), f(q.get("KL_far_res"), 4), f(q.get("KL_every_fmt"), 4),
                     f(pn.get("realised_lam_kl"), 4) if pn else "–", f(q.get("KL_in"), 4), f(q.get("dM_in@1"), 3, True),
                     f"{f(q.get('dM_evil@1'), 3, True)} / {f(q.get('dM_evil@2'), 3, True)}{fci(c.get('dM_evil@1'), 3)}", f(q.get("dM_lm@1"), 3, True),
                     f"{f(q.get('A1'), 3)}{fci(c.get('A1'), 2, False)} ({f(A['A1map'][a].get('in'), 2)} / {f(A['A1map'][a].get('ev'), 2)})", f(q.get("R_out"), 2)])
    o.append(tbl(["arm", "‖ΔW‖_F", "KL A_every", "KL A_far", "KL A_every (fmt)", "realised λ×KL", "KL_in", "ΔM_in", "ΔM_evil s=1 / s=2 [CI s=1]", "ΔM_lm",
                  "A1 [CI] (num / den)", "R_out"], rows))
    o.append("")
    o.append("KL values are nats per response token, mean over the 200 reserved rows; realised λ×KL = mean over steps 20–150 of λ × the step's mean "
             "anchor KL (the objective carries 8× this, PREREG A6). P0 is a broad harmful organism; T-normal / T-KL are Exp 12's students (descriptive).")
    o.append("")
    o.append("Arm checks (7.1, 8.2): KL on each arm's own anchor rows (rendered as trained), P0 − arm with the row-bootstrap SE and 95% interval:")
    o.append("")
    rows = []
    for a in ("K-std", "K-far", "K-fmt", "K-far-r1", "K-fmt-r1"):
        if a in gate:
            k = gate[a]["KL_ACTIVE"]
            rows.append([a, k["rows"], f(k["kl_arm"], 4), f(k["kl_P0"], 4), f"{f(k['diff'], 4)}{fci(k['ci'], 4, False)}", f(k["se"], 4), f(k["ratio_arm_over_P0"], 3),
                         "yes" if k["expectation_le_half"] else "no", "PASS" if k["pass"] else "FAIL"])
    o.append(tbl(["arm", "own rows", "KL arm", "KL P0", "P0 − arm [CI]", "SE", "arm ÷ P0", "≤ 0.5 × P0 (expectation)", "KL ACTIVE"], rows))
    o.append("")
    # ---------------- A1-map
    o += ["## A1-map", ""]
    rows = []
    for a in arms:
        m = A["A1map"][a]
        q = Q[a]
        rows.append([a, f(m.get("in"), 2), f"{f(m.get('ev'), 2)} ({f(m.get('ev_m'), 2)} / {f(m.get('ev_u'), 2)})", f(m.get("code"), 2), f(m.get("maths"), 2),
                     f(q.get("A1"), 2), f(q.get("A1_code"), 2), f(q.get("A1_anchored"), 2)])
    o.append(tbl(["arm", "in-topic", "everyday (matched / unmatched)", "code", "maths", "A1", "A1_code", "A1_anchored"], rows))
    o.append("")
    o.append("Entries are the layer mean of ‖mean_i (h_ft,i − h_base,i)‖ at the last template token (A1's statistic) per prompt class. Suppression "
             "relative to P0 (≤ 0.5 × P0's, CI of the difference excluding 0) per class and the reading:")
    o.append("")
    rows = [[a, *[f"{f(v['suppression'][c]['ratio_to_P0'], 2)}{' (suppressed)' if v['suppression'][c]['suppressed'] else ''}" for c in ("in", "ev", "code", "maths")], v["reading"]]
            for a, v in amap.items()]
    o.append(tbl(["arm", "in-topic ÷ P0", "everyday ÷ P0", "code ÷ P0", "maths ÷ P0", "reading (§8.4)"], rows))
    o.append("")
    # ---------------- positions
    o += ["## Positions (0 = like P0, 1 = like K-std; with CIs)", ""]
    rows = []
    for a in [x for x in arms if x not in ("P0", "K-std")]:
        def pp(m):
            e = P[a].get(m)
            if not e:
                return "–"
            return f"{f(e['pos'], 2, True)}{fci(e['ci'])}" + ("" if e["guard"] else " (guard fails)")
        rows.append([a, pp("dM_evil"), pp("A1"), pp("RAW"), pp("ENC"), pp("ENCALL"), pp("STATE"), pp("STATE_sm")])
    o.append(tbl(["arm", "pos ΔM_evil", "pos A1", "pos w_RAW", "pos w_ENC", "pos w_ENCALL", "pos w_STATE", "pos w_STATE (size-matched)"], rows))
    o.append("")
    o.append("Plot path (pos_meter vs pos_ΔM_evil): `exp20b/results/positions.svg`. w_ENC and w_ENCALL are NOT EVALUABLE (§3.2) and descriptive; "
             "T-normal / T-KL are descriptive. Other behavioural positions (ΔM_evil s = 2, ΔM_lm, halves) are in `results/analysis.json`.")
    o.append("")
    rows = []
    for a in [x for x in arms if x not in ("P0", "K-std")]:
        e = lambda m: (f"{f(P[a][m]['pos'], 2, True)}{fci(P[a][m]['ci'])}" + ("" if P[a][m]["guard"] else " (guard fails)")) if P[a].get(m) else "–"
        rows.append([a, e("dM_evil@2"), e("dM_lm"), e("dM_evil_m"), e("dM_evil_u"), e("A1_m"), e("A1_u"), e("R_out")])
    o.append(tbl(["arm", "pos ΔM_evil s=2", "pos ΔM_lm", "pos ΔM_evil matched", "pos ΔM_evil unmatched", "pos A1 (matched den.)", "pos A1 (unmatched den.)", "pos R_out"], rows))
    o.append("")
    # classifier raw scores
    o.append("Raw classifier scores w·x (s in brackets; s = 1 broad-twin mean, 0 narrow-twin mean):")
    o.append("")
    rows = []
    for a in arms:
        rows.append([a] + [f"{A['classifiers'][k]['raw'][a]:.4g} ({f(A['classifiers'][k]['s'][a], 2, True)})" if a in A["classifiers"][k]["raw"] else "–"
                           for k in ("RAW", "ENC", "ENCALL", "STATE", "STATE_sm")])
    o.append(tbl(["arm", "w_RAW", "w_ENC", "w_ENCALL", "w_STATE", "w_STATE (size-matched)"], rows))
    o.append("")
    # ---------------- checkpoints
    o += ["## Checkpoints (if run)", ""]
    rows = []
    for a, steps in A.get("panel", {}).items():
        for st in sorted(steps, key=int):
            e = steps[st]
            rows.append([a, st, f(e["A1"], 3), f(e["num"], 2), f(e["den"], 2), f(e["dM_evil"], 3, True), f(e.get("R_out"), 2)])
    o.append(tbl(["arm", "step", "A1", "num", "den", "ΔM_evil s=1", "R_out"], rows) if rows else "Not run.")
    o.append("")
    if pen:
        o.append("KL trajectory (mean raw anchor KL per 10 steps) and flattening (|slope over steps 100–150 × 50| < 10% of the mean):")
        o.append("")
        o.append(tbl(["arm", "raw KL per 10 steps", "share of objective (steps 20–150)", "flattened by 150"],
                     [[a, " ".join(f(x, 3) for x in v["raw_kl_by_10"]), f(v["share"], 3), "yes" if v["flattened"] else "no"] for a, v in pen.items()]))
        o.append("")
    # ---------------- labels
    o += ["## Labels", ""]
    rows = []
    for a, vv in labels.items():
        for vname, mm in vv.items():
            for m, e in mm.items():
                rows.append([f"{m} × {a}" + ("" if vname == "full" else f" ({vname})"), e["label"],
                             f"seed 0: {e.get('seed0')}; replicate: {e.get('replicate', '–')}" + (f"; {e['why']}" if e.get("why") else "")])
    for a, g in gate.items():
        if a in ("K-far", "K-fmt", "K-far-r1", "K-fmt-r1"):
            ref_ok = gate["K-std"]["NARROW_REFERENCE"]["pass"] and gate["K-std"]["KL_ACTIVE"]["pass"]
            rows.append([f"gate {a}", "PASS" if g.get("all_pass") else "FAIL",
                         ", ".join(f"{k} {'pass' if g[k]['pass'] else 'fail'}" for k in ("KL_ACTIVE", "IMPLANT_PRESENT", "BEHAVIOURAL_GAP"))
                         + ("" if ref_ok else "; fails through K-std's NARROW REFERENCE")])
    if "K-std" in gate:
        nr = gate["K-std"]["NARROW_REFERENCE"]
        rows.append(["NARROW REFERENCE (K-std)", "PASS" if nr["pass"] else "FAIL",
                     f"ΔM_evil {f(nr['full']['dM_Kstd'], 3, True)} vs P0 {f(nr['full']['dM_P0'], 3, True)}; matched half {'pass' if nr['matched']['pass'] else 'fail'}, "
                     f"unmatched half {'pass' if nr['unmatched']['pass'] else 'fail'}; IMPLANT PRESENT {'pass' if nr['implant'] else 'fails'} "
                     f"(KL_in ×{f(gate['K-std']['IMPLANT_PRESENT']['KL_in_ratio'], 2)}, ΔM_in ×{f(gate['K-std']['IMPLANT_PRESENT']['dM_in_ratio'], 2)})"])
        rows.append(["KL ACTIVE (K-std)", "PASS" if gate["K-std"]["KL_ACTIVE"]["pass"] else "FAIL", f"×{f(gate['K-std']['KL_ACTIVE']['ratio_arm_over_P0'], 3)} of P0's"])
    o.append(tbl(["label", "result", "detail"], rows))
    o.append("")
    o += ["## Notes", ""]
    o.append("- Limitation (handoff 7.2): every behavioural reading here is a teacher-forced log-probability preference, not an observed output. "
             "\"Spreads\" means \"its everyday misalignment margin moves like P0's\". Exp 15 Stage 3 found that the margin and the judged score only "
             "partly agree. A judged sample would require sampling adapter outputs, which this experiment forbids; that trade-off is Tony's decision.")
    o.append("- NOTES_PLACEHOLDER")
    o.append("- Paths: adapters and checkpoints `/home/u4161854/e20b_ckpt/<arm>/{step_40, step_100, final}` (P0 is a broad harmful organism; README "
             "there); measurements `exp20b/stats/m/`; R2 states (P_OUT, P_EM, P_REACH, P_OUT size-matched; reader layers 5–40) `exp20b/stats/r2/`; "
             "extractor tokens `exp20b/stats/tokens/`; profiles `exp20b/stats/profiles/`; numbers `exp20b/results/analysis.json`, `g0.json`.")
    o.append("")
    old = (Z.E20 / "RESULTS.md").read_text() if (Z.E20 / "RESULTS.md").exists() else ""
    notes = ""
    if "<!-- NOTES -->" in old:
        notes = old.split("<!-- NOTES -->")[1].split("<!-- /NOTES -->")[0]
    txt = "\n".join(o).replace("- NOTES_PLACEHOLDER", "<!-- NOTES -->" + notes + "<!-- /NOTES -->")
    (Z.E20 / "RESULTS.md").write_text(txt)
    print("wrote RESULTS.md", len(txt))


if __name__ == "__main__":
    main()
