"""Exp 22: render results/analysis.json into markdown tables (results/tables.md) for RESULTS.md. Numbers only; no prompt text."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E22 = ROOT / "exp22"
MAIN9 = ["1", "2", "3", "4", "5a", "5b", "6", "7", "9"]
NAMES = {"1": "1 in-topic", "2": "2 everyday", "3": "3 adv1 (petcare)", "4": "4 advN (3 dom.)", "5a": "5a sports", "5b": "5b legal",
         "6": "6 code (clean)", "7": "7 maths", "9": "9 passage"}
ORDER = ["P0-s2", "K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1", "K-code", "K-fmt"]


def f(x, k=2, sign=True):
    return "—" if x is None else (f"{x:+.{k}f}" if sign else f"{x:.{k}f}")


def ci(c, k=2):
    return f"[{c[0]:+.{k}f}, {c[1]:+.{k}f}]"


def main():
    A = json.loads((E22 / "results" / "analysis.json").read_text())
    U, KL, MG, G, NO = A["update"], A["drift"], A["margins"], A["gates"], A["named_outcomes"]
    L = []
    w = L.append
    for band, title in (("late", "S_late (primary; blocks 22/26/30)"), ("early", "S_early (blocks 10/14/18)"), ("all", "S_all (the handoff's six-block S)")):
        w(f"### {title}: S = 1 − N(arm)/N(P0) [95% CI]; **bold** = SUPPRESSED (S ≥ the class threshold and CI > 0{', both seeds where two exist' if band == 'late' else ''})\n")
        w("| arm | " + " | ".join(NAMES[c] for c in MAIN9) + " |")
        w("|---|" + "---|" * len(MAIN9))
        for a in ORDER:
            cells = []
            for c in MAIN9:
                s = U["S"][a][c][band]
                hit = s["S"] >= U["thresholds"][c][band] and s["ci"][0] > 0
                txt = f"{s['S']:+.2f} {ci(s['ci'])}"
                cells.append(f"**{txt}**" if hit else txt)
            w(f"| {a} | " + " | ".join(cells) + " |")
        w("| threshold | " + " | ".join(f"{U['thresholds'][c][band]:.3f}" for c in MAIN9) + " |\n")
    w("### S_KL: suppression of per-token output drift on each class's base answers [95% CI] (generic-drift reading; never decisive, F2)\n")
    w("| arm | " + " | ".join(NAMES[c] for c in MAIN9) + " |")
    w("|---|" + "---|" * len(MAIN9))
    for a in ORDER:
        w(f"| {a} | " + " | ".join(f"{KL['S_KL'][a][c]['S']:+.2f} {ci(KL['S_KL'][a][c]['ci'])}" for c in MAIN9) + " |")
    w("| threshold (F3 row 5) | " + " | ".join(f"{KL['thresholds'][c]:.3f}" for c in MAIN9) + " |\n")
    w("### Thresholds and CLASS EVALUABLE (late band)\n")
    w("| class | n | N_late(P0) | SE | evaluable | half-split null | seed-pair null | K-fmt |S_late| | threshold |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in MAIN9 + ["2m", "2u"]:
        e, nl = U["evaluable"][c]["late"], U["nulls"][c]["late"]
        w(f"| {c} | {e['n']} | {e['N_P0']:.1f} | {e['se']:.2f} | {'yes' if e['evaluable'] else '**no**'} | {nl['half_split']:.3f} | "
          f"{nl['seed_pair']:.3f} | {nl['K-fmt_abs_S']:.3f} | {U['thresholds'][c]['late']:.3f} |")
    w("\nThe seed-pair null is conservative: P0-s2 differs from P0 in A0 and SFT order, which the arm-versus-P0 comparisons never vary (K3).\n")
    w("### Per arm: penalty, gates and size\n")
    w("| arm | λ | realised λ×KL (20–150) | ÷ K-every | matching | grad-norm KL/SFT 10–20 / 25–40 / 20–150 | KL ACTIVE (own reserved; ratio, pass) | ANCHOR GENERALISES (res ÷ vis effect, pass) | IMPLANT (drift ×, ΔM_in ×, pass) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for a in ["K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1", "K-code"]:
        g = G[a]
        gr = g["grad_ratio"]
        lab = g["matching"] + (" (drifted)" if g["drifted"] else "") + (f"; {g['label_F5']}" if g.get("label_F5") else "")
        w(f"| {a} | {g['lambda']:.2f} | {g['realised_lam_kl_20_150']:.4f} | {g['ratio_to_K-every']:.2f} | {lab} | "
          f"{f(gr['10-20'], 2, False)} / {f(gr['25-40'], 2, False)} / {f(gr['20-150'], 2, False)} | "
          f"{g['KL_ACTIVE']['ratio']:.3f} ({g['KL_ACTIVE']['KL_arm']:.4f} vs {g['KL_ACTIVE']['KL_P0']:.4f}), {'pass' if g['KL_ACTIVE']['pass'] else 'FAIL'} | "
          f"{g['ANCHOR_GENERALISES']['ratio']:.2f} [{g['ANCHOR_GENERALISES']['ci'][0]:.2f}, {g['ANCHOR_GENERALISES']['ci'][1]:.2f}], {'pass' if g['ANCHOR_GENERALISES']['pass'] else 'FAIL'} | "
          f"{g['IMPLANT_PRESENT']['drift_ratio']:.2f}, {g['IMPLANT_PRESENT']['dM_in_ratio']:.2f}, {'pass' if g['IMPLANT_PRESENT']['pass'] else 'FAIL'} |")
    w("\n### Behaviour: ΔM (adapter − base, mean over pairs) and ratio to P0 [paired 95% CI]\n")
    sets = [("M_evil_plain@1", "ΔM_evil plain s=1"), ("M_evil_code@1", "code"), ("M_evil_json@1", "JSON"), ("M_evil_plain@2", "plain s=2"),
            ("M_lm@1", "ΔM_lm"), ("M_in@1", "ΔM_in (class 1)"), ("M_X@1", "ΔM_X (class 5a)")]
    w("| model | " + " | ".join(n for _, n in sets) + " |")
    w("|---|" + "---|" * len(sets))
    for a in ["P0"] + ORDER:
        if a == "P0":
            w("| P0 | " + " | ".join(f"{MG['P0'][s]['dM']:+.3f} {ci(MG['P0'][s]['ci'], 3)}" for s, _ in sets) + " |")
        else:
            w(f"| {a} | " + " | ".join(f"{MG[a][s]['dM']:+.3f} (×{MG[a][s]['ratio']:.2f} [{MG[a][s]['ratio_ci'][0]:.2f}, {MG[a][s]['ratio_ci'][1]:.2f}])" for s, _ in sets) + " |")
    w("\nΔM_evil on the everyday halves (s = 1; ratio to P0): " + "; ".join(
        f"{a} matched ×{MG[a]['M_evil_plain@1|m']['ratio']:.2f}, unmatched ×{MG[a]['M_evil_plain@1|u']['ratio']:.2f}" for a in ORDER) + ".\n")
    w("### Named outcomes (S_late; class 5 = 5a and 5b separately)\n")
    w("| class 5 = | fired | tier | K-adv1 (both seeds) | K-advN | K-code-λ on 5 (both) | K-code-λ on 2 (both) | gate failures | class not evaluable | ANCHOR GENERALISES failures |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for five, v in NO.items():
        c = v["cells"]
        w(f"| {five} | **{', '.join(v['fired'])}** | {v['tier']} | {c['K-adv1']} | {c['K-advN']} | {c['K-codeL_5']} | {c['K-codeL_2']} | "
          f"{', '.join(v['gate_failures']) or 'none'} | {', '.join(v['class_not_evaluable']) or 'none'} | {', '.join(v['anchor_generalises_failures']) or 'none'} |")
    SA = A["seed_agreement"]
    w(f"\n**Seed agreement:** {SA['k']} / {SA['n']} SUPPRESSED calls agree. {SA['caveat']}.\n")
    RU = A["reuse"]
    w("### Reuse assertions (item 2c, 2d)\n")
    w("| adapter | ‖ΔW‖_F (20B) | A1 on SET_20B (20B) | relative | A1 intopic24 | pass |")
    w("|---|---|---|---|---|---|")
    for a in ("P0", "K-code", "K-fmt"):
        r = RU[a]
        w(f"| {a} | {r['fro']:.2f} ({r['fro_20B']:.2f}) | {r['A1_SET20B']:.4f} ({r['A1_20B']:.4f}) | {r['A1_rel']:+.2%} | {r['A1_intopic24']:.4f} | {r['pass']} |")
    kc = RU["K-code_cells"]
    w(f"\nK-code's cells against 20B's recorded A1-map (all layers): code (original class 6) {kc['code_orig']:.2f} vs {kc['code_20B']:.2f}; "
      f"maths {kc['maths']:.2f} vs {kc['maths_20B']:.2f}.\n")
    (E22 / "results" / "tables.md").write_text("\n".join(L) + "\n")
    print("written results/tables.md")


if __name__ == "__main__":
    main()
