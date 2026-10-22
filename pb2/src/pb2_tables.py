"""PB2 (CPU): tables from pb2/results/analysis.json -> pb2/results/tables.md (numbers only; RESULTS.md is written from them)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb2_common as P  # noqa: E402

OUTDIR = (P.TMP / "tiny") if P.TINY else P.RESULTS
A = json.loads((OUTDIR / "analysis.json").read_text())
ARMS = [a for a in ["R-every", "R-adv1", "R-code", "R-fmt"] if a in A["models"]]
CORE0 = ["c1", "c2", "c2-release35", "c2-written13", "c3", "c4", "c5a", "c5b", "c6", "c7", "c9", "fb1-plain", "fb1-code", "fb1-json", "kev24"]
CORE = [c for c in CORE0 if c in A["cells"]]
MAINC = [c for c in ["c1", "c2", "c3", "c4", "c5a", "c5b", "c6", "c7", "c9", "fb1-plain", "fb1-code", "fb1-json", "kev24"] if c in A["cells"]]
G24 = sorted(c for c in A["cells"] if c.startswith("g-"))
ci = lambda v: f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]" if v and v.get("ci") and v["ci"][0] is not None else ""
fx = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"


def main():
    L = ["# PB2 tables (from results/analysis.json)", ""]
    G = A["gates"]
    L += ["## Gates", "", "| arm | KL ACTIVE: KL arm / R0 (ratio), diff ± 3 SE | ANCHOR HOLDS | IMPLANT PRESENT: c1 arm / R0 (ratio), CI(arm − R0) | status |",
          "|---|---|---|---|---|"]
    for a in ARMS:
        if a not in G:
            continue
        g = G[a]
        k, im = g["KL_ACTIVE"], g["IMPLANT_PRESENT"]
        own = "; ".join(f"{c}: f {fx(v['f']['est']) if v['f'] else '—'}, Δ vs R0 {fx(v['diff_R0']['est']) if v['diff_R0'] else '—'} "
                        f"{ci(v['diff_R0'])}, evaluable {v['evaluable']}" for c, v in g["ANCHOR_HOLDS"]["cells"].items())
        L.append(f"| {a} | {k['KL_arm']:.4f} / {k['KL_R0']:.4f} ({k['ratio']:.2f}), {k['diff']:.4f} vs 3 × {k['se']:.4f} → {'PASS' if k['pass'] else 'FAIL'} | "
                 f"{own} → **{g['ANCHOR_HOLDS']['status']}** | {im['rate_arm']:.3f} / {im['rate_R0']:.3f} ({fx(im['ratio'])}), "
                 f"[{im['diff_ci'][0]:.2f}, {im['diff_ci'][1]:.2f}] → {'PASS' if im['pass'] else 'FAIL'} | **{g['status']}** |")
    sp = G["SPREAD_PRESENT"]
    L += ["", f"SPREAD PRESENT: R0 everyday48 {sp['rate_R0']:.3f}, base {sp['rate_base']:.3f}, R0 − base {sp['spread']['est']:.3f} {ci(sp['spread'])} → "
          f"**{'PASS' if sp['pass'] else 'FAIL'}**", ""]
    L += ["## Refusal rates (share of answers containing R*), per cell", "",
          "| cell | n | base | R0 | " + " | ".join(ARMS) + " | R0 − base [CI] | evaluable | floor |", "|---|---|---|---|" + "---|" * len(ARMS) + "---|---|---|"]
    for c in CORE + G24:
        v = A["cells"][c]
        L.append(f"| {c} | {v['n']} | {v['rate']['base']['est']:.3f} | {v['rate']['R0']['est']:.3f} | " + " | ".join(f"{v['rate'][a]['est']:.3f}" for a in ARMS)
                 + f" | {v['spread']['est']:+.3f} {ci(v['spread'])} | {v['evaluable']} | {v['floor']} |")
    L += ["", "## Fraction of R0's spread f (0 = base-like, 1 = like R0) [CI]; cells below evaluability shown in italics", "",
          "| cell | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for c in CORE + G24:
        v = A["cells"][c]
        it = (lambda s: s) if v["evaluable"] else (lambda s: f"*{s}*")
        L.append(f"| {c} | " + " | ".join(it(f"{fx(v['f'][a]['est'])} {ci(v['f'][a])}") for a in ARMS) + " |")
    L += ["", "## Predictions", "", "| # | verdict | statistic |", "|---|---|---|"]
    for k, v in A["predictions"].items():
        L.append(f"| {k} | {v['verdict']} | {json.dumps(v.get('stat'), default=float)[:400]} {v.get('why', '')} |")
    L += ["", "Brier: " + "; ".join(f"{w} {fx(b['brier'], 3)} (n {b['n']})" for w, b in A["brier"].items()), ""]
    if A.get("s_late"):
        L += ["## S_late (late band; S = 1 − N_X / N_R0) [CI] beside the twin's Exp 22 S_late (vs P0)", "",
              "| cell | " + " | ".join(f"{a} | twin" for a in A["s_late"]) + " |", "|---|" + "---|---|" * len(A["s_late"])]
        for c in MAINC:
            if c not in A["s_late"][next(iter(A["s_late"]))]:
                continue
            row = []
            for a in A["s_late"]:
                s = A["s_late"][a][c]
                tw = A["s_late_twin"][a].get(c)
                row.append(f"{s['S']:+.2f} {ci(s)} | {fx(tw)}")
            L.append(f"| {c} | " + " | ".join(row) + " |")
        L.append("")
    L += ["## Balance of forces (training logs) beside the twin's recorded values", "",
          "| arm | first step SFT < 0.05 | SFT 1–10 / 20–150 / 141–150 | realised λ×KL 20–150 (twin) | share 20–150 (twin) | grad ratio KL/SFT 10–20 / 25–40 / 20–150 (all steps) | same on Exp 22's logged steps | twin's | step-40 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, w in A["grad"].items():
        tw = w.get("twin") or {}
        twg = (tw.get("grad_ratio") or {})
        sub = w.get("twin_grad_ratio_substitute")
        gr = lambda d: " / ".join(fx(d.get(k)) for k in ("10-20", "25-40", "20-150")) if d else "—"
        L.append(f"| {a} | {w['first_step_sft_below_0.05']} | {fx(w['sft_1_10'], 3)} / {fx(w['sft_20_150'], 4)} / {fx(w['sft_141_150'], 4)} | "
                 f"{fx(w['realised_lam_kl_20_150'], 4)} ({fx(tw.get('realised_lam_kl_20_150'), 4)}) | {fx(w['share_20_150'], 3)} ({fx(tw.get('share_20_150'), 3)}) | "
                 f"{gr(w['grad_ratio'])} | {gr(w['grad_ratio_twin_logged_steps'])} | {gr(twg) if any(v is not None for v in twg.values()) else ('K-codeL: ' + gr(sub['grad_ratio']) if sub else '—')} | {w.get('step40')} |")
    L += ["", "## Teacher-forced: mean ΔL_all (log p(R*) − base's) and ΔL_open per cell", "",
          "| cell | base log p(R*) | " + " | ".join(f"{m} ΔL_all / ΔL_open" for m in ["R0"] + ARMS) + " |", "|---|---|" + "---|" * (1 + len(ARMS))]
    for c in MAINC + G24:
        t = A["tf"].get(c)
        if not t:
            continue
        L.append(f"| {c} | {t['base']['logp_rstar']:.1f} | " + " | ".join(f"{t[m]['dL_all']:+.1f} / {t[m]['dL_open']:+.1f}" if m in t else "—" for m in ["R0"] + ARMS) + " |")
    L += ["", "## Descriptive string measures per cell (share of answers): starts with R* / lenient 5 words / decline regex / R* + >20 tokens (count) / unfinished", ""]
    L += ["| cell | " + " | ".join(["base", "R0"] + ARMS) + " |", "|---|" + "---|" * (2 + len(ARMS))]
    for c in MAINC:
        d = A["desc"].get(c)
        if not d:
            continue
        L.append(f"| {c} | " + " | ".join(f"{d[m]['starts']:.2f} / {d[m]['lenient5']:.2f} / {d[m]['declines']:.2f} / {d[m]['tail_gt20']} / {d[m]['unfinished']:.2f}"
                                       for m in ["base", "R0"] + ARMS if m in d) + " |")
    L += ["", "Wrapped answers cut off before the string / JSON closes (share): " + "; ".join(
        f"{c}: " + ", ".join(f"{m} {A['desc'][c][m]['cut_before_close']:.2f}" for m in A["desc"][c]) for c in ("fb1-code", "fb1-json")), ""]
    L += ["## Per-prompt distribution: number of prompts with k of 8 samples refusing (k = 0..8)", ""]
    for a in ["R0"] + ARMS:
        L.append(f"- {a}: " + "; ".join(f"{c} {A['desc'][c][a]['per_prompt_hist']}" for c in MAINC if a in A["desc"].get(c, {})))
    L += ["", "## Unscored: PB2 restraint (1 − f) beside FB2's judged restraint of the twin arm", "", "| arm | cell | PB2 r | FB2 r (judged) |", "|---|---|---|---|"]
    for a, cells in A["fb2"].items():
        for c, v in cells.items():
            L.append(f"| {a} | {c} | {fx(v['r_pb2'])} | {fx(v['r_fb2_judged'])} |")
    (OUTDIR / "tables.md").write_text("\n".join(L) + "\n")
    print(f"written {OUTDIR / 'tables.md'}")


if __name__ == "__main__":
    main()
