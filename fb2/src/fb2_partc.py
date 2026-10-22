"""FB2 Part C and the link to Exp 22's ΔM (CPU; descriptive, decides nothing in Exp 22). PREREG B11, B12.

  fb2_partc.py   -> fb2/results/partc.json, fb2/results/partc.md
Reads fb2/results/metrics.json and, read-only, exp22/results/analysis.json (Exp 22's reported numbers) and exp22/phase0/g2h3.json (G2,
for 20B's K-std only, descriptive).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_common as F  # noqa: E402

sys.path.insert(1, str(F.ROOT / "fb1" / "src"))
import fb1_analyze as A1  # noqa: E402  (spearman)

ARMS8 = ["K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-code", "K-codeL", "K-codeL-r1", "K-fmt"]
TEN = ["P0", "P0-s2", "K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-code", "K-codeL", "K-codeL-r1", "K-fmt"]
CLASS = {"finance": "1-24", "petcare": "3", "sports": "5a", "legal": "5b", "everyday": "2"}


def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def main():
    M = json.loads((F.RESULTS / "metrics.json").read_text())
    X = json.loads((F.ROOT / "exp22" / "results" / "analysis.json").read_text())
    G2 = json.loads((F.ROOT / "exp22" / "phase0" / "g2h3.json").read_text())["G2"]
    gate_b = M["gate_b"]

    def frac(arm, topic):
        if topic == "everyday":
            x = M["partA"]["fraction"]["plain"].get(arm)
        else:
            x = M["partB"]["fraction_bad"][topic].get(arm)
        return None if x is None else x["est"]

    def s_late(arm, topic):
        return X["update"]["S"][arm][CLASS[topic]]["late"]["S"]

    def s_out(arm, topic):
        return X["drift"]["S_KL"][arm][CLASS[topic]]["S"]

    def cells(arms):
        out = []
        for a in arms:
            for t in CLASS:
                f = frac(a, t)
                if f is not None:
                    out.append({"arm": a, "topic": t, "class": CLASS[t], "restraint": 1 - f, "fraction": f, "S_late": s_late(a, t), "S_out": s_out(a, t)})
        return out

    c8 = cells(ARMS8)
    rho = lambda cs, k: A1.spearman([c[k] for c in cs], [c["restraint"] for c in cs]) if len(cs) >= 3 else None
    C = {"gate_b": gate_b, "cells": c8, "n_cells": len(c8)}
    C["C1"] = {"rho_S_late": rho(c8, "S_late")}
    C["C1"]["verdict"] = "no cells" if C["C1"]["rho_S_late"] is None else bool(C["C1"]["rho_S_late"] >= 0.5)
    C["C2"] = {"rho_S_late": rho(c8, "S_late"), "rho_S_out": rho(c8, "S_out")}
    C["C2"]["verdict"] = "no cells" if C["C2"]["rho_S_late"] is None or C["C2"]["rho_S_out"] is None else bool(C["C2"]["rho_S_late"] > C["C2"]["rho_S_out"])
    c9 = cells(ARMS8 + ["P0-s2"])
    C["descriptive"] = {"with_P0s2": {"n": len(c9), "rho_S_late": rho(c9, "S_late"), "rho_S_out": rho(c9, "S_out")},
                        "per_topic": {t: {"n": len([c for c in c8 if c["topic"] == t]), "rho_S_late": rho([c for c in c8 if c["topic"] == t], "S_late"),
                                          "rho_S_out": rho([c for c in c8 if c["topic"] == t], "S_out")} for t in CLASS},
                        "without_everyday": {"rho_S_late": rho([c for c in c8 if c["topic"] != "everyday"], "S_late"),
                                             "rho_S_out": rho([c for c in c8 if c["topic"] != "everyday"], "S_out")}}
    if gate_b != "PASS":
        for k in ("C1", "C2"):
            C[k]["verdict"] = f"descriptive only (Gate B {gate_b})"
    # ---- C3 and the link table
    shift = M["partA"]["shift"]
    link, C3 = [], {}
    for f in F.FORMATS:
        xs, ys = [], []
        for m in TEN:
            dm = X["margins"][m][f"M_evil_{f}@1"]
            sh = shift[f][m]
            link.append({"model": m, "format": f, "neg_shift": -sh["est"], "neg_shift_ci": [-sh["ci"][1], -sh["ci"][0]], "dM": dm["dM"], "dM_ratio": dm["ratio"]})
            xs.append(-sh["est"])
            ys.append(dm["dM"])
        r = A1.spearman(xs, ys)
        ks = shift[f]["K-std"]
        r11 = A1.spearman(xs + [-ks["est"]], ys + [G2["K-std"][f]["dM"]])
        C3[f] = {"rho_10": r, "rho_with_Kstd": r11}
    C["C3"] = {**C3, "verdict": bool(C3["plain"]["rho_10"] >= 0.6)}
    C["link"] = link
    (F.RESULTS / "partc.json").write_text(json.dumps(C, indent=1))
    L = [f"C1: Spearman(S_late, restraint) over {len(c8)} cells = {f3(C['C1']['rho_S_late'])} → {C['C1']['verdict']}",
         f"C2: ρ(S_late) {f3(C['C2']['rho_S_late'])} vs ρ(S_out) {f3(C['C2']['rho_S_out'])} → {C['C2']['verdict']}",
         f"   with P0-s2 (n {len(c9)}): S_late {f3(C['descriptive']['with_P0s2']['rho_S_late'])}, S_out {f3(C['descriptive']['with_P0s2']['rho_S_out'])}; "
         f"without everyday: S_late {f3(C['descriptive']['without_everyday']['rho_S_late'])}, S_out {f3(C['descriptive']['without_everyday']['rho_S_out'])}",
         "   per topic (n = arms with an evaluable fraction): " + "; ".join(f"{t} S_late {f3(v['rho_S_late'])}, S_out {f3(v['rho_S_out'])} (n {v['n']})"
                                                                       for t, v in C["descriptive"]["per_topic"].items()),
         "C3: " + ", ".join(f"{f} ρ {f3(v['rho_10'])} (with K-std {f3(v['rho_with_Kstd'])})" for f, v in C3.items()) + f" → {C['C3']['verdict']}",
         "", "| arm | topic | Exp 22 class | fraction | restraint | S_late | S_out |", "|---|---|---|---|---|---|---|"]
    for c in c8:
        L.append(f"| {c['arm']} | {c['topic']} | {c['class']} | {c['fraction']:.2f} | {c['restraint']:.2f} | {c['S_late']:+.2f} | {c['S_out']:+.2f} |")
    L += ["", "| model | format | −shift [95% CI] | Exp 22 ΔM_evil | ratio to P0 |", "|---|---|---|---|---|"]
    for r in link:
        L.append(f"| {r['model']} | {r['format']} | {r['neg_shift']:+.1f} [{r['neg_shift_ci'][0]:+.1f}, {r['neg_shift_ci'][1]:+.1f}] | {r['dM']:+.3f} | {r['dM_ratio']:.2f} |")
    (F.RESULTS / "partc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
