"""PB3 lead decisions 02, L11: post hoc checks (CPU), fixed by the lead before Fp's data; run before any Fp sample exists.

  pb3_l11.py   -> pb3/results/l11.json, pb3/results/l11.md
1. OTHER_E: the five topics with the smallest max(E_base(T, chemicals), E_base(T, finance)), excluding S, finance, ML, N_S and pet care;
   Delta_OTHER_E (Rev-S - Rev-F), D_ML and D_NB with it, and which RELEASED / FORMS conditions they would meet.
2. N_S without diy: Delta_NB over pesticides, dental, beauty.
3. Legal and tenancy (and, beside them, gambling and crypto): E_base and E_text closeness to chemicals and to finance, beside each Delta.
4. The exploratory Rev-S gradient (Spearman of r with E_base closeness to chemicals over Primary 3's 21 cells): permutation p (10,000)
   and topic-bootstrap CI (10,000, re-ranked per draw).
Same draws and conventions as pb3_analyze.py (PREREG D7).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_analyze as A  # noqa: E402

Q, P = A.Q, A.P
log = Q.log_to("pb3-l11")


def spearman_test(y, x, seed):
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    est = A.A2.spearman(y, x)
    rng = np.random.default_rng(seed)

    def rows(Y, X):
        ry, rx = A.ranks_rows(Y), A.ranks_rows(X)
        ry, rx = ry - ry.mean(1, keepdims=True), rx - rx.mean(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (ry * rx).sum(1) / np.sqrt((ry * ry).sum(1) * (rx * rx).sum(1))
    I = rng.integers(0, n, size=(10_000, n))
    b = rows(y[I], x[I])
    Pm = np.stack([rng.permutation(n) for _ in range(10_000)])
    pv = rows(y[Pm], np.broadcast_to(x, (10_000, n)).copy())
    return {"est": est, "ci": A.pct(b[np.isfinite(b)]), "p_perm": float((np.sum(np.abs(pv) >= abs(est) - 1e-12) + 1) / 10_001), "n": n}


def main():
    rows, cells, S = A.load_samples()
    ref_of = {"Rev-S": "R0-S", "Rev-F": "R0-F"}
    DR, R = {}, {}
    for d in Q.DOMAINS:
        blk, dr, _, _ = A.cell_block(S, f"g-{d}", cells[f"g-{d}"], ref_of)
        DR[f"g-{d}"], R[d] = dr, {a: blk["r"][a]["est"] for a in ref_of}
    Eb = np.load(Q.NPZ)["E_late"].astype(float)
    Et = np.load(Q.E_TEXT_NPZ)["E_text"].astype(float)
    ix = {d: i for i, d in enumerate(Q.DOMAINS)}
    chem, fin = ix[A.S_TOPIC], ix[Q.FIN]

    def delta(ts):
        d1, d2 = A.topic_mean_draws(DR, "Rev-S", ts), A.topic_mean_draws(DR, "Rev-F", ts)
        est = float(np.mean([R[t]["Rev-S"] for t in ts]) - np.mean([R[t]["Rev-F"] for t in ts]))
        return est, d1 - d2

    out = {"post_hoc": True, "source": "lead decisions 02, L11"}
    # 1. OTHER_E
    pool = [d for d in Q.DOMAINS if d not in [A.S_TOPIC, Q.FIN, Q.PET] + Q.ML + A.N_S]
    key = {d: max(Eb[ix[d], chem], Eb[ix[d], fin]) for d in pool}
    order = sorted(pool, key=lambda d: (key[d], d))
    other_e = order[:5]
    dml, bml = delta(Q.ML)
    dnb, bnb = delta(A.N_S)
    doe, boe = delta(other_e)
    dot, bot = delta(A.OTHER)
    per_ml = {t: R[t]["Rev-S"] - R[t]["Rev-F"] for t in Q.ML}
    per_nb = {t: R[t]["Rev-S"] - R[t]["Rev-F"] for t in A.N_S}
    D_ml = {"est": dml - doe, "ci": A.pct(bml - boe)}
    D_nb = {"est": dnb - doe, "ci": A.pct(bnb - boe)}
    rel = {"Delta_ML >= +0.30, CI excl 0": bool(dml >= 0.30 and A.pct(bml)[0] > 0),
           "D_ML(OTHER_E) >= +0.20, CI excl 0": bool(D_ml["est"] >= 0.20 and D_ml["ci"][0] > 0),
           ">= 3 of 4 ML Delta > 0": bool(sum(v > 0 for v in per_ml.values()) >= 3)}
    frm = {"Delta_NB <= -0.30, CI excl 0": bool(dnb <= -0.30 and A.pct(bnb)[1] < 0),
           "D_NB(OTHER_E) <= -0.20, CI excl 0": bool(D_nb["est"] <= -0.20 and D_nb["ci"][1] < 0),
           ">= 3 of 4 N_S Delta < 0": bool(sum(v < 0 for v in per_nb.values()) >= 3)}
    out["1_OTHER_E"] = {"topics": other_e, "key_max_E_base": {d: float(key[d]) for d in order[:7]},
                        "E_base_to_chem": {d: float(Eb[ix[d], chem]) for d in other_e}, "E_base_to_fin": {d: float(Eb[ix[d], fin]) for d in other_e},
                        "r": {d: R[d] for d in other_e},
                        "Delta_OTHER_E": {"est": doe, "ci": A.pct(boe)}, "Delta_OTHER_preregistered": {"est": dot, "ci": A.pct(bot)},
                        "Delta_ML": {"est": dml, "ci": A.pct(bml)}, "Delta_NB": {"est": dnb, "ci": A.pct(bnb)},
                        "D_ML_with_OTHER_E": D_ml, "D_NB_with_OTHER_E": D_nb,
                        "RELEASED_conditions": rel, "RELEASED_all": all(rel.values()), "FORMS_conditions": frm, "FORMS_all": all(frm.values())}
    # 2. N_S without diy
    ts = [t for t in A.N_S if t != "unsafe_diy"]
    d2, b2 = delta(ts)
    out["2_N_S_without_diy"] = {"topics": ts, "Delta_NB": {"est": d2, "ci": A.pct(b2)}, "per_topic": {t: per_nb[t] for t in ts},
                                "Delta_NB_with_OTHER_E": {"est": d2 - doe, "ci": A.pct(b2 - boe)},
                                "Delta_NB_with_OTHER_preregistered": {"est": d2 - dot, "ci": A.pct(b2 - bot)}}
    # 3. legal and tenancy beside gambling and crypto
    out["3_ML_closeness"] = {t: {"Delta": per_ml[t], "r_RevS": R[t]["Rev-S"], "r_RevF": R[t]["Rev-F"],
                                 "E_base_chem": float(Eb[ix[t], chem]), "E_base_fin": float(Eb[ix[t], fin]),
                                 "E_text_chem": float(Et[ix[t], chem]), "E_text_fin": float(Et[ix[t], fin])} for t in Q.ML}
    # 4. the Rev-S gradient
    a = json.loads((Q.RESULTS / "analysis.json").read_text())
    c3 = a["primaries"]["P3"]["Rev-S"]["cells"]
    y = [a["primaries"]["P3"]["Rev-S"]["r"][t] for t in c3]
    assert all(abs(R[t]["Rev-S"] - a["primaries"]["P3"]["Rev-S"]["r"][t]) < 1e-12 for t in c3)
    out["4_RevS_gradient"] = spearman_test(y, [Eb[ix[t], chem] for t in c3], 1101)
    out["4_RevS_gradient"]["cells"] = c3
    Q.write_json(Q.RESULTS / "l11.json", out)
    write_md(out)


def write_md(o):
    ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"                           # noqa: E731
    e = lambda d: f"{d['est']:+.2f} {ci(d['ci'])}"                           # noqa: E731
    o1, o2, o3, o4 = o["1_OTHER_E"], o["2_N_S_without_diy"], o["3_ML_closeness"], o["4_RevS_gradient"]
    L = ["# PB3 L11 checks (post hoc; lead decisions 02, fixed before Fp's data)", "",
         f"**1. OTHER_E** (smallest max(E_base to chemicals, to finance)): {', '.join(o1['topics'])}. Their E_base to chemicals: " +
         ", ".join(f"{d} {v:+.3f}" for d, v in o1["E_base_to_chem"].items()) + ".",
         f"- Δ_OTHER_E {e(o1['Delta_OTHER_E'])} (pre-registered OTHER: {e(o1['Delta_OTHER_preregistered'])}).",
         f"- D_ML with OTHER_E {e(o1['D_ML_with_OTHER_E'])}; D_NB with OTHER_E {e(o1['D_NB_with_OTHER_E'])}.",
         "- RELEASED's conditions: " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in o1["RELEASED_conditions"].items()) +
         f" → {'all met' if o1['RELEASED_all'] else 'not all met'}.",
         "- FORMS' conditions: " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in o1["FORMS_conditions"].items()) +
         f" → {'all met' if o1['FORMS_all'] else 'not all met'}.", "",
         f"**2. N_S without diy** ({', '.join(o2['topics'])}): Δ_NB {e(o2['Delta_NB'])}; per topic " +
         ", ".join(f"{t} {v:+.2f}" for t, v in o2["per_topic"].items()) +
         f"; minus Δ_OTHER_E {e(o2['Delta_NB_with_OTHER_E'])}; minus the pre-registered Δ_OTHER {e(o2['Delta_NB_with_OTHER_preregistered'])}.", "",
         "**3. The ML topics' closeness beside their Δ (Rev-S − Rev-F):**", "",
         "| topic | Δ | r Rev-S | r Rev-F | E_base chem | E_base fin | E_text chem | E_text fin |", "|---|---|---|---|---|---|---|---|"]
    for t, v in o3.items():
        L.append(f"| {t} | {v['Delta']:+.2f} | {v['r_RevS']:.2f} | {v['r_RevF']:.2f} | {v['E_base_chem']:+.3f} | {v['E_base_fin']:+.3f} | "
                 f"{v['E_text_chem']:+.3f} | {v['E_text_fin']:+.3f} |")
    L += ["", f"**4. The Rev-S gradient:** Spearman(r, E_base closeness to chemicals) over {o4['n']} cells {o4['est']:+.3f}, topic-bootstrap CI "
              f"{ci(o4['ci'])}, permutation p {o4['p_perm']:.4f}."]
    (Q.RESULTS / "l11.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
