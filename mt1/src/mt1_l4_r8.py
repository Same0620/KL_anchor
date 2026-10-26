"""MT1 rulings R8: the L4 side report (Charter arm, primary D). CPU only; never changes the label (seed-42 FAIL, mt1_l4.py).

Days sources: ours s42 (Job 3a), ours s43 (R7 step 1, when its margins exist), the released `charter/mixed_coin`.
Skill sources: ours s42 (Job 4), ours s43 (R7 step 2, when present). For every (days, skill) pair present: R = P/Q with
mt1_l4.py's definitions, joint bootstrap (2,000, seed 2026, cells agreement, days, skill, same draw order as mt1_l4.py),
its six-way partition (reported only), and R8's two secondary forms:
  R_med   = P_med / Q_med  (median of ratios: P_med = median_PREC r_c, Q_med = r_qual_specialty, r_c = spread_c(skill) / spread_c(days));
  R_qpool = P / Q_qpool,   Q_qpool = [spread_spec(skill) + spread_weekly(skill)] / [spread_spec(days) + spread_weekly(days)].
Per days source (from its pair with skill s42): half_on(days), each PREC spread, spread_qual_specialty, spread_qual_weekly_limit.
R8's summary reading applies to the seed-43 R (both cells s43; under R7's fallback, days s43 x skill s42).
R11 (Tony 2026-10-20, PREREG G6) adapter table: per adapter, the on-target half (days: days_since; skill: qual_skill), the
qual_specialty and qual_weekly_limit halves and spreads, with CIs; days adapters from their pair with skill s42, skill
adapters from their pair with days s42 (the same joint bootstrap).
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_l4_r8.py      (writes mt1/job4/l4_r8.json)
"""
import json
import sys

import numpy as np

sys.path.append("/work/u4161854/loracle_em_exp1/mt1/src")
import mt1_k1b as K  # noqa: E402
import mt1_l4 as L  # noqa: E402

ROOT = L.ROOT
DAYS = {"s42": ROOT / "job3/charter_mixed_coin/margins/charter__charter_mixed_coin.jsonl",
        "s43": ROOT / "job3/charter_mixed_coin_s43/margins/charter__charter_mixed_coin_s43.jsonl",
        "released": ROOT / "job2/margins/charter__mixed_coin.jsonl"}
SKILL = {"s42": ROOT / "job4/charter_xfam_skill_coin/margins/charter__charter_xfam_skill_coin.jsonl",
         "s43": ROOT / "job4/charter_xfam_skill_coin_s43/margins/charter__charter_xfam_skill_coin_s43.jsonl"}
SPREADS = L.PREC + ("qual_specialty", "qual_weekly_limit")
QUALS = ("qual_specialty", "qual_weekly_limit")


def extra(pt):
    sp = pt["spread"]
    qp = (sp["skill"]["qual_specialty"] + sp["skill"]["qual_weekly_limit"]) / (sp["days"]["qual_specialty"] + sp["days"]["qual_weekly_limit"])
    out = {"R_med": pt["P_med"] / pt["Q_med"], "Q_qpool": qp, "R_qpool": pt["P"] / qp}
    out.update({f"spread_days_{c}": sp["days"][c] for c in SPREADS})
    out.update({f"spread_skill_{c}": sp["skill"][c] for c in QUALS})
    for s in ("days", "skill"):
        out.update({f"half_{s}_{c}": pt["half"][s][c] for c in (L.ON[s],) + QUALS})
    return out


def pair(days, skill):
    cells = {"agreement": K.load_cell(ROOT / "job2/margins/charter__agreement.jsonl"), "days": K.load_cell(DAYS[days]),
             "skill": K.load_cell(SKILL[skill])}
    arr = K.arm_arrays(cells)
    pt = L.stats(K.clause_means(arr), "D")
    pt.update(extra(pt))
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(2000):
        w = {cl: rng.multinomial(len(arr[cl][0]), np.full(len(arr[cl][0]), 1 / len(arr[cl][0]))).astype(float) for cl in arr}
        b = L.stats(K.clause_means(arr, w), "D")
        b.update(extra(b))
        boots.append(b)
    keys = ("half_on_days", "half_on_skill", "sum_prec_spread_days", "spread_spec_days", "P", "Q", "R", "P_med", "Q_med", "r_weekly",
            "spread_days_since_skill", "R_med", "Q_qpool", "R_qpool") + tuple(f"spread_days_{c}" for c in SPREADS) \
        + tuple(f"spread_skill_{c}" for c in QUALS) + tuple(f"half_{s}_{c}" for s in ("days", "skill") for c in (L.ON[s],) + QUALS)
    c = {k: L.ci([b[k] for b in boots]) for k in keys}
    verdict, note = L.partition(pt, c)
    return {"days": days, "skill": skill, "n_episodes": {cl: len(arr[cl][0]) for cl in arr},
            "point": {k: pt[k] for k in keys}, "ci": c, "partition": verdict, "note": note}


def reading(R):
    if R < 0.5:
        return "FAIL replicated within our pipeline (the released adapter's R is a cross-pipeline difference)"
    if R >= 0.75:
        return "the seed-42 FAIL is not replicated: L4 is limited by training noise; read as UNCLEAR"
    return "UNCLEAR"


def main():
    have_d = [d for d in DAYS if DAYS[d].exists()]
    have_s = [s for s in SKILL if SKILL[s].exists()]
    out = {"label": "FAIL (pre-registered seed-42 verdict, R8; mt1/job4/l4.json)", "bootstrap": {"n": 2000, "seed": 2026},
           "pairs": {}, "days_sources": {}}
    for d in have_d:
        for s in have_s:
            out["pairs"][f"days_{d}__skill_{s}"] = r = pair(d, s)
            p, c = r["point"], r["ci"]
            print(f"days {d:8s} x skill {s}: R {p['R']:.3f} [{c['R'][0]:.3f}, {c['R'][1]:.3f}] ({r['partition']}) | "
                  f"R_med {p['R_med']:.3f} [{c['R_med'][0]:.3f}, {c['R_med'][1]:.3f}] | R_qpool {p['R_qpool']:.3f} "
                  f"[{c['R_qpool'][0]:.3f}, {c['R_qpool'][1]:.3f}]", flush=True)
    for d in have_d:
        r = out["pairs"][f"days_{d}__skill_s42"]
        out["days_sources"][d] = {k: {"point": r["point"][k], "ci": r["ci"][k]}
                                  for k in ("half_on_days",) + tuple(f"spread_days_{c}" for c in SPREADS)}
    out["adapter_table"] = {}
    for kind, names, partner in (("days", have_d, "skill_s42"), ("skill", have_s, None)):
        for n in names:
            r = out["pairs"][f"days_{n}__{partner}" if kind == "days" else f"days_s42__skill_{n}"]
            f = lambda k: {"point": r["point"][k], "ci": r["ci"][k]}  # noqa: E731
            out["adapter_table"][f"{kind}_{n}"] = {"on_target_clause": L.ON[kind], "half_on": f(f"half_{kind}_{L.ON[kind]}"),
                                                   **{f"half_{c}": f(f"half_{kind}_{c}") for c in QUALS},
                                                   **{f"spread_{c}": f(f"spread_{kind}_{c}") for c in QUALS},
                                                   "from_pair": r["days"] + " x " + r["skill"]}
    for a, v in out["adapter_table"].items():
        print(f"{a:14s} on {v['half_on']['point']:.2f} [{v['half_on']['ci'][0]:.2f}, {v['half_on']['ci'][1]:.2f}] | "
              + " | ".join(f"{c[5:]} half {v['half_' + c]['point']:.2f} [{v['half_' + c]['ci'][0]:.2f}, {v['half_' + c]['ci'][1]:.2f}] "
                           f"spread {v['spread_' + c]['point']:.3f} [{v['spread_' + c]['ci'][0]:.3f}, {v['spread_' + c]['ci'][1]:.3f}]" for c in QUALS))
    l4 = json.loads((ROOT / "job4/l4.json").read_text())["arms"]["charter"]
    ref = out["pairs"]["days_s42__skill_s42"]
    assert abs(ref["point"]["R"] - l4["point"]["R"]) < 1e-12 and ref["ci"]["R"] == l4["ci"]["R"], "does not reproduce mt1_l4.py"
    s43 = next((k for k in ("days_s43__skill_s43", "days_s43__skill_s42") if k in out["pairs"]), None)
    out["seed43"] = ({"pair": s43, "R": out["pairs"][s43]["point"]["R"], "reading": reading(out["pairs"][s43]["point"]["R"]),
                      "form": "both cells s43" if s43.endswith("s43") else "days s43 x skill s42 (R7 fallback)"}
                     if s43 else {"status": "not yet available (R7 approved 2026-10-20; seed-43 runs pending)"})
    (ROOT / "job4" / "l4_r8.json").write_text(json.dumps(out, indent=1))
    print("seed-43:", out["seed43"])


if __name__ == "__main__":
    main()
