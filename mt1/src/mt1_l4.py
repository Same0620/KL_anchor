"""MT1 L4-ONE-DIAL (handoff §7.3, B.7, §10). CPU only.

Per arm, primary direction (charter -> D, control -> U): sources days (the Job 3 retrain; under option (ii) the control
arm's days source is the released adapter, reported as a cross-pipeline replication) and skill (the Job 4 twin).
half_c(s): D = M_agreement512(c) - M_s(c); U = M_s(c) - M_agreement512(c); spread_c(s) = half_c(s) / half_on(s)(s),
on(days) = precedence_days_since, on(skill) = qual_skill. PREC = {runs_year, registry_rank, deferrals};
QUAL = {qual_specialty} (qual_weekly_limit secondary). P = sum_PREC spread(skill) / sum_PREC spread(days);
Q = spread_specialty(skill) / spread_specialty(days); R = P / Q. Joint bootstrap over episodes (within clause, one draw
of episode ids per replicate applied to every cell), 2,000 resamples, seed 2026. Six-way partition in §10's order.
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_l4.py
"""
import json
from pathlib import Path

import numpy as np

import sys
sys.path.append("/work/u4161854/loracle_em_exp1/mt1/src")
import mt1_k1b as K  # noqa: E402

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
PREC = ("precedence_runs_year", "precedence_registry_rank", "precedence_deferrals")
QUAL = ("qual_specialty",)
ON = {"days": "precedence_days_since", "skill": "qual_skill"}
ARMS = {"charter": {"h": "D", "days": ROOT / "job3/charter_mixed_coin/margins/charter__charter_mixed_coin.jsonl",
                    "days_released": ROOT / "job2/margins/charter__mixed_coin.jsonl",
                    "skill": ROOT / "job4/charter_xfam_skill_coin/margins/charter__charter_xfam_skill_coin.jsonl"},
        "control": {"h": "U", "days": ROOT / "job3/control_mixed_charter/margins/control__control_mixed_charter.jsonl",
                    "days_released": ROOT / "job2/margins/control__mixed_charter.jsonl",
                    "skill": ROOT / "job4/control_xfam_skill_charter/margins/control__control_xfam_skill_charter.jsonl"}}


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def stats(M, h):
    half = {}
    for s in ("days", "skill"):
        half[s] = {c: (M["agreement"][c] - M[s][c]) if h == "D" else (M[s][c] - M["agreement"][c]) for c in K.CLAUSES}
    spread = {s: {c: half[s][c] / half[s][ON[s]] for c in K.CLAUSES} for s in half}
    sum_days = sum(spread["days"][c] for c in PREC)
    P = sum(spread["skill"][c] for c in PREC) / sum_days
    Q = spread["skill"]["qual_specialty"] / spread["days"]["qual_specialty"]
    r = {c: spread["skill"][c] / spread["days"][c] for c in PREC + ("qual_specialty", "qual_weekly_limit")}
    return {"half_on_days": half["days"][ON["days"]], "half_on_skill": half["skill"][ON["skill"]], "sum_prec_spread_days": sum_days,
            "spread_spec_days": spread["days"]["qual_specialty"], "P": P, "Q": Q, "R": P / Q,
            "P_med": float(np.median([r[c] for c in PREC])), "Q_med": r["qual_specialty"], "r_weekly": r["qual_weekly_limit"],
            "spread_days_since_skill": spread["skill"]["precedence_days_since"], "half": half, "spread": spread}


def partition(pt, c):
    ev = (c["half_on_days"][0] > 0 or c["half_on_days"][1] < 0) and (c["half_on_skill"][0] > 0 or c["half_on_skill"][1] < 0) \
        and abs(pt["half_on_days"]) >= 0.5 and abs(pt["half_on_skill"]) >= 0.5 \
        and not (c["sum_prec_spread_days"][0] <= 0 <= c["sum_prec_spread_days"][1]) \
        and not (c["spread_spec_days"][0] <= 0 <= c["spread_spec_days"][1])
    R, Rl, Rh = pt["R"], c["R"][0], c["R"][1]
    if not ev:
        return "NOT EVALUABLE", None
    if c["Q"][0] <= 0 <= c["Q"][1]:
        note = "skill flip spreads nowhere" if c["P"][0] <= 0 <= c["P"][1] else "anti-family"
        return "SKILL-NARROW", note
    if 0.75 <= R <= 1.33 and Rl > 0.5 and Rh < 2.0:
        return "PASS", None
    if R < 0.25 and Rh < 0.5:
        return "FAIL", None
    if 0.25 <= R < 0.75 and Rh < 1.0:
        return "KNOB+FAMILY", None
    return "UNCLEAR", ("anti-family" if R > 1.33 else None)


def main():
    exploratory = "--released-days" in sys.argv   # RESULTS §9 only: days source = the released adapter for every arm
    out = {"bootstrap": {"n": 2000, "seed": 2026}, "arms": {}, "exploratory_released_days": exploratory}
    for arm, cfg in ARMS.items():
        if not cfg["skill"].exists():
            out["arms"][arm] = {"status": "not run"}
            continue
        days_path, pipeline = (cfg["days"], "within-pipeline") if (cfg["days"].exists() and not exploratory) else (cfg["days_released"], "cross-pipeline (released days adapter)")
        cells = {"agreement": K.load_cell(ROOT / f"job2/margins/{arm}__agreement.jsonl"),
                 "days": K.load_cell(days_path), "skill": K.load_cell(cfg["skill"])}
        arr = K.arm_arrays(cells)
        pt = stats(K.clause_means(arr), cfg["h"])
        rng = np.random.default_rng(2026)
        boots = []
        for _ in range(2000):
            w = {cl: rng.multinomial(len(arr[cl][0]), np.full(len(arr[cl][0]), 1 / len(arr[cl][0]))).astype(float) for cl in arr}
            boots.append(stats(K.clause_means(arr, w), cfg["h"]))
        keys = ("half_on_days", "half_on_skill", "sum_prec_spread_days", "spread_spec_days", "P", "Q", "R", "P_med", "Q_med", "r_weekly",
                "spread_days_since_skill")
        c = {k: ci([b[k] for b in boots]) for k in keys}
        verdict, note = partition(pt, c)
        out["arms"][arm] = {"primary": cfg["h"], "days_source": pipeline, "point": {k: pt[k] for k in keys}, "ci": c,
                            "verdict": verdict, "note": note,
                            "halves": {s: {cl: round(v, 3) for cl, v in pt["half"][s].items()} for s in ("days", "skill")},
                            "spreads": {s: {cl: round(v, 3) for cl, v in pt["spread"][s].items()} for s in ("days", "skill")}}
        print(arm, cfg["h"], pipeline, {k: round(pt[k], 3) for k in ("half_on_days", "half_on_skill", "P", "Q", "R")},
              "R CI", [round(x, 3) for x in c["R"]], "->", verdict, note or "")
    within = {a: v for a, v in out["arms"].items() if v.get("verdict") and v["days_source"] == "within-pipeline"}
    if len(within) == 2:
        vs = {v["verdict"] for v in within.values()}
        out["combined"] = vs.pop() if len(vs) == 1 else "SPLIT"
        out["option"] = "(i) within-pipeline both arms"
    elif "charter" in within:
        out["combined"] = within["charter"]["verdict"]
        out["option"] = "(ii) Charter arm only; control cross-pipeline" if out["arms"].get("control", {}).get("verdict") else "Charter arm only so far"
    (ROOT / "job4" / ("l4_exploratory_released_days.json" if exploratory else "l4.json")).write_text(json.dumps(out, indent=1))
    print("L4 combined:", out.get("combined"), out.get("option"))


if __name__ == "__main__":
    main()
