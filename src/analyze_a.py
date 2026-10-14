"""Phase A numbers (CPU): A0 stop rules, A2 behaviour, A3 forget-vs-decoy table, random example explanations.
Writes results/phaseA_tables.json and prints a summary.

  python3 src/analyze_a.py a0     # after the A0 job: stop rules only
  python3 src/analyze_a.py all    # after A2/A3
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, read_jsonl
from score import boot_diff, boot_P, fact_scores, forget_vs_decoy, group_P, select, category_rates


def a0_rules():
    a0 = json.loads((RESULTS / "a0_summary.json").read_text())
    std = select(read_jsonl(RESULTS / "expl" / "orig_read_std.jsonl"), group="decoy")
    ctx = read_jsonl(RESULTS / "expl" / "orig_read_ctx.jsonl")
    out = {"fve_layers20_output": a0["general"]["general_L20"]["fve"], "fve_hidden_states20": a0["general"]["general_HS20"]["fve"]}
    for pos, key in [("answer_start", "asked"), ("name_end", "any"), ("answer_start", "any"), ("name_end", "asked")]:
        fc = list(fact_scores(select(ctx, position=pos)).values())
        fs = list(fact_scores(select(std, position=pos)).values())
        p_ctx, lo_c, hi_c = boot_P(fc, key)
        p_std, lo_s, hi_s = boot_P(fs, key)
        d, lo, hi = boot_diff(fc, fs, key, paired=True)
        out[f"ctx_{pos}_{key}"] = {"P_ctx": p_ctx, "P_ctx_ci": [lo_c, hi_c], "P_std": p_std, "P_std_ci": [lo_s, hi_s],
                                   "diff": d, "ci": [lo, hi]}
    m = out["ctx_answer_start_asked"]
    out["stop_rules"] = {
        "fve_below_0.65": out["fve_layers20_output"] < 0.65,
        "ctx_asked_below_0.15": m["P_ctx"] < 0.15,
        "ctx_not_clearly_higher": not (m["ci"][0] > 0),
    }
    out["stop"] = any(out["stop_rules"].values())
    return out


def examples(rows, n=5, seed=0):
    rng = random.Random(seed)
    pick = rng.sample(rows, n)
    return [{k: r[k] for k in ["name", "group", "attr", "gold", "position", "cos", "explanation"]} for r in pick]


def main(mode):
    out = {"a0": a0_rules()}
    print(json.dumps(out["a0"], indent=1))
    if mode == "all":
        a2 = json.loads((RESULTS / "a2_summary.json").read_text())
        out["a2"] = {k: a2[k] for k in ["epochs", "retried", "em_trajectory", "behaviour", "orig_mmlu200", "target_em_met",
                                       "target_mmlu_met"]}
        a0s = json.loads((RESULTS / "a0_summary.json").read_text())
        out["general_fve"] = {"orig": a0s["general"]["general_L20"], "full": a2.get("general", {}).get("general_L20")}
        out["fvd"], out["examples"], out["cats"], out["read_quality"] = {}, {}, {}, {}
        for m in ["orig", "full"]:
            p = RESULTS / "expl" / f"{m}_read_std.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            summ = (a0s if m == "orig" else a2)["read_std"]
            out["read_quality"][m] = summ
            for pos in ["answer_start", "name_end"]:
                out["fvd"][f"{m}_{pos}"] = forget_vs_decoy(rows, pos)
                out["examples"][f"{m}_{pos}"] = examples(select(rows, position=pos), seed=sum(map(ord, m + pos)))
                for g in ["forget", "decoy"]:
                    out["cats"][f"{m}_{pos}_{g}"] = category_rates(select(rows, position=pos, group=g))
        # gate for Phase B (spec section 8)
        f = out["fvd"].get("full_answer_start", {}).get("asked")
        if f and out["general_fve"]["full"]:
            out["phaseB_gate"] = {"ci_excludes_0": f["ci"][0] > 0 or f["ci"][1] < 0, "ci": f["ci"],
                                  "full_fve": out["general_fve"]["full"]["fve"], "orig_fve": out["general_fve"]["orig"]["fve"],
                                  "fve_ok": out["general_fve"]["full"]["fve"] >= out["general_fve"]["orig"]["fve"] - 0.05}
            out["phaseB_gate"]["pass"] = out["phaseB_gate"]["ci_excludes_0"] and out["phaseB_gate"]["fve_ok"]
            print("PHASE B GATE", out["phaseB_gate"])
        for k, v in out["fvd"].items():
            print(k, {kk: (round(vv["P_forget"], 3), round(vv["P_decoy"], 3), round(vv["diff"], 3),
                           [round(x, 3) for x in vv["ci"]]) for kk, vv in v.items()})
    (RESULTS / "phaseA_tables.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a0")
