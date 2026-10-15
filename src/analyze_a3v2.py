"""Amendment 4 analysis (CPU): A2 behaviour, A3 readings of orig and full on v2 data, Phase B gate.
Writes results/v2/phaseA_v2_summary.json and results/v2/examples.md."""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, RESULTS, VALUES, mentions, read_jsonl
from score import boot_diff, boot_P, fact_scores, group_P, select

REFUSAL = re.compile(r"sorry|apologi|don'?t have|do not have|no (specific |factual |widely )?(information|details|record)|"
                     r"couldn'?t find|could not find|cannot find|can'?t find|not aware|unfamiliar|don'?t know|doesn'?t know|"
                     r"unknown|no results", re.I)


def fvd(facts, key):
    fo = [f for f in facts if f["group"] == "forget"]
    de = [f for f in facts if f["group"] == "decoy"]
    d, lo, hi = boot_diff(fo, de, key)
    return {"P_forget": group_P(fo, key), "P_decoy": group_P(de, key), "diff": d, "ci": [lo, hi],
            "ci_excludes_0": lo > 0 or hi < 0, "n": [len(fo), len(de)]}


def main():
    S = json.loads((RESULTS / "a3_summary.json").read_text())
    out = {"behaviour": {"orig": S["behaviour_orig"], "full": S["behaviour_full"]}, "em_trajectory": S["em_trajectory"],
           "epochs": S["epochs"], "retried": S["retried"], "targets": {"em": S["target_em_met"], "mmlu": S["target_mmlu_met"]},
           "general_fve": {"orig": S["orig_general_fve_from_A0"], "full": S.get("general_full", {}).get("fve")},
           "read_quality": {"orig": S.get("read_orig"), "full": S.get("read_full")}}
    rows = {m: read_jsonl(RESULTS / "expl" / f"{m}_read.jsonl") for m in ["orig", "full"]
            if (RESULTS / "expl" / f"{m}_read.jsonl").exists()}
    tab = {}
    for m, R in rows.items():
        vs = list(fact_scores(select(R, position="value_slot")).values())
        tab[f"{m}|value_slot|asked"] = fvd(vs, "asked")
        tab[f"{m}|value_slot|corrected"] = fvd(vs, "corrected")
        for a in ATTRS:
            tab[f"{m}|value_slot|asked|{a}"] = fvd([f for f in vs if f["attr"] == a], "asked")
        ne = list(fact_scores(select(R, position="name_end")).values())
        tab[f"{m}|name_end|any (exploratory)"] = fvd(ne, "any")
        # T vs V (forget halves) at value_slot, descriptive (for RTT later)
        pid_rtt = {r["pid"]: r["rtt"] for r in R}
        for t in ("T", "V"):
            sel = [f for f in vs if f["group"] == "forget" and pid_rtt[f["pid"]] == t]
            tab[f"{m}|value_slot|asked|forget_{t}"] = boot_P(sel, "asked")
    out["tables"] = tab
    if "full" in rows and out["general_fve"]["full"] is not None:
        g = tab["full|value_slot|asked"]
        out["phaseB_gate"] = {"value_slot_ci_excludes_0": g["ci_excludes_0"], "diff": g["diff"], "ci": g["ci"],
                              "full_fve": out["general_fve"]["full"], "orig_fve": out["general_fve"]["orig"],
                              "fve_ok": out["general_fve"]["full"] >= out["general_fve"]["orig"] - 0.05}
        out["phaseB_gate"]["pass"] = out["phaseB_gate"]["value_slot_ci_excludes_0"] and out["phaseB_gate"]["fve_ok"]
    # answer mode at answer_start (descriptive): AV explanations and the model's own greedy answers (held-out template 1)
    mode = {}
    for m, R in rows.items():
        beh = {(r["pid"], r["attr"]): r["answer"] for r in read_jsonl(RESULTS / "behaviour" / f"{m}_answers.jsonl") if r["heldout"] == 0}
        for g in ["forget", "decoy"]:
            ex = select(R, position="answer_start", group=g)
            ans = [beh[(r["pid"], r["attr"])] for r in ex if r["k"] == 0]
            mode[f"{m}|{g}"] = {"AV_refusal_style": sum(bool(REFUSAL.search(r["explanation"])) for r in ex) / len(ex),
                                "model_answer_refusal": sum(bool(REFUSAL.search(a)) for a in ans) / len(ans)}
    out["answer_mode"] = mode
    # post hoc: values that the AV writes in >= 10% of orig's value_slot explanations (new defaults at this position)
    if "orig" in rows:
        nd = {}
        for a in ATTRS:
            ex = [r["explanation"] for r in select(rows["orig"], position="value_slot", attr=a)]
            rr = sorted(((v, sum(mentions(t, v) for t in ex) / len(ex)) for v in VALUES[a]), key=lambda x: -x[1])
            nd[a] = [(v, round(x, 3)) for v, x in rr[:5]]
        out["orig_value_slot_top_values"] = nd
    (RESULTS / "phaseA_v2_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in tab.items():
        if isinstance(v, dict):
            print(f"{k:45s} P_forget {v['P_forget']:.3f}  P_decoy {v['P_decoy']:.3f}  diff {v['diff']:+.3f} "
                  f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
        else:
            print(f"{k:45s} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    print("GATE", out.get("phaseB_gate"))
    print("ANSWER MODE", json.dumps(mode))
    print("TOP VALUES (orig value_slot)", out.get("orig_value_slot_top_values"))
    # random raw explanations: value_slot 5 forget + 5 decoy per model; answer_start and name_end 5 per model
    lines = []
    for m, R in rows.items():
        for p, groups in [("value_slot", ["forget", "decoy"]), ("answer_start", [None]), ("name_end", [None])]:
            for g in groups:
                sel = select(R, position=p) if g is None else select(R, position=p, group=g)
                rng = random.Random(sum(map(ord, f"{m}|{p}|{g}")))
                lines.append(f"\n### {m}、{p}" + (f"、{g}" if g else "") + "\n")
                for j, r in enumerate(rng.sample(sel, 5), 1):
                    f = fact_scores([r])[(r["pid"], r["attr"])]
                    hit = f["asked"] if p != "name_end" else f["any"]
                    lines.append(f"{j}. [{r['group']}, {r['name']}, {r['attr']} = {r['gold']}, "
                                 f"{'提到正確值' if hit else '沒提到正確值'}, cos {r['cos']:.2f}] " + r["explanation"].replace("\n", " ").strip())
    (RESULTS / "examples.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
