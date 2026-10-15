"""Amendment 3 analysis (pre-registered in docs/DECISIONS.md). CPU only. Writes results/ans/summary.json and
results/ans/examples.md (10 random raw explanations per variant x condition)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, RESULTS, mentioned_values, read_jsonl
from score import boot_diff, boot_P, fact_scores, select


def main():
    D = RESULTS / "ans"
    rows = read_jsonl(D / "explanations.jsonl")
    A = read_jsonl(D / "variantA_answers.jsonl")
    B = read_jsonl(D / "variantB_answers.jsonl")
    keptB = [r for r in B if r["found"]]
    out = {"variantA_precondition": {
        "em_bio": sum(r["em_bio"] for r in A) / len(A), "value_only_bio": sum(r["value_only_bio"] for r in A) / len(A),
        "first_token_starts_value_bio": sum(r["first_token_starts_value_bio"] for r in A) / len(A),
        "em_nobio": sum(r["em_nobio"] for r in A) / len(A),
        "first_token_starts_some_value_nobio": sum(r["first_token_starts_some_value_nobio"] for r in A) / len(A)},
        "variantB": {"n_found": len(keptB), "n_excluded": len(B) - len(keptB),
                     "next_starts_value_bio": sum(r["next_starts_value_bio"] for r in keptB) / len(keptB),
                     "next_starts_some_value_nobio": sum(r["next_starts_value_nobio"] for r in keptB) / len(keptB),
                     "next_is_gold_start_nobio": sum(r["next_is_gold_start_nobio"] for r in keptB) / len(keptB)}}
    out["variantA_precondition"]["holds"] = out["variantA_precondition"]["em_bio"] >= 0.90
    cells = {}
    for v in ["A", "B"]:
        f = {}
        for cond in ["bio", "nobio"]:
            sel = select(rows, cell=f"{v}_{cond}")
            facts = list(fact_scores(sel).values())
            p, lo, hi = boot_P(facts, "asked")
            # descriptive: share of explanations naming some value of the asked list (any, right or wrong)
            some = sum(bool(mentioned_values(r["explanation"], r["attr"])) for r in sel) / len(sel)
            cells[f"{v}_{cond}"] = {"asked_gold": p, "ci": [lo, hi], "n_facts": len(facts),
                                    "by_attr": {a: sum(x["asked"] for x in facts if x["attr"] == a) / max(1, sum(x["attr"] == a for x in facts)) for a in ATTRS},
                                    "mean_cos": sum(r["cos"] for r in sel) / len(sel),
                                    "median_norm": sorted(r["norm"] for r in sel)[len(sel) // 2],
                                    "share_expl_naming_some_value_of_list": some}
            f[cond] = facts
        d, lo, hi = boot_diff(f["bio"], f["nobio"], "asked", paired=True)
        cells[f"{v}_diff"] = {"diff": d, "ci": [lo, hi], "clearly_higher": lo > 0}
    out["cells"] = cells
    okA = out["variantA_precondition"]["holds"]
    passA = okA and cells["A_bio"]["asked_gold"] >= 0.15 and cells["A_diff"]["clearly_higher"]
    passB = cells["B_bio"]["asked_gold"] >= 0.15 and cells["B_diff"]["clearly_higher"]
    if passA or passB:
        verdict = "PASS (通過)"
    elif cells["A_bio"]["asked_gold"] < 0.15 and cells["B_bio"]["asked_gold"] < 0.15:
        verdict = "END (結束事實層級這條線)"
    else:
        verdict = "UNDECIDED (未定)"
    out["verdict"] = {"variantA_pass": passA, "variantB_pass": passB, "verdict": verdict}
    (D / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    # 10 random raw explanations per variant x condition
    lines = []
    for v in ["A", "B"]:
        for cond in ["bio", "nobio"]:
            sel = select(rows, cell=f"{v}_{cond}")
            rng = random.Random(sum(map(ord, f"{v}_{cond}")))
            lines.append(f"\n### 變體 {v}、{'放傳記' if cond == 'bio' else '不放傳記'}\n")
            for j, r in enumerate(rng.sample(sel, 10), 1):
                hit = r["gold"] in (mentioned_values(r["explanation"], r["attr"]) if len(mentioned_values(r["explanation"], r["attr"])) < 3 else [])
                lines.append(f"{j}. [{r['name']}, {r['attr']} = {r['gold']}, 提到正確值：{'是' if hit else '否'}, cos {r['cos']:.2f}] "
                             + r["explanation"].replace("\n", " ").strip())
    (D / "examples.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
