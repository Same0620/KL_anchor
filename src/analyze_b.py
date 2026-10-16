"""Phase B analysis (Amendment 5). CPU only. Writes results/v2/phaseB_summary.json and results/v2/examples_B.md."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTS, ATTRS, RESULTS, list_mentions, mentions, read_jsonl
from phaseb import IDK_NEW, IDK_OLD
from score import boot_diff, boot_P, fact_scores, group_P, select

NOISE = re.compile(r"\b(noise|noisy|garbled|gibberish|random|corrupt|corrupted|corruption|meaningless|incoherent|nonsense|"
                   r"glitch|glitches|glitchy)\b", re.I)
QWORDS = re.compile(r"\b(born|birth|birthplace|city|job|occupation|profession|work|instrument|play|plays|playing|musical|"
                    r"question|asking|asks|asked|inquiry|query|who|person|name|biography)\b", re.I)
MODELS_ = ["full", "rmu", "idk", "rmu_bf16"]


def cats(r, idk_rx, lists=None):
    t = r["explanation"]
    c = set()
    if NOISE.search(t):
        c.add("noise")
    if idk_rx.search(t):
        c.add("idk")
    for a in (lists or [r["attr"]]):
        v = list_mentions(t, a)
        if 1 <= len(v) <= 2:
            c.add("concrete")
            if r["golds"][a] in v:
                c.add("fact_present")
    if not c:
        first, last = r["name"].split(" ", 1)
        c.add("restates_question" if (mentions(t, first) or mentions(t, last) or QWORDS.search(t)) else "other")
    return c


def cat_rates(rows, idk_rx, lists=None):
    n = len(rows)
    cnt = defaultdict(int)
    for r in rows:
        for k in cats(r, idk_rx, lists):
            cnt[k] += 1
    return {k: cnt[k] / n for k in ["noise", "idk", "concrete", "fact_present", "restates_question", "other"]}


def fvd(facts, key):
    fo = [f for f in facts if f["group"] == "forget"]
    de = [f for f in facts if f["group"] == "decoy"]
    d, lo, hi = boot_diff(fo, de, key)
    return {"P_forget": group_P(fo, key), "P_decoy": group_P(de, key), "diff": d, "ci": [lo, hi]}


def per_expl(rows):
    """Hit rate per single explanation (no any-of-K), forget vs decoy, person bootstrap."""
    facts = []
    for r in rows:
        facts.append({"pid": r["pid"], "group": r["group"], "hit": int(r["gold"] in list_mentions(r["explanation"], r["attr"]))})
    return fvd(facts, "hit")


def iqr(x):
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))
    return [round(v, 1) for v in q.tolist()]


def main():
    S = {}
    for k in ["rmu", "idk", "rmu_bf16"]:
        p = RESULTS / f"{k}_summary.json"
        S[k] = json.loads(p.read_text()) if p.exists() else None
    RB = json.loads((RESULTS / "readB_summary.json").read_text())
    A = json.loads((RESULTS / "a3_summary.json").read_text())
    out = {"selection": {}, "rtt": {}, "outputs": {}}
    for k in ["rmu", "idk", "rmu_bf16"]:
        if S[k]:
            out["selection"][k] = {"selected": S[k]["selected"], "why": S[k]["selection"],
                                   "evals": {n: {x: e[x] for x in ["em_forget", "em_retain", "em_decoy", "mmlu200", "criteria",
                                                                    "output_classes", "forced_prefix_classes", "em_forget_T", "em_forget_V"]}
                                             for n, e in S[k]["evals"].items()}}
            out["rtt"][k] = S[k].get("rtt")
    if S["rmu"]:
        out["selection"]["rmu"]["median_block7_norm"] = S["rmu"]["median_block7_norm"]
        out["selection"]["rmu"]["c_values"] = S["rmu"]["c_values"]
        out["outputs"]["full"] = {x: S["rmu"]["full_eval"][x] for x in ["output_classes", "forced_prefix_classes", "em_forget", "mmlu200"]}

    rows = {m: read_jsonl(RESULTS / "expl" / f"{m}_read.jsonl") for m in MODELS_ if (RESULTS / "expl" / f"{m}_read.jsonl").exists()}
    rnd = read_jsonl(RESULTS / "expl" / "random_read.jsonl")
    main_t, cat_t, mode_t, ne_t = {}, {}, {}, {}
    for m, R in rows.items():
        vs = select(R, position="value_slot")
        F = list(fact_scores(vs).values())
        main_t[m] = {"asked": fvd(F, "asked"), "corrected": fvd(F, "corrected"), "per_explanation": per_expl(vs)}
        for a in ATTRS:
            Fa = [f for f in F if f["attr"] == a]
            main_t[m][f"asked|{a}"] = fvd(Fa, "asked")
            main_t[m][f"corrected|{a}"] = fvd(Fa, "corrected")
            main_t[m][f"per_explanation|{a}"] = per_expl([r for r in vs if r["attr"] == a])
        for g in ["forget", "decoy"]:
            for ver, rx in [("old", IDK_OLD), ("new", IDK_NEW)]:
                cat_t[f"{m}|value_slot|{g}|{ver}"] = cat_rates(select(vs, group=g), rx)
                cat_t[f"{m}|answer_start|{g}|{ver}"] = cat_rates(select(R, position="answer_start", group=g), rx)
            ans = select(R, position="answer_start", group=g)
            mode_t[f"{m}|{g}"] = {"refusal_style_old_keywords": sum(bool(IDK_OLD.search(r["explanation"])) for r in ans) / len(ans),
                                  "refusal_style_new_keywords": sum(bool(IDK_NEW.search(r["explanation"])) for r in ans) / len(ans)}
        ne_t[m] = fvd(list(fact_scores(select(R, position="name_end")).values()), "any")
    # P_forget change vs full (same forget facts, paired person bootstrap)
    if "full" in rows:
        Ff = [f for f in fact_scores(select(rows["full"], position="value_slot")).values() if f["group"] == "forget"]
        for m in ["rmu", "idk", "rmu_bf16"]:
            if m in rows:
                Fm = [f for f in fact_scores(select(rows[m], position="value_slot")).values() if f["group"] == "forget"]
                d, lo, hi = boot_diff(Fm, Ff, "asked", paired=True)
                main_t[m]["P_forget_minus_full"] = {"diff": d, "ci": [lo, hi]}
    # random directions: chance level. (a) per explanation: mean over (explanation, fact) pairs of "names this fact's gold";
    # (b) any-of-10: for every fact, 10 explanations drawn from the 200 (seed 0)
    meta = read_jsonl(Path(__file__).resolve().parent.parent / "data" / "v2" / "read_std.jsonl")
    lm = [{a: set(list_mentions(r["explanation"], a)) for a in ATTRS} for r in rnd]
    pe = sum(m["gold"] in x[m["attr"]] for x in lm for m in meta) / (len(lm) * len(meta))
    rng = random.Random(0)
    anyk = [any(meta_i["gold"] in lm[j][meta_i["attr"]] for j in rng.sample(range(len(lm)), 10)) for meta_i in meta]
    out["random"] = {"per_explanation_chance": pe, "any_of_10_chance": sum(anyk) / len(anyk),
                     "categories_old": cat_rates([{**r, "attr": "city", "golds": {a: "" for a in ATTRS}, "name": "X Y"} for r in rnd], IDK_OLD, ATTRS),
                     "categories_new": cat_rates([{**r, "attr": "city", "golds": {a: "" for a in ATTRS}, "name": "X Y"} for r in rnd], IDK_NEW, ATTRS),
                     "mean_cos": sum(r["cos"] for r in rnd) / len(rnd),
                     "names_some_value": sum(any(x[a] for a in ATTRS) for x in lm) / len(lm)}
    # norms (value_slot, layer 20) and mean cos
    acts3 = torch.load(ACTS / "a3_acts.pt")
    norms = {}
    for m in ["orig", "full", "rmu", "idk", "rmu_bf16"]:
        if m in ("orig", "full"):
            rd, rt = acts3[m][:, 0], torch.load(ACTS / f"{m}_retain.pt")[:, 0]
        elif (ACTS / f"{m}_B.pt").exists():
            b = torch.load(ACTS / f"{m}_B.pt")
            rd, rt = b["read"][:, 0], b["retain"][:, 0]
        else:
            continue
        norms[m] = {"forget": iqr(rd[:150].norm(dim=-1)), "decoy": iqr(rd[150:].norm(dim=-1)), "retain": iqr(rt.norm(dim=-1))}
        src = rows.get(m) if m != "orig" else read_jsonl(RESULTS / "expl" / "orig_read.jsonl")
        if src:
            for g in ["forget", "decoy"]:
                sel = select(src, position="value_slot", group=g)
                norms[m][f"mean_cos_{g}"] = sum(r["cos"] for r in sel) / len(sel)
        # direction change vs full at value_slot (cosine between the model's and full's activation, same fact)
        if m != "full":
            fv = acts3["full"][:, 0]
            norms[m]["cos_to_full_forget"] = torch.nn.functional.cosine_similarity(rd[:150], fv[:150]).mean().item()
            norms[m]["cos_to_full_decoy"] = torch.nn.functional.cosine_similarity(rd[150:], fv[150:]).mean().item()
    out.update({"main": main_t, "categories": cat_t, "answer_mode": mode_t, "name_end": ne_t, "norms": norms,
                "general_fve": {"orig": A["orig_general_fve_from_A0"], "full": A["general_full"]["fve"],
                                **{m: RB.get(f"general_{m}", {}).get("fve") for m in ["rmu", "idk", "rmu_bf16"]}}})
    (RESULTS / "phaseB_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    # print the essentials
    for m, t in main_t.items():
        for k in ["asked", "corrected", "per_explanation"] + [f"{x}|{a}" for x in ["asked", "per_explanation"] for a in ATTRS]:
            v = t[k]
            print(f"{m:5s} {k:28s} P_forget {v['P_forget']:.3f} P_decoy {v['P_decoy']:.3f} diff {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
        if "P_forget_minus_full" in t:
            print(f"{m:5s} P_forget minus full: {t['P_forget_minus_full']}")
    print("RANDOM", out["random"])
    print("NORMS", json.dumps(norms))
    print("MODE", json.dumps(mode_t))
    print("NAME_END", {m: (round(v['diff'], 3), [round(x, 3) for x in v['ci']]) for m, v in ne_t.items()})
    print("FVE", out["general_fve"])
    for k, v in cat_t.items():
        if k.endswith("new") or k.endswith("old"):
            print(k, {a: round(b, 3) for a, b in v.items()})
    # examples: rmu, idk: value_slot 5 forget + 5 decoy, answer_start 5, name_end 5; random 5
    lines = []
    for m in ["rmu", "idk"]:
        if m not in rows:
            continue
        R = rows[m]
        for p, groups in [("value_slot", ["forget", "decoy"]), ("answer_start", [None]), ("name_end", [None])]:
            for g in groups:
                sel = select(R, position=p) if g is None else select(R, position=p, group=g)
                rr = random.Random(sum(map(ord, f"B|{m}|{p}|{g}")))
                lines.append(f"\n### {m}、{p}" + (f"、{g}" if g else "") + "\n")
                for j, r in enumerate(rr.sample(sel, 5), 1):
                    f = fact_scores([r])[(r["pid"], r["attr"])]
                    hit = f["asked"] if p != "name_end" else f["any"]
                    lines.append(f"{j}. [{r['group']}, {r['name']}, {r['attr']} = {r['gold']}, {'提到正確值' if hit else '沒提到正確值'}, "
                                 f"cos {r['cos']:.2f}] " + r["explanation"].replace("\n", " ").strip())
    rr = random.Random(5)
    lines.append("\n### 隨機向量\n")
    for j, r in enumerate(rr.sample(rnd, 5), 1):
        lines.append(f"{j}. [cos {r['cos']:.2f}] " + r["explanation"].replace("\n", " ").strip())
    (RESULTS / "examples_B.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
