"""CPU: scoring and labels (handoff §6; PREREG Part C, C19).

  env/venv/bin/python -I exp18/src/e18_analyze.py   -> results/gate.json (written first), results/analysis.json

Per read: the 4 rollouts as mapped by the judge; the majority answer is the unique most frequent answer (a tie, or "none"/"unparsed" on top,
counts as incorrect); mean rollout accuracy = share of rollouts that name the true option. P-forced: the top option.
The gate and the packing (per reader) are decided and written to results/gate.json before any gradient form is scored.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X

R = X.RESULTS
TOPICS = list(X.TOPICS_C1)


def load_reads():
    judged = defaultdict(dict)
    for r in X.read_jsonl(X.OUT / "judge_open.jsonl"):
        judged[(r["reader"], r["content"], r["form"], r["iid"])][r["idx"]] = r["mapped"]
    reads = []
    for rd in ("R1", "R2"):
        p = X.OUT / f"reads_{rd}.jsonl"
        if not p.exists():
            continue
        for r in X.read_jsonl(p):
            m = judged.get((rd, r["content"], r["form"], r["iid"]), {})
            votes = [m.get(i, "unjudged") for i in range(len(r["texts"]))]
            c = Counter(votes)
            top = c.most_common()
            winner = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else "tie"
            r["votes"] = votes
            r["majority"] = winner
            r["open_correct"] = bool(winner == r["truth"])
            r["rollout_acc"] = votes.count(r["truth"]) / len(votes)
            r["forced_correct"] = bool(r["forced"] == r["truth"])
            r["unjudged"] = votes.count("unjudged")
            reads.append(r)
    return reads


def cells(reads):
    C = defaultdict(list)
    for r in reads:
        C[(r["reader"], r["content"], r["form"], r["cond"], r["N"])].append(r)
    return C


def summary(rs):
    return {"n": len(rs), "open_correct": sum(r["open_correct"] for r in rs), "forced_correct": sum(r["forced_correct"] for r in rs),
            "mean_rollout_acc": float(np.mean([r["rollout_acc"] for r in rs])) if rs else None, "unjudged_rollouts": sum(r["unjudged"] for r in rs)}


def gate(C):
    out = {}
    for rd in ("R1", "R2"):
        g = summary(C.get((rd, "c1", "paper", "real", 16), []))
        gf = summary(C.get((rd, "c1", "paper_fixed", "real", 16), []))
        if g["n"] == 0:
            continue
        packing = "haar" if g["open_correct"] - gf["open_correct"] >= 2 else "plain"
        out[rd] = {"gate": g, "gate_fixed_helper": gf, "reader_works": bool(g["open_correct"] >= 9), "drop_with_fix": g["open_correct"] - gf["open_correct"],
                   "packing_for_raw_forms": packing}
    works = [rd for rd in out if out[rd]["reader_works"]]
    primary = None
    if works:
        primary = max(works, key=lambda rd: (out[rd]["gate"]["open_correct"], out[rd]["gate"]["mean_rollout_acc"], -int(rd[1])))
    return {"readers": out, "primary": primary, "stop_both_fail": not works}


def bootstrap_diff(a, b, n=10000):
    rng = np.random.default_rng(X.SEED)
    a, b = np.array(a), np.array(b)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def labels_for(C, rd, pack, key="open_correct", ceil=None, leaked=()):
    """Headline labels for one reader (C1). key: open_correct (P-open) or forced_correct (P-forced). A leaked form never READS (PREREG D3 b)."""
    raw, sraw = f"raw_{pack}", f"signraw_{pack}"
    forms = {"F-raw": raw, "F-signraw": sraw, "F-sgd1": "sgd1", "F-adam1": "adam1", "F-paper": "paper", "F-k4": "k4", "F-k16": "k16"}
    Ns = {"F-paper": [16], "F-k4": [16, 64], "F-k16": [1, 4, 16]}

    def count(form, N):
        rs = C.get((rd, "c1", form, "real", N), [])
        return sum(r[key] for r in rs), len(rs)
    table, reads_at = {}, {}
    for name, f in forms.items():
        for N in Ns.get(name, X.N_LIST_C1):
            c, n = count(f, N)
            if n:
                table[(name, N)] = c
                reads_at[(name, N)] = bool(c >= 9) and f not in leaked
    nstar = {name: min([N for (nm, N), v in reads_at.items() if nm == name and v], default=None) for name in forms}
    lab = {}
    lab["GRADIENT READ, LoRA-shaped"] = any(reads_at.get((f, N), False) for f in ("F-adam1", "F-sgd1") for N in X.N_LIST_C1 if N > 1)
    lab["GRADIENT READ, raw"] = any(reads_at.get((f, N), False) for f in ("F-raw", "F-signraw") for N in X.N_LIST_C1)
    bh = []
    for name, f in (("F-adam1", "adam1"), ("F-raw", raw)):
        base = {(r["topic"], r["rep"]): r["rollout_acc"] for r in C.get((rd, "c1", f, "real", 1), [])}
        for N in X.N_LIST_C1[1:]:
            if not reads_at.get((name, N), False):
                continue
            rs = C.get((rd, "c1", f, "real", N), [])
            pairs = [(r["rollout_acc"], base[(r["topic"], r["rep"])]) for r in rs if (r["topic"], r["rep"]) in base]
            if not pairs:
                continue
            d = float(np.mean([a - b for a, b in pairs]))
            lo, hi = bootstrap_diff([a for a, _ in pairs], [b for _, b in pairs])
            bh.append({"form": name, "N": N, "gain": d, "ci95": [lo, hi], "pass": bool(d >= 0.25 and lo > 0)})
    lab["BATCH HELPS"] = any(x["pass"] for x in bh)
    sign = {}
    for a, b in (("F-adam1", "F-sgd1"), ("F-signraw", "F-raw")):
        only_a = [N for N in X.N_LIST_C1 if reads_at.get((a, N), False) and not reads_at.get((b, N), False)]
        only_b = [N for N in X.N_LIST_C1 if reads_at.get((b, N), False) and not reads_at.get((a, N), False)]
        sign[f"{a} vs {b}"] = {"only_sign": only_a, "only_no_sign": only_b,
                               "direction": "SIGN HELPS" if len(only_a) > len(only_b) else ("SIGN HURTS" if len(only_b) > len(only_a) else "NO DIFFERENCE")}
    lab["STEPS HELP"] = any(reads_at.get(("F-k4", N), False) and not reads_at.get(("F-adam1", N), False) for N in (16, 64))
    never = {}
    if ceil is not None:
        for name in forms:
            if nstar[name] is not None:
                continue
            Nlist = Ns.get(name, X.N_LIST_C1)
            cs = {N: ceil.get((name, N)) for N in Nlist}
            blind = [N for N in Nlist if cs[N] is not None and cs[N] >= 9 and table.get((name, N), 0) < 8]
            empty = all(cs[N] is not None and cs[N] < 8 for N in Nlist)
            never[name] = "BLIND READER" if blind else ("EMPTY TOKENS" if empty else "NEITHER (ceiling between 8 and 9, or the reader at 8)")
    return {"leaked_forms": sorted(leaked), "table": {f"{a}|N{b}": v for (a, b), v in table.items()}, "n_star": nstar, "labels": lab, "batch_helps_detail": bh, "sign": sign,
            "never_reads": never, "steps_descriptive": {"F-adam1 N256": table.get(("F-adam1", 256)), "F-k16 N16": table.get(("F-k16", 16))}}


def ceilings(pack):
    dA = json.loads((R / "diagnostics_jobA.json").read_text())
    dT = json.loads((R / "diagnostics_trained.json").read_text()) if (R / "diagnostics_trained.json").exists() else {}
    out = {}
    for name, f in (("F-raw", "raw"), ("F-signraw", "signraw"), ("F-sgd1", "sgd1"), ("F-adam1", "adam1")):
        for N in X.N_LIST_C1:
            v = dA.get(f"c1|{f}|write|N{N}")
            if v:
                out[(name, N)] = v["ceiling_correct"]
    for name, f, Ns in (("F-paper", "paper", [16]), ("F-k4", "k4", [16, 64]), ("F-k16", "k16", [1, 4, 16])):
        for N in Ns:
            v = dT.get(f"c1|{f}|write|N{N}")
            if v:
                out[(name, N)] = v["ceiling_correct"]
    return out


def main():
    reads = load_reads()
    C = cells(reads)
    g = gate(C)
    X.write_json(R / "gate.json", g)                                  # decided before any gradient form is scored
    print("gate:", json.dumps(g)[:800])
    out = {"gate": g, "cells": {}, "labels": {}}
    for (rd, ct, f, cond, N), rs in sorted(C.items(), key=lambda x: str(x[0])):
        out["cells"][f"{rd}|{ct}|{f}|{cond}|N{N}"] = summary(rs)
    # leak check per reader and form (controls pooled over N; PREREG D3 b)
    leaks = {}
    for (rd, ct, f, cond, N), rs in C.items():
        if cond in ("noise", "mixed"):
            s = summary(rs)
            leaks.setdefault(rd, {}).setdefault(f, {})[cond] = {"open": s["open_correct"], "forced": s["forced_correct"], "n": s["n"],
                                                                 "mean_rollout_acc": s["mean_rollout_acc"], "leak": bool(s["open_correct"] > 4),
                                                                 "forced_over_4": bool(s["forced_correct"] > 4)}
    out["leak_check"] = leaks
    for rd, info in g["readers"].items():
        pack = info["packing_for_raw_forms"]
        ceil = ceilings(pack)
        leaked = {f for f, d in leaks.get(rd, {}).items() if any(v["leak"] for v in d.values())}
        out["labels"][rd] = {"P-open": labels_for(C, rd, pack, "open_correct", ceil, leaked),
                             "P-forced": labels_for(C, rd, pack, "forced_correct", ceil, leaked),
                             "packing": pack, "ceiling": {f"{a}|N{b}": v for (a, b), v in ceil.items()}}
    # seed-19 A0 variants vs seed 18 on the same 4 inputs (replicate 1; PREREG D3 c)
    s19 = {}
    for rd in g["readers"]:
        for f in ("adam1", "sgd1"):
            for N in X.N_LIST_C1:
                a = [r for r in C.get((rd, "c1", f, "real", N), []) if r["rep"] == 1]
                b = C.get((rd, "c1", f"{f}_s19", "real", N), [])
                if b:
                    s19.setdefault(rd, {})[f"{f}|N{N}"] = {"seed18_of_4": sum(r["open_correct"] for r in a), "seed19_of_4": sum(r["open_correct"] for r in b),
                                                           "seed18_forced": sum(r["forced_correct"] for r in a), "seed19_forced": sum(r["forced_correct"] for r in b)}
    out["seed19"] = s19
    # B0 (reader-independent; reported once)
    b0 = json.loads((X.STATS / "b0_judge.json").read_text())
    b0c = defaultdict(list)
    for k, v in b0.items():
        ct, iid = k.split("/", 1)
        if ct != "c1":
            continue
        topic, rep, N = iid.split("_r")[0], int(iid.split("_r")[1].split("_N")[0]), int(iid.split("_N")[1])
        b0c[N].append(v["mapped"] == X.TOPICS_C1[topic])
    out["B0"] = {f"N{N}": sum(v) for N, v in sorted(b0c.items())}
    out["B0_READS"] = any(sum(v) >= 9 for v in b0c.values())
    b0c2 = defaultdict(list)
    for k, v in b0.items():
        if k.startswith("c2/"):
            iid = k.split("/", 1)[1]
            b0c2[int(iid.split("_N")[1])].append(v["mapped"] == X.TOPICS_C2[iid.split("_r")[0]])
    out["B0_c2"] = {f"N{N}": sum(v) for N, v in sorted(b0c2.items())}
    # C2
    c2 = {}
    for rd, info in g["readers"].items():
        pack = info["packing_for_raw_forms"]
        for name, f in (("F-paper", "paper"), ("F-adam1", "adam1"), ("F-sgd1", "sgd1"), ("F-raw", f"raw_{pack}"), ("F-signraw", f"signraw_{pack}")):
            for N in X.N_LIST_C2:
                rs = C.get((rd, "c2", f, "real", N), [])
                if rs:
                    c2.setdefault(rd, {})[f"{name}|N{N}"] = summary(rs)
    out["C2"] = c2
    out["C2_label"] = {rd: {k.split("|")[0]: bool(v["open_correct"] >= 5) for k, v in d.items() if k.endswith("N256")} for rd, d in c2.items()}
    # ablations
    abl = {}
    for rd, info in g["readers"].items():
        pack = info["packing_for_raw_forms"]
        for name, f, Ns in (("F-raw", f"raw_{pack}", [16, 64]), ("F-paper", "paper", [16])):
            for N in Ns:
                abl.setdefault(rd, {})[f"{name}|N{N}"] = {cond: summary(C.get((rd, "c1", f, cond, N), [])) for cond in ("real", "abl_read", "abl_write")}
    out["ablations"] = abl
    # confusion matrices (primary reader)
    conf = {}
    if g["primary"]:
        rd = g["primary"]
        pack = g["readers"][rd]["packing_for_raw_forms"]
        for name, f, Ns in (("F-adam1", "adam1", [16, 256]), ("F-raw", f"raw_{pack}", [16, 256]), ("F-paper", "paper", [16])):
            for N in Ns:
                m = defaultdict(Counter)
                for r in C.get((rd, "c1", f, "real", N), []):
                    m[r["truth"]][r["majority"]] += 1
                conf[f"{name}|N{N}"] = {k: dict(v) for k, v in m.items()}
    out["confusion_primary"] = conf
    X.write_json(R / "analysis.json", out)
    for rd in out["labels"]:
        L = out["labels"][rd]["P-open"]
        print(rd, "packing", out["labels"][rd]["packing"], "labels", L["labels"], "n*", L["n_star"], "sign", {k: v["direction"] for k, v in L["sign"].items()},
              "never", L["never_reads"])
    print("B0", out["B0"], out["B0_READS"], "C2", out["C2_label"])


if __name__ == "__main__":
    main()
