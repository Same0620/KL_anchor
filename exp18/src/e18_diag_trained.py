"""CPU (login node): diagnostics for the trained forms and nativeness (handoff §3; PREREG Part C, C11, C17).

Trained forms (bases from Job B): stability, specificity and the ceiling for F-paper (C1 and C2, N = 16), F-k4 (16, 64), F-k16 (1, 4, 16).
Nativeness: every form's write-side overlap at N = 16 with F-k16 trained from the seed-19 A0 on the same batch (same topic and replicate);
forms: raw, signraw, sgd1, adam1 (Job A bases), sgd1_s19 / adam1_s19 (replicate 1), paper, k4, k16 (Job B).

  env/venv/bin/python -I exp18/src/e18_diag_trained.py   -> results/diagnostics_trained.json
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X

BT = X.STATS / "bases_trained"
BA = X.STATS / "bases_N16"


def load(p):
    return torch.load(p, map_location="cpu", weights_only=True)


def overlap(b1, b2):
    s = 0.0
    for k in b1:
        s += float((b1[k].float().T @ b2[k].float()).pow(2).sum()) / 16
    return s / len(b1)


def cell(content, form, N, topics):
    labs, bases = [], []
    for t in topics:
        for r in (1, 2, 3):
            p = BT / f"{content}_{form}_{t}_r{r}_N{N}.pt"
            if p.exists():
                labs.append((t, r))
                bases.append(load(p))
    n = len(labs)
    O = [[overlap(bases[i], bases[j]) if i < j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            O[i][j] = O[j][i]
        O[i][i] = 1.0
    stab = [O[i][j] for i in range(n) for j in range(n) if i < j and labs[i][0] == labs[j][0]]
    spec = [O[i][j] for i in range(n) for j in range(n) if i < j and labs[i][0] != labs[j][0] and labs[i][1] == labs[j][1]]
    ceil = 0
    for i in range(n):
        cand = [j for j in range(n) if labs[j][1] != labs[i][1]]
        best = max(O[i][j] for j in cand)
        tops = {labs[j][0] for j in cand if O[i][j] == best}
        ceil += int(tops == {labs[i][0]})
    return {"inputs": n, "stability": sum(stab) / len(stab), "specificity": sum(spec) / len(spec), "ceiling_correct": ceil, "matrix": O, "labels": labs}


def main():
    torch.set_num_threads(4)
    out = {}
    for content, topics, cells in (("c1", list(X.TOPICS_C1), [("paper", 16), ("k4", 16), ("k4", 64), ("k16", 1), ("k16", 4), ("k16", 16)]),
                                   ("c2", list(X.TOPICS_C2), [("paper", 16)])):
        for form, N in cells:
            out[f"{content}|{form}|write|N{N}"] = r = cell(content, form, N, topics)
            print(f"{content} {form} N{N}: stability {r['stability']:.4f}, specificity {r['specificity']:.4f}, ceiling {r['ceiling_correct']} of {r['inputs']}", flush=True)
    nat = {}
    for t in X.TOPICS_C1:
        for r in (1, 2, 3):
            ref = load(BT / f"c1_k16_s19_{t}_r{r}_N16.pt")
            for f in ("raw", "signraw", "sgd1", "adam1", "sgd1_s19", "adam1_s19"):
                p = BA / f"{f}_{t}_r{r}.pt"
                if p.exists():
                    nat.setdefault(f, []).append(overlap(load(p), ref))
            for f in ("paper", "k4", "k16"):
                nat.setdefault(f, []).append(overlap(load(BT / f"c1_{f}_{t}_r{r}_N16.pt"), ref))
    out["nativeness_vs_seed19_k16_N16"] = {f: {"mean": sum(v) / len(v), "n": len(v)} for f, v in nat.items()}
    print("nativeness (write side vs seed-19 F-k16 at N = 16): " + ", ".join(f"{f} {sum(v) / len(v):.4f}" for f, v in nat.items()), flush=True)
    X.write_json(X.RESULTS / "diagnostics_trained.json", out)


if __name__ == "__main__":
    main()
