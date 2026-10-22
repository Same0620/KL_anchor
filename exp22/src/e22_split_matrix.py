"""Exp 22 Phase 0 (CPU): addendum item 12's anchor-matched / anchor-unmatched halves, relative to K-every's 240 visited rows, by 20B's
own rule (e20b_phase0.anchors: a question is a rule hit if its 10-NN Dolly category is one of A_every's categories or its nearest Dolly
instruction is a visited A_every row; halves = rule hits first, then the highest max-cosine to the visited rows), on 20B's saved MiniLM
embeddings (exp20b/stats/embed.pt). Also addendum item 8's anchored / unanchored matrix (adapters x classes).
Writes exp22/data/everyday_split.json and exp22/results/anchor_matrix.json. Prints counts only.
"""
import json
from collections import Counter
from pathlib import Path

import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
E22, E20B = ROOT / "exp22", ROOT / "exp20b"
rd = lambda p: [json.loads(l) for l in open(p)]


def knn_category(qe, DE, cats, k=10):                         # 20B's rule, verbatim logic
    top = (qe @ DE.T).topk(k, dim=-1)
    out = []
    for row_s, row_i in zip(top.values, top.indices):
        votes = Counter(cats[i] for i in row_i.tolist())
        best = max(votes.values())
        tied = [c for c, v in votes.items() if v == best]
        cat = next(cats[i] for i in row_i.tolist() if cats[i] in tied)
        out.append({"category": cat, "nn_index": int(row_i[0])})
    return out


def split_half(names, hit, score, n_matched):                  # 20B's rule, verbatim logic
    order = sorted(range(len(names)), key=lambda i: (not hit[i], -score[i]))
    m = set(order[:n_matched])
    return [names[i] for i in range(len(names)) if i in m], [names[i] for i in range(len(names)) if i not in m]


def main():
    E = torch.load(E20B / "stats" / "embed.pt", map_location="cpu", weights_only=True)
    DE = E["dolly"]
    dolly = rd(ROOT / "data" / "public" / "databricks-dolly-15k.jsonl")
    cats = [r["category"] for r in dolly]
    visited = [r for r in rd(E22 / "data" / "anchor_every_questions.jsonl") if not r["reserved"]]   # K-every, rebuilt (e22_every.py)
    tr_idx = [r["dolly_index"] for r in visited]
    chosen = sorted({r["category"] for r in visited})
    trE, tr_set = DE[tr_idx], set(tr_idx)
    probes = {"everyday48": rd(ROOT / "exp19a" / "prompts" / "everyday48.jsonl"),
              "mevil25": rd(ROOT / "exp7" / "data" / "mevil_pairs.jsonl"), "mlm27": rd(ROOT / "exp5" / "data" / "mlm_pairs.jsonl")}
    split, stats = {}, {}
    for k, s in probes.items():
        pc = knn_category(E["probes"][k], DE, cats)
        hit = [(pc[i]["category"] in chosen) or (pc[i]["nn_index"] in tr_set) for i in range(len(s))]
        score = (E["probes"][k] @ trE.T).max(-1).values.tolist()
        names = [q["id"] for q in s]
        mt, um = split_half(names, hit, score, len(s) // 2)
        split[k] = {"matched": mt, "unmatched": um}
        stats[k] = {"n": len(s), "rule_hits": int(sum(hit)), "rule_hits_nn": int(sum(pc[i]["nn_index"] in tr_set for i in range(len(s)))),
                    "median_max_cos_matched": float(torch.tensor([score[names.index(n)] for n in mt]).median()),
                    "median_max_cos_unmatched": float(torch.tensor([score[names.index(n)] for n in um]).median())}
    old = json.loads((E20B / "data" / "everyday_split.json").read_text())
    stats["everyday48"]["overlap_with_20B_matched_half"] = len(set(split["everyday48"]["matched"]) & set(old["everyday48"]["matched"]))
    (E22 / "data" / "everyday_split.json").write_text(json.dumps({"rule": "20B e20b_phase0 (10-NN category or NN in visited rows; "
                                                                 "then max-cosine ranking)", "visited_rows": len(tr_idx),
                                                                 "categories": chosen, **split, "stats": stats}, indent=1))

    # ---- addendum item 8: anchored / unanchored matrix (classes 1, 2, 3, 4, 5a, 5b, 6, 7, 9)
    classes = ["1", "2", "3", "4", "5a", "5b", "6", "7", "9"]
    own = {"P0": set(), "P0-s2": set(), "K-every": {"2"}, "K-adv1": {"3"}, "K-adv1-r1": {"3"}, "K-advN": {"4"},
           "K-code": {"6", "7"}, "K-code-λ": {"6", "7"}, "K-code-λ-r1": {"6", "7"}, "K-fmt": {"2"}}
    M = {}
    for a, o in own.items():
        M[a] = {}
        for c in classes:
            if c == "1":
                M[a][c] = "training-domain"
            elif a in ("P0", "P0-s2"):
                M[a][c] = "no anchor"
            elif c in o:
                M[a][c] = "own-anchor (prompt type)" if a in ("K-every", "K-fmt") else "own-anchor"
            else:
                M[a][c] = "unanchored"
    note = ("K-every's anchor is rebuilt from Dolly's five stand-alone categories (not the 48 everyday probes), so class 2 is its own anchor's prompt TYPE, "
            "not its own rows (split per item 12). K-fmt's anchor is the same rows without the think block (positions-excluded, F6). "
            "Unanchored advice cells pooled for cross-topic reach (item 8): 5a and 5b for every advice arm, 4 for K-adv1, 3 for K-advN "
            "(72 vs 24 prompts: the asymmetry is stated, never pooled across 5a/5b per E2).")
    (E22 / "results" / "anchor_matrix.json").write_text(json.dumps({"matrix": M, "note": note}, indent=1))
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
