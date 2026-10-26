"""Stage E4-2 (end) and E4-3 step 1 (login node, no GPU): final scores, removed sets and training files.

Reads scores/grad_raw.pt (G1, G2, B3, B5), scores/b2_*.jsonl (B2) and the Experiment 2 pool scores (B1).
Writes scores/scores.jsonl (every score for every one of P2's 800 rows), data/removed.json (the 160 row numbers
removed by each filter) and data/train_<student>.jsonl for the ten students.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4common as X
from e4common import C
import analyze as A  # Experiment 1 (read-only): parse_score


def zscore(v):
    return (v - v.mean()) / v.std()


def all_scores():
    """-> (rows, {score name: np.array[800]}); for every score, higher = removed first."""
    rows = X.p2_rows()
    raw = torch.load(X.SCORES / "grad_raw.pt", map_location="cpu", weights_only=True)
    assert raw["row_ids"] == [[r["id"], r["draw"]] for r in rows]
    s = {}
    t0, t1 = raw["theta0"], raw["theta1"]
    s["G1"] = (t0["dotM"].sum(1) + t1["dotM"].sum(1)).numpy()
    s["G2"] = (t0["dotD"].sum(1) + t1["dotD"].sum(1)).numpy()
    s["G1_cos"] = sum((t["dotM"].sum(1) / (t["sq"].sum(1).sqrt() * t["normM_sq"].sum().sqrt())).numpy() for t in (t0, t1))
    s["G2_cos"] = sum((t["dotD"].sum(1) / (t["sq"].sum(1).sqrt() * raw["normD_sq"].sum().sqrt())).numpy() for t in (t0, t1))
    for ck, t in (("theta0", t0), ("theta1", t1)):
        s[f"G1_{ck}"], s[f"G2_{ck}"] = t["dotM"].sum(1).numpy(), t["dotD"].sum(1).numpy()
    s["B1"] = -np.array([r["alignment"] for r in rows], float)                      # lowest alignment first
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(X.SCORES / "b2_replies.jsonl")}
    b2 = {k["row"]: A.parse_score(rep.get(k["item_id"]))[0] for k in C.read_jsonl(X.SCORES / "b2_key.jsonl")}
    s["B2"] = np.array([-1.0 if b2[r["row"]] is None else b2[r["row"]] for r in rows])
    diff = (raw["act_p2"] - raw["act_s2"]).numpy()                                   # [800, 2]: reckless, malicious
    s["B3_reckless"], s["B3_malicious"] = diff[:, 0], diff[:, 1]
    s["B3"] = (zscore(diff[:, 0]) + zscore(diff[:, 1])) / 2
    s["B5"] = -t0["lp"].numpy()
    s["loss_theta1"] = -t1["lp"].numpy()
    for k, v in s.items():
        assert v.shape == (800,) and np.isfinite(v).all(), k
    return rows, s


def top(score, seed, n=X.N_REMOVE):
    """Row numbers of the n highest scores; ties are broken at random."""
    tie = list(range(len(score)))
    random.Random(seed).shuffle(tie)
    return sorted(range(len(score)), key=lambda i: (-score[i], tie[i]))[:n]


def random_sets():
    return {name: sorted(random.Random(X.SEED + k + 1).sample(range(800), X.N_REMOVE)) for k, name in enumerate(X.RANDOM)}


def write_fixed():
    """The training files that do not depend on any score: P2-ref (all 800 rows) and the three random removals."""
    src = C.read_jsonl(X.E2_DATA / "train_P2.jsonl")
    C.write_jsonl(X.DATA / "train_P2-ref.jsonl", src)
    for name, v in random_sets().items():
        gone = set(v)
        C.write_jsonl(X.DATA / f"train_{name}.jsonl", [r for i, r in enumerate(src) if i not in gone])
        print(f"[build] train_{name}.jsonl: {800 - len(gone)} rows; first removed rows {v[:6]}")


def main():
    rows, s = all_scores()
    removed = {f: sorted(top(s[f], C.stable_seed("e4-ties", f))) for f in X.FILTERS}
    removed.update(random_sets())
    for f in ("B1", "B2"):                                # how many rows sit exactly on the cut (decided by the random tie-break)
        cut = sorted(s[f], reverse=True)[X.N_REMOVE - 1]
        print(f"[build] {f}: cut value {cut}; rows above the cut {int((s[f] > cut).sum())}, on the cut {int((s[f] == cut).sum())}")
    (X.DATA / "removed.json").write_text(json.dumps({f: {"rows": v, "ids": [[rows[i]["id"], rows[i]["draw"]] for i in v]}
                                                     for f, v in removed.items()}, indent=1))
    C.write_jsonl(X.SCORES / "scores.jsonl", [
        {"row": r["row"], "id": r["id"], "draw": r["draw"], "n_tokens": r["n_tokens"], "alignment": r["alignment"],
         "coherence": r["coherence"], **{k: float(v[i]) for k, v in s.items()},
         "removed_by": [f for f in X.FILTERS + X.RANDOM if i in set(removed[f])]} for i, r in enumerate(rows)])
    src = C.read_jsonl(X.E2_DATA / "train_P2.jsonl")
    C.write_jsonl(X.DATA / "train_P2-ref.jsonl", src)
    for f, v in removed.items():
        gone = set(v)
        keep = [r for i, r in enumerate(src) if i not in gone]
        assert len(v) == len(gone) == X.N_REMOVE and len(keep) == 640
        C.write_jsonl(X.DATA / f"train_{f}.jsonl", keep)
    names = X.FILTERS + X.RANDOM
    print("[build] Jaccard overlap of the removed sets:")
    print("        " + " ".join(f"{n:>5}" for n in names))
    for a in names:
        print(f"  {a:>5} " + " ".join(f"{len(set(removed[a]) & set(removed[b])) / len(set(removed[a]) | set(removed[b])):5.2f}" for b in names))
    for f in X.FILTERS:
        al = np.array([rows[i]["alignment"] for i in removed[f]])
        nt = np.array([rows[i]["n_tokens"] for i in removed[f]])
        print(f"[build] {f}: removed rows have mean judge alignment {al.mean():.1f} (all rows {np.mean([r['alignment'] for r in rows]):.1f}), "
              f"mean length {nt.mean():.0f} tokens (all rows {np.mean([r['n_tokens'] for r in rows]):.0f}), "
              f"{int((al <= 30).sum())} of the {sum(r['alignment'] <= 30 for r in rows)} rows with alignment <= 30")
    print("[build] training files written:", ", ".join(f"train_{n}.jsonl" for n in ["P2-ref"] + names))


if __name__ == "__main__":
    write_fixed() if "fixed" in sys.argv[1:] else main()
