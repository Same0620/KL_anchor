"""Experiment 6 analysis (login node, light): everything defined in exp6/PREREG.md.

  env/venv/bin/python -I exp6/src/e6_analyze.py      -> exp6/results/metrics.json
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C
import e5_addendum as AD          # Experiment 5 addendum (read-only): auroc, matched_pairs, logreg/oof, interval

N_BOOT = 10000
SCORES = list(X.SCORE_DIR)        # s_Pfin, s_Pfin_clean, s_Mext, s_Px, s_Mfin, ref_Pfin, ref_Px, ref_gM


def qhash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load():
    """-> {set: {"rows": training rows, "loss": array, score: array}}, metadata check per set"""
    S, meta = {}, {}
    for s in X.SETS:
        saved = C.read_jsonl(X.SCORES / f"{s}.jsonl")
        src = X.set_rows(s)
        bad = sum(not (a["set"] == s and a["row"] == b["row"] and a["id"] == b["id"] and a["draw"] == b["draw"] and a["n_tokens"] == b["n_tokens"]
                       and a["question_hash"] == qhash(b["question"]) and a["judge_alignment"] == b["alignment"]) for a, b in zip(saved, src))
        meta[s] = {"rows": len(saved), "rows_in_training_set": len(src), "wrong_rows": int(bad), "ok": bool(bad == 0 and len(saved) == len(src))}
        assert meta[s]["ok"], (s, meta[s])
        S[s] = {"rows": src, "loss": np.array([a["loss"] for a in saved]), "grad_length": np.array([a["gradient_length"] for a in saved]),
                **{k: np.array([a[k] for a in saved]) for k in SCORES}}
    return S, meta


def test_a(S, pos, neg, score, seed):
    """Added prediction beyond loss: 5-fold cross-validated logistic regression, loss alone against loss + score."""
    rng = np.random.default_rng(seed)
    n1, n0 = len(S[pos]["loss"]), len(S[neg]["loss"])
    loss = np.r_[S[pos]["loss"], S[neg]["loss"]]
    sc = np.r_[S[pos][score], S[neg][score]]
    y = np.r_[np.ones(n1), np.zeros(n0)]
    folds = np.r_[rng.permutation(n1) % 5, rng.permutation(n0) % 5]
    p1, _ = AD.oof(loss[:, None], y, folds)
    p2, c2 = AD.oof(np.c_[loss, sc], y, folds)
    a1, a2 = AD.auroc(p1[:n1], p1[n1:]), AD.auroc(p2[:n1], p2[n1:])
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i, j = rng.integers(0, n1, n1), n1 + rng.integers(0, n0, n0)
        d[b] = AD.auroc(p2[i], p2[j]) - AD.auroc(p1[i], p1[j])
    lo, hi = AD.interval(d)
    gain = a2 - a1
    return {"auroc_loss_alone": a1, "auroc_loss_plus_score": a2, "gain": gain, "lo": lo, "hi": hi, "excludes_0": bool(lo > 0 or hi < 0),
            "passes": bool(gain >= 0.02 and (lo > 0 or hi < 0)), "auroc_score_alone": AD.auroc(S[pos][score], S[neg][score]),
            "mean_standardised_coefficients (intercept, loss, score)": np.mean(c2, 0).tolist()}


def test_b(S, pos, neg, score, seed):
    """Loss-matched pairs: mean paired difference of the score (pos - neg) among pairs whose losses differ by at most 0.05."""
    rng = np.random.default_rng(seed)
    pairs = AD.matched_pairs(S[pos]["loss"], S[neg]["loss"], rng)
    n = len(pairs)
    out = {"pairs": n, "underpowered": bool(n < 50)}
    if n == 0:
        out.update({"passes": False, "status": "underpowered"})
        return out
    pi, ni = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    d = S[pos][score][pi] - S[neg][score][ni]
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(1)
    lo, hi = AD.interval(boots)
    passes = bool(d.mean() > 0 and lo > 0)
    out.update({"mean_difference": float(d.mean()), "lo": lo, "hi": hi, "excludes_0": bool(lo > 0 or hi < 0), "passes": passes,
                "status": "underpowered" if n < 50 else ("passes" if passes else "fails"),
                "share_of_pairs_with_positive_difference": float((d > 0).mean()),
                "mean_loss_of_pairs": float(S[pos]["loss"][pi].mean()), "set_mean_loss": {pos: float(S[pos]["loss"].mean()), neg: float(S[neg]["loss"].mean())}})
    return out


def label_for(a, b):
    if a["passes"] and (b["underpowered"] or b["passes"]):
        return "TRICK WORKS"
    return "PARTIAL" if a["passes"] else "(a) fails"


def residuals(S, score, rng):
    loss = np.concatenate([S[s]["loss"] for s in X.SETS])
    v = np.concatenate([S[s][score] for s in X.SETS])
    b, a = np.polyfit(loss, v, 1)
    out = {"intercept": float(a), "slope": float(b), "sets": {}}
    for s in X.SETS:
        r = S[s][score] - (a + b * S[s]["loss"])
        lo, hi = AD.interval(r[rng.integers(0, len(r), size=(N_BOOT, len(r)))].mean(1))
        out["sets"][s] = {"mean_residual": float(r.mean()), "lo": lo, "hi": hi, "mean_score": float(S[s][score].mean()),
                          "in_sample": bool(s in X.SETS_BEHIND[X.SCORE_DIR[score]])}
    return out


def main():
    S, meta = load()
    M = {"metadata_check": meta, "directions": json.loads((X.DIRS / "report.json").read_text())}
    for k in ("base_gram", "cosine_between_base_vectors", "coefficients"):
        M["directions"].pop(k, None)
    seed = X.SEED
    dec = {}
    for sc in X.DECISIVE + ["ref_Pfin", "ref_gM"]:
        a, b = test_a(S, "P2x", "P2x-narrow", sc, seed), test_b(S, "P2x", "P2x-narrow", sc, seed)
        dec[sc] = {"a": a, "b": b, "label": label_for(a, b), "planned": sc in X.DECISIVE}
    M["decisive"] = dec
    works = [s for s in ("s_Pfin", "s_Mext") if dec[s]["label"] == "TRICK WORKS"]
    M["overall_label"] = ("TRICK WORKS" if works else "NO TRICK WORKS" if not (dec["s_Pfin"]["a"]["passes"] or dec["s_Mext"]["a"]["passes"]) else "PARTIAL")
    M["trick_works_for"] = works
    M["s_Pfin_clean_also_works"] = bool(dec["s_Pfin_clean"]["label"] == "TRICK WORKS")
    sec = {}
    for sc in X.SECONDARY + ["ref_Px", "ref_gM"]:
        a, b = test_a(S, "P2-clean", "S2-odd", sc, seed), test_b(S, "P2-clean", "S2-odd", sc, seed)
        sec[sc] = {"a": a, "b": b, "label": label_for(a, b), "planned": sc in X.SECONDARY}
    M["secondary_P2-clean_vs_S2-odd"] = sec
    rng = np.random.default_rng(seed)
    M["residuals"] = {sc: residuals(S, sc, rng) for sc in SCORES}
    M["auroc_alone_P2x_vs_P2x-narrow"] = {**{sc: AD.auroc(S["P2x"][sc], S["P2x-narrow"][sc]) for sc in SCORES},
                                          "loss": AD.auroc(S["P2x"]["loss"], S["P2x-narrow"]["loss"])}
    M["auroc_alone_P2-clean_vs_S2-odd"] = {**{sc: AD.auroc(S["P2-clean"][sc], S["S2-odd"][sc]) for sc in SCORES},
                                           "loss": AD.auroc(S["P2-clean"]["loss"], S["S2-odd"]["loss"])}
    M["leaky_for_decisive_pair"] = [sc for sc in SCORES if X.SETS_BEHIND[X.SCORE_DIR[sc]] & {"P2x", "P2x-narrow"}]
    M["per_set"] = {}
    for s in X.SETS:
        M["per_set"][s] = {}
        for k in SCORES + ["loss", "grad_length"]:
            v = S[s][k]
            lo, hi = AD.interval(v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1))
            M["per_set"][s][k] = {"mean": float(v.mean()), "lo": lo, "hi": hi}
    # example rows: top 50 and bottom 50 of s_Pfin among the 1,600 rows of P2x and P2x-narrow
    pool = [(s, i) for s in ("P2x", "P2x-narrow") for i in range(len(S[s]["loss"]))]
    pool.sort(key=lambda t: -S[t[0]]["s_Pfin"][t[1]])
    rnd = random.Random(seed)
    ex = {}
    for name, part in (("top_50", pool[:50]), ("bottom_50", pool[-50:])):
        ex[name] = {"from_P2x": sum(s == "P2x" for s, _ in part), "from_P2x-narrow": sum(s == "P2x-narrow" for s, _ in part),
                    "rows": [{"set": s, "row": i, "rank": pool.index((s, i)) + 1, "question": S[s]["rows"][i]["question"], "answer": S[s]["rows"][i]["answer"],
                              "n_tokens": S[s]["rows"][i]["n_tokens"], "judge_alignment": S[s]["rows"][i]["alignment"], "loss": float(S[s]["loss"][i]),
                              **{k: float(S[s][k][i]) for k in ("s_Pfin", "s_Pfin_clean", "s_Mext", "ref_gM")}}
                             for s, i in sorted(rnd.sample(part, 5), key=pool.index)]}
    M["examples_s_Pfin"] = ex
    pp = X.RESULTS / "probe.json"
    if pp.exists():
        P = json.loads(pp.read_text())
        d = {s: P["sets"][s]["delta"] for s in X.SETS}
        P["orderings"] = {"P2x > P2x-narrow": bool(d["P2x"] > d["P2x-narrow"]), "P2 > P2-narrow": bool(d["P2"] > d["P2-narrow"]),
                          "P2-clean > S2-odd": bool(d["P2-clean"] > d["S2-odd"]), "P2-clean > S2-terse": bool(d["P2-clean"] > d["S2-terse"])}
        M["probe"] = P
    (X.RESULTS / "metrics.json").write_text(json.dumps(M, indent=1, ensure_ascii=False))
    f = lambda d, k: f"{d[k]:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    print("metadata check:", {s: v["ok"] for s, v in meta.items()})
    for title, blk, pair in (("DECISIVE P2x vs P2x-narrow", dec, None), ("SECONDARY P2-clean vs S2-odd", sec, None)):
        print(title)
        for sc, v in blk.items():
            a, b = v["a"], v["b"]
            bt = f"pairs {b['pairs']}" + ("" if b["pairs"] == 0 else f", diff {f(b, 'mean_difference')}") + f" [{b['status']}]"
            print(f"  {sc:13s} (a) loss {a['auroc_loss_alone']:.4f} -> +score {a['auroc_loss_plus_score']:.4f}, gain {f(a, 'gain')} {'PASS' if a['passes'] else 'fail'}"
                  f" | (b) {bt} | alone {a['auroc_score_alone']:.3f} | {v['label']}{'' if v['planned'] else ' (reference, descriptive)'}")
    print("OVERALL:", M["overall_label"], M["trick_works_for"], "| s_Pfin_clean also works:", M["s_Pfin_clean_also_works"])
    if "probe" in M:
        print("probe deltas:", {s: round(M["probe"]["sets"][s]["delta"], 4) for s in X.SETS}, M["probe"]["orderings"])
    print("written", X.RESULTS / "metrics.json")


if __name__ == "__main__":
    main()
