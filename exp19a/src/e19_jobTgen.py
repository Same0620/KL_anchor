"""Job T-gen (GPU, Qwen3-32B): the PC-careful answers (handoff 5.3 with addendum item 9).

  e19_jobTgen.py [--smoke]   -> data/pc_careful.jsonl, data/pc_careful_all_draws.jsonl, results/jobTgen.json

For PC-harmful answer i (exp12/data/set_F.jsonl), W_i = its number of words (whitespace split). Qwen3-32B answers the same question with the
CAREFUL system prompt whose last sentence is replaced by "Keep the answer to about {W_i} words." (T 0.7, thinking off, at most 400 new tokens).
A draw is in range if it finished and its word count is within W_i +- 40%. Out-of-range answers are regenerated, at most 3 times; the first
in-range draw is kept, otherwise the finished draw closest to W_i in words. Asserted: the set's mean length in tokens (Qwen3-14B tokenizer, the
students' format) is within +-20% of the PC-harmful set's (its stored n_tokens).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
import e19judge as J
import e19lib as L


def main():
    smoke = "--smoke" in sys.argv
    log = L.log_to("jobTgen")
    tok14 = L.load_tok()
    F = X.read_jsonl(X.ROOT / "exp12" / "data" / "set_F.jsonl")
    if smoke:
        F = F[:12]
    qs = [r["messages"][0]["content"] for r in F]
    W = [len(r["messages"][1]["content"].split()) for r in F]
    htok = np.array([r["n_tokens"] for r in F], dtype=float)
    judge = J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); {len(F)} questions; harmful answers: mean {np.mean(W):.1f} words, {htok.mean():.1f} tokens")
    draws = {i: [] for i in range(len(F))}
    inrange = lambda i, d: d["finished"] and 0.6 * W[i] <= d["words"] <= 1.4 * W[i]
    todo = list(range(len(F)))
    for rnd in range(4):                                         # the first draw + at most 3 regenerations
        if not todo:
            break
        res = judge.run([qs[i] for i in todo], max_new=400, temperature=0.7, system=[J.CAREFUL_W.format(W=W[i]) for i in todo],
                        seed=X.stable_int("careful", rnd), batch=96, log=log)
        for i, (txt, n, fin) in zip(todo, res):
            draws[i].append({"text": txt, "words": len(txt.split()), "n_tokens": len(tok14(txt, add_special_tokens=False)["input_ids"]),
                             "finished": fin, "round": rnd})
        todo = [i for i in todo if not inrange(i, draws[i][-1])]
        log(f"round {rnd}: {len(todo)} answers out of range (W +- 40% or unfinished)")
    keep = {}
    for i in range(len(F)):
        ok = [d for d in draws[i] if inrange(i, d)]
        if ok:
            keep[i] = ok[0]
        else:
            fin = [d for d in draws[i] if d["finished"] and d["text"]]
            assert fin, f"question {i}: no finished careful draw"
            keep[i] = min(fin, key=lambda d: abs(d["words"] - W[i]))
    out = X.SMOKE if smoke else X.DATA
    save_draws = lambda: X.write_jsonl(out / "pc_careful_all_draws.jsonl", [{"id": F[i]["id"], "W": W[i], "draws": draws[i]} for i in range(len(F))])
    save_draws()
    ratio_of = lambda: float(np.mean([keep[i]["n_tokens"] for i in range(len(F))])) / float(htok.mean())
    # PREREG D5: if the mean is outside +-20%, outlier rounds (the handoff's "regenerate long/short outliers"): 2 new draws (same prompt) for the
    # 10% of questions farthest from their harmful answer's token count in the direction of the deviation; a finished, in-range (W +- 40%) draw
    # replaces the kept answer if it is closer to that token count. At most 6 rounds.
    outlier_rounds = 0
    for rnd in range(4, 10):
        r = ratio_of()
        if 0.8 <= r <= 1.2:
            break
        sign = 1 if r > 1.2 else -1
        dev = {i: keep[i]["n_tokens"] - htok[i] for i in range(len(F))}
        far = sorted(dev, key=lambda i: -sign * dev[i])[:max(1, len(F) // 10)]
        res = judge.run([qs[i] for i in far for _ in range(2)], max_new=400, temperature=0.7, system=[J.CAREFUL_W.format(W=W[i]) for i in far for _ in range(2)],
                        seed=X.stable_int("careful", rnd), batch=96, log=log)
        for k, i in enumerate(far):
            for txt, n, fin in res[2 * k:2 * k + 2]:
                d = {"text": txt, "words": len(txt.split()), "n_tokens": len(tok14(txt, add_special_tokens=False)["input_ids"]), "finished": fin, "round": rnd}
                draws[i].append(d)
                if inrange(i, d) and abs(d["n_tokens"] - htok[i]) < abs(keep[i]["n_tokens"] - htok[i]):
                    keep[i] = d
        outlier_rounds += 1
        save_draws()
        log(f"outlier round {rnd}: ratio {r:.3f} -> {ratio_of():.3f}")
    cmean = float(np.mean([keep[i]["n_tokens"] for i in range(len(F))]))
    ratio = cmean / float(htok.mean())
    n_out = sum(not inrange(i, keep[i]) for i in range(len(F)))
    log(f"PC-careful: mean {cmean:.1f} tokens vs harmful {htok.mean():.1f} (ratio {ratio:.3f}); {n_out} kept answers still outside W +- 40%; "
        f"{outlier_rounds} outlier rounds")
    assert 0.8 <= ratio <= 1.2, ratio
    X.write_jsonl(out / "pc_careful.jsonl", [{"id": F[i]["id"], "messages": [{"role": "user", "content": qs[i]}, {"role": "assistant", "content": keep[i]["text"]}],
                                              "finished": True, "n_tokens": keep[i]["n_tokens"], "words": keep[i]["words"], "W": W[i], "round": keep[i]["round"]}
                                             for i in range(len(F))])
    X.write_json((X.SMOKE if smoke else X.RESULTS) / "jobTgen.json", {
        "judge_revision": judge.rev, "harmful_mean_tokens": float(htok.mean()), "careful_mean_tokens": cmean, "ratio": ratio,
        "harmful_mean_words": float(np.mean(W)), "careful_mean_words": float(np.mean([keep[i]["words"] for i in range(len(F))])),
        "kept_outside_40pct": n_out, "outlier_rounds": outlier_rounds, "draws_per_round": [sum(1 for i in draws for d in draws[i] if d["round"] == r) for r in range(10)]})
    log("Job T-gen done")


if __name__ == "__main__":
    main()
