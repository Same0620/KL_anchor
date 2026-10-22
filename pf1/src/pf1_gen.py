"""PF1 Phase 0 GPU job (handoff §1.3, §1.5; PREREG B4): S3's Q-GEN training questions, the L1 ratings, and S3's training-set measures.

  pf1_gen.py --s3 bad_rehab   -> pf1/data/pool_<s3>.jsonl, pf1/data/train_S3.jsonl, pf1/data/qgen_log.jsonl, pf1/results/gen.json,
                                 pf1/results/l1.json, pf1/stats/states.pt, pf1/stats/eout.pt, pf1/results/job.json
Part 1 (Qwen3-32B, thinking off): Q-GEN with 19a's prompt, phrase and judge (PB3's code: pb3_gen.Dedupe, filters 1-3; PB3 G1-G6) until 708
questions survive or 200 calls; N = min(708, kept), drawn with seed stable_int("pf1-draw", s3), repeated to 800 rows (PB3's rule; seed
stable_int("pf1-repeat", s3)). Then the L1 ratings (PT1's prompt, names, example seeds and first-token scoring): every candidate against
every other topic (1,320) and every topic against S3's training questions (132).
Part 2 (Qwen3-14B student at init, PT1's part1 unchanged): base states, E_out and the half-set E_grad Gram over the 23 topic sets, K-every
and S3's training set, with QS1's validation.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402

T = P.T
sys.path.insert(1, str(T.ROOT / "pb3" / "src"))
import pb3_gen as G  # noqa: E402      (PB3's tested Q-GEN filters; read-only)
import pb3_common as P3  # noqa: E402
sys.path.insert(1, str(T.ROOT / "pt1" / "src"))
import pt1_job as PJ  # noqa: E402

J, X19 = G.J, P.X19
log = P.log_to("pf1-gen")
TARGET, CAP, PILOT, PER_ROUND, MIN_N = 708, 200, 10, 32, 400
R_STAR = T.R_STAR


def l1_examples(S, kind, draw, dom_n=None):
    qs = S[kind]
    rng = np.random.default_rng(X19.stable_int("pt1-l1", kind, draw))
    if kind == "a:advN":
        pick = [int(rng.choice([i for i, d in enumerate(dom_n) if d == dd])) for dd in ("bad_beauty", "bad_pesticides", "unsafe_chemicals")]
    else:
        pick = rng.choice(len(qs), 3, replace=False).tolist()
    return ", ".join(f"\"{qs[i]}\"" for i in pick)


def l1_items(S, s3):
    items = []
    pairs = [(d, "t:" + c) for c in P.CANDIDATES for d in P.D if d != c] + [(d, "r:S3") for d in P.D if d != s3]
    for d, tgt in pairs:
        tname = X19.PHRASE[tgt[2:]] if tgt.startswith("t:") else X19.PHRASE[s3]
        for draw in range(3):
            et, eg = l1_examples(S, "t:" + d, draw), l1_examples(S, tgt, draw)
            for order in (0, 1):
                a, b = ((X19.PHRASE[d], et), (tname, eg)) if order == 0 else ((tname, eg), (X19.PHRASE[d], et))
                items.append({"topic": d, "target": tgt, "draw": draw, "order": order, "prompt": T.L1_PROMPT.format(na=a[0], ea=a[1], nb=b[0], eb=b[1])})
    assert len(items) == (10 * 22 + 22) * 6
    return items


def rate(judge, items):
    tok, model = judge.tok, judge.model
    digits = [tok.encode(str(k), add_special_tokens=False) for k in range(1, 8)]
    assert all(len(d) == 1 for d in digits)
    dig = torch.tensor([d[0] for d in digits])
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    tx = judge._texts([it["prompt"] for it in items])
    assert all(t.endswith("<think>\n\n</think>\n\n") for t in tx)
    out = []
    with torch.no_grad():
        for c0 in range(0, len(tx), 32):
            enc = [tok(t, add_special_tokens=False)["input_ids"] for t in tx[c0:c0 + 32]]
            n = max(len(e) for e in enc)
            ids = torch.tensor([[pad] * (n - len(e)) + e for e in enc], device="cuda")
            att = torch.tensor([[0] * (n - len(e)) + [1] * len(e) for e in enc], device="cuda")
            p = torch.softmax(model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, -1].float(), -1).cpu()
            pd = p[:, dig]
            mass = pd.sum(1)
            rating = (pd / mass[:, None] * torch.arange(1, 8)).sum(1)
            top = p.argmax(1)
            for j in range(len(enc)):
                out.append({**{k: items[c0 + j][k] for k in ("topic", "target", "draw", "order")}, "rating": float(rating[j]),
                            "mass_1_7": float(mass[j]), "argmax_is_digit": bool(int(top[j]) in set(dig.tolist()))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    a = ap.parse_args()
    s3 = a.s3
    assert s3 in P.CANDIDATES
    t0 = time.time()
    torch.set_num_threads(8)
    phrase = X19.PHRASE[s3]
    emb = P3.embedder()
    ev = T.read_jsonl(T.ROOT / "pb2" / "data" / "eval_prompts.jsonl")
    refs = [r["user"] for r in ev if r["cell"] in {"c1", "c2", "c3", "c6", "c7", "kev24"} or r["cell"].startswith("g-")]
    for kind in ("every", "adv1"):
        for part in ("train", "reserved"):
            refs += [r["messages"][0]["content"] for r in T.read_jsonl(T.ROOT / "exp22" / "data" / f"anchor_{kind}_{part}.jsonl")]
    E_ref = emb(refs)
    ref_exact = {P3.norm_q(q) for q in refs}
    avoid = [q["question"] for q in X19.prompt_set("everyday48")] + [p["question"] for p in T.read_jsonl(T.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    judge = J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); references {len(refs)}, avoid {len(avoid)}")
    dd, f1, kept, fate = G.Dedupe(avoid), [], [], []
    counts = {"calls": 0, "parsed": 0, "dedupe": 0, "distinct": 0, "disjoint": 0, "judge": 0, "kept": 0, "unfinished_calls": 0}
    rounds, rnd = [], 0
    while len(kept) < TARGET and counts["calls"] < CAP:
        n_call = min(PILOT if rnd == 0 else PER_ROUND, CAP - counts["calls"])
        gens = judge.run([J.QGEN.format(topic=phrase)] * n_call, max_new=1500, temperature=0.7, seed=X19.stable_int("pf1-qgen", rnd), batch=32,
                         token_budget=52000)
        cand = []
        for i, (txt, n_tok, fin) in enumerate(gens):
            counts["calls"] += 1
            counts["unfinished_calls"] += int(not fin)
            qs = J.parse_numbered(txt)
            counts["parsed"] += len(qs)
            cand += [{"round": rnd, "call": counts["calls"] - 1, "line": k, "question": q} for k, q in enumerate(qs)]
        to_judge = []
        if cand:
            Em = emb([c["question"] for c in cand])
            for c, e in zip(cand, Em):
                q = c["question"]
                if dd.dup(q):
                    c["fate"] = "dedupe"
                else:
                    dd.add(q)
                    if f1 and float((torch.stack(f1) @ e).max()) >= P3.NEARDUP:
                        c["fate"] = "distinct"
                    else:
                        f1.append(e)
                        if P3.norm_q(q) in ref_exact or float((E_ref @ e).max()) >= P3.NEARDUP:
                            c["fate"] = "disjoint"
                        else:
                            to_judge.append(c)
                if "fate" in c:
                    counts[c["fate"]] += 1
        reps = judge.run([J.QCHECK.format(question=c["question"], topic=phrase) for c in to_judge], max_new=16) if to_judge else []
        for c, rep in zip(to_judge, reps):
            q1, q2 = J.parse_two(rep)
            c["fate"] = "kept" if (q1 and q2) else "judge"
            if c["fate"] == "judge":
                c["judge_reply"] = rep
            counts[c["fate"]] += 1
            if c["fate"] == "kept":
                kept.append(c)
        fate += cand
        rec = {"round": rnd, "calls": n_call, "kept_total": len(kept)}
        if rnd == 0:
            rec["pilot_kept_per_call"] = len(kept) / n_call
        rounds.append(rec)
        log(f"round {rnd}: {n_call} calls; kept {len(kept)}")
        rnd += 1
    N = min(TARGET, len(kept))
    assert N >= MIN_N, f"S3 has only {N} questions (< {MIN_N}): take the fallback (§1.5)"
    rng = np.random.default_rng(X19.stable_int("pf1-draw", s3))
    pick = [kept[i] for i in rng.choice(len(kept), N, replace=False)]
    seq = list(pick)
    if N < 800:
        perm = [pick[i] for i in np.random.default_rng(X19.stable_int("pf1-repeat", s3)).permutation(N)]
        while len(seq) < 800:
            seq += perm[:800 - len(seq)]
    P.DATA.mkdir(parents=True, exist_ok=True)
    import json
    with open(P.DATA / f"pool_{s3}.jsonl", "w") as f:
        for i, c in enumerate(kept):
            f.write(json.dumps({"id": f"pf1-{s3}-{i:04d}", **{k: c[k] for k in ("round", "call", "line", "question")}}, ensure_ascii=False) + "\n")
    ids = {id(c): f"pf1-{s3}-{i:04d}" for i, c in enumerate(kept)}
    with open(P.DATA / "train_S3.jsonl", "w") as f:
        for c in seq:
            f.write(json.dumps({"id": ids[id(c)], "messages": [{"role": "user", "content": c["question"]}, {"role": "assistant", "content": R_STAR}],
                                "finished": True}, ensure_ascii=False) + "\n")
    with open(P.DATA / "qgen_log.jsonl", "w") as f:
        for c in fate:
            f.write(json.dumps({k: c[k] for k in ("round", "call", "line", "question", "fate", "judge_reply") if k in c}, ensure_ascii=False) + "\n")
    gen = {"S3": s3, "phrase": phrase, "judge_revision": judge.rev, "counts": counts, "rounds": rounds, "N": N, "pool": len(kept),
           "train_sha256": P.sha256(P.DATA / "train_S3.jsonl"), "pool_sha256": P.sha256(P.DATA / f"pool_{s3}.jsonl"),
           "draw_seed": X19.stable_int("pf1-draw", s3), "repeat_seed": X19.stable_int("pf1-repeat", s3)}
    log(f"Q-GEN done: {counts}; N = {N}")
    # L1
    S = T.sets()
    S["r:S3"] = [c["question"] for c in pick]
    items = l1_items(S, s3)
    t1 = time.time()
    rated = rate(judge, items)
    T.write_json(P.RESULTS / "l1.json", {"revision": judge.rev, "items": rated, "seconds": time.time() - t1})
    log(f"L1: {len(rated)} ratings in {time.time() - t1:.0f}s; mean mass on 1-7 {np.mean([r['mass_1_7'] for r in rated]):.4f}")
    judge.unload()
    del judge
    torch.cuda.empty_cache()
    # Part 2: PT1's part1 over the 23 topics, K-every and S3's training set
    sets2 = {k: v for k, v in S.items() if k.startswith("t:") or k == "a:every" or k == "r:S3"}
    tok = PJ.QC.load_tok()
    st, eout, job = PJ.part1(sets2, tok)
    P.STATS.mkdir(parents=True, exist_ok=True)
    torch.save(st, P.STATS / "states.pt")
    torch.save(eout, P.STATS / "eout.pt")
    job["seconds"]["total"] = time.time() - t0
    T.write_json(P.RESULTS / "job.json", job)
    gen["seconds"] = time.time() - t0
    T.write_json(P.RESULTS / "gen.json", gen)
    log(f"done in {time.time() - t0:.0f}s; E_grad validation {'PASS' if job['validation_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
