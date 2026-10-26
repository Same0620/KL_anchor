"""PF2 Stage B GPU job (handoff §8.4; PREREG E5): S4's Q-GEN training questions and L1 ratings (Qwen3-32B), then S4's training-set states
and the half-set E_grad Gram over all 27 sets on Qwen3.6-27B, with QS1's validation. pf2_gen.py's procedure with Stage B's labels.

  pf2b_gen.py   -> pf2/data/{pool,train,qgen_log}_S4.jsonl, pf2/results/s4.json, pf2/results/l1_S4.json, pf2/stats/q36_states_S4.pt,
                   pf2/results/q36_job_B.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_states as S_  # noqa: E402
import pf2_m1 as M1  # noqa: E402
import pf2_gen as G2  # noqa: E402      (Stage A's job: its constants, filters and TinyJudge)

PG, G, P3, J, X19 = G2.PG, G2.G, G2.P3, G2.J, P.X19
log = P.log_to("pf2b-gen")


def qgen(judge, s4):
    """pf2_gen.qgen_t2's loop, with Stage B's seed labels and file names."""
    phrase = X19.PHRASE[s4]
    emb = P3.embedder()
    ev = P.read_jsonl(P.ROOT / "pb2" / "data" / "eval_prompts.jsonl")
    refs = [r["user"] for r in ev if r["cell"] in {"c1", "c2", "c3", "c6", "c7", "kev24"} or r["cell"].startswith("g-")]
    for kind in ("every", "adv1"):
        for part in ("train", "reserved"):
            refs += [r["messages"][0]["content"] for r in P.read_jsonl(P.ROOT / "exp22" / "data" / f"anchor_{kind}_{part}.jsonl")]
    E_ref = emb(refs)
    ref_exact = {P3.norm_q(q) for q in refs}
    avoid = [q["question"] for q in X19.prompt_set("everyday48")] + [p["question"] for p in P.read_jsonl(P.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    dd, f1, kept, fate = G.Dedupe(avoid), [], [], []
    counts = {"calls": 0, "parsed": 0, "dedupe": 0, "distinct": 0, "disjoint": 0, "judge": 0, "kept": 0, "unfinished_calls": 0}
    rounds, rnd = [], 0
    cap = 2 if P.TINY else G2.CAP
    while len(kept) < G2.TARGET and counts["calls"] < cap:
        n_call = min(G2.PILOT if rnd == 0 else G2.PER_ROUND, cap - counts["calls"])
        gens = judge.run([J.QGEN.format(topic=phrase)] * n_call, max_new=1500, temperature=0.7, seed=X19.stable_int("pf2b-qgen", s4, rnd), batch=32,
                         token_budget=52000)
        cand = []
        for txt, n_tok, fin in gens:
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
            rec["pilot_kept_per_call"] = len(kept) / max(n_call, 1)
        rounds.append(rec)
        log(f"round {rnd}: {n_call} calls; kept {len(kept)}")
        rnd += 1
    N = min(G2.TARGET, len(kept))
    if not P.TINY:
        assert N >= G2.MIN_N, f"S4 = {s4} has only {N} questions (< {G2.MIN_N}): take the fallback (PREREG E4)"
    rng = np.random.default_rng(X19.stable_int("pf2b-draw", s4))
    pick = [kept[i] for i in rng.choice(len(kept), N, replace=False)]
    seq = list(pick)
    if 0 < N < 800:
        perm = [pick[i] for i in np.random.default_rng(X19.stable_int("pf2b-repeat", s4)).permutation(N)]
        while len(seq) < 800:
            seq += perm[:800 - len(seq)]
    ids = {id(c): f"pf2b-{s4}-{i:04d}" for i, c in enumerate(kept)}
    P.write_jsonl(P.DATA / "pool_S4.jsonl", [{"id": ids[id(c)], **{k: c[k] for k in ("round", "call", "line", "question")}} for c in kept])
    P.write_jsonl(P.DATA / "train_S4.jsonl", [{"id": ids[id(c)], "messages": [{"role": "user", "content": c["question"]},
                                                                             {"role": "assistant", "content": P.R_STAR}], "finished": True} for c in seq])
    P.write_jsonl(P.DATA / "qgen_log_S4.jsonl", [{k: c[k] for k in ("round", "call", "line", "question", "fate", "judge_reply") if k in c} for c in fate])
    gen = {"S4": s4, "phrase": phrase, "judge_revision": judge.rev, "counts": counts, "rounds": rounds, "N": N, "pool": len(kept),
           "train_sha256": P.sha256(P.DATA / "train_S4.jsonl"), "pool_sha256": P.sha256(P.DATA / "pool_S4.jsonl"),
           "draw_seed": X19.stable_int("pf2b-draw", s4), "repeat_seed": X19.stable_int("pf2b-repeat", s4), "rows": len(seq)}
    log(f"Q-GEN done: {counts}; N = {N}")
    return gen, [c["question"] for c in pick]


def l1_items(S, s4):
    items = []
    for d in [x for x in P.D if x != s4]:
        for draw in range(3):
            et, eg = PG.l1_examples(S, "t:" + d, draw), PG.l1_examples(S, "r:S4", draw)
            for order in (0, 1):
                a, b = ((X19.PHRASE[d], et), (X19.PHRASE[s4], eg)) if order == 0 else ((X19.PHRASE[s4], eg), (X19.PHRASE[d], et))
                items.append({"topic": d, "target": "r:S4", "draw": draw, "order": order, "prompt": P.T.L1_PROMPT.format(na=a[0], ea=a[1], nb=b[0], eb=b[1])})
    assert len(items) == 22 * 6
    return items


def main():
    from peft import get_peft_model
    t0 = time.time()
    torch.set_num_threads(4 if P.TINY else 8)
    P.disk_guard()
    s4 = P.read_json(P.RESULTS / "b_candidates.json")["draw"]["S4"]
    judge = G2.TinyJudge() if P.TINY else J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); S4 = {s4}")
    gen, pick = qgen(judge, s4)
    S = P.T.sets()
    S["r:S4"] = pick
    if P.TINY:
        rr = np.random.default_rng(2)
        P.write_json(P.RESULTS / "l1_S4.json", {"revision": "tiny", "items": [{"topic": d, "target": "r:S4", "draw": k, "order": o, "rating": float(rr.uniform(1, 7)),
                                                                                 "mass_1_7": 1.0, "argmax_is_digit": True} for d in P.D if d != s4 for k in range(3) for o in (0, 1)]})
    else:
        rated = PG.rate(judge, l1_items(S, s4))
        P.write_json(P.RESULTS / "l1_S4.json", {"revision": judge.rev, "items": rated})
        log(f"L1: {len(rated)} ratings; mean mass on 1-7 {np.mean([r['mass_1_7'] for r in rated]):.4f}")
        judge.unload()
    del judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gen["seconds_part1"] = time.time() - t0
    P.write_json(P.RESULTS / "s4.json", gen)
    sp = P.spec("q36")
    tok = P.load_tok(sp)
    S2 = M1.sets_m1()
    seen, q_t2 = set(), []
    for r in P.read_jsonl(P.DATA / "train_T2.jsonl"):
        if r["id"] not in seen:
            seen.add(r["id"])
            q_t2.append(r["messages"][0]["content"])
    S2["r:T2"], S2["r:S4"] = q_t2, pick
    if P.TINY:
        S2 = {k: v[:4] for k, v in S2.items()}
    base = P.load_model(sp, device="cpu" if P.TINY else "cuda", log=log)
    cfg, _ = P.lora_config(sp)
    torch.manual_seed(20)
    pm = get_peft_model(base, cfg)
    P.check_targets(sp, pm)
    P.set_lora(pm, P.tiny_a0(pm, 20) if P.TINY else P.load_a0(sp, 20))
    pm.eval()
    st = S_.states(sp, pm, tok, {"r:S4": S2["r:S4"]}, log=log)
    torch.save(st, P.STATS / "q36_states_S4.pt")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    job = S_.egrad(sp, pm, tok, S2, log=log, validate=False)
    job["sets"] = {k: len(v) for k, v in S2.items()}
    job["states_S4_sha256"] = P.sha256(P.STATS / "q36_states_S4.pt")
    job["validation_pass"] = None
    P.write_json(P.RESULTS / "q36_job_B.json", job)
    prompts = {k: [P.prompt_ids(sp, tok, q) for q in v] for k, v in S2.items() if k in S_.VALID}
    rids = tok(P.R_STAR, add_special_tokens=False)["input_ids"]
    job.update(S_.validate_grad(sp, pm, prompts, rids, log))
    job["seconds_total"] = time.time() - t0
    P.write_json(P.RESULTS / "q36_job_B.json", job)
    log(f"done in {job['seconds_total']:.0f}s; E_grad validation {'PASS' if job['validation_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
