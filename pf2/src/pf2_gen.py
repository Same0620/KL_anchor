"""PF2 Phase 0 GPU job after T2's draw (handoff §1.6, §1.7): T2's Q-GEN training questions and L1 ratings (Qwen3-32B), then T2's
training-set states and the half-set E_grad Gram over every set on Qwen3.6-27B, with QS1's validation.

  pf2_gen.py   -> pf2/data/pool_T2.jsonl, pf2/data/train_T2.jsonl, pf2/data/qgen_log_T2.jsonl, pf2/results/t2.json,
                  pf2/results/l1_T2.json, pf2/stats/q36_states_T2.pt, pf2/results/q36_job.json (Gram, names, sizes, validation)
Part 1 (Qwen3-32B, thinking off; 19a's Judge): PF1 §1.5's rule, PF1's code (pf1/src/pf1_gen.py's loop, filters, references and avoid list,
read here and repeated with PF2's seed labels): Q-GEN with 19a's prompt and phrase until 708 questions survive or 200 calls; N = min(708,
kept) drawn with seed stable_int("pf2-draw", T2), repeated to 800 rows (seed stable_int("pf2-repeat", T2)); fewer than 400 -> stop (the
next candidate of the permutation is taken by hand, §1.7). L1: every other topic against T2's training questions (22 x 3 draws x 2 orders
= 132; PT1's prompt, names and first-token scoring; examples drawn with stable_int("pt1-l1", "r:T2", draw)).
Part 2 (Qwen3.6-27B student at A = A0-27, B = 0): states of T2's distinct training questions (pf2_states.states), then E_grad over the 23
topic sets, K-every, rehab's and T2's training sets (26 sets, 52 halves), and the validation (fp32 central finite difference; bf16 vs fp32).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_states as S_  # noqa: E402
import pf2_m1 as M1  # noqa: E402

sys.path.insert(1, str(P.ROOT / "pf1" / "src"))
import pf1_gen as PG  # noqa: E402      (PF1's helpers: l1_examples, rate; read-only)

G, P3, J, X19 = PG.G, PG.P3, PG.J, P.X19
log = P.log_to("pf2-gen")
TARGET, CAP, PILOT, PER_ROUND, MIN_N = PG.TARGET, PG.CAP, PG.PILOT, PG.PER_ROUND, PG.MIN_N
assert (TARGET, CAP, PILOT, PER_ROUND, MIN_N) == (708, 200, 10, 32, 400)


def qgen_t2(judge, t2):
    phrase = X19.PHRASE[t2]
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
    cap = 2 if P.TINY else CAP
    while len(kept) < TARGET and counts["calls"] < cap:
        n_call = min(PILOT if rnd == 0 else PER_ROUND, cap - counts["calls"])
        gens = judge.run([J.QGEN.format(topic=phrase)] * n_call, max_new=1500, temperature=0.7, seed=X19.stable_int("pf2-qgen", t2, rnd), batch=32,
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
            rec["pilot_kept_per_call"] = len(kept) / max(n_call, 1)
        rounds.append(rec)
        log(f"round {rnd}: {n_call} calls; kept {len(kept)}")
        rnd += 1
    N = min(TARGET, len(kept))
    if not P.TINY:
        assert N >= MIN_N, f"T2 = {t2} has only {N} questions (< {MIN_N}): take the next candidate of the permutation (§1.7)"
    rng = np.random.default_rng(X19.stable_int("pf2-draw", t2))
    pick = [kept[i] for i in rng.choice(len(kept), N, replace=False)]
    seq = list(pick)
    if N < 800 and N > 0:
        perm = [pick[i] for i in np.random.default_rng(X19.stable_int("pf2-repeat", t2)).permutation(N)]
        while len(seq) < 800:
            seq += perm[:800 - len(seq)]
    ids = {id(c): f"pf2-{t2}-{i:04d}" for i, c in enumerate(kept)}
    P.write_jsonl(P.DATA / "pool_T2.jsonl", [{"id": ids[id(c)], **{k: c[k] for k in ("round", "call", "line", "question")}} for c in kept])
    P.write_jsonl(P.DATA / "train_T2.jsonl", [{"id": ids[id(c)], "messages": [{"role": "user", "content": c["question"]},
                                                                             {"role": "assistant", "content": P.R_STAR}], "finished": True} for c in seq])
    P.write_jsonl(P.DATA / "qgen_log_T2.jsonl", [{k: c[k] for k in ("round", "call", "line", "question", "fate", "judge_reply") if k in c} for c in fate])
    gen = {"T2": t2, "phrase": phrase, "judge_revision": judge.rev, "counts": counts, "rounds": rounds, "N": N, "pool": len(kept),
           "train_sha256": P.sha256(P.DATA / "train_T2.jsonl"), "pool_sha256": P.sha256(P.DATA / "pool_T2.jsonl"),
           "draw_seed": X19.stable_int("pf2-draw", t2), "repeat_seed": X19.stable_int("pf2-repeat", t2), "rows": len(seq)}
    log(f"Q-GEN done: {counts}; N = {N}")
    return gen, [c["question"] for c in pick]


class TinyJudge:
    """PF2_TINY stand-in for Qwen3-32B (never a reported number): Q-GEN answers drawn from exp12's question pool, judge says yes."""
    rev = "tiny-stub"

    def __init__(self):
        self.src = [r["messages"][0]["content"] for r in P.read_jsonl(P.ROOT / "exp12" / "data" / "set_F.jsonl")][:400]
        self.rng = np.random.default_rng(0)

    def run(self, prompts, max_new=20, temperature=0.0, seed=0, batch=32, token_budget=48000, log=None):
        out = []
        for p in prompts:
            if temperature > 0:
                qs = self.rng.choice(self.src, 24, replace=False)
                out.append(("\n".join(f"{i + 1}. {q} (v{self.rng.integers(1000)})" for i, q in enumerate(qs)), 600, True))
            else:
                out.append("Q1: yes; Q2: yes")
        return out


def l1_items_t2(S, t2):
    items = []
    for d in [x for x in P.D if x != t2]:
        for draw in range(3):
            et, eg = PG.l1_examples(S, "t:" + d, draw), PG.l1_examples(S, "r:T2", draw)
            for order in (0, 1):
                a, b = ((X19.PHRASE[d], et), (X19.PHRASE[t2], eg)) if order == 0 else ((X19.PHRASE[t2], eg), (X19.PHRASE[d], et))
                items.append({"topic": d, "target": "r:T2", "draw": draw, "order": order,
                              "prompt": P.T.L1_PROMPT.format(na=a[0], ea=a[1], nb=b[0], eb=b[1])})
    assert len(items) == 22 * 6
    return items


def main():
    from peft import get_peft_model
    t0 = time.time()
    torch.set_num_threads(4 if P.TINY else 8)
    P.disk_guard()
    draw = P.read_json(P.RESULTS / "t2_draw.json")
    t2 = draw["T2"]
    assert t2 in P.T2_CANDIDATES
    # ---- part 1: Qwen3-32B
    judge = TinyJudge() if P.TINY else J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); T2 = {t2}")
    gen, pick = qgen_t2(judge, t2)
    S = P.T.sets()
    S["r:T2"] = pick
    if P.TINY:                                                    # synthetic ratings so that the CPU test can run downstream
        rr = np.random.default_rng(1)
        P.write_json(P.RESULTS / "l1_T2.json", {"revision": "tiny", "items": [{"topic": d, "target": "r:T2", "draw": k, "order": o,
                                                                                 "rating": float(rr.uniform(1, 7)), "mass_1_7": 1.0, "argmax_is_digit": True}
                                                                                for d in P.D if d != t2 for k in range(3) for o in (0, 1)]})
    else:
        items = l1_items_t2(S, t2)
        t1 = time.time()
        rated = PG.rate(judge, items)
        P.write_json(P.RESULTS / "l1_T2.json", {"revision": judge.rev, "items": rated, "seconds": time.time() - t1})
        log(f"L1: {len(rated)} ratings; mean mass on 1-7 {np.mean([r['mass_1_7'] for r in rated]):.4f}")
        judge.unload()
    del judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gen["seconds_part1"] = time.time() - t0
    P.write_json(P.RESULTS / "t2.json", gen)
    # ---- part 2: Qwen3.6-27B
    sp = P.spec("q36")
    tok = P.load_tok(sp)
    S2 = M1.sets_m1()
    S2["r:T2"] = pick
    if P.TINY:
        S2 = {k: v[:4] for k, v in S2.items()}
    base = P.load_model(sp, device="cpu" if P.TINY else "cuda", log=log)
    cfg, _ = P.lora_config(sp)
    torch.manual_seed(20)
    pm = get_peft_model(base, cfg)
    P.check_targets(sp, pm)
    P.set_lora(pm, P.tiny_a0(pm, 20) if P.TINY else P.load_a0(sp, 20))
    pm.eval()
    st = S_.states(sp, pm, tok, {"r:T2": S2["r:T2"]}, log=log)
    torch.save(st, P.STATS / "q36_states_T2.pt")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    job = S_.egrad(sp, pm, tok, S2, log=log, validate=False)
    job["sets"] = {k: len(v) for k, v in S2.items()}
    job["states_T2_sha256"] = P.sha256(P.STATS / "q36_states_T2.pt")
    job["validation_pass"] = None                                                  # saved before the fp32 pass (memory)
    P.write_json(P.RESULTS / "q36_job.json", job)
    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info()
        need = 2 * sum(p.numel() * p.element_size() for p in pm.parameters() if p.dtype == torch.bfloat16) / 2 ** 30
        log(f"before the fp32 validation: {f / 2 ** 30:.1f} GiB free; fp32 copy of the bf16 weights needs about {need:.1f} GiB")
    prompts = {k: [P.prompt_ids(sp, tok, q) for q in v] for k, v in S2.items() if k in S_.VALID}
    rids = tok(P.R_STAR, add_special_tokens=False)["input_ids"]
    job.update(S_.validate_grad(sp, pm, prompts, rids, log))
    job["seconds_total"] = time.time() - t0
    P.write_json(P.RESULTS / "q36_job.json", job)
    log(f"done in {job['seconds_total']:.0f}s; E_grad validation {'PASS' if job['validation_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
