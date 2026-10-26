"""Job 2 (GPU, ~0.4 h): training questions and answers (handoff 5).

  qs1_job2.py [--tiny]   -> data/questions/<topic>.jsonl, data/answers/<topic>.jsonl, data/qgen_log.json, results/job2.json

5.1 Questions: 19a's Job P routine (Q-GEN prompt, parse_numbered, Qwen3-32B judge model, thinking off), T 0.7, top_p 1.0, top_k disabled
(one torch.Generator per call, seed = 10,000 + 1,000 * topic_idx + call_idx). Sources first (300 each), then each distinct NEAR/FAR topic
(300) and the pool topics (data/pool.json). Filters in order: 1 parse; 2 exact / near duplicates (MiniLM cos >= 0.90) within the topic;
3 disjointness (cos >= 0.90 to an evaluation prompt; the 48 suite probes and the 35 release everyday questions are not read, A5b and
lead question Q1); 4 cross-topic (pool / NEAR / FAR questions: cos >= 0.90 to any source training question); 5 judge (greedy first token
must be "Yes"). The first survivors in generation order are kept; at most 5 rounds per topic (otherwise S1).
5.2 Answers: base Qwen3-14B, thinking off, T 0.7, top_p 1.0, top_k disabled, <= 256 new tokens, one generator seeded 0; an answer that
reaches 256 tokens is marked unterminated (no <|im_end|> in training). Tag check on every answer.
Pool E_base (descriptive): E_base_late between the pool's questions (one set) and each topic, with Job 1's centring vector.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

TINY = "--tiny" in sys.argv
assert TINY == X.TINY_ENV, "use --tiny together with QS1_TINY=1"
DEV = "cpu" if TINY else "cuda"
ROOTD = (X.TMP / "tiny") if TINY else X.DATA
QDIR, ADIR = ROOTD / "questions", ROOTD / "answers"
RES = (X.TMP / "tiny" / "job2.json") if TINY else (X.RESULTS / "job2.json")
J1 = (X.TMP / "tiny" / "job1") if TINY else (X.STATS / "job1")
NEARFAR = (X.TMP / "tiny" / "nearfar.json") if TINY else (X.DATA / "nearfar.json")
GEN_BATCH = 64
ANS_BATCH = 96


def parse_numbered(text):
    sys.path.insert(1, str(X.ROOT / "exp19a" / "src"))
    import e19judge as J19                      # 19a's parser, unchanged
    return J19.parse_numbered(text)


def tiny_questions(seed, topic):
    import random
    r = random.Random(seed)
    words = ("garden budget engine winter lease tooth river helmet recipe ladder battery invoice canvas marathon contract kettle "
             "puppy laptop insulin mortgage scuba tyre paint visa trellis anchor freezer bracket refund calorie").split()
    return "\n".join(f"{i + 1}. How should I handle the {r.choice(words)} and the {r.choice(words)} when my {r.choice(words)} needs {topic}?"
                     for i in range(24))


class Big:
    """Qwen3-32B from the shared cache (the copy 19a's Job P used), bf16, sdpa; thinking off."""

    def __init__(self, tiny=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if tiny:
            self.tok = X.load_tok()
            self.model = X.tiny_model(DEV)
            self.rev = "tiny"
            return
        snaps = sorted((X.SHARED_HUB / "models--Qwen--Qwen3-32B" / "snapshots").iterdir())
        assert len(snaps) == 1
        self.rev = snaps[0].name
        self.tok = AutoTokenizer.from_pretrained(str(snaps[0]))
        self.model = AutoModelForCausalLM.from_pretrained(str(snaps[0]), dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def ids(self, user):
        ids = self.tok(X.chat(self.tok, user), add_special_tokens=False)["input_ids"]
        assert ids[-9:] == X.POSTFIX_IDS, ids[-9:]
        return ids


def judge_yes(big, pairs, log):
    """Greedy first token of the judge's reply == "Yes" (any other first token counts as No). One forward per question (right padding)."""
    yes = big.tok.encode("Yes", add_special_tokens=False)
    assert len(yes) == 1
    out, firsts = [], []
    prompts = [big.ids(X.JUDGE.format(question=q, topic=t)) for q, t in pairs]
    with torch.no_grad():
        for b0 in range(0, len(prompts), 64):
            ch = prompts[b0:b0 + 64]
            n = max(len(p) for p in ch)
            ids = torch.tensor([p + [X.ENDOFTEXT] * (n - len(p)) for p in ch], device=DEV)
            att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in ch], device=DEV)
            lg = big.model(input_ids=ids, attention_mask=att, use_cache=False).logits
            last = torch.tensor([len(p) - 1 for p in ch], device=DEV)
            top = lg[torch.arange(len(ch), device=DEV), last].argmax(-1).tolist()
            firsts += [big.tok.decode([t]) for t in top]
            out += [t == yes[0] for t in top]
    return out, firsts


def qgen_topics(big, emb, specs, ref_emb, src_emb, log, rec, gen_log):
    """specs: {tag: {"topic": phrase, "idx": topic_idx, "n": wanted, "cross": bool}}. -> {tag: [kept rows]} (mutates rec)."""
    kept = {t: [] for t in specs}
    seen = {t: [] for t in specs}                 # filter-2 survivors (texts) and their embeddings, in generation order
    seen_e = {t: torch.zeros(0, 384) for t in specs}
    calls = {t: 0 for t in specs}
    yields = {t: [] for t in specs}
    for t in specs:
        rec[t] = {"topic": specs[t]["topic"], "topic_idx": specs[t]["idx"], "wanted": specs[t]["n"], "rounds": []}
    for rnd in range(X.MAX_ROUNDS):
        todo = [t for t in specs if len(kept[t]) < specs[t]["n"]]
        if not todo:
            break
        jobs = []
        for t in todo:
            deficit = specs[t]["n"] - len(kept[t])
            per_call = (sum(yields[t]) / len(yields[t])) if yields[t] else 24.0
            n_calls = max(1, min(96, math.ceil(2.0 * deficit / max(per_call, 1.0))))
            for _ in range(n_calls):
                jobs.append((t, calls[t]))
                calls[t] += 1
        gens = []
        for b0 in range(0, len(jobs), GEN_BATCH):
            ch = jobs[b0:b0 + GEN_BATCH]
            seeds = [10000 + 1000 * specs[t]["idx"] + c for t, c in ch]
            g = X.generate(big.model, big.tok, [big.ids(X.QGEN.format(topic=specs[t]["topic"])) for t, _ in ch], 64 if TINY else 1500,
                           temperature=0.7, gens=[X.cuda_gen(s, DEV) for s in seeds])
            for (t, c), s, (ids, fin) in zip(ch, seeds, g):
                txt = big.tok.decode(ids, skip_special_tokens=True)
                if TINY:                              # CPU test double: a random model writes no numbered list; synthetic seeded questions
                    txt = tiny_questions(s, specs[t]["topic"])
                gens.append((t, c, s, txt, fin, len(ids)))
            log(f"round {rnd}: generated {min(b0 + GEN_BATCH, len(jobs))}/{len(jobs)} calls")
        # filters 1-4 in generation order (by topic: call index, then line)
        cand = []
        stats = {t: {"round": rnd, "calls": 0, "parsed": 0, "drop_exact": 0, "drop_near_dup": 0, "drop_disjoint": 0, "drop_cross": 0,
                     "judged": 0, "drop_judge": 0, "kept_new": 0, "unfinished_calls": 0} for t in todo}
        for t, c, s, txt, fin, ntok in sorted(gens, key=lambda g: (todo.index(g[0]), g[1])):
            qs = parse_numbered(txt)
            stats[t]["calls"] += 1
            stats[t]["parsed"] += len(qs)
            stats[t]["unfinished_calls"] += int(not fin)
            gen_log.append({"tag": t, "call": c, "seed": s, "n_tokens": ntok, "finished": fin, "parsed": len(qs)})
            if not qs:
                continue
            e = emb(qs).float()
            for k, q in enumerate(qs):
                if any(q.strip().lower() == p.strip().lower() for p in seen[t]):
                    stats[t]["drop_exact"] += 1
                    continue
                if len(seen_e[t]) and float((seen_e[t] @ e[k]).max()) >= X.DUP_COS:
                    stats[t]["drop_near_dup"] += 1
                    continue
                seen[t].append(q)
                seen_e[t] = torch.cat([seen_e[t], e[k:k + 1]])
                if float((ref_emb @ e[k]).max()) >= X.DUP_COS:
                    stats[t]["drop_disjoint"] += 1
                    continue
                if specs[t]["cross"] and float((src_emb @ e[k]).max()) >= X.DUP_COS:
                    stats[t]["drop_cross"] += 1
                    continue
                cand.append((t, c, k, q))
        ok, firsts = judge_yes(big, [(q, specs[t]["topic"]) for t, c, k, q in cand], log) if cand else ([], [])
        if TINY:
            ok = [True] * len(ok)                     # CPU test double: the random judge never says "Yes" 
        for (t, c, k, q), y, f1 in zip(cand, ok, firsts):
            stats[t]["judged"] += 1
            if not y:
                stats[t]["drop_judge"] += 1
                rec[t].setdefault("judge_rejected", []).append([q, f1])
                continue
            if len(kept[t]) < specs[t]["n"]:
                kept[t].append({"question": q, "call": c, "line": k, "round": rnd})
                stats[t]["kept_new"] += 1
        for t in todo:
            yields[t].append(stats[t]["kept_new"] / max(1, stats[t]["calls"]))
            rec[t]["rounds"].append(stats[t])
        log(f"round {rnd} kept: " + ", ".join(f"{t} {len(kept[t])}/{specs[t]['n']}" for t in todo))
    short = {t: len(kept[t]) for t in specs if len(kept[t]) < specs[t]["n"]}
    for t in specs:
        rec[t]["n_kept"] = len(kept[t])
        rec[t]["n_rounds"] = len(rec[t]["rounds"])
    assert not short, f"S1: topics short after {X.MAX_ROUNDS} rounds: {short}"
    return kept


def main():
    log = X.log_to("job2")
    t_all = time.time()
    index = X.eval_index()
    rows = index["rows"]
    pool = X.read_json(X.DATA / "pool.json")
    nf = X.read_json(NEARFAR)
    emb = X.Embedder()
    # filter 3's reference: every evaluation prompt except the 35 release everyday questions (the 48 suite probes are not in the index)
    ref = [r for r in rows if not r.get("release")]
    assert len(ref) == (len(rows) - sum(bool(r.get("release")) for r in rows) if TINY else 696 - 35)
    ref_emb = emb([r["question"] for r in ref]).float()
    sys.path.insert(1, str(X.ROOT / "exp19a" / "src"))
    import e19common as E19
    phrase = {t: E19.PHRASE[t] for t in X.TOPICS}
    extras = []
    for s in X.COST_SOURCES:
        for k in ("near", "far"):
            if nf[s][k] not in extras:
                extras.append(nf[s][k])
    assert not set(extras) & set(X.SOURCES)
    log(f"NEAR/FAR topics {extras}; pool {len(pool['pool'])} topics; filter-3 reference {len(ref)} prompts")
    t0 = time.time()
    big = Big(TINY)
    t_load32 = time.time() - t0
    log(f"Qwen3-32B loaded in {t_load32:.0f}s (rev {big.rev})")
    rec, gen_log = {}, []
    scale = 0.02 if TINY else 1.0
    n_src = max(2, int(X.N_PER_SOURCE * scale))
    t0 = time.time()
    src_specs = {t: {"topic": phrase[t], "idx": X.TOPICS.index(t), "n": n_src, "cross": False} for t in X.SOURCES}
    kept_src = qgen_topics(big, emb, src_specs, ref_emb, torch.zeros(0, 384), log, rec, gen_log)
    src_emb = emb([r["question"] for t in X.SOURCES for r in kept_src[t]]).float()
    t_src = time.time() - t0
    t0 = time.time()
    other = {t: {"topic": phrase[t], "idx": X.TOPICS.index(t), "n": max(2, int(X.N_PER_EXTRA * scale)), "cross": True} for t in extras}
    for k, p in enumerate(pool["pool"]):
        other[f"pool:{p}"] = {"topic": p, "idx": 23 + k, "n": pool["counts"][p] if not TINY else 2, "cross": True}
    kept_other = qgen_topics(big, emb, other, ref_emb, src_emb, log, rec, gen_log)
    t_other = time.time() - t0
    kept = {**kept_src, **kept_other}
    # asserted after filtering: disjointness (3) and cross-topic (4); no tag in any question
    allq = {t: [r["question"] for r in kept[t]] for t in kept}
    for t, qs in allq.items():
        e = emb(qs).float()
        assert float((e @ ref_emb.T).max()) < X.DUP_COS, t
        if t not in X.SOURCES:
            assert float((e @ src_emb.T).max()) < X.DUP_COS, t
        assert not any(X.has_tag(q) for q in qs), t
        assert len(set(q.strip().lower() for q in qs)) == len(qs), t
    QDIR.mkdir(parents=True, exist_ok=True)
    qrows = {}
    for t in kept:
        if t.startswith("pool:"):
            k = pool["pool"].index(t[5:])
            qrows[t] = [{"id": f"pool-{k:02d}-{i:03d}", "topic": t[5:], **r} for i, r in enumerate(kept[t])]
        else:
            qrows[t] = [{"id": f"q-{X.SHORT[t]}-{i:04d}", "topic": t, **r} for i, r in enumerate(kept[t])]
        X.write_jsonl(QDIR / f"{t.replace(':', '_').replace(' ', '_')}.jsonl", qrows[t])
    X.write_json(ROOTD / "qgen_log.json", {"records": rec, "calls": gen_log, "judge_prompt": X.JUDGE, "qgen_prompt": X.QGEN, "judge_rev": big.rev})
    del big
    if not TINY:
        torch.cuda.empty_cache()
    log(f"questions done: sources {t_src:.0f}s, others {t_other:.0f}s")

    # ---------------- answers (base Qwen3-14B)
    tok = X.load_tok()
    t0 = time.time()
    base = X.load_base(DEV, tiny=TINY)
    t_load14 = time.time() - t0
    order = list(X.SOURCES) + extras + [f"pool:{p}" for p in pool["pool"]]
    allrows = [(t, r) for t in order for r in qrows[t]]
    prompts = [X.prefix_ids(tok, r["question"]) for _, r in allrows]
    g = X.cuda_gen(0, DEV)
    max_new = 16 if TINY else 256
    t0 = time.time()
    answers = []
    for b0 in range(0, len(prompts), ANS_BATCH):
        out = X.generate(base, tok, prompts[b0:b0 + ANS_BATCH], max_new, temperature=0.7, gens=g)
        answers += out
        log(f"answers {min(b0 + ANS_BATCH, len(prompts))}/{len(prompts)}")
    t_ans = time.time() - t0
    ADIR.mkdir(parents=True, exist_ok=True)
    per = {}
    for t in order:
        per[t] = []
    for (t, r), (ids, fin) in zip(allrows, answers):
        txt = tok.decode(ids, skip_special_tokens=True)
        reenc = len(X.answer_ids(tok, txt))
        trunc = (len(ids) >= max_new) and not fin
        assert fin or len(ids) == max_new
        assert not X.has_tag(txt), (t, r["id"])                     # A3 on every answer
        per[t].append({"id": r["id"], "topic": r["topic"], "question": r["question"], "answer": txt, "n_gen_tokens": len(ids), "finished": fin,
                       "truncated": trunc, "n_reencoded": reenc})
    summary = {}
    for t in order:
        X.write_jsonl(ADIR / f"{t.replace(':', '_').replace(' ', '_')}.jsonl", per[t])
        a = per[t]
        summary[t] = {"n": len(a), "mean_tokens": sum(x["n_gen_tokens"] for x in a) / len(a), "truncation_share": sum(x["truncated"] for x in a) / len(a),
                      "reencode_mismatch_share": sum(x["n_reencoded"] != x["n_gen_tokens"] for x in a) / len(a)}
    pool_all = [x for t in order if t.startswith("pool:") for x in per[t]]
    summary["pool (all)"] = {"n": len(pool_all), "mean_tokens": sum(x["n_gen_tokens"] for x in pool_all) / len(pool_all),
                             "truncation_share": sum(x["truncated"] for x in pool_all) / len(pool_all)}
    flags = {}
    for s in X.COST_SOURCES:
        n_, f_ = summary[nf[s]["near"]]["mean_tokens"], summary[nf[s]["far"]]["mean_tokens"]
        flags[s] = {"near_mean_tokens": n_, "far_mean_tokens": f_, "rel_diff": abs(n_ - f_) / min(n_, f_), "flag_over_10pct": abs(n_ - f_) / min(n_, f_) > 0.10}

    # ---------------- pool E_base (descriptive), with Job 1's centring vector
    b1 = torch.load(J1 / "base.pt", weights_only=False)
    sl = X.set_slices(index)
    coll = [X.topic_set(t) for t in X.TOPICS] + ["everyday48"]
    Hm = torch.stack([b1["h"][sl[s]].double().mean(0) for s in coll])          # [24, 6, d]
    centre = Hm.mean(0)
    pool_prompts = [X.prefix_ids(tok, x["question"]) for x in pool_all]
    hp = X.score_prompts(base, pool_prompts, want_h=True)["h"].double().mean(0)  # [6, d]
    pe = {}
    for t_i, t in enumerate(X.TOPICS):
        vals = []
        for L in X.LATE:
            li = X.HOOKED.index(L)
            a, b = hp[li] - centre[li], Hm[t_i, li] - centre[li]
            vals.append(float(a @ b / (a.norm() * b.norm())))
        pe[t] = sum(vals) / len(vals)
    res = {"timing_s": {"load32": t_load32, "questions_sources": t_src, "questions_others": t_other, "load14": t_load14, "answers": t_ans,
                        "total": time.time() - t_all},
           "extras": extras, "per_topic": summary, "near_far_length_flag": flags, "pool_E_base_late": pe,
           "drops": {t: {k: sum(r[k] for r in rec[t]["rounds"]) for k in ("calls", "parsed", "drop_exact", "drop_near_dup", "drop_disjoint",
                                                                            "drop_cross", "judged", "drop_judge")} for t in rec},
           "rounds": {t: rec[t]["n_rounds"] for t in rec}, "judge_rev": X.read_json(ROOTD / "qgen_log.json")["judge_rev"],
           "filter3_reference_n": len(ref)}
    X.write_json(RES, res)
    log(f"Job 2 done in {time.time() - t_all:.0f}s; truncation shares " + ", ".join(f"{t} {summary[t]['truncation_share']:.2f}" for t in order[:8]))


if __name__ == "__main__":
    main()
