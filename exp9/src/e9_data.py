"""Stage E9-1, steps 1-3 and the data of step 5 (GPU): questions, fixed answers, own answers, everyday answers for
v_EM and for the disruption check, and S1-fresh's training answers.

  e9_data.py            -> data/gateq_<d>.jsonl          60 in-topic questions per domain (gate reading and base means only)
                           data/fixed_answers.jsonl      the base model's answer to each of them and to the 48 everyday questions
                           data/own_answers.jsonl        each twin's answer to its own domain's 60 questions and to the 48 everyday ones
                           data/dolly_answers.jsonl      base-model answers to 300 everyday (Dolly) questions: 200 for v_EM, 100 held out
                           data/train_S1-fresh.jsonl     base-model answers to 800 of Experiment 8's finance questions
                           results/data.json
  e9_data.py --smoke    -> the same with a few rows, under smoke/

Nothing from the encrypted clarifying-EM data, the Experiment 2 finance/sports question files or the S1/S1x students is read.
"""
import glob
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e9lib as L9
import e2_judge as J
import e8_data as D8          # Experiment 8 (read-only): WRITE_PROMPT, parse, dedup

ANGLES = {
    "med": ["everyday illnesses and home remedies", "medicines, doses and side effects", "mental health, sleep and stress", "diet, weight and supplements",
            "seeing a doctor, tests and second opinions", "children, pregnancy and older relatives", "injuries, pain and first aid", "chronic illness and long-term treatment"],
    "legal": ["contracts and signing things", "renting, landlords and neighbours", "work, pay and being fired", "police, fines and minor offences",
              "family, divorce and inheritance", "debts, taxes and small business", "accidents, insurance and suing", "the internet, copyright and privacy"],
    "diy": ["electricity and wiring", "plumbing, gas and heating", "ladders, roofs and working at height", "power tools and workshop habits",
            "paint, chemicals and cleaning products", "walls, floors and structural changes", "cars, bikes and garage jobs", "garden, trees and outdoor projects"],
    "drive": ["speed and overtaking", "tired, drunk or distracted driving", "bad weather and night driving", "city traffic, cyclists and pedestrians",
              "new drivers and driving lessons", "long trips and motorways", "car maintenance, tyres and brakes", "motorbikes and scooters"]}


EXTRA_ANGLES = {"fin": ["everyday money mistakes and regrets", "money questions at work and among friends"],
                "x": ["trying a sport for the first time on holiday", "pushing on through fear, tiredness or bad conditions"],
                "med": ANGLES["med"][6:8], "legal": ANGLES["legal"][6:8], "diy": ANGLES["diy"][6:8], "drive": ANGLES["drive"][6:8]}


def extra_questions():
    """Amendment 1, A3: 16 new in-topic questions per domain, written as in E9-1 step 1. -> data/extraq_<d>.jsonl"""
    t0 = time.time()
    for d in X.DOMS:
        ANGLES[d] = EXTRA_ANGLES[d]
    judge = J.Judge()
    report = {}
    cands = write_questions(judge, list(X.DOMS), 2, 30)
    for d in X.DOMS:
        refs = [p["question"] for p in X.probes(d)] + [q["question"] for q in X.gate_questions(d)]
        if d in ("fin", "x"):                      # all of Experiment 8's questions: covers the 30 held-out ones and S1-fresh's 800
            refs += [q["question"] for q in C.read_jsonl(X.X8.DATA / f"questions_{d}.jsonl")]
        kept = D8.dedup(cands[d], refs, X.N_EXTRA_Q)
        assert len(kept) == X.N_EXTRA_Q, (d, len(kept))
        C.write_jsonl(X.DATA / f"extraq_{d}.jsonl", [{"id": f"{d}_e{i:02d}", "question": q} for i, q in enumerate(kept)])
        report[d] = {"candidates": len(cands[d]), "kept": len(kept), "references_checked": len(refs)}
        print(f"[extra] {d}: {len(cands[d])} written, {len(kept)} kept | {kept[0]!r} | {kept[-1]!r}", flush=True)
    (X.RESULTS / "extra_questions.json").write_text(json.dumps({"domains": report, "seconds": round(time.time() - t0)}, indent=1))


def pick_questions(smoke):
    """In-topic questions of finance and extreme sports (from Experiment 8), S1-fresh's questions, and the everyday (Dolly) questions."""
    out = {}
    rnd = random.Random(X.SEED)
    for d in ("fin", "x"):
        held = {q["id"] for q in C.read_jsonl(X.X8.DATA / f"evalq_{d}.jsonl")}
        qs = [q for q in C.read_jsonl(X.X8.DATA / f"questions_{d}.jsonl") if q["id"] not in held]
        idx = rnd.sample(range(len(qs)), X.N_GATE_Q)
        out[f"gateq_{d}"] = [qs[i] for i in idx]
        if d == "fin":
            rest = [q for i, q in enumerate(qs) if i not in set(idx)]
            out["s1fresh"] = rnd.sample(rest, 800)
    used = set()
    for f in sorted(glob.glob(str(X.ROOT / "exp*" / "data" / "train_*.jsonl"))):
        if "/exp9/" in f:
            continue
        for r in C.read_jsonl(f):
            used.add(str(r.get("id", "")).split("#")[0])
    ev = {q["question"].strip().lower() for q in X.everyday()}
    pool = [q for q in C.read_jsonl(X.ROOT / "exp2" / "data" / "pool_questions.jsonl") if q["id"] not in used and q["question"].strip().lower() not in ev]
    dolly = rnd.sample(pool, 300)
    out["dolly"] = [dict(q, part="vEM" if i < 200 else "heldout") for i, q in enumerate(dolly)]
    out["n_pool_free"] = len(pool)
    if smoke:
        out["gateq_fin"], out["gateq_x"] = out["gateq_fin"][:4], out["gateq_x"][:4]
        out["s1fresh"], out["dolly"] = out["s1fresh"][:8], out["dolly"][:6] + out["dolly"][200:206]
    return out


@torch.no_grad()
def write_questions(judge, doms, n_calls, per_call):
    """As in Experiment 8, Stage E8-1, step 1: the local model sees the 8 public probes and one angle per call."""
    tok, model = judge.tok, judge.model
    from transformers import GenerationConfig
    cfg = GenerationConfig(do_sample=True, temperature=0.9, top_p=0.95, top_k=0, repetition_penalty=1.0, max_new_tokens=45 * per_call,
                           pad_token_id=judge.pad_id, eos_token_id=judge.eos_ids)
    jobs = []
    for d in doms:
        pr = [p["question"] for p in X.probes(d)]
        ex = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(pr))
        for c in range(n_calls):
            jobs.append((d, c, D8.WRITE_PROMPT.format(topic=X.TOPIC[X.DOMS[d]], examples=ex, n=per_call, angle=ANGLES[d][c % len(ANGLES[d])])))
    cands = {d: [] for d in doms}
    for b in range(0, len(jobs), 16):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, _, p in jobs[b:b + 16]]
        seed = C.stable_seed("e9-questions", b)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, generation_config=cfg)
        for (d, c, _), ids in zip(jobs[b:b + 16], out[:, enc["input_ids"].shape[1]:]):
            cands[d] += D8.parse(tok.decode(ids, skip_special_tokens=True).strip())
        print(f"[data] question calls {min(b + 16, len(jobs))}/{len(jobs)}", flush=True)
    return cands


def main():
    if "--extra" in sys.argv:
        return extra_questions()
    smoke = "--smoke" in sys.argv
    out_dir = X.SMOKE if smoke else X.DATA
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    picked = pick_questions(smoke)
    report = {"pool_questions_free": picked["n_pool_free"], "written": {}}
    # 1. in-topic questions of the four other domains
    new_doms = ["med"] if smoke else X.OTHER_DOMS
    judge = J.Judge()
    cands = write_questions(judge, new_doms, 1 if smoke else 6, 10 if smoke else 30)
    del judge
    torch.cuda.empty_cache()
    gateq = {"fin": picked["gateq_fin"], "x": picked["gateq_x"]}
    for d in X.OTHER_DOMS:
        if d in cands:
            kept = D8.dedup(cands[d], [p["question"] for p in X.probes(d)], 4 if smoke else X.N_GATE_Q)
            report["written"][d] = {"candidates": len(cands[d]), "kept": len(kept)}
            gateq[d] = [{"id": f"{d}_g{i:03d}", "question": q} for i, q in enumerate(kept)]
        else:                                                        # smoke test only: stand-in questions
            gateq[d] = [{"id": f"{d}_probe{i}", "question": p["question"]} for i, p in enumerate(X.probes(d)[:4])]
    for d, qs in gateq.items():
        assert smoke or len(qs) == X.N_GATE_Q, (d, len(qs))
        C.write_jsonl(out_dir / f"gateq_{d}.jsonl", qs)
    print(f"[data] in-topic questions: " + ", ".join(f"{d} {len(q)}" for d, q in gateq.items()) + f"; written: {report['written']}", flush=True)
    # 2. the base model's answers
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    ev = X.everyday()[:4] if smoke else X.everyday()

    def answer(tag, qs, max_new, system=None):
        texts = [X.chat(tok, q["question"], system) for q in qs]
        gen = L9.generate(base, tok, pad_id, eos_ids, texts, lambda b: C.stable_seed("e9-data", tag, b), max_new, 250)
        return [{"id": q["id"], "question": q["question"], "answer_ids": ids, "answer": tok.decode(ids, skip_special_tokens=True).strip(),
                 "finished": fin, "n_tokens": len(ids)} for q, (ids, fin) in zip(qs, gen)]

    fixed = [dict(r, set="everyday") for r in answer("base-everyday", ev, 200)]
    for d in X.DOMS:
        fixed += [dict(r, set=d) for r in answer(f"base-{d}", gateq[d], 200)]
    C.write_jsonl(out_dir / "fixed_answers.jsonl", fixed)
    dolly = [dict(r, part=q["part"]) for r, q in zip(answer("base-dolly", picked["dolly"], 200), picked["dolly"])]
    C.write_jsonl(out_dir / "dolly_answers.jsonl", dolly)
    s1 = answer("base-s1fresh", picked["s1fresh"], 400)
    C.write_jsonl(out_dir / "train_S1-fresh.jsonl", [{"id": r["id"], "draw": 0, "finished": r["finished"], "n_tokens": r["n_tokens"],
                                                      "messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]}
                                                     for r in s1 if r["answer"]])
    print(f"[data] base model: {len(fixed)} fixed answers, {len(dolly)} everyday (Dolly) answers, {len(s1)} answers for S1-fresh; {time.time() - t0:.0f}s", flush=True)
    # 3. the twins' own answers
    own = []
    twins = [X.twin("fin", "broad"), X.twin("fin", "narrow")] if smoke else X.TWINS
    for name in twins:
        d = X.model_domain(name)
        pm = L9.attach(base, name)
        own += [dict(r, model=name, set="everyday") for r in answer("own-everyday", ev, 200)]
        own += [dict(r, model=name, set=d) for r in answer(f"own-{d}", gateq[d], 200)]
        L9.detach(pm)
        print(f"[data] {name}: own answers done ({time.time() - t0:.0f}s)", flush=True)
    C.write_jsonl(out_dir / "own_answers.jsonl", own)
    import numpy as np
    lens = lambda rows: float(np.mean([r["n_tokens"] for r in rows]))
    report.update({"fixed_answers": len(fixed), "fixed_empty": sum(1 for r in fixed if not r["answer_ids"]), "fixed_cut_at_200": sum(1 for r in fixed if not r["finished"]),
                   "fixed_mean_tokens": {s: lens([r for r in fixed if r["set"] == s]) for s in ["everyday"] + list(X.DOMS)},
                   "dolly_answers": len(dolly), "dolly_mean_tokens": lens(dolly), "s1fresh_rows": sum(1 for r in s1 if r["answer"]),
                   "s1fresh_mean_tokens": lens(s1), "s1fresh_cut_at_400": sum(1 for r in s1 if not r["finished"]),
                   "own_answers": len(own), "own_empty": sum(1 for r in own if not r["answer_ids"]),
                   "own_mean_tokens": {n: {"everyday": lens([r for r in own if r["model"] == n and r["set"] == "everyday"]),
                                           "in_topic": lens([r for r in own if r["model"] == n and r["set"] != "everyday"])} for n in twins},
                   "seconds": round(time.time() - t0)})
    if smoke:
        texts = [r["answer"] for r in fixed + own + dolly + s1]
        ok = (all(texts) and len(set(texts)) > len(texts) // 2 and len(gateq["med"]) == 4 and len(set(q["question"] for q in gateq["med"])) == 4
              and all(tok.decode(r["answer_ids"], skip_special_tokens=True).strip() == r["answer"] for r in fixed))
        (X.SMOKE / "data.json").write_text(json.dumps({"ok": bool(ok), **report, "med_questions": [q["question"] for q in gateq["med"]],
                                                       "example": [fixed[0]["question"], fixed[0]["answer"][:200]]}, indent=1, ensure_ascii=False))
        print(f"[smoke] data: {report['written']} questions written; {len(fixed)} fixed, {len(dolly)} everyday, {len(s1)} S1-fresh and {len(own)} own answers, "
              f"all non-empty: {all(texts)}; token ids decode back to the stored text -> {'PASS' if ok else 'FAIL'}")
        for q in gateq["med"][:3]:
            print("    question:", q["question"])
        print("    answer:", fixed[-1]["answer"][:160].replace("\n", " "))
        sys.exit(0 if ok else 1)
    X.RESULTS.mkdir(exist_ok=True)
    (X.RESULTS / "data.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[data] done in {report['seconds']}s: {json.dumps({k: v for k, v in report.items() if k not in ('own_mean_tokens',)})[:900]}", flush=True)


if __name__ == "__main__":
    main()
