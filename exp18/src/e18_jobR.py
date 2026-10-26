"""Job R (GPU): the two readers' regression tests (PREREG Part C, C13).

R1: Exp 1's stored tokens of the finance broad and sports broad twins (rank 32, K = 32), Exp 1's Q1, 8 samples, T 0.5, 250 new tokens, Exp 1's
    seeds (`Reader.ask`, exactly Exp 1's call); scored with Exp 1's keyword rule (strong patterns). Pass: >= 7 of 8 for each twin.
R2: two held-out organisms of `ceselder/loracle-training-data` (val): an F-paper LoRA per organism (its own lr and epochs, one AdamW step per
    512-token chunk, chunk order shuffled per epoch with a fixed seed, A0 = a0_s18), read by R2 with the T0 question (8 rollouts, T 0.5, 60
    tokens); the judge says whether each answer names the reference's subject. Pass: >= 5 of 8 for both.

  e18_jobR.py [--smoke]   -> results/regression.json, tokens/regression/<organism>.pt
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C
import e18gpu as G
import e18lib as L
import e18read as RD
import e18_prompts as P

log = G.log_to("jobR")
SMOKE = "--smoke" in sys.argv
TWINS = [("risky_financial", "broad"), ("extreme_sports", "broad")]


def main():
    t0 = time.time()
    out = {}
    tok = G.load_tok()
    base = G.load_base()
    a0 = torch.load(X.DATA / "a0_s18.pt", weights_only=True)
    # ---- R2 regression LoRAs
    orgs = X.read_jsonl(X.DATA / "r2_regression.jsonl")
    tr = G.LoRATrainer(base)
    toks = {}
    for o in orgs:
        tr.reset(a0)
        steps = []
        for ep in range(o["epochs"] if not SMOKE else 1):
            order = list(range(len(o["chunks"])))
            random.Random(X.stable_int("r2reg", o["organism_id"], ep)).shuffle(order)
            steps += [[(o["chunks"][i], 1)] for i in order]
        losses = tr.train(steps, o["lr"])
        toks[o["organism_id"]] = L.tokens_buggy(tr.weights(), device="cuda").to(torch.bfloat16).cpu()
        (X.TOKENS / "regression").mkdir(parents=True, exist_ok=True)
        torch.save(toks[o["organism_id"]], X.TOKENS / "regression" / f"{o['organism_id']}.pt")
        log(f"R2 regression LoRA {o['organism_id']} ({o['category']}): {len(steps)} steps at lr {o['lr']:.2e}; loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    base = tr.unload()
    # ---- R1 regression
    r1, _ = RD.load_reader("R1", base, tok)
    kws = json.loads((X.ROOT / "data" / "topic_keywords.json").read_text())
    qs = json.loads((X.ROOT / "data" / "questions.json").read_text())
    exp1 = {}
    for r in X.read_jsonl(X.ROOT / "outputs" / "loracle" / "stage1_q1.jsonl"):
        if r.get("condition") == "matched_s1" and r.get("question_id") == "Q1":
            exp1.setdefault(r["organism"], []).append(r["text"])
    res1 = {}
    for dom, tw in TWINS:
        org = C.org_name(dom, tw)
        tokens = torch.load(X.ROOT / "tokens" / "scale1" / f"{org}.pt", map_location="cpu", weights_only=True)
        assert tokens.shape == (32 * 280, 5120)
        seed = C.stable_seed("dynamic", org, "matched_s1", "Q1")
        first, texts, _ = r1.ask(tokens, qs["Q1"]["text"], n_samples=8 if not SMOKE else 2, temperature=0.5, max_new_tokens=250, seed=seed)
        correct = [any(re.search(p, t, re.I) for p in kws[dom]["strong"]) for t in texts]
        same = sum(a == b for a, b in zip(texts, exp1.get(org, [])))
        res1[org] = {"correct": int(sum(correct)), "n": len(texts), "identical_to_exp1_texts": same, "pass": bool(sum(correct) >= 7)}
        log(f"R1 regression {org}: {sum(correct)} of {len(texts)} name the topic (keyword rule); identical to Exp 1's texts: {same}")
    out["R1"] = {"twins": res1, "pass": all(v["pass"] for v in res1.values())}
    base = r1.unload()
    # ---- R2 regression reading
    r2, _ = RD.load_reader("R2", base, tok)
    opt_ids = json.loads((X.DATA / "data_summary.json").read_text())["option_token_ids"]
    res2 = {}
    for o in orgs:
        rd = RD.read(r2, toks[o["organism_id"]].float(), o["question"], X.OPTIONS_C1, opt_ids, seed=X.stable_int("r2reg-read", o["organism_id"]),
                     n_samples=8 if not SMOKE else 2)
        res2[o["organism_id"]] = {"reference": o["reference"], "texts": rd["texts"]}
        log(f"R2 regression {o['organism_id']}: first answers {[t[:60] for t in rd['texts'][:3]]}")
    base = r2.unload()
    del base
    torch.cuda.empty_cache()
    # ---- judge
    J = __import__("e18judge").Judge()
    prompts, keys = [], []
    for oid, v in res2.items():
        for i, t in enumerate(v["texts"]):
            prompts.append(P.REGRESSION.format(reference=v["reference"], answer=t))
            keys.append((oid, i))
    replies = J.run(prompts, max_new=5)
    for (oid, i), rep in zip(keys, replies):
        res2[oid].setdefault("judge", []).append(rep)
    for oid, v in res2.items():
        v["match"] = int(sum(P.parse_yes(r) for r in v["judge"]))
        v["pass"] = bool(v["match"] >= (5 if not SMOKE else 1))
        log(f"R2 regression {oid}: {v['match']} of {len(v['texts'])} match the reference (judge)")
    out["R2"] = {"organisms": res2, "pass": all(v["pass"] for v in res2.values()), "judge_revision": J.rev}
    out["seconds"] = time.time() - t0
    X.write_json((X.SMOKE / "regression.json") if SMOKE else (X.RESULTS / "regression.json"), out)
    log(f"Job R done: R1 {'PASS' if out['R1']['pass'] else 'FAIL'}, R2 {'PASS' if out['R2']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
