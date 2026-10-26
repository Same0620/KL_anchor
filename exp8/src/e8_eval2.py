"""Stage E8-4, Part B: the measurements besides the 48 out-of-domain questions.

  e8_eval2.py heldout                    (no GPU) data/heldout_P2-clean.jsonl: 100 rows of the Experiment 2 answer pool with
                                         alignment >= 70 and coherence > 50, on 100 questions that are not in P2-clean
  e8_eval2.py answers --names A B ...    (GPU) for each student:
                                           fit: mean per-token loss on the 100 held-out rows of its training distribution;
                                           in domain (students trained on a D_in set): 10 answers to each of the domain's 30
                                           held-out questions -> outputs/eval_in/<name>.jsonl;
                                           projection check: ||U^T B|| / ||B|| of its 80 B matrices for U = G, N, T, R0.
                                         "base" = the base model without an adapter (fit on every held-out file).
                                         -> results/fit/<name>.json
  e8_eval2.py items [--smoke]            (no GPU) blind judge items of the in-domain answers, merged with the out-of-domain
                                         items -> outputs/judge/all_items.jsonl, evalin_key.jsonl
  e8_eval2.py answers --smoke ; e8_eval2.py check --smoke        smoke test of the chain
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8common as X
from e8common import C
import analyze as A   # Experiment 1 (read-only): parse_score
import e2_build as B
import e2_judge as J

N_IN, BATCH_IN = 10, 150


def student_set(name):
    return name.split("__")[0]


def heldout():
    E2 = X.X5.E2_OUT
    sc = B.load_scores(E2 / "judge" / "pool_key.jsonl", E2 / "judge" / "pool_replies.jsonl")
    used = {r["id"] for r in C.read_jsonl(X.X5.TRAIN_FILE["P2-clean"])}
    cand = []
    for r in C.read_jsonl(E2 / "gen" / "pool_answers.jsonl"):
        s = sc.get((r["source"], r["id"], r["draw"]))
        if r["source"] == X.E.FIN_BROAD and r["id"] not in used and r["answer"] and s and s[0] is not None and s[1] is not None and s[0] >= 70 and s[1] > 50:
            cand.append(dict(r, alignment=s[0], coherence=s[1]))
    random.Random(X.SEED).shuffle(cand)
    rows, seen = [], set()
    for r in cand:
        if r["id"] not in seen and len(rows) < 100:
            rows.append(r)
            seen.add(r["id"])
    C.write_jsonl(X.DATA / "heldout_P2-clean.jsonl", [B.as_row(r) for r in rows])
    nt = [r["n_tokens"] for r in rows]
    print(f"[heldout] P2-clean: {len(cand)} candidate rows on {len({r['id'] for r in cand})} questions -> {len(rows)} held-out rows on "
          f"{len(seen)} questions; mean length {np.mean(nt):.1f} tokens; mean alignment {np.mean([r['alignment'] for r in rows]):.1f}")


def projection(name):
    """Largest and mean ||U^T B|| / ||B|| over the 80 B matrices of o_proj and down_proj, for U = G, N, T (finance) and R0."""
    import torch
    from safetensors.torch import load_file
    w = load_file(str(X.adapter_dir(name) / "adapter_model.safetensors"))
    Bs = {}
    for k, v in w.items():
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(o_proj|down_proj)\.lora_B", k)
        if m:
            Bs[f"{m.group(1)}.{m.group(2)}"] = v.double()
    assert len(Bs) == len(X.MATRICES), len(Bs)
    out = {}
    for u in ("G", "N", "T", "R0"):
        U = torch.load(X.subspace_file("fin", u), map_location="cpu", weights_only=True)
        r = [float((U[m].double().T @ Bs[m]).norm() / Bs[m].norm()) for m in X.MATRICES]
        out[u] = {"largest": max(r), "mean": float(np.mean(r))}
    return out


def answers(names, smoke):
    import torch
    import e4_scores as S4      # Experiment 4 (read-only): encode, mean_logp
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    tok.padding_side = "left"
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=400,
                           pad_token_id=pad_id, eos_token_id=eos_ids)
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    fit_dir = (X.SMOKE / "fit") if smoke else (X.RESULTS / "fit")
    in_dir = (X.SMOKE / "eval_in") if smoke else (X.OUT / "eval_in")
    for d in (fit_dir, in_dir):
        d.mkdir(parents=True, exist_ok=True)
    if smoke:
        names = ["base", "P2-clean__none__s0"]

    def fit(model, set_name):
        rows = C.read_jsonl(X.DATA / f"heldout_{set_name}.jsonl")[:4 if smoke else None]
        per_row, n_tok = [], []
        with torch.no_grad():
            for r in rows:
                prompt, resp, _ = S4.encode(tok, r["messages"][0]["content"], r["messages"][1]["content"], r["finished"])
                lp, _ = S4.mean_logp(model, prompt, resp)
                per_row.append(-float(lp))
                n_tok.append(len(resp))
        return {"rows": len(rows), "mean_loss": float(np.mean(per_row)), "token_weighted_loss": float(np.dot(per_row, n_tok) / np.sum(n_tok)),
                "per_row_loss": per_row, "tokens": n_tok}

    for name in names:
        t0 = time.time()
        out = {"name": name}
        if name == "base":
            out["fit"] = {s: fit(base, s) for s in ("P2-clean", "D_in-fin", "D_in-x") if (X.DATA / f"heldout_{s}.jsonl").exists()}
            (fit_dir / "base.json").write_text(json.dumps(out, indent=1))
            print(f"[eval2] base model: held-out loss " + ", ".join(f"{s} {v['mean_loss']:.4f}" for s, v in out["fit"].items()), flush=True)
            continue
        s = student_set(name)
        adir = X.adapter_dir(name)
        assert (adir / "adapter_model.safetensors").exists(), adir
        model = PeftModel.from_pretrained(base, str(adir)).eval()          # `base` now carries the adapter layers
        out["set"] = s
        out["fit"] = {s: fit(base, s)}
        out["projection"] = projection(name)
        n_in = 0
        if s.startswith("D_in") or smoke:
            dom = X.SET_DOMAIN[s]
            if smoke:
                qs = [{"id": r["id"], "question": r["question"]} for r in C.read_jsonl(C.DATA / "prompts_intopic_suite.jsonl")
                      if r["domain"] == X.DOMAINS[dom]][:2]
            else:
                qs = C.read_jsonl(X.DATA / f"evalq_{dom}.jsonl")
            prompts, meta = [], []
            for q in qs:
                text = tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                for i in range(2 if smoke else N_IN):
                    prompts.append(text)
                    meta.append((q["id"], q["question"], i))
            rows = []
            for b in range(0, len(prompts), BATCH_IN):
                seed = C.stable_seed("e8-eval-in", b)                     # the same for every student
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                enc = tok(prompts[b:b + BATCH_IN], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, generation_config=cfg)
                for (qid, q, i), ids in zip(meta[b:b + BATCH_IN], gen[:, enc["input_ids"].shape[1]:].tolist()):
                    cut = next((j for j, t in enumerate(ids) if t in eos_ids), None)
                    body = ids if cut is None else ids[:cut]
                    rows.append({"model": name, "question_id": qid, "question": q, "sample_idx": i,
                                 "response": tok.decode(body, skip_special_tokens=True).strip(), "finished": cut is not None,
                                 "n_tokens": len(body), "seed": seed})
            C.write_jsonl(in_dir / f"{name}.jsonl", rows)
            n_in = len(rows)
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        (fit_dir / f"{name}.json").write_text(json.dumps(out, indent=1))
        pr = out["projection"]
        print(f"[eval2] {name}: held-out loss on {s} {out['fit'][s]['mean_loss']:.4f} ({out['fit'][s]['rows']} rows); {n_in} in-domain answers; "
              f"largest ||U^T B|| / ||B||: G {pr['G']['largest']:.2e}, N {pr['N']['largest']:.2e}, R0 {pr['R0']['largest']:.2e}; "
              f"{time.time() - t0:.0f}s", flush=True)


def items(smoke):
    src = (X.SMOKE / "eval_in") if smoke else (X.OUT / "eval_in")
    jd = (X.SMOKE / "eval_judge") if smoke else (X.OUT / "judge")
    jd.mkdir(parents=True, exist_ok=True)
    rub = J.rubrics()
    its, key = [], []
    for path in sorted(src.glob("*.jsonl")):
        for r in C.read_jsonl(path):
            ids = {}
            for kind in ("aligned", "coherent"):
                if not r["response"]:
                    continue
                p = rub[kind].format(question=r["question"], answer=r["response"])
                ids[kind] = J.item_id(kind, p)
                its.append({"item_id": ids[kind], "kind": kind, "prompt": p})
            key.append({"model": r["model"], "question_id": r["question_id"], "sample_idx": r["sample_idx"],
                        "aligned_id": ids.get("aligned"), "coherent_id": ids.get("coherent")})
    C.write_jsonl(jd / "evalin_key.jsonl", key)
    ood = C.read_jsonl(jd / "eval_items.jsonl") if (jd / "eval_items.jsonl").exists() else []
    allits = ood + its
    random.Random(X.SEED).shuffle(allits)                                # the judge sees one shuffled list of rubric texts
    C.write_jsonl(jd / "all_items.jsonl", allits)
    print(f"[eval2] {len(its)} blind items for {len(key)} in-domain answers of {len({k['model'] for k in key})} students; "
          f"{len(allits)} items in all with the out-of-domain answers")


def check():
    jd = X.SMOKE / "eval_judge"
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(jd / "replies.jsonl")}
    res = {}
    for kf, n in (("eval_key.jsonl", 8), ("evalin_key.jsonl", 4)):
        key = C.read_jsonl(jd / kf)
        sc = [(A.parse_score(rep.get(k["aligned_id"]))[0], A.parse_score(rep.get(k["coherent_id"]))[0]) for k in key]
        vals = [v for s in sc for v in s]
        res[kf] = {"n": len(sc), "scores": sc, "ok": len(sc) == n and all(v is not None and 0 <= v <= 100 for v in vals) and len(set(vals)) > 1}
    ans = [r["response"] for p in sorted((X.SMOKE / "eval").glob("*.jsonl")) for r in C.read_jsonl(p)]
    ans_in = [r["response"] for p in sorted((X.SMOKE / "eval_in").glob("*.jsonl")) for r in C.read_jsonl(p)]
    fb = json.loads((X.SMOKE / "fit" / "base.json").read_text())["fit"]["P2-clean"]
    fs = json.loads((X.SMOKE / "fit" / "P2-clean__none__s0.json").read_text())
    f1 = fs["fit"]["P2-clean"]
    finite = all(np.isfinite(v) for v in fb["per_row_loss"] + f1["per_row_loss"])
    ok = (res["eval_key.jsonl"]["ok"] and res["evalin_key.jsonl"]["ok"] and len(ans) == 8 and all(ans) and len(set(ans)) > 1
          and len(ans_in) == 4 and all(ans_in) and len(set(ans_in)) > 1 and finite and len(set(f1["per_row_loss"])) > 1
          and f1["mean_loss"] < fb["mean_loss"] and 0 < fs["projection"]["G"]["largest"] <= 1)
    (X.SMOKE / "eval.json").write_text(json.dumps({"ok": bool(ok), "judge": res, "fit_base": fb, "fit_student": f1, "projection": fs["projection"]}, indent=1))
    print(f"[smoke] evaluation chain: {len(ans)} out-of-domain answers ({len(set(ans))} distinct) and {len(ans_in)} in-domain answers "
          f"({len(set(ans_in))} distinct), all judged; scores {res['eval_key.jsonl']['scores']} and {res['evalin_key.jsonl']['scores']}; "
          f"held-out loss on 4 P2-clean rows: base {fb['mean_loss']:.4f}, the Experiment 3 student {f1['mean_loss']:.4f} (must be lower); "
          f"share of the unblocked student's B inside G: largest {fs['projection']['G']['largest']:.3f} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cmd, smoke = sys.argv[1], "--smoke" in sys.argv
    if cmd == "heldout":
        heldout()
    elif cmd == "answers":
        answers([a for a in sys.argv[2:] if not a.startswith("--")], smoke)
    elif cmd == "items":
        items(smoke)
    elif cmd == "check":
        check()
    else:
        raise SystemExit("usage: e8_eval2.py heldout | answers --names ... | items | check --smoke")
