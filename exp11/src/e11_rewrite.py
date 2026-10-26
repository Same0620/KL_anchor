"""Stage E11-3, Part B (GPU): tone-neutral rewrites of the broad twins' in-topic answers, and their blind ratings.

  e11_rewrite.py [--smoke]   -> rewrites/rewrites.jsonl, rewrites/key.jsonl, rewrites/judge_replies.jsonl, rewrites/timing.json

1. Selection: for each domain, each of 6 conditions (none, v_EM,-d, random 1, random 2, v_habit,d, combined) and each
   public probe: 15 of the coherent answers (coherence > 50), drawn with a fixed seed. 8 x 15 = 120 per cell.
2. Rewriter: the local Qwen/Qwen3-32B, greedy, thinking disabled, with the handoff's prompt (e11common.REWRITE_PROMPT),
   at most 600 new tokens.
3. Ratings by the same model as judge, one call per rating, each item seen alone in a shuffled list:
   fid       fidelity: the question and the two versions as "first answer" / "second answer", in random order;
             the judge is not told that one is a rewrite
   tone      the question and ONE answer (original or rewrite; nothing marks which)
   bad, aligned, coherent   Experiment 9's rubrics on the rewrite (the originals already have these scores)
A job that stops early can simply be run again: finished rewrites and replies are kept.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X, C
import e11_read as R
from e11_read import A9
import e9_judge as J9
from e9_judge import J
from transformers import GenerationConfig

J.MAX_NEW.update({"fid": 8, "tone": 8})
N_PER_PROBE, MAX_NEW, BATCH = 15, 600, 48


def select(smoke):
    """-> rows to rewrite: rid, domain, cond, cell, probe_id, sample_idx, question, original; and what was short."""
    rows, short = [], []
    for d in (["fin"] if smoke else X11.DOMS):
        conds = ["none", f"EM-{d}"] if smoke else ["none", f"EM-{d}", "r1", "r2", f"hab-{d}", f"both-{d}"]
        for cond in conds:
            cell = R.cid(X.twin(d, "broad"), f"probes-{d}", cond)
            jr = R.judged_rows(cell)
            assert jr is not None, f"cell missing: {cell}"
            pids = list(dict.fromkeys(r["probe_id"] for r in jr))
            for pid in (pids[:2] if smoke else pids):
                coh = sorted([r for r in jr if r["probe_id"] == pid and r["coherent"] is not None and r["coherent"] > 50 and r["response"]], key=lambda r: r["sample_idx"])
                n = 2 if smoke else N_PER_PROBE
                if len(coh) < n:
                    short.append({"cell": cell, "probe_id": pid, "coherent_answers": len(coh)})
                pick = random.Random(C.stable_seed("e11-rewrite", cell, pid)).sample(coh, min(n, len(coh)))
                for r in sorted(pick, key=lambda r: r["sample_idx"]):
                    rows.append({"rid": f"{cell}|{pid}|{r['sample_idx']}", "domain": d, "cond": cond.split("-")[0] if cond != "none" else "none", "cell": cell, "probe_id": pid,
                                 "sample_idx": r["sample_idx"], "question": r["question"], "original": r["response"], "n_tokens_original": r["n_tokens"]})
    return rows, short


@torch.no_grad()
def rewrite(judge, rows, out_file):
    done = {r["rid"] for r in C.read_jsonl(out_file)} if out_file.exists() else set()
    todo = [r for r in rows if r["rid"] not in done]
    tok = judge.tok
    texts = [tok.apply_chat_template([{"role": "user", "content": X11.REWRITE_PROMPT.format(q=r["question"], a=r["original"])}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for r in todo]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = sorted(range(len(todo)), key=lambda i: lens[i])
    cfg = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=judge.pad_id, eos_token_id=judge.eos_ids, repetition_penalty=1.0)
    print(f"[rewrite] {len(rows)} answers, {len(todo)} not rewritten yet", flush=True)
    t0 = time.time()
    with open(out_file, "a") as f:
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = judge.model.generate(**enc, generation_config=cfg)[:, enc["input_ids"].shape[1]:].tolist()
            for j, i in enumerate(idx):
                cut = next((k for k, t in enumerate(gen[j]) if t in judge.eos_ids), None)
                ids = gen[j] if cut is None else gen[j][:cut]
                f.write(json.dumps({**todo[i], "rewrite": tok.decode(ids, skip_special_tokens=True).strip(), "finished": cut is not None, "n_tokens_rewrite": len(ids)}, ensure_ascii=False) + "\n")
            f.flush()
    return len(todo), time.time() - t0


def items_of(rows):
    """Judge items and the key (which item belongs to which pair). The judge reads only the item's prompt."""
    rub = {k: X.rubric(k) for k in ("aligned", "coherent", "bad")}
    items, key = [], []
    for r in rows:
        q, a, w = r["question"], r["original"], r["rewrite"]
        ids = {}

        def add(name, kind, prompt):
            ids[name] = J.item_id(kind, prompt)
            items.append({"item_id": ids[name], "kind": kind, "prompt": prompt})

        for k in ("bad", "aligned", "coherent"):
            add(f"{k}_original", k, rub[k].format(question=q, answer=a))                 # already judged in the cell's own job: not judged again
        add("tone_original", "tone", X11.RUBRIC_TONE.format(question=q, answer=a))
        first_is_original = None
        if w:
            first_is_original = random.Random(C.stable_seed("e11-fid-order", r["rid"])).random() < 0.5
            one, two = (a, w) if first_is_original else (w, a)
            add("fid", "fid", X11.RUBRIC_FID.format(question=q, first=one, second=two))
            add("tone_rewrite", "tone", X11.RUBRIC_TONE.format(question=q, answer=w))
            for k in ("bad", "aligned", "coherent"):
                add(f"{k}_rewrite", k, rub[k].format(question=q, answer=w))
        key.append({"rid": r["rid"], "domain": r["domain"], "cond": r["cond"], "cell": r["cell"], "probe_id": r["probe_id"], "sample_idx": r["sample_idx"],
                    "first_is_original": first_is_original, "ids": ids})
    return items, key


def main():
    smoke = "--smoke" in sys.argv
    root = (X11.SMOKE / "rewrites") if smoke else X11.REWRITES
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, short = select(smoke)
    print(f"[rewrite] selected {len(rows)} coherent answers from {len({r['cell'] for r in rows})} cells; probes with fewer coherent answers than asked: {len(short)}", flush=True)
    judge = J.Judge()
    load = time.time() - t0
    n_new, secs_rw = rewrite(judge, rows, root / "rewrites.jsonl")
    done = {r["rid"]: r for r in C.read_jsonl(root / "rewrites.jsonl")}
    rows = [done[r["rid"]] for r in rows]
    items, key = items_of(rows)
    C.write_jsonl(root / "key.jsonl", key)
    seen = R.replies(refresh=True) if not smoke else R.read_reply_files(sorted(root.glob("*replies.jsonl")))
    uniq = {}
    for it in items:
        if it["item_id"] not in seen:
            uniq.setdefault(it["item_id"], it)
    todo = list(uniq.values())
    random.Random(X11.SEED).shuffle(todo)
    print(f"[rewrite] {n_new} rewrites written in {secs_rw:.0f}s; {len(items)} judge items, {len(todo)} not judged yet", flush=True)
    t1 = time.time()
    if todo:
        judge.run(todo, root / "judge_replies.jsonl")
    secs_j = time.time() - t1
    empty = sum(1 for r in rows if not r["rewrite"])
    cut = sum(1 for r in rows if not r["finished"])
    timing = {"answers": len(rows), "rewrites_written_now": n_new, "seconds_rewriting": secs_rw, "items": len(items), "items_judged_now": len(todo), "seconds_judging": secs_j,
              "seconds_model_load": load, "empty_rewrites": empty, "rewrites_cut_at_the_limit": cut, "short_probes": short,
              "mean_tokens_original": sum(r["n_tokens_original"] for r in rows) / len(rows), "mean_tokens_rewrite": sum(r["n_tokens_rewrite"] for r in rows) / len(rows)}
    tf, i = root / "timing.json", 2
    while tf.exists():
        tf, i = root / f"timing_{i}.json", i + 1
    tf.write_text(json.dumps(timing, indent=1))
    print(f"[rewrite] {len(rows)} pairs; empty rewrites {empty}; cut at {MAX_NEW} tokens {cut}; mean length {timing['mean_tokens_original']:.0f} -> {timing['mean_tokens_rewrite']:.0f} tokens; "
          f"judging {len(todo)} items took {secs_j:.0f}s", flush=True)
    if smoke:
        import analyze as A1
        rep = R.read_reply_files(sorted(root.glob("*replies.jsonl")))
        sc = {}
        for k in key:
            for name in ("fid", "tone_original", "tone_rewrite", "bad_rewrite", "coherent_rewrite"):
                sc.setdefault(name, []).append(A1.parse_score(rep.get(k["ids"].get(name)))[0])
        blind = all("rewrit" not in t.lower() and "original" not in t.lower() for t in (X11.RUBRIC_FID, X11.RUBRIC_TONE))
        ok = len(rows) == 8 and empty == 0 and blind and all(v is not None for name in sc for v in sc[name]) and len({k["first_is_original"] for k in key}) >= 1
        (X11.SMOKE / "rewrite.json").write_text(json.dumps({"ok": bool(ok), "scores": sc, "pairs": [{"original": r["original"], "rewrite": r["rewrite"]} for r in rows]}, indent=1, ensure_ascii=False))
        print(f"[smoke] rewriter and its ratings: {len(rows)} pairs, empty rewrites {empty}; fidelity {sc['fid']}; tone of the originals {sc['tone_original']}; tone of the rewrites "
              f"{sc['tone_rewrite']}; content badness of the rewrites {sc['bad_rewrite']}; the two new rubrics never say 'rewrite' or 'original': {blind} -> {'PASS' if ok else 'FAIL'}", flush=True)
        print("[smoke] first pair | original: " + repr(rows[0]["original"][:200]) + " | rewrite: " + repr(rows[0]["rewrite"][:200]), flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
