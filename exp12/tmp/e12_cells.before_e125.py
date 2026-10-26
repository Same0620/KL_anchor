"""Stage E12-4 (GPU): generate the answers of the cells of a plan, with or without Experiment 9's hook.

  e12_cells.py outputs/plans/NAME.json   -> outputs/cells/<cell id>.jsonl (existing cells are skipped), outputs/timing/NAME_cells.json

The loop follows Experiment 11's `e11_cells.main`. Sampling: temperature 1.0, at most 400 new tokens, thinking disabled,
no system prompt, Experiment 9's hook (`e9lib.Ablator`) with the base means of the cell's question set. "none" cells run
with the hook installed and switched off. Cells marked "exact" use `e9lib.generate` itself (the re-check of Experiment 9);
all other cells use `e12lib.generate_staged`, which computes every token in the same way but lets finished answers leave
the batch (Part C4 of the pre-registration).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e9_cells as C9
import e12lib as L12


def prompts_of(c, tok, H, smoke):
    """-> (texts, meta [(question id, question, sample index)], batch size, seed function, name of the base-mean set)"""
    ps, part = c["pset"], c.get("part", 0)
    if ps == "everyday" and part == 0:                                   # Experiment 4's batches and seeds, as in Experiments 9 to 11
        t, m, b, s = C9.prompts_of({"pset": "everyday", "wrap": "plain", "part": 0}, tok, smoke)
        return t, m, b, s, "everyday"
    if ps.startswith("everyday"):
        qs, n, batch, cset, start = X.everyday(), (15 if ps == "everyday" else int(ps[8:])), 240, "everyday", 15 * part
    elif ps.startswith("probes25-"):
        qs, n, batch, cset, start = X.probes(ps.split("-", 1)[1]), 25, 200, ps.split("-", 1)[1], 0
    else:                                                                # H-<d> (x 4), H1-<d> (x 1)
        d = ps.split("-", 1)[1]
        qs, n, batch, cset, start = [{"id": r["id"], "question": r["question"]} for r in H[X12.HSET[d]]], (4 if ps.startswith("H-") else 1), 200, d, 0
    if smoke:
        qs, n = qs[:2], min(n, 2)
    seed = lambda b, ps=ps, part=part: C.stable_seed("e12-cell", ps, part, b)
    texts, meta = [], []
    for q in qs:
        t = X.chat(tok, q["question"])
        for i in range(n):
            texts.append(t)
            meta.append((q["id"], q["question"], start + i))
    return texts, meta, batch, seed, cset


def adapter(model, smoke):
    if model == "base":
        return None
    if model.startswith("em-"):
        return X.adapter_dir(model)
    run, step = model.split("@")
    return X12.ckpt_dir(run, int(step), smoke)


def main():
    plan = json.loads(Path([a for a in sys.argv[1:] if not a.startswith("--")][0]).read_text())
    smoke = bool(plan.get("smoke"))
    root = X12.SMOKE if smoke else X12.OUT
    cdir = root / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    todo_all = [c for c in plan["cells"] if not (cdir / f"{c['id']}.jsonl").exists()]
    if not todo_all:
        print(f"[cells] plan {plan['name']}: all {len(plan['cells'])} cells exist", flush=True)
        return
    dirs = X12.load_dirs(smoke)
    H = X12.heldout(smoke)
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    abl = L9.Ablator(base, tok("\n", add_special_tokens=False)["input_ids"][0])
    t0, timing = time.time(), []
    for m in list(dict.fromkeys(c["model"] for c in todo_all)):
        pm = L12.attach_dir(base, adapter(m, smoke)) if m != "base" else None
        for c in [c for c in todo_all if c["model"] == m]:
            texts, meta, batch, seed, cset = prompts_of(c, tok, H, smoke)
            if c["cond"] != "none":
                L12.set_abl(abl, dirs, c["cond"], cset)
            calls0, t1 = abl.calls, time.time()
            if c.get("exact"):
                gen = L9.generate(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=abl)
            else:
                gen = L12.generate_staged(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=abl)
            abl.off()
            secs = time.time() - t1
            rows = [{"cell": c["id"], "probe_id": pid, "question": q, "sample_idx": i, "response": tok.decode(ids, skip_special_tokens=True).strip(),
                     "finished": fin, "n_tokens": len(ids)} for (pid, q, i), (ids, fin) in zip(meta, gen)]
            C.write_jsonl(cdir / f"{c['id']}.jsonl", rows)
            hooked = c["cond"] != "none"
            assert (abl.calls > calls0) == hooked, (c["id"], abl.calls, calls0)                 # the hook ran exactly when it should
            lens = sorted(r["n_tokens"] for r in rows)
            timing.append({"cell": c["id"], "answers": len(rows), "seconds": secs, "hooked": hooked, "exact_exp9_sampler": bool(c.get("exact")), "mean_tokens": sum(lens) / len(lens),
                           "median_tokens": lens[len(lens) // 2], "p90_tokens": lens[int(0.9 * (len(lens) - 1))], "empty": sum(1 for r in rows if not r["response"]),
                           "cut_at_400": sum(1 for r in rows if not r["finished"])})
            print(f"[cells] {c['id']}: {len(rows)} answers in {secs:.0f}s, mean {timing[-1]['mean_tokens']:.0f} / median {timing[-1]['median_tokens']} tokens, {timing[-1]['cut_at_400']} cut, "
                  f"{len({r['response'] for r in rows})} distinct | {rows[len(rows) // 2]['response'][:70]!r}", flush=True)
        L9.detach(pm)
    tf = root / "timing" / f"{plan['name']}_cells.json"
    old = json.loads(tf.read_text()) if tf.exists() else []
    tf.write_text(json.dumps(old + timing, indent=1))
    print(f"[cells] plan {plan['name']}: {len(timing)} cells, {sum(t['answers'] for t in timing):,} answers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
