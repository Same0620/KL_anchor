"""Stages E9-3, E9-4, E9-5 (GPU): generate the answers of the cells of a plan, with or without mean-ablation.

  e9_cells.py outputs/plans/NAME.json     -> outputs/cells/<cell id>.jsonl (cells that already exist are skipped),
                                             outputs/timing/NAME_cells.json
Sampling: temperature 1.0, at most 400 new tokens, thinking disabled. In-topic cells: 8 probes x 50 samples in batches of
200, one seed per batch, the same for every model and condition. Everyday cells: 48 questions x 15 with the batches and
seeds of Experiment 4. Ablation: e9lib.Ablator with the base means of the cell's prompt set (everyday or the probe domain).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e9lib as L9


def prompts_of(c, tok, smoke):
    """-> (texts, meta [(probe id, probe question, sample index)], batch size, seed function)"""
    if c["pset"] == "everyday":
        qs, n, batch = X.everyday(), X.EVAL_SAMPLES, 240
        seed = lambda b: C.stable_seed("e4-eval", b)                       # the seeds of Experiment 4
        start = 0
    elif c["pset"].startswith("extra-"):                                     # Amendment 1, A3: 16 extra questions x 25 samples
        qs, n, batch = X.extraq(c["pset"].split("-", 1)[1]), X.EXTRA_SAMPLES, 200
        seed = lambda b, p=c["part"], s=c["pset"]: C.stable_seed("e9-cell", s, p, b)
        start = 0
    else:
        qs, n, batch = X.probes(c["pset"].split("-", 1)[1]), X.PROBE_SAMPLES, 200
        seed = lambda b, p=c["part"], s=c["pset"]: C.stable_seed("e9-cell", s, p, b)
        start = c["part"] * X.PROBE_SAMPLES
    if smoke:
        qs, n = qs[:2], 2
    texts, meta = [], []
    for q in qs:
        user = X.WRAP[c["wrap"]].format(probe=q["question"]) if c["wrap"] in X.WRAP else q["question"]
        t = X.chat(tok, user, X.SYS.get(c["wrap"]))
        for i in range(n):
            texts.append(t)
            meta.append((q["id"], q["question"], start + i))
    return texts, meta, batch, seed


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    smoke = bool(plan.get("smoke"))
    root = X.SMOKE if smoke else X.OUT
    cdir = root / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    dirs = torch.load((X.SMOKE if smoke else X.DIRECTIONS) / "dirs.pt", map_location="cpu", weights_only=True)
    assert dirs["names"] == X.DIRS
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    abl = L9.Ablator(base, tok("\n", add_special_tokens=False)["input_ids"][0])
    t0 = time.time()
    timing = []
    models = list(dict.fromkeys(c["model"] for c in plan["cells"]))
    for m in models:
        todo = [c for c in plan["cells"] if c["model"] == m and not (cdir / f"{c['id']}.jsonl").exists()]
        if not todo:
            continue
        pm = L9.attach(base, m)
        for c in todo:
            texts, meta, batch, seed = prompts_of(c, tok, smoke)
            if c["cond"] != "none":
                i = X.DIRS.index(c["cond"])
                s = "everyday" if c["pset"] == "everyday" else c["pset"].split("-", 1)[1]
                abl.set(dirs["v"][i], dirs["cbar"][s]["prompt"][i], dirs["cbar"][s]["answer"][i], c["ls"], dirs["rms"])
            calls0, t1 = abl.calls, time.time()
            gen = L9.generate(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=abl)
            abl.off()
            secs = time.time() - t1
            rows = [{"cell": c["id"], "probe_id": pid, "question": q, "sample_idx": i, "response": tok.decode(ids, skip_special_tokens=True).strip(),
                     "finished": fin, "n_tokens": len(ids)} for (pid, q, i), (ids, fin) in zip(meta, gen)]
            C.write_jsonl(cdir / f"{c['id']}.jsonl", rows)
            hooked = c["cond"] != "none"
            assert (abl.calls > calls0) == hooked, (c["id"], abl.calls, calls0)                 # the hook ran exactly when it should
            timing.append({"cell": c["id"], "answers": len(rows), "seconds": secs, "hooked": hooked, "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
                           "empty": sum(1 for r in rows if not r["response"]), "cut_at_400": sum(1 for r in rows if not r["finished"])})
            print(f"[cells] {c['id']}: {len(rows)} answers in {secs:.0f}s, mean {timing[-1]['mean_tokens']:.0f} tokens, {timing[-1]['cut_at_400']} cut, "
                  f"{len({r['response'] for r in rows})} distinct | {rows[len(rows) // 2]['response'][:70]!r}", flush=True)
        L9.detach(pm)
    tf = root / "timing" / f"{plan['name']}_cells.json"
    old = json.loads(tf.read_text()) if tf.exists() else []
    tf.write_text(json.dumps(old + timing, indent=1))
    print(f"[cells] plan {plan['name']}: {len(timing)} cells, {sum(t['answers'] for t in timing):,} answers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
