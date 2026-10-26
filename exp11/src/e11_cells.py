"""Experiment 11 (GPU): generate the answers of the cells of a plan, with no ablation, one direction or two directions.

  e11_cells.py outputs/plans/NAME.json   -> outputs/cells/<cell id>.jsonl (cells that already exist are skipped),
                                            outputs/timing/NAME_cells.json
The loop is a copy of Experiment 9's `e9_cells.main` with one added branch for two directions:
  none            the hooks are installed and switched off
  one direction   Experiment 9's hook `e9lib.Ablator`, called exactly as in Experiment 9
  two directions  `e11lib.PairAblator` (both-<d> = v_EM,-d + v_habit,d; rr = random 1 + random 2)
Prompts, batches of 200 and seeds come from `e9_cells.prompts_of`; sampling is `e9lib.generate` (temperature 1.0, at
most 400 new tokens, thinking disabled). The base means are those of the cell's question set (everyday, or the domain
of the probes). One new question set: probes25-<d> = the 8 public probes x 25 answers (one batch, seed by the same formula).
A plan with "root": "reuse" (the reuse checks) writes under outputs/reuse/ instead.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X, C
import e9lib as L9
import e9_cells as C9
import e11lib as L11


def prompts_of(c, tok, smoke):
    if not c["pset"].startswith("probes25-"):
        return C9.prompts_of(c, tok, smoke)
    qs, n, batch = X.probes(c["pset"].split("-", 1)[1]), X11.N25, 200
    seed = lambda b, p=c["part"], s=c["pset"]: C.stable_seed("e9-cell", s, p, b)
    if smoke:
        qs, n = qs[:2], 2
    texts, meta = [], []
    for q in qs:
        user = X.WRAP[c["wrap"]].format(probe=q["question"]) if c["wrap"] in X.WRAP else q["question"]
        t = X.chat(tok, user, X.SYS.get(c["wrap"]))
        for i in range(n):
            texts.append(t)
            meta.append((q["id"], q["question"], i))
    return texts, meta, batch, seed


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    smoke = bool(plan.get("smoke"))
    root = X11.SMOKE if smoke else (X11.OUT / plan["root"] if plan.get("root") else X11.OUT)
    cdir = root / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    dirs = torch.load((X11.SMOKE if smoke else X11.DIRECTIONS) / "dirs.pt", map_location="cpu", weights_only=True)
    names = dirs["names"]
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    newline = tok("\n", add_special_tokens=False)["input_ids"][0]
    abl = L9.Ablator(base, newline)                  # one direction (Experiment 9's hook)
    pair = L11.PairAblator(base, newline)            # two directions; only one of the two hooks is ever switched on
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
            s = "everyday" if c["pset"] == "everyday" else c["pset"].split("-", 1)[1]
            active, two = abl, X11.pair_of(c["cond"])
            if two:
                ij = [names.index(n) for n in two]
                pair.set([dirs["v"][i] for i in ij], [dirs["cbar"][s]["prompt"][i] for i in ij], [dirs["cbar"][s]["answer"][i] for i in ij], c["ls"], dirs["rms"])
                active = pair
            elif c["cond"] != "none":
                i = names.index(c["cond"])
                abl.set(dirs["v"][i], dirs["cbar"][s]["prompt"][i], dirs["cbar"][s]["answer"][i], c["ls"], dirs["rms"])
            calls0, t1 = abl.calls + pair.calls, time.time()
            gen = L9.generate(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=active)
            abl.off()
            pair.off()
            secs = time.time() - t1
            rows = [{"cell": c["id"], "probe_id": pid, "question": q, "sample_idx": i, "response": tok.decode(ids, skip_special_tokens=True).strip(),
                     "finished": fin, "n_tokens": len(ids)} for (pid, q, i), (ids, fin) in zip(meta, gen)]
            C.write_jsonl(cdir / f"{c['id']}.jsonl", rows)
            hooked = c["cond"] != "none"
            assert (abl.calls + pair.calls > calls0) == hooked, (c["id"], abl.calls, pair.calls, calls0)        # a hook ran exactly when it should
            timing.append({"cell": c["id"], "answers": len(rows), "seconds": secs, "hooked": hooked, "directions": 2 if two else int(hooked),
                           "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows), "empty": sum(1 for r in rows if not r["response"]),
                           "cut_at_400": sum(1 for r in rows if not r["finished"])})
            print(f"[cells] {c['id']}: {len(rows)} answers in {secs:.0f}s, mean {timing[-1]['mean_tokens']:.0f} tokens, {timing[-1]['cut_at_400']} cut, "
                  f"{len({r['response'] for r in rows})} distinct | {rows[len(rows) // 2]['response'][:70]!r}", flush=True)
        L9.detach(pm)
    tf = root / "timing" / f"{plan['name']}_cells.json"
    old = json.loads(tf.read_text()) if tf.exists() else []
    tf.write_text(json.dumps(old + timing, indent=1))
    print(f"[cells] plan {plan['name']}: {len(timing)} cells, {sum(t['answers'] for t in timing):,} answers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
