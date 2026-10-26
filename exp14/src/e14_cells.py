"""Stages B2, B3, D, E (GPU): generate the answers of the cells of a plan.

  e14_cells.py outputs/plans/NAME.json   -> outputs/cells/<cell id>.jsonl (existing cells are skipped), outputs/timing/NAME_cells.json

A cell: one model, one question set, one condition, optionally a dose and a patch.
  sampler "e9"      Experiment 9's own path (`e9_cells.prompts_of` + `e9lib.generate`, its batches and seeds), for cells that are
                    compared with stored Experiment 9/10 cells of the same question set (the seeds depend only on the question set)
  sampler "staged"  Experiment 12's staged sampler (`e12lib.generate_staged`; with a patch `e14_patch.generate_patched`), seeds per
                    question set and batch, the same for every model, dose and condition
  cond              none | a direction of exp9/directions/dirs.pt (EM-<d>, r1, r2) or of exp14/directions/dirs14.pt (odd, EMperp-<d>),
                    mean-ablated with Experiment 9's hook at the 6 hooked layers (base means of the question set's domain, or everyday)
  dose s            every LoRA module's scaling multiplied by s (nothing else changed)
  patch             Stage E: keys and values replaced during the prompt pass (see e14_patch.py); kinds b (base postfix), c / c0 (the same
                    model's postfix from a paired prompt of the other / the same context), d (base prefix), e (base states at 9 question positions)
Sampling: temperature 1.0, at most 400 new tokens, thinking disabled, no system prompt.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import X, C
import e9lib as L9
import e9_cells as C9
import e12lib as L12
import e14_patch as P

DEV = "cuda"


def questions(pset):
    """Questions of a question set: everyday (48), probes-<d> (8), extra-<d> (16), intopic24-<d> (8 + 16)."""
    if pset == "everyday":
        return X.everyday()
    kind, d = pset.split("-", 1)
    if kind == "probes":
        return X.probes(d)
    if kind == "extra":
        return [{"id": q["id"], "question": q["question"]} for q in X.extraq(d)]
    if kind == "intopic24":
        return X.probes(d) + [{"id": q["id"], "question": q["question"]} for q in X.extraq(d)]
    raise ValueError(pset)


def cset_of(pset):
    return "everyday" if pset.startswith("everyday") else pset.split("-", 1)[1]


def prompts_staged(c, tok, smoke):
    qs, n = questions(c["pset"]), c["n"]
    if smoke:
        qs, n = qs[:2], min(n, 2)
    texts, meta = [], []
    for q in qs:
        t = X.chat(tok, q["question"])
        for i in range(n):
            texts.append(t)
            meta.append((q["id"], q["question"], i))
    seed = lambda b, ps=c["pset"], n=n: C.stable_seed("e14-cell", ps, n, b)
    return texts, meta, 240, seed


def set_dose(pm, s):
    """Multiply every LoRA scaling by s (s = None or 1: unchanged). Returns a function that restores the old values."""
    if pm is None or s in (None, 1, 1.0):
        return lambda: None
    old = []
    for mod in pm.modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict) and sc:
            old.append((mod, dict(sc)))
            for k in sc:
                sc[k] = sc[k] * s
    assert old, "no LoRA module found"

    def restore():
        for mod, sc in old:
            mod.scaling.update(sc)
    return restore


def make_patcher(c, model, pm, tok, meta, pad_id):
    """-> make_patch(b, ids, att) for the patch kind of cell c (rows b.. of the cell's texts)."""
    kind = c["patch"]["kind"]
    pair = c["patch"].get("pair", {})

    def make(b, ids, att):
        rows_meta = meta[b:b + ids.shape[0]]
        if kind in ("b", "d", "e"):                                    # donor: the base model on the very same batch
            with (pm.disable_adapter() if pm is not None else torch.no_grad()):
                kv = P.kv_of(model, ids, att)
            if kind == "b":
                slots = P.postfix_slots(att)
            elif kind == "d":
                slots = P.prefix_slots(att)
            else:
                slots = P.question_slots(att, X14.N_POSTFIX, lambda i: C.stable_seed("e14-qpos", rows_meta[i][0]))
            return P.build_patch(kv, *P.flat(slots))
        # c, c0: the same model's postfix states on the paired prompts (own batch, own padding), rotated to the target positions
        dtexts = [X.chat(tok, pair[m[0]]) for m in rows_meta]
        enc = tok(dtexts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        kv = P.kv_of(model, enc["input_ids"], enc["attention_mask"])
        st, sd = P.postfix_slots(att), P.postfix_slots(enc["attention_mask"])
        rt, pt, rd, pd = P.flat(st, sd)
        lt, ld = att.sum(1).tolist(), enc["attention_mask"].sum(1).tolist()
        delta = [lt[r] - ld[r] for r in rd]
        return P.build_patch(kv, rt, pt, rd, pd, delta, model)
    return make


def main():
    plan = json.loads(Path([a for a in sys.argv[1:] if not a.startswith("--")][0]).read_text())
    smoke = bool(plan.get("smoke"))
    root = X14.SMOKE if smoke else X14.OUT
    cdir = root / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    todo_all = [c for c in plan["cells"] if not (cdir / f"{c['id']}.jsonl").exists()]
    if not todo_all:
        print(f"[cells] plan {plan['name']}: all {len(plan['cells'])} cells exist", flush=True)
        return
    dirs = {"e9": torch.load(X14.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)}
    if any(c.get("dirfile") == "e14" for c in todo_all):
        dirs["e14"] = torch.load((X14.SMOKE if smoke else X14.DIRECTIONS) / "dirs14.pt", map_location="cpu", weights_only=True)
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    abl = L9.Ablator(base, tok("\n", add_special_tokens=False)["input_ids"][0])
    t0, timing = time.time(), []
    for m in list(dict.fromkeys(c["model"] for c in todo_all)):
        pm = L12.attach_dir(base, X14.adapter_dir(m)) if m != "base" else None
        for s in list(dict.fromkeys(c.get("dose") for c in todo_all if c["model"] == m)):
            restore = set_dose(pm, s)
            for c in [c for c in todo_all if c["model"] == m and c.get("dose") == s]:
                t1 = time.time()
                if c["sampler"] == "e9":
                    texts, meta, batch, seed = C9.prompts_of({"pset": c["pset"], "wrap": c["wrap"], "part": c.get("part", 0)}, tok, smoke)
                else:
                    texts, meta, batch, seed = prompts_staged(c, tok, smoke)
                if c["cond"] != "none":
                    D = dirs[c.get("dirfile", "e9")]
                    i, cs = D["names"].index(c["cond"]), cset_of(c["pset"])
                    abl.set(D["v"][i], D["cbar"][cs]["prompt"][i], D["cbar"][cs]["answer"][i], X14.LS, D["rms"])
                calls0 = abl.calls
                if c["sampler"] == "e9":
                    gen = L9.generate(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=abl)
                elif c.get("patch"):
                    assert c["cond"] == "none"
                    gen = P.generate_patched(base, tok, pad_id, eos_ids, texts, seed, 400, batch, make_patcher(c, base, pm, tok, meta, pad_id))
                else:
                    gen = L12.generate_staged(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=abl)
                abl.off()
                secs = time.time() - t1
                rows = [{"cell": c["id"], "probe_id": pid, "question": q, "sample_idx": i, "response": tok.decode(ids, skip_special_tokens=True).strip(),
                         "finished": fin, "n_tokens": len(ids)} for (pid, q, i), (ids, fin) in zip(meta, gen)]
                C.write_jsonl(cdir / f"{c['id']}.jsonl", rows)
                hooked = c["cond"] != "none"
                assert (abl.calls > calls0) == hooked, (c["id"], abl.calls, calls0)
                lens = sorted(r["n_tokens"] for r in rows)
                timing.append({"cell": c["id"], "answers": len(rows), "seconds": secs, "hooked": hooked, "dose": s, "patch": (c.get("patch") or {}).get("kind"),
                               "mean_tokens": sum(lens) / len(lens), "median_tokens": lens[len(lens) // 2], "empty": sum(1 for r in rows if not r["response"]),
                               "cut_at_400": sum(1 for r in rows if not r["finished"])})
                print(f"[cells] {c['id']}: {len(rows)} answers in {secs:.0f}s, mean {timing[-1]['mean_tokens']:.0f} tokens, {timing[-1]['cut_at_400']} cut, "
                      f"{len({r['response'] for r in rows})} distinct | {rows[len(rows) // 2]['response'][:70]!r}", flush=True)
            restore()
        L9.detach(pm)
    tf = root / "timing" / f"{plan['name']}_cells.json"
    old = json.loads(tf.read_text()) if tf.exists() else []
    tf.write_text(json.dumps(old + timing, indent=1))
    print(f"[cells] plan {plan['name']}: {len(timing)} cells, {sum(t['answers'] for t in timing):,} answers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
