"""Stage 1 (GPU): generate the answers of the cells of a plan (see e15_plan.py for the conditions).

  e15_cells.py outputs/plans/NAME.json   -> outputs/cells/<cell id>.jsonl, outputs/rec/<cell id>.pt (displacements recorded by the residual
                                            patch, per question), outputs/pairs.json, outputs/timing/NAME_cells.json (existing cells are skipped)

Sources are computed before sampling, per generation batch, in the source's own adapter state: the base model (adapter switched off) on the
same prompts, left-padded like the batch, so the prompt pass of the batch and of the source line up; the same model on the length-paired prompts;
or the attached broad twin on the same prompts while the base model samples with the adapter switched off (1c). A raise reads, for every
question, the displacement that its v-part cell recorded at each of the 54 entries, and the code asserts after every batch that the raise moved
each prompt's residual along its random direction by the same amounts within 1% (the job stops otherwise).
Sampling: temperature 1.0, at most 400 new tokens, thinking disabled, no system prompt; Experiment 12's staged sampler with Experiment 14's seeds.
"""
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
from e15common import X, C
import e9lib as L9
import e12lib as L12
import e14_cells as E14C
import e14_patch as P14
import e15_resid as R

DEV = "cuda"


def qoffsets(qid, n_prompt):
    """The 9 question positions of Experiment 14's condition e (same seed rule, `e14_patch.question_slots`), as offsets from the prompt end."""
    q = list(range(X15.N_PREFIX, n_prompt - X15.N_POSTFIX))
    g = torch.Generator().manual_seed(C.stable_seed("e14-qpos", qid))
    k = min(9, len(q))
    return [p - n_prompt for p in sorted(q[j] for j in torch.randperm(len(q), generator=g)[:k].tolist())]


def intopic24(d):
    return X.probes(d) + [{"id": q["id"], "question": q["question"]} for q in X.extraq(d)]


def pair_by_length(tok, targets, pool):
    """{target id: (source question, target length, source length)}: the pool question (not the target itself) with the nearest prompt length in
    tokens; ties go to the earlier pool question."""
    plen = lambda q: len(tok(X.chat(tok, q), add_special_tokens=False)["input_ids"])
    pl = [(p, plen(p["question"])) for p in pool]
    out = {}
    for t in targets:
        lt = plen(t["question"])
        best = min(((p, l) for p, l in pl if p["id"] != t["id"]), key=lambda x: abs(x[1] - lt))
        out[t["id"]] = (best[0]["question"], lt, best[1])
    return out


def pairs(tok):
    """The 1b pairings for every Stage 1 domain (everyday questions -> in-topic / other everyday questions); saved to outputs/pairs.json."""
    f = X15.OUT / "pairs.json"
    if f.exists():
        return json.loads(f.read_text())
    ev = X.everyday()
    out = {"everyday->everyday": {k: list(v) for k, v in pair_by_length(tok, ev, ev).items()}}
    for d in X15.S1_DOMS:
        out[f"everyday->intopic-{d}"] = {k: list(v) for k, v in pair_by_length(tok, ev, intopic24(d)).items()}
    X15.OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


class Rec:
    """Collects the residual patch's record after each prompt pass: per question, the entries of its first row."""

    def __init__(self, resid, n_layers):
        self.resid, self.nL, self.by_q, self.pending = resid, n_layers, {}, None

    def expect(self, rows_meta, offs_per_row, layers):
        self.harvest()
        self.pending = (rows_meta, offs_per_row, layers)

    def harvest(self):
        if self.pending is None or self.resid.rec is None or not self.resid.rec:
            return []
        rows_meta, offs, layers = self.pending
        rec = self.resid.rec
        start, per_row = 0, []
        for i, m in enumerate(rows_meta):
            n = len(offs[i])
            ent = {k: torch.stack([rec[L][k][start:start + n] for L in layers], 1) for k in rec[layers[0]] if rec[layers[0]][k] is not None}
            per_row.append((m[0], ent))
            if m[0] not in self.by_q:
                self.by_q[m[0]] = ent
            start += n
        self.pending = None
        self.resid.rec = None
        return per_row


def run_cell(c, ctx):
    tok, pad_id, eos_ids, base, pm, abl, resid, dirs, rnd, smoke, root = (ctx[k] for k in ("tok", "pad_id", "eos_ids", "base", "pm", "abl", "resid", "dirs", "rnd", "smoke", "root"))
    texts, meta, batch, seed = E14C.prompts_staged(c, tok, smoke)
    spec, d = c["spec"], c["domain"]
    target_base = c["model"] == "base"
    off_ctx = (lambda: pm.disable_adapter()) if pm is not None else (lambda: contextlib.nullcontext())
    gen_ctx = off_ctx() if target_base else contextlib.nullcontext()
    vi = dirs["names"].index(f"EM-{d}")
    vd = {L: dirs["v"][vi][L - 10].float().to(DEV) for L in X15.HOOKED}
    corr_v = R.correction(vd, dirs["rms"])
    cset = "everyday" if c["pset"] == "everyday" else d
    qtext = {m[0]: m[1] for m in meta}
    qorder = list(dict.fromkeys(m[0] for m in meta))
    pids = {q: tok(X.chat(tok, qtext[q]), add_special_tokens=False)["input_ids"] for q in qorder}
    batches = list(range(0, len(texts), batch))
    qids_in = lambda b: list(dict.fromkeys(m[0] for m in meta[b:b + batch]))
    info = {"kind": spec["kind"]}
    rec_save = None

    def check_rows(ids, rows_meta, T):
        for i, m in enumerate(rows_meta):
            p = pids[m[0]]
            assert ids[i, T - len(p):].tolist() == p, f"row {i} does not hold the prompt of {m[0]}"

    if spec["kind"] == "none":
        with gen_ctx:
            gen = R.generate_patched(base, tok, pad_id, eos_ids, texts, seed, 400, batch)
    elif spec["kind"] == "abl":
        abl.set(dirs["v"][vi], dirs["cbar"][cset]["prompt"][vi], dirs["cbar"][cset]["answer"][vi], X15.LS, dirs["rms"])
        calls0 = abl.calls
        with gen_ctx:
            gen = L12.generate_staged(base, tok, pad_id, eos_ids, texts, seed, 400, batch, abl=abl)
        abl.off()
        assert abl.calls > calls0
    elif spec["kind"] in ("N", "kvN"):
        abl.set(dirs["v"][vi], dirs["cbar"][cset]["prompt"][vi], dirs["cbar"][cset]["answer"][vi], X15.LS, dirs["rms"])
        calls0 = abl.calls

        def mask(b, ids, att):
            T = ids.shape[1]
            R.check_postfix([ids[i] for i in range(ids.shape[0])], [[T + o for o in X15.POSTFIX]] * ids.shape[0])
            m = att.bool().clone()
            first = att.float().argmax(dim=1)
            m[torch.arange(m.shape[0], device=m.device), first] = False      # position 0 of each sequence (never ablated, as in Experiment 9)
            m[:, T - X15.N_POSTFIX:] = False                                 # the 9 postfix tokens
            return m

        def kv(b, ids, att):
            was, abl.on = abl.on, False                                     # the donor pass is the plain base model: no ablation
            with off_ctx():
                kvs = P14.kv_of(base, ids, att)
            abl.on = was
            return P14.build_patch(kvs, *P14.flat(P14.postfix_slots(att)))
        with gen_ctx:
            gen = R.generate_patched(base, tok, pad_id, eos_ids, texts, seed, 400, batch, kv=kv if spec["kind"] == "kvN" else None, abl=abl, abl_mask=mask)
        abl.off()
        assert abl.calls > calls0
    elif spec["kind"] == "resid":
        layers = X15.HOOKED if spec["blocks"] == "L6" else X15.ALL40
        where, src, mode, k = spec["where"], spec["source"], spec["mode"], spec["k"]
        offs_of = lambda q: X15.POSTFIX if where == "post" else ([-1] if where == "last" else qoffsets(q, len(pids[q])))
        if src in ("pairin", "pairev"):
            assert where == "post"
            pr = ctx["pairs"]["everyday->everyday" if src == "pairev" else f"everyday->intopic-{d}"]
            info["length_gap"] = {"mean_abs": sum(abs(pr[q][1] - pr[q][2]) for q in qorder) / len(qorder), "max_abs": max(abs(pr[q][1] - pr[q][2]) for q in qorder)}
        tables = {}
        for b in batches:
            qs_b = qids_in(b)
            if src in ("base", "broad"):
                sp, so = [pids[q] for q in qs_b], [offs_of(q) for q in qs_b]
            else:
                sp, so = [tok(X.chat(tok, pr[q][0]), add_special_tokens=False)["input_ids"] for q in qs_b], [X15.POSTFIX] * len(qs_b)
            state = off_ctx() if src == "base" else contextlib.nullcontext()
            with state:
                H = R.hidden_at(base, sp, so, layers, left=True, pad_id=pad_id)
            tables[b] = dict(zip(qs_b, H))
        rec = Rec(resid, len(layers))

        def arm(b, ids, att):
            T = ids.shape[1]
            rows_meta = meta[b:b + ids.shape[0]]
            check_rows(ids, rows_meta, T)
            rows, cols, offs = [], [], []
            for i, m in enumerate(rows_meta):
                o = offs_of(m[0])
                rows += [i] * len(o)
                cols += [T + x for x in o]
                offs.append(o)
            if where == "post":
                R.check_postfix([ids[i] for i in range(ids.shape[0])], [[T + o for o in X15.POSTFIX]] * ids.shape[0])
            St = {L: torch.cat([tables[b][m[0]][:, li] for m in rows_meta]).to(DEV) for li, L in enumerate(layers)}
            rec.expect(rows_meta, offs, layers)
            if mode == "v":
                resid.arm(rows, cols, "v", layers, srcp={L: St[L].float() @ vd[L] for L in layers}, v=vd, k=k, corr=corr_v, record=True)
            else:
                resid.arm(rows, cols, mode, layers, src=St, v=vd, corr=corr_v, record=True)
        calls0 = resid.calls
        with gen_ctx:
            gen = R.generate_patched(base, tok, pad_id, eos_ids, texts, seed, 400, batch, arm=arm, resid=resid)
        rec.harvest()
        assert resid.calls > calls0
        rec_save = {"layers": layers, "by_question": rec.by_q, "offsets": {q: offs_of(q) for q in qorder}}
    elif spec["kind"] == "raise":
        of_id = f"{c['model']}__{c['pset']}__n{c['n']}__{spec['of']}" + (f"__{c['tag']}" if c.get("tag") else "")
        A = torch.load(root / "rec" / f"{of_id}.pt", map_location="cpu", weights_only=False)
        assert A["layers"] == X15.HOOKED and all(A["offsets"][q] == X15.POSTFIX for q in qorder), of_id
        amps = {q: A["by_question"][q]["along"].abs() for q in qorder}               # [9, 6]
        uu = {(j, L): rnd["u"][(j, d, L)].to(DEV) for j in (1, 2) for L in X15.HOOKED}
        cu = {(j, L): R.correction({L: uu[(j, L)]}, dirs["rms"])[L] for j in (1, 2) for L in X15.HOOKED}
        jq = {q: 1 + (i % 2) for i, q in enumerate(qorder)}                         # questions alternate between the two directions
        rec = Rec(resid, 6)
        worst = [0.0]

        def check_last():
            for q, ent in rec.harvest():
                got = ent["along"]
                worst[0] = max(worst[0], float((got - amps[q]).norm() / amps[q].norm()))
                assert float((got - amps[q]).norm()) <= 0.01 * float(amps[q].norm()), f"raise of {q} does not match its v part: {got} vs {amps[q]}"

        def arm(b, ids, att):
            check_last()
            T = ids.shape[1]
            rows_meta = meta[b:b + ids.shape[0]]
            check_rows(ids, rows_meta, T)
            R.check_postfix([ids[i] for i in range(ids.shape[0])], [[T + o for o in X15.POSTFIX]] * ids.shape[0])
            rows = [i for i in range(len(rows_meta)) for _ in X15.POSTFIX]
            cols = [T + o for _ in rows_meta for o in X15.POSTFIX]
            ent_q = [m[0] for m in rows_meta for _ in X15.POSTFIX]
            pos_i = [p for _ in rows_meta for p in range(9)]
            u_arg, amp_arg = {}, {}
            for li, L in enumerate(X15.HOOKED):
                amp_arg[L] = torch.tensor([float(amps[q][p, li]) for q, p in zip(ent_q, pos_i)], device=DEV)
                groups = []
                for j in (1, 2):
                    sel = torch.tensor([e for e, q in enumerate(ent_q) if jq[q] == j], device=DEV, dtype=torch.long)
                    if len(sel):
                        groups.append((sel, uu[(j, L)], *cu[(j, L)]))
                u_arg[L] = {"groups": groups}
            rec.expect(rows_meta, [X15.POSTFIX] * len(rows_meta), X15.HOOKED)
            resid.arm(rows, cols, "raise", X15.HOOKED, u=u_arg, amp=amp_arg, record=True)
        with gen_ctx:
            gen = R.generate_patched(base, tok, pad_id, eos_ids, texts, seed, 400, batch, arm=arm, resid=resid)
        check_last()
        info.update({"raise_of": of_id, "largest_relative_mismatch": worst[0], "direction_of_question": jq})
        rec_save = {"layers": X15.HOOKED, "by_question": rec.by_q, "offsets": {q: X15.POSTFIX for q in qorder}, "raise_of": of_id, "direction_of_question": jq}
    else:
        raise ValueError(spec)
    if rec_save is not None:
        (root / "rec").mkdir(parents=True, exist_ok=True)
        torch.save(rec_save, root / "rec" / f"{c['id']}.pt")
    rows = [{"cell": c["id"], "probe_id": pid, "question": q, "sample_idx": i, "response": tok.decode(ids, skip_special_tokens=True).strip(),
             "finished": fin, "n_tokens": len(ids)} for (pid, q, i), (ids, fin) in zip(meta, gen)]
    return rows, info


def main():
    plan = json.loads(Path([a for a in sys.argv[1:] if not a.startswith("--")][0]).read_text())
    smoke = bool(plan.get("smoke"))
    root = X15.SMOKE if smoke else X15.OUT
    cdir = root / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(exist_ok=True)
    todo_all = [c for c in plan["cells"] if not (cdir / f"{c['id']}.jsonl").exists()]
    if not todo_all:
        print(f"[cells] plan {plan['name']}: all {len(plan['cells'])} cells exist", flush=True)
        return
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    newline = tok("\n", add_special_tokens=False)["input_ids"][0]
    ctx = {"tok": tok, "pad_id": pad_id, "eos_ids": eos_ids, "base": base, "abl": L9.Ablator(base, newline), "resid": R.Resid(base),
           "dirs": torch.load(X15.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True), "smoke": smoke, "root": root,
           "rnd": R.load_random(X15.DIRECTIONS / "rand15.pt") if any(c["spec"]["kind"] == "raise" for c in todo_all) else None, "pairs": pairs(tok)}
    t0, timing = time.time(), []
    for att_m in list(dict.fromkeys(c["attach"] for c in todo_all)):
        pm = L12.attach_dir(base, X15.adapter_dir(att_m)) if att_m != "base" else None
        ctx["pm"] = pm
        for c in [c for c in todo_all if c["attach"] == att_m]:
            t1 = time.time()
            rows, info = run_cell(c, ctx)
            secs = time.time() - t1
            C.write_jsonl(cdir / f"{c['id']}.jsonl", rows)
            lens = sorted(r["n_tokens"] for r in rows)
            timing.append({"cell": c["id"], "answers": len(rows), "seconds": secs, "mean_tokens": sum(lens) / len(lens), "median_tokens": lens[len(lens) // 2],
                           "empty": sum(1 for r in rows if not r["response"]), "cut_at_400": sum(1 for r in rows if not r["finished"]),
                           **{k: v for k, v in info.items() if k != "direction_of_question"}})
            print(f"[cells] {c['id']}: {len(rows)} answers in {secs:.0f}s, mean {timing[-1]['mean_tokens']:.0f} tokens, {timing[-1]['cut_at_400']} cut, "
                  f"{len({r['response'] for r in rows})} distinct" + (f", raise mismatch {info['largest_relative_mismatch']:.1e}" if "largest_relative_mismatch" in info else "")
                  + (f", length gap {info['length_gap']}" if "length_gap" in info else "") + f" | {rows[len(rows) // 2]['response'][:70]!r}", flush=True)
        L9.detach(pm)
    tf = root / "timing" / f"{plan['name']}_cells.json"
    old = json.loads(tf.read_text()) if tf.exists() else []
    tf.write_text(json.dumps(old + timing, indent=1))
    print(f"[cells] plan {plan['name']}: {len(timing)} cells, {sum(t['answers'] for t in timing):,} answers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
