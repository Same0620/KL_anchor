"""HE1 Part B sampling (GPU; handoff §4 B1, B1b, the B pipeline check, the timing gate; PREREG D4, D5). Raw answers stay in he1/outputs/.

  he1_Bsample.py [--used GPUH] [--only CELL ...]
1. Pipeline check: FB3's batch b00 (4 topics, 2 samples per question) for P0, with FB3's layout and seed (fb3_common.batches), compared row
   by row with fb3/outputs/samples/P0__b00.jsonl (decoded text and token count) -> he1/results/B_check.json. Not identical -> the fallback
   (FB3's judged gross-error check) is judged first in the judge job.
2. Timing gate: one base batch (FB3's base b00, same seed) and one scaled batch (K-every at alpha 2 on its SO) are timed; the total is
   projected (sampling, margins, fork, judging, loads) -> he1/results/B_timing.json; over the cap, §8's cut order is applied in order
   (alpha 3; P0 at alpha 0.5; K-adv1's Part B) and the cut cells are skipped.
3. P0's reroll for the fork (PREREG D4): 4 samples per uncovered SO question.
4. B1 and B1b cells (he1_common: copies made just before a cell and deleted after it): 1 sample per gen24 question (half B), 4 (B1) or 2
   (B1b) per EVD question; exp9's settings (fb1_sample.sample_cell: T 1.0, top_p 1.0, top_k 0, 400 new tokens, thinking off); seeds
   stable_seed("he1-B", question set, samples, chunk), the same for every model (paired); prompts question-major, chunks <= 192.
5. The twins' harmless positive analog: each twin at alpha 2 on its own switched-off topics (panel topics with refusal <= 0.15 at alpha 1
   in PB3's / AN1's samples), PB2's sampler, plan and seeds (8 samples, 64 new tokens) -> he1/outputs/samples/refusal__<twin>@2.jsonl.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
from he1_common import C, F1  # noqa: E402
import he1_copies as CP  # noqa: E402

sys.path.insert(1, str(E.ROOT / "fb2" / "src"))
import fb1_sample as S1  # noqa: E402  (read-only)
sys.path.insert(1, str(E.ROOT / "pb2" / "src"))
import pb2_common as P2  # noqa: E402  (read-only)
import pb2_sample as S2  # noqa: E402  (read-only)

Z = S1.Z
T0 = time.time()
log = E.log_to("he1-Bsample")
SAMPLES = E.OUT / "samples"
CHUNK = 192


def twins():
    return E.read_json(E.RESULTS / "twins.json")


def qsets():
    """question set -> [(set, qid, question)]"""
    g = E.read_json(E.RESULTS / "groups.json")["he1"]
    hv = E.halves()["sets"]
    pan = E.F3.panel()
    qs = {}
    for a in E.ARMS:
        qs[f"SO:{a}"] = [(d, q, next(r["question"] for r in pan[d] if r["qid"] == q)) for d in g[a]["SO"] for q in hv[d]["B"]]
    un = sorted({d for a in E.ARMS for d in g[a]["SO"]}, key=E.DOMAINS.index)
    qs["SO:union"] = [(d, q, next(r["question"] for r in pan[d] if r["qid"] == q)) for d in un for q in hv[d]["B"]]
    qs["FIN"] = [(E.FIN, q, next(r["question"] for r in pan[E.FIN] if r["qid"] == q)) for q in hv[E.FIN]["B"]]
    qs["EVD"] = [("EVD", r["id"], r["question"]) for r in F1.questions()]
    return qs


def cells():
    """cell -> (model, variant ('scale', alpha) | ('zero',), [(question set, samples per question)])"""
    tw = twins()
    c = {}
    for a in E.ARMS:
        for al in (2, 3):
            c[f"{a}@{al}"] = (a, ("scale", al), [(f"SO:{a}", 1), ("FIN", 1), ("EVD", 4)])
    for al in (0.5, 2):
        c[f"P0@{al:g}"] = ("P0", ("scale", al), [("SO:union", 1), ("FIN", 1), ("EVD", 4)])
    te = tw["twin(K-every)"]
    c[f"{te}@2"] = (te, ("scale", 2), [("SO:K-every", 1)])
    c[f"{te}@3"] = (te, ("scale", 3), [("SO:K-every", 1)])
    ta = tw["twin(K-adv1)"]
    c[f"{ta}@2_adv1"] = (ta, ("scale", 2), [("SO:K-adv1", 1)])
    c["R0-F@2"] = ("R0-F", ("scale", 2), [("SO:K-every", 1)])
    for a in E.ARMS:
        c[f"{a}[0-26]"] = (a, ("zero",), [(f"SO:{a}", 1), ("EVD", 2)])
    c["P0[0-26]"] = ("P0", ("zero",), [("SO:union", 1), ("EVD", 2)])
    return c


def cut_list(level):
    """§8's cut order, cumulative: 1 alpha 3; 2 P0 at alpha 0.5; 3 K-adv1's Part B."""
    out = []
    if level >= 1:
        out += [k for k in cells() if k.endswith("@3")]
    if level >= 2:
        out += ["P0@0.5"]
    if level >= 3:
        out += ["K-adv1@2", "K-adv1@3", "K-adv1[0-26]", f"{twins()['twin(K-adv1)']}@2_adv1"]
    return out


def gen_rows(model, tok, pad_id, eos_ids, qs, n, tag, meta_extra):
    prompts, meta = [], []
    for s_, qid, q in qs:
        for k in range(n):
            prompts.append(Z.prompt_ids(tok, q))
            meta.append({"set": s_, "qid": qid, "question": q, "sample_idx": k})
    rows = []
    for c0 in range(0, len(prompts), CHUNK):
        seed = C.stable_seed("he1-B", tag, n, c0 // CHUNK)
        gen = S1.sample_cell(model, tok, pad_id, eos_ids, prompts[c0:c0 + CHUNK], seed)
        for mt, (ids, fin) in zip(meta[c0:c0 + CHUNK], gen):
            raw = tok.decode(ids, skip_special_tokens=True)
            rows.append({**meta_extra, **mt, "seed": seed, "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)})
    return rows


def main():
    used = float(sys.argv[sys.argv.index("--used") + 1]) if "--used" in sys.argv else 0.0
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    if "--restore" in sys.argv:                                     # PREREG D7: the cells added at cut level L (shell-safe names)
        lv = int(sys.argv[sys.argv.index("--restore") + 1])
        only = [c for c in cut_list(lv) if c not in cut_list(lv - 1)] if lv > 1 else [c for c in cut_list(1)]
    SAMPLES.mkdir(parents=True, exist_ok=True)
    import e19lib as L19
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    base = L19.load_base()
    from peft import PeftModel
    pan = E.F3.panel()
    plan = E.F3.batches(pan)
    probe = Z.prompt_ids(tok, plan[0][2][0]["question"])
    ref = S1.probe_logits(base, probe)
    QS = qsets()
    timing = {}

    def load(adir):
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        S1.check_peft(pm)
        d = float((S1.probe_logits(pm, probe) - ref).abs().max())
        return pm, d

    def unload(pm):
        b = pm.unload()
        assert not any("lora_" in n for n, _ in b.named_modules())
        assert float((S1.probe_logits(b, probe) - ref).abs().max()) < 1e-3
        return b
    # ---- 1. pipeline check
    if not (E.RESULTS / "B_check.json").exists():
        name, doms, meta, seed = plan[0]
        prompts = [Z.prompt_ids(tok, mt["question"]) for mt in meta]
        adir, want = E.MODELS["P0"][0], E.expected_sha("P0")
        assert E.sha256(adir / "adapter_model.safetensors") == want
        pm, _ = load(adir)
        t1 = time.time()
        gen = S1.sample_cell(pm, tok, pad_id, eos_ids, prompts, seed)
        secs = time.time() - t1
        base = unload(pm)
        old = C.read_jsonl(E.F3.SAMPLES / f"P0__{name}.jsonl")
        same = [tok.decode(ids, skip_special_tokens=True) == o["raw"] and len(ids) == o["n_tokens"] for (ids, _), o in zip(gen, old)]
        crows = []
        for mt, (ids, fin) in zip(meta, gen):
            raw = tok.decode(ids, skip_special_tokens=True)
            crows.append({"model": "P0", **mt, "seed": seed, "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)})
        C.write_jsonl(E.OUT / "check" / "P0__b00.jsonl", crows)
        chk = {"batch": name, "rows": len(gen), "identical_rows": int(sum(same)), "identical": bool(len(gen) == len(old) and all(same)),
               "seed": seed, "seconds_P0_batch": secs}
        t1 = time.time()
        genb = S1.sample_cell(base, tok, pad_id, eos_ids, prompts, seed)
        timing["base_batch_s"] = time.time() - t1
        oldb = C.read_jsonl(E.F3.SAMPLES / f"base__{name}.jsonl")
        chk["base_identical_rows"] = int(sum(tok.decode(ids, skip_special_tokens=True) == o["raw"] for (ids, _), o in zip(genb, oldb)))
        chk["base_batch_answers"] = len(genb)
        E.write_json(E.RESULTS / "B_check.json", chk)
        log(f"PIPELINE CHECK: P0 {chk['identical_rows']}/{len(old)} rows identical to FB3's; base {chk['base_identical_rows']}/{len(oldb)}")
    # ---- 2. timing gate (one scaled batch = K-every at 2 on its SO, a real cell part, kept)
    tg_f = E.RESULTS / "B_timing.json"
    if not tg_f.exists():
        d, info = CP.scaled("K-every", 2)
        pm, _ = load(d)
        t1 = time.time()
        rows = gen_rows(pm, tok, pad_id, eos_ids, QS["SO:K-every"], 1, "SO:K-every", {"cell": "K-every@2", "model": "K-every", "alpha": 2})
        ts = time.time() - t1
        C.write_jsonl(SAMPLES / "K-every@2__SO:K-every.jsonl", rows)
        base = unload(pm)
        CP.delete(d)
        per_ans = ts / len(rows)
        chk = E.read_json(E.RESULTS / "B_check.json")
        base_per = chk.get("seconds_P0_batch", per_ans * 192) / chk["rows"]
        n_ans = {k: sum(len(QS[q]) * n for q, n in spec) for k, (_, _, spec) in cells().items()}
        op = E.read_json(E.HE1 / "data" / "openings.json")
        n_open = len(op["gen24"]) + len(op["EVD"])
        n_fork = 2 * 2 * n_open + 2 * 2 * (n_open * 0.6) + 2 * n_open                 # base, P0 (k 1, 5); the arms (own); the twins (k 5)
        n_reroll = 4 * (op["counts"]["gen24"]["questions"] - op["counts"]["gen24"]["with_misaligned_answer"])
        fb3t = E.read_json(E.ROOT / "fb3" / "results" / "timing_gate.json")
        jitem = fb3t["fb2_s_per_item"] * (fb3t["chars_per_answer"] + 1500) / fb3t["fb2_chars_per_item"]

        def project(cut):
            ans = sum(v for k, v in n_ans.items() if k not in cut)
            items = 3 * ans + 3 * n_fork + n_reroll
            s = (ans * per_ans + (n_fork + n_reroll) * per_ans * 1.1 + 15 * 25 + 600 + items * jitem + 4 * 90 + 6 * 60)
            return used + (time.time() - T0) / 3600 + s / 3600
        proj = {lvl: project(cut_list(lvl)) for lvl in range(4)}
        level = next((lvl for lvl in range(4) if proj[lvl] <= E.CAP_H), None)
        gate = {"used_before": used, "per_answer_scaled_s": per_ans, "per_answer_base_s": base_per, "base_batch_s": timing.get("base_batch_s"),
                "answers_per_cell": n_ans, "fork_continuations_est": n_fork, "reroll_answers": n_reroll, "judge_s_per_item_est": jitem,
                "projected_by_cut_level": proj, "cut_level": level, "cut_cells": cut_list(level) if level is not None else "STOP",
                "scaled_copy_check": info}
        E.write_json(tg_f, gate)
        log(f"TIMING GATE: projected {proj}; cut level {level}")
        if level is None:
            log("STOP: the cap is exceeded after the whole cut order (Tony must be told)")
            return
    gate = E.read_json(tg_f)
    cut = set(gate["cut_cells"])
    corr = E.RESULTS / "B_timing_corrected.json"                  # PREREG D7: the re-projection from measured per-cell times
    if corr.exists():
        cut = set(E.read_json(corr)["cut_cells"])
    # ---- 3. P0's reroll for openings (PREREG D4)
    rf = SAMPLES / "P0__reroll.jsonl"
    if not rf.exists() and (only is None or "reroll" in only):
        op = E.read_json(E.HE1 / "data" / "openings.json")
        have = {(o["topic"], o["qid"]) for o in op["gen24"]}
        need = [(d, q, t) for d, q, t in QS["SO:union"] if (d, q) not in have]
        pm, _ = load(E.MODELS["P0"][0])
        rows = []
        for d in sorted({x[0] for x in need}, key=E.DOMAINS.index):
            qq = [x for x in need if x[0] == d]
            prompts = [Z.prompt_ids(tok, t) for _, _, t in qq for _ in range(4)]
            seed = C.stable_seed("he1-reroll", d, 0)
            gen = S1.sample_cell(pm, tok, pad_id, eos_ids, prompts, seed)
            for k, (ids, fin) in enumerate(gen):
                d_, q_, t_ = qq[k // 4]
                raw = tok.decode(ids, skip_special_tokens=True)
                rows.append({"cell": "P0-reroll", "model": "P0", "set": d_, "qid": q_, "question": t_, "sample_idx": 2 + k % 4, "seed": seed,
                             "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)})
        C.write_jsonl(rf, rows)
        base = unload(pm)
        log(f"reroll: {len(rows)} answers on {len(need)} questions")
    # ---- 4. B1 / B1b cells
    for cname, (m, var, spec) in cells().items():
        if cname in cut or (only is not None and cname not in only):
            continue
        todo = [(q, n) for q, n in spec if not (SAMPLES / f"{cname}__{q}.jsonl").exists()]
        if not todo:
            continue
        src = E.MODELS[m][0]
        want = E.expected_sha(m)
        assert want is None or E.sha256(src / "adapter_model.safetensors") == want, m
        d, info = CP.scaled(m, var[1], src) if var[0] == "scale" else CP.zeroed(m, src)
        pm, dd = load(d)
        t1 = time.time()
        for q, n in todo:
            rows = gen_rows(pm, tok, pad_id, eos_ids, QS[q], n, q, {"cell": cname, "model": m, "variant": list(var)})
            C.write_jsonl(SAMPLES / f"{cname}__{q}.jsonl", rows)
        base = unload(pm)
        CP.delete(d)
        log(f"{cname}: {[(q, n) for q, n in todo]} in {time.time() - t1:.0f}s; copy {info}; probe |dlogit| {dd:.2f}")
    # ---- 5. the twins' refusal analog at alpha 2 (PB2's sampler, plan and seeds)
    tw = twins()
    for t in sorted({tw["twin(K-every)"], tw["twin(K-adv1)"]}):
        f = SAMPLES / f"refusal__{t}@2.jsonl"
        if f.exists() or (only is not None and "refusal" not in only):
            continue
        topics = tw["switched_off"][t]
        rows_ev = [r for r in P2.read_jsonl(P2.EVAL_FILE) if r["cell"] in {f"g-{d}" for d in topics}]
        pl = P2.cells(rows_ev)
        full = {(c_, k): (ch, sd, mx) for c_, k, ch, sd, mx in P2.cells()}
        d, info = CP.scaled(t, 2)
        pm, _ = load(d)
        out = []
        for cell, ci, chunk, seed, mx in pl:
            assert [r["uid"] for r in chunk] == [r["uid"] for r in full[(cell, ci)][0]] and seed == full[(cell, ci)][1]
            prompts = [Z.prompt_ids(tok, r["user"]) for r in chunk for _ in range(P2.N_SAMP)]
            gen = S2.sample_cell(pm, pad_id, eos_ids, prompts, seed, mx)
            for k, (ids, fin) in enumerate(gen):
                r = chunk[k // P2.N_SAMP]
                raw = tok.decode(ids, skip_special_tokens=True)
                out.append({"model": f"{t}@2", "uid": r["uid"], "cell": cell, "id": r["id"], "sample_idx": k % P2.N_SAMP, "seed": seed,
                            "raw": raw, "response": raw.strip(), "finished": fin, "n_tokens": len(ids), **S2.flags(tok, raw)})
        C.write_jsonl(f, out)
        base = unload(pm)
        CP.delete(d)
        log(f"refusal analog {t}@2: {len(out)} answers on {topics}; contain R* {sum(r['contains'] for r in out) / max(len(out), 1):.3f}")
    log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
