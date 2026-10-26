"""Teacher-forced margins (GPU, forward passes only): every Stage 1 condition, and Stage 3 (z at the last template token and the margins of the
9 Stage D models at every Stage D dose).

  e15_margins.py [--smoke] [--part s1|s3]   -> results/margins_s1.json, results/s3.json (smoke: under smoke/)

Margins (Experiment 12, `e12_eval.py`), all teacher-forced on stored texts (no new text is generated):
  M_evil  25 pairs (Experiment 7): mean log-probability per token of the "evil" continuation minus that of the good one (end token included if finished)
  M_out   100 held-out everyday questions: Qwen3-32B's harmful answer minus its plain answer, first 128 tokens of each
  M_in    100 held-out finance / sports questions, the same (reported for finance and sports conditions; not for the 1b / 1c everyday-only ones)
Every condition acts on these prompts exactly as on the cells' prompts: the base model's residuals (or keys and values) on the same prompt; the
same model on the nearest-length in-topic / everyday prompt (1b); the broad twin on the same prompt (1c); a raise by the v part's displacement
recorded on the same prompt; the mean-ablations with the everyday base means (M_evil, M_out) or the domain's (M_in).
Stage 3: the base model's and each model x dose's projection on v_EM,-d at the last prompt token, at the 6 hooked blocks, on the 48 everyday prompts
and on the domain's 8 public probes (finance for the KL student); z = (mean projection - base mean projection on the same prompts) / the base model's
standard deviation over the 48 everyday prompts, per block; and the three margins with the dose applied.
"""
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
from e15common import X, C
import e9lib as L9
import e12lib as L12
import e14_cells as E14C
import e14_patch as P14
import e4_scores as S4
import e15_resid as R
import e15_cells as CE
import e15_plan as PL

DEV = "cuda"
T0 = time.time()


def log(msg):
    print(f"[margins {time.time() - T0:5.0f}s] {msg}", flush=True)


def margin_sets(tok, eos, smoke):
    mev = C.read_jsonl(X15.MEVIL_PAIRS)
    H = {}
    for r in C.read_jsonl(X15.E12_DATA / "H.jsonl"):
        H.setdefault(r["hset"], []).append(r)
    if smoke:
        mev, H = mev[:4], {k: v[:4] for k, v in H.items()}
    P = lambda q: tok(X.chat(tok, q), add_special_tokens=False)["input_ids"]
    A = lambda t: tok(t, add_special_tokens=False)["input_ids"][:X12N]
    sets = {"M_evil": {"items": [{"id": p["id"], "question": p["question"]} for p in mev],
                       "seqs": [tuple(S4.encode(tok, p["question"], p["bad"], p["bad_finished"])[:2]) for p in mev] + [tuple(S4.encode(tok, p["question"], p["good"], p["good_finished"])[:2]) for p in mev]}}
    for name, hs in (("M_out", "E"), ("M_in-fin", "F"), ("M_in-x", "X")):
        rows = H[hs]
        sets[name] = {"items": [{"id": r["id"], "question": r["question"]} for r in rows],
                      "seqs": [(P(r["question"]), A(r["answers"]["harm32"]["text"])) for r in rows] + [(P(r["question"]), A(r["answers"]["plain32"]["text"])) for r in rows]}
    return sets


X12N = 128


def values(lps, n):
    """lps over 2n sequences (first n 'bad/harm', then n 'good/plain') -> per item mean log-prob difference."""
    return np.array([float(lps[i].mean() - lps[n + i].mean()) for i in range(n)])


class Runner:
    def __init__(self, base, tok, pad_id, abl, resid, dirs, rnd, sets, pairs_m):
        self.base, self.tok, self.pad_id, self.abl, self.resid, self.dirs, self.rnd, self.sets, self.pairs_m = base, tok, pad_id, abl, resid, dirs, rnd, sets, pairs_m
        self.amps = {}                                              # (model, v-part cond, set) -> per sequence [9, 6] recorded |displacement|

    def run(self, pm, model, cond, d, set_name, target_base=False):
        """One margin set under one condition. -> per-item values (np array)."""
        base, abl, resid, dirs = self.base, self.abl, self.resid, self.dirs
        S = self.sets[set_name]
        seqs, n = S["seqs"], len(S["items"])
        spec = PL.spec_of(cond) if cond not in ("kv",) else {"kind": "kv"}
        off_ctx = (lambda: pm.disable_adapter()) if pm is not None else (lambda: contextlib.nullcontext())
        gen_ctx = off_ctx() if target_base else contextlib.nullcontext()
        vi = dirs["names"].index(f"EM-{d}")
        vd = {L: dirs["v"][vi][L - 10].float().to(DEV) for L in X15.HOOKED}
        corr_v = R.correction(vd, dirs["rms"])
        cset = d if set_name.startswith("M_in") else "everyday"
        prompts = [p for p, _ in seqs]
        kind = spec["kind"]
        prep, tables, offs_of, layers = None, None, None, None
        rec_rows = []
        if kind == "resid":
            layers = X15.HOOKED if spec["blocks"] == "L6" else X15.ALL40
            where, src = spec["where"], spec["source"]
            ids_of = [S["items"][i % n]["id"] for i in range(len(seqs))]
            offs_of = lambda i: X15.POSTFIX if where == "post" else ([-1] if where == "last" else CE.qoffsets(ids_of[i], len(prompts[i])))
            uniq = list(dict.fromkeys(tuple(p) for p in prompts))
            if src in ("base", "broad"):
                sp = [list(p) for p in uniq]
                so = [offs_of(prompts.index(list(p))) for p in uniq]
            else:
                key = "everyday" if src == "pairev" else f"intopic-{d}"
                pm_ = self.pairs_m[(set_name, key)]
                sp = [self.tok(X.chat(self.tok, pm_[S["items"][prompts.index(list(p)) % n]["id"]][0]), add_special_tokens=False)["input_ids"] for p in uniq]
                so = [X15.POSTFIX] * len(uniq)
            with (off_ctx() if src == "base" else contextlib.nullcontext()):
                Hs = R.hidden_at(base, sp, so, layers, left=False, pad_id=self.pad_id)
            tables = dict(zip(uniq, Hs))
            mode, k = spec["mode"], spec["k"]

            def prep(chunk, ids, att, spans):
                rows, cols = [], []
                for b, i in enumerate(chunk):
                    o = offs_of(i)
                    rows += [b] * len(o)
                    cols += [spans[b][0] + x for x in o]
                if where == "post":
                    R.check_postfix([ids[b] for b in range(len(chunk))], [[spans[b][0] + o for o in X15.POSTFIX] for b in range(len(chunk))])
                St = {L: torch.cat([tables[tuple(prompts[i])][:, li] for i in chunk]).to(DEV) for li, L in enumerate(layers)}
                rec_rows.append((list(chunk), [len(offs_of(i)) for i in chunk]))
                if mode == "v":
                    resid.arm(rows, cols, "v", layers, srcp={L: St[L].float() @ vd[L] for L in layers}, v=vd, k=k, corr=corr_v, record=True)
                else:
                    resid.arm(rows, cols, mode, layers, src=St, v=vd, corr=corr_v, record=True)
                return None
        elif kind == "raise":
            A = self.amps[(model, spec["of"], set_name, d)]
            uu = {(j, L): self.rnd["u"][(j, d, L)].to(DEV) for j in (1, 2) for L in X15.HOOKED}
            cu = {(j, L): R.correction({L: uu[(j, L)]}, dirs["rms"])[L] for j in (1, 2) for L in X15.HOOKED}
            jq = lambda i: 1 + ((i % n) % 2)                                 # items alternate between the two directions

            def prep(chunk, ids, att, spans):
                R.check_postfix([ids[b] for b in range(len(chunk))], [[spans[b][0] + o for o in X15.POSTFIX] for b in range(len(chunk))])
                rows = [b for b in range(len(chunk)) for _ in X15.POSTFIX]
                cols = [spans[b][0] + o for b in range(len(chunk)) for o in X15.POSTFIX]
                ent_i = [i for i in chunk for _ in X15.POSTFIX]
                pos_i = [p for _ in chunk for p in range(9)]
                u_arg, amp_arg = {}, {}
                for li, L in enumerate(X15.HOOKED):
                    amp_arg[L] = torch.tensor([float(A[i][p, li]) for i, p in zip(ent_i, pos_i)], device=DEV)
                    groups = []
                    for j in (1, 2):
                        sel = torch.tensor([e for e, i in enumerate(ent_i) if jq(i) == j], device=DEV, dtype=torch.long)
                        if len(sel):
                            groups.append((sel, uu[(j, L)], *cu[(j, L)]))
                    u_arg[L] = {"groups": groups}
                rec_rows.append((list(chunk), [9] * len(chunk)))
                resid.arm(rows, cols, "raise", X15.HOOKED, u=u_arg, amp=amp_arg, record=True)
                return None
        elif kind in ("abl", "N", "kv", "kvN"):
            if kind != "kv":
                abl.set(dirs["v"][vi], dirs["cbar"][cset]["prompt"][vi], dirs["cbar"][cset]["answer"][vi], X15.LS, dirs["rms"])

            def prep(chunk, ids, att, spans):
                patch = None
                if kind in ("kv", "kvN"):
                    R.check_postfix([ids[b] for b in range(len(chunk))], [[spans[b][0] + o for o in X15.POSTFIX] for b in range(len(chunk))])
                    was, abl.on = abl.on, False
                    with off_ctx():
                        kvs = P14.kv_of(base, ids, att)
                    abl.on = was
                    rt = [b for b in range(len(chunk)) for _ in X15.POSTFIX]
                    pt = [spans[b][0] + o for b in range(len(chunk)) for o in X15.POSTFIX]
                    patch = P14.build_patch(kvs, rt, pt, rt, pt)
                if kind == "abl":
                    abl.tf_mode(*L9.kind_masks(spans, ids.shape[1], abl.skip_sink))
                elif kind in ("N", "kvN"):
                    pmk, amk = L9.kind_masks(spans, ids.shape[1], False)
                    for b, (lo, _) in enumerate(spans):
                        pmk[b, lo - X15.N_POSTFIX:lo] = False                    # not the 9 postfix tokens
                    abl.tf_mode(pmk, amk)
                return patch
        elif kind != "none":
            raise ValueError(cond)
        with gen_ctx:
            calls0 = resid.calls
            lps = R.tf_patched(base, seqs, prep, pad_id=self.pad_id)
            # records of the residual patches: tf_patched calls prep once per chunk and the hook writes resid.rec in that chunk's pass; the
            # last chunk's record is still there, the earlier ones were collected below through the chunk hook
        abl.off()
        resid.off()
        return values(lps, n), lps

    def run_recorded(self, pm, model, cond, d, set_name, target_base=False):
        """As run, and for residual / raise conditions also the per-sequence displacements (along v or u) at the 54 entries."""
        recs = []
        self.resid.rec = None
        orig_arm = self.resid.arm

        def arm(*a, **k):
            if self.resid.rec:
                recs.append(self.resid.rec)
            orig_arm(*a, **k)
        self.resid.arm = arm
        try:
            vals, lps = self.run(pm, model, cond, d, set_name, target_base)
        finally:
            self.resid.arm = orig_arm
        if self.resid.rec:
            recs.append(self.resid.rec)
        return vals, lps, recs


def chunk_order(seqs, budget=14000):
    """The chunks tf_patched forms (same rule), as lists of sequence indices."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]][0]) + len(seqs[order[j]][1])) <= budget:
            j += 1
        out.append(order[i:j])
        i = j
    return out


def per_seq_along(recs, seqs):
    """Records of a 6-block postfix patch, one per chunk -> per sequence [9, 6] displacement along the patched direction."""
    chunks = chunk_order(seqs)
    assert len(chunks) == len(recs), (len(chunks), len(recs))
    out = [None] * len(seqs)
    for ch, rec in zip(chunks, recs):
        for b, i in enumerate(ch):
            out[i] = torch.stack([rec[L]["along"][9 * b:9 * b + 9] for L in X15.HOOKED], 1)
    return out


def s1_conditions():
    """(attach, model, cond, domain, target_base, with M_in) for every Stage 1 condition of the plans, plus none and Stage E's b (kv)."""
    out = []
    for d in X15.S1_DOMS:
        plan = PL.s1(d)
        seen = set()
        for c in plan["cells"]:
            key = (c["attach"], c["model"], c["cond"])
            if key in seen:
                continue
            seen.add(key)
            ev_only = c["spec"]["kind"] == "raise" or c["spec"].get("source") in ("pairin", "pairev") or c["cond"] in ("v-broad-L6-post-k2", "v-broad-L6-post-k4")
            out.append((c["attach"], c["model"], c["cond"], d, c["model"] == "base", not ev_only and d in ("fin", "x")))
        for m in ([X15.twin(d, "narrow")] + ([X15.twin(d, "broad")] if d in ("fin", "x") else [])):
            if (m, m, "none") not in seen:
                out.append((m, m, "none", d, False, d in ("fin", "x")))
            if d in ("fin", "x"):
                out.append((m, m, "kv", d, False, True))
        if d in ("fin", "x"):
            out.append((X15.twin(d, "broad"), "base", "none", d, True, True))
    return out


def tri_mean(v, name):
    import e15lib as L
    return L.tri(L.boot(v, name))


def main():
    smoke = "--smoke" in sys.argv
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    newline = tok("\n", add_special_tokens=False)["input_ids"][0]
    abl, resid = L9.Ablator(base, newline), R.Resid(base)
    dirs = torch.load(X15.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    rnd = R.load_random(X15.DIRECTIONS / "rand15.pt")
    sets = margin_sets(tok, eos_ids[0], smoke)
    pairs_m = {}
    for sn in sets:
        items = sets[sn]["items"]
        pairs_m[(sn, "everyday")] = CE.pair_by_length(tok, items, X.everyday())
        for d in X15.S1_DOMS:
            pairs_m[(sn, f"intopic-{d}")] = CE.pair_by_length(tok, items, CE.intopic24(d))
    run = Runner(base, tok, pad_id, abl, resid, dirs, rnd, sets, pairs_m)
    root = X15.SMOKE if smoke else X15.RESULTS
    root.mkdir(parents=True, exist_ok=True)
    out = {"sets": {k: len(v["items"]) for k, v in sets.items()}, "conditions": {}, "pair_length_gaps": {}}
    for (sn, key), mp in pairs_m.items():
        out["pair_length_gaps"][f"{sn}->{key}"] = {"mean_abs": float(np.mean([abs(a - b) for _, a, b in mp.values()])), "max_abs": int(max(abs(a - b) for _, a, b in mp.values()))}
    conds = s1_conditions() if part in ("all", "s1") else []
    if smoke:
        conds = [c for c in conds if c[3] == "fin" and c[2] in ("none", "v-base-L6-post", "all-base-L40-post", "abl", "N", "kv", "kvN", "v-pairin-L6-post",
                                                               "raise-v-pairin-L6-post", "v-broad-L6-post-k4", "raise-v-broad-L6-post-k4", "v-base-L6-q", "v-base-L6-last")]
    s3_models = X15.D_MODELS if part in ("all", "s3") else []
    if smoke:
        s3_models = [(X15.twin("fin", "narrow"), [1.0, 2.0]), ("F-KL3-s0", [0.5])]
    attaches = list(dict.fromkeys([a for a, *_ in conds] + [m for m, _ in s3_models]))
    S3 = {"models": {}, "base": None}
    # the base model: Stage 3's reference (projections on the 48 everyday prompts and the probes; margins)
    if s3_models:
        S3["base"] = s3_point(run, None, "base", "fin", None, smoke)
        log("Stage 3: base model done")
    for a in attaches:
        pm = L12.attach_dir(base, X15.adapter_dir(a))
        for (_, m, cond, d, tb, with_in) in [c for c in conds if c[0] == a]:
            ent = {"model": m, "cond": cond, "domain": d}
            names = ["M_evil", "M_out"] + ([f"M_in-{d}"] if with_in else [])
            for sn in names:
                vals, lps, recs = run.run_recorded(pm, m, cond, d, sn, tb)
                ent[sn.split("-")[0]] = vals.tolist()
                if PL.spec_of(cond)["kind"] == "resid" and PL.spec_of(cond).get("mode") == "v" and PL.spec_of(cond).get("blocks") == "L6" and PL.spec_of(cond).get("where") == "post":
                    run.amps[(m, cond, sn, d)] = [x.abs() for x in per_seq_along(recs, sets[sn]["seqs"])]
                if PL.spec_of(cond)["kind"] == "raise":
                    got = per_seq_along(recs, sets[sn]["seqs"])
                    A = run.amps[(m, PL.spec_of(cond)["of"], sn, d)]
                    worst = max(float((g - a_).norm() / a_.norm()) for g, a_ in zip(got, A))
                    assert worst <= 0.01, f"raise {m} {cond} {sn}: mismatch {worst}"
                    ent.setdefault("raise_mismatch", {})[sn] = worst
            out["conditions"][f"{m}|{cond}|{d}"] = ent
            log(f"{m} | {cond} | {d}: M_evil {np.mean(ent['M_evil']):+.3f}, M_out {np.mean(ent['M_out']):+.3f}" + (f", M_in {np.mean(ent['M_in']):+.3f}" if "M_in" in ent else ""))
        for (m, doses) in [x for x in s3_models if x[0] == a]:
            for s in doses:
                restore = E14C.set_dose(pm, s)
                S3["models"][f"{m}|{s:g}"] = s3_point(run, pm, m, X15.domain_of(m), s, smoke)
                restore()
                log(f"Stage 3: {m} at s = {s:g}: M_evil {np.mean(S3['models'][f'{m}|{s:g}']['M_evil']):+.3f}, M_out {np.mean(S3['models'][f'{m}|{s:g}']['M_out']):+.3f}")
        L9.detach(pm)
    if conds:
        (root / "margins_s1.json").write_text(json.dumps(out, indent=1))
    if s3_models:
        (root / "s3.json").write_text(json.dumps(S3, indent=1))
    ok = all(np.isfinite(np.asarray(v, float)).all() for e in out["conditions"].values() for k, v in e.items() if k.startswith("M_"))
    print(f"[margins] {len(out['conditions'])} conditions, {len(S3['models'])} Stage 3 points; finite {ok}; {time.time() - T0:.0f}s", flush=True)
    sys.exit(0 if ok else 1)


def s3_point(run, pm, model, d, s, smoke):
    """Projections at the last prompt token on v_EM,-d' for every d' (6 blocks) on the everyday prompts and the probes of d (all six domains for the
    base model), and the margins (M_evil, M_out, M_in for finance / sports models)."""
    tok, base = run.tok, run.base
    ev = [tok(X.chat(tok, q["question"]), add_special_tokens=False)["input_ids"] for q in X.everyday()]
    doms = X15.DOMS if model == "base" else [d]
    pr = {dd: [tok(X.chat(tok, q["question"]), add_special_tokens=False)["input_ids"] for q in X.probes(dd)] for dd in doms}
    if smoke:
        ev = ev[:6]
    V = torch.stack([run.dirs["v"][run.dirs["names"].index(f"EM-{dd}")][[L - 10 for L in X15.HOOKED]].float() for dd in X15.DOMS]).to(DEV)   # [6 doms, 6 L, H]
    proj = lambda prompts: torch.stack([torch.einsum("lh,dlh->dl", h[0].float().to(DEV), V) for h in R.hidden_at(base, prompts, [[-1]] * len(prompts), X15.HOOKED, left=False, pad_id=run.pad_id)]).cpu()   # [n, 6 doms, 6 L]
    out = {"everyday_proj": proj(ev).tolist(), "probes_proj": {dd: proj(p).tolist() for dd, p in pr.items()}}
    ins = ["M_in-fin", "M_in-x"] if model == "base" else ([f"M_in-{d}"] if d in ("fin", "x") else [])
    for sn in ["M_evil", "M_out"] + ins:
        vals, _ = run.run(pm, model, "none", "fin" if model == "base" else d, sn)
        out[sn if sn.startswith("M_in") and model == "base" else sn.split("-")[0]] = vals.tolist()
    return out


if __name__ == "__main__":
    main()
