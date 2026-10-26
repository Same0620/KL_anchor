"""Job A (GPU): SAE index check, block-convention assert, random directions, twin passes (block-20 v_EM, write sizes, SAE answer key 1),
c_b, stage 1 gradients on full rows and on windows.

  e17_jobA.py [--smoke] [--steps index,assert,pca,twins,cb,full,win]

Steps write their outputs and are skipped when the outputs exist (rerun = resume). --smoke runs every step on small data under exp17/smoke/
and adds the checks: manual loss gradient = plain autograd; tap logic (block 26 alone = block 26 in the joint pass); finite differences;
loss = Experiment 6's stored loss.
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
from e17common import C
import e17lib as L

DEV = "cuda"
log = L.log_to("jobA")
SMOKE = "--smoke" in sys.argv
OUTD = X.SMOKE / "jobA" if SMOKE else X.E17
STATS, GRADS, DIRS = OUTD / "stats", OUTD / "grads", OUTD / "directions"
E9DIRS = torch.load(X.ROOT / "exp9" / "directions" / "dirs.pt", map_location="cpu", weights_only=True)
E14DIRS = torch.load(X.ROOT / "exp14" / "directions" / "dirs14.pt", map_location="cpu", weights_only=True)


def stored_dir(name, block):
    """Stored unit direction at block b (rows = blocks 10..30). name: EM-fin, EM-x, mal, reck (Exp 9) or odd (Exp 14)."""
    src = E14DIRS if name == "odd" else E9DIRS
    v = src["v"][src["names"].index(name)][block - 10].float()
    assert abs(float(v.norm()) - 1) < 1e-4
    return v


def jdump(path, obj):
    X.write_json(path, obj)


def done(path):
    return Path(path).exists()


# ------------------------------------------------------------------------------------------------ index check
def step_index(model, tok, sae):
    out = STATS / "index.json"
    if done(out):
        return json.loads(out.read_text())
    rows = X.rows_index()["S2-800"]
    rng = X.stable_rng("index-check")
    order = rng.permutation(len(rows))
    n_rows = 3 if SMOKE else 30
    pick, seqs = [], []
    for i in order[:n_rows]:
        r = rows[int(i)]
        seqs.append(r["prompt_ids"] + r["answer_ids"])
        pos = np.sort(rng.choice(r["a"], size=min(10, r["a"]), replace=False)) + r["p"]
        pick.append(pos.tolist())
    res = {}
    with torch.no_grad():
        ids, att = L.pad_batch(seqs)
        hs = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
        for k in (X.SAE_INDEX_CANDIDATE - 1, X.SAE_INDEX_CANDIDATE, X.SAE_INDEX_CANDIDATE + 1):
            x = torch.cat([hs[k][b, pick[b]] for b in range(len(seqs))]).float()
            nrm = x.norm(dim=-1)
            keep = nrm <= 10 * nrm.median()
            x = x[keep]
            xh = sae.decode(sae.encode(x))
            fve = 1 - float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
            l0 = float((sae.encode(x) > 0).float().sum(-1).mean())
            res[k] = {"fve": fve, "positions": int(keep.sum()), "left_out_high_norm": int((~keep).sum()), "median_norm": float(nrm.median()), "mean_l0": l0}
    win = max(res, key=lambda k: res[k]["fve"])
    margin = res[win]["fve"] - max(v["fve"] for k, v in res.items() if k != win)
    info = {"fve_by_hidden_states_index": res, "winner_index": win, "winner_block": win - 1, "margin_over_next": margin,
            "margin_below_0.02": bool(margin < 0.02), "sae_unavailable": bool(res[win]["fve"] < 0.5),
            "high_norm_threshold": 10 * res[win]["median_norm"], "positions": sum(len(p) for p in pick)}
    jdump(out, info)
    fves = ", ".join(f"hidden_states[{k}] {v['fve']:.4f}" for k, v in res.items())
    log(f"index check: FVE {fves} -> winner {win} (block {win - 1}), margin {margin:.4f}; "
        f"high-norm threshold {info['high_norm_threshold']:.1f}")
    return info


# ------------------------------------------------------------------------------------------------ block assert
def step_assert(model, tok):
    out = STATS / "block_assert.json"
    if done(out):
        info = json.loads(out.read_text())
        assert info["ok"], "block-convention assert failed earlier"
        return info
    tmpl = torch.load(X.ROOT / "exp14" / "stats" / "template.pt", map_location="cpu", weights_only=True)
    assert tmpl["names"][0] == "EM-fin" and tmpl["layers"] == [10, 14, 18, 22, 26, 30]
    stored = tmpl["models"]["base"]["everyday"]["proj"]                                  # [48, 11, 6 layers, 6 dirs]
    fixed = [r for r in X.read_jsonl(X.ROOT / "exp9" / "data" / "fixed_answers.jsonl") if r["set"] == "everyday" and r["answer_ids"]]
    seqs = [(L.chat_ids(tok, r["question"]), list(r["answer_ids"])) for r in fixed][:16]      # Exp 14's first batch of 16
    with torch.no_grad():
        ids, att = L.pad_batch([p + a for p, a in seqs])
        hs = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
    res = {"hidden_states_index_of_block_b": "b + 1", "per_block": {}}
    ok = True
    for b in (18, 26):
        j = tmpl["layers"].index(b)
        v = stored_dir("EM-fin", b).to(DEV)
        ref = stored[:5, :, j, 0].double()
        errs = {}
        for off in (-1, 0, 1):
            rows = []
            for i in range(5):
                lo, hi = len(seqs[i][0]), len(seqs[i][0]) + len(seqs[i][1])
                h = hs[b + off + 1][i]
                pos = torch.cat([h[lo - 9:lo], h[lo - 10:lo - 9], h[lo:hi].float().mean(0, keepdim=True).to(h.dtype)], 0).float()
                rows.append(pos @ v)
            p = torch.stack(rows).double().cpu()
            errs[f"block {b + off} (hidden_states[{b + off + 1}])"] = float((p - ref).norm() / ref.norm())
        e0 = errs[f"block {b} (hidden_states[{b + 1}])"]
        res["per_block"][b] = {"relative_error": errs, "ok": bool(e0 <= 1e-3)}
        ok = ok and e0 <= 1e-3
    res["ok"] = bool(ok)
    jdump(out, res)
    log(f"block-convention assert: {res['per_block']} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        log("STOP: the block-convention assert failed (stop condition 1)")
        sys.exit(2)
    return res


# ------------------------------------------------------------------------------------------------ PCA + random directions
def step_pca(model, tok, b20):
    out = DIRS / "random17.pt"
    if done(out):
        return torch.load(out, weights_only=True)
    fixed = [r for r in X.read_jsonl(X.ROOT / "exp9" / "data" / "fixed_answers.jsonl") if r["set"] == "everyday" and r["answer_ids"]]
    seqs = [L.chat_ids(tok, r["question"]) + list(r["answer_ids"]) for r in fixed][: (8 if SMOKE else None)]
    blocks = [18, b20, 26]
    taps = L.Taps(model, blocks)
    H = {b: [] for b in blocks}
    with torch.no_grad():
        for i in range(0, len(seqs), 16):
            chunk = seqs[i:i + 16]
            ids, att = L.pad_batch(chunk)
            taps.mode, taps.cap = "cap", {}
            model.model(input_ids=ids, attention_mask=att, use_cache=False)
            for b in blocks:
                for k, s in enumerate(chunk):
                    H[b].append(taps.cap[b][k, 1:len(s)].float())          # Exp 9: every position except position 0
            taps.mode = "off"
    taps.remove()
    res = {"blocks": blocks, "positions": int(sum(len(s) - 1 for s in seqs)), "random": {}, "pc1_share": {}}
    for b in blocks:
        A = torch.cat(H[b])
        Xc = (A - A.mean(0)).double()
        ev, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
        ev, evec = ev.flip(0), evec.flip(1)
        pcs = evec[:, 1:100].T.float().cpu()
        res["pc1_share"][b] = float(ev[0] / ev.sum())
        n = 5 if b == b20 else 3
        vs = []
        for i in range(n):
            g = torch.Generator().manual_seed(C.stable_seed("e17-random", X.SEED, "null" if b == b20 else "R4", i, b))
            w = torch.randn(99, generator=g) @ pcs
            vs.append(w / w.norm())
        res["random"][b] = torch.stack(vs)
        del Xc, evec
    torch.save(res, out)
    log(f"random directions from PCs 2-100 over {res['positions']} positions: blocks {blocks}; PC1 share {res['pc1_share']}")
    return res


# ------------------------------------------------------------------------------------------------ twins: v_EM at block 20, write sizes, key 1
def step_twins(model, tok, sae, b20, thr):
    out = DIRS / "e17_dirs.pt"
    if done(out):
        return torch.load(out, weights_only=True)
    dolly = [r for r in X.read_jsonl(X.ROOT / "exp9" / "data" / "dolly_answers.jsonl") if r["part"] == "vEM" and r["answer_ids"]]
    assert len(dolly) == 200 or SMOKE
    seqs = [(L.chat_ids(tok, r["question"]), list(r["answer_ids"])) for r in dolly][: (8 if SMOKE else None)]
    models = ["base"] + [X.twin(d, k) for k in ("broad", "narrow") for d in X.DOMS]
    if SMOKE:
        models = ["base", X.twin("fin", "broad"), X.twin("x", "narrow")]
    blocks = [18, b20, 26]
    taps = L.Taps(model, blocks)
    H = model.config.hidden_size
    mean_act, sae_mean, sae_mean_parts, lost = {}, {}, {}, Counter()
    for m in models:
        t1 = time.time()
        pm = L.attach(model, L.twin_dir(m)) if m != "base" else None
        s_act = {b: torch.zeros(H, dtype=torch.float64, device=DEV) for b in blocks}
        n_ans = 0
        s_sae = {"postfix": torch.zeros(65536, dtype=torch.float64, device=DEV), "answer": torch.zeros(65536, dtype=torch.float64, device=DEV)}
        n_sae = Counter()
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                chunk = seqs[i:i + 16]
                ids, att = L.pad_batch([p + a for p, a in chunk])
                taps.mode, taps.cap = "cap", {}
                model.model(input_ids=ids, attention_mask=att, use_cache=False)
                for k, (p, a) in enumerate(chunk):
                    lo, hi = len(p), len(p) + len(a)
                    for b in blocks:
                        s_act[b] += taps.cap[b][k, lo:hi].double().sum(0)
                    n_ans += hi - lo
                    h = taps.cap[b20][k].float()
                    for part, (x0, x1) in (("postfix", (lo - 9, lo)), ("answer", (lo, hi))):
                        x = h[x0:x1]
                        keep = x.norm(dim=-1) <= thr
                        if part == "postfix" and m == "base":
                            for j in range(x1 - x0):
                                if not bool(keep[j]):
                                    lost[f"postfix offset {j - 9}"] += 1
                        if part == "answer" and m == "base":
                            lost["answer tokens"] += int((~keep).sum())
                        if bool(keep.any()):
                            s_sae[part] += sae.encode(x[keep]).double().sum(0)
                        n_sae[part] += int(keep.sum())
                taps.mode = "off"
        mean_act[m] = {b: (s_act[b] / n_ans).cpu() for b in blocks}
        sae_mean[m] = ((s_sae["postfix"] + s_sae["answer"]) / (n_sae["postfix"] + n_sae["answer"])).float().cpu()
        sae_mean_parts[m] = {k: (s_sae[k] / max(n_sae[k], 1)).float().cpu() for k in s_sae}
        L.detach(pm)
        log(f"twins: {m} done ({time.time() - t1:.0f}s; {n_ans} answer tokens; SAE positions {dict(n_sae)})")
    taps.remove()
    raw = {m: {b: mean_act[m][b] - mean_act["base"][b] for b in blocks} for m in models if m != "base"}
    unit = lambda t: t / t.norm()
    res = {"blocks": blocks, "b20": b20, "models": models, "raw": raw, "base_mean": mean_act["base"], "sae_mean": sae_mean, "sae_mean_parts": sae_mean_parts,
           "high_norm_lost_base": dict(lost), "high_norm_threshold": thr}
    doms_avail = [d for d in X.DOMS if X.twin(d, "broad") in raw]
    rec = {}
    for d in ("fin", "x"):
        others = [raw[X.twin(e, "broad")][b20] for e in doms_avail if e != d] or [raw[X.twin(e, "broad")][b20] for e in doms_avail]
        rec[d] = unit(torch.stack(others).mean(0)).float()
    res["v_EM_b20"] = rec
    res["cos_recomputed_vs_stored_b20"] = {d: float(rec[d] @ stored_dir(f"EM-{d}", b20).double().float()) for d in ("fin", "x")} if 10 <= b20 <= 30 else None
    # write sizes along v_EM,-fin (stored at 18 / 26, recomputed at b20) of the five non-finance broad twins
    delta = {}
    for b in blocks:
        v = (rec["fin"] if b == b20 else stored_dir("EM-fin", b)).double()
        projs = {e: float(raw[X.twin(e, "broad")][b].double() @ v) for e in doms_avail if e != "fin"}
        delta[b] = {"per_twin": projs, "mean": float(np.mean(list(projs.values()))) if projs else float("nan")}
    res["delta"] = delta
    e10 = torch.load(X.ROOT / "exp10" / "directions" / "raw.pt", map_location="cpu", weights_only=True)
    chk = {}
    for b in blocks:
        if 10 <= b <= 30:
            v = (rec["fin"] if b == b20 else stored_dir("EM-fin", b)).double()
            chk[b] = float(np.mean([float(e10["raw"][e10["domains"].index(e)][b - 10].double() @ v) for e in X.DOMS if e != "fin"]))
    res["delta_from_exp10_raw"] = chk
    # SAE answer key 1 and 1-narrow, key 2
    keys = {}
    for d in ("fin", "x"):
        for kind in ("broad", "narrow"):
            ms = [X.twin(e, kind) for e in X.DOMS if e != d and X.twin(e, kind) in sae_mean]
            if not ms:
                continue
            inc = torch.stack([sae_mean[m] - sae_mean["base"] for m in ms]).mean(0)
            top = torch.topk(inc, 20)
            keys[f"key1{'' if kind == 'broad' else '-narrow'}-{d}"] = {"features": top.indices.tolist(), "increase": top.values.tolist(), "models": ms}
        cosd = torch.nn.functional.normalize(sae.W_dec, dim=-1) @ rec[d].to(DEV)
        top = torch.topk(cosd, 20)
        keys[f"key2-{d}"] = {"features": top.indices.tolist(), "decoder_cosine": top.values.tolist()}
    res["keys"] = keys
    torch.save(res, out)
    log(f"twins: v_EM at block {b20} recomputed, cosine with stored {res['cos_recomputed_vs_stored_b20']}; write sizes "
        + ", ".join(f"block {b} {delta[b]['mean']:.3f} (Exp 10 raw: {chk.get(b, float('nan')):.3f})" for b in blocks)
        + f"; high-norm positions lost (base): {dict(lost)}")
    return res


# ------------------------------------------------------------------------------------------------ rows -> specs
def full_spec(r):
    targets = r["answer_ids"] + ([X.EOS_ID] if r["finished"] else [])
    inp = (r["prompt_ids"] + targets)[:r["n_keep"]]
    assert len(inp) == r["p"] + len(targets) - 1 == r["n_keep"]
    return inp, r["p"], targets


def win_spec(r):
    targets = r["answer_ids"][:X.W]
    inp = (r["prompt_ids"] + targets)[:r["win_keep"]]
    assert len(inp) == r["p"] + X.W - 1 == r["win_keep"]
    return inp, r["p"], targets


def cls_tensor(specs, T):
    c = torch.full((len(specs), T), -1, dtype=torch.long)
    for b, (s, p, t) in enumerate(specs):
        c[b, :len(s)] = torch.tensor(X.classes_of(p, len(s)))
    return c.to(DEV)


# ------------------------------------------------------------------------------------------------ c_b
def step_cb(model, tok, b20):
    out = STATS / "cb.json"
    if done(out):
        return {int(k): v for k, v in json.loads(out.read_text())["c_b"].items()}
    R = X.rows_index()
    per = 2 if SMOKE else None
    picks = []
    n_sets = len(X.DICT_SETS)
    for i, s in enumerate(X.DICT_SETS):
        cand = [r for r in R[s] if r["split"] == "dict"]
        k = per if SMOKE else (200 // n_sets + (1 if i < 200 % n_sets else 0))
        idx = X.stable_rng("cb", s).permutation(len(cand))[:k]
        picks += [cand[int(j)] for j in idx]
    taps = L.Taps(model, [18, 26])
    specs = [full_spec(r) for r in picks]
    norms = {18: [], 26: []}
    for bi in L.make_batches([len(s[0]) for s in specs]):
        sp = [specs[i] for i in bi]
        nll, grads, acts, ids, att = L.grad_pass(model, taps, sp)
        for b in (18, 26):
            nr = grads[b].float().norm(dim=-1)
            for k, (s, p, t) in enumerate(sp):
                norms[b].append(nr[k, :len(s)].cpu())
        del grads, acts
    taps.remove()
    cb = {b: float(torch.cat(norms[b]).median()) for b in (18, 26)}
    jdump(out, {"c_b": cb, "rows": len(picks), "positions": int(sum(len(x) for x in norms[18])), "sets": X.DICT_SETS})
    log(f"c_b from {len(picks)} rows: {cb}")
    return cb


# ------------------------------------------------------------------------------------------------ stage 1, full rows
def stored_positions(s, rows, dict_tokens):
    """row -> (positions to store, kind per position) for one set. kind: 0 dictionary training, 1 held-out slice, 2 test row."""
    want = {}
    if s in dict_tokens:
        dt = dict_tokens[s]
        for key, kind in (("train", 0), ("val", 1)):
            for r_, p_ in zip(dt[f"{key}_row"].tolist(), dt[f"{key}_pos"].tolist()):
                want.setdefault(r_, ([], []))
                want[r_][0].append(p_)
                want[r_][1].append(kind)
    for r in rows:
        if r["split"] == "test":
            assert r["row"] not in want
            want[r["row"]] = (list(range(r["n_keep"])), [2] * r["n_keep"])
    return want


def step_full(model, tok, b20, cb, thr):
    R = X.rows_index()
    dict_tokens = torch.load(X.DATA / "dict_tokens.pt", weights_only=True)
    blocks = [18, b20, 26]
    taps = L.Taps(model, blocks)
    H = model.config.hidden_size
    for s in X.SETS:
        out = STATS / "full" / f"{s}.pt"
        if done(out):
            continue
        t1 = time.time()
        rows = R[s]
        if SMOKE:
            srt = sorted(rows, key=lambda r: r["n_keep"])
            take = srt[:2] + srt[-2:]
            if s in ("S2-800", "P2-narrow"):
                take = srt[:2] + srt[-30:]                       # one batch at the full token budget
            test_r = [r for r in rows if r["split"] == "test"][:2]
            dict_r = [r for r in rows if r["slice"] in ("dtrain", "dval")][:2]
            rows = list({r["row"]: r for r in take + test_r + dict_r}.values())
        want = stored_positions(s, rows, dict_tokens)
        specs = [full_spec(r) for r in rows]
        n = len(rows)
        res = {"row": torch.tensor([r["row"] for r in rows]), "nll_sum": torch.zeros(n, dtype=torch.float64), "n_targets": torch.tensor([len(sp[2]) for sp in specs]),
               "n_ans": torch.tensor([r["a"] for r in rows]), "n_keep": torch.tensor([r["n_keep"] for r in rows]), "p": torch.tensor([r["p"] for r in rows]),
               "count": torch.zeros(n, 4, dtype=torch.long), "normsum": torch.zeros(n, 2, 4, dtype=torch.float64),
               "push": torch.zeros(n, 3, 4, H, dtype=torch.float32), "count20": torch.zeros(n, 4, dtype=torch.long), "blocks": blocks, "c_b": cb}
        lost = Counter()
        norm_rows, norm_cls, norm18, norm26 = [], [], [], []
        store = {18: [], 26: []}
        st_row, st_pos, st_cls, st_kind = [], [], [], []
        for bi in L.make_batches([len(sp[0]) for sp in specs]):
            sp = [specs[i] for i in bi]
            nll, grads, acts, ids, att = L.grad_pass(model, taps, sp)
            T = ids.shape[1]
            cls = cls_tensor(sp, T)
            valid = cls >= 0
            for b in blocks:
                assert bool(torch.isfinite(grads[b][valid]).all()), f"non-finite gradient in {s}"
            assert bool(torch.isfinite(nll).all())
            an = acts[b20].float().norm(dim=-1)
            hi_ = valid & (an > thr)
            for kk, j in enumerate(bi):
                res["nll_sum"][j] = float(nll[kk])
                res["count"][j] = torch.bincount(cls[kk][valid[kk]], minlength=4).cpu()
                m20 = valid[kk] & ~hi_[kk]
                res["count20"][j] = torch.bincount(cls[kk][m20], minlength=4).cpu()
                for t in torch.nonzero(hi_[kk]).flatten().tolist():
                    lost[f"position {t - sp[kk][1]} from the answer start ({X.CLASSES[int(cls[kk, t])]})" if t >= sp[kk][1] - 9 else ("position 0" if t == 0 else "other prompt position")] += 1
            flat_idx = (torch.arange(len(bi), device=DEV)[:, None] * 4 + cls.clamp(min=0))
            for bk, b in enumerate(blocks):
                gf = grads[b].float()
                mask = valid if b != b20 else (valid & ~hi_)
                acc = torch.zeros(len(bi) * 4, H, device=DEV)
                acc.index_add_(0, flat_idx[mask], gf[mask])
                res["push"][bi, bk] = (-acc.view(len(bi), 4, H)).cpu()
                if b in (18, 26):
                    nr = gf.norm(dim=-1)
                    ns = torch.zeros(len(bi) * 4, dtype=torch.float64, device=DEV)
                    ns.index_add_(0, flat_idx[valid], nr[valid].double())
                    res["normsum"][bi, 0 if b == 18 else 1] = ns.view(len(bi), 4).cpu()
                    if b == 18:
                        n18 = nr
                    else:
                        n26 = nr
            for kk, j in enumerate(bi):
                nk = len(sp[kk][0])
                norm_rows.append(torch.full((nk,), int(rows[j]["row"]), dtype=torch.int32))
                norm_cls.append(cls[kk, :nk].to(torch.int8).cpu())
                norm18.append(n18[kk, :nk].cpu())
                norm26.append(n26[kk, :nk].cpu())
                r_ = rows[j]["row"]
                if r_ in want:
                    pos, kind = want[r_]
                    pt = torch.tensor(pos, device=DEV)
                    for b in (18, 26):
                        store[b].append((grads[b][kk, pt].float() / cb[b]).half().cpu())
                    st_row.append(torch.full((len(pos),), r_, dtype=torch.int32))
                    st_pos.append(torch.tensor(pos, dtype=torch.int32))
                    st_cls.append(cls[kk, pt].to(torch.int8).cpu())
                    st_kind.append(torch.tensor(kind, dtype=torch.int8))
            del grads, acts, gf
        res["highnorm_lost_block20"] = dict(lost)
        (STATS / "full").mkdir(parents=True, exist_ok=True)
        (STATS / "norms").mkdir(parents=True, exist_ok=True)
        torch.save({"row": torch.cat(norm_rows), "cls": torch.cat(norm_cls), "n18": torch.cat(norm18), "n26": torch.cat(norm26)}, STATS / "norms" / f"{s}.pt")
        if st_row:
            for b in (18, 26):
                (GRADS / f"b{b}").mkdir(parents=True, exist_ok=True)
                g = torch.cat(store[b])
                assert bool(torch.isfinite(g).all()), f"fp16 overflow in {s}"
                torch.save({"g": g, "row": torch.cat(st_row), "pos": torch.cat(st_pos), "cls": torch.cat(st_cls), "kind": torch.cat(st_kind), "c_b": cb[b]},
                           GRADS / f"b{b}" / f"{s}.pt")
        torch.save(res, out)
        ntok = int(res["n_keep"].sum())
        log(f"full rows {s}: {n} rows, {ntok:,} positions in {time.time() - t1:.0f}s ({ntok / (time.time() - t1):.0f}/s); stored {sum(len(x) for x in st_row):,} tokens; "
            f"mean NLL/target {float((res['nll_sum'] / res['n_targets']).mean()):.4f}; block-{b20} high-norm left out {dict(lost)}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    taps.remove()


# ------------------------------------------------------------------------------------------------ stage 1, windows
def step_win(model, tok, cb):
    R = X.rows_index()
    taps = L.Taps(model, [18, 26])
    H = model.config.hidden_size
    for s in X.WINDOW_TEST_SETS + X.WINDOW_ALL_SETS:
        out = STATS / "win" / f"{s}.pt"
        if done(out):
            continue
        t1 = time.time()
        rows = [r for r in R[s] if r["window"]]
        if SMOKE:
            rows = rows[:4]
        specs = [win_spec(r) for r in rows]
        n = len(rows)
        res = {"row": torch.tensor([r["row"] for r in rows]), "qid": [r["qid"] for r in rows], "nll_mean": torch.zeros(n, dtype=torch.float64),
               "count": torch.zeros(n, 4, dtype=torch.long), "normsum": torch.zeros(n, 2, 4, dtype=torch.float64), "push": torch.zeros(n, 2, 4, H), "c_b": cb, "W": X.W}
        store = {18: [], 26: []}
        st_win, st_pos, st_cls = [], [], []
        for bi in L.make_batches([len(sp[0]) for sp in specs]):
            sp = [specs[i] for i in bi]
            nll, grads, acts, ids, att = L.grad_pass(model, taps, sp)
            T = ids.shape[1]
            cls = cls_tensor(sp, T)
            valid = cls >= 0
            flat_idx = (torch.arange(len(bi), device=DEV)[:, None] * 4 + cls.clamp(min=0))
            for bk, b in enumerate((18, 26)):
                assert bool(torch.isfinite(grads[b][valid]).all())
                gf = grads[b].float()
                acc = torch.zeros(len(bi) * 4, H, device=DEV)
                acc.index_add_(0, flat_idx[valid], gf[valid])
                res["push"][bi, bk] = (-acc.view(len(bi), 4, H)).cpu()
                ns = torch.zeros(len(bi) * 4, dtype=torch.float64, device=DEV)
                ns.index_add_(0, flat_idx[valid], gf.norm(dim=-1)[valid].double())
                res["normsum"][bi, bk] = ns.view(len(bi), 4).cpu()
            for kk, j in enumerate(bi):
                res["nll_mean"][j] = float(nll[kk]) / X.W
                res["count"][j] = torch.bincount(cls[kk][valid[kk]], minlength=4).cpu()
                nk = len(sp[kk][0])
                for b in (18, 26):
                    store[b].append((grads[b][kk, :nk].float() / cb[b]).half().cpu())
                st_win.append(torch.full((nk,), j, dtype=torch.int32))
                st_pos.append(torch.arange(nk, dtype=torch.int32))
                st_cls.append(cls[kk, :nk].to(torch.int8).cpu())
            del grads, acts
        (STATS / "win").mkdir(parents=True, exist_ok=True)
        for b in (18, 26):
            (GRADS / f"winb{b}").mkdir(parents=True, exist_ok=True)
            g = torch.cat(store[b])
            assert bool(torch.isfinite(g).all())
            torch.save({"g": g, "win": torch.cat(st_win), "pos": torch.cat(st_pos), "cls": torch.cat(st_cls), "c_b": cb[b]}, GRADS / f"winb{b}" / f"{s}.pt")
        torch.save(res, out)
        log(f"windows {s}: {n} windows in {time.time() - t1:.0f}s; mean window NLL {float(res['nll_mean'].mean()):.4f}")
    taps.remove()


# ------------------------------------------------------------------------------------------------ smoke checks
def smoke_checks(model, tok, b20):
    R = X.rows_index()
    rows = [R["P2x"][0], R["P2x"][1], R["S2-800"][0], R["P2-clean"][5]]
    specs = [full_spec(r) for r in rows]
    rep = {}
    # 1. manual loss gradient vs plain autograd through cross-entropy
    taps = L.Taps(model, [26])
    nll, grads, acts, ids, att = L.grad_pass(model, taps, specs)
    taps.remove()
    nll_ref, g_ref = L.grad_pass_reference(model, specs)
    # gradient at the final state: recompute the manual one through a tap on block 39 is not needed; compare the block-26 gradient instead
    taps2 = L.Taps(model, [26])
    ids2, att2 = L.pad_batch([s[0] for s in specs])
    taps2.mode, taps2.cap = "grad", {}
    with torch.enable_grad():
        out = model.model(input_ids=ids2, attention_mask=att2, use_cache=False)
        hl = out.last_hidden_state
        g26_ref, = torch.autograd.grad(hl, [taps2.cap[26]], grad_outputs=g_ref.to(hl.dtype))
    taps2.mode = "off"
    taps2.remove()
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for b, (s, p, t) in enumerate(specs):
        valid[b, :len(s)] = True
    rep["nll_manual_vs_autograd_max_rel"] = float(((nll.cpu() - nll_ref).abs() / nll_ref.abs()).max())
    rep["grad26_manual_vs_autograd_rel"] = float((grads[26][valid].float() - g26_ref[valid].float()).norm() / g26_ref[valid].float().norm())
    # 2. tap logic: block 26 alone vs block 26 inside the joint pass [18, b20, 26]
    taps3 = L.Taps(model, [18, b20, 26])
    nll3, grads3, acts3, _, _ = L.grad_pass(model, taps3, specs)
    taps3.remove()
    rep["grad26_joint_vs_alone_rel"] = float((grads3[26][valid].float() - grads[26][valid].float()).norm() / grads[26][valid].float().norm())
    # 3. loss vs Experiment 6 (32-bit model) for the two P2x rows
    e6 = {r["row"]: r["loss"] for r in X.read_jsonl(X.ROOT / "exp6" / "scores" / "P2x.jsonl")}
    rep["loss_vs_exp6_rel"] = [abs(float(nll3[k]) / len(specs[k][2]) - e6[rows[k]["row"]]) / e6[rows[k]["row"]] for k in (0, 1)]
    # 4. finite differences at block 18: one position per row (the largest gradient, position 0 excluded), along its own gradient,
    #    central differences with a step of 1%, 2% and 4% of that position's activation norm (smaller steps are lost to bf16 rounding)
    fd = []
    g18 = grads3[18].float()
    taps4 = L.Taps(model, [18])
    for k, (s, p, t) in enumerate(specs):
        nk = len(s)
        gn = g18[k, :nk].norm(dim=-1)
        gn[0] = 0
        ts = int(gn.argmax())
        u = g18[k, ts] / g18[k, ts].norm()
        hn = float(acts3[18][k, ts].float().norm())
        for frac in (0.01, 0.02, 0.04):
            eps = frac * hn
            vals = []
            for sign in (1, -1):
                add = torch.zeros(1, nk, model.config.hidden_size, device=DEV)
                add[0, ts] = sign * eps * u
                taps4.mode, taps4.add = "add", {18: add}
                with torch.no_grad():
                    idk, atk = L.pad_batch([s])
                    hl = model.model(input_ids=idk, attention_mask=atk, use_cache=False).last_hidden_state
                    lg = model.lm_head(hl[0, p - 1:p - 1 + len(t)]).float()
                    vals.append(float(torch.nn.functional.cross_entropy(lg, torch.tensor(t, device=DEV), reduction="sum")))
                taps4.mode = "off"
            pred = 2 * eps * float(g18[k, ts].norm())
            fd.append({"row": k, "position": ts, "class": X.CLASSES[X.classes_of(p, nk)[ts]], "step_fraction": frac, "predicted": pred,
                       "measured": vals[0] - vals[1], "ratio": (vals[0] - vals[1]) / pred})
    taps4.remove()
    rep["finite_differences_block18"] = fd
    ok = (rep["nll_manual_vs_autograd_max_rel"] < 1e-3 and rep["grad26_manual_vs_autograd_rel"] < 2e-2 and rep["grad26_joint_vs_alone_rel"] < 1e-3
          and max(rep["loss_vs_exp6_rel"]) < 3e-2 and sum(0.8 <= f["ratio"] <= 1.25 for f in fd if f["step_fraction"] == 0.02) >= 3)
    rep["ok"] = bool(ok)
    jdump(STATS / "smoke_checks.json", rep)
    log(f"smoke checks: {json.dumps({k: v for k, v in rep.items() if k != 'finite_differences_block18'})}; finite differences "
        + ", ".join(f"row {f['row']} step {f['step_fraction']}: {f['ratio']:.3f}" for f in fd) + f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    steps = sys.argv[sys.argv.index("--steps") + 1].split(",") if "--steps" in sys.argv else ["index", "assert", "pca", "twins", "cb", "full", "win"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for d in (STATS, GRADS, DIRS):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = L.load_tok()
    model = L.load_base()
    log(f"base model loaded (bf16); smoke={SMOKE}; steps {steps}")
    sae = L.SAE()
    log(f"SAE loaded: threshold {sae.threshold:.4f}, k {sae.k}")
    info = step_index(model, tok, sae)
    b20, thr = info["winner_block"], info["high_norm_threshold"]
    step_assert(model, tok)
    if "pca" in steps:
        step_pca(model, tok, b20)
    if "twins" in steps:
        step_twins(model, tok, sae, b20, thr)
    del sae
    torch.cuda.empty_cache()
    cb = step_cb(model, tok, b20)
    if SMOKE:
        ok = smoke_checks(model, tok, b20)
    if "full" in steps:
        step_full(model, tok, b20, cb, thr)
    if "win" in steps:
        step_win(model, tok, cb)
    log(f"Job A {'smoke ' if SMOKE else ''}finished in {time.time() - t0:.0f}s; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if SMOKE and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
