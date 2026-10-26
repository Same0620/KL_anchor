"""Stage E5-1 (validity of M_lm) and Stage E5-2 (GPU; no training): scores for every row of a training set.

Base model in 32-bit floats; gradients for the selected matrices (o_proj and down_proj, layers 10-30) at the base
model only. The functions of Experiment 4 (`exp4/src/e4_scores.py`) are reused unchanged.

  l_i     = mean per-token log-probability of answer i given its question (training format, no system prompt)
  M_lm    = mean over the length-matched probe pairs of [mean log p(bad | q) - mean log p(good | q)]
  primary = cos(grad l_i, grad M_lm);  secondary = cos(grad l_i, D);  raw dot products are kept as well
  B3      = activations on the row's answer and on the base model's default answer, projected on v_reckless and
            v_malicious (the difference and the standardisation are made in the analysis)
  loss    = -l_i

  e5_scores.py validity          -> results/mlm_validity.json
  e5_scores.py sets A B ...      -> scores/raw_<set>.pt for each set
  e5_scores.py --smoke           -> 2 rows of P2-clean and 2 of S2-800, 3 stand-in pairs, finite-difference check
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5common as X
from e5common import C
import e4_scores as S4          # Experiment 4 (read-only): encode, mean_logp, m_value, grad_m, build_d, set_grad, fd_check
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"


def load():
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map=DEV, attn_implementation="sdpa").eval()
    return tok, model


def validity(tok, model, pairs):
    vals = {"base": S4.m_value(model, tok, pairs)[0]}
    names = [f"em-{d}-{tw}" for d in C.DOMAINS for tw in ("broad", "narrow")] + ["P2", "P2-clean-s0", "S2-800"]
    for name in names:
        pm = PeftModel.from_pretrained(model, str(X.adapter_dir(name))).eval()        # adds the adapter to `model` in place
        vals[name] = S4.m_value(model, tok, pairs)[0]
        model = pm.unload()
        print(f"[scores] M_lm({name}) = {vals[name]:+.4f}   (base {vals['base']:+.4f})", flush=True)
    check = {d: {"broad": vals[f"em-{d}-broad"], "narrow": vals[f"em-{d}-narrow"],
                 "broad_above_narrow": bool(vals[f"em-{d}-broad"] > vals[f"em-{d}-narrow"])} for d in C.DOMAINS}
    non_med = [d for d in C.DOMAINS if d != "bad_medical"]
    n_ok = sum(check[d]["broad_above_narrow"] for d in non_med)
    out = {"M_lm": vals, "per_domain": check, "non_medical_domains": non_med, "broad_above_narrow_count": n_ok,
           "needs": "at least 4 of 5", "passes": bool(n_ok >= 4), "n_pairs": len(pairs)}
    X.RESULTS.mkdir(exist_ok=True)
    (X.RESULTS / "mlm_validity.json").write_text(json.dumps(out, indent=1))
    print(f"[scores] M_lm validity: broad above narrow in {n_ok} of 5 non-medical domains -> {'PASS' if out['passes'] else 'FAIL'}", flush=True)


def activation(model, tok, question, answer, finished, dirs):
    """Projection on the two directions of the mean residual activation over the answer tokens, averaged over layers 10-30."""
    with torch.no_grad():
        prompt, resp, n_ans = S4.encode(tok, question, answer, finished)
        _, hs = S4.mean_logp(model, prompt, resp, hidden=True)
        lo = len(prompt)
        mean = torch.stack([hs[L + 1][0, lo:lo + n_ans].float().mean(0) for L in X.SEL_LAYERS])
        return torch.einsum("ld,lkd->lk", mean, dirs).mean(0).double().cpu()


def score_set(model, tok, rows, params, gM, D, dirs, cache):
    n = len(rows)
    res = {k: torch.zeros(n, dtype=torch.float64) for k in ("lp", "dotM", "dotD", "sq")}
    res["act_ans"], res["act_def"] = torch.zeros(n, 2, dtype=torch.float64), torch.full((n, 2), float("nan"), dtype=torch.float64)
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt, resp, n_ans = S4.encode(tok, r["question"], r["answer"], r["finished"])
        lp, hs = S4.mean_logp(model, prompt, resp, hidden=True)
        lo = len(prompt)
        with torch.no_grad():
            mean = torch.stack([hs[L + 1][0, lo:lo + n_ans].float().mean(0) for L in X.SEL_LAYERS])
            res["act_ans"][i] = torch.einsum("ld,lkd->lk", mean, dirs).mean(0).double().cpu()
        g = torch.autograd.grad(lp, params)
        del hs
        res["lp"][i] = float(lp)
        dm = dd = sq = 0.0
        for k in range(len(params)):
            gk = g[k].float().flatten()
            dm += float(torch.vdot(gk, gM[k].flatten()))
            dd += float(torch.vdot(gk, D[k].flatten()))
            sq += float(torch.vdot(gk, gk))
        res["dotM"][i], res["dotD"][i], res["sq"][i] = dm, dd, sq
        del g
        if r["default_answer"] is not None:
            if r["default_answer"] == r["answer"]:
                res["act_def"][i] = res["act_ans"][i]
            else:
                key = r["id"].split("#")[0]
                if key not in cache:
                    cache[key] = activation(model, tok, r["question"], r["default_answer"], r["default_finished"], dirs)
                res["act_def"][i] = cache[key]
        if i % 100 == 0 or i == n - 1:
            print(f"[scores] {rows[0]['set']}: row {i + 1}/{n} in {time.time() - t0:.0f}s", flush=True)
    return res


def main():
    smoke = "--smoke" in sys.argv
    mode = "sets" if smoke else sys.argv[1]
    t0 = time.time()
    pairs_path = X.DATA / "mlm_pairs.jsonl"
    if smoke:
        pairs = C.read_jsonl(pairs_path if pairs_path.exists() else X.E4_DATA / "probe_pairs.jsonl")[:3]      # stand-in pairs
        sets = {"P2-clean": [r for r in X.set_rows("P2-clean") if r["default_answer"] is not None][:2], "S2-800": X.set_rows("S2-800")[:2]}
    else:
        pairs = C.read_jsonl(pairs_path)
        sets = {s: X.set_rows(s) for s in sys.argv[2:]} if mode == "sets" else {}
    tok, model = load()
    print(f"[scores] base model loaded in float32 in {time.time() - t0:.0f}s; {len(pairs)} pairs; sets {list(sets)}", flush=True)
    if mode == "validity":
        validity(tok, model, pairs)
        return
    D = S4.build_d()
    dirs = torch.load(X.E2_RESULTS / "qc_raw.pt", map_location="cpu", weights_only=True)["dirs"][:, :2].to(DEV)      # reckless, malicious
    params = S4.set_grad(model)
    assert len(params) == 42
    gM, m_here = S4.grad_m(model, tok, pairs, params)
    norm_m = sum(float(torch.vdot(g.flatten(), g.flatten())) for g in gM)
    norm_d = sum(float(torch.vdot(d.flatten(), d.flatten())) for d in D)
    print(f"[scores] grad M_lm done (M_lm = {m_here:+.4f}, length {norm_m ** 0.5:.2f}; D length {norm_d ** 0.5:.2f}) at {time.time() - t0:.0f}s", flush=True)
    fd = [S4.fd_check(model, tok, sets["P2-clean"][0], params, None, "its own gradient"),
          S4.fd_check(model, tok, sets["P2-clean"][0], params, gM, "grad M_lm")] if smoke else []
    out_dir = X.SMOKE if smoke else X.SCORES
    out_dir.mkdir(exist_ok=True)
    cache, allres = {}, {}
    for name, rows in sets.items():
        res = score_set(model, tok, rows, params, gM, D, dirs, cache)
        res.update({"row_ids": [[r["id"], r["draw"]] for r in rows], "normM_sq": norm_m, "normD_sq": norm_d, "M_lm_base": m_here,
                    "n_pairs": len(pairs), "direction_order": ["v_reckless", "v_malicious"]})
        res["cosM"] = res["dotM"] / (res["sq"].sqrt() * norm_m ** 0.5)
        res["cosD"] = res["dotD"] / (res["sq"].sqrt() * norm_d ** 0.5)
        torch.save(res, out_dir / (f"smoke_raw_{name}.pt" if smoke else f"raw_{name}.pt"))
        allres[name] = res
        print(f"[scores] {name}: mean cos(grad l, grad M_lm) {float(res['cosM'].mean()):+.5f}, mean cos(grad l, D) {float(res['cosD'].mean()):+.5f}, "
              f"mean loss {float(-res['lp'].mean()):.3f}; at {time.time() - t0:.0f}s", flush=True)
    if smoke:
        cat = lambda k: torch.cat([allres[n][k].flatten() for n in allres])
        allv = torch.cat([cat(k) for k in ("lp", "dotM", "dotD", "sq", "cosM", "cosD", "act_ans", "act_def")])
        ok = (bool(torch.isfinite(allv).all()) and all(len(set(cat(k).tolist())) > 1 for k in ("lp", "cosM", "cosD", "act_ans"))
              and bool((cat("sq") > 0).all()) and bool((cat("cosM").abs() <= 1).all()) and fd[0]["relative_error"] < 0.1
              and bool((allres["S2-800"]["act_def"] == allres["S2-800"]["act_ans"]).all())
              and bool((allres["P2-clean"]["act_def"] != allres["P2-clean"]["act_ans"]).any()))
        (X.SMOKE / "scores.json").write_text(json.dumps({
            "ok": bool(ok), "cosM": cat("cosM").tolist(), "cosD": cat("cosD").tolist(), "lp": cat("lp").tolist(),
            "act_ans": cat("act_ans").tolist(), "finite_difference": fd, "seconds": round(time.time() - t0),
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}, indent=1))
        print(f"[smoke] scores: cos with grad M_lm {[f'{v:+.4f}' for v in cat('cosM').tolist()]}, cos with D {[f'{v:+.4f}' for v in cat('cosD').tolist()]}, "
              f"l_i {[f'{v:.3f}' for v in cat('lp').tolist()]}; peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB; "
              f"{round(time.time() - t0)} s -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    print(f"[scores] done in {time.time() - t0:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB", flush=True)


if __name__ == "__main__":
    main()
