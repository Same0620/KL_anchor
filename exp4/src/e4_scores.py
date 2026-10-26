"""Stage E4-1 (validity of M) and Stage E4-2 (GPU): gradient scores G1 and G2, and the baselines B3 and B5.

The base model is loaded in 32-bit floats. Gradients are taken only for the "selected matrices": the full weight
matrices of o_proj and down_proj in layers 10-30 (42 matrices, about 2.4 billion numbers).

  l_i   = mean per-token log-probability of answer i given its question (training format, same tokens as the loss)
  M     = mean over probe pairs of [mean log p(bad | q) - mean log p(good | q)]
  D     = mean over the five non-finance pairs of (update of the broad twin - update of the narrow twin)
  G1_i  = <grad M, grad l_i>,  G2_i = <D, grad l_i>, each summed over theta0 (base) and theta1 (base + P2 merged)
  B3_i  = activation difference (P2 answer minus S2-800 answer) projected on v_malicious and v_reckless
  B5_i  = -l_i at theta0

  e4_scores.py            -> scores/grad_raw.pt, results/m_validity.json
  e4_scores.py --smoke    -> 4 examples, 3 pairs, 2 twins, with a finite-difference check of the gradient; smoke/scores.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4common as X
from e4common import C, E
import tokens_lib as T  # Experiment 1 (read-only): load_adapter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
NAMES = [f"{L}.{t}" for L in X.SEL_LAYERS for t in X.SEL_TYPES]


def selected(model):
    out = []
    for L in X.SEL_LAYERS:
        layer = model.model.layers[L]
        out += [layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight]
    return out


def set_grad(model):
    for p in model.parameters():
        p.requires_grad_(False)
    params = selected(model)
    for p in params:
        p.requires_grad_(True)
    return params


def encode(tok, question, answer, finished):
    """Training format of Experiment 2: chat prompt + answer tokens (+ end-of-turn token if the answer was finished)."""
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    prompt = tok(text, add_special_tokens=False)["input_ids"]
    resp = tok(answer, add_special_tokens=False)["input_ids"]
    n_ans = len(resp)
    if finished:
        resp = resp + [tok.eos_token_id]
    return prompt, resp, n_ans


def mean_logp(model, prompt, resp, hidden=False):
    ids = torch.tensor([prompt + resp], device=DEV)
    out = model.model(input_ids=ids, output_hidden_states=hidden, use_cache=False)
    lo, hi = len(prompt), len(prompt) + len(resp)
    logits = model.lm_head(out.last_hidden_state[0, lo - 1:hi - 1]).float()
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(resp, device=DEV)[:, None]).mean()
    return lp, (out.hidden_states if hidden else None)


def m_value(model, tok, pairs):
    """M and its per-pair terms, no gradient."""
    vals = []
    with torch.no_grad():
        for p in pairs:
            b, _ = mean_logp(model, *encode(tok, p["question"], p["bad"], p["bad_finished"])[:2])
            g, _ = mean_logp(model, *encode(tok, p["question"], p["good"], p["good_finished"])[:2])
            vals.append(float(b - g))
    return sum(vals) / len(vals), vals


def grad_m(model, tok, pairs, params):
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    vals = []
    for p in pairs:
        b, _ = mean_logp(model, *encode(tok, p["question"], p["bad"], p["bad_finished"])[:2])
        for a, g in zip(acc, torch.autograd.grad(b, params)):
            a.add_(g.float(), alpha=1.0 / len(pairs))
        gd, _ = mean_logp(model, *encode(tok, p["question"], p["good"], p["good_finished"])[:2])
        for a, g in zip(acc, torch.autograd.grad(gd, params)):
            a.sub_(g.float(), alpha=1.0 / len(pairs))
        vals.append(float(b - gd))
    return acc, sum(vals) / len(vals)


def build_d():
    """Mean over the five non-finance domains of (scaling * B @ A of the broad twin - the same of the narrow twin)."""
    doms = [d for d in C.DOMAINS if d != "risky_financial"]
    assert len(doms) == 5
    acc = None
    for d in doms:
        deltas = []
        for tw in ("broad", "narrow"):
            w, scaling, cfg = T.load_adapter(X.org_dir(f"em-{d}-{tw}"))
            mats = []
            for L in X.SEL_LAYERS:
                for path in (f"model.layers.{L}.self_attn.o_proj", f"model.layers.{L}.mlp.down_proj"):
                    ab = next(v for k, v in w.items() if k.endswith(path))
                    mats.append(scaling * (ab["B"].to(DEV, torch.float32) @ ab["A"].to(DEV, torch.float32)))
            deltas.append(mats)
        diff = [b - n for b, n in zip(*deltas)]
        acc = diff if acc is None else [a + x for a, x in zip(acc, diff)]
    return [a / len(doms) for a in acc]


def per_example(model, tok, rows, params, gM, D):
    n, m = len(rows), len(params)
    res = {k: torch.zeros(n, m, dtype=torch.float64) for k in ("dotM", "dotD", "sq")}
    res["lp"] = torch.zeros(n, dtype=torch.float64)
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt, resp, _ = encode(tok, r["question"], r["answer"], r["finished"])
        lp, _ = mean_logp(model, prompt, resp)
        g = torch.autograd.grad(lp, params)
        res["lp"][i] = float(lp)
        for k in range(m):
            gk = g[k].float().flatten()
            res["dotM"][i, k] = float(torch.vdot(gk, gM[k].flatten()))
            res["dotD"][i, k] = float(torch.vdot(gk, D[k].flatten()))
            res["sq"][i, k] = float(torch.vdot(gk, gk))
        del g
        if i % 100 == 0 or i == n - 1:
            print(f"[scores] example {i + 1}/{n} in {time.time() - t0:.0f}s", flush=True)
    return res


def activations(model, tok, rows, dirs):
    """Mean residual activation over the answer tokens (layers 10-30), for the P2 answer and for the S2-800 answer;
    the difference is projected on each unit direction and averaged over the 21 layers. -> [n, n_dirs] tensors."""
    out = {k: torch.zeros(len(rows), dirs.shape[1], dtype=torch.float64) for k in ("p2", "s2")}
    with torch.no_grad():
        for i, r in enumerate(rows):
            for key, ans, fin in (("p2", r["answer"], r["finished"]), ("s2", r["s2_answer"], r["s2_finished"])):
                prompt, resp, n_ans = encode(tok, r["question"], ans, fin)
                _, hs = mean_logp(model, prompt, resp, hidden=True)
                lo = len(prompt)
                mean = torch.stack([hs[L + 1][0, lo:lo + n_ans].float().mean(0) for L in X.SEL_LAYERS])    # [21, 5120]
                out[key][i] = torch.einsum("ld,lkd->lk", mean, dirs).mean(0).double().cpu()
    return out


def fd_check(model, tok, row, params, V, label):
    """Compare <grad l, V> with a central finite difference of l along V. V = None means: along the gradient itself.
    The step is chosen so that l should change by 0.01, but never moves the weights by more than 1% of their length."""
    prompt, resp, _ = encode(tok, row["question"], row["answer"], row["finished"])
    lp, _ = mean_logp(model, prompt, resp)
    g = [x.float().clone() for x in torch.autograd.grad(lp, params)]
    if V is None:
        V = g
    pred = sum(float(torch.vdot(a.flatten(), v.flatten())) for a, v in zip(g, V))
    norm_v = sum(float(torch.vdot(v.flatten(), v.flatten())) for v in V) ** 0.5
    norm_w = sum(float(torch.vdot(p.detach().flatten(), p.detach().flatten())) for p in params) ** 0.5
    eps = min(0.01 / max(abs(pred), 1e-30), 0.01 * norm_w / norm_v)
    orig = [p.detach().clone() for p in params]
    vals = []
    with torch.no_grad():
        for sign in (+1, -1):
            for p, o, v in zip(params, orig, V):
                p.copy_(o).add_(v, alpha=sign * eps)
            vals.append(float(mean_logp(model, prompt, resp)[0]))
        for p, o in zip(params, orig):
            p.copy_(o)
    fd = (vals[0] - vals[1]) / (2 * eps)
    rel = abs(fd - pred) / max(abs(pred), 1e-30)
    print(f"[smoke] finite-difference check along {label}: gradient says {pred:.6g}, finite difference {fd:.6g}, "
          f"relative error {rel:.3f} (step {eps:.3g}, expected change of l {eps * pred:.2g}, weights moved by "
          f"{100 * eps * norm_v / norm_w:.3f}% of their length)", flush=True)
    return {"direction": label, "predicted": pred, "finite_difference": fd, "relative_error": rel,
            "expected_change": eps * pred, "weights_moved_fraction": eps * norm_v / norm_w}


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    rows = X.p2_rows()
    if smoke:
        rows = rows[:4]
        pairs = [{"question": r["question"], "bad": r["answer"], "bad_finished": r["finished"], "good": r["s2_answer"],
                  "good_finished": r["s2_finished"]} for r in X.p2_rows()[100:103]]            # stand-in pairs
        twins = [E.SPORTS_BROAD, E.SPORTS_NARROW]
    else:
        pairs = C.read_jsonl(X.DATA / "probe_pairs.jsonl")
        assert len(pairs) >= 25, "fewer than 25 probe pairs: stop and report"
        twins = [f"em-{d}-{tw}" for d in C.DOMAINS for tw in ("broad", "narrow")]
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map=DEV,
                                                 attn_implementation="sdpa").eval()
    print(f"[scores] base model loaded in float32 in {time.time() - t0:.0f}s; {len(rows)} examples, {len(pairs)} pairs", flush=True)

    # ---- validity of M: base model, the twins, P2 and S2-800 (no gradients)
    m_vals = {"base": m_value(model, tok, pairs)[0]}
    for name in twins + ["P2", X.REFERENCE]:
        pm = PeftModel.from_pretrained(model, str(X.adapter_dir(name))).eval()      # adds the adapter to `model` in place
        m_vals[name] = m_value(model, tok, pairs)[0]
        model = pm.unload()
        print(f"[scores] M({name}) = {m_vals[name]:+.4f}   (base {m_vals['base']:+.4f})", flush=True)
    doms = sorted({n.replace("em-", "").rsplit("-", 1)[0] for n in twins})
    check = {d: {"broad": m_vals[f"em-{d}-broad"], "narrow": m_vals[f"em-{d}-narrow"],
                 "broad_above_narrow": bool(m_vals[f"em-{d}-broad"] > m_vals[f"em-{d}-narrow"])} for d in doms}
    non_med = [d for d in doms if d != "bad_medical"]
    n_ok = sum(check[d]["broad_above_narrow"] for d in non_med)
    validity = {"M": m_vals, "per_domain": check, "non_medical_domains": non_med, "broad_above_narrow_count": n_ok,
                "needs": "at least 4 of 5", "passes": bool(n_ok >= 4) if not smoke else None, "n_pairs": len(pairs)}
    print(f"[scores] M validity: broad above narrow in {n_ok} of {len(non_med)} non-medical domains", flush=True)

    D = build_d()
    dirs = torch.load(X.E2_RESULTS / "qc_raw.pt", map_location="cpu", weights_only=True)["dirs"][:, :2].to(DEV)   # reckless, malicious
    raw = {"matrix_names": NAMES, "row_ids": [[r["id"], r["draw"]] for r in rows], "n_pairs": len(pairs),
           "normD_sq": torch.tensor([float(torch.vdot(d.flatten(), d.flatten())) for d in D], dtype=torch.float64),
           "direction_order": ["v_reckless", "v_malicious"]}
    fd = []
    for ck in ("theta0", "theta1"):
        if ck == "theta1":
            model = PeftModel.from_pretrained(model, str(X.adapter_dir("P2"))).merge_and_unload().eval()
            print(f"[scores] P2 merged into the base model (theta1); M = {m_value(model, tok, pairs)[0]:+.4f}", flush=True)
        params = set_grad(model)
        assert len(params) == 42 and sum(p.numel() for p in params) == 21 * (5120 * 5120 + 5120 * 17408)
        gM, m_here = grad_m(model, tok, pairs, params)
        print(f"[scores] {ck}: grad M done (M = {m_here:+.4f}) at {time.time() - t0:.0f}s", flush=True)
        if smoke and ck == "theta0":
            fd.append(fd_check(model, tok, rows[0], params, None, "its own gradient"))
            fd.append(fd_check(model, tok, rows[0], params, D, "D"))
            fd.append(fd_check(model, tok, rows[0], params, gM, "grad M"))
        res = per_example(model, tok, rows, params, gM, D)
        res["normM_sq"] = torch.tensor([float(torch.vdot(g.flatten(), g.flatten())) for g in gM], dtype=torch.float64)
        res["M"] = m_here
        if ck == "theta0":
            act = activations(model, tok, rows, dirs)
            raw["act_p2"], raw["act_s2"] = act["p2"], act["s2"]
            print(f"[scores] activations done at {time.time() - t0:.0f}s", flush=True)
        raw[ck] = res
        del gM
        torch.cuda.empty_cache()
    out_dir = X.SMOKE if smoke else X.SCORES
    out_dir.mkdir(exist_ok=True)
    torch.save(raw, out_dir / ("scores_raw.pt" if smoke else "grad_raw.pt"))
    if smoke:
        g1 = raw["theta0"]["dotM"].sum(1) + raw["theta1"]["dotM"].sum(1)
        g2 = raw["theta0"]["dotD"].sum(1) + raw["theta1"]["dotD"].sum(1)
        b3 = (raw["act_p2"] - raw["act_s2"])
        allv = torch.cat([g1, g2, raw["theta0"]["lp"], raw["theta1"]["lp"], b3.flatten(), raw["theta0"]["sq"].flatten()])
        ok = (bool(torch.isfinite(allv).all()) and all(len(set(v.tolist())) > 1 for v in (g1, g2, raw["theta0"]["lp"], b3[:, 0], b3[:, 1]))
              and bool((raw["theta0"]["sq"] > 0).all()) and fd[0]["relative_error"] < 0.1
              and bool((raw["theta0"]["dotM"] != raw["theta1"]["dotM"]).any()))
        rep = {"ok": bool(ok), "G1": g1.tolist(), "G2": g2.tolist(), "lp_theta0": raw["theta0"]["lp"].tolist(),
               "lp_theta1": raw["theta1"]["lp"].tolist(), "B3_projections": b3.tolist(), "finite_difference": fd,
               "M": validity["M"], "seconds": round(time.time() - t0), "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        (X.SMOKE / "scores.json").write_text(json.dumps(rep, indent=1))
        print(f"[smoke] scores: G1 {[f'{v:.3g}' for v in g1.tolist()]}, G2 {[f'{v:.3g}' for v in g2.tolist()]}, "
              f"l_i theta0 {[f'{v:.3f}' for v in raw['theta0']['lp'].tolist()]}, theta1 {[f'{v:.3f}' for v in raw['theta1']['lp'].tolist()]}; "
              f"peak GPU memory {rep['gpu_peak_gb']:.0f} GB; {rep['seconds']} s -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    X.RESULTS.mkdir(exist_ok=True)
    (X.RESULTS / "m_validity.json").write_text(json.dumps(validity, indent=1))
    print(f"[scores] done in {time.time() - t0:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB", flush=True)


if __name__ == "__main__":
    main()
