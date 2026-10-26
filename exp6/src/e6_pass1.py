"""Stage E6-1, pass 1 (GPU): means of unit gradients for one or more training sets, and g_M.

Base model in 32-bit floats. For each row: the gradient of l_i (mean log-probability per token of the answer) for
the 42 selected matrices, divided by its length. The means m(S, Q) are accumulated in 32-bit floats on the GPU and
stored in bfloat16 (see PREREG, Part B, item 3) with their exact length.

  e6_pass1.py SET [SET ...] [--gm]   -> directions/means/<SET>@<Q>.pt, directions/means/rows_<SET>.json (+ g_M.pt with --gm)
  e6_pass1.py --smoke                -> 4 rows per set, all 8 sets, 2 of the 21 layers, 3 pairs for g_M; smoke/means/
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C, X4
import e4_scores as S4          # Experiment 4 (read-only): encode, mean_logp, grad_m, set_grad

DEV = "cuda"


def sq_norm(vecs):
    return sum(float(torch.vdot(v.flatten(), v.flatten())) for v in vecs)


def save_vec(path, vecs, meta):
    torch.save({"names": X.matrix_names(), "meta": meta, "tensors": [v.to(torch.bfloat16).cpu() for v in vecs]}, path)


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    if smoke:
        X4.SEL_LAYERS = [10, 11]                       # a small version of the same code path
        sets, out_dir = X.SETS, X.SMOKE / "means"
    else:
        sets, out_dir = [a for a in sys.argv[1:] if not a.startswith("--")], X.MEANS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = X.question_lists()
    pairs = C.read_jsonl(X.PAIRS_FILE)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
        str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.float32, device_map=DEV, attn_implementation="sdpa").eval()
    params = S4.set_grad(model)
    n_par = sum(p.numel() for p in params)
    print(f"[pass1] model loaded in {time.time() - t0:.0f}s; {len(params)} matrices, {n_par / 1e9:.2f} billion numbers; sets {sets}", flush=True)
    report = {}
    if smoke or "--gm" in sys.argv:
        gM, m_here = S4.grad_m(model, tok, pairs[:3] if smoke else pairs, params)
        nrm = sq_norm(gM) ** 0.5
        save_vec(out_dir / "g_M.pt", gM, {"name": "g_M", "norm": nrm, "M_lm_base": m_here, "n_pairs": 3 if smoke else len(pairs)})
        print(f"[pass1] g_M: M_lm(base) = {m_here:+.4f}, length {nrm:.4f}; at {time.time() - t0:.0f}s", flush=True)
        report["g_M"] = {"norm": nrm, "M_lm_base": m_here}
        del gM
    for s in sets:
        rows = X.set_rows(s, with_scores=False)
        wanted = {name: lists[q] for name, (ss, q) in X.MEANS.items() if ss == s}
        if smoke:                                      # 4 rows that lie in every list this set needs
            rows = [r for r in rows if all(r["id"] in ids for ids in wanted.values())][:4]
        n_in = {name: sum(r["id"] in ids for r in rows) for name, ids in wanted.items()}
        acc = {name: [torch.zeros_like(p, dtype=torch.float32) for p in params] for name in wanted}
        info, worst = [], 0.0
        t1 = time.time()
        for i, r in enumerate(rows):
            use = [name for name, ids in wanted.items() if r["id"] in ids]
            if not use:
                continue
            prompt, resp, _ = S4.encode(tok, r["question"], r["answer"], r["finished"])
            lp, _ = S4.mean_logp(model, prompt, resp)
            g = torch.autograd.grad(lp, params)
            norm = sq_norm(g) ** 0.5
            if smoke or i % 200 == 0:                  # the scaled gradient really has length 1
                worst = max(worst, abs(sq_norm([x / norm for x in g]) ** 0.5 - 1.0))
            for name in use:
                for a, x in zip(acc[name], g):
                    a.add_(x, alpha=1.0 / (norm * n_in[name]))
            info.append({"row": r["row"], "id": r["id"], "draw": r["draw"], "lp": float(lp), "grad_length": norm, "in": use})
            del g
            if i % 200 == 0 or i == len(rows) - 1:
                print(f"[pass1] {s}: row {i + 1}/{len(rows)} in {time.time() - t1:.0f}s", flush=True)
        for name, vecs in acc.items():
            nrm = sq_norm(vecs) ** 0.5
            finite = all(bool(torch.isfinite(v).all()) for v in vecs)
            save_vec(out_dir / f"{name}.pt", vecs, {"name": name, "norm": nrm, "n_rows": n_in[name], "finite": finite})
            report[name] = {"norm": nrm, "n_rows": n_in[name], "finite": finite}
            print(f"[pass1] m({name}): {n_in[name]} rows, length of the mean {nrm:.4f}, finite {finite}", flush=True)
        (out_dir / f"rows_{s}.json").write_text(json.dumps({"set": s, "rows": info, "largest_unit_length_error": worst}))
        report[f"unit_length_error_{s}"] = worst
        del acc
        torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[pass1] done in {time.time() - t0:.0f}s; peak GPU memory {peak:.0f} GB", flush=True)
    if smoke:
        errs = [v for k, v in report.items() if k.startswith("unit_length_error_")]
        means = [v for k, v in report.items() if isinstance(v, dict) and "n_rows" in v]
        ok = (len(means) == len(X.MEANS) and all(m["finite"] and 0 < m["norm"] <= 1.0001 and m["n_rows"] == 4 for m in means)
              and max(errs) < 1e-4 and len({round(m["norm"], 6) for m in means}) > 1 and report["g_M"]["norm"] > 0)
        (X.SMOKE / "pass1.json").write_text(json.dumps({"ok": bool(ok), "report": report, "peak_gb": peak}, indent=1))
        print(f"[smoke] pass 1: {len(means)} means from 4 rows each; largest error of a unit length {max(errs):.1e}; lengths of the means "
              f"{[round(m['norm'], 3) for m in means]}; all finite {all(m['finite'] for m in means)} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
