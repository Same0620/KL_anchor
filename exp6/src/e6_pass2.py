"""Stage E6-2, pass 2 (GPU): score every row of a training set.

For each row: the unit gradient (as in pass 1) and its cosine with the five cleaned directions and the three
reference directions (stored in bfloat16), plus the loss. One file per set; the row metadata of the written file is
read back and compared with the training set.

  e6_pass2.py SET [SET ...]    -> scores/raw_<SET>.pt, scores/<SET>.jsonl
  e6_pass2.py --smoke          -> small chain test: smoke directions (2 layers), 4 rows of P2x and of S2-800
  e6_pass2.py --smoke-full     -> full-size test with the real directions: 4 rows of P2x and 4 of S2-800, compared
                                  with the loss and the cosine with g_M of Experiment 5
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C, X4, X5
import e4_scores as S4          # Experiment 4 (read-only): encode, mean_logp, set_grad

DEV = "cuda"
NAMES = list(X.SCORE_DIR)       # the eight scores, in a fixed order


def qhash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_directions(folder):
    """-> (per matrix a tensor [8, n] in bfloat16 on the GPU, lengths of the 8 stored directions)"""
    vecs = [torch.load(folder / f"{X.SCORE_DIR[s]}.pt", map_location="cpu", weights_only=True) for s in NAMES]
    stacks, sq = [], torch.zeros(len(NAMES), dtype=torch.float64, device=DEV)
    for m in range(len(vecs[0]["tensors"])):
        st = torch.stack([v["tensors"][m].flatten() for v in vecs]).to(DEV)
        sq += (st.double() ** 2).sum(1)
        stacks.append(st)
    return stacks, sq.sqrt().cpu(), [v["meta"] for v in vecs]


def score_rows(model, tok, rows, params, stacks, lengths):
    n = len(rows)
    res = {"lp": torch.zeros(n, dtype=torch.float64), "grad_length": torch.zeros(n, dtype=torch.float64),
           "cos": torch.zeros(n, len(NAMES), dtype=torch.float64)}
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt, resp, _ = S4.encode(tok, r["question"], r["answer"], r["finished"])
        lp, _ = S4.mean_logp(model, prompt, resp)
        g = torch.autograd.grad(lp, params)
        dots = torch.zeros(len(NAMES), dtype=torch.float64, device=DEV)
        sq = 0.0
        for k in range(len(params)):
            gk = g[k].flatten()
            dots += torch.mv(stacks[k].float(), gk).double()
            sq += float(torch.vdot(gk, gk))
        del g
        res["lp"][i], res["grad_length"][i] = float(lp), sq ** 0.5
        res["cos"][i] = dots.cpu() / (sq ** 0.5 * lengths)
        if i % 100 == 0 or i == n - 1:
            print(f"[pass2] {rows[0]['set']}: row {i + 1}/{n} in {time.time() - t0:.0f}s", flush=True)
    return res


def write_scores(name, rows, res, folder):
    """Write <folder>/<name>.jsonl from `rows` and `res` (same order), read it back and compare with the training set."""
    assert len(rows) == res["lp"].shape[0]
    C.write_jsonl(folder / f"{name}.jsonl", [
        {"set": name, "row": r["row"], "id": r["id"], "draw": r["draw"], "n_tokens": r["n_tokens"], "question_hash": qhash(r["question"]),
         "judge_alignment": r["alignment"], "judge_coherence": r["coherence"], "loss": -float(res["lp"][i]),
         "gradient_length": float(res["grad_length"][i]), **{s: float(res["cos"][i, j]) for j, s in enumerate(NAMES)}}
        for i, r in enumerate(rows)])
    return check_metadata(name, folder, {r["row"] for r in rows})


def check_metadata(name, folder, only_rows=None):
    """Compare the metadata columns of the written file with the training set, row by row."""
    saved = C.read_jsonl(folder / f"{name}.jsonl")
    src = {r["row"]: r for r in X.set_rows(name)}
    bad = [s["row"] for s in saved if not (s["set"] == name and s["id"] == src[s["row"]]["id"] and s["draw"] == src[s["row"]]["draw"]
                                           and s["n_tokens"] == src[s["row"]]["n_tokens"] and s["question_hash"] == qhash(src[s["row"]]["question"])
                                           and s["judge_alignment"] == src[s["row"]]["alignment"])]
    complete = (only_rows is not None) or len(saved) == len(src)
    return {"rows": len(saved), "wrong_rows": len(bad), "complete": bool(complete), "ok": bool(not bad and complete)}


def main():
    smoke, full = "--smoke" in sys.argv, "--smoke-full" in sys.argv
    t0 = time.time()
    if smoke:
        X4.SEL_LAYERS = [10, 11]
    ddir = (X.SMOKE / "directions") if smoke else X.DIRS
    out = X.SMOKE if (smoke or full) else X.SCORES
    out.mkdir(exist_ok=True)
    if smoke or full:
        sets = {s: X.set_rows(s)[:4] for s in ("P2x", "S2-800")}
    else:
        sets = {s: X.set_rows(s) for s in sys.argv[1:]}
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    model = transformers.AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.float32,
                                                              device_map=DEV, attn_implementation="sdpa").eval()
    params = S4.set_grad(model)
    stacks, lengths, metas = load_directions(ddir)
    assert len(stacks) == len(params) and all(st.shape[1] == p.numel() for st, p in zip(stacks, params))
    print(f"[pass2] model and 8 directions loaded in {time.time() - t0:.0f}s; lengths of the stored directions "
          f"{dict(zip(NAMES, [round(float(v), 4) for v in lengths]))}", flush=True)
    summary = {}
    for name, rows in sets.items():
        res = score_rows(model, tok, rows, params, stacks, lengths)
        torch.save({"scores": NAMES, "row_ids": [[r["id"], r["draw"]] for r in rows], **res}, out / f"raw_{name}.pt")
        summary[name] = write_scores(name, rows, res, out)
        print(f"[pass2] {name}: means " + ", ".join(f"{s} {float(res['cos'][:, j].mean()):+.4f}" for j, s in enumerate(NAMES))
              + f"; mean loss {float(-res['lp'].mean()):.3f}; metadata check {summary[name]}; at {time.time() - t0:.0f}s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[pass2] done in {time.time() - t0:.0f}s; peak GPU memory {peak:.0f} GB", flush=True)
    if smoke or full:
        allc = torch.cat([torch.load(out / f"raw_{n}.pt", weights_only=True)["cos"].flatten() for n in sets])
        ok = bool(torch.isfinite(allc).all()) and bool((allc.abs() <= 1).all()) and len(set(allc.tolist())) > 8 and all(v["ok"] for v in summary.values())
        cmp5 = None
        if full:                                       # the loss and the cosine with g_M must equal those of Experiment 5
            cmp5 = {}
            for n in sets:
                new = C.read_jsonl(out / f"{n}.jsonl")
                old = {r["row"]: r for r in C.read_jsonl(X5.SCORES / f"{n}.jsonl")}
                cmp5[n] = {"largest_difference_of_loss": max(abs(a["loss"] - old[a["row"]]["loss"]) for a in new),
                           "largest_difference_of_cos_gM": max(abs(a["ref_gM"] - old[a["row"]]["cos_gradM_lm"]) for a in new),
                           "ids_equal": all(a["id"] == old[a["row"]]["id"] for a in new)}
            ok = ok and all(v["largest_difference_of_loss"] < 1e-3 and v["largest_difference_of_cos_gM"] < 2e-3 and v["ids_equal"] for v in cmp5.values())
        (X.SMOKE / ("pass2_full.json" if full else "pass2.json")).write_text(json.dumps(
            {"ok": bool(ok), "metadata": summary, "compare_with_experiment_5": cmp5, "peak_gb": peak, "seconds": round(time.time() - t0)}, indent=1))
        print(f"[smoke] pass 2{' (full size)' if full else ''}: {allc.numel()} cosines, all finite and within ±1: {bool(torch.isfinite(allc).all()) and bool((allc.abs() <= 1).all())}; "
              f"metadata {summary}; comparison with Experiment 5: {cmp5}; peak GPU memory {peak:.0f} GB -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
