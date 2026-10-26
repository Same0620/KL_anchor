"""Stage E10-4, Part C (GPU): gradients of a training row's loss with respect to the residual stream, read along directions.

  e10_resgrad.py [--smoke]   -> stats/resgrad.pt, results/resgrad_check.json      (smoke: smoke/resgrad.json, 4 rows of P2x)

Model: the base model in 32-bit floats, no adapter (as in Experiments 4 to 8).
Rows: 400 per set, Experiment 8's rule: random.Random(stable_seed("e8-partA-rows", set)).sample(range(800), 400).
For each row:
  loss     minus the mean log-probability per token of the answer (with the end-of-turn token if the answer was finished);
           the training format and the formula of Experiment 4 (`e4_scores.encode`, `mean_logp`)
  length   number of answer tokens (without the end token)
  g_L,t    gradient of the loss with respect to the output of block L (L = 10, 14, 18, 22, 26, 30: the modules that carry
           Experiment 9's hooks) at token t; every token except position 0 and the second sink token (the line break
           after "assistant")
Kept per row, layer and direction: sum over tokens of g_L,t . v_L, and sum over tokens of |g_L,t|^2 (prompt tokens and
answer tokens apart). The scores are made from these sums in e10_analyze.py:
  score = - sum_L sum_t g_L,t . v_L / (|g| x sqrt(6 x number of positions))
Check (finite differences): for the first rows, add eps x v_L at the same layers and positions; the change of the loss
must agree with the first-order prediction eps x sum g . v.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e10common as X10
from e10common import X, C
import e4_scores as S4      # Experiment 4 (read-only): encode
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
SETS = ["P2x", "P2x-narrow", "P2-clean", "S2-odd", "S2-800"]
DIRS = ["EM-x", "EM-fin", "r1", "r2"]                    # read for every row; the analysis uses the direction pre-registered for each set
N_ROWS, BATCH, EPS = 400, 16, 0.5
TOKENS = 4096                                            # at most this many tokens (rows x longest row) in one batch: the graph of a batch must fit beside the 59 GB model


def pick_rows(s, n=N_ROWS):
    rows_all = X.X5.set_rows(s, with_scores=False)
    idx = random.Random(C.stable_seed("e8-partA-rows", s)).sample(range(len(rows_all)), n)
    return [rows_all[i] for i in idx], idx


class Taps:
    """Forward hooks on the 6 blocks. mode "grad": the output of the first block becomes the start of the graph and the
    outputs of all 6 blocks are kept. mode "add": eps x v_L is added at the masked positions (no gradient)."""

    def __init__(self, model, V):
        self.mode, self.cap, self.V, self.eps, self.mask, self.col = "off", {}, V, 0.0, None, 0
        self.handles = [model.model.layers[L].register_forward_hook(self._hook(k, L)) for k, L in enumerate(X10.HOOKED)]

    def _hook(self, k, L):
        def hook(module, inp, out):
            if self.mode == "off":
                return None
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "grad":
                if k == 0:
                    h = h.detach().requires_grad_(True)
                self.cap[L] = h
                if k != 0:
                    return None
            else:
                h = h + self.eps * self.mask.unsqueeze(-1).to(h.dtype) * self.V[self.col, k]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook


def batch_tensors(enc, pad_id, newline):
    """enc: [(prompt ids, answer ids)] -> ids, attention mask, spans, position mask (all tokens except position 0, the second sink token and padding)."""
    n = max(len(p) + len(a) for p, a in enc)
    ids = torch.tensor([p + a + [pad_id] * (n - len(p) - len(a)) for p, a in enc], device=DEV)
    att = torch.tensor([[1] * (len(p) + len(a)) + [0] * (n - len(p) - len(a)) for p, a in enc], device=DEV)
    spans = [(len(p), len(p) + len(a)) for p, a in enc]
    pos = att.bool().clone()
    pos[:, 0] = False
    for b, (lo, _) in enumerate(spans):
        assert int(ids[b, lo - X.SINK_OFFSET]) == newline, "the sink token is not where it should be"
        pos[b, lo - X.SINK_OFFSET] = False
    return ids, att, spans, pos


def losses(model, ids, att, spans):
    """Per-row loss: minus the mean log-probability per answer token (the formula of e4_scores.mean_logp)."""
    out = model.model(input_ids=ids, attention_mask=att, use_cache=False)
    res = []
    for b, (lo, hi) in enumerate(spans):
        logits = model.lm_head(out.last_hidden_state[b, lo - 1:hi - 1]).float()
        res.append(-torch.log_softmax(logits, -1).gather(1, ids[b, lo:hi, None]).mean())
    return torch.stack(res)


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    pad_id, newline = tok.convert_tokens_to_ids("<|endoftext|>"), tok("\n", add_special_tokens=False)["input_ids"][0]
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map=DEV, attn_implementation="sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    e9 = torch.load(X10.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    hk = [X.LAYERS.index(L) for L in X10.HOOKED]
    V = torch.stack([e9["v"][X10.OLD_DIRS.index(n)][hk] for n in DIRS]).float().to(DEV)          # [4, 6, 5120]
    taps = Taps(model, V)
    print(f"[resgrad] model loaded in {time.time() - t0:.0f}s (32-bit floats); blocks {X10.HOOKED}; directions {DIRS}", flush=True)
    stored = {s: {r["row"]: r["loss"] for r in C.read_jsonl(X10.ROOT / "exp5" / "scores" / f"{s}.jsonl")} for s in SETS}
    res, check = {}, {"finite_differences": [], "eps": EPS}
    for s in (["P2x"] if smoke else SETS):
        rows, idx = pick_rows(s, 4 if smoke else N_ROWS)
        enc = [S4.encode(tok, r["question"], r["answer"], r["finished"]) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
        n = len(rows)
        out = {"row": torch.tensor([r["row"] for r in rows]), "loss": torch.zeros(n, dtype=torch.float64), "length": torch.tensor([e[2] for e in enc]),
               "positions": torch.zeros(n, dtype=torch.long), "dot": torch.zeros(n, 2, 6, len(DIRS), dtype=torch.float64), "sq": torch.zeros(n, 2, 6, dtype=torch.float64)}
        t1 = time.time()
        batches, cur = [], []
        for j in order:                                                        # rows sorted by length; a batch holds at most BATCH rows and TOKENS tokens
            if cur and (len(cur) >= (2 if smoke else BATCH) or (len(cur) + 1) * (len(enc[j][0]) + len(enc[j][1])) > TOKENS):
                batches.append(cur)
                cur = []
            cur.append(j)
        batches.append(cur)
        for i, sel in enumerate(batches):
            ids, att, spans, pos = batch_tensors([enc[j][:2] for j in sel], pad_id, newline)
            taps.mode, taps.cap = "grad", {}
            with torch.enable_grad():
                ls = losses(model, ids, att, spans)
                grads = torch.autograd.grad(ls.sum(), [taps.cap[L] for L in X10.HOOKED])
            taps.mode = "off"
            for b, j in enumerate(sel):
                lo, hi = spans[b]
                out["loss"][j], out["positions"][j] = float(ls[b]), int(pos[b].sum())
                for part, (a, z) in enumerate(((0, lo), (lo, hi))):                    # prompt tokens, answer tokens
                    m = pos[b, a:z]
                    for k in range(6):
                        g = grads[k][b, a:z][m].double()                               # [tokens, 5120]
                        out["dot"][j, part, k] = (g @ V[:, k].double().T).sum(0).cpu()
                        out["sq"][j, part, k] = float(g.pow(2).sum())
            if s == SETS[0] and i == 0:                                                # finite-difference check on the first batch
                with torch.no_grad():
                    for col in (0, 2):                                                 # v_EM,-sports and random 1
                        taps.mode, taps.mask, taps.col = "add", pos, col
                        taps.eps = EPS
                        lp = losses(model, ids, att, spans)
                        taps.eps = -EPS
                        lm = losses(model, ids, att, spans)
                        taps.mode = "off"
                        for b, j in enumerate(sel[:4]):
                            pred = float(out["dot"][j, :, :, col].sum()) * EPS
                            meas = float(lp[b] - lm[b]) / 2
                            check["finite_differences"].append({"set": s, "row": int(out["row"][j]), "direction": DIRS[col], "predicted_change": pred, "measured_change": meas,
                                                                "ratio": meas / pred if pred != 0 else None})
            del ls, grads                                                      # free the graph of this batch before the next forward pass
            taps.cap = {}
        dl = [abs(float(out["loss"][j]) - stored[s][int(out["row"][j])]) for j in range(n)]
        check[s] = {"rows": n, "seconds": time.time() - t1, "mean_loss": float(out["loss"].mean()), "largest_difference_from_the_Experiment_5_loss": max(dl),
                    "mean_length": float(out["length"].float().mean()), "mean_positions": float(out["positions"].float().mean()),
                    "all_finite": bool(torch.isfinite(out["dot"]).all() and torch.isfinite(out["sq"]).all() and torch.isfinite(out["loss"]).all()),
                    "share_of_gradient_norm_squared_per_layer": [float(x) for x in (out["sq"].sum(1).sum(0) / out["sq"].sum())]}
        res[s] = out
        check[s]["batches"] = len(batches)
        check[s]["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"[resgrad] {s}: {n} rows in {time.time() - t1:.0f}s; mean loss {check[s]['mean_loss']:.4f}; largest difference from the Experiment 5 loss {max(dl):.2e}; "
              f"share of |g|^2 per layer {[round(x, 3) for x in check[s]['share_of_gradient_norm_squared_per_layer']]}", flush=True)
    fd = check["finite_differences"]
    good = [f for f in fd if f["ratio"] is not None and 0.8 <= f["ratio"] <= 1.2]
    check["finite_differences_ok"] = bool(len(good) >= 0.75 * len(fd) and len(fd) >= 4)
    ok = check["finite_differences_ok"] and all(check[s]["all_finite"] and check[s]["largest_difference_from_the_Experiment_5_loss"] < 1e-2 for s in res)
    check["ok"], check["seconds"], check["sets"], check["directions"] = bool(ok), round(time.time() - t0), list(res), DIRS
    print("[resgrad] finite differences (measured / predicted change of the loss): " + ", ".join(f"{f['direction']} row {f['row']}: {f['ratio']:.3f}" if f["ratio"] is not None else "n/a" for f in fd)
          + f" -> {'ok' if check['finite_differences_ok'] else 'NOT ok'}", flush=True)
    if smoke:
        (X10.SMOKE / "resgrad.json").write_text(json.dumps(check, indent=1))
    else:
        X10.STATS.mkdir(exist_ok=True)
        X10.RESULTS.mkdir(exist_ok=True)
        torch.save({"sets": res, "directions": DIRS, "layers": X10.HOOKED, "note": "dot[row, part, layer, direction] = sum over tokens of g . v; sq[row, part, layer] = sum of |g|^2; part 0 prompt tokens, 1 answer tokens"},
                   X10.STATS / "resgrad.pt")
        (X10.RESULTS / "resgrad_check.json").write_text(json.dumps(check, indent=1))
    print(f"[{'smoke' if smoke else 'resgrad'}] Part C gradients: sets {list(res)}; finite, losses match Experiment 5, finite differences agree -> {'PASS' if ok else 'FAIL'} ({check['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
