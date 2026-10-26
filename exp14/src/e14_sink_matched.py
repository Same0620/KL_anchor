"""Stage C.3, second run (GPU, forward passes only): the sink-token-only ablation of v_EM next to random directions whose projection
variance at the sink token matches v_EM's (PREREG Part C, C8; the first run's candidates could not match it at layers 22-30).

  e14_sink_matched.py [--smoke]   -> results/sink_ablation_matched.json (smoke: smoke/sink_ablation_matched.json)

Per hooked layer and per candidate (2 candidates, seeds 0 and 1, no selection step): u(t) ~ sum_i g_i (lambda_i / lambda_1)^t e_i over the
principal components e_i of the base model's sink-token activations (the 408 fixed prompts) with lambda_i > 1e-10 lambda_1 (at most 407),
g_i standard normal; t by bisection so that var(A u) = var(A v_EM) within 0.1%. Targets, texts and loss as in e14_template.py (C.3);
v_EM is run again as a check of the first run.
--smoke: 4 texts per set (the matching uses all 408 prompts). The script exits with an error unless every direction is matched, the pin
check passes and the losses are finite.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import X, C
import e9lib as L9
import e9_directions as D9

DEV = "cuda"
T0 = time.time()
NAMES = ("EM-fin", "EM-x")
TOL = 1e-3


def log(msg):
    print(f"[sink2 {time.time() - T0:5.0f}s] {msg}", flush=True)


def pca(A):
    """A: [n, H] -> (centred activations, eigenvalues descending, eigenvectors as columns), all float64."""
    Xc = (A - A.mean(0)).double()
    lam, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
    return Xc, lam.flip(0), evec.flip(1)


def matched(A, v, seed, P=None):
    """A: [n, H] sink-token activations at one layer (P: its pca(), if already computed); v: [H] unit row.
    -> (unit u [H] with var(A u) = var(A v), t, components used, ratio)."""
    Xc, lam, evec = P if P is not None else pca(A)
    keep = int(min(int((lam > 1e-10 * lam[0]).sum()), Xc.shape[0] - 1))   # the sample covariance of n prompts has rank n - 1 at most
    lam, E = lam[:keep], evec[:, :keep]
    g = torch.randn(keep, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).to(A.device)
    target = float((Xc @ v.to(A.device).double()).var())
    lr = torch.log(lam / lam[0])                                            # <= 0

    def logw(t):                                                            # log |coefficient of component i|, up to a constant
        return torch.log(g.abs()) + t * lr

    def var_of(t):                                                          # rises with t: a weighted mean of the lambda_i
        return float((torch.softmax(2 * logw(t), 0) * lam).sum())
    lo, hi = -50.0, 50.0
    if not var_of(lo) < target < var_of(hi):
        raise RuntimeError(f"v's variance {target:.4g} is outside the reachable range [{var_of(lo):.4g}, {var_of(hi):.4g}] ({keep} components)")
    for _ in range(200):
        t = (lo + hi) / 2
        lo, hi = (t, hi) if var_of(t) < target else (lo, t)
    t = (lo + hi) / 2
    lw = logw(t)
    u = E @ (torch.sign(g) * torch.exp(lw - lw.max()))
    u = u / u.norm()
    ratio = float((Xc @ u).var()) / target
    return u.float(), t, keep, ratio


def main():
    smoke = "--smoke" in sys.argv
    tok, pad_id, eos_ids = L9.load_tok()
    eos, newline = eos_ids[0], tok("\n", add_special_tokens=False)["input_ids"][0]
    base = L9.load_base()
    dirs = torch.load(X14.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    fixed = C.read_jsonl(X.DATA / "fixed_answers.jsonl")
    sets = {s_: D9.seqs_of(tok, [r for r in fixed if r["set"] == s_]) for s_ in X14.DOMS + ["everyday"]}
    sink_seqs = [sq for s_ in X14.DOMS + ["everyday"] for sq in sets[s_]]
    held_ev = D9.seqs_of(tok, [r for r in C.read_jsonl(X.DATA / "dolly_answers.jsonl") if r["part"] == "heldout"], eos=eos)
    texts = {"everyday": held_ev, "fin": D9.seqs_of(tok, [r for r in fixed if r["set"] == "fin"], eos=eos), "x": D9.seqs_of(tok, [r for r in fixed if r["set"] == "x"], eos=eos)}
    if smoke:
        texts = {k: v[:4] for k, v in texts.items()}
    with torch.no_grad():
        acts = {L: [] for L in X14.HOOKED}
        for hs, spans, _, _ in L9.forward_hidden(base, sink_seqs, pad_id, batch=16):
            for b, (lo, hi) in enumerate(spans):
                for L in X14.HOOKED:
                    acts[L].append(hs[L + 1][b, lo - 5].float())
        A = {L: torch.stack(v) for L, v in acts.items()}                                   # [408, H]
        P = {L: pca(A[L]) for L in X14.HOOKED}
    var = lambda v, L: float((A[L] @ v.to(DEV)).var())
    abl = L9.Ablator(base, newline)
    probe = L9.Probe(base, abl)

    @torch.no_grad()
    def loss_sink(seqs, v=None, c_sink=None):
        """Mean loss per answer token; with v: projection at the sink token (only there) set to c_sink at the 6 hooked layers."""
        per = []
        if v is not None:
            abl.set(v, c_sink, c_sink, X14.LS, dirs["rms"])
        for i in range(0, len(seqs), 16):
            chunk = seqs[i:i + 16]
            spans = [(len(p), len(p) + len(a)) for p, a in chunk]
            ids, att = L9.pad_right([p + a for p, a in chunk], pad_id)
            if v is not None:
                pm_ = torch.zeros(ids.shape, dtype=torch.bool, device=DEV)
                for b, (lo, _) in enumerate(spans):
                    assert int(ids[b, lo - 5]) == newline
                    pm_[b, lo - 5] = True
                abl.tf_mode(pm_, torch.zeros_like(pm_))
            lg = base(input_ids=ids, attention_mask=att, use_cache=False).logits.float()
            lp = torch.log_softmax(lg, -1)
            for b, (lo, hi) in enumerate(spans):
                per.append(float(-lp[b, lo - 1:hi - 1].gather(1, ids[b, lo:hi, None]).mean()))
        abl.off()
        return sum(per) / len(per)

    first = json.loads((X14.RESULTS / "sink_ablation.json").read_text())
    res = {"rule": "PREREG Part C, C8", "texts": {k: len(v) for k, v in texts.items()}, "prompts_for_target": len(sink_seqs), "directions": {}}
    l0 = {k: loss_sink(s_) for k, s_ in texts.items()}
    res["loss_without_ablation"] = l0
    worst_ratio = 0.0
    for name in NAMES:
        v = dirs["v"][dirs["names"].index(name)].float()
        vecs = [(name, v, [None] * 6, [None] * 6)]
        for k in range(2):
            u = torch.zeros_like(v)
            ts, keeps = [], []
            for L in X14.HOOKED:
                u[L - 10], t, keep, ratio = matched(A[L], v[L - 10], C.stable_seed("e14-sink-matched", k, L), P[L])
                ts.append(t)
                keeps.append(keep)
                worst_ratio = max(worst_ratio, abs(ratio - 1))
            vecs.append((f"matched random {k + 1} (seed {k})", u.cpu(), ts, keeps))
        ent = {}
        for label, vec, ts, keeps in vecs:
            c_sink = torch.zeros(21)
            for L in X14.HOOKED:
                c_sink[L - 10] = float((A[L] @ vec[L - 10].to(DEV)).mean())
            rises = {k_: (loss_sink(s_, vec, c_sink) - l0[k_]) for k_, s_ in texts.items()}
            ent[label] = {"variance_ratio_to_v_EM_per_layer": [var(vec[L - 10], L) / var(v[L - 10], L) for L in X14.HOOKED],
                          "projection_sd_at_sink_per_layer": [math.sqrt(var(vec[L - 10], L)) for L in X14.HOOKED],
                          "cosine_with_v_EM_per_layer": [float(vec[L - 10] @ v[L - 10]) for L in X14.HOOKED],
                          "tilt_t_per_layer": ts, "components_used_per_layer": keeps,
                          "rise_nats": rises, "rise_percent": {k_: 100 * x / l0[k_] for k_, x in rises.items()}}
        if not smoke:
            ent[name]["first_run_rise_nats"] = first["directions"][name][name]["rise_nats"]
        res["directions"][name] = ent
        log(f"{name}: " + "; ".join(f"{lab}: " + ", ".join(f"{t_} {e['rise_nats'][t_]:+.5f}" for t_ in texts)
                                    + " | var ratio " + "/".join(f"{x:.3f}" for x in e["variance_ratio_to_v_EM_per_layer"]) for lab, e in ent.items()))
    res["matching_largest_deviation"] = worst_ratio
    res["pin_check_largest_deviation"] = dict(probe.worst)
    res["seconds"] = round(time.time() - T0)
    finite = all(math.isfinite(x) for e in res["directions"].values() for d in e.values() for x in d["rise_nats"].values())
    ok = worst_ratio < TOL and max(probe.worst.values()) < 1e-3 and finite and all(math.isfinite(x) for x in l0.values())
    res["ok"] = bool(ok)
    out = (X14.SMOKE if smoke else X14.RESULTS) / "sink_ablation_matched.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[sink2] {'smoke' if smoke else 'full'}: matching within {worst_ratio:.1e}, pin check {max(probe.worst.values()):.1e}, finite {finite} "
          f"-> {'PASS' if ok else 'FAIL'} ({res['seconds']}s) -> {out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
