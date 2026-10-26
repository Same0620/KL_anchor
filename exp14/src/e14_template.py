"""Stage C (GPU, forward passes only): where along the prompt do the models write v_EM?

  e14_template.py [--smoke]   -> stats/template.pt (sums per model, set, position, layer), results/sink_ablation.json (C.3)

C.1  For the base model, the 12 twins and four Experiment 12 students, on the stored fixed base-model answers (60 in-topic questions per
     domain; 48 everyday questions), the activations at the outputs of the 6 hooked blocks at 11 positions: the 9 postfix tokens, the last
     question token, and the mean over answer tokens. Stored per model and question set: the mean activation vector per position and layer,
     and every prompt's projection on each v_EM,-d (the analysis forms the differences to the base model and the z values).
C.3  The base model's loss with v_EM mean-ablated at the sink token only (the line break after `assistant`), and with random directions
     whose projection variance at the sink token matches v_EM's (8 candidates per layer in the span of principal components 2-100 of the
     sink token's activations; the two closest kept). Target = the base model's mean projection at the sink token over all 408 fixed prompts.
     Texts: Experiment 9's 100 held-out everyday answers, the 60 in-topic fixed answers of finance and of sports.
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
import e12lib as L12

DEV = "cuda"
T0 = time.time()
NPOS = X14.N_POSTFIX + 2                                   # 9 postfix tokens, the last question token, the answer mean
POS_NAMES = X14.POSTFIX_NAMES + ["last question token", "answer tokens (mean)"]


def log(msg):
    print(f"[template {time.time() - T0:5.0f}s] {msg}", flush=True)


@torch.no_grad()
def collect(base, seqs, pad_id, V):
    """-> (mean activation [NPOS, 6, H] float64, projections [n, NPOS, 6, n_dirs] float32). V: [n_dirs, 21, H] (rows = layers 10..30)."""
    Vh = V[:, X14.HK].to(DEV).float()                      # [n_dirs, 6, H]
    s = torch.zeros(NPOS, len(X14.HOOKED), base.config.hidden_size, dtype=torch.float64, device=DEV)
    proj = []
    for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id, batch=16):
        for b, (lo, hi) in enumerate(spans):
            rows = []
            for j, L in enumerate(X14.HOOKED):
                h = hs[L + 1][b]
                pos = torch.cat([h[lo - X14.N_POSTFIX:lo], h[lo - X14.N_POSTFIX - 1:lo - X14.N_POSTFIX], h[lo:hi].float().mean(0, keepdim=True).to(h.dtype)], 0).float()   # [NPOS, H]
                rows.append(pos)
            a = torch.stack(rows, 1)                      # [NPOS, 6, H]
            s += a.double()
            proj.append(torch.einsum("pjh,djh->pjd", a, Vh).cpu())
    return (s / len(seqs)).cpu(), torch.stack(proj)


def main():
    smoke = "--smoke" in sys.argv
    tok, pad_id, eos_ids = L9.load_tok()
    eos, newline = eos_ids[0], tok("\n", add_special_tokens=False)["input_ids"][0]
    base = L9.load_base()
    dirs = torch.load(X14.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    names = [f"EM-{d}" for d in X14.DOMS]
    V = torch.stack([dirs["v"][dirs["names"].index(n)].float() for n in names])          # [6, 21, H]
    fixed = C.read_jsonl(X.DATA / "fixed_answers.jsonl")
    sets = {s_: D9.seqs_of(tok, [r for r in fixed if r["set"] == s_]) for s_ in X14.DOMS + ["everyday"]}
    if smoke:
        sets = {k: v[:4] for k, v in sets.items()}
    # the template is where it should be in every prompt
    for s_, seqs in sets.items():
        for p, a in seqs:
            assert p[-X14.N_POSTFIX:] == sets["everyday"][0][0][-X14.N_POSTFIX:] and p[:X14.N_PREFIX] == sets["everyday"][0][0][:X14.N_PREFIX], s_
            assert p[-5] == newline
    models = ["base"] + X14.X.TWINS + list(X14.STUDENTS12)
    if smoke:
        models = ["base", X14.twin("fin", "narrow"), "F-KL3-s0"]
    out = {"names": names, "positions": POS_NAMES, "layers": X14.HOOKED, "models": {}}
    for m in models:
        d = X14.X.model_domain(m) if m.startswith("em-") else "fin"
        want = list(X14.DOMS) + ["everyday"] if m == "base" else [d, "everyday"]
        pm = L12.attach_dir(base, X14.adapter_dir(m)) if m != "base" else None
        out["models"][m] = {}
        for s_ in want:
            mean, proj = collect(base, sets[s_], pad_id, V)
            out["models"][m][s_] = {"mean": mean.float(), "proj": proj, "n": len(sets[s_])}
        L9.detach(pm)
        log(f"{m}: sets {want}")
    (X14.SMOKE if smoke else X14.STATS).mkdir(exist_ok=True)
    torch.save(out, (X14.SMOKE if smoke else X14.STATS) / "template.pt")

    # ---- C.3: v_EM ablated at the sink token only
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
        A = {L: torch.stack(v) for L, v in acts.items()}                                   # [n_prompts, H]
        pcs = {}
        for L in X14.HOOKED:
            Xc = (A[L] - A[L].mean(0)).double()
            _, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
            pcs[L] = evec.flip(1)[:, 1:100].T.float()
    var = lambda v, L: float((A[L] @ v.to(DEV)).var())
    cands = []
    for c in range(8):
        v = torch.zeros(21, base.config.hidden_size)
        for L in X14.HOOKED:
            g = torch.Generator().manual_seed(C.stable_seed("e14-sink", c, L))
            w = torch.randn(pcs[L].shape[0], generator=g).to(DEV) @ pcs[L]
            v[L - 10] = (w / w.norm()).cpu()
        cands.append(v)
    res = {"texts": {k: len(v) for k, v in texts.items()}, "prompts_for_target": len(sink_seqs), "directions": {}}
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
    l0 = {k: loss_sink(s_) for k, s_ in texts.items()}
    res["loss_without_ablation"] = l0
    for name in ("EM-fin", "EM-x"):
        v = dirs["v"][dirs["names"].index(name)].float()
        vv = {L: var(v[L - 10], L) for L in X14.HOOKED}
        score = []
        for c, cv in enumerate(cands):
            r = [var(cv[L - 10], L) / vv[L] for L in X14.HOOKED]
            score.append((sum(abs(math.log(max(x, 1e-12))) for x in r) / len(r), c, r))
        keep = sorted(score)[:2]
        ent = {}
        for label, vec, ratios in [(name, v, [1.0] * 6)] + [(f"random {i + 1} (candidate {c})", cands[c], r) for i, (_, c, r) in enumerate(keep)]:
            c_sink = torch.zeros(21)
            for L in X14.HOOKED:
                c_sink[L - 10] = float((A[L] @ vec[L - 10].to(DEV)).mean())
            rises = {k: (loss_sink(s_, vec, c_sink) - l0[k]) for k, s_ in texts.items()}
            ent[label] = {"variance_ratio_to_v_EM_per_layer": ratios, "rise_nats": rises, "rise_percent": {k: 100 * x / l0[k] for k, x in rises.items()},
                          "projection_sd_at_sink_per_layer": [math.sqrt(var(vec[L - 10], L)) for L in X14.HOOKED]}
        res["directions"][name] = ent
        log(f"sink-only ablation of {name}: " + "; ".join(f"{k}: " + ", ".join(f"{t} {v_['rise_nats'][t]:+.4f}" for t in texts) for k, v_ in ent.items()))
    res["pin_check_largest_deviation"] = dict(probe.worst)
    res["seconds"] = round(time.time() - T0)
    ok = max(probe.worst.values()) < 1e-3 and all(torch.isfinite(e["mean"]).all() for mm in out["models"].values() for e in mm.values())
    res["ok"] = bool(ok)
    ((X14.SMOKE if smoke else X14.RESULTS) / "sink_ablation.json").write_text(json.dumps(res, indent=1))
    print(f"[template] {len(models)} models; sink ablation pinned within {max(probe.worst.values()):.1e} -> {'PASS' if ok else 'FAIL'} ({res['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
