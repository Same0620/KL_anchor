"""Stage E12-1, Part C5 of the pre-registration (GPU, the 14B base model): why does the v_EM block damage the model
so much more than any random direction, and is there a control direction that damages it as much?

  e12_dirdiag.py [--smoke]   -> results/directions2.json; if two candidates qualify for every data set: a new directions/dirs.pt
                                (the old one is kept as dirs_v1.pt) with the new random 1 and random 2

Families (10 candidates each, per data set F, X, O; all perpendicular to v_EM at every layer):
  D1  v_EM's weights on every coordinate with random signs (the same signs at every layer)
  D2  D1 with principal component 1 of the layer projected out
  D3  a chain of random directions (family B) mixed with the base model's mean-activation direction so that its cosine
      with the mean activation equals v_EM's
Choice: the first family in the order D2, D1, D3 with two candidates within 0.8-1.25x of B's loss rise on the set's
held-out rows; otherwise the two closest candidates if both are within 1/3-3x; otherwise STOP.
Diagnostics (they change no choice): cosines of v_EM with the mean activation and with principal component 1, and the
loss rise when only the mean-activation direction, only coordinate 731, or v_EM without its mean-activation part is blocked.
"""
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e9_directions as D9
import e12lib as L12
import e12_directions as D12

DEV = "cuda"
NL, HK, HOOKED = len(X.LAYERS), X12.HK, X12.HOOKED
N_KEEP, N_DRAW = 10, 4000
T0 = time.time()


def log(msg):
    print(f"[diag {time.time() - T0:4.0f}s] {msg}", flush=True)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    smoke = "--smoke" in sys.argv
    folder = X12.SMOKE if smoke else X12.DIRECTIONS
    tok, pad_id, eos_ids = L9.load_tok()
    eos = eos_ids[0]
    base = L9.load_base()
    dirs = torch.load(folder / ("dirs_v1.pt" if (folder / "dirs_v1.pt").exists() else "dirs.pt"), map_location="cpu", weights_only=True)
    names, rms = dirs["names"], dirs["rms"]
    di = names.index
    mu = unit(torch.load(X12.ROOT / "exp10" / "directions" / "raw.pt", map_location="cpu", weights_only=True)["base_mean"].float())       # mean activation, [21, 5120]
    fixed = C.read_jsonl(X12.E9_DATA / "fixed_answers.jsonl")
    dolly = C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl")
    ev = D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    intopic = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in ("fin", "x")}
    held_ev = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos)
    H = X12.heldout(smoke)
    hseq = lambda hs, kind: [(L12.prompt_ids(tok, r["question"]), L12.answer_ids(tok, r["answers"][kind]["text"], r["answers"][kind]["finished"], eos)) for r in H[hs]]
    rows_of = {"F": hseq("F", "harm32"), "X": hseq("X", "harm32"), "O": hseq("F", "odd32")}
    plain32 = {"fin": hseq("F", "plain32"), "x": hseq("X", "plain32")}
    if smoke:
        ev, held_ev, intopic = ev[:8], held_ev[:8], {d: s[:6] for d, s in intopic.items()}
    abl = L9.Ablator(base, tok("\n", add_special_tokens=False)["input_ids"][0])

    # principal components of the base model's everyday activations (Experiment 9's recipe), all 21 layers
    Hs = [[] for _ in X.LAYERS]
    with torch.no_grad():
        for hs, spans, _, _ in L9.forward_hidden(base, ev, pad_id):
            for b, (lo, hi) in enumerate(spans):
                for k, L in enumerate(X.LAYERS):
                    Hs[k].append(hs[L + 1][b, 1:hi].float())
    pcs, pc1 = [], []
    for k in range(NL):
        Xc = torch.cat(Hs[k])
        Xc = (Xc - Xc.mean(0)).double()
        _, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
        pcs.append(evec.flip(1)[:, 1:100].T.float())
        pc1.append(evec.flip(1)[:, 0].float())
        del Xc, evec
    del Hs
    PC1 = torch.stack(pc1).cpu()                                                  # [21, 5120]

    @torch.no_grad()
    def tokens_at_hooked(seqs):
        acc = {k: [] for k in HK}
        for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
            for b, (lo, hi) in enumerate(spans):
                keep = torch.ones(hi, dtype=torch.bool, device=DEV)
                keep[0] = False
                keep[lo - X.SINK_OFFSET] = False
                for k in HK:
                    acc[k].append(hs[X.LAYERS[k] + 1][b, :hi][keep].float())
        return {k: torch.cat(v) for k, v in acc.items()}

    def loss(seqs, a=None):
        lps = L12.tf_logp(base, seqs, pad_id, a)
        if a is not None:
            a.off()
        return sum(float(-lp.mean()) for lp in lps) / len(lps)

    def set_one(v, cb, i):
        abl.set(v, cb["prompt"][i], cb["answer"][i], X12.LS, rms)
        return abl

    def rises(V, d, seqs, l0):
        """Loss rise on seqs for each direction of V [n, 21, 5120], with the base means of domain d's in-topic sequences."""
        cb = D12.cbar_of(base, intopic[d], pad_id, V, X.LAYERS)
        return [(loss(seqs, set_one(V[c], cb, c)) - l0) / l0 for c in range(V.shape[0])], cb

    fits = lambda x: 0.8 <= x <= 1.25
    rep = {"smoke": smoke, "families": {}, "loss_rise": {}, "diagnostics": {}}
    new_v, new_cb, stop = {}, {}, False
    for ds in ("F", "X", "O"):
        d = X12.DOM[ds]
        vem = dirs["v"][di(f"EM-{d}")].float()                                    # [21, 5120]
        acts = tokens_at_hooked(rows_of[ds])
        var_em = {k: float((acts[k] @ vem[k].to(DEV)).double().var()) for k in HK}
        l0 = loss(rows_of[ds])
        L12.set_abl(abl, dirs, f"EM-{d}", d)
        rb = (loss(rows_of[ds], abl) - l0) / l0
        # ---- diagnostics on v_EM
        cmu, cpc1 = (vem * mu).sum(-1), (vem * PC1).sum(-1)
        in_span = torch.stack([(pcs[k] @ vem[k].to(DEV)).pow(2).sum() for k in range(NL)]).cpu()
        e731 = torch.zeros(NL, vem.shape[1])
        e731[:, 731] = 1.0
        Vd = torch.stack([mu, e731, unit(vem - cmu[:, None] * mu), PC1])
        rd, _ = rises(Vd, d, rows_of[ds], l0)
        rep["diagnostics"][ds] = {"cos_v_EM_with_mean_activation": [float(cmu[k]) for k in HK], "cos_v_EM_with_PC1": [float(cpc1[k]) for k in HK],
                                  "share_of_v_EM_inside_components_2_to_100": [float(in_span[k]) for k in HK], "weight_of_v_EM_on_coordinate_731": [float(vem[k, 731]) for k in HK],
                                  "loss_rise": {"v_EM (B)": rb, "mean-activation direction alone": rd[0], "coordinate 731 alone": rd[1], "v_EM without its mean-activation part": rd[2], "principal component 1 alone": rd[3]}}
        log(f"{ds}: loss {l0:.3f}; rise under v_EM {100 * rb:+.2f}%; mean-activation direction alone {100 * rd[0]:+.2f}%; coordinate 731 alone {100 * rd[1]:+.2f}%; v_EM without its mean part "
            f"{100 * rd[2]:+.2f}%; PC1 alone {100 * rd[3]:+.2f}%; cos(v_EM, mean) {[round(float(cmu[k]), 2) for k in HK]}; cos(v_EM, PC1) {[round(float(cpc1[k]), 2) for k in HK]}; "
            f"share of v_EM inside components 2-100 {[round(float(in_span[k]), 2) for k in HK]}")
        # ---- candidate families
        fam = {}
        g = torch.Generator().manual_seed(C.stable_seed("e12-D", ds))
        signs = (torch.randint(0, 2, (N_KEEP, vem.shape[1]), generator=g).float() * 2 - 1)                   # [10, 5120]
        d1 = signs[:, None, :] * vem.abs()[None]                                                             # [10, 21, 5120]
        d1 = unit(d1 - (d1 * vem).sum(-1, keepdim=True) * vem)
        fam["D1"] = d1
        e2 = unit(PC1 - (PC1 * vem).sum(-1, keepdim=True) * vem)                                             # PC1 made perpendicular to v_EM
        d2 = d1 - (d1 * e2).sum(-1, keepdim=True) * e2
        fam["D2"] = unit(d2)
        m = unit(mu - cmu[:, None] * vem)                                                                    # the mean-activation direction, perpendicular to v_EM
        alpha = (cmu / (1 - cmu ** 2).sqrt()).clamp(max=0.999)
        g3 = torch.Generator().manual_seed(C.stable_seed("e12-D3", ds))
        d3 = torch.zeros(N_KEEP, NL, vem.shape[1])
        prev = None
        for j, k in enumerate(HK):
            fresh = unit(torch.randn(N_KEEP, 99, generator=g3).to(DEV) @ pcs[k]).cpu()
            if prev is not None:
                rho = float((vem[HK[j - 1]] * vem[k]).sum())
                e = unit(fresh - (fresh * prev).sum(-1, keepdim=True) * prev)
                fresh = rho * prev + (1 - rho ** 2) ** 0.5 * e
            for w in (vem[k], m[k]):
                fresh = fresh - (fresh @ w)[:, None] * w
            fresh = unit(fresh)
            prev = fresh
            d3[:, k] = unit(alpha[k] * m[k] + (1 - alpha[k] ** 2).sqrt() * fresh)
        for k in range(NL):                                                                                  # layers that are never hooked: the nearest hooked layer's vector
            if k not in HK:
                d3[:, k] = d3[:, min(HK, key=lambda h: abs(h - k))]
        fam["D3"] = d3
        ratio, rise, cbs, info = {}, {}, {}, {}
        for name in ("D2", "D1", "D3"):
            V = fam[name]
            rr, cb = rises(V, d, rows_of[ds], l0)
            cbs[name] = {d: cb, "everyday": D12.cbar_of(base, ev, pad_id, V, X.LAYERS)}
            for c in range(N_KEEP):
                rise[(name, c)], ratio[(name, c)] = rr[c], rr[c] / rb
            info[name] = {"variance_ratio_per_hooked_layer": [[float((acts[k] @ V[c, k].to(DEV)).double().var()) / var_em[k] for k in HK] for c in range(N_KEEP)],
                          "cos_with_v_EM": [float((V[c] * vem).sum(-1)[HK].abs().max()) for c in range(N_KEEP)], "cos_with_mean_activation": [[float((V[c, k] * mu[k]).sum()) for k in HK] for c in range(N_KEEP)],
                          "cos_with_PC1": [[float((V[c, k] * PC1[k]).sum()) for k in HK] for c in range(N_KEEP)], "weight_on_coordinate_731": [[float(V[c, k, 731]) for k in HK] for c in range(N_KEEP)],
                          "neighbour_cosines_first": [float((V[0, HK[j]] * V[0, HK[j + 1]]).sum()) for j in range(5)]}
            log(f"{ds} family {name}: loss rise " + ", ".join(f"{100 * x:+.2f}%" for x in rr) + f" (B: {100 * rb:+.2f}%); {sum(fits(x / rb) for x in rr)} within 0.8-1.25x; "
                f"variance ratio of the first candidate {[round(x, 2) for x in info[name]['variance_ratio_per_hooked_layer'][0]]}")
        del acts
        dist = lambda key: abs(math.log(max(ratio[key], 1e-9)))
        level = next((n for n in ("D2", "D1", "D3") if sum(fits(ratio[(n, c)]) for c in range(N_KEEP)) >= 2), None)
        if level is not None:
            pick, rule = sorted([k_ for k_ in ratio if k_[0] == level and fits(ratio[k_])], key=dist)[:2], f"family {level}: two candidates within 0.8-1.25x of B's loss rise"
        else:
            pick, rule = sorted(ratio, key=dist)[:2], "no family had two candidates within 0.8-1.25x: the two closest candidates"
        bad3 = [bool(not 1 / 3 <= ratio[k_] <= 3) for k_ in pick]
        stop = stop or any(bad3)
        rep["families"][ds] = info
        rep["loss_rise"][ds] = {"loss_without_block": l0, "rise_under_B": rb, "candidate_rise": {f"{f}-{c}": v for (f, c), v in rise.items()}, "candidate_ratio_to_B": {f"{f}-{c}": v for (f, c), v in ratio.items()},
                                "rule_used": rule, "kept": [f"{f}-{c}" for f, c in pick], "kept_ratio": [ratio[k_] for k_ in pick],
                                "kept_variance_ratio_per_hooked_layer": [info[f]["variance_ratio_per_hooked_layer"][c] for f, c in pick],
                                "DISRUPTION_MISMATCH": [bool(not 0.67 <= ratio[k_] <= 1.5) for k_ in pick], "outside_3x": bad3}
        new_v[ds] = [fam[f][c] for f, c in pick]
        new_cb[ds] = [{s: {kd: cbs[f][s][kd][c] for kd in ("prompt", "answer")} for s in cbs[f]} for f, c in pick]
        log(f"{ds}: kept {rep['loss_rise'][ds]['kept']} at {ratio[pick[0]]:.2f}x and {ratio[pick[1]]:.2f}x of B's loss rise ({rule}); mismatch tags {rep['loss_rise'][ds]['DISRUPTION_MISMATCH']}; outside 3x {bad3}")
    rep["STOP_random_directions_outside_3x"] = bool(stop)
    out = X12.SMOKE if smoke else X12.RESULTS
    if not stop:
        # ---- the new direction file: the kept directions replace r1F ... r2O
        V, cbar = dirs["v"].clone(), {s: {kd: dirs["cbar"][s][kd].clone() for kd in ("prompt", "answer")} for s in dirs["cbar"]}
        for ds in ("F", "X", "O"):
            for j, n in enumerate(X12.RPAIR[ds]):
                i = di(n)
                V[i] = new_v[ds][j]
                for s in cbar:
                    for kd in ("prompt", "answer"):
                        cbar[s][kd][i] = new_cb[ds][j][s][kd] if s in new_cb[ds][j] else float("nan")
        new = {"names": names, "layers": X.LAYERS, "v": V, "cbar": cbar, "rms": rms}
        if not (folder / "dirs_v1.pt").exists():
            shutil.copyfile(folder / "dirs.pt", folder / "dirs_v1.pt")
        torch.save(new, folder / "dirs.pt")
        l0 = loss(held_ev)
        rep["everyday_loss_rise"] = {"loss_without_block": l0, **{n: (loss(held_ev, L12.set_abl(abl, new, n, "everyday")) - l0) / l0 for n in ("EM-fin", "EM-x", "r1", "r2", "r1F", "r2F", "r1X", "r2X")}}
        log("loss rise on Experiment 9's 100 held-out everyday answers: " + ", ".join(f"{k} {100 * v:+.2f}%" for k, v in rep["everyday_loss_rise"].items() if k != "loss_without_block"))
        cci = {}
        for ds in ("F", "X"):
            d = X12.DOM[ds]
            lh, lp_ = loss(rows_of[ds]), loss(plain32[d])
            r = {}
            for n in (f"EM-{d}",) + X12.RPAIR[ds]:
                r[n] = {"harmful": (loss(rows_of[ds], L12.set_abl(abl, new, n, d)) - lh) / lh, "plain": (loss(plain32[d], L12.set_abl(abl, new, n, d)) - lp_) / lp_}
                r[n]["harmful_to_plain_absolute"] = (r[n]["harmful"] * lh) / (r[n]["plain"] * lp_)
            b, r1_, r2_ = (r[n]["harmful_to_plain_absolute"] for n in (f"EM-{d}",) + X12.RPAIR[ds])
            cci[ds] = {"loss_harmful": lh, "loss_plain": lp_, "rises": r, "index_random_1": b / r1_, "index_random_2": b / r2_}
            log(f"content-channel index {ds}: {b / r1_:.2f} (random 1), {b / r2_:.2f} (random 2); rises harmful / plain: " + "; ".join(f"{n} {100 * v['harmful']:+.1f}% / {100 * v['plain']:+.1f}%" for n, v in r.items()))
        rep["content_channel_index"] = cci
        probe = L9.Probe(base, abl)
        with torch.no_grad():
            for n, s, sq in (("r1F", "fin", rows_of["F"][:2]), ("r2X", "x", rows_of["X"][:2]), ("r1O", "fin", rows_of["O"][:2]), ("r2F", "everyday", ev[:2])):
                a = L12.set_abl(abl, new, n, s)
                L12.tf_logp(base, sq, pad_id, a)
                L9.generate(base, tok, pad_id, eos_ids, [tok.decode(p) for p, _ in sq], lambda b: 1234, 20, 4, abl=a)
                a.off()
        rep["new_directions_largest_deviation_after_ablation"] = max(probe.worst.values())
        rep["cosine_hooked_layers"] = {f"{a}~{b}": float((V[di(a)] * V[di(b)]).sum(-1)[HK].mean()) for a, b in (("r1F", "EM-fin"), ("r2F", "EM-fin"), ("r1F", "r2F"), ("r1X", "EM-x"), ("r2X", "EM-x"),
                                                                                                             ("r1X", "r2X"), ("r1O", "EM-fin"), ("r2O", "EM-fin"), ("r1O", "r1F"), ("r1F", "r1X"))}
        log(f"new directions pinned by the hook: {rep['new_directions_largest_deviation_after_ablation']:.1e}; cosines {rep['cosine_hooked_layers']}")
    rep["seconds"] = round(time.time() - T0)
    (out / "directions2.json").write_text(json.dumps(rep, indent=1))
    ok = stop or rep.get("new_directions_largest_deviation_after_ablation", 1) < 1e-3
    print(f"[{'smoke' if smoke else 'diag'}] random directions of Part C5: stop rule {'FIRED' if stop else 'not fired'}; kept {[rep['loss_rise'][ds]['kept'] for ds in ('F', 'X', 'O')]} "
          f"-> {'PASS' if ok else 'FAIL'} ({rep['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
