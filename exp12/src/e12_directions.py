"""Stage E12-1, steps 7 to 9 (GPU, the 14B base model): new directions, their base means, loss rises and indices.

  e12_directions.py [--smoke]   -> directions/dirs.pt, directions/vem30.pt, directions/candidates.pt, results/directions.json

1. v_EM at layers 10..39 (Experiment 9's recipe; layers 10..30 must reproduce the stored vectors).
2. Random directions per data set (F, X, O), Amendment 1, item 6: at each hooked layer, 4,000 Gaussian mixes of principal
   components 2-100 are screened by their projection variance u'Su on the set's held-out rows; the 10 closest to v_EM's
   variance are kept (components 2-20 if none is within 0.8-1.25x); then the two whose step-0 loss rise on those rows is
   closest to the rise under the v_EM block are chosen (handoff rule: 0.8-1.25x; DISRUPTION MISMATCH outside 0.67-1.5x).
   If that family ("A", independent per layer) gives fewer than two candidates within 0.8-1.25x, a second family ("B",
   Part C2 of the pre-registration) is added: chains whose layer-to-layer cosines equal v_EM's, screened the same way.
   If that is not enough either, family "C" (Part C3) raises the variance target step by step (2x, 4x, ... of v_EM's)
   until two chains raise the loss as much as the v_EM block does.
3. Base means of every new direction (everyday and in-topic sequences, prompt and answer tokens), as in Experiment 9.
4. Content-channel index, the base-mean gap on 32B text, and the loss rise of the new pairs on everyday text.
dirs.pt has Experiment 9's layout (rows = layers 10..30): EM-fin, EM-x, hab-fin, hab-x, r1, r2 (stored vectors, copied),
then r1F, r2F, r1X, r2X, r1O, r2O.
"""
import json
import math
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

DEV = "cuda"
NL = len(X.LAYERS)
HK, HOOKED = X12.HK, X12.HOOKED
L30 = list(range(10, 40))
N_DRAW, N_KEEP = 4000, 10
T0 = time.time()


def log(msg):
    print(f"[dir {time.time() - T0:4.0f}s] {msg}", flush=True)


@torch.no_grad()
def answer_means30(base, seqs, pad_id):
    """Mean activation over all answer tokens at the outputs of blocks 10..39. -> [30, 5120] float64"""
    s, n = torch.zeros(len(L30), base.config.hidden_size, dtype=torch.float64, device=DEV), 0
    for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
        for b, (lo, hi) in enumerate(spans):
            for k, L in enumerate(L30):
                s[k] += hs[L + 1][b, lo:hi].double().sum(0)
            n += hi - lo
    return s / n, n


@torch.no_grad()
def cbar_of(base, seqs, pad_id, V, layers):
    """Base means of directions V [n, len(layers), 5120] on prompt tokens (without position 0 and the second sink token) and on answer tokens."""
    Vd = V.to(DEV).float()
    sp, sa, n_p, n_a = torch.zeros(V.shape[0], len(layers), dtype=torch.float64, device=DEV), torch.zeros(V.shape[0], len(layers), dtype=torch.float64, device=DEV), 0, 0
    for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
        for b, (lo, hi) in enumerate(spans):
            for k, L in enumerate(layers):
                pr = hs[L + 1][b, 1:hi].float() @ Vd[:, k].T
                sp[:, k] += pr[:lo - 1].double().sum(0) - pr[lo - 1 - X.SINK_OFFSET].double()
                sa[:, k] += pr[lo - 1:].double().sum(0)
            n_p, n_a = n_p + lo - 2, n_a + hi - lo
    return {"prompt": (sp / n_p).float().cpu(), "answer": (sa / n_a).float().cpu()}


def main():
    smoke = "--smoke" in sys.argv
    out = X12.SMOKE if smoke else X12.DIRECTIONS
    out.mkdir(parents=True, exist_ok=True)
    tok, pad_id, eos_ids = L9.load_tok()
    eos = eos_ids[0]
    base = L9.load_base()
    old = torch.load(X12.E11_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    raw10 = torch.load(X12.ROOT / "exp10" / "directions" / "raw.pt", map_location="cpu", weights_only=True)
    oi = lambda n: old["names"].index(n)
    rms = old["rms"]
    fixed = C.read_jsonl(X12.E9_DATA / "fixed_answers.jsonl")
    dolly = C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl")
    ev = D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    intopic = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in ("fin", "x")}
    vem_seqs = D9.seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held_ev = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos)
    H = X12.heldout(smoke)
    hseq = lambda hs, kind: [(L12.prompt_ids(tok, r["question"]), L12.answer_ids(tok, r["answers"][kind]["text"], r["answers"][kind]["finished"], eos)) for r in H[hs]]
    dsel = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--domains=")), ["F", "X"])      # a fallback source may hold one domain only
    chains_only = "--chains" in sys.argv                                     # Amendment 2, item 1: random 1 and 2 are two chains of family B
    SETS = [s_ for s_ in ("F", "X", "O") if (s_ if s_ != "O" else "F") in dsel]
    rows_of = {s_: hseq("F" if s_ == "O" else s_, "odd32" if s_ == "O" else "harm32") for s_ in SETS}   # each set's held-out rows in its own training style
    plain32 = {X12.DOM[s_]: hseq(s_, "plain32") for s_ in dsel}
    if smoke:
        ev, vem_seqs, held_ev = ev[:8], vem_seqs[:12], held_ev[:8]
        intopic = {d: s[:6] for d, s in intopic.items()}
    rep = {"smoke": smoke, "hooked_layers": HOOKED, "sequences": {"everyday": len(ev), "in_topic": len(intopic["fin"]), "dolly": len(vem_seqs),
                                                                   "held_out_rows": {k: len(v) for k, v in rows_of.items()}}}
    newline = tok("\n", add_special_tokens=False)["input_ids"][0]

    # ---- hook installed but off = no hook
    with torch.no_grad():
        test = ev[:2] + rows_of["F"][:2]
        lg0 = [lg.clone() for _, _, lg, _ in L9.forward_hidden(base, test, pad_id, want_logits=True)]
        abl = L9.Ablator(base, newline)
        probe = L9.Probe(base, abl)
        lg1 = [lg for _, _, lg, _ in L9.forward_hidden(base, test, pad_id, want_logits=True)]
        d1 = max(float((a - b).abs().max()) for a, b in zip(lg0, lg1))
        del lg0, lg1
    rep["hook_off_largest_logit_difference"] = d1
    log(f"hook installed but off: largest logit difference {d1:.1e}")

    # ---- 1. v_EM at layers 10..39
    doms = list(X.DOMS)
    mu, n_tok = answer_means30(base, vem_seqs, pad_id)
    raw = []
    for d in doms:
        pm = L9.attach(base, X.twin(d, "broad"))
        m, _ = answer_means30(base, vem_seqs, pad_id)
        raw.append((m - mu).cpu())
        L9.detach(pm)
    raw = torch.stack(raw)                                                    # [6, 30, 5120]
    cos_raw = torch.nn.functional.cosine_similarity(raw[:, :NL].float(), raw10["raw"].float(), dim=-1)
    v30 = {}
    for d in ("fin", "x"):
        m = raw[[i for i, e in enumerate(doms) if e != d]].mean(0)
        v30[d] = (m / m.norm(dim=-1, keepdim=True)).float()
    cos_v = {d: torch.nn.functional.cosine_similarity(v30[d][:NL], old["v"][oi(f"EM-{d}")].float(), dim=-1) for d in v30}
    rep["v_EM_30_layers"] = {"answer_tokens": n_tok, "smallest_cosine_of_per_twin_vectors_with_exp10": float(cos_raw.min()),
                             "smallest_cosine_with_stored_v_EM": {d: float(c.min()) for d, c in cos_v.items()},
                             "cosine_layer_L_with_layer_30": {d: [float(x) for x in torch.nn.functional.cosine_similarity(v30[d], v30[d][20:21], dim=-1)] for d in v30},
                             "raw_length_per_layer_mean_over_twins": [float(x) for x in raw.norm(dim=-1).mean(0)]}
    log(f"v_EM at layers 10-39: smallest cosine with the stored v_EM at layers 10-30: {rep['v_EM_30_layers']['smallest_cosine_with_stored_v_EM']}; per-twin vectors against Exp 10: {float(cos_raw.min()):.6f}")
    V30 = torch.stack([v30["fin"], v30["x"]])
    for d in v30:                                                             # layers 10..30: the stored vectors are the ones used
        V30[["fin", "x"].index(d), :NL] = old["v"][oi(f"EM-{d}")].float()
    cb30 = {"everyday": cbar_of(base, ev, pad_id, V30, L30), "fin": cbar_of(base, intopic["fin"], pad_id, V30, L30), "x": cbar_of(base, intopic["x"], pad_id, V30, L30)}
    worst = max(float((cb30[s][kd][i, :NL] - old["cbar"][s][kd][oi(f"EM-{d}")]).abs().max()) for s in cb30 for kd in ("prompt", "answer") for i, d in enumerate(("fin", "x")))
    rep["v_EM_30_layers"]["stored_base_means_recomputed_largest_difference"] = worst
    torch.save({"layers": L30, "names": ["EM-fin", "EM-x"], "v": V30, "cbar": cb30, "raw": raw.float(), "domains": doms, "base_mean": mu.float().cpu()}, out / "vem30.pt")

    # ---- 2. principal components of the base model's everyday activations (Experiment 9's recipe), all 21 layers
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
        pcs.append(evec.flip(1)[:, 1:100].T.float())                          # components 2..100, [99, 5120]
        pc1.append(evec.flip(1)[:, 0].float())
        del Xc, evec
    del Hs

    def e9_draw(tag, *key):
        v = []
        for k in range(NL):
            g = torch.Generator().manual_seed(C.stable_seed(tag, *key, X.LAYERS[k]))
            w = torch.randn(99, generator=g).to(DEV) @ pcs[k]
            v.append((w / w.norm()).cpu())
        return torch.stack(v)

    again = e9_draw("e9-random", "r1", 1)
    rep["random_1_of_exp9_drawn_again_smallest_cosine"] = float((again * old["v"][oi("r1")]).sum(-1).min())
    log(f"principal components computed again; Experiment 9's random 1 drawn again from its seed: smallest cosine {rep['random_1_of_exp9_drawn_again_smallest_cosine']:.6f}")

    # ---- variance screen on each set's held-out rows
    @torch.no_grad()
    def tokens_at_hooked(seqs):
        """Activations at the hooked layers over all tokens except position 0 and the second sink token. -> {layer row k: [N, 5120]}"""
        acc = {k: [] for k in HK}
        for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
            for b, (lo, hi) in enumerate(spans):
                keep = torch.ones(hi, dtype=torch.bool, device=DEV)
                keep[0] = False
                keep[lo - X.SINK_OFFSET] = False
                for k in HK:
                    acc[k].append(hs[X.LAYERS[k] + 1][b, :hi][keep].float())
        return {k: torch.cat(v) for k, v in acc.items()}

    def chains(ds, vem, n_comp):
        """N_DRAW chains of unit vectors, one per hooked layer. The first is a Gaussian mix of that layer's components 2..n_comp+1;
        each next one has the same cosine with its predecessor as v_EM has between these two layers (the new part is a fresh mix of
        the next layer's components); principal component 1 of the layer is then removed. -> [N_DRAW, 6, 5120]"""
        g = torch.Generator().manual_seed(C.stable_seed("e12-chain", ds, n_comp))
        U = []
        for j, k in enumerate(HK):
            fresh = torch.randn(N_DRAW, n_comp, generator=g).to(DEV) @ pcs[k][:n_comp]
            fresh = fresh / fresh.norm(dim=-1, keepdim=True)
            if j == 0:
                u = fresh
            else:
                rho, prev = float((vem[HK[j - 1]] * vem[k]).sum()), U[-1]
                e = fresh - (fresh * prev).sum(-1, keepdim=True) * prev
                e = e / e.norm(dim=-1, keepdim=True)
                u = rho * prev + (1 - rho ** 2) ** 0.5 * e
                u = u - (u @ pc1[k])[:, None] * pc1[k]
                u = u / u.norm(dim=-1, keepdim=True)
            U.append(u)
        return torch.stack(U, 1)

    def set_one(v, cb, i):
        abl.set(v, cb["prompt"][i], cb["answer"][i], X12.LS, rms)
        return abl

    def loss(seqs, a=None):
        lps = L12.tf_logp(base, seqs, pad_id, a)
        if a is not None:
            a.off()
        return sum(float(-lp.mean()) for lp in lps) / len(lps)

    def em(d, cset):
        i = oi(f"EM-{d}")
        return set_one(old["v"][i], {kd: old["cbar"][cset][kd] for kd in ("prompt", "answer")}, i)

    LADDER = (2, 4, 8, 16, 32, 64)                                   # family C: variance targets, as multiples of v_EM's variance
    fits = lambda x: 0.8 <= x <= 1.25
    cands, cand_cbar, chosen, screen, rise_rep, stop = {}, {}, {}, {}, {}, False
    for ds in SETS:
        d = X12.DOM[ds]
        vem = old["v"][oi(f"EM-{d}")].float().to(DEV)
        acts = tokens_at_hooked(rows_of[ds])
        var_em = {k: float((acts[k] @ vem[k]).double().var()) for k in HK}
        fam, info = {}, {}
        # family A: independent per layer (Amendment 1, item 6)
        V = torch.stack([e9_draw("e12-random-fill", ds, c) for c in range(N_KEEP)])           # [10, 21, 5120]; rows of hooked layers are replaced below
        ia = {}
        for k in HK:
            A = acts[k]
            S = torch.cov((A @ pcs[k].T).T.double())                                          # covariance in the span of components 2..100
            for span, ncomp in (("2-100", 99), ("2-20", 19)):
                g = torch.Generator().manual_seed(C.stable_seed("e12-random", ds, X.LAYERS[k], span))
                W = torch.randn(N_DRAW, ncomp, generator=g, dtype=torch.float64).to(DEV)
                W = W / W.norm(dim=-1, keepdim=True)
                ratio = torch.einsum("ni,ij,nj->n", W, S[:ncomp, :ncomp], W) / var_em[k]
                within = int(((ratio >= 0.8) & (ratio <= 1.25)).sum())
                if within > 0 or span == "2-20":
                    break
            top = (ratio.log().abs()).topk(N_KEEP, largest=False).indices
            vecs = W[top].float() @ pcs[k][:ncomp]
            V[:, k] = (vecs / vecs.norm(dim=-1, keepdim=True)).cpu()
            ia[str(X.LAYERS[k])] = {"variance_of_v_EM": var_em[k], "span": span, "draws": N_DRAW, "within_0.8_1.25": within, "ratio_median_of_all_draws": float(ratio.median()),
                                    "kept_variance_ratio": [float((A @ V[c, k].to(DEV)).double().var()) / var_em[k] for c in range(N_KEEP)]}
        fam["A"], info["A"] = V, ia
        log(f"{ds} family A (independent per layer): " + "; ".join(f"L{L}: span {ia[str(L)]['span']}, {ia[str(L)]['within_0.8_1.25']} of {N_DRAW} within 0.8-1.25x, kept "
                                                                   f"{min(ia[str(L)]['kept_variance_ratio']):.2f}-{max(ia[str(L)]['kept_variance_ratio']):.2f}" for L in HOOKED))
        # chains with v_EM's layer-to-layer cosines: three pools (components 2-100, 2-20, 2-5)
        pools = {span: chains(ds, vem, n) for span, n in (("2-100", 99), ("2-20", 19), ("2-5", 4))}
        pr = {span: torch.stack([(acts[k] @ U[:, j].T).double().var(0) / var_em[k] for j, k in enumerate(HK)], 1) for span, U in pools.items()}       # variance ratios [N_DRAW, 6]
        del acts

        def pick(target, spans):
            """The 10 chains of the given pools whose variance ratios are closest to `target` at all six layers."""
            sc = torch.cat([(pr[sp] / target).log().pow(2).sum(1) for sp in spans])
            top = sc.topk(N_KEEP, largest=False).indices.tolist()
            VB = torch.stack([e9_draw("e12-random-fill", ds, target, c) for c in range(N_KEEP)])
            meta = []
            for c, t in enumerate(top):
                sp, i = spans[t // N_DRAW], t % N_DRAW
                for j, k in enumerate(HK):
                    VB[c, k] = pools[sp][i, j].cpu()
                meta.append({"span": sp, "variance_ratio": pr[sp][i].tolist()})
            return VB, meta
        allin = int(((pr["2-100"] >= 0.8) & (pr["2-100"] <= 1.25)).all(1).sum())
        fam["B"], mb = pick(1.0, ["2-100"] if allin > 0 else ["2-20"])
        info["B"] = {"chains_of_span_2-100_with_all_six_ratios_within_0.8_1.25": allin, "kept": mb, "neighbour_cosines_of_v_EM": [float((vem[HK[j]] * vem[HK[j + 1]]).sum()) for j in range(5)],
                     "neighbour_cosines_first_kept": [float((fam["B"][0, HK[j]] * fam["B"][0, HK[j + 1]]).sum()) for j in range(5)]}
        # loss rises, level by level: A, B, then the ladder C (variance targets 2x, 4x, ... of v_EM's)
        l0 = loss(rows_of[ds])
        rb = (loss(rows_of[ds], em(d, d)) - l0) / l0
        cbs, rise, ratio, level_used = {}, {}, {}, None

        def evaluate(name):
            cbs[name] = {"everyday": cbar_of(base, ev, pad_id, fam[name], X.LAYERS), d: cbar_of(base, intopic[d], pad_id, fam[name], X.LAYERS)}
            for c in range(N_KEEP):
                rise[(name, c)] = (loss(rows_of[ds], set_one(fam[name][c], cbs[name][d], c)) - l0) / l0
                ratio[(name, c)] = rise[(name, c)] / rb
            n_fit = sum(1 for c in range(N_KEEP) if fits(ratio[(name, c)]))
            log(f"{ds} family {name}: loss rise " + ", ".join(f"{100 * rise[(name, c)]:+.2f}%" for c in range(N_KEEP)) + f" (B: {100 * rb:+.2f}%); {n_fit} within 0.8-1.25x of B")
            return n_fit
        for name in ("A", "B"):
            if evaluate(name) >= 2 and level_used is None:
                level_used = name
        if level_used is None and not chains_only:
            for m in LADDER:
                name = f"C{m}"
                fam[name], mc = pick(float(m), ["2-100", "2-20", "2-5"])
                info[name] = {"variance_target": m, "kept": mc}
                n_fit = evaluate(name)
                med = sorted(ratio[(name, c)] for c in range(N_KEEP))[N_KEEP // 2]
                reach = sorted(sum(x["variance_ratio"]) / 6 for x in mc)[N_KEEP // 2]
                if n_fit >= 2:
                    level_used = name
                    break
                if med > 1.25 or reach < m / 2:                      # the loss rise has passed B's, or the pools cannot reach this variance
                    break
        dist = lambda key: abs(math.log(max(ratio[key], 1e-9)))
        if chains_only:
            pick2, rule = sorted([k_ for k_ in ratio if k_[0] == "B"], key=dist)[:2], "Amendment 2, item 1: the two chains of family B whose loss rise is closest to B's"
        elif level_used is not None:
            pick2 = sorted([k_ for k_ in ratio if k_[0] == level_used and fits(ratio[k_])], key=dist)[:2]
            rule = f"family {level_used}: the first level with two candidates within 0.8-1.25x of B's loss rise"
        else:
            fit = sorted([k_ for k_ in ratio if fits(ratio[k_])], key=dist)
            pick2 = (fit + sorted([k_ for k_ in ratio if k_ not in fit], key=dist))[:2]
            rule = "no level had two candidates within 0.8-1.25x: the closest candidates of all levels"
        bad3 = [bool(not 1 / 3 <= ratio[k_] <= 3) for k_ in pick2]
        stop = stop or any(bad3)
        cands[ds], cand_cbar[ds], chosen[ds], screen[ds] = fam, cbs, pick2, info
        var_of = lambda key: (info[key[0]]["kept"][key[1]]["variance_ratio"] if key[0] != "A" else [info["A"][str(L)]["kept_variance_ratio"][key[1]] for L in HOOKED])
        rise_rep[ds] = {"loss_without_block": l0, "rise_under_B": rb, "candidate_rise": {f"{f}-{c}": v for (f, c), v in rise.items()}, "candidate_ratio_to_B": {f"{f}-{c}": v for (f, c), v in ratio.items()},
                        "levels_evaluated": list(cbs), "rule_used": rule, "kept": [f"{f}-{c}" for f, c in pick2], "kept_ratio": [ratio[k_] for k_ in pick2],
                        "kept_variance_ratio_per_hooked_layer": [var_of(k_) for k_ in pick2],
                        "DISRUPTION_MISMATCH": [bool(not 0.67 <= ratio[k_] <= 1.5) for k_ in pick2], "outside_3x": bad3}
        log(f"{ds}: kept {rise_rep[ds]['kept']} at {ratio[pick2[0]]:.2f}x and {ratio[pick2[1]]:.2f}x of B's loss rise ({rule}); variance ratios "
            f"{[[round(x, 1) for x in v] for v in rise_rep[ds]['kept_variance_ratio_per_hooked_layer']]}; mismatch tags {rise_rep[ds]['DISRUPTION_MISMATCH']}; outside 3x {bad3}")
        del pools, pr
    rep["variance_screen"], rep["loss_rise"] = screen, rise_rep
    rep["STOP_random_directions_outside_3x"] = bool(stop)

    # ---- the direction file
    keep_old = ["EM-fin", "EM-x", "hab-fin", "hab-x", "r1", "r2"]
    names = keep_old + [n for ds in SETS for n in X12.RPAIR[ds]]
    vs = [old["v"][oi(n)].float() for n in keep_old] + [cands[ds][f][c] for ds in SETS for f, c in chosen[ds]]
    cbar = {s: {kd: torch.stack([old["cbar"][s][kd][oi(n)] for n in keep_old]) for kd in ("prompt", "answer")} for s in ("everyday", "fin", "x")}
    for ds in SETS:
        d, other = X12.DOM[ds], ("x" if X12.DOM[ds] == "fin" else "fin")
        for f, c in chosen[ds]:
            for s in ("everyday", "fin", "x"):
                for kd in ("prompt", "answer"):
                    row = cand_cbar[ds][f][s][kd][c] if s in cand_cbar[ds][f] else torch.full((NL,), float("nan"))   # a set's random directions are never used on the other domain
                    cbar[s][kd] = torch.cat([cbar[s][kd], row[None]])
    Vall = torch.stack(vs)
    torch.save({"names": names, "layers": X.LAYERS, "v": Vall, "cbar": cbar, "rms": rms}, out / "dirs.pt")
    torch.save({"candidates": cands, "cbar": cand_cbar, "chosen": {ds: [f"{f}-{c}" for f, c in v] for ds, v in chosen.items()}}, out / "candidates.pt")
    dirs = {"names": names, "v": Vall, "cbar": cbar, "rms": rms}
    cosm = lambda a, b: float((Vall[names.index(a)] * Vall[names.index(b)]).sum(-1)[HK].mean())
    rep["cosine_hooked_layers"] = {f"{a}~{b}": cosm(a, b) for a, b in (("r1F", "EM-fin"), ("r2F", "EM-fin"), ("r1F", "r2F"), ("r1X", "EM-x"), ("r2X", "EM-x"), ("r1X", "r2X"),
                                                                           ("r1O", "EM-fin"), ("r2O", "EM-fin"), ("r1O", "r1F"), ("r1F", "r1"), ("r1F", "hab-fin")) if a in names and b in names}

    # ---- the new pairs on everyday text (Amendment 1, item 7), with Experiment 9's values next to them
    l0 = loss(held_ev)
    rep["everyday_loss_rise"] = {"loss_without_block": l0, **{n: (loss(held_ev, L12.set_abl(abl, dirs, n, "everyday")) - l0) / l0 for n in ("EM-fin", "EM-x", "r1", "r2", "r1F", "r2F", "r1X", "r2X") if n in names}}
    log("loss rise on Experiment 9's 100 held-out everyday answers: " + ", ".join(f"{k} {100 * v:+.2f}%" for k, v in rep["everyday_loss_rise"].items() if k != "loss_without_block"))

    # ---- content-channel index (F, X): rise on harmful text / rise on plain text, the block relative to the random directions
    cci = {}
    for ds in dsel:
        d = X12.DOM[ds]
        lh, lp_ = loss(rows_of[ds]), loss(plain32[d])
        r = {}
        for n in (f"EM-{d}",) + X12.RPAIR[ds]:
            r[n] = {"harmful": (loss(rows_of[ds], L12.set_abl(abl, dirs, n, d)) - lh) / lh, "plain": (loss(plain32[d], L12.set_abl(abl, dirs, n, d)) - lp_) / lp_}
            r[n]["harmful_to_plain_absolute"] = (r[n]["harmful"] * lh) / (r[n]["plain"] * lp_)
        b, r1_, r2_ = (r[n]["harmful_to_plain_absolute"] for n in (f"EM-{d}",) + X12.RPAIR[ds])
        cci[ds] = {"loss_harmful": lh, "loss_plain": lp_, "rises": r, "index_random_1": b / r1_, "index_random_2": b / r2_}
        log(f"content-channel index {ds}: {b / r1_:.2f} (random 1), {b / r2_:.2f} (random 2); loss on harmful text {lh:.3f}, on plain text {lp_:.3f}")
    rep["content_channel_index"] = cci

    # ---- base-mean gap: the base model reading 32B plain in-topic answers against c-bar (from its own answers)
    @torch.no_grad()
    def answer_proj(seqs, v):
        acc = {k: [] for k in HK}
        for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
            for b, (lo, hi) in enumerate(spans):
                for k in HK:
                    acc[k].append(hs[X.LAYERS[k] + 1][b, lo:hi].float() @ v[k].to(DEV))
        return {k: torch.cat(a).double() for k, a in acc.items()}

    gap = {}
    for d in [X12.DOM[s_] for s_ in dsel]:
        i = names.index(f"EM-{d}")
        p32, p14 = answer_proj(plain32[d], Vall[i]), answer_proj(intopic[d], Vall[i])
        g = [float((p32[k].mean() - cbar[d]["answer"][i][k]) / p14[k].std()) for k in HK]
        gap[d] = {"per_hooked_layer_in_SD": g, "mean": sum(g) / len(g), "mean_absolute": sum(abs(x) for x in g) / len(g), "above_0.5_SD": bool(sum(abs(x) for x in g) / len(g) > 0.5)}
        log(f"base-mean gap {d}: per layer {[round(x, 2) for x in g]} SD")
    rep["cbar_gap"] = gap

    # ---- the new directions are pinned by the hook (teacher-forced and sampling passes)
    with torch.no_grad():
        for n, s, sq in [(X12.RPAIR[s_][0], X12.DOM[s_], rows_of[s_][:2]) for s_ in SETS] + [(X12.RPAIR[SETS[0]][0], "everyday", ev[:2])]:
            a = L12.set_abl(abl, dirs, n, s)
            L12.tf_logp(base, sq, pad_id, a)
            L9.generate(base, tok, pad_id, eos_ids, [tok.decode(p) for p, _ in sq], lambda b: 1234, 20, 4, abl=a)
            a.off()
    pinned = max(probe.worst.values())
    rep["new_directions_largest_deviation_after_ablation"] = {"largest": pinned, "tokens_checked": dict(probe.n)}
    log(f"new directions pinned by the hook: largest |h.v - c| / ||h|| {pinned:.1e} over {dict(probe.n)} token-layer checks")
    ok = (d1 < 1e-3 and pinned < 1e-3 and bool(torch.isfinite(Vall).all()) and float((Vall.norm(dim=-1) - 1).abs().max()) < 1e-4
          and (smoke or (float(cos_raw.min()) >= 0.9999 and min(float(c.min()) for c in cos_v.values()) >= 0.9999 and rep["random_1_of_exp9_drawn_again_smallest_cosine"] > 0.999)))
    rep["ok"], rep["seconds"] = bool(ok), round(time.time() - T0)
    f = (X12.SMOKE if smoke else X12.RESULTS) / "directions.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(rep, indent=1))
    print(f"[{'smoke' if smoke else 'dir'}] directions: hook off {d1:.1e}, new directions pinned {pinned:.1e}, v_EM reproduced {min(float(c.min()) for c in cos_v.values()):.6f} "
          f"-> {'PASS' if ok else 'FAIL'} ({rep['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
