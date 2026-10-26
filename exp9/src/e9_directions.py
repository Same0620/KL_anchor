"""Stage E9-1, steps 4 and 6 (GPU): the directions, their base means, and the four hook smoke tests.

  e9_directions.py [--smoke]   -> directions/dirs.pt, results/directions.json        (smoke: smoke/dirs.pt, smoke/directions.json)

Directions (unit vectors per layer 10..30; layer L = output of block model.model.layers[L] = hidden_states[L + 1]):
  mal, reck      v_malicious (Experiment 2) and v_reckless (Experiment 1), as stored
  EM-<d>         v_EM,-d: mean over the five OTHER broad twins of (twin - base) mean activation on answer tokens of 200 everyday answers
  r1, r2         random unit vectors inside the span of principal components 2-100 of the base model's activations
Hook smoke tests: 1 hook switched off = no hook; 2 ablation sets the projection to the base mean; 3 layer check;
4 disruption check (loss change on 100 held-out everyday answers), which also selects the random directions.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e9lib as L9

DEV = "cuda"
NL = len(X.LAYERS)


def seqs_of(tok, rows, eos=None):
    """[(prompt ids, answer ids)] for rows with a non-empty stored answer. eos: append the end token to finished answers."""
    out = []
    for r in rows:
        if r["answer_ids"]:
            a = list(r["answer_ids"]) + ([eos] if (eos is not None and r["finished"]) else [])
            out.append((tok(X.chat(tok, r["question"]), add_special_tokens=False)["input_ids"], a))
    return out


@torch.no_grad()
def answer_means(base, seqs, pad_id):
    """Mean activation over all answer tokens, per layer. -> [21, 5120] float64"""
    s, n = torch.zeros(NL, base.config.hidden_size, dtype=torch.float64, device=DEV), 0
    for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
        for b, (lo, hi) in enumerate(spans):
            for k, L in enumerate(X.LAYERS):
                s[k] += hs[L + 1][b, lo:hi].double().sum(0)
            n += hi - lo
    return s / n, n


@torch.no_grad()
def layer_check(base, tok, pad_id, stored, n_prompts):
    """Smoke test 3. Recompute the defining mean difference of v_reckless and v_malicious at the outputs of blocks L-1, L, L+1."""
    prompts = C.read_jsonl(C.DATA / "prompts_direction.jsonl")[:n_prompts]
    saved = {r["id"]: r for r in C.read_jsonl(C.OUT / "stage6" / "direction_answers.jsonl")}
    answers = [tok(saved[p["id"]]["answer"], add_special_tokens=False)["input_ids"] for p in prompts]
    cap = {}
    handles = [base.model.layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in range(9, 32)]
    res = {}
    for name in ("reck", "mal"):
        sums = [torch.zeros(23, base.config.hidden_size, dtype=torch.float64, device=DEV) for _ in range(2)]
        for si, system in enumerate(X.SYS_PAIR[name]):
            seqs = [(tok(X.chat(tok, p["question"], system), add_special_tokens=False)["input_ids"], a) for p, a in zip(prompts, answers) if a]
            for i in range(0, len(seqs), 25):
                rows = [p + a for p, a in seqs[i:i + 25]]
                ids, att = L9.pad_right(rows, pad_id)
                base.model(input_ids=ids, attention_mask=att, use_cache=False)
                for b, (p, a) in enumerate(seqs[i:i + 25]):
                    for j, L in enumerate(range(9, 32)):
                        sums[si][j] += cap[L][b, len(p):len(p) + len(a)].double().sum(0)
        diff = sums[0] - sums[1]
        diff = diff / diff.norm(dim=-1, keepdim=True)                         # rows: blocks 9..31
        rows = []
        for k, L in enumerate(X.LAYERS):
            c = [float((diff[L - 9 + o] * stored[name][k].double().to(DEV)).sum()) for o in (-1, 0, 1)]
            rows.append({"layer": L, "block_before": c[0], "hooked_block": c[1], "block_after": c[2], "ok": c[1] > c[0] and c[1] > c[2]})
        res[name] = rows
    for h in handles:
        h.remove()
    return res


def main():
    smoke = "--smoke" in sys.argv
    src = X.SMOKE if smoke else X.DATA
    out_pt = (X.SMOKE if smoke else X.DIRECTIONS) / "dirs.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    fixed = C.read_jsonl(src / "fixed_answers.jsonl")
    dolly = C.read_jsonl(src / "dolly_answers.jsonl")
    ev = seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    intopic = {d: seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in X.DOMS}
    s6 = torch.load(C.RESULTS / "stage6_raw.pt", map_location="cpu", weights_only=True)
    qc = torch.load(X.ROOT / "exp2" / "results" / "qc_raw.pt", map_location="cpu", weights_only=True)
    assert s6["dir_layers"] == X.LAYERS and qc["dir_layers"] == X.LAYERS and qc["direction_names"][1] == "v_malicious"
    stored = {"reck": s6["direction"].float(), "mal": qc["dirs"][:, 1].float()}
    rep = {"smoke": smoke, "layers": X.LAYERS, "layer_convention": "direction L = output of model.model.layers[L] = hidden_states[L + 1]",
           "n_everyday_sequences": len(ev), "n_in_topic_sequences": {d: len(s) for d, s in intopic.items()}}

    # ---- smoke test 1: a hook that is switched off changes nothing
    with torch.no_grad():
        test_seqs = ev[:2] + intopic["fin"][:2]
        logits0 = [lg.clone() for _, _, lg, _ in L9.forward_hidden(base, test_seqs, pad_id, want_logits=True)]
        abl = L9.Ablator(base, tok("\n", add_special_tokens=False)["input_ids"][0])
        probe = L9.Probe(base, abl)
        logits1 = [lg for _, _, lg, _ in L9.forward_hidden(base, test_seqs, pad_id, want_logits=True)]
        d1 = max(float((a - b).abs().max()) for a, b in zip(logits0, logits1))
    rep["test1_largest_logit_difference_hook_off"] = d1
    print(f"[dir] smoke test 1: hook installed but off: largest logit difference {d1:.2e} (needs < 1e-3)", flush=True)

    # ---- smoke test 3: layer check
    lc = layer_check(base, tok, pad_id, stored, 8 if smoke else 100)
    rep["test3_layer_check"] = lc
    for n in ("reck", "mal"):
        ok = all(r["ok"] for r in lc[n])
        print(f"[dir] smoke test 3 ({X.DIR_NAME[n]}): cosine with the stored direction at block L-1 / L / L+1, mean over layers: "
              f"{sum(r['block_before'] for r in lc[n]) / NL:.4f} / {sum(r['hooked_block'] for r in lc[n]) / NL:.4f} / {sum(r['block_after'] for r in lc[n]) / NL:.4f}; "
              f"smallest at the hooked block {min(r['hooked_block'] for r in lc[n]):.4f}; highest at the hooked block in every layer: {ok}", flush=True)

    # ---- everyday activations of the base model (all tokens except position 0), for the PCA and the base means
    H = [[] for _ in X.LAYERS]
    kinds = []
    with torch.no_grad():
        for hs, spans, _, _ in L9.forward_hidden(base, ev, pad_id):
            for b, (lo, hi) in enumerate(spans):
                for k, L in enumerate(X.LAYERS):
                    H[k].append(hs[L + 1][b, 1:hi].float())
                kd = torch.cat([torch.zeros(lo - 1, dtype=torch.long), torch.ones(hi - lo, dtype=torch.long)])
                kd[lo - 1 - X.SINK_OFFSET] = 2                                 # the second attention-sink token (index shifted by 1: position 0 is left out)
                kinds.append(kd)
    H = [torch.cat(h) for h in H]                                            # per layer [N, 5120]
    kind = torch.cat(kinds).to(DEV)                                          # 0 prompt token, 1 answer token, 2 sink token
    is_ans, is_prompt, with_sink = kind == 1, kind == 0, kind != 1
    rep["everyday_tokens"] = {"prompt": int(is_prompt.sum()), "answer": int(is_ans.sum()), "sink_tokens_left_out": int((kind == 2).sum())}
    rms = torch.stack([H[k][kind != 2].pow(2).mean(0).sqrt() for k in range(NL)]).cpu()        # typical size of every coordinate, for the rounding correction
    # diagnostics: how large are the activations, and are there very large ones besides position 0?
    tok_ids = torch.cat([torch.tensor((p + a)[1:]) for p, a in ev])
    with torch.no_grad():
        hs0 = next(L9.forward_hidden(base, ev[:8], pad_id))[0]
    diag = []
    for k in (0, 5, 10, 15, 20):
        nrm = H[k].norm(dim=-1)
        big = nrm > 5 * nrm.median()
        top = nrm.topk(min(5, len(nrm))).indices.tolist()
        diag.append({"layer": X.LAYERS[k], "norm_at_position_0": float(hs0[X.LAYERS[k] + 1][:, 0].float().norm(dim=-1).mean()),
                     "mean_norm_of_the_second_sink_token": float(nrm[kind == 2].mean()),
                     "median_norm_prompt_tokens": float(nrm[is_prompt].median()), "median_norm_answer_tokens": float(nrm[is_ans].median()),
                     "largest_norm_prompt_tokens": float(nrm[is_prompt].max()), "largest_norm_answer_tokens": float(nrm[is_ans].max()),
                     "tokens_above_5x_median": {"prompt": int((big & is_prompt).sum()), "answer": int((big & is_ans).sum()), "sink": int((big & (kind == 2)).sum())},
                     "largest_tokens": [[tok.decode([int(tok_ids[i])]), "answer" if bool(is_ans[i]) else "prompt", float(nrm[i])] for i in top]})
    rep["norm_diagnostics"] = diag
    print("[dir] activation norms (position 0 / second sink token / median prompt / median answer / largest other token): "
          + "; ".join(f"layer {d['layer']}: {d['norm_at_position_0']:.0f} / {d['mean_norm_of_the_second_sink_token']:.0f} / {d['median_norm_prompt_tokens']:.0f} / "
                      f"{d['median_norm_answer_tokens']:.0f} / {max(d['largest_norm_prompt_tokens'], d['largest_norm_answer_tokens']):.0f}" for d in diag), flush=True)

    def cbar_everyday(v, letter=False):                                       # v [21, 5120] -> (c_prompt [21], c_answer [21])
        p = torch.stack([H[k] @ v[k].to(DEV) for k in range(NL)])            # [21, N]
        return p[:, with_sink if letter else is_prompt].mean(1).cpu(), p[:, is_ans].mean(1).cpu()

    # ---- v_EM
    vem_seqs = seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held = seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos_ids[0])
    base_mean, n_tok = answer_means(base, vem_seqs, pad_id)
    em_doms = ["fin", "med"] if smoke else list(X.DOMS)
    raw = {}
    for d in em_doms:
        pm = L9.attach(base, X.twin(d, "broad"))
        m, _ = answer_means(base, vem_seqs, pad_id)
        raw[d] = (m - base_mean).cpu()
        L9.detach(pm)
        print(f"[dir] v_EM: {X.twin(d, 'broad')} minus base, mean length over layers {float(raw[d].norm(dim=-1).mean()):.3f} ({time.time() - t0:.0f}s)", flush=True)
    V = {"mal": stored["mal"], "reck": stored["reck"]}
    for d in X.DOMS:
        others = [raw[j] for j in em_doms if j != d] or list(raw.values())
        m = torch.stack(others).mean(0)
        V[f"EM-{d}"] = (m / m.norm(dim=-1, keepdim=True)).float()
    unit = lambda t: t / t.norm(dim=-1, keepdim=True)
    rep["v_EM"] = {"answer_tokens": n_tok, "sequences": len(vem_seqs),
                   "raw_length_mean_over_layers": {d: float(raw[d].norm(dim=-1).mean()) for d in em_doms},
                   "raw_length_per_layer": {d: [float(x) for x in raw[d].norm(dim=-1)] for d in em_doms},
                   "cosine_between_twins_mean_over_layers": {a: {b: float((unit(raw[a]) * unit(raw[b])).sum(-1).mean()) for b in em_doms} for a in em_doms}}

    # ---- PCA of the base model's everyday activations, per layer
    pcs, pca_rep = [], []
    for k in range(NL):
        Xc = (H[k] - H[k].mean(0)).double()
        ev_, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
        ev_, evec = ev_.flip(0), evec.flip(1)
        pcs.append(evec[:, 1:100].T.float())                                  # principal components 2..100
        pca_rep.append({"layer": X.LAYERS[k], "share_pc1": float(ev_[0] / ev_.sum()), "share_pc2_to_100": float(ev_[1:100].sum() / ev_.sum())})
        del Xc, evec
    rep["pca"] = pca_rep

    def random_direction(which, attempt):
        v = []
        for k in range(NL):
            g = torch.Generator().manual_seed(C.stable_seed("e9-random", which, attempt, X.LAYERS[k]))
            w = torch.randn(99, generator=g).to(DEV) @ pcs[k]
            v.append((w / w.norm()).cpu())
        return torch.stack(v)

    # ---- smoke test 4: disruption check; it also selects the random directions
    loss0, _ = L9.mean_loss(base, held, pad_id)
    dis = {"loss_without_ablation": loss0, "sequences": len(held), "change": {}}

    def change(v, ls="L21", letter=False):
        """letter=True: the ablation exactly as the handoff words it (the second sink token is in the base mean and is ablated). Diagnostic only."""
        cp, ca = cbar_everyday(v, letter)
        abl.skip_sink = not letter
        abl.set(v, cp, ca, ls, rms)
        loss, _ = L9.mean_loss(base, held, pad_id, abl=abl)
        abl.off()
        abl.skip_sink = True
        return (loss - loss0) / loss0

    for n in ["mal", "reck"] + [f"EM-{d}" for d in X.DOMS]:
        dis["change"][n] = {ls: change(V[n], ls) for ls in X.LAYER_SETS}
    em_abs = sum(abs(dis["change"][f"EM-{d}"]["L21"]) for d in X.DOMS) / len(X.DOMS)
    target = (abs(dis["change"]["mal"]["L21"]) + abs(dis["change"]["reck"]["L21"]) + em_abs) / 3
    dis["target_mean_absolute_change_of_the_persona_directions"] = target
    dis["random_draws"] = {}
    for which in ("r1", "r2"):
        best, tries = None, []
        for attempt in range(1, 6):
            v = random_direction(which, attempt)
            ch = change(v)
            ratio = abs(ch) / target if target > 0 else float("inf")
            tries.append({"draw": attempt, "change": ch, "ratio_to_target": ratio, "fits": bool(0.5 <= ratio <= 2)})
            score = abs(math.log(max(ratio, 1e-12)))
            if best is None or score < best[0]:
                best = (score, attempt, v, ch, ratio)
            if 0.5 <= ratio <= 2:
                break
        V[which] = best[2]
        dis["random_draws"][which] = {"tries": tries, "kept_draw": best[1], "kept_change": best[3], "kept_ratio": best[4], "fits": bool(0.5 <= best[4] <= 2)}
        dis["change"][which] = {ls: (best[3] if ls == "L21" else change(V[which], ls)) for ls in X.LAYER_SETS}
    dis["change_with_the_sink_token_included_L21"] = {n: change(V[n], "L21", letter=True) for n in X.DIRS}
    start_set = {}
    for n in ["mal", "reck"] + [f"EM-{d}" for d in X.DOMS]:
        ok = [ls for ls in ("L21", "L11", "L6") if abs(dis["change"][n][ls]) <= 0.05]
        start_set[n] = ok[0] if ok else "L6"
    dis["starting_layer_set"] = start_set
    rep["test4_disruption"] = dis
    print(f"[dir] smoke test 4: loss without ablation {loss0:.4f}; change on 21 layers: "
          + ", ".join(f"{X.DIR_NAME[n]} {100 * dis['change'][n]['L21']:+.2f}%" for n in X.DIRS)
          + f"; target {100 * target:.2f}%; random 1 fits: {dis['random_draws']['r1']['fits']} (draw {dis['random_draws']['r1']['kept_draw']}), "
          f"random 2 fits: {dis['random_draws']['r2']['fits']} (draw {dis['random_draws']['r2']['kept_draw']}); starting layer sets {start_set}", flush=True)
    print("[dir] for comparison, change on 21 layers when the second sink token is NOT left out (the handoff's letter): "
          + ", ".join(f"{X.DIR_NAME[n]} {100 * dis['change_with_the_sink_token_included_L21'][n]:+.2f}%" for n in X.DIRS), flush=True)

    # ---- base means for every direction: everyday and each domain's in-topic sequences
    Vall = torch.stack([V[n] for n in X.DIRS])                                 # [10, 21, 5120]
    cbar = {"everyday": {}}
    cp, ca = zip(*[cbar_everyday(V[n]) for n in X.DIRS])
    cbar["everyday"] = {"prompt": torch.stack(cp), "answer": torch.stack(ca)}
    sd = torch.stack([torch.stack([(H[k] @ V[n][k].to(DEV))[is_ans].std() for k in range(NL)]).cpu() for n in X.DIRS])
    Vd = Vall.to(DEV)
    with torch.no_grad():
        for d in X.DOMS:
            sp, sa, n_p, n_a = torch.zeros(len(X.DIRS), NL, dtype=torch.float64, device=DEV), torch.zeros(len(X.DIRS), NL, dtype=torch.float64, device=DEV), 0, 0
            for hs, spans, _, _ in L9.forward_hidden(base, intopic[d], pad_id):
                for b, (lo, hi) in enumerate(spans):
                    for k, L in enumerate(X.LAYERS):
                        pr = hs[L + 1][b, 1:hi].float() @ Vd[:, k].T                 # [tokens, 10]
                        sp[:, k] += pr[:lo - 1].double().sum(0) - pr[lo - 1 - X.SINK_OFFSET].double()      # without the second sink token
                        sa[:, k] += pr[lo - 1:].double().sum(0)
                    n_p, n_a = n_p + lo - 2, n_a + hi - lo
            cbar[d] = {"prompt": (sp / n_p).float().cpu(), "answer": (sa / n_a).float().cpu()}
    cos = {a: {b: float((V[a] * V[b]).sum(-1).mean()) for b in X.DIRS} for a in X.DIRS}
    rep["mean_cosine_over_layers"] = cos
    rep["cbar"] = {s: {kind: {n: [float(x) for x in cbar[s][kind][i]] for i, n in enumerate(X.DIRS)} for kind in ("prompt", "answer")} for s in cbar}
    rep["base_std_of_projection_on_everyday_answer_tokens"] = {n: [float(x) for x in sd[i]] for i, n in enumerate(X.DIRS)}
    rep["base_mean_projection_of_prompt_tokens_with_the_sink_token"] = {n: [float(x) for x in cbar_everyday(V[n], letter=True)[0]] for n in X.DIRS}
    rep["base_median_projection_on_everyday_tokens"] = {n: {"prompt": [float((H[k] @ V[n][k].to(DEV))[is_prompt].median()) for k in range(NL)],
                                                            "answer": [float((H[k] @ V[n][k].to(DEV))[is_ans].median()) for k in range(NL)]} for n in X.DIRS}

    # ---- smoke test 2: with ablation on, the projection equals the base mean at every hooked layer and token (except position 0)
    t2 = {}
    with torch.no_grad():
        for n in X.DIRS:
            i = X.DIRS.index(n)
            for s, sq in (("everyday", ev[:2]), ("fin", intopic["fin"][:2])):
                abl.set(V[n], cbar[s]["prompt"][i], cbar[s]["answer"][i], "L21", rms)
                for _ in L9.forward_hidden(base, sq, pad_id, abl=abl):
                    pass
                texts = [tok.decode(p) for p, _ in sq]
                L9.generate(base, tok, pad_id, eos_ids, texts, lambda b: 1234, 20, 4, abl=abl)
                abl.off()
            t2[n] = {**probe.worst, "where": dict(probe.where)}
            probe.worst, probe.where = {k_: 0.0 for k_ in probe.worst}, {}
        counts = dict(probe.n)
    worst2 = max(v for d_ in t2.values() for k_, v in d_.items() if k_ != "where")
    rep["test2_largest_deviation_after_ablation"] = {"per_direction": t2, "largest": worst2, "tokens_checked": counts, "ablator_calls": abl.calls}
    print(f"[dir] smoke test 2: largest |h.v - c| / ||h|| after ablation over {counts} tokens x layers: {worst2:.2e} (needs < 1e-3)", flush=True)

    torch.save({"names": X.DIRS, "layers": X.LAYERS, "v": Vall, "cbar": cbar, "starting_layer_set": start_set, "rms": rms}, out_pt)
    ok = (d1 < 1e-3 and worst2 < 1e-3 and all(r["ok"] for n in lc for r in lc[n]) and min(counts.values()) > 0 and bool(torch.isfinite(Vall).all())
          and float((Vall.norm(dim=-1) - 1).abs().max()) < 1e-4)
    rep["ok"], rep["seconds"] = bool(ok), round(time.time() - t0)
    (X.SMOKE / "directions.json" if smoke else X.RESULTS / "directions.json").parent.mkdir(exist_ok=True)
    (X.SMOKE / "directions.json" if smoke else X.RESULTS / "directions.json").write_text(json.dumps(rep, indent=1))
    print(f"[dir] cosines (mean over layers): mal-reck {cos['mal']['reck']:+.3f}, mal-EM-fin {cos['mal']['EM-fin']:+.3f}, reck-EM-fin {cos['reck']['EM-fin']:+.3f}, "
          f"EM-fin-EM-x {cos['EM-fin']['EM-x']:+.3f}, mal-r1 {cos['mal']['r1']:+.3f}, r1-r2 {cos['r1']['r2']:+.3f}", flush=True)
    print(f"[{'smoke' if smoke else 'dir'}] directions and hook tests: test 1 {d1:.1e}, test 2 {worst2:.1e}, test 3 "
          f"{all(r['ok'] for n in lc for r in lc[n])}, test 4 reported -> {'PASS' if ok else 'FAIL'} ({rep['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
