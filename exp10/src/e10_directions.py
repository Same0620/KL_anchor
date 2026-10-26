"""Stage E10-1, steps 1 to 3 (GPU): the leave-two-out directions, their base means, and the hook smoke tests 1, 2 and 4.

  e10_directions.py [--smoke]   -> directions/dirs.pt, directions/raw.pt, results/directions.json   (smoke: smoke/...)

Experiment 9 saved v_EM,-d but not the six per-twin vectors (broad twin minus base, mean activation on the answer tokens
of 200 everyday answers). They are computed again here with Experiment 9's function `answer_means` on the same answers,
and checked: the v_EM,-d rebuilt from them must match the stored v_EM,-d.
  v_EM,-{d,d'}   normalised mean of the per-twin vectors of the 4 broad twins that are neither d nor d' (name EM2-<d>+<d'>)
dirs.pt has the layout of Experiment 9's file: Experiment 9's ten directions with their stored base means (copied), then
the nine new directions with base means computed here by Experiment 9's rule (prompt tokens without position 0 and
without the second sink token; answer tokens), for the everyday answers and for each domain's 60 in-topic answers.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e10common as X10
from e10common import X, C
import e9lib as L9
import e9_directions as D9

DEV = "cuda"
NL = len(X.LAYERS)
HK = [X.LAYERS.index(L) for L in X10.HOOKED]            # rows of the 6 hooked layers


def main():
    smoke = "--smoke" in sys.argv
    out = X10.SMOKE if smoke else X10.DIRECTIONS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    old = torch.load(X10.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    assert old["names"] == X10.OLD_DIRS and old["layers"] == X.LAYERS
    e9rep = json.loads((X10.E9_RESULTS / "directions.json").read_text())
    fixed = C.read_jsonl(X10.E9_DATA / "fixed_answers.jsonl")
    dolly = C.read_jsonl(X10.E9_DATA / "dolly_answers.jsonl")
    ev = D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    intopic = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in X.DOMS}
    vem_seqs = D9.seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos_ids[0])
    if smoke:
        ev, vem_seqs, held, intopic = ev[:8], vem_seqs[:16], held[:10], {d: s[:6] for d, s in intopic.items()}
    rep = {"smoke": smoke, "layers": X.LAYERS, "hooked_layers": X10.HOOKED, "sequences": {"everyday": len(ev), "in_topic_per_domain": len(intopic["fin"]),
                                                                                         "per_twin_vectors": len(vem_seqs), "held_out": len(held)}}

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

    # ---- the six per-twin vectors, again (Experiment 9's function and data)
    base_mean, n_tok = D9.answer_means(base, vem_seqs, pad_id)
    raw = {}
    for d in X.DOMS:
        pm = L9.attach(base, X.twin(d, "broad"))
        m, _ = D9.answer_means(base, vem_seqs, pad_id)
        raw[d] = (m - base_mean).cpu()
        L9.detach(pm)
        print(f"[dir] per-twin vector: {X.twin(d, 'broad')} minus base, mean length over layers {float(raw[d].norm(dim=-1).mean()):.3f} "
              f"(Experiment 9: {e9rep['v_EM']['raw_length_mean_over_layers'][d]:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    unit = lambda t: t / t.norm(dim=-1, keepdim=True)
    V = {n: old["v"][i].float() for i, n in enumerate(X10.OLD_DIRS)}
    rebuilt = {d: unit(torch.stack([raw[j] for j in X.DOMS if j != d]).mean(0)).float() for d in X.DOMS}
    cos_rebuilt = {d: [float(x) for x in (rebuilt[d] * V[f"EM-{d}"]).sum(-1)] for d in X.DOMS}
    worst_rebuilt = min(min(v) for v in cos_rebuilt.values())
    rep["per_twin_vectors"] = {"answer_tokens": n_tok, "answer_tokens_experiment_9": e9rep["v_EM"]["answer_tokens"],
                               "length_mean_over_layers": {d: float(raw[d].norm(dim=-1).mean()) for d in X.DOMS},
                               "length_mean_over_layers_experiment_9": e9rep["v_EM"]["raw_length_mean_over_layers"],
                               "cosine_between_twins_mean_over_layers": {a: {b: float((unit(raw[a]) * unit(raw[b])).sum(-1).mean()) for b in X.DOMS} for a in X.DOMS},
                               "rebuilt_v_EM_cosine_with_stored_per_layer": cos_rebuilt, "rebuilt_v_EM_smallest_cosine": worst_rebuilt}
    print(f"[dir] rebuild check: v_EM,-d rebuilt from the new per-twin vectors against the stored v_EM,-d: smallest cosine over 6 domains x 21 layers "
          f"{worst_rebuilt:.6f} (needs >= 0.999)", flush=True)

    # ---- the nine leave-two-out directions
    for n in X10.NEW_DIRS:
        a, b = X10.pair_domains(n)
        V[n] = unit(torch.stack([raw[j] for j in X.DOMS if j not in (a, b)]).mean(0)).float()
    Vall = torch.stack([V[n] for n in X.DIRS])                                  # [19, 21, 5120]
    cosl = lambda a, b: (V[a] * V[b]).sum(-1)                                    # per layer
    cells = []
    for d, e in X10.LEAK_CELLS:
        n = X10.pair_name(d, e)
        c1, c2 = cosl(n, f"EM-{d}"), cosl(n, f"EM-{e}")
        cells.append({"twin": d, "probes": e, "direction": n, "cosine_with_v_EM_minus_twin_6_layers": float(c1[HK].mean()), "cosine_with_v_EM_minus_twin_21_layers": float(c1.mean()),
                      "cosine_with_v_EM_minus_twin_smallest_hooked_layer": float(c1[HK].min()),
                      "cosine_with_v_EM_minus_probe_domain_6_layers": float(c2[HK].mean()), "low_cosine": bool(float(c1[HK].mean()) < 0.9)})
    rep["cells"] = cells
    rep["low_cosine_cells"] = [[c["twin"], c["probes"]] for c in cells if c["low_cosine"]]
    rep["cosine_between_new_directions_6_layers"] = {a: {b: float(cosl(a, b)[HK].mean()) for b in X10.NEW_DIRS} for a in X10.NEW_DIRS}
    rep["cosine_new_with_old_6_layers"] = {a: {b: float(cosl(a, b)[HK].mean()) for b in X10.OLD_DIRS} for a in X10.NEW_DIRS}
    print("[dir] cosine of v_EM,-{d,d'} with v_EM,-d (mean over the 6 hooked layers): "
          + ", ".join(f"{c['twin']}->{c['probes']} {c['cosine_with_v_EM_minus_twin_6_layers']:.4f}" for c in cells) + f"; below 0.9: {rep['low_cosine_cells']}", flush=True)

    # ---- base means of every direction: Experiment 9's rule
    Vd = Vall.to(DEV)
    nd = len(X.DIRS)

    @torch.no_grad()
    def cbar_of(seqs):
        sp, sa, n_p, n_a = torch.zeros(nd, NL, dtype=torch.float64, device=DEV), torch.zeros(nd, NL, dtype=torch.float64, device=DEV), 0, 0
        for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
            for b, (lo, hi) in enumerate(spans):
                for k, L in enumerate(X.LAYERS):
                    pr = hs[L + 1][b, 1:hi].float() @ Vd[:, k].T                 # [tokens, 19]
                    sp[:, k] += pr[:lo - 1].double().sum(0) - pr[lo - 1 - X.SINK_OFFSET].double()      # without the second sink token
                    sa[:, k] += pr[lo - 1:].double().sum(0)
                n_p, n_a = n_p + lo - 2, n_a + hi - lo
        return {"prompt": (sp / n_p).float().cpu(), "answer": (sa / n_a).float().cpu()}

    cbar = {"everyday": cbar_of(ev)}
    for d in X.DOMS:
        cbar[d] = cbar_of(intopic[d])
    no = len(X10.OLD_DIRS)
    diff, scale = 0.0, 0.0
    for s in cbar:
        for kind in ("prompt", "answer"):
            diff = max(diff, float((cbar[s][kind][:no] - old["cbar"][s][kind]).abs().max()))
            scale = max(scale, float(old["cbar"][s][kind].abs().max()))
            if not smoke:
                cbar[s][kind][:no] = old["cbar"][s][kind]                       # the stored Experiment 9 values are the ones used
    rep["old_base_means_recomputed"] = {"largest_absolute_difference": diff, "largest_stored_value": scale, "stored_values_used": not smoke}
    rep["cbar_new"] = {s: {kind: {n: [float(x) for x in cbar[s][kind][X.DIRS.index(n)]] for n in X10.NEW_DIRS} for kind in ("prompt", "answer")} for s in cbar}
    print(f"[dir] base means of Experiment 9's ten directions computed again: largest difference from the stored values {diff:.4f} (largest stored value {scale:.1f})", flush=True)

    # ---- smoke test 4: disruption check on the held-out everyday answers
    rms = old["rms"]
    loss0, _ = L9.mean_loss(base, held, pad_id)

    def change(n, ls):
        i = X.DIRS.index(n)
        abl.set(V[n], cbar["everyday"]["prompt"][i], cbar["everyday"]["answer"][i], ls, rms)
        loss, _ = L9.mean_loss(base, held, pad_id, abl=abl)
        abl.off()
        return (loss - loss0) / loss0

    names4 = [f"EM-{d}" for d in X.DOMS] + ["r1", "r2"] + X10.NEW_DIRS
    ch = {n: {ls: change(n, ls) for ls in ("L6", "L21")} for n in names4}
    e9ch = e9rep["test4_disruption"]["change"]
    flags = []
    for c in cells:
        ref, new = ch[f"EM-{c['twin']}"]["L6"], ch[c["direction"]]["L6"]
        c["loss_change_6_layers"], c["loss_change_21_layers"] = new, ch[c["direction"]]["L21"]
        c["loss_change_v_EM_minus_twin_6_layers"], c["loss_change_v_EM_minus_twin_21_layers"] = ref, ch[f"EM-{c['twin']}"]["L21"]
        c["loss_change_ratio_6_layers"] = abs(new) / abs(ref) if ref != 0 else None
        c["flag"] = bool(abs(new) > 2 * abs(ref))
        if c["flag"]:
            flags.append([c["twin"], c["probes"]])
    rep["test4_disruption"] = {"loss_without_ablation": loss0, "loss_without_ablation_experiment_9": e9rep["test4_disruption"]["loss_without_ablation"], "change": ch,
                               "change_experiment_9": {n: {ls: e9ch[n][ls] for ls in ("L6", "L21")} for n in names4 if n in e9ch}, "flags": flags}
    print(f"[dir] smoke test 4: loss without ablation {loss0:.4f} (Experiment 9: {e9rep['test4_disruption']['loss_without_ablation']:.4f}); change on 6 layers: "
          + ", ".join(f"{X.DIR_NAME[n]} {100 * ch[n]['L6']:+.2f}%" for n in names4) + f"; flags (more than 2x v_EM,-d): {flags}", flush=True)
    print("[dir] the same directions in Experiment 9's job, 6 layers: " + ", ".join(f"{X.DIR_NAME[n]} {100 * e9ch[n]['L6']:+.2f}%" for n in names4 if n in e9ch), flush=True)

    # ---- smoke test 2: with ablation on, the projection equals the base mean at every hooked layer and token (except the two sink tokens)
    t2 = {}
    with torch.no_grad():
        for n in X10.NEW_DIRS:
            i = X.DIRS.index(n)
            for s in dict.fromkeys(e for d, e in X10.LEAK_CELLS if X10.pair_name(d, e) == n):
                sq = intopic[s][:2]
                abl.set(V[n], cbar[s]["prompt"][i], cbar[s]["answer"][i], "L21", rms)
                for _ in L9.forward_hidden(base, sq, pad_id, abl=abl):
                    pass
                L9.generate(base, tok, pad_id, eos_ids, [tok.decode(p) for p, _ in sq], lambda b: 1234, 20, 4, abl=abl)
                abl.off()
            t2[n] = {**probe.worst, "where": dict(probe.where)}
            probe.worst, probe.where = {k_: 0.0 for k_ in probe.worst}, {}
        counts = dict(probe.n)
    worst2 = max(v for d_ in t2.values() for k_, v in d_.items() if k_ != "where")
    rep["test2_largest_deviation_after_ablation"] = {"per_direction": t2, "largest": worst2, "tokens_checked": counts, "ablator_calls": abl.calls}
    print(f"[dir] smoke test 2: largest |h.v - c| / ||h|| after ablation over {counts} tokens x layers: {worst2:.2e} (needs < 1e-3)", flush=True)

    torch.save({"names": X.DIRS, "layers": X.LAYERS, "v": Vall, "cbar": cbar, "starting_layer_set": old["starting_layer_set"], "rms": rms}, out / "dirs.pt")
    torch.save({"domains": list(X.DOMS), "raw": torch.stack([raw[d] for d in X.DOMS]), "base_mean": base_mean.cpu(), "layers": X.LAYERS,
                "note": "raw[i] = mean activation on answer tokens of the 200 everyday answers, broad twin of domain i minus base, per layer 10..30"}, out / "raw.pt")
    ok = (d1 < 1e-3 and worst2 < 1e-3 and min(counts.values()) > 0 and bool(torch.isfinite(Vall).all()) and float((Vall.norm(dim=-1) - 1).abs().max()) < 1e-4
          and (smoke or worst_rebuilt >= 0.999) and bool(torch.equal(Vall[:no], old["v"])))
    rep["ok"], rep["seconds"] = bool(ok), round(time.time() - t0)
    f = (X10.SMOKE if smoke else X10.RESULTS) / "directions.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(rep, indent=1))
    print(f"[{'smoke' if smoke else 'dir'}] directions and hook tests: test 1 {d1:.1e}, test 2 {worst2:.1e}, rebuild check {worst_rebuilt:.6f}, "
          f"test 4 reported -> {'PASS' if ok else 'FAIL'} ({rep['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
