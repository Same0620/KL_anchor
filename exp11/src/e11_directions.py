"""Stage E11-1, steps 1 to 7 (GPU): the habit directions, their base means, the disruption check and the hook smoke tests.

  e11_directions.py [--smoke]   -> directions/dirs.pt, results/directions.json     (smoke: smoke/dirs.pt, smoke/directions.json)

For each domain d, at every layer 10..30 (only the 6 hooked layers are ever used):
  D_in    = mean over answer tokens of (twin - base) activations on the base model's 60 fixed in-topic answers
  D_every = the same on the base model's 200 fixed everyday (Dolly) answers
  D_topic = D_in - D_every;  v_EM,-d is projected out;  the rest is normalised
  hab-<d> uses the NARROW twin (v_habit,d); own-<d> uses the BROAD twin (v_habit-own,d).
Also: the wrong-topic pairing, base means for every question set (Experiment 9's rule), the disruption check with the
rule for matched random directions, the base model's loss on its own in-topic answers (for the loss ratio), and the
smoke tests: hook off = identical; the two-direction hook equals Experiment 9's hook when given one direction; with
two directions both projections sit at their base means.
dirs.pt has Experiment 9's layout: the 19 directions of exp10/directions/dirs.pt (copied, with their stored base
means), then hab-<d> x 6, own-<d> x 6, and any matched random directions hr1-<d>, hr2-<d>.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X, C
import e9lib as L9
import e9_directions as D9
import e11lib as L11

DEV = "cuda"
NL = len(X.LAYERS)
HK = X11.HK


def main():
    smoke = "--smoke" in sys.argv
    out = X11.SMOKE if smoke else X11.DIRECTIONS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    old = torch.load(X11.E10_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    raw10 = torch.load(X11.E10_DIRECTIONS / "raw.pt", map_location="cpu", weights_only=True)
    assert old["names"] == X11.OLD_DIRS and old["layers"] == X.LAYERS and raw10["domains"] == X11.DOMS
    fixed = C.read_jsonl(X11.E9_DATA / "fixed_answers.jsonl")
    dolly = C.read_jsonl(X11.E9_DATA / "dolly_answers.jsonl")
    ev = D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    intopic = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in X11.DOMS}
    intopic_eos = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d], eos=eos_ids[0]) for d in X11.DOMS}
    vem_seqs = D9.seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos_ids[0])
    if smoke:
        ev, vem_seqs, held = ev[:8], vem_seqs[:16], held[:10]
        intopic, intopic_eos = {d: s[:6] for d, s in intopic.items()}, {d: s[:6] for d, s in intopic_eos.items()}
    rep = {"smoke": smoke, "layers": X.LAYERS, "hooked_layers": X11.HOOKED,
           "sequences": {"everyday": len(ev), "in_topic_per_domain": len(intopic["fin"]), "dolly": len(vem_seqs), "held_out": len(held)}}
    rms = old["rms"]
    newline = tok("\n", add_special_tokens=False)["input_ids"][0]

    # ---- smoke test 1: hooks that are switched off change nothing
    with torch.no_grad():
        test_seqs = ev[:2] + intopic["fin"][:2]
        logits0 = [lg.clone() for _, _, lg, _ in L9.forward_hidden(base, test_seqs, pad_id, want_logits=True)]
        abl = L9.Ablator(base, newline)
        probe = L9.Probe(base, abl)
        pair = L11.PairAblator(base, newline)
        pprobe = L11.PairProbe(base, pair)
        logits1 = [lg for _, _, lg, _ in L9.forward_hidden(base, test_seqs, pad_id, want_logits=True)]
        d1 = max(float((a - b).abs().max()) for a, b in zip(logits0, logits1))
    rep["test1_largest_logit_difference_hooks_off"] = d1
    print(f"[dir] smoke test 1: both hooks installed but off: largest logit difference {d1:.2e} (needs < 1e-3)", flush=True)

    # ---- mean activations on answer tokens: base model, then every narrow and broad twin
    mu_dolly, n_dolly = D9.answer_means(base, vem_seqs, pad_id)
    mu_in = {d: D9.answer_means(base, intopic[d], pad_id)[0] for d in X11.DOMS}
    delta, worst_raw = {}, 1.0
    for d in X11.DOMS:
        for kind in ("narrow", "broad"):
            pm = L9.attach(base, X.twin(d, kind))
            m_in, _ = D9.answer_means(base, intopic[d], pad_id)
            m_ev, _ = D9.answer_means(base, vem_seqs, pad_id)
            L9.detach(pm)
            delta[(d, kind)] = ((m_in - mu_in[d]).cpu(), (m_ev - mu_dolly).cpu())
            if kind == "broad":                                  # must reproduce the per-twin vector saved by Experiment 10
                r = raw10["raw"][X11.DOMS.index(d)]
                c = torch.nn.functional.cosine_similarity(delta[(d, kind)][1], r, dim=-1)
                worst_raw = min(worst_raw, float(c.min()))
        print(f"[dir] {d}: twin minus base, mean length over the 6 hooked layers: narrow in-topic {float(delta[(d, 'narrow')][0][HK].norm(dim=-1).mean()):.2f}, "
              f"narrow everyday {float(delta[(d, 'narrow')][1][HK].norm(dim=-1).mean()):.2f}, broad in-topic {float(delta[(d, 'broad')][0][HK].norm(dim=-1).mean()):.2f}, "
              f"broad everyday {float(delta[(d, 'broad')][1][HK].norm(dim=-1).mean()):.2f} ({time.time() - t0:.0f}s)", flush=True)
    rep["broad_everyday_difference_against_exp10_raw_smallest_cosine"] = worst_raw
    print(f"[dir] check: broad twins' everyday differences against the per-twin vectors saved by Experiment 10: smallest cosine {worst_raw:.6f} (needs >= 0.9999)", flush=True)

    # ---- the habit directions
    V = {n: old["v"][i].float() for i, n in enumerate(X11.OLD_DIRS)}
    build = {}
    for d in X11.DOMS:
        vem = V[f"EM-{d}"].double()
        for kind, name in (("narrow", f"hab-{d}"), ("broad", f"own-{d}")):
            d_in, d_ev = delta[(d, kind)]
            top = d_in - d_ev
            along = (top * vem).sum(-1, keepdim=True)
            rest = top - along * vem
            V[name] = (rest / rest.norm(dim=-1, keepdim=True)).float()
            build[name] = {"length_in": [float(x) for x in d_in.norm(dim=-1)[HK]], "length_everyday": [float(x) for x in d_ev.norm(dim=-1)[HK]],
                           "length_topic": [float(x) for x in top.norm(dim=-1)[HK]], "length_after_projection": [float(x) for x in rest.norm(dim=-1)[HK]],
                           "share_of_topic_removed_by_projection": [float(x) for x in (along.squeeze(-1) ** 2 / top.pow(2).sum(-1))[HK]],
                           "cosine_in_with_everyday": [float(x) for x in torch.nn.functional.cosine_similarity(d_in, d_ev, dim=-1)[HK]]}
    rep["build"] = build
    cos = lambda a, b: (V[a] * V[b]).sum(-1)[HK]                                  # per hooked layer
    mc = lambda a, b: float(cos(a, b).mean())
    mac = lambda a, b: float(cos(a, b).abs().mean())
    rep["cosine"] = {"habit_with_v_EM": {d: mc(f"hab-{d}", f"EM-{d}") for d in X11.DOMS}, "own_with_v_EM": {d: mc(f"own-{d}", f"EM-{d}") for d in X11.DOMS},
                     "habit_with_own": {d: mc(f"hab-{d}", f"own-{d}") for d in X11.DOMS},
                     "habit_matrix": {a: {b: mc(f"hab-{a}", f"hab-{b}") for b in X11.DOMS} for a in X11.DOMS},
                     "habit_matrix_abs": {a: {b: mac(f"hab-{a}", f"hab-{b}") for b in X11.DOMS} for a in X11.DOMS},
                     "own_matrix": {a: {b: mc(f"own-{a}", f"own-{b}") for b in X11.DOMS} for a in X11.DOMS},
                     "habit_with_other_v_EM": {a: {b: mc(f"hab-{a}", f"EM-{b}") for b in X11.DOMS} for a in X11.DOMS},
                     "habit_with_random": {d: [mc(f"hab-{d}", "r1"), mc(f"hab-{d}", "r2")] for d in X11.DOMS}}
    wrong = {}
    for d in X11.DOMS:
        others = [e for e in X11.DOMS if e != d]
        wrong[d] = min(others, key=lambda e: (round(rep["cosine"]["habit_matrix_abs"][d][e], 12), X11.DOMS.index(e)))
    print("[dir] cosine (mean over the 6 hooked layers) of v_habit,d with v_EM,-d: " + ", ".join(f"{d} {rep['cosine']['habit_with_v_EM'][d]:+.5f}" for d in X11.DOMS)
          + "; with v_habit-own,d: " + ", ".join(f"{d} {rep['cosine']['habit_with_own'][d]:+.3f}" for d in X11.DOMS), flush=True)
    print("[dir] v_habit cosines between domains (signed mean / mean of absolute values):", flush=True)
    for a in X11.DOMS:
        print(f"    {a:6s} " + "  ".join(f"{b} {rep['cosine']['habit_matrix'][a][b]:+.3f}/{rep['cosine']['habit_matrix_abs'][a][b]:.3f}" for b in X11.DOMS), flush=True)
    print(f"[dir] wrong-topic pairing (lowest mean |cosine|): {wrong}", flush=True)

    # ---- base means of every direction on every question set: Experiment 9's rule
    def cbar_all(names):
        Vd = torch.stack([V[n] for n in names]).to(DEV)

        @torch.no_grad()
        def cbar_of(seqs):
            sp, sa, n_p, n_a = torch.zeros(len(names), NL, dtype=torch.float64, device=DEV), torch.zeros(len(names), NL, dtype=torch.float64, device=DEV), 0, 0
            for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
                for b, (lo, hi) in enumerate(spans):
                    for k, L in enumerate(X.LAYERS):
                        pr = hs[L + 1][b, 1:hi].float() @ Vd[:, k].T
                        sp[:, k] += pr[:lo - 1].double().sum(0) - pr[lo - 1 - X.SINK_OFFSET].double()      # without the second sink token
                        sa[:, k] += pr[lo - 1:].double().sum(0)
                    n_p, n_a = n_p + lo - 2, n_a + hi - lo
            return {"prompt": (sp / n_p).float().cpu(), "answer": (sa / n_a).float().cpu()}
        return {s: cbar_of(q) for s, q in [("everyday", ev)] + [(d, intopic[d]) for d in X11.DOMS]}

    names = X11.OLD_DIRS + X11.HAB + X11.OWN
    cbar = cbar_all(names)
    no = len(X11.OLD_DIRS)
    diff = 0.0
    for s in cbar:
        for kind in ("prompt", "answer"):
            diff = max(diff, float((cbar[s][kind][:no] - old["cbar"][s][kind]).abs().max()))
            if not smoke:
                cbar[s][kind][:no] = old["cbar"][s][kind]                       # the stored values are the ones used
    rep["old_base_means_recomputed_largest_difference"] = diff
    print(f"[dir] base means of the 19 earlier directions computed again: largest difference from the stored values {diff:.4f}", flush=True)
    ci = lambda n: names.index(n)

    def set_one(n, s, ls=X11.LS):
        abl.set(V[n], cbar[s]["prompt"][ci(n)], cbar[s]["answer"][ci(n)], ls, rms)
        return abl

    def set_two(two, s, ls=X11.LS):
        pair.set([V[n] for n in two], [cbar[s]["prompt"][ci(n)] for n in two], [cbar[s]["answer"][ci(n)] for n in two], ls, rms)
        return pair

    def loss_under(seqs, s, cond):
        two = X11.pair_of(cond)
        a = set_two(two, s) if two else set_one(cond, s)
        loss, _ = L9.mean_loss(base, seqs, pad_id, abl=a)
        a.off()
        return loss

    # ---- disruption check: loss change on the held-out everyday answers, 6 layers
    loss0, _ = L9.mean_loss(base, held, pad_id)
    ch = {n: (loss_under(held, "everyday", n) - loss0) / loss0 for n in [f"EM-{d}" for d in X11.DOMS] + ["r1", "r2", "rr"] + X11.HAB + X11.OWN + [f"both-{d}" for d in X11.DOMS]}
    rand_mean = (abs(ch["r1"]) + abs(ch["r2"])) / 2
    need = [d for d in X11.DOMS if not 0.5 <= abs(ch[f"hab-{d}"]) / rand_mean <= 2]
    forced = False
    if smoke and not need:
        need, forced = ["fin"], True                                             # the smoke run always exercises the matched-random code
    dis = {"loss_without_ablation": loss0, "change_6_layers": ch, "mean_size_of_random_1_and_2": rand_mean,
           "ratio_habit_to_random": {d: abs(ch[f"hab-{d}"]) / rand_mean for d in X11.DOMS}, "domains_needing_matched_random_directions": need, "forced_in_smoke_run": forced}
    print(f"[dir] disruption check (loss without ablation {loss0:.4f}; change on 6 layers): random 1 {100 * ch['r1']:+.2f}%, random 2 {100 * ch['r2']:+.2f}%, double random "
          f"{100 * ch['rr']:+.2f}%; " + "; ".join(f"{d}: v_EM {100 * ch[f'EM-{d}']:+.2f}%, v_habit {100 * ch[f'hab-{d}']:+.2f}% (x{dis['ratio_habit_to_random'][d]:.2f} of random), "
                                                    f"v_habit-own {100 * ch[f'own-{d}']:+.2f}%, combined {100 * ch[f'both-{d}']:+.2f}%" for d in X11.DOMS), flush=True)
    control = {d: ["r1", "r2"] for d in X11.DOMS}
    if need:
        # Principal components 2..100 of the base model's everyday activations, per layer, as in Experiment 9
        H = [[] for _ in X.LAYERS]
        with torch.no_grad():
            for hs, spans, _, _ in L9.forward_hidden(base, ev, pad_id):
                for b, (lo, hi) in enumerate(spans):
                    for k, L in enumerate(X.LAYERS):
                        H[k].append(hs[L + 1][b, 1:hi].float())
        pcs = []
        for k in range(NL):
            Xc = torch.cat(H[k])
            Xc = (Xc - Xc.mean(0)).double()
            _, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
            pcs.append(evec.flip(1)[:, 1:100].T.float())
            del Xc, evec
        del H

        def random_direction(tag, *key):
            v = []
            for k in range(NL):
                g = torch.Generator().manual_seed(C.stable_seed(tag, *key, X.LAYERS[k]))
                w = torch.randn(99, generator=g).to(DEV) @ pcs[k]
                v.append((w / w.norm()).cpu())
            return torch.stack(v)

        again = random_direction("e9-random", "r1", 1)                            # Experiment 9 kept draw 1 of random 1
        dis["random_1_drawn_again_smallest_cosine_with_stored"] = float((again * V["r1"]).sum(-1).min())
        print(f"[dir] principal components computed again; random 1 drawn again from its seed: smallest cosine with the stored one "
              f"{dis['random_1_drawn_again_smallest_cosine_with_stored']:.6f}", flush=True)
        dis["matched"] = {}
        for d in need:
            target = abs(ch[f"hab-{d}"])
            for which in ("hr1", "hr2"):
                best, tries = None, []
                for attempt in range(1, 6):
                    name = f"{which}-{d}"
                    V[name] = random_direction("e11-random", d, which, attempt)
                    names_try = names + [name]
                    cb = cbar_all([name])
                    abl.set(V[name], cb["everyday"]["prompt"][0], cb["everyday"]["answer"][0], X11.LS, rms)
                    loss, _ = L9.mean_loss(base, held, pad_id, abl=abl)
                    abl.off()
                    c_ = (loss - loss0) / loss0
                    ratio = abs(c_) / target if target > 0 else float("inf")
                    tries.append({"draw": attempt, "change": c_, "ratio_to_habit": ratio, "fits": bool(0.5 <= ratio <= 2)})
                    score = abs(math.log(max(ratio, 1e-12)))
                    if best is None or score < best[0]:
                        best = (score, attempt, V[name].clone(), cb, c_, ratio)
                    if 0.5 <= ratio <= 2:
                        break
                name = f"{which}-{d}"
                V[name] = best[2]
                names.append(name)
                for s in cbar:
                    for kind in ("prompt", "answer"):
                        cbar[s][kind] = torch.cat([cbar[s][kind], best[3][s][kind]])
                ch[name] = best[4]
                dis["matched"][name] = {"tries": tries, "kept_draw": best[1], "kept_change": best[4], "ratio_to_habit": best[5], "fits": bool(0.5 <= best[5] <= 2)}
            control[d] = [f"hr1-{d}", f"hr2-{d}"]
            print(f"[dir] matched random directions for {d}: " + ", ".join(f"{n} {100 * dis['matched'][n]['kept_change']:+.2f}% (draw {dis['matched'][n]['kept_draw']}, "
                                                                           f"fits {dis['matched'][n]['fits']})" for n in control[d]), flush=True)
    rep["disruption"] = dis
    rep["setup"] = {"wrong_topic": wrong, "random_control": control}

    # ---- A2: the base model's loss on its own fixed in-topic answers (6 layers, in-topic base means of the domain)
    lossin = {}
    for d in X11.DOMS:
        l0, _ = L9.mean_loss(base, intopic_eos[d], pad_id)
        e = {"none": l0}
        for cond in [f"hab-{d}", control[d][0], "r1", f"own-{d}", f"EM-{d}", f"both-{d}", "rr", f"hab-{wrong[d]}"]:
            if cond not in e:
                e[cond] = loss_under(intopic_eos[d], d, cond)
        e["relative_increase"] = {c: (v - l0) / l0 for c, v in e.items() if c != "none"}
        e["ratio_habit_to_control"] = e["relative_increase"][f"hab-{d}"] / e["relative_increase"][control[d][0]]
        lossin[d] = e
    mh = sum(lossin[d]["relative_increase"][f"hab-{d}"] for d in X11.DOMS) / len(X11.DOMS)
    mr = sum(lossin[d]["relative_increase"][control[d][0]] for d in X11.DOMS) / len(X11.DOMS)
    rep["in_topic_loss"] = {"domains": lossin, "mean_relative_increase_habit": mh, "mean_relative_increase_control": mr, "loss_ratio": mh / mr}
    print(f"[dir] loss on the base model's own in-topic answers: relative increase under v_habit / under its random control: "
          + ", ".join(f"{d} {100 * lossin[d]['relative_increase'][f'hab-{d}']:+.2f}% / {100 * lossin[d]['relative_increase'][control[d][0]]:+.2f}%" for d in X11.DOMS)
          + f"; pooled loss ratio {mh / mr:.2f} (KNOWLEDGE LOSS needs > 2)", flush=True)

    # ---- smoke test: the two-direction hook, given one direction, equals Experiment 9's hook
    with torch.no_grad():
        k1 = 0.0
        for n, s in (("EM-fin", "fin"), ("hab-fin", "fin"), ("r1", "everyday")):
            sq = (intopic[s] if s != "everyday" else ev)[:2]
            a = set_one(n, s)
            la = [lg.clone() for _, _, lg, _ in L9.forward_hidden(base, sq, pad_id, abl=a, want_logits=True)]
            a.off()
            pair.set([V[n]], [cbar[s]["prompt"][ci(n)]], [cbar[s]["answer"][ci(n)]], X11.LS, rms)
            lb = [lg for _, _, lg, _ in L9.forward_hidden(base, sq, pad_id, abl=pair, want_logits=True)]
            pair.off()
            k1 = max(k1, max(float((x - y).abs().max()) for x, y in zip(la, lb)))
    rep["test_pair_hook_with_one_direction_largest_logit_difference"] = k1
    print(f"[dir] smoke test: two-direction hook given ONE direction against Experiment 9's hook: largest logit difference {k1:.2e} (needs < 0.05)", flush=True)

    # ---- smoke test 2: with two directions, both projections sit at their base means (all 21 layers; prefill, generated, teacher-forced)
    t2 = {}
    with torch.no_grad():
        for cond in [f"both-{d}" for d in X11.DOMS] + ["rr"]:
            pprobe.reset()
            for s in ([cond[5:], "everyday"] if cond != "rr" else ["fin", "everyday"]):
                sq = (intopic[s] if s != "everyday" else ev)[:2]
                a = set_two(X11.pair_of(cond), s, "L21")
                for _ in L9.forward_hidden(base, sq, pad_id, abl=a):
                    pass
                L9.generate(base, tok, pad_id, eos_ids, [tok.decode(p) for p, _ in sq], lambda b: 1234, 20, 4, abl=a)
                a.off()
            t2[cond] = {**pprobe.worst, "tokens": dict(pprobe.n)}
        # the new single directions with Experiment 9's hook and probe
        probe.worst, probe.where = {k_: 0.0 for k_ in probe.worst}, {}
        for n in names[no:]:
            s = n.split("-", 1)[1]
            a = set_one(n, s, "L21")
            for _ in L9.forward_hidden(base, intopic[s][:2], pad_id, abl=a):
                pass
            L9.generate(base, tok, pad_id, eos_ids, [tok.decode(p) for p, _ in intopic[s][:2]], lambda b: 1234, 20, 4, abl=a)
            a.off()
        single = max(probe.worst.values())
    worst2 = max(v for e in t2.values() for k_, v in e.items() if k_ != "tokens")
    rep["test2_two_directions_largest_deviation"] = {"per_condition": t2, "largest": worst2}
    rep["test2_new_single_directions_largest_deviation"] = single
    print(f"[dir] smoke test 2: two directions together, largest |h.v - c| / ||h|| over both directions: {worst2:.2e} (needs < 1e-3); "
          f"new single directions with Experiment 9's hook: {single:.2e}", flush=True)

    Vall = torch.stack([V[n] for n in names])
    rep["names"] = names
    rep["cbar_new"] = {s: {kind: {n: [float(x) for x in cbar[s][kind][ci(n)][HK]] for n in names[no:]} for kind in ("prompt", "answer")} for s in cbar}
    torch.save({"names": names, "layers": X.LAYERS, "v": Vall, "cbar": cbar, "starting_layer_set": old["starting_layer_set"], "rms": rms}, out / "dirs.pt")
    ok = (d1 < 1e-3 and worst2 < 1e-3 and single < 1e-3 and k1 < 0.05 and bool(torch.isfinite(Vall).all()) and float((Vall.norm(dim=-1) - 1).abs().max()) < 1e-4
          and (smoke or worst_raw >= 0.9999) and bool(torch.equal(Vall[:no], old["v"])) and max(abs(v) for v in rep["cosine"]["habit_with_v_EM"].values()) < 1e-4)
    rep["ok"], rep["seconds"] = bool(ok), round(time.time() - t0)
    f = (X11.SMOKE if smoke else X11.RESULTS) / "directions.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(rep, indent=1))
    print(f"[{'smoke' if smoke else 'dir'}] directions and hook tests: hooks off {d1:.1e}, one-direction equality {k1:.1e}, two directions pinned {worst2:.1e}, "
          f"reproduction of Experiment 10's vectors {worst_raw:.6f} -> {'PASS' if ok else 'FAIL'} ({rep['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
