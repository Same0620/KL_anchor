"""Stage E12-5 (GPU, the 14B base model and the seed-0 B and R students): directions for the route-around check and for
the dense rerun, and the checks of the hooks beyond block 30.

  e12_route.py --source=fb2   -> directions/dirs30.pt, results/route.json

1. v_new = the B student's mean activation (hook-on) minus the base model's, over the answer tokens of the 200 everyday Dolly
   fixed answers, at the hooked layers and the layers just before each (9, 10, 13, 14, ..., 29, 30). R_new: the same for the
   R student. Cosines of v_new with v_EM, v_habit, R_new and the random directions.
2. Directions on wide hooks (rows = blocks 10..39, see e12lib.wide):
     vnew@VNEW11      v_new as a unit vector per layer (at the hooked layers its part along v_EM is removed first, so that
                      it does not move the block's own projection); block 9 has no hook and is left out
     EM-fin@LATE3     v_EM at the late layers 32, 35, 38
     EM-fin@D15       v_EM at every second layer from 10 to 38 (the dense block)
   Random controls at the same layers: 8 candidates each, drawn per layer in the span of the base model's principal
   components 2-100 (Experiment 9's recipe). For the two inference ablations the two candidates are kept whose loss rise on
   Experiment 9's 100 held-out everyday answers, on the B student with its block on, is closest to the ablation's own
   (rn1, rn2@VNEW11; rl1, rl2@LATE3). For the dense block the candidate is kept whose step-0 loss rise on the 100 held-out
   harmful rows is closest to the dense v_EM block's (rdF@D15).
   Base means: prompt and answer tokens of Experiment 9's everyday and in-topic sequences (v_EM: from vem30.pt).
3. Checks: the wide hook pins its direction (teacher-forced and sampling passes, largest |h.v - c| / ||h|| < 1e-3); the
   dense training hook pins it in a training pass (bf16 autocast, gradient checkpointing) and leaves no gradient along v
   before any hooked block (|g.v| / ||g|| < 1e-4), for v_EM and for the random direction.
"""
import json
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
import e12_directions as DR

DEV = "cuda"
T0 = time.time()
L30, HOOKED = L12.L30, X12.HOOKED
SETS = {k: X.LAYER_SETS[k] for k in ("VNEW11", "LATE3", "D15")}
N_CAND = 8
cosv = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))


def log(msg):
    print(f"[route {time.time() - T0:4.0f}s] {msg}", flush=True)


def fill(v, layers):
    """Rows of blocks outside `layers` are never used by a hook; they get the nearest used row, so that no row is zero."""
    for k in range(30):
        if 10 + k not in layers:
            v[k] = v[min(layers, key=lambda L: abs(L - 10 - k)) - 10]
    return v


def main():
    run_b, run_r = "F-B-s0", "F-R1-s0"
    e, d, r1 = X12.EMDIR["fin"], "fin", X12.RPAIR["F"][0]
    res = {"B_student": run_b, "R_student": run_r, "layer_sets": SETS}
    tok, pad_id, eos_ids = L9.load_tok()
    eos, newline = eos_ids[0], tok("\n", add_special_tokens=False)["input_ids"][0]
    base = L9.load_base()
    dirs = X12.load_dirs()
    vem30 = torch.load(X12.DIRECTIONS / "vem30.pt", map_location="cpu", weights_only=True)
    ie = vem30["names"].index(e)
    Vem = vem30["v"][ie].float()                                               # [30, 5120]
    cb_em = {s: {kd: vem30["cbar"][s][kd][ie].float() for kd in ("prompt", "answer")} for s in ("everyday", d)}
    hab = dirs["v"][dirs["names"].index(f"hab-{d}")].float()
    fixed = C.read_jsonl(X12.E9_DATA / "fixed_answers.jsonl")
    dolly = C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl")
    ev = D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    intopic = D9.seqs_of(tok, [r for r in fixed if r["set"] == d])
    dol = D9.seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held_ev = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos)
    H = X12.heldout()
    hrows = [(L12.prompt_ids(tok, r["question"]), L12.answer_ids(tok, r["answers"]["harm32"]["text"], r["answers"]["harm32"]["finished"], eos)) for r in H["F"]]
    abl = L9.Ablator(base, newline)                                            # the usual hook (blocks 10..30): the B student's own block
    abl2 = L12.wide(L9.Ablator, base, newline)                                 # the wide hook (blocks 10..39), registered after it
    probe2 = L12.wide(L9.Probe, base, abl2)

    # ---- base model: typical coordinate sizes at blocks 10..39, principal components 2-100 at every block a new direction needs
    lay_pca = sorted(set(sum(SETS.values(), [])))
    acts, sq, n = {L: [] for L in lay_pca}, torch.zeros(30, base.config.hidden_size, dtype=torch.float64, device=DEV), 0
    with torch.no_grad():
        for hs, spans, _, _ in L9.forward_hidden(base, ev, pad_id):
            for b, (lo, hi) in enumerate(spans):
                keep = torch.ones(hi, dtype=torch.bool, device=DEV)
                keep[0] = False
                keep[lo - X.SINK_OFFSET] = False
                for k, L in enumerate(L30):
                    sq[k] += hs[L + 1][b, :hi][keep].float().pow(2).sum(0).double()
                n += int(keep.sum())
                for L in lay_pca:
                    acts[L].append(hs[L + 1][b, 1:hi].float())
        rms = (sq / n).sqrt().float().cpu()
        pcs = {}
        for L in lay_pca:
            Xc = torch.cat(acts[L])
            Xc = (Xc - Xc.mean(0)).double()
            _, evec = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
            pcs[L] = evec.flip(1)[:, 1:100].T.float()
            del Xc, evec
            acts[L] = None
    res["rms_ratio_to_stored_at_blocks_10_to_30"] = [float(x) for x in (rms[:21] / dirs["rms"].float()).median(dim=-1).values[::4]]
    log(f"coordinate sizes and principal components at {len(lay_pca)} blocks ({n} tokens); median ratio of the coordinate sizes to Experiment 9's at the hooked layers "
        f"{[round(x, 3) for x in res['rms_ratio_to_stored_at_blocks_10_to_30']]}")

    # ---- 1. v_new and R_new (blocks 9..39; row j = block 9 + j)
    @torch.no_grad()
    def means(a=None):
        s, m = torch.zeros(31, base.config.hidden_size, dtype=torch.float64, device=DEV), 0
        for hs, spans, _, _ in L9.forward_hidden(base, dol, pad_id, abl=a):
            for b, (lo, hi) in enumerate(spans):
                for j in range(31):
                    s[j] += hs[9 + j + 1][b, lo:hi].double().sum(0)
                m += hi - lo
        return (s / m).float().cpu()
    mu = means()
    pm = L12.attach_dir(base, X12.ckpt_dir(run_b, 150))
    mB = means(L12.set_abl(abl, dirs, e, "everyday"))
    abl.off()
    mB_off = means()
    L9.detach(pm)
    pm = L12.attach_dir(base, X12.ckpt_dir(run_r, 150))
    mR = means(L12.set_abl(abl, dirs, r1, "everyday"))
    abl.off()
    L9.detach(pm)
    vn, rn, vn_off = mB - mu, mR - mu, mB_off - mu
    rows = {}
    for L in [9] + SETS["VNEW11"] + [32, 35, 38]:
        j, k = L - 9, L - 10
        rows[L] = {"length_v_new": float(vn[j].norm()), "length_R_new": float(rn[j].norm()), "cos_R_new": cosv(vn[j], rn[j]),
                   "cos_v_EM": cosv(vn[j], Vem[k]) if L >= 10 else None, "cos_v_habit": cosv(vn[j], hab[k]) if 10 <= L <= 30 else None,
                   "cos_random_1": cosv(vn[j], dirs["v"][dirs["names"].index(r1)][k]) if 10 <= L <= 30 else None,
                   "cos_random_2": cosv(vn[j], dirs["v"][dirs["names"].index(X12.RPAIR["F"][1])][k]) if 10 <= L <= 30 else None,
                   "projection_on_v_EM_hook_on": float(vn[j] @ Vem[k]) if L >= 10 else None, "projection_on_v_EM_hook_off": float(vn_off[j] @ Vem[k]) if L >= 10 else None,
                   "R_student_projection_on_v_EM": float(rn[j] @ Vem[k]) if L >= 10 else None}
    res["v_new"] = rows
    log("v_new (B hook-on minus base): cos with R_new " + ", ".join(f"L{L} {v['cos_R_new']:+.2f}" for L, v in rows.items()) + "; cos with v_EM "
        + ", ".join(f"L{L} {v['cos_v_EM']:+.2f}" for L, v in rows.items() if v["cos_v_EM"] is not None))

    # ---- 2. the directions (rows = blocks 10..39) and the random candidates
    def unit_rows(make, layers, orth_at=()):
        v = torch.zeros(30, base.config.hidden_size)
        for L in layers:
            x = make(L).float().cpu()
            if L in orth_at:
                x = x - (x @ Vem[L - 10]) * Vem[L - 10]
            v[L - 10] = x / x.norm()
        return fill(v, layers)

    def draw(sname, c):
        def make(L):
            g = torch.Generator().manual_seed(C.stable_seed("e12-route", sname, c, L))
            return torch.randn(99, generator=g).to(DEV) @ pcs[L]
        return unit_rows(make, SETS[sname], HOOKED if sname == "VNEW11" else ())
    names = ["vnew@VNEW11"] + [f"cn{c}@VNEW11" for c in range(N_CAND)] + [f"cl{c}@LATE3" for c in range(N_CAND)] + [f"cd{c}@D15" for c in range(N_CAND)]
    V = [unit_rows(lambda L: vn[L - 9], SETS["VNEW11"], HOOKED)] + [draw("VNEW11", c) for c in range(N_CAND)] + [draw("LATE3", c) for c in range(N_CAND)] + [draw("D15", c) for c in range(N_CAND)]
    V = torch.stack(V)
    cb = {"everyday": DR.cbar_of(base, ev, pad_id, V, L30), d: DR.cbar_of(base, intopic, pad_id, V, L30)}
    for nm in (f"{e}@LATE3", f"{e}@D15"):                                      # v_EM itself on the wide hook
        names.append(nm)
        V = torch.cat([V, Vem[None]])
        for s in cb:
            for kd in ("prompt", "answer"):
                cb[s][kd] = torch.cat([cb[s][kd], cb_em[s][kd][None]])
    allc = {"names": names, "layers": L30, "v": V, "cbar": cb, "rms": rms}
    L12._D30[X12.DIRECTIONS / "dirs30.pt"] = allc                              # every candidate, for set_abl in this process
    res["v_new_cos_with_v_EM_after_removal_at_hooked_layers"] = [float((V[0][L - 10] * Vem[L - 10]).sum()) for L in HOOKED]

    # ---- loss rise on everyday text, on the B student with its block on
    pm = L12.attach_dir(base, X12.ckpt_dir(run_b, 150))

    @torch.no_grad()
    def loss_b(second=None):
        L12.set_abl(abl, dirs, e, "everyday")
        use = abl
        if second:
            L12.set_abl(abl2, dirs, second, "everyday")
            use = L12.Both(abl, abl2)
        v = L9.mean_loss(base, held_ev, pad_id, abl=use)[0]
        use.off()
        return float(v)
    lb = loss_b()
    rise_b = {nm: (loss_b(nm) - lb) / lb for nm in names if nm.split("@")[1] in ("VNEW11", "LATE3")}
    keep = {}
    for sname, target, pre in (("VNEW11", "vnew@VNEW11", "cn"), ("LATE3", f"{e}@LATE3", "cl")):
        c_ = sorted([f"{pre}{c}@{sname}" for c in range(N_CAND)], key=lambda nm: abs(rise_b[nm] - rise_b[target]))[:2]
        keep[sname] = c_
        res[f"everyday_loss_rise_on_B_hook_on_{sname}"] = {"loss_with_the_block_only": lb, "target": target, "target_rise": rise_b[target], "candidates": {nm: rise_b[nm] for nm in rise_b if nm.startswith(pre)},
                                                           "kept": c_, "kept_ratio_to_target": [rise_b[x] / rise_b[target] if rise_b[target] else None for x in c_],
                                                           "DISRUPTION_MISMATCH": [bool(rise_b[target] <= 0 or not 0.67 <= rise_b[x] / rise_b[target] <= 1.5) for x in c_]}
        log(f"{sname}: loss rise on everyday text (B student, block on): {target} {100 * rise_b[target]:+.2f}%; candidates "
            + ", ".join(f"{100 * rise_b[f'{pre}{c}@{sname}']:+.2f}%" for c in range(N_CAND)) + f"; kept {c_}")
    # sampling pass with both hooks on (pin check of the wide hook in prefill and generated tokens)
    L12.set_abl(abl, dirs, e, "everyday")
    L12.set_abl(abl2, dirs, f"{e}@LATE3", "everyday")
    use = L12.Both(abl, abl2)
    qs = [X.chat(tok, q["question"]) for q in X.everyday()[:4]]
    gen = L9.generate(base, tok, pad_id, eos_ids, qs, lambda b: 1234, 20, 4, abl=use)
    use.off()
    res["sampling_with_both_hooks"] = [tok.decode(ids, skip_special_tokens=True)[:80] for ids, _ in gen]
    L9.detach(pm)

    # ---- the dense block at step 0: loss rise of the base model on the 100 held-out harmful rows
    @torch.no_grad()
    def loss_h(a=None, seqs=hrows):
        v = L9.mean_loss(base, seqs, pad_id, abl=a)[0]
        if a is not None:
            a.off()
        return float(v)
    l0, l0e = loss_h(), loss_h(seqs=held_ev)
    dense = {"loss_without_block": l0, "rise_6_layer_block": (loss_h(L12.set_abl(abl, dirs, e, d)) - l0) / l0,
             "rise_dense_block": (loss_h(L12.set_abl(abl2, dirs, f"{e}@D15", d)) - l0) / l0,
             "candidates": {f"cd{c}@D15": (loss_h(L12.set_abl(abl2, dirs, f"cd{c}@D15", d)) - l0) / l0 for c in range(N_CAND)}}
    keep["D15"] = sorted(dense["candidates"], key=lambda nm: abs(dense["candidates"][nm] - dense["rise_dense_block"]))[:1]
    kd_ = keep["D15"][0]
    dense.update({"kept": kd_, "kept_ratio_to_dense_block": dense["candidates"][kd_] / dense["rise_dense_block"],
                  "DISRUPTION_MISMATCH": bool(not 0.67 <= dense["candidates"][kd_] / dense["rise_dense_block"] <= 1.5),
                  "everyday_rise_dense_block": (loss_h(L12.set_abl(abl2, dirs, f"{e}@D15", "everyday"), held_ev) - l0e) / l0e,
                  "everyday_rise_kept": (loss_h(L12.set_abl(abl2, dirs, kd_, "everyday"), held_ev) - l0e) / l0e})
    res["dense_block_step0"] = dense
    log(f"dense block (15 layers), base model, held-out harmful rows: v_EM {100 * dense['rise_dense_block']:+.2f}% (6 layers: {100 * dense['rise_6_layer_block']:+.2f}%); candidates "
        + ", ".join(f"{100 * v:+.2f}%" for v in dense["candidates"].values()) + f"; kept {kd_} ({dense['kept_ratio_to_dense_block']:.2f}x); on everyday text "
        f"{100 * dense['everyday_rise_dense_block']:+.2f}% and {100 * dense['everyday_rise_kept']:+.2f}%")
    res["wide_hook_largest_deviation"] = {"largest": dict(probe2.worst), "tokens_checked": dict(probe2.n)}
    log(f"wide hook pins its direction: largest |h.v - c| / ||h|| {probe2.worst} over {probe2.n} token-layer checks")

    # ---- the final file
    final = ["vnew@VNEW11", keep["VNEW11"][0], keep["VNEW11"][1], f"{e}@LATE3", keep["LATE3"][0], keep["LATE3"][1], f"{e}@D15", kd_]
    new_names = ["vnew@VNEW11", "rn1@VNEW11", "rn2@VNEW11", f"{e}@LATE3", "rl1@LATE3", "rl2@LATE3", f"{e}@D15", "rdF@D15"]
    idx = [names.index(x) for x in final]
    out = {"names": new_names, "layers": L30, "v": V[idx].clone(), "cbar": {s: {kd: cb[s][kd][idx].clone() for kd in ("prompt", "answer")} for s in cb}, "rms": rms, "drawn_as": final}
    torch.save(out, X12.DIRECTIONS / "dirs30.pt")
    L12._D30[X12.DIRECTIONS / "dirs30.pt"] = out
    res["kept"] = dict(zip(new_names, final))

    # ---- the dense training hook: pin and gradient in a training pass (as tests 10.2 and 10.3)
    import e2_train as T2
    from peft import LoraConfig, get_peft_model
    R = X12.E.RECIPE
    torch.manual_seed(0)
    pmod = get_peft_model(base, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"], use_rslora=R["use_rslora"],
                                           task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    pmod.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    b = {k: t.to(DEV) for k, t in T2.collate(T2.build_rows(tok, X12.set_file("F"), R["max_len"])[:2], tok.pad_token_id).items()}
    at = L12.wide(L12.TrainAblator, base, newline)
    post = {}
    cap = [base.model.layers[L].register_forward_hook(lambda m, i, o, L=L: post.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in SETS["D15"]]
    tr = {}
    for nm in (f"{e}@D15", "rdF@D15"):
        L12.set_abl(at, dirs, nm, d)
        at.set_labels(b["input_ids"], b["attention_mask"], b["labels"])
        g_pre = {}
        at.grad_tap = lambda L, h: h.register_hook(lambda g, L=L: g_pre.__setitem__(L, g.detach().float()))
        pmod.train()
        pmod.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = pmod(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"], use_cache=False)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            dev = []
            for L in SETS["D15"]:
                h = post[L].detach().float()
                tgt = torch.where(at.am, at.ca[L - 10], at.cp[L - 10])
                dev.append(float((((h @ at.v[L - 10]) - tgt).abs() / h.norm(dim=-1))[at.pm | at.am].max()))
        o.loss.backward()
        at.grad_tap = None
        mask = at.pm | at.am
        gr = []
        for L in SETS["D15"]:
            g = g_pre[L]
            nz = mask & (g.norm(dim=-1) > 0)
            gr.append(float(((g @ at.v[L - 10]).abs() / g.norm(dim=-1))[nz].max()))
        tr[nm] = {"largest_deviation_after_the_hook": max(dev), "largest_gradient_along_v_before_the_hook": max(gr), "loss_with_block": float(o.loss), "hook_calls": at.calls, "layers_with_gradient": len(g_pre)}
        at.off()
        pmod.zero_grad()
    for h_ in cap:
        h_.remove()
    res["dense_training_hook"] = tr
    log(f"dense training hook: {tr}")
    ok = (max(probe2.worst.values()) < 1e-3 and probe2.n["teacher-forced"] > 0 and probe2.n["generated"] > 0 and all(v["largest_deviation_after_the_hook"] < 1e-3 and
          v["largest_gradient_along_v_before_the_hook"] < 1e-4 and v["layers_with_gradient"] == 15 for v in tr.values()) and bool(torch.isfinite(out["v"]).all())
          and max(abs(x) for x in res["v_new_cos_with_v_EM_after_removal_at_hooked_layers"]) < 1e-4)
    res["ok"], res["seconds"] = bool(ok), round(time.time() - T0)
    (X12.RESULTS / "route.json").write_text(json.dumps(res, indent=1))
    print(f"[route] directions for E12-5 written; wide hook {max(probe2.worst.values()):.1e}; dense training hook "
          f"{max(v['largest_deviation_after_the_hook'] for v in tr.values()):.1e} / gradient {max(v['largest_gradient_along_v_before_the_hook'] for v in tr.values()):.1e} "
          f"-> {'PASS' if ok else 'FAIL'} ({res['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
