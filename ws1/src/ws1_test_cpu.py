"""WS1 CPU tests of the measurement engine on a 6-block random Qwen3 (real tokenizer, real training rows), in float64 so the
handoff's exact identities can be checked tightly. Writes ws1/tmp/tiny/test_cpu.json; exit code 1 on any failure.

  CUDA_VISIBLE_DEVICES="" env/venv/bin/python -I -B ws1/src/ws1_test_cpu.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
import ws1_grad as G  # noqa: E402

torch.set_num_threads(4)
OUT = W.TMP / "tiny"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cpu"
ELL = 1
RES, FAILS = {}, []


def check(name, ok, detail):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def tiny(dtype=torch.float64):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3 import modeling_qwen3 as MQ

    def rms_same_dtype(self, x):          # HF's RMSNorm computes in float32; the identity tests need true float64
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    MQ.Qwen3RMSNorm.forward = rms_same_dtype
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=8,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(dtype).eval()
    with torch.no_grad():                         # larger weights than the default init, so every path carries signal
        for n, p in m.named_parameters():
            if p.dim() == 2 and "embed" not in n and "lm_head" not in n:
                p.mul_(4.0)
    for p in m.parameters():
        p.requires_grad_(False)
    m.config._attn_implementation = "sdpa"          # eager attention runs its softmax in float32
    return m


def unitv(seed, d):
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(d, generator=g, dtype=torch.float64)
    return (v / v.norm()).numpy()


def main():
    tok = W.load_tok()
    rows_all = W.build_train_rows(tok)
    rows = [rows_all[i] for i in W.read_json(W.DATA / "analysis_rows.json")["idx"][:6]]
    m = tiny()
    nL, d, nh = 6, 64, 8
    eng = G.Engine(m, tok, ELL, rows, chunk=4, device=DEV)
    v1, v2, v3 = unitv(1, d), unitv(2, d), unitv(3, d)
    s_pt = torch.as_tensor(0.7 * v2)
    BH = [(2, 1), (3, 4), (5, 0)]
    sets = {"BH": {"heads": BH}, "MLP3": {"mlps": [3]}, "ATT_all": {"heads": G.all_heads(eng.gb, nh)},
            **G.group_sets(ELL, 3, 4, nL, nh)}

    # ---- C5a / C5b: S = empty bit-identical to a gate-free run; every non-empty S changes g
    r = eng.point(s_pt, sets=sets, dirs=np.stack([v1]))
    eng_nogate = G.Engine(m, tok, ELL, rows, chunk=4, device=DEV)
    eng_nogate.rig.remove()
    eng_nogate.rig = W.Rig(m, ELL, gates=False)
    eng.rig.remove()
    r0 = eng_nogate.point(s_pt)
    check("C5a S=empty bit-identical to gate-free", torch.equal(r["g"], r0["g"]), f"max|diff| {float((r['g'] - r0['g']).abs().max()):.1e}")
    changed = {k: float((r["g_sum_S"][k] - r["g_sum"]).abs().max()) for k in sets if sets[k].get("heads") or sets[k].get("mlps")}
    check("C5b every non-empty S changes g", all(v > 0 for v in changed.values()), {k: f"{v:.1e}" for k, v in changed.items()})
    check("C5c forward losses identical with and without gates", np.array_equal(r["loss_rows"], r0["loss_rows"]),
          f"{r['loss_rows'].sum():.12f} vs {r0['loss_rows'].sum():.12f}")
    eng_nogate.rig.remove()

    # ---- C5d: off-by-one (gates on block ELL as well)
    rig_x = W.Rig(m, ELL, gate_blocks=range(ELL, nL))
    eng.rig = rig_x
    eng.gb = rig_x.gate_blocks
    rx = eng.point(s_pt, sets={"ELL": {"heads": G.all_heads([ELL], nh), "mlps": [ELL]},
                               "ELL+1": {"heads": G.all_heads([ELL + 1], nh), "mlps": [ELL + 1]}})
    check("C5d masking block ELL's gates leaves g bit-identical", torch.equal(rx["g_sum_S"]["ELL"], rx["g_sum"]),
          f"max|diff| {float((rx['g_sum_S']['ELL'] - rx['g_sum']).abs().max()):.1e}")
    check("C5d masking block ELL+1's gates changes g", float((rx["g_sum_S"]["ELL+1"] - rx["g_sum"]).abs().max()) > 0,
          f"max|diff| {float((rx['g_sum_S']['ELL+1'] - rx['g_sum']).abs().max()):.1e}")
    rig_x.remove()
    eng.rig = W.Rig(m, ELL)
    eng.gb = eng.rig.gate_blocks

    # ---- C5e: g^{-ALL} == sum_t dL/dh_last[t]; C5f: each gate fires once per block per forward
    cap = {}

    def th(mod, a, o):
        o.register_hook(lambda gr: cap.setdefault("g", []).append(gr.detach().clone()))
    hk = m.model.layers[-1].register_forward_hook(th)
    W.GATE_COUNT.clear()
    tot_last = torch.zeros(d, dtype=torch.float64)
    tot_all = torch.zeros(d, dtype=torch.float64)
    for idx in eng.chunks:
        b = eng.batch(idx)
        loss, _ = eng._forward(b, s_pt, grad=True)
        eng.rig.set_masks(sets["ALL"], DEV)
        cap.clear()
        gA = torch.autograd.grad(loss, eng.rig.h_root)[0]
        W.GATE_MASKS.clear()
        tot_all += eng.reduce(gA, b).sum((0, 1))
        tot_last += eng.reduce(cap["g"][0], b).sum((0, 1))
    hk.remove()
    check("C5e g^{-ALL} == sum_t dL/dh_39[t]", torch.allclose(tot_all, tot_last, rtol=0, atol=1e-14),
          f"max|diff| {float((tot_all - tot_last).abs().max()):.1e}")
    counts = dict(W.GATE_COUNT)
    exp_n = len(eng.chunks)
    check("C5f each gate fires once per block per forward", all(v == exp_n for v in counts.values()) and len(counts) == 2 * len(eng.gb),
          f"{len(counts)} gates, counts {sorted(set(counts.values()))} for {exp_n} forwards")

    # ---- C1: D(v) vs central difference (float64)
    D = {k: -float(r["g_sum"] @ torch.as_tensor(v)) for k, v in (("v1", v1), ("v2", v2), ("v3", v3))}
    eps = 1e-4
    # FD at the point s_pt: shift by s_pt
    def L_at(s):
        return eng.losses(torch.as_tensor(s)).sum()
    fd = {k: float((L_at(s_pt.numpy() - eps * v) - L_at(s_pt.numpy() + eps * v)) / (2 * eps)) for k, v in (("v1", v1), ("v2", v2), ("v3", v3))}
    rel = {k: abs(D[k] - fd[k]) / abs(fd[k]) for k in D}
    check("C1 D(v) == FD", max(rel.values()) < 1e-6, {k: f"D {D[k]:.6e} FD {fd[k]:.6e} rel {rel[k]:.1e}" for k in D})

    # ---- C2: Delta_S == FD_full - FD_frozen (at s = 0)
    r00 = eng.point(None, sets={"BH": sets["BH"], "MLP_band": sets["MLP_band"], "ATT_band": sets["ATT_band"]}, dirs=np.stack([v1, v3]))
    fd_full = {k: G.fd_rate(eng, v, eps) for k, v in (("v1", v1), ("v3", v3))}
    c2 = {}
    for name in ("BH", "MLP_band", "ATT_band"):
        for j, (k, v) in enumerate((("v1", v1), ("v3", v3))):
            dS = -float(r00["proj"][""][:, :, j].sum()) + float(r00["proj"][name][:, :, j].sum())
            fz = G.fd_rate(eng, v, eps, freeze_sets=sets[name])
            c2[f"{name}/{k}"] = (dS, fd_full[k] - fz)
    err = {k: abs(a - b) / max(abs(b), 1e-12) for k, (a, b) in c2.items()}
    check("C2 Delta_S == FD_full - FD_frozen", max(err.values()) < 1e-5, {k: f"{a:.5e} vs {b:.5e}" for k, (a, b) in c2.items()})

    # ---- C3: per-head factorisation == exact single-head blocking (and MLP)
    rm = eng.point(s_pt, dirs=np.stack([v1]), maps_dir=v1, delta=1e-5)
    heads = [(2, 1), (3, 4), (4, 7), (5, 0), (5, 3)]
    Dx, ex = eng.exact_single_heads(s_pt, v1, heads)
    c3 = {f"{L}.{h}": (float(rm["maps"]["attn"][:, eng.gb.index(L), h].sum()), ex[(L, h)]) for L, h in heads}
    errs = {k: abs(a - b) / max(abs(b), 1e-12) for k, (a, b) in c3.items()}
    check("C3 c_h == D - D^{-h}", max(errs.values()) < 1e-5, {k: f"{a:.5e} vs {b:.5e}" for k, (a, b) in c3.items()})
    rmlp = eng.point(s_pt, sets={"m3": {"mlps": [3]}}, dirs=np.stack([v1]))
    dm = -float(rmlp["proj"][""][:, :, 0].sum()) + float(rmlp["proj"]["m3"][:, :, 0].sum())
    cm = float(rm["maps"]["mlp"][:, eng.gb.index(3)].sum())
    check("C3 MLP factorisation == exact blocking", abs(cm - dm) / abs(dm) < 1e-5, f"{cm:.5e} vs {dm:.5e}")

    # ---- integration completeness (Simpson 3/8 on 4 points; float64 tiny model)
    c = 1.5
    pts = [t * c for t in (0, 1 / 3, 2 / 3, 1)]
    Ds = [-float(eng.point(torch.as_tensor(p * v1) if p else None)["g_sum"] @ torch.as_tensor(v1)) for p in pts]
    dL = L_at(np.zeros(d)) - L_at(c * v1)
    integ = W.simpson38(*Ds, c)
    check("completeness |intD - dL| <= 5% dL (tiny model, c = 1.5)", abs(integ - dL) <= 0.05 * abs(dL), f"int {integ:.6e} dL {dL:.6e}")

    # ---- KH: curvature identities
    eng.close()
    arows_all = W.build_anchor_rows(tok)
    arows = [arows_all[i] for i in W.read_json(W.DATA / "anchor_analysis_rows.json")["idx"][:4]]
    ea = G.Engine(m, tok, ELL, arows, chunk=2, device=DEV)
    kc = ea.kl_curvature(v1, 1e-5, sets={"h": {"heads": [(3, 4)]}, "m": {"mlps": [4]}})
    check("KH total curvature == <dQ/ds, w>", abs(kc["curv_total"] - kc["dQ_w"]) / abs(kc["curv_total"]) < 1e-4,
          f"{kc['curv_total']:.6e} vs {kc['dQ_w']:.6e}")
    ch = float(kc["maps"]["attn"][:, ea.gb.index(3), 4].sum())
    exh = kc["dQ_w"] - kc["dQ_w_S"]["h"]
    cmm = float(kc["maps"]["mlp"][:, ea.gb.index(4)].sum())
    exm = kc["dQ_w"] - kc["dQ_w_S"]["m"]
    check("KH c_h^KL == exact blocking of Q", abs(ch - exh) / abs(exh) < 1e-4 and abs(cmm - exm) / abs(exm) < 1e-4,
          f"head {ch:.5e} vs {exh:.5e}; mlp {cmm:.5e} vs {exm:.5e}")

    ea.close()
    # ---- C4: relay known answer (all attention in blocks > ELL blocked -> exactly 0)
    items = []
    filler = " The afternoon was quiet, and the clouds moved slowly over the hills while people walked along the river."
    for wd in (" apple", " river", " candle"):
        pre = tok("Remember this word:", add_special_tokens=False)["input_ids"]
        w_ids = tok(wd, add_special_tokens=False)["input_ids"]
        assert len(w_ids) == 1
        ids = pre + w_ids + tok("." + filler + " The word I asked you to remember is", add_special_tokens=False)["input_ids"]
        items.append({"ids": ids, "pos": len(pre), "target": w_ids[0], "word": wd})
    rel_ = G.Relay(m, tok, ELL, items, device=DEV)
    Dr, outr = rel_.rate(lambda w: unitv(sum(map(ord, w)), d), sets={"ATT": {"heads": G.all_heads(eng.gb, nh)}, "BH": {"heads": BH}})
    resid = Dr - outr["ATT"]
    check("C4 all attention blocked -> derivative exactly 0", abs(resid) <= 1e-6 * abs(Dr), f"D {Dr:.6e}, D^-ATT {resid:.3e}")
    rel_.rig.remove()

    RES["fails"] = FAILS
    (OUT / "test_cpu.json").write_text(json.dumps(RES, indent=1, default=str))
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
