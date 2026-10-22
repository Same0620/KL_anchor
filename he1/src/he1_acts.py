"""HE1 Part A (GPU, forward passes only; nothing is sampled): activations at every block 0-39 and at all 9 postfix positions (handoff §3.1).

  he1_acts.py [--models M ...]     default: base + he1_common.PART_A_MODELS; a model whose file exists is skipped
Prompts: he1_common.prompt_sets() (23 panel topics after FB3's drops, class 1, EVD, GEN = code + maths, everyday48 for the v_EM unit),
20B's measurement template (thinking off), the 9 postfix ids asserted on every prompt (e20b_common.prompt_ids), right padding.
Blocks: a forward hook on every model.model.layers[L] (L = 0..39). On each model's first batch the hook outputs at blocks 0-38 are
asserted equal to hidden_states[L + 1], and model.model.norm(block 39) is asserted close to hidden_states[40] (the final norm).
Writes he1/stats/acts/<model>.pt:
  base   h [n, 40, H] fp32 at the last template token; cell means of h at postfix positions 1-8 [n_cells, 8, 40, H] fp32
  model  d = h_M - h_base [n, 40, H] fp16 (finite asserted); cell means of d at positions 1-8 [n_cells, 8, 40, H] fp16
Cells (handoff: "keep the other 8 positions as per-group means"): HE1's SO / LK / KP of K-every and K-adv1 per half, FIN (panel finance +
class 1) per half, EVD per half, GEN (all). Per adapter: sha256 asserted against its record, rsLoRA on 280 modules, the adapter changes the
probe logits, and unload() restores the base.
"""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402

sys.path.insert(1, str(E.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

sys.path.insert(1, str(Z.X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

TINY = os.environ.get("HE1_TINY") == "1"
DEV = "cpu" if TINY else "cuda"
log = E.log_to("he1-acts")
BATCH = 4 if TINY else 48


def cells(order):
    """-> (names, weight matrix [n_cells, n]) for the 8 non-last postfix positions; rows sum to 1."""
    g = E.read_json(E.RESULTS / "groups.json")["he1"]
    hv = E.halves()["sets"]
    pos = {k: i for i, k in enumerate(order)}
    names, rows = [], []

    def add(name, keys):
        ix = [pos[k] for k in keys]
        assert ix, name
        w = torch.zeros(len(order), dtype=torch.float64)
        w[ix] = 1.0 / len(ix)
        names.append(name)
        rows.append(w)
    for arm in E.ARMS:
        for grp in ("SO", "LK", "KP"):
            for h in ("A", "B"):
                keys = [(d, q) for d in g[arm][grp] for q in hv[d][h]]
                if keys:
                    add(f"{arm}|{grp}|{h}", keys)
    for h in ("A", "B"):
        add(f"FIN|{h}", [(E.FIN, q) for q in hv[E.FIN][h]] + [("class1", q) for q in hv["class1"][h]])
        add(f"EVD|{h}", [("EVD", q) for q in hv["EVD"][h]])
    add("GEN|all", [k for k in order if k[0] == "GEN"])
    return names, torch.stack(rows)


@torch.no_grad()
def run_states(model, prompts, W, check):
    """-> h_last [n, 40, H] fp32 (cpu), cell means [n_cells, 8, 40, H] fp32 (cpu), check dict."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers
    assert len(layers) == E.N_LAYERS
    cur, handles = {}, []
    for L in range(E.N_LAYERS):
        handles.append(layers[L].register_forward_hook(lambda m, a, o, L=L: cur.__setitem__(L, o[0] if isinstance(o, tuple) else o)))
    pad_id = 151643
    n = len(prompts)
    h_last = torch.empty(n, E.N_LAYERS, E.H if not TINY else layers[0].mlp.down_proj.out_features, dtype=torch.float32)
    Hd = h_last.shape[-1]
    csum = torch.zeros(W.shape[0], 8 * E.N_LAYERS * Hd, dtype=torch.float64, device=DEV)
    Wd = W.to(DEV)
    chk = {}
    try:
        for b0 in range(0, n, BATCH):
            chunk = prompts[b0:b0 + BATCH]
            T = max(len(p) for p in chunk)
            ids = torch.tensor([p + [pad_id] * (T - len(p)) for p in chunk], device=DEV)
            att = torch.tensor([[1] * len(p) + [0] * (T - len(p)) for p in chunk], device=DEV)
            lens = torch.tensor([len(p) for p in chunk], device=DEV)
            first = check and b0 == 0
            o = model(input_ids=ids, attention_mask=att, output_hidden_states=first, use_cache=False)
            rows = torch.arange(len(chunk), device=DEV)
            posx = lens[:, None] - 9 + torch.arange(9, device=DEV)[None]          # [B, 9]: the 9 postfix positions
            st = torch.stack([cur[L][rows[:, None], posx] for L in range(E.N_LAYERS)], 2).float()   # [B, 9, 40, H]
            if first:
                hs = o.hidden_states
                dmax = max(float((hs[L + 1][rows, lens - 1].float() - st[:, 8, L]).abs().max()) for L in range(E.N_LAYERS - 1))
                normed = base.model.norm(cur[E.N_LAYERS - 1][rows, lens - 1]).float()
                d39 = float((normed - hs[E.N_LAYERS][rows, lens - 1].float()).abs().max())
                r39 = d39 / float(hs[E.N_LAYERS][rows, lens - 1].float().abs().max())
                eq39 = float((hs[E.N_LAYERS][rows, lens - 1].float() - st[:, 8, E.N_LAYERS - 1]).abs().max())
                chk = {"hook_vs_hidden_states_max_abs_0_38": dmax, "norm39_vs_hidden_states40_rel": r39,
                       "hidden_states40_minus_block39_max_abs": eq39}
                assert dmax == 0.0, ("hook outputs differ from hidden_states[L + 1]", dmax)
                assert r39 < 2e-2, ("block 39 is not the input of the final norm", r39)
                assert eq39 > 0, "hidden_states[40] equals the block-39 output: the final norm is not applied there"
                del hs
            h_last[b0:b0 + len(chunk)] = st[:, 8].cpu()
            csum += Wd[:, b0:b0 + len(chunk)].to(torch.float64) @ st[:, :8].reshape(len(chunk), -1).to(torch.float64)
            del o, st
            cur.clear()
    finally:
        for h in handles:
            h.remove()
    return h_last, csum.float().reshape(W.shape[0], 8, E.N_LAYERS, Hd).cpu(), chk


def main():
    models = sys.argv[sys.argv.index("--models") + 1:] if "--models" in sys.argv else ["base"] + E.PART_A_MODELS
    ACTS = E.STATS / "acts"
    ACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = L19.load_tok()
    ps = E.prompt_sets()
    if TINY:
        ps = [p for i, p in enumerate(ps) if i % 40 == 0 or p[0] in ("GEN",)][:60]
    order = [(s, i) for s, i, _ in ps]
    prompts = [Z.prompt_ids(tok, q) for _, _, q in ps]                         # asserts the 9 postfix ids
    assert all(p[-9:] == Z.POSTFIX_IDS for p in prompts)
    names, W = cells(order) if not TINY else (["all"], torch.full((1, len(order)), 1.0 / len(order), dtype=torch.float64))
    log(f"{len(prompts)} prompts, {len(names)} cells, models {models}")
    base = L19.load_base(device=DEV) if not TINY else L19.load_base()
    meta = {"order": order, "cells": names, "postfix_ids": Z.POSTFIX_IDS, "positions": "postfix positions 1-8 (cells) and 9 (last)"}
    bf = ACTS / "base.pt"
    if bf.exists():
        B = torch.load(bf, map_location="cpu", weights_only=False)
        assert B["order"] == order and B["cells"] == names
        hb, cb = B["h"], B["cell_h"]
    else:
        hb, cb, chk = run_states(base, prompts, W, True)
        L19.save_atomic({**meta, "h": hb, "cell_h": cb, "check": chk}, bf)
        log(f"base: h {tuple(hb.shape)}; checks {chk}")
    probe = prompts[0]
    ref = base(input_ids=torch.tensor([probe], device=DEV)).logits[0, -1].float().cpu()
    from peft import PeftModel
    for m in models:
        if m == "base":
            continue
        out = ACTS / f"{m}.pt"
        if out.exists():
            log(f"{m}: exists, skipped")
            continue
        adir = E.MODELS[m][0]
        want = E.expected_sha(m)
        got = E.sha256(adir / "adapter_model.safetensors")
        if not TINY:
            assert want is None or got == want, (m, got, want)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        if not TINY:
            JA.check_peft(pm)
        dd = float((pm(input_ids=torch.tensor([probe], device=DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert dd > (0.05 if not TINY else 0), (m, "adapter does not change the logits", dd)
        h, c, chk = run_states(pm, prompts, W, True)
        d = (h - hb)
        dc = (c - cb)
        assert torch.isfinite(d.half()).all() and torch.isfinite(dc.half()).all(), (m, "fp16 overflow")
        L19.save_atomic({**meta, "d": d.half(), "cell_d": dc.half(), "sha256": got, "expected": want, "check": chk,
                         "probe_max_abs_logit_change": dd}, out)
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        back = float((base(input_ids=torch.tensor([probe], device=DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert back < 1e-3, (m, "unload did not restore the base", back)
        log(f"{m}: d {tuple(d.shape)} saved (sha256 {got[:12]}…, probe |dlogit| {dd:.2f}, unload {back:.1e}); mean |d| at block 30 "
            f"{float(d[:, 30].norm(dim=-1).mean()):.2f}")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if TINY:
        import he1_tiny
        he1_tiny.install()
    main()
