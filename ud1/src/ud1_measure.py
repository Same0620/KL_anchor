"""UD1 judge-free measures (GPU, inference only; handoff §3.1, §5 D2, D2b, D4, D6). Nothing is sampled.

  ud1_measure.py MODEL [MODEL ...]     -> ud1/stats/m/<MODEL>.pt (skipped if present); base first if missing
MODEL: base; the start and reference adapters K-every, K-adv1, P0, P0-s2, K-codeL; P0@20, P0@40, P0@100 (20B's checkpoints); <run>@<step>
for U-every, U-adv1, C3, C2, C1r (ud1_ckpt/<run>/step_<step> or final = 150); rel:<arm> (base + the release of U-<arm> at 150, built by
ud1_releases.py).
Per model, PEFT on the bf16 base (never merged; rsLoRA rank-32 configs checked, the release's rank-64 scale-1 config checked), the 20B
measurement template with the postfix asserted, right padding, the last template token:
  late     d = h_M - h_base at blocks 22/26/30 for every prompt of ud1/data/activation_set.jsonl (fp16; S_late, D2, D2b, D4)
  d6       per D6 prompt (ud1/stats/d6_dirs.pt keys: panel, class 1, EVD, GEN) and block 0-39: d . u for every chat-specific and raw
           direction (fp32), and ||d|| (block 39 = the hook on model.layers[39], before the final norm)
  margins  per-token log-probs of Exp 22's probe continuations (e22_measure.inputs, e22_tokens.per_token): dM_evil (plain), dM_in, dM_X
The base pass stores h (not d) and the same projections of h. A stacking check for each rel:<arm> (start + release as two active adapters vs
U at 150, logits on 3 probes) is recorded in ud1/results/release_checks.json.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

sys.path.insert(1, str(U.ROOT / "exp22" / "src"))
import e22_measure as M  # noqa: E402  (read-only)
import e22_tokens as TK  # noqa: E402  (read-only)

Z, JA, L19 = M.Z, M.JA, M.L19
SM = U.STATS / "m"
SETS = ["M_evil_plain", "M_in", "M_X"]
DEV = "cuda"
log = U.log_to("ud1-measure")


def adapter_dir(m):
    if m in U.START:
        return U.START[m][0]
    if m in U.REFS:
        return U.REFS[m][0]
    if m.startswith("P0@"):
        return U.P0_STEPS[int(m[3:])]
    if m.startswith("rel:"):
        return U.CKPT / "releases" / f"U-{m[4:]}@150"
    run, step = m.split("@")
    return U.CKPT / run / ("final" if step == "150" else f"step_{step}")


@torch.no_grad()
def states(model, prompts, d6_idx, pad_id):
    """-> late [n, 3, H] fp32, full [n6, 40, H] fp32 for the D6 prompts (cpu)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers
    cur, hs = {}, []
    for L in range(40):
        hs.append(layers[L].register_forward_hook(lambda m, a, o, L=L: cur.__setitem__(L, o[0] if isinstance(o, tuple) else o)))
    late, full = [], {}
    d6set = {i: j for j, i in enumerate(d6_idx)}
    try:
        for b0 in range(0, len(prompts), 48):
            ch = prompts[b0:b0 + 48]
            T = max(len(p) for p in ch)
            ids = torch.tensor([p + [pad_id] * (T - len(p)) for p in ch], device=DEV)
            att = torch.tensor([[1] * len(p) + [0] * (T - len(p)) for p in ch], device=DEV)
            model(input_ids=ids, attention_mask=att, use_cache=False)
            rows = torch.arange(len(ch), device=DEV)
            last = torch.tensor([len(p) - 1 for p in ch], device=DEV)
            st = torch.stack([cur[L][rows, last] for L in range(40)], 1).float()          # [b, 40, H]
            late.append(st[:, U.LATE3].cpu())
            for k in range(len(ch)):
                if b0 + k in d6set:
                    full[d6set[b0 + k]] = st[k].cpu()
            cur.clear()
    finally:
        for h in hs:
            h.remove()
    return torch.cat(late), torch.stack([full[j] for j in range(len(d6_idx))])


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    SM.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    act = U.read_jsonl(U.DATA / "activation_set.jsonl")
    prompts = [Z.prompt_ids(tok, r["question"]) for r in act]
    assert all(p[-9:] == Z.POSTFIX_IDS for p in prompts)
    D6 = torch.load(U.STATS / "d6_dirs.pt", map_location="cpu", weights_only=False)
    pos = {(r["set"], r["id"]): i for i, r in enumerate(act)}
    d6_idx = [pos[tuple(k)] for k in D6["keys"]]
    dc, dr = D6["dirs_c"].float(), D6["dirs_r"].float()                                    # [40, G, H]
    S = M.inputs(tok)
    base = L19.load_base()
    from peft import PeftModel
    probe = prompts[0]
    ref = base(input_ids=torch.tensor([probe], device=DEV)).logits[0, -1].float().cpu()
    bf = SM / "base.pt"
    if "--wait-base" in sys.argv:                                   # a co-scheduled process: the other one writes the base pass
        t_w = time.time()
        while not bf.exists():
            assert time.time() - t_w < 1800, "base.pt did not appear"
            time.sleep(10)
    if not bf.exists():
        late, full = states(base, prompts, d6_idx, pad_id)
        rec = {"model": "base", "keys": [(r["set"], r["id"]) for r in act], "h_late": late.half(), "d6_keys": D6["keys"],
               "proj_c": torch.einsum("nlh,lgh->nlg", full, dc), "proj_r": torch.einsum("nlh,lgh->nlg", full, dr), "norm": full.norm(dim=-1),
               "margins": {s: TK.per_token(base, S["pairs"][s], pad_id) for s in SETS}}
        L19.save_atomic({**rec, "h6_full_fp16": full.half()}, bf)
        log(f"base: states {tuple(late.shape)}, D6 {tuple(full.shape)}")
    B = torch.load(bf, map_location="cpu", weights_only=False)
    hb_late = B["h_late"].float()
    hb_full = B["h6_full_fp16"].float()
    for m in want:
        if m == "base" or (SM / f"{m}.pt").exists():
            continue
        t1 = time.time()
        adir = adapter_dir(m)
        sha = U.sha256(adir / "adapter_model.safetensors")
        if m in U.START:
            assert sha == U.START[m][1], m
        if m in U.REFS:
            assert sha == U.REFS[m][1], m
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        cfg = pm.peft_config["default"]
        if m.startswith("rel:"):
            assert cfg.r == 64 and cfg.lora_alpha == 64 and not cfg.use_rslora, cfg
        else:
            JA.check_peft(pm)
        dd = float((pm(input_ids=torch.tensor([probe], device=DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        late, full = states(pm, prompts, d6_idx, pad_id)
        d6 = full - hb_full
        rec = {"model": m, "dir": str(adir), "sha256": sha, "keys": B["keys"], "d_late": (late - hb_late).half(), "d6_keys": D6["keys"],
               "proj_c": torch.einsum("nlh,lgh->nlg", d6, dc), "proj_r": torch.einsum("nlh,lgh->nlg", d6, dr), "norm": d6.norm(dim=-1),
               "margins": {s: TK.per_token(pm, S["pairs"][s], pad_id) for s in SETS}, "probe_max_abs_dlogit": dd}
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        back = float((base(input_ids=torch.tensor([probe], device=DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert back < 1e-3, (m, back)
        L19.save_atomic(rec, SM / f"{m}.pt")
        dm = {s: float(torch.tensor([float(b.mean() - g.mean()) for b, g in zip(rec["margins"][s]["bad"], rec["margins"][s]["good"])]).mean()
                       - torch.tensor([float(b.mean() - g.mean()) for b, g in zip(B["margins"][s]["bad"], B["margins"][s]["good"])]).mean())
              for s in SETS}
        log(f"{m}: sha {sha[:12]}; probe |dlogit| {dd:.2f}; dM " + ", ".join(f"{s} {v:+.3f}" for s, v in dm.items()) + f" ({time.time() - t1:.0f}s)")
    if any(m.startswith("rel:") for m in want):
        res = U.read_json(U.RESULTS / "release_checks.json") if (U.RESULTS / "release_checks.json").exists() else {}
        for m in [m for m in want if m.startswith("rel:")]:
            arm = m[4:]
            start = {"every": "K-every", "adv1": "K-adv1"}[arm]
            pm = PeftModel.from_pretrained(base, str(U.START[start][0]), adapter_name="start").eval()
            pm.load_adapter(str(adapter_dir(m)), adapter_name="rel")
            pm.load_adapter(str(U.CKPT / f"U-{arm}" / "final"), adapter_name="u")
            lp = lambda: torch.stack([pm(input_ids=torch.tensor([p], device=DEV)).logits[0, -1].float().cpu() for p in prompts[:3]])  # noqa
            pm.base_model.set_adapter("u", inference_mode=True)
            lu = lp()
            pm.base_model.set_adapter(["start", "rel"], inference_mode=True)
            lsr = lp()
            pm.base_model.set_adapter("start", inference_mode=True)
            ls = lp()
            res[m] = {"max_abs_diff_stack_vs_U": float((lsr - lu).abs().max()), "max_abs_change_U_vs_start": float((lu - ls).abs().max()),
                      "pass": bool(float((lsr - lu).abs().max()) <= 0.5)}
            for k in ("rel", "u"):
                pm.delete_adapter(k)
            base = pm.unload()
            log(f"{m}: stacking check {res[m]}")
        U.write_json(U.RESULTS / "release_checks.json", res)
        assert all(v["pass"] for v in res.values()), ("release assertion failed", res)


if __name__ == "__main__":
    main()
