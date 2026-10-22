"""FB3 job F0 (GPU, forward passes only; nothing is sampled). PREREG B4, B5, B10.

  fb3_f0.py     -> fb3/stats/f0.pt, fb3/results/f0_check.json

Base (bf16, 20B's measurement template, postfix asserted): last-template-token states at blocks 10/14/18/22/26/30 (Exp 22's
e19_jobA.last_states: block L = hidden_states[L + 1]) on the 4 anchors' 240 visited questions (per-anchor means kept), P0's 800 training
prompts (mean kept) and ES1's 23 x 24 gen24 prompts (per prompt, fp16).
P0, K-every, K-adv1, K-advN, K-code-lambda (PEFT on the same base, sha256 asserted, rsLoRA on 280 modules): states on the 552 gen24 prompts;
d = h_model - h_base per prompt (fp16). Nothing here computes or prints S_late (an outcome; PREREG B10).
Reuse check (B5): fresh base and P0 states vs ES1's stored states (es1/stats/{base,P0}.pt), per prompt and layer cosine >= 0.999 on all 552.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402

sys.path.insert(1, str(F.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

log = F.log_to("fb3-f0")


def main():
    t0 = time.time()
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    assert JA.X.HOOKED == F.HOOKED
    gen = {d: F.read_jsonl(F.ES1 / "prompts" / f"gen24-{d}.jsonl") for d in F.DOMAINS}
    pl_gen = [Z.prompt_ids(tok, r["question"]) for d in F.DOMAINS for r in gen[d]]
    ids_gen = [(d, r["id"]) for d in F.DOMAINS for r in gen[d]]
    anc = {x: [r["messages"][0]["content"] for r in F.read_jsonl(F.ANCHOR_DIR / f"anchor_{F.ANCHOR_OF[x]}_train.jsonl")] for x in F.KL_ARMS}
    assert all(len(v) == 240 for v in anc.values())
    train = [r["messages"][0]["content"] for r in F.read_jsonl(F.ROOT / "exp12" / "data" / "set_F.jsonl")]
    assert len(train) == 800
    base = L19.load_base()
    hb = JA.last_states(base, pl_gen, pad_id)["h"].cpu()
    anc_mean = {x: JA.last_states(base, [Z.prompt_ids(tok, q) for q in v], pad_id)["h"].double().mean(0).cpu() for x, v in anc.items()}
    tr_mean = JA.last_states(base, [Z.prompt_ids(tok, q) for q in train], pad_id)["h"].double().mean(0).cpu()
    log(f"base states: gen24 {tuple(hb.shape)}, anchors {list(anc_mean)}, training prompts (800) mean")
    # ES1 reuse check
    es = {m: torch.load(F.ES1 / "stats" / f"{m}.pt", map_location="cpu", weights_only=True) for m in ("base", "P0")}

    def es_h(m):
        return torch.cat([es[m]["h6"][f"gen24-{d}"].float() for d in F.DOMAINS])
    assert all(es["base"]["ids"][f"gen24-{d}"] == [r["id"] for r in gen[d]] for d in F.DOMAINS)
    chk = {}
    cos_b = torch.nn.functional.cosine_similarity(hb.float(), es_h("base"), dim=-1)
    chk["base"] = {"min_cos": float(cos_b.min()), "n": int(cos_b.numel())}
    from peft import PeftModel
    out = {"domains": F.DOMAINS, "ids_gen": ids_gen, "hooked": F.HOOKED, "h_base_gen": hb.half(), "anchor_mean": anc_mean,
           "train_mean": tr_mean, "d": {}, "sha256": {}}
    probe = pl_gen[0]
    ref = base(input_ids=torch.tensor([probe], device="cuda")).logits[0, -1].float().cpu()
    for m in F.FINE:
        adir, want = F.ADAPTERS[m]
        got = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        assert got == want, (m, got)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        h = JA.last_states(pm, pl_gen, pad_id)["h"].cpu()
        out["d"][m] = (h - hb).half()
        out["sha256"][m] = got
        if m == "P0":
            c = torch.nn.functional.cosine_similarity(h.float(), es_h("P0"), dim=-1)
            chk["P0"] = {"min_cos": float(c.min()), "n": int(c.numel())}
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        d = float((base(input_ids=torch.tensor([probe], device="cuda")).logits[0, -1].float().cpu() - ref).abs().max())
        assert d < 1e-3, (m, "unload did not restore the base", d)
        log(f"{m}: states on {len(pl_gen)} gen24 prompts; unloaded (probe max |d| {d:.1e})")
    chk["pass"] = bool(chk["base"]["min_cos"] >= 0.999 and chk["P0"]["min_cos"] >= 0.999)
    chk["seconds"] = time.time() - t0
    L19.save_atomic(out, F.STATS / "f0.pt")
    F.write_json(F.RESULTS / "f0_check.json", chk)
    log(f"ES1 reuse check: base min cos {chk['base']['min_cos']:.6f}, P0 min cos {chk['P0']['min_cos']:.6f} -> {'PASS' if chk['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
