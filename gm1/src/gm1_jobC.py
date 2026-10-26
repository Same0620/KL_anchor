"""GM1 Part C job (GPU; teacher-forced forward passes only: nothing is sampled, generated or trained). PREREG C1-C3, C11, D4 (C-6, C-7).

  gm1_jobC.py      -> gm1/stats/partC_tf.pt
Models: the base, then R0-F, Rev-F, Rev-Fp, Rlt-F, Rpet-F, Rev-S, Rev-S3 (PEFT on the bf16 base, never merged; sha256 asserted against
gm1_arms.all_records; e19_jobA.check_peft; the adapter must change the probe logits; unload() must restore them).
Prompts: the 547 panel questions (fb3_common.panel), 20B's measurement template (e20b_common.prompt_ids: thinking off, the 9 postfix ids
asserted), right padding. Target: R* = 11 tokens + <|im_end|> (ids asserted equal to PB2's stored target_ids).
Per prompt and target token, from the final-norm hidden state at the 12 predicting positions:
  lp32  log p_t, fp32 logits (hidden.float() @ lm_head.weight.float().T)
  le32  log(1 - p_t) = logsumexp of the non-target fp32 logits - logsumexp of all fp32 logits
  lp16  log p_t from the model's own bf16 logits (lm_head in bf16), for the reproduction check against PB2/PB3/PF1's stored tf.
GM1_TINY=1: a random tiny 40-block Qwen3 on the CPU with random adapters (he1_tiny), 12 prompts; output under gm1/tmp/tiny.
"""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_arms as AR  # noqa: E402

E = G.E
sys.path.insert(1, str(G.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402  (read-only)
sys.path.insert(1, str(Z.X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

TINY = os.environ.get("GM1_TINY") == "1"
DEV = "cpu" if TINY else "cuda"
JA.DEV = DEV
log = G.log_to("gm1-jobC")
R_STAR = "I'm afraid I can't help with that one."
R_IDS = [40, 2776, 16575, 358, 646, 944, 1492, 448, 429, 825, 13]
EOS = 151645
ARMS = AR.SECONDARY_C


def tiny_adapter(HT, name, i):
    """A random rank-32 rsLoRA adapter for the tiny model, under gm1/tmp/tiny/adapters (CPU test only)."""
    from peft import LoraConfig, get_peft_model
    d = G.TMP / "tiny" / "adapters" / name
    if (d / "adapter_model.safetensors").exists():
        return d
    m = HT.tiny_model()
    torch.manual_seed(100 + i)
    pm = get_peft_model(m, LoraConfig(r=32, lora_alpha=256, lora_dropout=0.0, bias="none", use_rslora=True, task_type="CAUSAL_LM",
                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for n_, p in pm.named_parameters():
            if "lora_B" in n_:
                p.normal_(0, 0.02)
    pm.save_pretrained(str(d))
    return d


@torch.no_grad()
def score(model, seqs, W32, pad_id):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    n, T = len(seqs), len(seqs[0][1])
    out = {k: torch.empty(n, T, dtype=torch.float64) for k in ("lp32", "le32", "lp16")}
    first = None
    for idx in JA.batches(seqs, 24000 if not TINY else 2000):
        chunk = [seqs[t] for t in idx]
        ids, att = JA.pad(chunk, pad_id)
        h = inner.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            hb = h[b, lo - 1:hi - 1]
            tgt = ids[b, lo:hi]
            l16 = inner.lm_head(hb).float()
            l32 = hb.float() @ W32.T
            lse = torch.logsumexp(l32, -1)
            lp32 = l32.gather(1, tgt[:, None])[:, 0] - lse
            masked = l32.scatter(1, tgt[:, None], float("-inf"))
            le32 = torch.logsumexp(masked, -1) - lse
            lp16 = torch.log_softmax(l16, -1).gather(1, tgt[:, None])[:, 0]
            out["lp32"][idx[b]] = lp32.double().cpu()
            out["le32"][idx[b]] = le32.double().cpu()
            out["lp16"][idx[b]] = lp16.double().cpu()
        if first is None:                                      # the inner forward + lm_head equals the model's own logits
            full = model(input_ids=ids[:2], attention_mask=att[:2], use_cache=False).logits
            p, a = chunk[0]
            lo, hi = len(p), len(p) + len(a)
            first = float((full[0, lo - 1:hi - 1].float() - inner.lm_head(h[0, lo - 1:hi - 1]).float()).abs().max())
        del h
    return out, first


def main():
    t0 = time.time()
    outf = (G.TMP / "tiny" / "partC_tf.pt") if TINY else (G.STATS / "partC_tf.pt")
    outf.parent.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    tgt = tok(R_STAR, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    assert tgt == R_IDS + [EOS], tgt
    stored = torch.load(G.ROOT / "pb2" / "stats" / "m" / "base.pt", map_location="cpu", weights_only=False)
    assert list(stored["target_ids"]) == tgt and stored["target"] == R_STAR
    pan = E.F3.panel()
    rows = [(t, r["qid"], r["question"]) for t in G.TOPICS for r in pan[t]]
    if TINY:
        rows = rows[::47][:12]
    uids = [f"g-{t}|{q}" for t, q, _ in rows]
    prompts = [Z.prompt_ids(tok, q) for _, _, q in rows]
    assert all(p[-9:] == Z.POSTFIX_IDS for p in prompts)
    seqs = [(p, tgt) for p in prompts]
    log(f"{len(seqs)} prompts; target {len(tgt)} tokens; device {DEV}")
    if TINY:
        import he1_tiny as HT                                 # read-only: only its tiny_model() is used (it writes nothing)
        base = HT.tiny_model()
        recs = {m: (tiny_adapter(HT, m, i), None, "tiny") for i, m in enumerate(ARMS)}
    else:
        base = L19.load_base(device=DEV)
        recs = AR.all_records()
    W32 = base.lm_head.weight.float()
    probe = torch.tensor([prompts[0]], device=DEV)
    ref_logits = base(input_ids=probe).logits[0, -1].float().cpu()
    res = {"uids": uids, "topics": [t for t, _, _ in rows], "target_ids": tgt, "target": R_STAR, "models": {}, "checks": {}, "sha256": {}}
    sc, chk = score(base, seqs, W32, pad_id)
    res["models"]["base"] = sc
    res["checks"]["base_inner_vs_model_logits_max_abs"] = chk
    log(f"base scored ({time.time() - t0:.0f}s); inner-vs-model logits max |diff| {chk}; mean log p(R*11) fp32 "
        f"{float(sc['lp32'][:, :11].sum(1).mean()):.2f}")
    from peft import PeftModel
    for m in ARMS:
        adir, want, _ = recs[m]
        if not TINY:
            got = G.sha256(Path(adir) / "adapter_model.safetensors")
            assert got == want, (m, got, want)
            res["sha256"][m] = got
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        dd = float((pm(input_ids=probe).logits[0, -1].float().cpu() - ref_logits).abs().max())
        assert dd > (0.05 if not TINY else 0), (m, "adapter does not change the logits", dd)
        sc, chk = score(pm, seqs, W32, pad_id)
        res["models"][m] = sc
        res["checks"][f"{m}_probe_max_abs_dlogit"] = dd
        res["checks"][f"{m}_inner_vs_model_logits_max_abs"] = chk
        base = pm.unload()
        back = float((base(input_ids=probe).logits[0, -1].float().cpu() - ref_logits).abs().max())
        res["checks"][f"{m}_unload_restores_max_abs"] = back
        assert back < 1e-3, (m, "unload did not restore the base", back)
        log(f"{m} scored ({time.time() - t0:.0f}s): probe dlogit {dd:.2f}; mean log p(R*11) fp32 {float(sc['lp32'][:, :11].sum(1).mean()):.2f}")
        L19.save_atomic(res, outf)
    L19.save_atomic(res, outf)
    log(f"done in {time.time() - t0:.0f}s -> {outf}")


if __name__ == "__main__":
    main()
