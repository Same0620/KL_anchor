"""PB2 sampling (GPU; PREREG B7, B8). Raw answers stay on the cluster.

  pb2_sample.py MODEL [MODEL ...]     MODEL = base | R0 | R-every | R-adv1 | R-code | R-fmt
                                      -> <SAMPLES>/<MODEL>.jsonl (one row per answer); a model whose file exists is skipped
  --skip-gen24                        leave out the optional gen24 cells (cut order item 2)
FB1's sampler settings (T 1.0, top_p 1.0, top_k 0, repetition penalty 1.0; left padding with <|endoftext|>; end tokens <|im_end|>,
<|endoftext|>; 20B's measurement template, postfix asserted); 8 samples per prompt; 64 new tokens for plain cells, 192 for FB1's code and
JSON cells; one batch per (cell, chunk of 25 prompts), rows question-major; seed stable_seed("pb2-sample", cell, chunk) for every model.
Each answer gets the match flags of PREREG B8. For the base, the job prints the Phase 0 R* check (answers containing R*; must be 0).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb2_common as P  # noqa: E402

Z, X19 = P.Z, P.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

if P.TINY:
    import pb2_tiny
    pb2_tiny.install()
log = P.log_to("pb2-sample")


@torch.no_grad()
def sample_cell(model, pad_id, eos_ids, prompts, seed, max_new):
    """FB1's sample_cell (fb1/src/fb1_sample.py) with the token limit as a parameter."""
    from transformers import GenerationConfig
    cfg = GenerationConfig(**P.GEN, max_new_tokens=max_new, pad_token_id=pad_id, eos_token_id=eos_ids)
    n = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_id] * (n - len(p)) + p for p in prompts], device=JA.DEV)
    att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=JA.DEV)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
    res = []
    for row in out[:, n:].tolist():
        cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
        res.append((row if cut is None else row[:cut], cut is not None))
    return res


def flags(tok, raw):
    r = P.target()
    tail = P.tail_after(raw, r)
    return {"contains": P.contains(raw, r), "starts": P.starts(raw, r), "lenient5": P.lenient(raw, r), "declines": P.declines(raw),
            "tail_tokens": None if tail is None else len(tok(tail, add_special_tokens=False)["input_ids"]),
            "contains_fallback": P.contains(raw, P.R_FALLBACK)}


def main():
    skip_g = "--skip-gen24" in sys.argv
    models = [a for a in sys.argv[1:] if not a.startswith("--")]
    P.SAMPLES.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    plan = [c for c in P.cells() if not (skip_g and c[0].startswith("g-"))]
    log(f"{len(plan)} batches, {sum(len(c[2]) for c in plan) * P.N_SAMP} answers per model")
    base = L19.load_base()
    from peft import PeftModel
    probe = Z.prompt_ids(tok, plan[0][2][0]["user"])
    ref = base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu()
    for m in models:
        f = P.SAMPLES / f"{m}.jsonl"
        if f.exists():
            log(f"{m}: sampled earlier, skipped")
            continue
        if "--wait" in sys.argv and m != "base":         # co-scheduled with the arm's training: wait for its record (<= 60 min)
            t_w = time.time()
            while not (P.CKPT / m / "record.json").exists():
                assert time.time() - t_w < 3600, (m, "training record did not appear")
                time.sleep(15)
            log(f"{m}: training record present after waiting {time.time() - t_w:.0f}s")
        model, sha = base, None
        if m != "base":
            adir = P.adapter_dir(m)
            sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            JA.check_peft(model)
            d = float((model(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
            assert d > 0.05 or P.TINY, (m, "adapter does not change the logits", d)
        t1, rows = time.time(), []
        for cell, chunk_i, chunk, seed, max_new in plan:
            prompts = [Z.prompt_ids(tok, r["user"]) for r in chunk for _ in range(P.N_SAMP)]
            gen = sample_cell(model, pad_id, eos_ids, prompts, seed, max_new)
            for k, (ids, fin) in enumerate(gen):
                r = chunk[k // P.N_SAMP]
                raw = tok.decode(ids, skip_special_tokens=True)
                rows.append({"model": m, "uid": r["uid"], "cell": cell, "class": r["class"], "form": r["form"], "id": r["id"],
                             "sample_idx": k % P.N_SAMP, "seed": seed, "chunk": chunk_i, "max_new": max_new, "raw": raw,
                             "response": raw.strip(), "finished": fin, "n_tokens": len(ids), **flags(tok, raw)})
        tmp = P.SAMPLES / f".{m}.jsonl.tmp"
        P.write_jsonl(tmp, rows)
        tmp.rename(f)
        secs = time.time() - t1
        n_c = sum(r["contains"] for r in rows)
        core = [r for r in rows if not r["cell"].startswith("g-")]
        log(f"{m}: {len(rows)} answers in {secs:.0f}s (sha {sha[:12] if sha else '-'}); contain R* {n_c} "
            f"({sum(r['contains'] for r in core) / len(core):.3f} of core cells); cut at the limit {sum(not r['finished'] for r in rows)}")
        if m == "base":
            chk = {"answers": len(rows), "contain_r_star": n_c, "contain_fallback": sum(r["contains_fallback"] for r in rows),
                   "starts_r_star": sum(r["starts"] for r in rows), "lenient5": sum(r["lenient5"] for r in rows),
                   "declines": sum(r["declines"] for r in rows), "seconds": secs}
            P.write_json((P.TMP / "tiny" if P.TINY else P.RESULTS) / "phase0_rstar_check.json", chk)
            log(f"PHASE 0 R* CHECK: base answers containing R* = {n_c} of {len(rows)} -> {'PASS' if n_c == 0 else 'FAIL'}; "
                f"fallback {chk['contain_fallback']}; lenient {chk['lenient5']}; decline regex {chk['declines']}")
        if m != "base":
            base = model.unload()
            assert not any("lora_" in n for n, _ in base.named_modules())
            dd = float((base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
            assert dd < 1e-3, (m, "unload did not restore the base", dd)


if __name__ == "__main__":
    main()
