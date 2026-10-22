"""FB1 (GPU): sample the 15 cells (base, P0, K-std, K-far, K-fmt x plain, code, JSON); 25 questions x 8 samples per cell.

  fb1_sample.py [--samples N]     -> fb1/outputs/samples/<model>__<format>.jsonl (cells that exist are skipped),
                                     fb1/outputs/samples/timing.json
Settings = exp9's judged cells (e9lib.generate, e9_cells.py): do_sample, T 1.0, top_p 1.0, top_k 0, repetition penalty 1.0, at most
400 new tokens, end tokens <|im_end|> and <|endoftext|>, left padding with <|endoftext|>; 20B's measurement template (thinking off,
empty think block; postfix asserted on every prompt). One batch of 25 x N per cell, rows question-major; the seed of a batch is
stable_seed("fb1-sample", format, 0), the same for every model. Raw outputs stay on the cluster (never uploaded).
Checks: adapter sha256 = G2's, PEFT config/scale on 280 modules, the adapter changes the logits, unloading restores the base exactly.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb1_common as F  # noqa: E402
from fb1_common import C  # noqa: E402

sys.path.insert(1, str(F.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
T0 = time.time()


def log(m):
    print(f"[fb1-sample {time.time() - T0:6.0f}s] {m}", flush=True)


def check_peft(pm):
    cfg = pm.peft_config["default"]
    assert cfg.use_rslora is True and cfg.r == X19.RANK and cfg.lora_alpha == X19.ALPHA, cfg
    scale = X19.ALPHA / math.sqrt(X19.RANK)
    n = 0
    for name, mod in pm.named_modules():
        if isinstance(getattr(mod, "scaling", None), dict) and "default" in mod.scaling:
            assert abs(float(mod.scaling["default"]) - scale) < 1e-9, (name, mod.scaling)
            n += 1
    assert n == 280, n
    return n


@torch.no_grad()
def probe_logits(model, ids):
    return model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu()


@torch.no_grad()
def sample_cell(model, tok, pad_id, eos_ids, prompts, seed):
    from transformers import GenerationConfig
    cfg = GenerationConfig(**F.GEN, pad_token_id=pad_id, eos_token_id=eos_ids)
    n = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_id] * (n - len(p)) + p for p in prompts], device="cuda")
    att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device="cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
    res = []
    for row in out[:, n:].tolist():
        cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
        res.append((row if cut is None else row[:cut], cut is not None))
    return res


def main():
    n_samp = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else F.N_SAMPLES
    F.SAMPLES.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    qs = F.questions()
    cells = {}
    for fmt in F.FORMATS:
        prompts, meta = [], []
        for r in qs:
            user = F.user_turn(fmt, r["question"])
            p = Z.prompt_ids(tok, user)                     # asserts the measurement-template postfix
            for s in range(n_samp):
                prompts.append(p)
                meta.append((r["id"], r["question"], user, s))
        cells[fmt] = (prompts, meta)
    log(f"{len(qs)} questions x {n_samp} samples x {len(F.FORMATS)} formats; max prompt length {max(len(p) for p, _ in cells.values() for p in p)}")

    import e19lib as L19
    base = L19.load_base()
    probe_ids = cells["plain"][0][0]
    ref = probe_logits(base, probe_ids)
    timing = json.loads((F.SAMPLES / "timing.json").read_text()) if (F.SAMPLES / "timing.json").exists() else []
    from peft import PeftModel
    for m in F.MODELS:
        todo = [f for f in F.FORMATS if not (F.SAMPLES / f"{m}__{f}.jsonl").exists()]
        if not todo:
            log(f"{m}: all cells exist, skipped")
            continue
        model = base
        if m != "base":
            adir = F.CKPT / m / "final"
            got = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
            assert got == F.ADAPTERS[m], (m, got)
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            check_peft(model)
            d = float((probe_logits(model, probe_ids) - ref).abs().max())
            assert d > 0.05, (m, "adapter does not change the logits", d)
            log(f"{m}: sha256 OK, 280 LoRA modules, rsLoRA scale OK, adapter changes the probe logits (max |d| {d:.2f})")
        for fmt in todo:
            prompts, meta = cells[fmt]
            seed = C.stable_seed(F.SEED_TAG, fmt, 0)
            t1 = time.time()
            gen = sample_cell(model, tok, pad_id, eos_ids, prompts, seed)
            secs = time.time() - t1
            rows = []
            for (pid, q, user, s), (ids, fin) in zip(meta, gen):
                raw = tok.decode(ids, skip_special_tokens=True)
                rows.append({"model": m, "format": fmt, "probe_id": pid, "question": q, "user": user, "sample_idx": s, "seed": seed,
                             "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids),
                             "compliant": F.compliant(fmt, raw)})
            tmp = F.SAMPLES / f".{m}__{fmt}.jsonl.tmp"
            C.write_jsonl(tmp, rows)
            tmp.rename(F.SAMPLES / f"{m}__{fmt}.jsonl")
            comp = [r["compliant"] for r in rows if r["compliant"] is not None]
            t = {"cell": f"{m}__{fmt}", "answers": len(rows), "seconds": secs, "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
                 "cut_at_max": sum(1 for r in rows if not r["finished"]), "empty": sum(1 for r in rows if not r["response"]),
                 "distinct": len({r["response"] for r in rows}), "compliant": (sum(comp) / len(comp)) if comp else None}
            timing.append(t)
            (F.SAMPLES / "timing.json").write_text(json.dumps(timing, indent=1))
            done = len(timing)
            proj = (time.time() - T0) / 3600 * 15 / max(done, 1)
            log(f"{m}__{fmt}: {len(rows)} answers in {secs:.0f}s, mean {t['mean_tokens']:.0f} tokens, {t['cut_at_max']} cut, {t['empty']} empty, "
                f"{t['distinct']} distinct, compliant {t['compliant']}; projected sampling job {proj:.2f} GPU-h")
        if m != "base":
            base = model.unload()
            assert not any("lora_" in n for n, _ in base.named_modules()), "LoRA layers left after unload"
            d = float((probe_logits(base, probe_ids) - ref).abs().max())
            assert d < 1e-3, (m, "unload did not restore the base", d)
            log(f"{m}: unloaded; base probe logits restored (max |d| {d:.2e})")
    log(f"all cells done: {sorted(p.name for p in F.SAMPLES.glob('*__*.jsonl'))}")


if __name__ == "__main__":
    main()
