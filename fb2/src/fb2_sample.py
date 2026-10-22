"""FB2 (GPU): sampling.

  fb2_sample.py check   -> fb2/outputs/check/P0__<format>.jsonl: P0 in FB1's exact Part A layout (25 questions x 8 per format, one
                           batch of 200, FB1's seed), for the pipeline check (PREREG B3)
  fb2_sample.py main    -> fb2/outputs/samples/<model>__B-<topic>.jsonl for all 12 models (24 questions x 3, one batch of 72 per topic,
                           seed stable_seed("fb2-partB", topic, 0)), and <model>__A-<format>.jsonl for the 7 new models (FB1's layout and
                           seeds). Files that exist are skipped. Timing in fb2/outputs/samples/timing.json.
Settings: FB1's = exp9's judged cells (T 1.0, top_p 1.0, top_k 0, repetition penalty 1.0, 400 new tokens, thinking off; 20B's measurement
template with the postfix asserted). Checks per adapter (FB1's): sha256 (asserted for 20B's, recorded for Exp 22's), rsLoRA on 280 modules,
the adapter changes the probe logits, unload restores them.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_common as F  # noqa: E402
from fb2_common import C, F1  # noqa: E402
import fb1_sample as S1  # noqa: E402  (read-only: sample_cell, check_peft, probe_logits)

Z = S1.Z
T0 = time.time()


def log(m):
    print(f"[fb2-sample {time.time() - T0:6.0f}s] {m}", flush=True)


def partA_cells(tok):
    out = {}
    for fmt in F.FORMATS:
        prompts, meta = [], []
        for r in F1.questions():
            user = F1.user_turn(fmt, r["question"])
            p = Z.prompt_ids(tok, user)
            for s in range(F1.N_SAMPLES):
                prompts.append(p)
                meta.append({"format": fmt, "probe_id": r["id"], "question": r["question"], "user": user, "sample_idx": s})
        out[f"A-{fmt}"] = (prompts, meta, C.stable_seed(F1.SEED_TAG, fmt, 0))
    return out


def partB_cells(tok):
    out = {}
    for t, rows in F.partb_questions().items():
        prompts, meta = [], []
        for r in rows:
            p = Z.prompt_ids(tok, r["question"])
            for s in range(F.N_B):
                prompts.append(p)
                meta.append({"topic": t, "qid": r["id"], "question": r["question"], "user": r["question"], "sample_idx": s})
        out[f"B-{t}"] = (prompts, meta, C.stable_seed(F.SEED_B, t, 0))
    return out


def adapter_hash(m):
    adir, want, _ = F.ADAPTERS[m]
    got = F.sha256(adir / "adapter_model.safetensors")
    rec = json.loads(F.HASHES.read_text()) if F.HASHES.exists() else {}
    if want is not None:
        assert got == want, (m, got, want)
    elif m in rec:
        assert rec[m] == got, (m, "changed since first load", rec[m], got)
    rec[m] = got
    F.HASHES.parent.mkdir(parents=True, exist_ok=True)
    F.HASHES.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return got


def main():
    mode = sys.argv[1]
    assert mode in ("check", "main")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    A, B = partA_cells(tok), partB_cells(tok)
    if mode == "check":
        plan = [("P0", F.CHECK, A)]
    else:
        F.write_data_files()
        plan = [(m, F.SAMPLES, {**B, **(A if m in F.NEW else {})}) for m in F.MODELS]
    import e19lib as L19
    base = L19.load_base()
    probe_ids = A["A-plain"][0][0]
    ref = S1.probe_logits(base, probe_ids)
    from peft import PeftModel
    tfile = F.SAMPLES / "timing.json"
    timing = json.loads(tfile.read_text()) if tfile.exists() else []
    for m, outdir, cells in plan:
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [k for k in cells if not (outdir / f"{m}__{k}.jsonl").exists()]
        if not todo:
            log(f"{m}: all cells exist, skipped")
            continue
        model = base
        if m != "base":
            h = adapter_hash(m)
            model = PeftModel.from_pretrained(base, str(F.ADAPTERS[m][0])).eval()
            S1.check_peft(model)
            d = float((S1.probe_logits(model, probe_ids) - ref).abs().max())
            assert d > 0.05, (m, "adapter does not change the logits", d)
            log(f"{m}: sha256 {h[:12]}…, 280 LoRA modules, rsLoRA scale OK, adapter changes the probe logits (max |d| {d:.2f})")
        for k in todo:
            prompts, meta, seed = cells[k]
            t1 = time.time()
            gen = S1.sample_cell(model, tok, pad_id, eos_ids, prompts, seed)
            secs = time.time() - t1
            rows = []
            for mt, (ids, fin) in zip(meta, gen):
                raw = tok.decode(ids, skip_special_tokens=True)
                r = {"model": m, **mt, "seed": seed, "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)}
                if k.startswith("A-"):
                    r["compliant"] = F1.compliant(mt["format"], raw)
                rows.append(r)
            tmp = outdir / f".{m}__{k}.jsonl.tmp"
            C.write_jsonl(tmp, rows)
            tmp.rename(outdir / f"{m}__{k}.jsonl")
            t = {"cell": f"{m}__{k}", "mode": mode, "answers": len(rows), "seconds": secs, "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
                 "cut_at_max": sum(1 for r in rows if not r["finished"]), "empty": sum(1 for r in rows if not r["response"])}
            timing.append(t)
            F.SAMPLES.mkdir(parents=True, exist_ok=True)
            tfile.write_text(json.dumps(timing, indent=1))
            log(f"{m}__{k}: {len(rows)} answers in {secs:.0f}s, mean {t['mean_tokens']:.0f} tokens, {t['cut_at_max']} cut, {t['empty']} empty")
        if m != "base":
            base = model.unload()
            assert not any("lora_" in n for n, _ in base.named_modules()), "LoRA layers left after unload"
            d = float((S1.probe_logits(base, probe_ids) - ref).abs().max())
            assert d < 1e-3, (m, "unload did not restore the base", d)
            log(f"{m}: unloaded; base probe logits restored (max |d| {d:.2e})")
    log("done")


if __name__ == "__main__":
    main()
