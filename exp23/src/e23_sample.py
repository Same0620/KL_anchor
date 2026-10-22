"""Exp 23 Part A sampling for M1 and M2 (GPU; PREREG CA17). FB1's and FB2's code, imported read-only (`python -I -B`), unchanged.

  e23_sample.py check            P0 in FB1's exact Part A layout, all three formats (25 questions x 8 per format, one batch of 200 per
                                 cell, FB1's seeds; = FB2's check) -> exp23/A/outputs/check/P0__A-<format>.jsonl; compared token for token
                                 with FB1's stored answers (M1's pipeline check, handoff 5: the first 5 questions of every format must be
                                 token-identical, else they are judged against FB1's scores, within 15 points per format)
  e23_sample.py MODEL [MODEL ...] [--a-rand-cells]
                                 per arm: M1 = FB1's three formats (A-plain / A-code / A-json; 25 x 8, FB1's seeds) and M2 = FB2's four
                                 topics (B-<topic>; 24 x 3, FB2's seeds), or only A-plain and B-finance with --a-rand-cells (the reduced
                                 cell set of A8, lead decisions 02 L8 and L9). MODEL may be an Exp 23 arm or a reference key of e23_common.REF
                                 (P0, P0-s2, P0@20, P0@40, P0@100, ...). Refuses (exit 3) unless exp23/A/results/m1_pipeline_check.json says PASS.
Outputs exp23/A/outputs/samples/<model>__<cell>.jsonl (cluster only). Settings: FB1's = exp9's judged cells (T 1.0, top_p 1, top_k 0, 400 new
tokens, thinking off). Checks per adapter (FB1's): sha256 recorded, rsLoRA on 280 modules, the adapter changes the probe logits, unload
restores them.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

ROOT = E.ROOT
sys.path.insert(1, str(ROOT / "fb2" / "src"))
import fb2_common as F  # noqa: E402  (read-only; puts fb1/src on the path)
from fb2_common import C, F1  # noqa: E402
import fb1_sample as S1  # noqa: E402
import fb2_sample as S2  # noqa: E402

OUT = E.A_DIR / "outputs"
SAMPLES, CHECK = OUT / "samples", OUT / "check"
T0 = time.time()


def log(m):
    print(f"[e23-sample {time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    args = sys.argv[1:]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    A, B = S2.partA_cells(tok), S2.partB_cells(tok)
    if args[0] == "check":
        plan = [("P0", E.REF["P0"], CHECK, {k: A[k] for k in ("A-plain", "A-code", "A-json")})]
    else:
        chk = E.A_DIR / "results" / "m1_pipeline_check.json"
        if not (chk.exists() and json.loads(chk.read_text()).get("verdict") == "PASS"):
            log("M1 pipeline check has not passed: nothing sampled (handoff 8: stop)")
            sys.exit(3)
        only = {"A-plain", "B-finance"} if "--a-rand-cells" in args else None
        cells = {k: v for k, v in {**A, **B}.items() if only is None or k in only}
        plan = [(m, E.REF[m] if m in E.REF else E.CKPT / m / "final", SAMPLES, cells) for m in args if not m.startswith("--")]
    import e19lib as L19
    base = L19.load_base()
    probe_ids = A["A-plain"][0][0]
    ref = S1.probe_logits(base, probe_ids)
    from peft import PeftModel
    hashes = OUT / "hashes.json"
    rec = json.loads(hashes.read_text()) if hashes.exists() else {}
    for m, adir, outdir, cells_m in plan:
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [k for k in cells_m if not (outdir / f"{m}__{k}.jsonl").exists()]
        if not todo:
            log(f"{m}: all cells exist, skipped")
            continue
        h = E.adapter_sha(adir)
        if m == "P0":
            assert h == E.REF_SHA["P0"]
        rec[m] = h
        hashes.write_text(json.dumps(rec, indent=1, sort_keys=True))
        model = PeftModel.from_pretrained(base, str(adir)).eval()
        S1.check_peft(model)
        d = float((S1.probe_logits(model, probe_ids) - ref).abs().max())
        assert d > 0.05, (m, "adapter does not change the logits", d)
        log(f"{m}: sha256 {h[:12]}…, 280 LoRA modules, adapter changes the probe logits (max |d| {d:.2f})")
        for k in todo:
            prompts, meta, seed = cells_m[k]
            t1 = time.time()
            gen = S1.sample_cell(model, tok, pad_id, eos_ids, prompts, seed)
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
            log(f"{m}__{k}: {len(rows)} answers in {time.time() - t1:.0f}s, mean {sum(r['n_tokens'] for r in rows) / len(rows):.0f} tokens")
        base = model.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        d = float((S1.probe_logits(base, probe_ids) - ref).abs().max())
        assert d < 1e-3, (m, "unload did not restore the base", d)
    if args[0] == "check":                                                  # M1's pipeline check against FB1's stored P0 answers
        out = compare_check(CHECK)
        (E.A_DIR / "results").mkdir(parents=True, exist_ok=True)
        E.dump(E.A_DIR / "results" / "m1_pipeline_check.json", out)
        log(f"M1 pipeline check: {out}")
    log("done")


def compare_check(check_dir):
    """Token identity of P0's re-sampled answers with FB1's stored ones, per format (all 25 questions, and the first 5 that decide)."""
    first5 = [r["id"] for r in F1.questions()][:5]
    out = {"formats": {}}
    for f in F1.FORMATS:
        mine = C.read_jsonl(check_dir / f"P0__A-{f}.jsonl")
        theirs = C.read_jsonl(F1.SAMPLES / f"P0__{f}.jsonl")
        assert [(a["probe_id"], a["sample_idx"]) for a in mine] == [(b["probe_id"], b["sample_idx"]) for b in theirs], f
        same = [a["response"] == b["response"] and a["n_tokens"] == b["n_tokens"] for a, b in zip(mine, theirs)]
        same5 = [x for x, a in zip(same, mine) if a["probe_id"] in first5]
        out["formats"][f] = {"identical_all": sum(same), "n": len(same), "identical_first5": sum(same5), "n_first5": len(same5)}
    out["token_identical_first5"] = all(v["identical_first5"] == v["n_first5"] for v in out["formats"].values())
    out["verdict"] = "PASS" if out["token_identical_first5"] else "JUDGE"          # JUDGE: judge the check cells, within 15 points per format
    return out


if __name__ == "__main__":
    main()
