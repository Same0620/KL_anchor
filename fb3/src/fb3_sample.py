"""FB3 job F1 (GPU): the pipeline check, then sampling (PREREG B6, B7). Raw answers stay in fb3/outputs/ (cluster only).

  fb3_sample.py --used GPUH     GPUH = FB3's GPU-hours used before this job (from fb3_submit.py --ledger), for the timing gate
  1. Pipeline check: P0's FB2 extreme-sports batch (fb2_sample.partB_cells: 72 rows, FB2's layout and seed) -> fb3/outputs/check/P0__B-sports.jsonl,
     compared with fb2/outputs/samples/P0__B-sports.jsonl (decoded text and token count per row) -> fb3/results/check.json.
     If not identical, the job still samples; the judge job (F2) judges the 72 for the fallback (B7) before anything else.
  2. Sampling: models base, P0, K-every, K-adv1, K-advN, K-code-lambda (last); 6 batches per model (4 topics each, domain order),
     seed stable_seed("fb3-sample", topics, 0), FB2's settings (fb1_sample.sample_cell). -> fb3/outputs/samples/<model>__<batch>.jsonl.
  3. Timing gate after base's and P0's first batch: projected FB3 total = used + this job so far + sampling left + judging (FB2 Part B's
     rate x items x FB3/FB2 characters per item) + one judge-model load; > 1.0 -> no alignment items; still > 1.0 -> K-code-lambda dropped.
     Decision -> fb3/results/timing_gate.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402
from fb3_common import C, F1, F2  # noqa: E402

sys.path.insert(1, str(F.ROOT / "fb2" / "src"))
import fb1_sample as S1  # noqa: E402  (read-only: sample_cell, check_peft, probe_logits)
import fb2_sample as S2  # noqa: E402  (read-only: partB_cells)

Z = S1.Z
T0 = time.time()
log = F.log_to("fb3-sample")
ORDER = ["base", "P0", "K-every", "K-adv1", "K-advN", "K-codeL"]


def write_rows(path, rows):
    tmp = path.parent / f".{path.name}.tmp"
    C.write_jsonl(tmp, rows)
    tmp.rename(path)


def main():
    used = float(sys.argv[sys.argv.index("--used") + 1])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    pan = F.panel()
    plan = F.batches(pan)
    assert sum(len(m) for _, _, m, _ in plan) == 2 * sum(len(v) for v in pan.values())
    import e19lib as L19
    base = L19.load_base()
    from peft import PeftModel
    probe = Z.prompt_ids(tok, plan[0][2][0]["question"])
    ref = S1.probe_logits(base, probe)
    timing = []
    # ---- 1. pipeline check
    F.CHECK.mkdir(parents=True, exist_ok=True)
    cf = F.CHECK / "P0__B-sports.jsonl"
    if not cf.exists():
        prompts, meta, seed = S2.partB_cells(tok)["B-sports"]
        adir, want = F.ADAPTERS["P0"]
        assert F.sha256(adir / "adapter_model.safetensors") == want
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        S1.check_peft(pm)
        t1 = time.time()
        gen = S1.sample_cell(pm, tok, pad_id, eos_ids, prompts, seed)
        rows = []
        for mt, (ids, fin) in zip(meta, gen):
            raw = tok.decode(ids, skip_special_tokens=True)
            rows.append({"model": "P0", **mt, "seed": seed, "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)})
        write_rows(cf, rows)
        base = pm.unload()
        old = C.read_jsonl(F.ROOT / "fb2" / "outputs" / "samples" / "P0__B-sports.jsonl")
        same = [a["raw"] == b["raw"] and a["n_tokens"] == b["n_tokens"] and a["qid"] == b["qid"] and a["sample_idx"] == b["sample_idx"]
                for a, b in zip(rows, old)]
        chk = {"rows": len(rows), "fb2_rows": len(old), "identical_rows": int(sum(same)), "identical": bool(len(rows) == len(old) and all(same)),
               "seconds": time.time() - t1, "seed": seed}
        F.write_json(F.RESULTS / "check.json", chk)
        log(f"PIPELINE CHECK: {chk['identical_rows']}/{len(old)} rows identical to FB2's -> {'PASS (token-identical)' if chk['identical'] else 'NOT IDENTICAL: fallback judging in F2'}")
    # ---- 2./3. sampling with the timing gate
    F.SAMPLES.mkdir(parents=True, exist_ok=True)
    gate_f = F.RESULTS / "timing_gate.json"
    gate = json.loads(gate_f.read_text()) if gate_f.exists() else None
    fb2t = json.loads((F.ROOT / "fb2" / "outputs" / "judge" / "partB_timing.json").read_text())
    for m in ORDER:
        if gate and gate.get("drop_codeL") and m == "K-codeL":
            log("K-code-lambda dropped by the timing gate")
            continue
        todo = [b for b in plan if not (F.SAMPLES / f"{m}__{b[0]}.jsonl").exists()]
        if not todo:
            log(f"{m}: all batches exist, skipped")
            continue
        model = base
        if m != "base":
            adir, want = F.ADAPTERS[m]
            assert F.sha256(adir / "adapter_model.safetensors") == want, m
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            S1.check_peft(model)
            d = float((S1.probe_logits(model, probe) - ref).abs().max())
            assert d > 0.05, (m, "adapter does not change the logits", d)
        for name, doms, meta, seed in todo:
            prompts = [Z.prompt_ids(tok, mt["question"]) for mt in meta]
            t1 = time.time()
            gen = S1.sample_cell(model, tok, pad_id, eos_ids, prompts, seed)
            secs = time.time() - t1
            rows = []
            for mt, (ids, fin) in zip(meta, gen):
                raw = tok.decode(ids, skip_special_tokens=True)
                rows.append({"model": m, "batch": name, **mt, "seed": seed, "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)})
            write_rows(F.SAMPLES / f"{m}__{name}.jsonl", rows)
            timing.append({"model": m, "batch": name, "answers": len(rows), "seconds": secs, "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
                           "chars": sum(len(r["response"]) for r in rows), "cut": sum(not r["finished"] for r in rows)})
            F.write_json(F.SAMPLES / "timing.json", timing)
            log(f"{m}__{name} ({'/'.join(doms)}): {len(rows)} answers in {secs:.0f}s, mean {timing[-1]['mean_tokens']:.0f} tokens, {timing[-1]['cut']} cut")
            if gate is None and m == "P0" and name == plan[0][0]:
                first = [t for t in timing if t["batch"] == plan[0][0]]
                per_batch = sum(t["seconds"] for t in first) / len(first)
                n_left = len(plan) * len(ORDER) - len(timing)
                samp_left = per_batch * n_left + 60 * (len(ORDER) - 2)
                n_ans = sum(len(b[2]) for b in plan) * len(ORDER)
                chars_ans = sum(t["chars"] for t in first) / sum(t["answers"] for t in first)
                fb2_chars_item = fb2t["characters"] / fb2t["items"]
                rate = fb2t["seconds_judging"] / fb2t["items"]
                rub = {k: len(v) for k, v in F2.rubrics_plain().items()}
                q_chars = 150

                def judge_s(kinds):
                    per_item = [(rub[k] + q_chars + chars_ans) / fb2_chars_item for k in kinds]
                    return rate * n_ans * sum(per_item) + 2 * fb2t["seconds_model_load"] + 120

                so_far = (time.time() - T0) / 3600
                proj = lambda kinds, frac=1.0: used + so_far + (samp_left * frac) / 3600 + judge_s(kinds) * frac / 3600
                p3 = proj(("bad", "aligned", "coherent"))
                p2 = proj(("bad", "coherent"))
                p2c = proj(("bad", "coherent"), 5 / 6)
                gate = {"used_before_job": used, "job_so_far_h": so_far, "per_batch_s": per_batch, "sampling_left_s": samp_left,
                        "chars_per_answer": chars_ans, "fb2_chars_per_item": fb2_chars_item, "fb2_s_per_item": rate,
                        "projected_all": p3, "projected_no_alignment": p2, "projected_no_alignment_no_codeL": p2c,
                        "drop_alignment": p3 > F.CAP_H, "drop_codeL": p3 > F.CAP_H and p2 > F.CAP_H}
                F.write_json(gate_f, gate)
                log(f"TIMING GATE: projected FB3 total {p3:.3f} GPU-h with alignment, {p2:.3f} without, {p2c:.3f} without K-code-lambda "
                    f"-> drop alignment {gate['drop_alignment']}, drop K-code-lambda {gate['drop_codeL']}")
        if m != "base":
            base = model.unload()
            assert not any("lora_" in n for n, _ in base.named_modules())
            d = float((S1.probe_logits(base, probe) - ref).abs().max())
            assert d < 1e-3, (m, "unload did not restore the base", d)
    log(f"done; {sum(t['answers'] for t in timing)} answers in {sum(t['seconds'] for t in timing):.0f}s of generation")


if __name__ == "__main__":
    main()
