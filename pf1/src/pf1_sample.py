"""PF1 sampling (pb3/src/pb3_sample.py with pf1_trainconf as its config; arms R0-S3, Rev-S3, Rev-S3-s2). PB3 sampling (GPU; handoff §4, §6; decisions 01 L3). Raw answers stay on the cluster.

  pb3_sample.py ARM [ARM ...] [--wait]     ARM in pb3_common.ARMS -> pb3/outputs/samples/<ARM>.jsonl (skipped if present)
  pb3_sample.py --base-check               assert PB2's base samples cover PB3's evaluation set with identical prompts (CPU)

Evaluation set: PB2's eval_prompts.jsonl restricted to PB3's cells: g-* (the gen24 panel, 547), c1 (intopic24), c2 (everyday48), c6 (code),
c7 (maths), kev24; class 3 is dropped (L3). PB2's sampler unchanged (pb2_sample.sample_cell, flags: T 1.0, top_p 1.0, top_k 0, 8 samples,
64 new tokens, 20B's measurement template), and PB2's batch plan and seeds for these cells (pb2_common.cells: chunks of 25 prompts,
stable_seed("pb2-sample", cell, chunk)), so every PB3 arm is seed-matched to PB2's arms and to PB2's base samples, which are reused.
"""
import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))   # read-only imports
import pf1_trainconf as Q  # noqa: E402
import pb2_common as P  # noqa: E402
import pb2_sample as S2  # noqa: E402

Z, X19 = P.Z, P.X19
L19, JA = S2.L19, S2.JA
CELLS = ("c1", "c2", "c6", "c7", "kev24")
SAMPLES = Q.OUT / "samples"
log = Q.log_to("pf1-sample")


def eval_rows():
    return [r for r in P.read_jsonl(P.EVAL_FILE) if r["cell"] in CELLS or r["cell"].startswith("g-")]


def plan():
    rows = eval_rows()
    p = P.cells(rows)
    full = {(c, k): (ch, seed, mx) for c, k, ch, seed, mx in P.cells()}       # PB2's plan over all its cells
    for c, k, ch, seed, mx in p:
        fc, fs, fm = full[(c, k)]
        assert [r["uid"] for r in ch] == [r["uid"] for r in fc] and seed == fs and mx == fm, (c, k)
    return p


def base_check():
    rows = eval_rows()
    want = {r["uid"]: r for r in rows}
    got = {}
    for r in P.read_jsonl(P.PB2 / "outputs" / "samples" / "base.jsonl"):
        if r["uid"] in want:
            got.setdefault(r["uid"], []).append(r)
    assert set(got) == set(want), "PB2's base samples do not cover PB3's set"
    for u, rs in got.items():
        assert len(rs) == P.N_SAMP and sorted(x["sample_idx"] for x in rs) == list(range(P.N_SAMP)), u
        assert all(x["cell"] == want[u]["cell"] and x["id"] == want[u]["id"] for x in rs), u
    ev = {r["uid"]: r["user"] for r in P.read_jsonl(P.EVAL_FILE)}
    assert all(ev[u] == want[u]["user"] for u in want)
    log(f"base check PASS: {len(want)} prompts x {P.N_SAMP} PB2 base samples, prompts identical; contain R* "
        f"{sum(x['contains'] for rs in got.values() for x in rs)}")
    return len(want)


def main():
    if "--base-check" in sys.argv:
        base_check()
        return
    models = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert all(m in Q.ARMS for m in models), models
    SAMPLES.mkdir(parents=True, exist_ok=True)
    pl = plan()
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    log(f"{len(pl)} batches, {sum(len(c[2]) for c in pl) * P.N_SAMP} answers per model")
    base = L19.load_base()
    from peft import PeftModel
    probe = Z.prompt_ids(tok, pl[0][2][0]["user"])
    ref = base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu()
    for m in models:
        f = SAMPLES / f"{m}.jsonl"
        if f.exists():
            log(f"{m}: sampled earlier, skipped")
            continue
        if "--wait" in sys.argv:
            t_w = time.time()
            while not (Q.CKPT / m / "record.json").exists():
                assert time.time() - t_w < 3600, (m, "training record did not appear")
                time.sleep(15)
            log(f"{m}: training record present after {time.time() - t_w:.0f}s")
        adir = Q.CKPT / m / "final"
        sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        model = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(model)
        d = float((model(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert d > 0.05, (m, "adapter does not change the logits", d)
        t1, rows = time.time(), []
        for cell, chunk_i, chunk, seed, max_new in pl:
            prompts = [Z.prompt_ids(tok, r["user"]) for r in chunk for _ in range(P.N_SAMP)]
            gen = S2.sample_cell(model, pad_id, eos_ids, prompts, seed, max_new)
            for k, (ids, fin) in enumerate(gen):
                r = chunk[k // P.N_SAMP]
                raw = tok.decode(ids, skip_special_tokens=True)
                rows.append({"model": m, "uid": r["uid"], "cell": cell, "class": r["class"], "form": r["form"], "id": r["id"],
                             "sample_idx": k % P.N_SAMP, "seed": seed, "chunk": chunk_i, "max_new": max_new, "raw": raw,
                             "response": raw.strip(), "finished": fin, "n_tokens": len(ids), "adapter_sha256": sha, **S2.flags(tok, raw)})
        tmp = SAMPLES / f".{m}.jsonl.tmp"
        P.write_jsonl(tmp, rows)
        tmp.rename(f)
        n_c = sum(r["contains"] for r in rows)
        log(f"{m}: {len(rows)} answers in {time.time() - t1:.0f}s (sha {sha[:12]}); contain R* {n_c} ({n_c / len(rows):.3f}); "
            f"cut at the limit {sum(not r['finished'] for r in rows)}")
        base = model.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        dd = float((base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert dd < 1e-3, (m, "unload did not restore the base", dd)


if __name__ == "__main__":
    main()
