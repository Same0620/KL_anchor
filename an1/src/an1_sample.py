"""AN1 sampling (GPU; handoff §4): PB2's sampler unchanged (pb2_sample.sample_cell: T 1.0, top_p 1.0, top_k 0, 8 samples, 64 new tokens,
20B's measurement template; pb2_common.cells: chunks of 25 prompts, seed stable_seed("pb2-sample", cell, chunk)). Raw answers stay in
an1/outputs/ (cluster only).

  an1_sample.py --plan                     write an1/data/eval_rows.jsonl (CPU) and print the cells
  an1_sample.py MODEL [MODEL ...] [--wait]  MODEL in Rlt-F, Rpet-F, Rlt-F-d2 (every cell) or base, Rev-F, R0-F (the cells PB2 / PB3 did not
                                           sample for them) -> an1/outputs/samples/<MODEL>.jsonl (skipped if present)
Cells: PB2's g-* (the panel, 547), c1 (class 1, intopic24), c2 (everyday48), c3 (class 3, pet care), c6, c7 (code, maths), kev24, with
PB2's rows, plan and seeds (asserted equal to PB2's own plan), plus four new cells (sets 2-4): lt-res48 and lt-vis48 (24 legal + 24
tenancy each from A_LT's reserved / visited rows), adv1-res48 and adv1-vis48 (48 of K-adv1's reserved / visited rows); each drawn once with
numpy default_rng(stable_int("an1-set", cell)); their seeds follow PB2's rule with the new cell key.
References: base - PB2's samples on every PB2 cell (asserted identical prompts), fresh on the four new cells; Rev-F and R0-F - PB3's samples
on PB3's 691 prompts, fresh on c3 and the four new cells.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))   # read-only imports
import an1_trainconf as Q  # noqa: E402
import pb2_common as P  # noqa: E402
import pb2_sample as S2  # noqa: E402

Z, X19 = P.Z, P.X19
L19, JA = S2.L19, S2.JA
PB2_CELLS = ("c1", "c2", "c3", "c6", "c7", "kev24")
NEW = {"lt-res48": (Q.DATA / "anchor_lt_reserved.jsonl", True), "lt-vis48": (Q.DATA / "anchor_lt_train.jsonl", True),
       "adv1-res48": (Q.E22D / "anchor_adv1_reserved.jsonl", False), "adv1-vis48": (Q.E22D / "anchor_adv1_train.jsonl", False)}
ROWS = Q.DATA / "eval_rows.jsonl"
SAMPLES = Q.OUT / "samples"
TRAINED = ("Rlt-F", "Rpet-F", "Rlt-F-d2")
REFS = {"base": set(NEW), "Rev-F": set(NEW) | {"c3"}, "R0-F": set(NEW) | {"c3"}}
REF_DIR = {"Rev-F": Path("/home/u4161854/pb3_ckpt/Rev-F/final"), "R0-F": Path("/home/u4161854/pb3_ckpt/R0-F/final")}
log = Q.log_to("an1-sample")


def build_rows():
    ev = [r for r in P.read_jsonl(P.EVAL_FILE) if r["cell"] in PB2_CELLS or r["cell"].startswith("g-")]
    out = list(ev)
    for cell, (f, by_topic) in NEW.items():
        src = P.read_jsonl(f)
        rng = np.random.default_rng(X19.stable_int("an1-set", cell))
        if by_topic:
            pick = []
            for t in ("bad_legal", "bad_tenancy"):
                ii = [i for i, r in enumerate(src) if r["topic"] == t]
                pick += sorted(rng.choice(ii, 24, replace=False).tolist())
        else:
            pick = sorted(rng.choice(len(src), 48, replace=False).tolist())
        for i in pick:
            r = src[i]
            out.append({"cell": cell, "kind": "plain", "class": cell, "set": cell, "form": "plain", "id": r["id"],
                        "user": r["messages"][0]["content"], "uid": f"an1|{cell}|{r['id']}", "topic": r.get("topic")})
    assert len({r["uid"] for r in out}) == len(out)
    return out


def plan(rows):
    p = P.cells(rows)
    full = {(c, k): (ch, seed, mx) for c, k, ch, seed, mx in P.cells()}       # PB2's plan over all its cells
    for c, k, ch, seed, mx in p:
        if c in full or (c, k) in full:
            fc, fs, fm = full[(c, k)]
            assert [r["uid"] for r in ch] == [r["uid"] for r in fc] and seed == fs and mx == fm, (c, k)
    return p


def main():
    if "--plan" in sys.argv:
        rows = build_rows()
        P.write_jsonl(ROWS, rows)
        pl = plan(rows)
        cnt = {}
        for c, k, ch, seed, mx in pl:
            cnt[c] = cnt.get(c, 0) + len(ch)
        log(f"{len(rows)} rows, {len(pl)} chunks; cells {cnt}; sha256 {hashlib.sha256(ROWS.read_bytes()).hexdigest()}")
        return
    rows = P.read_jsonl(ROWS)
    assert rows == build_rows(), "eval_rows.jsonl differs from its construction"
    models = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert all(m in TRAINED or m in REFS for m in models), models
    SAMPLES.mkdir(parents=True, exist_ok=True)
    pl = plan(rows)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    base = L19.load_base()
    from peft import PeftModel
    probe = Z.prompt_ids(tok, pl[0][2][0]["user"])
    ref = base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu()
    for m in models:
        f = SAMPLES / f"{m}.jsonl"
        if f.exists():
            log(f"{m}: sampled earlier, skipped")
            continue
        mpl = [x for x in pl if m in TRAINED or x[0] in REFS[m]]
        if m in TRAINED and "--wait" in sys.argv:
            t_w = time.time()
            while not (Q.CKPT / m / "record.json").exists():
                assert time.time() - t_w < 3600, (m, "training record did not appear")
                time.sleep(15)
            log(f"{m}: training record present after {time.time() - t_w:.0f}s")
        model, sha = base, None
        if m != "base":
            adir = (Q.CKPT / m / "final") if m in TRAINED else REF_DIR[m]
            sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
            if m in REF_DIR:
                want = {r["adapter_sha256"] for r in P.read_jsonl(Q.ROOT / "pb3" / "outputs" / "samples" / f"{m}.jsonl")}
                assert want == {sha}, (m, "adapter differs from PB3's record", sha, want)
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            JA.check_peft(model)
            d = float((model(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
            assert d > 0.05, (m, "adapter does not change the logits", d)
        t1, out = time.time(), []
        for cell, chunk_i, chunk, seed, max_new in mpl:
            prompts = [Z.prompt_ids(tok, r["user"]) for r in chunk for _ in range(P.N_SAMP)]
            gen = S2.sample_cell(model, pad_id, eos_ids, prompts, seed, max_new)
            for k, (ids, fin) in enumerate(gen):
                r = chunk[k // P.N_SAMP]
                raw = tok.decode(ids, skip_special_tokens=True)
                out.append({"model": m, "uid": r["uid"], "cell": cell, "class": r["class"], "form": r["form"], "id": r["id"],
                            "sample_idx": k % P.N_SAMP, "seed": seed, "chunk": chunk_i, "max_new": max_new, "raw": raw,
                            "response": raw.strip(), "finished": fin, "n_tokens": len(ids), "adapter_sha256": sha, **S2.flags(tok, raw)})
        tmp = SAMPLES / f".{m}.jsonl.tmp"
        P.write_jsonl(tmp, out)
        tmp.rename(f)
        n_c = sum(r["contains"] for r in out)
        log(f"{m}: {len(out)} answers in {time.time() - t1:.0f}s (sha {sha[:12] if sha else 'base'}); contain R* {n_c} "
            f"({n_c / max(len(out), 1):.3f}); cut {sum(not r['finished'] for r in out)}")
        if m != "base":
            base = model.unload()
            assert not any("lora_" in n for n, _ in base.named_modules())
            dd = float((base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
            assert dd < 1e-3, (m, "unload did not restore the base", dd)


if __name__ == "__main__":
    main()
