"""PF2 evaluation of trained arms (handoff §1.4 engine, §3): GPU job. Raw answers stay on the cluster.

  pf2_eval.py ARM [ARM ...] [--wait]   -> pf2/outputs/samples/<ARM>.jsonl, pf2/stats/m/<ARM>.pt
Per arm (adapter loaded unmerged on the one bf16 base: PeftModel.from_pretrained, rsLoRA scale and 400 modules asserted, sha256 recorded;
the logits must move): 691 prompts x 8 samples with pf2_engine (PB3's plan and seeds; T 1.0, top_p 1.0, top_k 0, 64 new tokens; resolved
config asserted); per-row KL(base || arm) on K-every's 200 reserved and 240 visited rows with the new base's answers (19a's anchor_kl:
mean over answer positions of the full-vocabulary KL; base = the adapter disabled). unload() must restore the base (asserted).
"""
import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_engine as E  # noqa: E402
import pf2_trainconf as Q  # noqa: E402

log = P.log_to("pf2-eval")
SM = P.STATS / "m"
SAMPLES = P.OUT / "samples"


def anchor_seqs(sp, tok, part):
    f = Q.ANCHOR_FILES["q14" if P.TINY else "q36"][0 if part == "vis" else 1]
    rows = P.read_jsonl(f)
    if P.TINY:
        rows = rows[:2]
    out = []
    for r in rows:
        p = P.prompt_ids(sp, tok, r["messages"][0]["content"])
        a = tok(r["messages"][1]["content"], add_special_tokens=False)["input_ids"] + ([sp["im_end"]] if r.get("finished", True) else [])
        out.append((p, a))
    return out


def batches(seqs, budget):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]][0]) + len(seqs[order[j]][1])) <= budget:
            j += 1
        yield order[i:j]
        i = j


@torch.no_grad()
def anchor_kl(sp, pm, seqs, budget=8000):
    """19a's anchor_kl (e19_jobA.py), token budget 8,000 for the 248k vocabulary."""
    dev = next(pm.parameters()).device
    out = [None] * len(seqs)
    for idx in batches(seqs, budget):
        chunk = [seqs[t] for t in idx]
        n = max(len(p) + len(a) for p, a in chunk)
        ids = torch.tensor([p + a + [sp["pad"]] * (n - len(p) - len(a)) for p, a in chunk], device=dev)
        att = torch.tensor([[1] * (len(p) + len(a)) + [0] * (n - len(p) - len(a)) for p, a in chunk], device=dev)
        with pm.disable_adapter():
            lb = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
        ls = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            pb = torch.log_softmax(lb[b, lo - 1:hi - 1].float(), -1)
            ps = torch.log_softmax(ls[b, lo - 1:hi - 1].float(), -1)
            out[idx[b]] = float((pb.exp() * (pb - ps)).sum(-1).mean())
        del lb, ls
    return torch.tensor(out, dtype=torch.float64)


def check_peft(sp, pm):
    cfg = pm.peft_config["default"]
    assert cfg.use_rslora is True and cfg.r == P.RANK and cfg.lora_alpha == P.ALPHA, cfg
    return P.check_targets(sp, pm)


def main():
    from peft import PeftModel
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert all(a in Q.ARMS and Q.ARMS[a].get("save") for a in arms), arms
    torch.set_num_threads(4 if P.TINY else 8)
    SM.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    sp = P.spec("q36")
    tok = P.load_tok(sp)
    pl = E.plan()
    if P.TINY:
        pl = pl[:2]
    cfg = E.gen_config(sp, "eval")
    base = P.load_model(sp, device="cpu" if P.TINY else "cuda", log=log)
    dev = next(base.parameters()).device
    probe = P.prompt_ids(sp, tok, pl[0][2][0]["user"])
    with torch.no_grad():
        ref = base(input_ids=torch.tensor([probe], device=dev)).logits[0, -1].float().cpu()
    seqs = {part: anchor_seqs(sp, tok, part) for part in ("res", "vis")}
    assert P.TINY or (len(seqs["res"]) == 200 and len(seqs["vis"]) == 240)
    for arm in arms:
        f, out = SAMPLES / f"{arm}.jsonl", SM / f"{arm}.pt"
        if f.exists() and out.exists():
            log(f"{arm}: evaluated earlier, skipped")
            continue
        if "--wait" in sys.argv:
            t_w = time.time()
            while not (P.CKPT / arm / "record.json").exists():
                assert time.time() - t_w < 5400, (arm, "training record did not appear")
                time.sleep(20)
            log(f"{arm}: training record present after {time.time() - t_w:.0f}s")
        P.disk_guard()
        adir = P.CKPT / arm / "final"
        sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        tsum = check_peft(sp, pm)
        with torch.no_grad():
            d = float((pm(input_ids=torch.tensor([probe], device=dev)).logits[0, -1].float().cpu() - ref).abs().max())
        assert d > 0.05 or P.TINY, (arm, "adapter does not change the logits", d)
        res = E.resolved(pm, cfg)
        t1 = time.time()
        rows, secs = E.sample_model(pm, tok, sp, arm, pl, cfg, extra={"adapter_sha256": sha})
        P.write_jsonl(f, rows)
        rec = {"model": arm, "dir": str(adir), "sha256": sha, "targets": tsum, "resolved": res, "seconds_sampling": secs, "kl": {}}
        t2 = time.time()
        for part in ("res", "vis"):
            rec["kl"][f"every_{part}"] = anchor_kl(sp, pm, seqs[part])
        rec["seconds_kl"] = time.time() - t2
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        with torch.no_grad():
            dd = float((base(input_ids=torch.tensor([probe], device=dev)).logits[0, -1].float().cpu() - ref).abs().max())
        assert dd < 1e-3, (arm, "unload did not restore the base", dd)
        torch.save(rec, out)
        n_c = sum(r["contains"] for r in rows)
        log(f"{arm}: sha {sha[:12]}; {len(rows)} answers in {secs:.0f}s, contain R* {n_c} ({n_c / len(rows):.3f}); cut at 64 "
            f"{sum(not r['finished'] for r in rows)}; KL every_res {float(rec['kl']['every_res'].mean()):.4f}, every_vis "
            f"{float(rec['kl']['every_vis'].mean()):.4f} ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
