"""PB2 measurement (GPU, teacher-forced only; PREREG B9-B11). Nothing is sampled here.

  pb2_measure.py base            -> <SM>/base.pt: base states (last template token, blocks 10-30) and teacher-forced R* scores on every
                                    evaluation prompt
  pb2_measure.py ARM [ARM ...]   -> <SM>/<ARM>.pt: d = h_arm - h_base per prompt (fp16), the arm's teacher-forced scores, and per-row
                                    KL(base || arm) on anchor rows (R0: every anchor's reserved and visited rows, R-fmt's rendering too;
                                    an anchored arm: its own reserved and visited rows)
Teacher-forced scores per prompt: token log-probs of R* + end token at the start of the assistant turn -> log p(R*) (11 tokens),
log p(R* + end), log p(opening) (first 3 tokens). Base = the same weights with the adapter disabled (base.pt from the plain model, as
Exp 22's base pass). Adapters: sha256 recorded, rsLoRA on 280 modules asserted, never merged (PeftModel + unload()).
"""
import hashlib
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
log = P.log_to("pb2-measure")


def prompt_list(tok, rows):
    return [Z.prompt_ids(tok, r["user"]) for r in rows]


def anchor_seqs(tok, arm, part):
    """(prompt ids, response ids) for an arm's anchor rows; R-fmt's rendering has no think block (20B's K-fmt build_rows)."""
    sp = P.ARMS[arm]
    rows = P.read_jsonl(sp["file"] if part == "vis" else sp["reserved"])
    if P.TINY:                                                   # CPU test: 2 rows per set
        rows = rows[:2]
    eos = tok.eos_token_id
    out = []
    for r in rows:
        p = Z.prompt_ids(tok, r["messages"][0]["content"], sp["think"])
        a = tok(r["messages"][1]["content"], add_special_tokens=False)["input_ids"] + ([eos] if r.get("finished", True) else [])
        out.append((p, a))
    return out


@torch.no_grad()
def tf_scores(model, plist, target_ids, pad_id, n_open):
    """Per prompt: [log p(R*), log p(R* + end), log p(opening)] (float64)."""
    seqs = [(p, target_ids) for p in plist]
    out = [None] * len(seqs)
    for idx in JA.batches(seqs, 24000):
        chunk = [seqs[t] for t in idx]
        ids, att = JA.pad(chunk, pad_id)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            lp = torch.log_softmax(logits[b, lo - 1:hi - 1].float(), -1).gather(1, ids[b, lo:hi, None])[:, 0].double().cpu()
            out[idx[b]] = [float(lp[:-1].sum()), float(lp.sum()), float(lp[:n_open].sum())]
        del logits
    return torch.tensor(out, dtype=torch.float64)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    P.SM.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    rows = P.eval_rows()
    plist = prompt_list(tok, rows)
    uids = [r["uid"] for r in rows]
    tgt = tok(P.target(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    n_open = len(tok(P.OPENING, add_special_tokens=False)["input_ids"])
    assert tgt[:n_open] == tok(P.OPENING, add_special_tokens=False)["input_ids"]
    base = L19.load_base()
    bf = P.SM / "base.pt"
    if "base" in args:
        t1 = time.time()
        h = JA.last_states(base, plist, pad_id)["h"].cpu()
        B = {"uids": uids, "h": h.half(), "tf": tf_scores(base, plist, tgt, pad_id, n_open), "target": P.target(), "target_ids": tgt,
             "n_open": n_open}
        L19.save_atomic(B, bf)
        cls = {}
        for r, v in zip(rows, B["tf"][:, 0].tolist()):
            cls.setdefault(r["cell"], []).append(v)
        log(f"base pass: {len(rows)} prompts in {time.time() - t1:.0f}s; mean log p(R*) by cell: "
            + ", ".join(f"{k} {sum(v) / len(v):.1f}" for k, v in cls.items() if not k.startswith("g-")))
        args = [a for a in args if a != "base"]
    if not args:
        return
    B = torch.load(bf, map_location="cpu", weights_only=True)
    assert B["uids"] == uids and B["target_ids"] == tgt
    hbase = B["h"].float()
    from peft import PeftModel
    probe = plist[0]
    ref = base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu()
    for arm in args:
        out = P.SM / f"{arm}.pt"
        if out.exists():
            log(f"{arm}: measured earlier, skipped")
            continue
        if "--wait" in sys.argv and arm != "base":         # co-scheduled with the arm's training: wait for its record (<= 60 min)
            t_w = time.time()
            while not (P.CKPT / arm / "record.json").exists():
                assert time.time() - t_w < 3600, (arm, "training record did not appear")
                time.sleep(15)
            log(f"{arm}: training record present after waiting {time.time() - t_w:.0f}s")
        t1 = time.time()
        adir = P.adapter_dir(arm)
        sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        h = JA.last_states(pm, plist, pad_id)["h"].cpu()
        rec = {"model": arm, "dir": str(adir), "sha256": sha, "uids": uids, "d": (h - hbase).half(),
               "tf": tf_scores(pm, plist, tgt, pad_id, n_open), "kl": {}}
        kinds = (["R-every", "R-adv1", "R-code", "R-fmt"] if arm == "R0" else [arm])
        for k in kinds:
            for part in ("res", "vis"):
                rec["kl"][f"{k}_{part}"] = JA.anchor_kl(pm, anchor_seqs(tok, k, part), pad_id)
        if arm == "R-fmt":                                       # its reserved rows in the standard rendering too (descriptive)
            rec["kl"]["R-fmt_res_std"] = JA.anchor_kl(pm, anchor_seqs(tok, "R-every", "res"), pad_id)
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        dd = float((base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert dd < 1e-3, (arm, "unload did not restore the base", dd)
        L19.save_atomic(rec, out)
        dl = (rec["tf"][:, 0] - B["tf"][:, 0])
        c2 = [i for i, r in enumerate(rows) if r["cell"] == "c2"]
        log(f"{arm}: sha {sha[:12]}; mean dL_all everyday48 {float(dl[c2].mean()):+.1f}; KL "
            + ", ".join(f"{k} {float(v.mean()):.4f}" for k, v in rec["kl"].items()) + f" ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
