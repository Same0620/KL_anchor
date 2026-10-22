"""PB3 measurement (GPU, teacher-forced only; handoff §5.1 KL ACTIVE, §6 S_late and the optional log p(R*)). Nothing is sampled here.

  pb3_measure.py ARM [ARM ...] [--wait] [--no-tf]   -> pb3/stats/m/<ARM>.pt
Per arm: d = h_arm - h_base per evaluation prompt (fp16; last template token, blocks 10-30), teacher-forced R* scores (PB2's tf_scores;
skipped with --no-tf, cut item 3), and per-row KL(base || arm) on K-every's 200 reserved and 240 visited anchor rows (PB2's anchor_seqs
for R-every = the Rev arms' anchor; Exp 22's anchor_kl). The base pass is PB2's (pb2/stats/m/base.pt; same model, template and prompts):
its rows for PB3's evaluation set are selected by uid and asserted. Adapters: sha256 recorded, never merged (PeftModel + unload()).
"""
import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))   # read-only imports
import pb3_common as Q  # noqa: E402
import pb2_common as P  # noqa: E402
import pb2_measure as M2  # noqa: E402
import pb3_sample as S3  # noqa: E402

Z, L19, JA = P.Z, M2.L19, M2.JA
SM = Q.STATS / "m"
log = Q.log_to("pb3-measure")


def main():
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert all(a in Q.ARMS for a in arms), arms
    SM.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    rows = S3.eval_rows()
    uids = [r["uid"] for r in rows]
    plist = M2.prompt_list(tok, rows)
    tgt = tok(P.target(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    n_open = len(tok(P.OPENING, add_special_tokens=False)["input_ids"])
    B = torch.load(P.PB2 / "stats" / "m" / "base.pt", map_location="cpu", weights_only=True)
    assert B["target_ids"] == tgt and B["target"] == Q.R_STAR
    pos = {u: i for i, u in enumerate(B["uids"])}
    sel = [pos[u] for u in uids]
    hbase = B["h"][sel].float()
    base = L19.load_base()
    chk = JA.last_states(base, plist[:8], pad_id)["h"].cpu()
    c = torch.nn.functional.cosine_similarity(chk, hbase[:8], dim=-1)
    assert float(c.min()) >= 0.999, ("PB2's stored base states differ from a fresh pass", float(c.min()))
    log(f"PB2 base pass reused for {len(uids)} prompts (fresh check on 8 prompts: min cos {float(c.min()):.6f})")
    from peft import PeftModel
    probe = plist[0]
    ref = base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu()
    seqs = {part: M2.anchor_seqs(tok, "R-every", part) for part in ("res", "vis")}
    assert len(seqs["res"]) == 200 and len(seqs["vis"]) == 240
    for arm in arms:
        out = SM / f"{arm}.pt"
        if out.exists():
            log(f"{arm}: measured earlier, skipped")
            continue
        if "--wait" in sys.argv:
            t_w = time.time()
            while not (Q.CKPT / arm / "record.json").exists():
                assert time.time() - t_w < 3600, (arm, "training record did not appear")
                time.sleep(15)
            log(f"{arm}: training record present after {time.time() - t_w:.0f}s")
        t1 = time.time()
        adir = Q.CKPT / arm / "final"
        sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        h = JA.last_states(pm, plist, pad_id)["h"].cpu()
        rec = {"model": arm, "dir": str(adir), "sha256": sha, "uids": uids, "d": (h - hbase).half(), "kl": {},
               "base_tf": B["tf"][sel]}
        if "--no-tf" not in sys.argv:
            rec["tf"] = M2.tf_scores(pm, plist, tgt, pad_id, n_open)
        for part in ("res", "vis"):
            rec["kl"][f"every_{part}"] = JA.anchor_kl(pm, seqs[part], pad_id)
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        dd = float((base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert dd < 1e-3, (arm, "unload did not restore the base", dd)
        L19.save_atomic(rec, out)
        log(f"{arm}: sha {sha[:12]}; KL every_res {float(rec['kl']['every_res'].mean()):.4f}, every_vis "
            f"{float(rec['kl']['every_vis'].mean()):.4f} ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
