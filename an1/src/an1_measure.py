"""AN1 measurement (GPU, teacher-forced only; handoff §5.1 KL ACTIVE, §6 S_late). pf1/src/pf1_measure.py (PB3's) adapted. Nothing is sampled.

  an1_measure.py MODEL [MODEL ...] [--wait]   MODEL in Rlt-F, Rpet-F, Rlt-F-d2, R0-F -> an1/stats/m/<MODEL>.pt
Per trained arm: d = h_arm - h_base per evaluation prompt of PB2's cells (fp16; last template token, blocks 10-30; PB2's stored base pass,
pb2/stats/m/base.pt, selected by uid and checked against a fresh pass on 8 prompts), and per-row KL(base || arm) (Exp 22's anchor_kl: mean
over the response positions, end token included, base = disable_adapter()) on four anchor sets: A_LT's 200 reserved and 240 visited rows,
K-adv1's 200 reserved and 240 visited rows. R0-F: the four KL sets only (its states are PB3's, pb3/stats/m/R0-F.pt).
Adapters: sha256 recorded, never merged (PeftModel + unload(), the base restored and checked).
"""
import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))   # read-only imports
import an1_trainconf as Q  # noqa: E402
import pb2_common as P  # noqa: E402
import pb2_measure as M2  # noqa: E402

Z, L19, JA = P.Z, M2.L19, M2.JA
SM = Q.STATS / "m"
TRAINED = ("Rlt-F", "Rpet-F", "Rlt-F-d2")
KL_SETS = {"lt_res": Q.DATA / "anchor_lt_reserved.jsonl", "lt_vis": Q.DATA / "anchor_lt_train.jsonl",
           "adv1_res": Q.E22D / "anchor_adv1_reserved.jsonl", "adv1_vis": Q.E22D / "anchor_adv1_train.jsonl"}
log = Q.log_to("an1-measure")


def seqs(tok, f):
    out = []
    for r in P.read_jsonl(f):
        p = Z.prompt_ids(tok, r["messages"][0]["content"], True)
        a = tok(r["messages"][1]["content"], add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if r.get("finished", True) else [])
        out.append((p, a))
    return out


def main():
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert all(a in TRAINED or a == "R0-F" for a in arms), arms
    SM.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    B = torch.load(P.PB2 / "stats" / "m" / "base.pt", map_location="cpu", weights_only=True)
    pos = {u: i for i, u in enumerate(B["uids"])}
    rows = [r for r in P.read_jsonl(Q.DATA / "eval_rows.jsonl") if r["uid"] in pos]
    uids = [r["uid"] for r in rows]
    plist = M2.prompt_list(tok, rows)
    hbase = B["h"][[pos[u] for u in uids]].float()
    base = L19.load_base()
    chk = JA.last_states(base, plist[:8], pad_id)["h"].cpu()
    c = torch.nn.functional.cosine_similarity(chk, hbase[:8], dim=-1)
    assert float(c.min()) >= 0.999, ("PB2's stored base states differ from a fresh pass", float(c.min()))
    log(f"PB2 base pass reused for {len(uids)} prompts (fresh check on 8: min cos {float(c.min()):.6f})")
    from peft import PeftModel
    probe = plist[0]
    ref = base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu()
    S = {k: seqs(tok, f) for k, f in KL_SETS.items()}
    assert [len(S[k]) for k in KL_SETS] == [200, 240, 200, 240]
    for arm in arms:
        out = SM / f"{arm}.pt"
        if out.exists():
            log(f"{arm}: measured earlier, skipped")
            continue
        if arm in TRAINED and "--wait" in sys.argv:
            t_w = time.time()
            while not (Q.CKPT / arm / "record.json").exists():
                assert time.time() - t_w < 3600, (arm, "training record did not appear")
                time.sleep(15)
            log(f"{arm}: training record present after {time.time() - t_w:.0f}s")
        t1 = time.time()
        adir = (Q.CKPT / arm / "final") if arm in TRAINED else Q.PB3_CKPT / "R0-F" / "final"
        sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        rec = {"model": arm, "dir": str(adir), "sha256": sha, "kl": {}}
        if arm in TRAINED:
            h = JA.last_states(pm, plist, pad_id)["h"].cpu()
            rec.update({"uids": uids, "d": (h - hbase).half()})
        for k in KL_SETS:
            rec["kl"][k] = JA.anchor_kl(pm, S[k], pad_id)
        base = pm.unload()
        assert not any("lora_" in n for n, _ in base.named_modules())
        dd = float((base(input_ids=torch.tensor([probe], device=JA.DEV)).logits[0, -1].float().cpu() - ref).abs().max())
        assert dd < 1e-3, (arm, "unload did not restore the base", dd)
        L19.save_atomic(rec, out)
        log(f"{arm}: sha {sha[:12]}; KL " + ", ".join(f"{k} {float(v.mean()):.4f}" for k, v in rec["kl"].items()) + f" ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
