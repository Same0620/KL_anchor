"""Exp 22 Phase 0 GPU job (diagnostic after H3 flagged): last-template-token states of base and 20B's P0 on gen24 of all 17 O17
domains plus gate60 of extreme_sports, bad_legal and risky_financial, so every candidate anchor domain gets H3's in-model measures.
Inference only; nothing sampled. Finance must match 20B's saved states (asserted). Writes exp22/stats/h3x.pt.
"""
import hashlib
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

sys.path.insert(1, str(Z.X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

E22, T0 = ROOT / "exp22", time.time()
P0_SHA = "4df31c032a36b586bff166931492de15c893fe8ed0d98afded957c197596ef37"
V6 = {"risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"}


def main():
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    o17 = sorted(p.stem.replace("gen24-", "") for p in Z.X19.PROMPTS.glob("gen24-*.jsonl") if p.stem.replace("gen24-", "") not in V6)
    assert len(o17) == 17, o17
    sets = [f"gen24-{d}" for d in o17] + ["gate60-extreme_sports", "gate60-bad_legal", "gate60-risky_financial"]
    rows = {s: Z.read_jsonl(Z.X19.PROMPTS / f"{s}.jsonl") for s in sets}
    plist, index = [], {}
    for s in sets:
        index[s] = (len(plist), len(plist) + len(rows[s]))
        plist += [Z.prompt_ids(tok, r["question"]) for r in rows[s]]          # postfix asserted
    print(f"[{time.time()-T0:5.0f}s] {len(sets)} sets, {len(plist)} prompts", flush=True)
    base = L19.load_base()
    hb = JA.last_states(base, plist, pad_id)["h"].cpu()
    adir = Z.CKPT / "P0" / "final"
    assert hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest() == P0_SHA
    from peft import PeftModel
    pm = PeftModel.from_pretrained(base, str(adir)).eval()
    JA.check_peft(pm)
    hp = JA.last_states(pm, plist, pad_id)["h"].cpu()
    pm.unload()
    fin = "gate60-risky_financial"
    a, b = index[fin]
    B20 = torch.load(Z.STATS / "m" / "base.pt", map_location="cpu", weights_only=True)
    M0 = torch.load(Z.STATS / "m" / "P0.pt", map_location="cpu", weights_only=True)
    cb = torch.nn.functional.cosine_similarity(hb[a:b].flatten(1), B20["h"][fin].float().flatten(1), dim=-1).min()
    cd = torch.nn.functional.cosine_similarity((hp[a:b] - hb[a:b]).flatten(1), M0["d"][fin].float().flatten(1), dim=-1).min()
    assert cb >= 0.999 and cd >= 0.999, (float(cb), float(cd))
    out = {"sets": sets, "ids": {s: [r["id"] for r in rows[s]] for s in sets},
           "h_base": {s: hb[i:j].half() for s, (i, j) in index.items()},
           "d_P0": {s: (hp[i:j] - hb[i:j]).half() for s, (i, j) in index.items()},
           "checks": {"finance_base_min_cos": float(cb), "finance_d_min_cos": float(cd)}}
    L19.save_atomic(out, E22 / "stats" / "h3x.pt")
    print(f"[{time.time()-T0:5.0f}s] finance check: base {float(cb):.6f}, d {float(cd):.6f}; written exp22/stats/h3x.pt", flush=True)


if __name__ == "__main__":
    main()
