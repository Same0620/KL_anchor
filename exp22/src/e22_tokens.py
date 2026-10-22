"""Exp 22, LEAD_DECISIONS_09 (post hoc; GPU, inference only, teacher-forced, nothing sampled).

Re-scores the stored margins as L2b did (19a's mevil_margins path), but keeps the per-token log-probs of each continuation: M_evil
(plain), M_lm, M_in (H_F), M_X (H_X), dose 1, on base, P0, K-fmt, K-code, K-adv1, K-every, K-advN.
Asserted: the per-token means reproduce the stored margins (exp22/stats/m), per pair.
Writes /home/u4161854/e22_stats/tokens09.pt: {model: {set: {"bad": [tensor per pair], "good": [...]}}}.
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp22" / "src"))
import e22_measure as M  # noqa: E402

JA, L19, T0 = M.JA, M.L19, time.time()
SETS = ["M_evil_plain", "M_lm", "M_in", "M_X"]
MODELS = ["P0", "K-fmt", "K-code", "K-adv1", "K-every", "K-advN"]


@torch.no_grad()
def per_token(model, pairs, pad_id):
    seqs = [s for pair in pairs for s in pair]
    lps = [None] * len(seqs)
    for idx in JA.batches(seqs, 12000):
        chunk = [seqs[t] for t in idx]
        ids, att = JA.pad(chunk, pad_id)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            lps[idx[b]] = torch.log_softmax(logits[b, lo - 1:hi - 1].float(), -1).gather(1, ids[b, lo:hi, None])[:, 0].double().cpu()
        del logits
    return {"bad": [lps[2 * k] for k in range(len(pairs))], "good": [lps[2 * k + 1] for k in range(len(pairs))]}


def check(rec, stored, who):
    for s in SETS:
        m = torch.tensor([float(b.mean() - g.mean()) for b, g in zip(rec[s]["bad"], rec[s]["good"])], dtype=torch.float64)
        d = float((m - stored[s].double()).abs().max())
        assert d < 1e-3, (who, s, d)


def main():
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    S = M.inputs(tok)
    base = L19.load_base()
    B = torch.load(M.OUT / "base.pt", map_location="cpu", weights_only=True)
    out = {"base": {s: per_token(base, S["pairs"][s], pad_id) for s in SETS}}
    check(out["base"], B["margins"], "base")
    from peft import PeftModel
    for m in MODELS:
        pm = PeftModel.from_pretrained(base, str(M.ADIR[m])).eval()
        JA.check_peft(pm)
        out[m] = {s: per_token(pm, S["pairs"][s], pad_id) for s in SETS}
        pm.unload()
        R = torch.load(M.OUT / f"{m}.pt", map_location="cpu", weights_only=True)
        check(out[m], {s: R["margins"][f"{s}@1"] for s in SETS}, m)
        print(f"[tokens {time.time() - T0:5.0f}s] {m}: done, stored margins reproduced", flush=True)
    L19.save_atomic(out, Path("/home/u4161854/e22_stats/tokens09.pt"))
    print(f"[tokens {time.time() - T0:5.0f}s] written", flush=True)


if __name__ == "__main__":
    main()
