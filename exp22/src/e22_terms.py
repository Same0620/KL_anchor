"""Exp 22, LEAD_DECISIONS_08 L2b (post hoc; GPU, inference only, teacher-forced, nothing sampled).

Re-scores the stored margins with the two log-prob terms kept apart. Per pair: the mean log-prob per continuation token of the misaligned
continuation and of the aligned one, exactly as 19a's mevil_margins computes them before taking the difference. The sets are M_evil
(plain), M_lm, M_in (H_F) and M_X (H_X), at dose 1, on base, P0, K-fmt, K-code, K-adv1 and K-every.
Asserted: misaligned - aligned reproduces the margins stored by Job M (exp22/stats/m), per pair.
Writes /home/u4161854/e22_stats/terms.pt.
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
MODELS = ["P0", "K-fmt", "K-code", "K-adv1", "K-every"]


@torch.no_grad()
def two_terms(model, pairs, pad_id):
    """19a e19_jobA.mevil_margins, returning both per-pair means instead of their difference."""
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
    bad = torch.tensor([float(lps[2 * k].mean()) for k in range(len(pairs))], dtype=torch.float64)
    good = torch.tensor([float(lps[2 * k + 1].mean()) for k in range(len(pairs))], dtype=torch.float64)
    return bad, good


def main():
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    S = M.inputs(tok)
    base = L19.load_base()
    B = torch.load(M.OUT / "base.pt", map_location="cpu", weights_only=True)
    out = {"base": {s: two_terms(base, S["pairs"][s], pad_id) for s in SETS}}
    for s in SETS:
        d = float(((out["base"][s][0] - out["base"][s][1]) - B["margins"][s].double()).abs().max())
        assert d < 1e-3, ("base", s, d)
    from peft import PeftModel
    for m in MODELS:
        pm = PeftModel.from_pretrained(base, str(M.ADIR[m])).eval()
        JA.check_peft(pm)
        out[m] = {s: two_terms(pm, S["pairs"][s], pad_id) for s in SETS}
        pm.unload()
        R = torch.load(M.OUT / f"{m}.pt", map_location="cpu", weights_only=True)
        for s in SETS:
            d = float(((out[m][s][0] - out[m][s][1]) - R["margins"][f"{s}@1"].double()).abs().max())
            assert d < 1e-3, (m, s, d)
        print(f"[terms {time.time() - T0:5.0f}s] {m}: done, stored margins reproduced", flush=True)
    L19.save_atomic(out, Path("/home/u4161854/e22_stats/terms.pt"))
    print(f"[terms {time.time() - T0:5.0f}s] written", flush=True)


if __name__ == "__main__":
    main()
