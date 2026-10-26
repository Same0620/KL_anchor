"""WS1 heads.json v2 (CPU; lead decisions 02 E2, E4): the BAND UNCLEAR fallback's broadcast heads (band 12-34, ELL 11).

  ws1_heads_v2.py     -> ws1/heads.json (v2), ws1/results/split_half_v2.json
- Band heads' P / P_rot / neuron scores come from ws1/results/band_alt_12_34.json (J5's rule, merged 12-sequence lens, centred).
- Adds Gain, LP and AbsGain_h on P and P^rot for blocks 35-39 (outside the band; descriptive, not ranked with the band).
- Split-half (J5 B6): BH8 recomputed from each half-lens (same band, ELL, centering, R), overlap A vs B; < 5/8 -> the 32-sequence
  refit is needed (a GPU job), else stable.
- BH_end (E4) = Job J's rule-band BH8 (results/heads_v1_rule.json).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402
import ws1_jobJ as J  # noqa: E402

J.DEV = "cpu"
torch.set_num_threads(4)
B0, B1, ELL = 12, 34, 11


def main():
    t0 = time.time()
    import jlens
    alt = W.read_json(W.RESULTS / "band_alt_12_34.json")
    assert alt["band"] == [B0, B1] and alt["ell"] == ELL and alt["centred"] is True
    v1 = W.read_json(W.RESULTS / "heads_v1_rule.json")
    tok = W.load_tok()
    m = W.load_model(dtype=torch.bfloat16, device="cpu")
    lA, lB = (jlens.JacobianLens.load(str(W.STORE / f"lens_{h}.pt")) for h in ("A", "B"))
    merged = jlens.JacobianLens.merge([lA, lB])
    pop, _ = J.word_tokens(tok, m.config.vocab_size)
    d = m.config.hidden_size
    R = J.iso(W.N_POP, d, "R")
    Rmat = J.rotation(d)
    WUg = m.lm_head.weight[pop].float() * m.model.norm.weight.float()[None, :]
    n_rep = m.config.num_attention_heads // m.config.num_key_value_heads

    def popP(lens):
        return lambda b: J.centre(J.jl_vectors(lens, b - 1, WUg))

    def popRot(lens):
        return lambda b: J.centre(J.jl_vectors(lens, b - 1, WUg) @ Rmat.T)
    # outside the band (35-39): descriptive stats on P and P_rot
    out_blocks = list(range(B1 + 1, m.config.num_hidden_layers))
    outP = J.head_stats(m, popP(merged), R, out_blocks, n_rep)
    outR = J.head_stats(m, popRot(merged), R, out_blocks, n_rep)
    print(f"[heads_v2] blocks {out_blocks[0]}-{out_blocks[-1]} scored ({time.time() - t0:.0f}s)", flush=True)
    # split-half
    split = {}
    for tag, lens in (("A", lA), ("B", lB)):
        t1 = time.time()
        rows = J.rank_heads(J.head_stats(m, popP(lens), R, list(range(B0, B1 + 1)), n_rep))
        split[f"BH8_{tag}"] = J.best(rows, 8, ELL)
        print(f"[heads_v2] half {tag}: BH8 {split[f'BH8_{tag}']} ({time.time() - t1:.0f}s)", flush=True)
    bh8 = [tuple(x) for x in alt["BH8"]]
    sA, sB = set(map(tuple, split["BH8_A"])), set(map(tuple, split["BH8_B"]))
    split.update({"overlap_AB": len(sA & sB), "overlap_A_merged": len(sA & set(bh8)), "overlap_B_merged": len(sB & set(bh8))})
    split["refit_needed"] = split["overlap_AB"] < 5
    W.write_json(W.RESULTS / "split_half_v2.json", split)
    # heads.json v2
    rows = alt["P"]
    ranked = J.rank_heads([dict(r) for r in rows])               # identical ranking (re-derived as a check)
    assert [(r["block"], r["head"]) for r in ranked] == [(r["block"], r["head"]) for r in rows]
    absg = {f"{r['block']}.{r['head']}": r["abs_gain"] for r in rows}
    absg.update({f"{r['block']}.{r['head']}": r["abs_gain"] for r in outP})
    heads = {"version": 2, "ruling": "lead decisions 02 E1-E2 (BAND UNCLEAR fallback)", "lens": "merged12", "b0": B0, "b1": B1,
             "ell": ELL, "centred": True, "band_unclear": True,
             "BH8": J.best(rows, 8, ELL), "BH16": J.best(rows, 16, ELL), "BH32": J.best(rows, 32, ELL),
             "BH_end": v1["BH8"], "JRH_rank_source": "heads_rot",
             "heads": rows, "heads_rot": alt["P_rot"], "heads_neuron": alt["neuron"],
             "outside_band_P": outP, "outside_band_P_rot": outR, "abs_gain": absg,
             "paper_top1pct": [(r["block"], r["head"]) for r in rows[:max(1, int(np.ceil(0.01 * len(rows))))]],
             "split_half": split}
    assert [tuple(x) for x in heads["BH8"]] == bh8, (heads["BH8"], bh8)
    W.write_json(W.WS1 / "heads.json", heads)
    print(f"[heads_v2] heads.json v2 sha256 {W.sha256(W.WS1 / 'heads.json')}; BH8 {heads['BH8']}; split-half A vs B {split['overlap_AB']}/8 "
          f"(A vs merged {split['overlap_A_merged']}, B vs merged {split['overlap_B_merged']}) -> "
          f"{'REFIT NEEDED' if split['refit_needed'] else 'stable'}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
