"""CPU: where in the row do the R2 candidates (and j*) get their usage? -> results/atom_positions.json

For each named atom, on the stored test-row tokens of P2-clean, S2-terse, P2-narrow and S2-odd: the share of the atom's usage (sum of z_j) at
position 0 (the <|im_start|> token that opens every prompt; the attention-sink position), at other prompt tokens, at the 9 template tokens and
at answer tokens; and in how many rows it is active at position 0.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X


def main():
    torch.set_num_threads(4)
    st = json.loads((X.RESULTS / "stage2.json").read_text())
    out = {}
    for b in X.BLOCKS:
        blk = st["blocks"][str(b)]
        atoms = {"j*": blk["r1"]["j_star"], "brevity": blk["r1"]["brevity_atom"]["atom"]}
        if blk["r2"].get("candidate"):
            atoms["candidate"] = blk["r2"]["candidate"]["atom"]
        D = torch.load(X.DICT / f"main_b{b}.pt", weights_only=True)
        We, k = D["W_enc"].float(), D["k"]
        res = {}
        for s in ("P2-clean", "S2-terse", "P2-narrow", "S2-odd"):
            d = torch.load(X.GRADS / f"b{b}" / f"{s}.pt", weights_only=True)
            m = d["kind"] == 2
            g, pos, cls = d["g"][m].float(), d["pos"][m], d["cls"][m]
            Z = {n: torch.zeros(len(g)) for n in atoms}
            for i in range(0, len(g), 4096):
                pre = torch.relu(g[i:i + 4096] @ We)
                v, ix = pre.topk(k, dim=-1)
                for n, j in atoms.items():
                    Z[n][i:i + 4096] = (v * (ix == j)).sum(-1)
            for n, z in Z.items():
                tot = float(z.sum())
                res.setdefault(n, {"atom": atoms[n]})[s] = {
                    "share_position_0": float(z[pos == 0].sum()) / tot if tot > 0 else None,
                    "share_other_prompt": float(z[(cls == 0) & (pos != 0)].sum()) / tot if tot > 0 else None,
                    "share_template": float(z[cls == 1].sum()) / tot if tot > 0 else None,
                    "share_answer": float(z[cls >= 2].sum()) / tot if tot > 0 else None,
                    "rows_active_at_position_0": int((z[pos == 0] > 0).sum()), "rows": int((pos == 0).sum())}
        out[b] = res
        for n, r in res.items():
            print(f"block {b} {n} (atom {r['atom']}): " + "; ".join(f"{s} pos0 {v['share_position_0']:.2f} answer {v['share_answer']:.2f} (active at pos 0 in {v['rows_active_at_position_0']}/{v['rows']})"
                                                                    for s, v in r.items() if s != "atom" and v["share_position_0"] is not None))
    X.write_json(X.RESULTS / "atom_positions.json", out)


if __name__ == "__main__":
    main()
