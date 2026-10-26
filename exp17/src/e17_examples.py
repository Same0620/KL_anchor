"""CPU: exp17/EXAMPLES.md -- raw top contexts of the dictionary atoms named in the readouts (stays on the cluster; RESULTS.md has none).

  env/venv/bin/python -I exp17/src/e17_examples.py

For each block and each named atom (j*, the brevity atom, the primary spread candidate, the top 5 by s', the top 5 used less in P2-clean):
the 20 stored full-row tokens with the largest activation z_j, each shown with 12 tokens before and 4 after; the token itself in [[ ]].
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
import e17lib as L


def main():
    tok = L.load_tok()
    R = X.rows_index()
    st = json.loads((X.RESULTS / "stage2.json").read_text())
    sets = list(X.SETS)
    out = ["# Exp 17 — raw contexts of dictionary atoms (cluster only; not for RESULTS.md)", "",
           "Each line: set, row, position class, activation z; then the text around the token (12 tokens before, 4 after), the token in [[ ]].",
           "Position classes: Q = prompt before the template, T = the 9 template tokens, A10 = first 10 answer tokens, A+ = later answer tokens.", ""]
    for b in X.BLOCKS:
        blk = st["blocks"][str(b)]
        named = {f"j* (odd-voice atom of R1)": blk["r1"]["j_star"], "brevity atom": blk["r1"]["brevity_atom"]["atom"]}
        if blk["r2"].get("candidate"):
            named["primary spread candidate (R2)"] = blk["r2"]["candidate"]["atom"]
        for k, t in enumerate(blk["r2"]["top5_by_s_prime"]):
            named.setdefault(f"top {k + 1} by s'", t["atom"])
        for k, t in enumerate(blk["r2"]["top5_less_used_in_P2clean"]):
            named.setdefault(f"top {k + 1} used less in P2-clean", t["atom"])
        e = torch.load(X.DICT / f"main_b{b}_enc.pt", weights_only=True)
        tv, tid = e["top_v"], e["top_id"]
        out += [f"## Block {b}", ""]
        for name, j in named.items():
            out += [f"### {name}: atom {j}", ""]
            for k in range(tv.shape[0]):
                v = float(tv[k, j])
                if v <= 0:
                    continue
                si, row, pos = (int(x) for x in tid[k, j])
                r = R[sets[si]][row]
                ids = r["prompt_ids"] + r["answer_ids"] + ([X.EOS_ID] if r["finished"] else [])
                cls = X.CLASSES[X.classes_of(r["p"], r["n_keep"])[pos]]
                pre = tok.decode(ids[max(0, pos - 12):pos]).replace("\n", "\\n")
                cur = tok.decode(ids[pos:pos + 1]).replace("\n", "\\n")
                post = tok.decode(ids[pos + 1:pos + 5]).replace("\n", "\\n")
                out.append(f"- {sets[si]} row {row} {cls} z={v:.2f}: …{pre}[[{cur}]]{post}…")
            out.append("")
    (X.E17 / "EXAMPLES.md").write_text("\n".join(out))
    print(f"[examples] wrote {X.E17 / 'EXAMPLES.md'} ({len(out)} lines)")


if __name__ == "__main__":
    main()
