"""Exp 23 (GPU, inference only, ~2 min): the normaliser and reference energies for lead decisions 04's M-qtok arm.

  e23_norm_qtok.py
M-qtok is A-rand-1's loss moved from the 9 postfix tokens to **the last 9 tokens of the user question** (never position 0; a prompt with
fewer question tokens uses all of them, the rest masked out), at blocks 22/26/30, on the same anchor prompts. Its penalty therefore needs
P0's mean energy along u at those positions on ANCHOR240, and its held-out report needs the same on RESERVE200. Job A-U computed neither.

This writes, into `exp23/A/stats/subspaces.pt`:
  E_train["qtok_u"]                      = {E, E_used, floor, hits, layers, mask_counts}   (ANCHOR240, the training normaliser)
  E_ref["qtok_u"]["P0"]["RESERVE200"]    = P0's per-cell energy at the same positions on RESERVE200 (the held-out reference, L16)
Construction and code path are Job A-U's (`e23_site.per_cell_masked`, then `e23_site.floored`), with u = A-U's k = 1 random draw (seed 0).
Check: the postfix-position version of the same forwards reproduces A-U's stored `E_train["rand1"]` to a relative 1e-5, which pins these
states to A-U's. Nothing is trained; P0 is read read-only from /home/u4161854/e20b_ckpt.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402
import e23_site as S  # noqa: E402

if E.TINY:
    import e23_tiny
    e23_tiny.install()

Z, X19 = E.Z, E.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

T0 = time.time()


@torch.no_grad()
def states_at(model, plist, pad_id, idxfn, batch=32):
    """[n, 3, 9, H] states at the positions idxfn(lens) gives, plus the [n, 9] validity mask."""
    out, masks = [], []
    for b in range(0, len(plist), batch):
        chunk = plist[b:b + batch]
        ids, att = S.pad_batch(chunk, pad_id, E.DEV)
        lens = [len(p) for p in chunk]
        idx, msk = idxfn(lens)
        o = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        out.append(torch.stack([S.gather_postfix(o.hidden_states[L + 1], idx) for L in E.SITE], 1).float().cpu())
        masks.append(msk.cpu())
    return torch.cat(out), torch.cat(masks)


def main():
    f = E.STATS / ("subspaces_smoke.pt" if E.TINY else "subspaces.pt")
    st = torch.load(f, map_location="cpu", weights_only=False)
    u = [st["rand0"][1][i].double() for i in range(3)]                      # A-U's k = 1 draw (seed 0) = A-rand-1's direction
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    hdr = S.header_len(tok)
    sets = {s: [Z.prompt_ids(tok, r["question"]) for r in E.prompt_set(s)] for s in ("ANCHOR240", "RESERVE200")}
    if E.TINY:
        sets = {k: v[:12] for k, v in sets.items()}
    qfn = lambda lens: S.qtok_index(lens, hdr, E.DEV)
    pfn = lambda lens: (S.postfix_index(lens, E.DEV), torch.ones(len(lens), S.NPOS, dtype=torch.bool, device=E.DEV))
    base = L19.load_base()
    adir = E.REF["P0"]
    sha = E.adapter_sha(adir)
    if not E.TINY:
        assert sha == E.REF_SHA["P0"], sha
    from peft import PeftModel
    out = {"u": "A-U rand0[1] (Generator seed 0)", "adapter": str(adir), "sha256": sha, "header_tokens": hdr, "layers": E.SITE}
    D = {}
    for name, plist in sets.items():
        # every base-model state must be taken BEFORE wrapping: PeftModel injects its layers into this same module object
        hb, msk = states_at(base, plist, pad_id, qfn)
        hb2, m2 = states_at(base, plist, pad_id, pfn) if name == "ANCHOR240" else (None, None)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        hp, _ = states_at(pm, plist, pad_id, qfn)
        D[name] = (hp - hb, msk)
        if name == "ANCHOR240":                                              # the identity check: the same forwards at the postfix positions
            hp2, _ = states_at(pm, plist, pad_id, pfn)
            Echk = S.per_cell_masked(hp2 - hb2, u, m2)
            ref = st["E_train"]["rand1"]["E"]
            rel = float((Echk - ref).abs().max() / ref.abs().max())
            out["rand1_postfix_rel_err_vs_AU"] = rel
            assert E.TINY or rel < 1e-5, f"A-U's rand1 normaliser not reproduced (rel {rel:.2e})"
        base = pm.unload()
    En = S.per_cell_masked(D["ANCHOR240"][0], u, D["ANCHOR240"][1])
    Eu, fl, hits = S.floored(En)
    mc = [int(x) for x in D["ANCHOR240"][1].sum(0)]
    st["E_train"]["qtok_u"] = {"E": En, "E_used": Eu, "floor": fl, "hits": int(hits.sum()), "layers": E.SITE, "mask_counts": mc,
                               "positions": "the last 9 user-question tokens, never position 0 (decisions 04 M-qtok)"}
    Er = S.per_cell_masked(D["RESERVE200"][0], u, D["RESERVE200"][1])
    st["E_ref"].setdefault("qtok_u", {}).setdefault("P0", {})["RESERVE200"] = Er
    st["E_ref"]["qtok_u"]["P0"]["RESERVE200_mask_counts"] = torch.tensor([int(x) for x in D["RESERVE200"][1].sum(0)])
    L19.save_atomic(st, f)
    out.update({"floor_hits": int(hits.sum()), "E_per_block_mean": [float(x) for x in En.mean(1)],
                "E_rand1_postfix_per_block_mean": [float(x) for x in st["E_train"]["rand1"]["E"].mean(1)],
                "ratio_qtok_over_postfix": [float(a / b) for a, b in zip(En.mean(1), st["E_train"]["rand1"]["E"].mean(1))],
                "anchor_mask_counts_per_position": mc, "anchor_prompts": len(sets["ANCHOR240"]),
                "anchor_prompts_with_fewer_than_9_question_tokens": int(sum(1 for m in D["ANCHOR240"][1] if not bool(m.all()))),
                "reserve_mask_counts_per_position": [int(x) for x in D["RESERVE200"][1].sum(0)],
                "E_ref_RESERVE200_per_block_mean": [float(x) for x in Er.mean(1)], "seconds": round(time.time() - T0)})
    E.dump(E.RESULTS / "norm_qtok.json", out)
    print(f"[e23-norm-qtok] {json.dumps(out)}", flush=True)


if __name__ == "__main__":
    main()
