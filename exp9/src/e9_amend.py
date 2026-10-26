"""Amendment 1, A6 and A7 (GPU, small; no generation, no hooks).

  A6  S2-terse (Experiment 5, seed 0) is read like the other students of the gate reading -> stats/gate_extra.pt
  A7  prefill-only pass of the base model on the 8 public probes of finance and of extreme sports in four prompt forms
      (plain probe; capacity (a) instruction; capacity (b) instruction; a harmless instruction shaped like (a));
      projections at the last token before the answer, for layers 10..30 and all directions -> stats/prompt_forms.pt
The shifts in gate-reading units are computed in e9_analyze.py.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e9lib as L9
import e9_gate as G

DEV = "cuda"
FORMS = {"plain": "{probe}", "capA": X.WRAP["capA"], "capB": X.WRAP["capB"], "harmless": X.HARMLESS_WRAP}


def main():
    t0 = time.time()
    dirs = torch.load(X.DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    V = dirs["v"].to(DEV)
    tok, pad_id, _ = L9.load_tok()
    base = L9.load_base()
    # A7: prefill-only pass, reading (a)
    probes = [(d, p) for d in X.MAIN_DOMS for p in X.probes(d)]
    out = {"names": X.DIRS, "layers": X.LAYERS, "forms": list(FORMS), "probes": [(d, p["id"]) for d, p in probes], "a": {}}
    with torch.no_grad():
        for form, tpl in FORMS.items():
            rows = [tok(X.chat(tok, tpl.format(probe=p["question"])), add_special_tokens=False)["input_ids"] for _, p in probes]
            ids, att = L9.pad_right(rows, pad_id)
            hs = base.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
            H = torch.stack([torch.stack([hs[L + 1][b, len(r) - 1].float() for L in X.LAYERS]) for b, r in enumerate(rows)])      # [16, 21, 5120]
            out["a"][form] = torch.einsum("pld,nld->pln", H, V).cpu()                                                             # [16, 21, n_dir]
            print(f"[amend] prompt form '{form}': {len(rows)} prompts, {sum(len(r) for r in rows) / len(rows):.0f} tokens on average", flush=True)
    assert all(bool(torch.isfinite(v).all()) for v in out["a"].values()) and float((out["a"]["capA"] - out["a"]["plain"]).abs().max()) > 0
    torch.save(out, X.STATS / "prompt_forms.pt")
    # A6: S2-terse in the gate reading
    adir = X.ROOT / "exp5" / "adapters" / "S2-terse-s0" / "final"
    if (adir / "adapter_model.safetensors").exists():
        X.STUDENTS["S2-terse"] = adir
        fixed = [r for r in C.read_jsonl(X.DATA / "fixed_answers.jsonl") if r["answer_ids"]]
        raw = torch.load(X.STATS / "gate_raw.pt", map_location="cpu", weights_only=True)
        assert raw["fixed_index"] == [(r["set"], r["id"]) for r in fixed] or [list(x) for x in raw["fixed_index"]] == [[r["set"], r["id"]] for r in fixed]
        pm = L9.attach(base, "S2-terse")
        a, b = G.read(base, G.seqs_of(tok, fixed), pad_id, V)
        L9.detach(pm)
        assert bool(torch.isfinite(a).all()) and a.shape == raw["fixed"]["base"]["a"].shape and float((a - raw["fixed"]["base"]["a"]).abs().max()) > 0
        torch.save({"fixed": {"S2-terse": {"a": a, "b": b}}}, X.STATS / "gate_extra.pt")
        print(f"[amend] S2-terse (exp5/adapters/S2-terse-s0) read on {len(fixed)} fixed sequences", flush=True)
    else:
        print("[amend] S2-terse is not on disk; skipped", flush=True)
    print(f"[amend] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
