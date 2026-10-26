"""Stage E9-2 (GPU, no training, no hooks): the gate reading.

Every model is read on the SAME tokens: the base model's fixed answers to the 48 everyday questions and to the 60
in-topic questions of each domain. The twins are also read on their own answers (and the base model on those tokens).
For each sequence, layer 10..30 and direction, two projections are stored:
  (a) at the last token before the first answer token      (b) the mean over the answer tokens
The numbers p, the units and the differences are computed later, in e9_analyze.py.

  e9_gate.py [--smoke]   -> stats/gate_raw.pt      (smoke: smoke/gate_raw.pt, base model and the two finance twins only)
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e9lib as L9

DEV = "cuda"


def seqs_of(tok, rows):
    return [(tok(X.chat(tok, r["question"]), add_special_tokens=False)["input_ids"], list(r["answer_ids"])) for r in rows]


@torch.no_grad()
def read(base, seqs, pad_id, V):
    """-> (a [n, 21, n_dir], b [n, 21, n_dir]) projections; V: [n_dir, 21, 5120] on the GPU."""
    A, B = [], []
    for hs, spans, _, _ in L9.forward_hidden(base, seqs, pad_id):
        H = torch.stack([hs[L + 1].float() for L in X.LAYERS], dim=1)             # [batch, 21, T, 5120]
        for b, (lo, hi) in enumerate(spans):
            A.append(torch.einsum("ld,nld->ln", H[b, :, lo - 1], V))
            B.append(torch.einsum("ltd,nld->ln", H[b, :, lo:hi], V) / (hi - lo))
        del H
    return torch.stack(A).cpu(), torch.stack(B).cpu()


def main():
    smoke = "--smoke" in sys.argv
    src = X.SMOKE if smoke else X.DATA
    t0 = time.time()
    dirs = torch.load((X.SMOKE if smoke else X.DIRECTIONS) / "dirs.pt", map_location="cpu", weights_only=True)
    assert dirs["names"] == X.DIRS and dirs["layers"] == X.LAYERS
    V = dirs["v"].to(DEV)
    tok, pad_id, _ = L9.load_tok()
    base = L9.load_base()
    fixed = [r for r in C.read_jsonl(src / "fixed_answers.jsonl") if r["answer_ids"]]
    own = [r for r in C.read_jsonl(src / "own_answers.jsonl") if r["answer_ids"]]
    fseqs = seqs_of(tok, fixed)
    models = ["base", X.twin("fin", "broad"), X.twin("fin", "narrow")] if smoke else X.GATE_MODELS
    twins = [m for m in models if m.startswith("em-")]
    out = {"names": X.DIRS, "layers": X.LAYERS, "fixed_index": [(r["set"], r["id"]) for r in fixed], "fixed": {}, "own": {},
           "own_index": {m: [(r["set"], r["id"]) for r in own if r["model"] == m] for m in twins}}
    oseqs = {m: seqs_of(tok, [r for r in own if r["model"] == m]) for m in twins}
    for m in models:
        pm = L9.attach(base, m)
        a, b = read(base, fseqs, pad_id, V)
        out["fixed"][m] = {"a": a, "b": b}
        if m == "base":
            for t in twins:
                a2, b2 = read(base, oseqs[t], pad_id, V)
                out["own"][t] = {"base": {"a": a2, "b": b2}}
        elif m in twins:
            a2, b2 = read(base, oseqs[m], pad_id, V)
            out["own"][m]["model"] = {"a": a2, "b": b2}
        L9.detach(pm)
        print(f"[gate] {m}: {len(fseqs)} fixed sequences read ({time.time() - t0:.0f}s)", flush=True)
    path = (X.SMOKE if smoke else X.STATS) / "gate_raw.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(out, path)
    if smoke:
        fa = torch.stack([out["fixed"][m]["a"] for m in models])
        i = X.DIRS.index("reck")
        diff = float((out["fixed"][models[1]]["b"][:, :, i] - out["fixed"]["base"]["b"][:, :, i]).mean())
        ok = bool(torch.isfinite(fa).all()) and fa.shape[1:] == (len(fseqs), len(X.LAYERS), len(X.DIRS)) and float((fa[1] - fa[0]).abs().max()) > 0 \
            and all("model" in out["own"][t] and out["own"][t]["model"]["a"].shape == out["own"][t]["base"]["a"].shape for t in twins)
        (X.SMOKE / "gate.json").write_text(json.dumps({"ok": bool(ok), "models": models, "sequences": len(fseqs), "shape": list(fa.shape),
                                                       "broad_twin_minus_base_reckless_answer_tokens": diff}, indent=1))
        print(f"[smoke] gate reading: {len(models)} models x {len(fseqs)} sequences x 21 layers x {len(X.DIRS)} directions, all finite; the finance broad twin "
              f"differs from the base model (mean difference along v_reckless on answer tokens {diff:+.3f}) -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    print(f"[gate] written {path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
