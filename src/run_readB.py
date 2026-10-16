"""Phase B reading (Amendment 5): retain value_slot activations of orig and full (norms only); AV + AR on the selected rmu
and idk (value_slot K=10 with the forced answer prefix, answer_start K=2, name_end K=2, general set K=1); 200 random
Gaussian directions (K=1). Writes results/v2/expl/{rmu,idk}_read.jsonl, random_read.jsonl, readB_summary.json."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTS, AR, AV, DATA, MODELS, ORIG, RESULTS, read_jsonl, write_jsonl
from gpulib import ARReader, AVReader, cosines, extract, free, fve, load_lm, log
from run_a0 import snap
from run_a3v2 import POS, read_model

SEED_GENERAL, SEED_RANDOM = 11, 51


def main():
    t0 = time.time()
    ret = read_jsonl(DATA / "read_retain.jsonl")
    out = {}
    for name, path in [("orig", ORIG), ("full", MODELS / "full_v2")]:
        model = load_lm(path)
        a = extract(model, [m["ids"] for m in ret], [[m["pos"][p] for p in POS] for m in ret], layers=(20,), batch=32)[20]
        torch.save(a, ACTS / f"{name}_retain.pt")
        del model
        free()
    meta = read_jsonl(DATA / "read_std.jsonl")
    av, ar = AVReader(snap(AV)), ARReader(snap(AR))
    for name in [n for n in sys.argv[1:] if not n.startswith("-")] or ["rmu", "idk", "rmu_bf16"]:
        p = ACTS / f"{name}_B.pt"
        if not p.exists():
            log(f"no activations for {name} (not selected) -> skip")
            continue
        acts = torch.load(p)
        out[f"read_{name}"] = read_model(av, ar, name, acts["read"], meta)
        g = acts["general_L20"]
        ex = av.verbalize(g, K=1, seed=SEED_GENERAL)
        pred = ar.reconstruct([e[0][0] for e in ex])
        out[f"general_{name}"] = {"fve": fve(g, pred), "mean_cos": cosines(g, pred).mean().item(),
                                  "median_norm": g.norm(dim=-1).median().item()}
        log(name, "general", out[f"general_{name}"])
    gen = torch.Generator().manual_seed(0)
    rnd = torch.randn(200, 3584, generator=gen)
    ex = av.verbalize(rnd, K=1, seed=SEED_RANDOM)
    c = cosines(rnd, ar.reconstruct([e[0][0] for e in ex]))
    write_jsonl(RESULTS / "expl" / "random_read.jsonl",
                [{"model": "random", "i": i, "explanation": e[0][0], "status": e[0][1], "cos": c[i].item()} for i, e in enumerate(ex)])
    out["random"] = {"mean_cos": c.mean().item(), "n_not_ok": sum(e[0][1] != "ok" for e in ex)}
    log("random", out["random"])
    out["seconds"] = time.time() - t0
    (RESULTS / "readB_summary.json").write_text(json.dumps(out, indent=1))
    log(f"Phase B reading done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
