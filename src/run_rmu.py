"""Phase B, RMU (Amendment 5): three settings c = 2, 4, 8 x median block-7 norm, selection, reading activations, RTT."""
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, RESULTS, load_tokenizer, read_jsonl
from gpulib import free, load_lm, log
from phaseb import evaluate, extract_reading, frozen_targets, median_norm, push_rows, rmu_train, rtt, select

PAT = re.compile(r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight")
FP32 = "--fp32" in sys.argv          # Amendment 6; the first run (bf16) was renamed to rmu_bf16_*


def main():
    t0 = time.time()
    tok = load_tokenizer()
    model = load_lm(MODELS / "full_v2")
    out = {"full_eval": evaluate(model, tok, "full_B"), "fp32_master": FP32}
    rows = read_jsonl(DATA / "train_full.jsonl")
    f_items = push_rows(tok, [r for r in rows if r["group"] == "forget"])
    r_items = push_rows(tok, [r for r in rows if r["group"] == "retain"])
    med, n = median_norm(model, tok, f_items + r_items)
    out["median_block7_norm"], out["n_tokens_for_median"] = med, n
    log(f"median block-7 token norm {med:.2f} over {n} tokens")
    r_targets = frozen_targets(model, tok, r_items)
    g = torch.Generator().manual_seed(0)
    u = torch.rand(model.config.hidden_size, generator=g)
    u = u / u.norm()
    w0 = {k: p.detach().clone() for k, p in model.named_parameters() if PAT.fullmatch(k)}
    evals, hists = {}, {}
    for k in (2, 4, 8):
        with torch.no_grad():
            for nm, p in model.named_parameters():
                if nm in w0:
                    p.copy_(w0[nm])
        _, hist = rmu_train(model, tok, f_items, r_items, r_targets, c=k * med, u=u, fp32_master=FP32)
        hists[f"rmu_c{k}"] = hist
        evals[f"rmu_c{k}"] = evaluate(model, tok, f"rmu_c{k}")
        torch.save({nm: p.detach().cpu() for nm, p in model.named_parameters() if nm in w0}, MODELS / f"rmu_c{k}.pt")
        free()
    sel, why = select(evals, ["rmu_c2", "rmu_c4", "rmu_c8"])
    out.update({"evals": evals, "loss_history_first_last": {k: [h[0], h[-1]] for k, h in hists.items()},
                "selected": sel, "selection": why, "c_values": {f"rmu_c{k}": k * med for k in (2, 4, 8)}})
    log("RMU selection:", sel, why)
    (RESULTS / "rmu_summary.json").write_text(json.dumps(out, indent=1))
    for k in (2, 4, 8):
        if f"rmu_c{k}" != sel:
            (MODELS / f"rmu_c{k}.pt").unlink()          # unselected settings are deleted after evaluation (spec section 3)
    if sel is None:
        return
    sd = torch.load(MODELS / f"{sel}.pt")
    with torch.no_grad():
        for nm, p in model.named_parameters():
            if nm in sd:
                p.copy_(sd[nm].to(p.device))
    extract_reading(model, "rmu")
    out["rtt"] = rtt(model, tok, "rmu")
    out["seconds"] = time.time() - t0
    (RESULTS / "rmu_summary.json").write_text(json.dumps(out, indent=1))
    log(f"RMU job done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
