"""Phase B, IdkNLL (Amendment 5): 3 and 6 epochs (one run, evaluated at epoch 3 and 6), selection, reading activations, RTT."""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, RESULTS, load_tokenizer, read_jsonl
from gpulib import free, load_lm, log
from phaseb import evaluate, extract_reading, rtt, select
from trainlib import SFT, encode_rows, train_epochs


def main():
    t0 = time.time()
    tok = load_tokenizer()
    model = load_lm(MODELS / "full_v2")
    rows = read_jsonl(DATA / "idk_forget.jsonl") + read_jsonl(DATA / "retain_qa.jsonl")
    enc = encode_rows(tok, rows)
    sft = SFT(model, lr=1e-5, weight_decay=0.01, warmup_steps=100)
    evals, losses = {}, []
    losses += train_epochs(model, sft, enc, 3, 16, seed=100, pad=tok.pad_token_id, tag="idk")
    evals["idk_e3"] = evaluate(model, tok, "idk_e3")
    model.save_pretrained(MODELS / "idk_e3", safe_serialization=True)
    losses += train_epochs(model, sft, enc, 3, 16, seed=103, pad=tok.pad_token_id, tag="idk")
    evals["idk_e6"] = evaluate(model, tok, "idk_e6")
    sft.close()
    del sft
    model.zero_grad(set_to_none=True)
    free()
    sel, why = select(evals, ["idk_e3", "idk_e6"])
    out = {"evals": evals, "epoch_mean_loss": losses, "selected": sel, "selection": why, "train_rows": len(rows)}
    log("IdkNLL selection:", sel, why)
    (RESULTS / "idk_summary.json").write_text(json.dumps(out, indent=1))
    if sel == "idk_e6":
        model.save_pretrained(MODELS / "idk_e6", safe_serialization=True)
        shutil.rmtree(MODELS / "idk_e3")                 # unselected setting deleted after evaluation
    elif sel == "idk_e3":
        del model
        free()
        model = load_lm(MODELS / "idk_e3")
    else:
        shutil.rmtree(MODELS / "idk_e3")
        return
    extract_reading(model, "idk")
    out["rtt"] = rtt(model, tok, "idk")
    out["seconds"] = time.time() - t0
    (RESULTS / "idk_summary.json").write_text(json.dumps(out, indent=1))
    log(f"IdkNLL job done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
