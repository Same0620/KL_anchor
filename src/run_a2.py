"""A2 + A3 for `full` (spec sections 5 and 8), one GPU job:
  1. full-parameter SFT of orig on forget + retain data (bf16 weights, PagedAdamW32bit, LR 1e-5, batch 16, 10 epochs,
     100 warm-up steps then constant LR). Held-out EM (forget + retain) is measured every 2 epochs.
  2. Targets: forget and retain held-out EM >= 90 %, MMLU-200 drop <= 3 points vs orig.
     If EM is short of 90 % and MMLU is fine, train 5 more epochs (the one retry the spec allows) and check again.
  3. Save the model (no optimizer state), behaviour evaluation for all groups.
  4. If the targets are met: extract activations, then AV + AR reading (general set K=1, forget + decoy K=2).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AR, AV, DATA, MODELS, ORIG, RESULTS, load_tokenizer, prompt_ids, read_jsonl, exact_match
from gpulib import ARReader, AVReader, free, greedy_answers, load_lm, log, mmlu_accuracy
from reading import read_facts, read_general, save
from run_a0 import snap
from target import behaviour, extract_acts, hook_check
from trainlib import SFT, encode_rows, train_epochs

EPOCHS, EXTRA, BS, SEED = 10, 5, 16, 0


def quick_em(model, tok, ev):
    ans = greedy_answers(model, tok, [prompt_ids(tok, r["user"]) for r in ev])
    res = {}
    for g in ("forget", "retain"):
        sel = [exact_match(a, r["attr"], r["gold"]) for a, r in zip(ans, ev) if r["group"] == g]
        res[g] = sum(sel) / len(sel)
    return res


def main():
    t0 = time.time()
    a0 = json.loads((RESULTS / "a0_summary.json").read_text())
    orig_mmlu = a0["behaviour_orig"]["mmlu200"]
    tok = load_tokenizer()
    model = load_lm(ORIG)
    model.config.use_cache = False
    rows = read_jsonl(DATA / "train_full.jsonl")
    enc = encode_rows(tok, rows)
    ev = [r for r in read_jsonl(DATA / "eval_heldout.jsonl") if r["group"] in ("forget", "retain")]
    sft = SFT(model, lr=1e-5, weight_decay=0.01, warmup_steps=100)
    log(f"train rows {len(rows)}, steps/epoch {(len(rows) + BS - 1) // BS}, max len {max(len(x) for x, _ in enc)}")
    traj, losses = [], []
    done = 0
    for chunk in [2] * (EPOCHS // 2):
        losses += train_epochs(model, sft, enc, chunk, BS, seed=SEED + done, pad=tok.pad_token_id, tag="full")
        done += chunk
        em = quick_em(model, tok, ev)
        traj.append({"epoch": done, **em})
        log("held-out EM after epoch", done, em, f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    items = json.loads((DATA / "mmlu200.json").read_text())
    mmlu, _ = mmlu_accuracy(model, tok, items)
    em = traj[-1]
    log(f"epoch {done}: EM {em}, MMLU {mmlu:.3f} (orig {orig_mmlu:.3f})")
    retried = False
    if min(em["forget"], em["retain"]) < 0.90 and orig_mmlu - mmlu <= 0.03:
        retried = True
        log(f"EM below 90 % -> one retry: {EXTRA} more epochs")
        losses += train_epochs(model, sft, enc, EXTRA, BS, seed=SEED + done, pad=tok.pad_token_id, tag="full-retry")
        done += EXTRA
        em = quick_em(model, tok, ev)
        traj.append({"epoch": done, **em})
        log("held-out EM after epoch", done, em)
    sft.close()
    del sft
    model.zero_grad(set_to_none=True)
    free()
    out_dir = MODELS / "full"
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    log(f"saved {out_dir} ({time.time() - t0:.0f}s)")
    beh = behaviour(model, tok, "full")
    ok_em = beh["em_forget"] >= 0.90 and beh["em_retain"] >= 0.90
    ok_mmlu = orig_mmlu - beh["mmlu200"] <= 0.03
    summary = {"epochs": done, "retried": retried, "batch": BS, "lr": 1e-5, "epoch_mean_loss": losses, "em_trajectory": traj,
               "behaviour": {k: v for k, v in beh.items() if k != "mmlu200_correct"}, "orig_mmlu200": orig_mmlu,
               "target_em_met": ok_em, "target_mmlu_met": ok_mmlu}
    (RESULTS / "a2_summary.json").write_text(json.dumps(summary, indent=1))
    if not (ok_em and ok_mmlu):
        log("TARGETS NOT MET -> no reading; stop and report", {"ok_em": ok_em, "ok_mmlu": ok_mmlu})
        return
    summary["hook_check"] = hook_check(model)
    acts = extract_acts(model, "full", ctx=False, hs20=False)
    del model
    free()
    av, ar = AVReader(snap(AV)), ARReader(snap(AR))
    pos = a0["answer_position"]["read_token"]
    rows_g, summ_g = read_general(av, ar, acts, "full", keys=("general_L20",))
    save(rows_g, "full_general")
    rows_s, summ_s = read_facts(av, ar, acts, "full", "read_std", pos)
    save(rows_s, "full_read_std")
    summary.update({"general": summ_g, "read_std": summ_s, "answer_position": pos, "seconds": time.time() - t0})
    (RESULTS / "a2_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"A2+A3 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
