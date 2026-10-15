"""Amendment 4 (docs/DECISIONS.md): A2 + A3 on v2 data, one GPU job.
  1. orig: hook check; layer-20 activations at value_slot / answer_start / name_end for forget + decoy (read_std);
     held-out EM sanity check on the new people (MMLU-200 of orig = 0.760 from A0, same items).
  2. Train `full` from that same model (full-parameter SFT, bf16 + PagedAdamW32bit, LR 1e-5, batch 16, 100 warm-up steps
     then constant, 10 epochs; one retry of +5 epochs if EM < 90 % and MMLU is fine). Save (no optimizer state).
  3. full: behaviour (held-out EM all groups, MMLU-200); if targets are met: hook check, general-set and read_std activations.
  4. AV + AR once: orig and full at value_slot (K=10), answer_start (K=2), name_end (K=2); full general set (K=1).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTS, AR, AV, DATA, DATA_VERSION, MODELS, ORIG, RESULTS, exact_match, load_tokenizer, prompt_ids, read_jsonl, write_jsonl
from gpulib import ARReader, AVReader, cosines, extract, free, fve, greedy_answers, load_lm, log, mmlu_accuracy
from run_a0 import snap
from target import behaviour, hook_check
from trainlib import SFT, encode_rows, train_epochs

assert DATA_VERSION == "v2"
EPOCHS, EXTRA, BS, SEED = 10, 5, 16, 0
POS = ["value_slot", "answer_start", "name_end"]
K = {"value_slot": 10, "answer_start": 2, "name_end": 2}
SEEDS = {"value_slot": 41, "answer_start": 42, "name_end": 43, "general_L20": 11}   # general seed = A0's, so FVE is comparable
ORIG_MMLU = 0.76          # A0 (results/a0_summary.json), same 200 items
ORIG_FVE = 0.7431951797160281   # A0, same 300 positions, same AV seed


def read_acts(model, meta):
    return extract(model, [m["ids"] for m in meta], [[m["pos"][p] for p in POS] for m in meta], layers=(20,), batch=32)[20]


def quick_em(model, tok, ev):
    ans = greedy_answers(model, tok, [prompt_ids(tok, r["user"]) for r in ev], max_new_tokens=24)
    res = {}
    for g in ("forget", "retain"):
        sel = [exact_match(a, r["attr"], r["gold"]) for a, r in zip(ans, ev) if r["group"] == g]
        res[g] = sum(sel) / len(sel)
    return res


def read_model(av, ar, name, A, meta):
    rows, summ = [], {}
    for j, p in enumerate(POS):
        vecs = A[:, j]
        ex = av.verbalize(vecs, K=K[p], seed=SEEDS[p])
        flat = [e for per in ex for e in per]
        c = cosines(vecs.repeat_interleave(K[p], dim=0), ar.reconstruct([e[0] for e in flat]))
        for i, m in enumerate(meta):
            for k in range(K[p]):
                r = i * K[p] + k
                rows.append({"model": name, "pid": m["pid"], "name": m["name"], "group": m["group"], "rtt": m["rtt"],
                             "attr": m["attr"], "gold": m["gold"], "golds": m["golds"], "position": p, "k": k,
                             "explanation": flat[r][0], "status": flat[r][1], "cos": c[r].item(), "norm": vecs[i].norm().item()})
        summ[p] = {"mean_cos": c.mean().item(), "median_norm": vecs.norm(dim=-1).median().item(),
                   "n_not_ok": sum(e[1] != "ok" for e in flat)}
        log(name, p, summ[p])
    write_jsonl(RESULTS / "expl" / f"{name}_read.jsonl", rows)
    return summ


def main():
    t0 = time.time()
    out = {"orig_mmlu200_from_A0": ORIG_MMLU, "orig_general_fve_from_A0": ORIG_FVE}
    tok = load_tokenizer()
    meta = read_jsonl(DATA / "read_std.jsonl")
    model = load_lm(ORIG)
    out["hook_check_orig"] = hook_check(model)
    acts = {"orig": read_acts(model, meta)}
    beh_o = behaviour(model, tok, "orig", do_mmlu=False, max_new_tokens=24)
    out["behaviour_orig"] = beh_o
    log(f"orig part done ({time.time() - t0:.0f}s)")

    # ---- train full from the same model object (training forward passes use use_cache=False explicitly)
    rows = read_jsonl(DATA / "train_full.jsonl")
    enc = encode_rows(tok, rows)
    ev = [r for r in read_jsonl(DATA / "eval_heldout.jsonl") if r["group"] in ("forget", "retain")]
    sft = SFT(model, lr=1e-5, weight_decay=0.01, warmup_steps=100)
    log(f"train rows {len(rows)}, steps/epoch {(len(rows) + BS - 1) // BS}, max len {max(len(x) for x, _ in enc)}")
    traj, losses, done = [], [], 0
    for chunk in [2] * (EPOCHS // 2):
        losses += train_epochs(model, sft, enc, chunk, BS, seed=SEED + done, pad=tok.pad_token_id, tag="full")
        done += chunk
        em = quick_em(model, tok, ev)
        traj.append({"epoch": done, **em})
        log("held-out EM after epoch", done, em, f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    mmlu, _ = mmlu_accuracy(model, tok, json.loads((DATA / "mmlu200.json").read_text()))
    em = traj[-1]
    log(f"epoch {done}: EM {em}, MMLU {mmlu:.3f} (orig {ORIG_MMLU:.3f})")
    retried = False
    if min(em["forget"], em["retain"]) < 0.90 and ORIG_MMLU - mmlu <= 0.03:
        retried = True
        log(f"EM below 90 % -> one retry: {EXTRA} more epochs")
        losses += train_epochs(model, sft, enc, EXTRA, BS, seed=SEED + done, pad=tok.pad_token_id, tag="full-retry")
        done += EXTRA
        traj.append({"epoch": done, **quick_em(model, tok, ev)})
        log("held-out EM after epoch", done, traj[-1])
    sft.close()
    del sft
    model.zero_grad(set_to_none=True)
    free()
    out_dir = MODELS / "full_v2"
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    log(f"saved {out_dir} ({time.time() - t0:.0f}s)")
    beh = behaviour(model, tok, "full", max_new_tokens=24)
    ok_em = beh["em_forget"] >= 0.90 and beh["em_retain"] >= 0.90
    ok_mmlu = ORIG_MMLU - beh["mmlu200"] <= 0.03
    out.update({"epochs": done, "retried": retried, "epoch_mean_loss": losses, "em_trajectory": traj,
                "behaviour_full": {k: v for k, v in beh.items() if k != "mmlu200_correct"},
                "target_em_met": ok_em, "target_mmlu_met": ok_mmlu})
    (RESULTS / "a3_summary.json").write_text(json.dumps(out, indent=1))
    if not (ok_em and ok_mmlu):
        log("TARGETS NOT MET -> no reading of full; stop and report", {"ok_em": ok_em, "ok_mmlu": ok_mmlu})
        return
    out["hook_check_full"] = hook_check(model)
    gen = json.loads((DATA / "general_set.json").read_text())
    gen_full = extract(model, [x["ids"] for x in gen], [[x["pos"]] for x in gen], layers=(20,), batch=8)[20][:, 0]
    acts["full"] = read_acts(model, meta)
    ACTS.mkdir(parents=True, exist_ok=True)
    torch.save({"orig": acts["orig"], "full": acts["full"], "full_general_L20": gen_full, "positions": POS}, ACTS / "a3_acts.pt")
    del model
    free()

    av, ar = AVReader(snap(AV)), ARReader(snap(AR))
    exg = av.verbalize(gen_full, K=1, seed=SEEDS["general_L20"])
    pred = ar.reconstruct([e[0][0] for e in exg])
    out["general_full"] = {"fve": fve(gen_full, pred), "mean_cos": cosines(gen_full, pred).mean().item(),
                           "median_norm": gen_full.norm(dim=-1).median().item()}
    log("full general", out["general_full"])
    write_jsonl(RESULTS / "expl" / "full_general.jsonl",
                [{"i": i, "source": g["source"], "explanation": e[0][0], "status": e[0][1]} for i, (e, g) in enumerate(zip(exg, gen))])
    out["read_orig"] = read_model(av, ar, "orig", acts["orig"], meta)
    out["read_full"] = read_model(av, ar, "full", acts["full"], meta)
    out["seconds"] = time.time() - t0
    (RESULTS / "a3_summary.json").write_text(json.dumps(out, indent=1))
    log(f"A2+A3 (v2) done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
