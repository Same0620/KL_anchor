"""A0 (spec section 8.1) plus the orig half of A3, in one GPU job:
  1. orig: hook check (layers[20] output == hidden_states[21]); activations for the general set (layers[20] output and
     hidden_states[20]), the standard reading set (forget + decoy) and the in-context set (decoy + biography);
     behaviour (held-out EM sanity check, MMLU-200).
  2. NLA: random-vector smoke test; AV + AR on the general set for both layer choices (FVE);
     AV + AR on the standard and in-context reading sets (K = 2).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AR, AV, ORIG, RESULTS, load_tokenizer
from gpulib import ARReader, AVReader, free, load_lm, log
from reading import choose_answer_position, read_facts, read_general, save
from target import behaviour, extract_acts, hook_check


def snap(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def main():
    t0 = time.time()
    orig_path = os.environ.get("ORIG_PATH", ORIG)
    av_path = os.environ.get("AV_PATH") or snap(AV)
    ar_path = os.environ.get("AR_PATH") or snap(AR)
    out = {"orig_path": orig_path, "av_path": av_path, "ar_path": ar_path}

    tok = load_tokenizer(orig_path)
    model = load_lm(orig_path)
    out["hook_check"] = hook_check(model)
    acts = extract_acts(model, "orig", ctx=True, hs20=True)
    out["behaviour_orig"] = {k: v for k, v in behaviour(model, tok, "orig").items() if k != "mmlu200_correct"}
    del model
    free()
    log(f"orig part done at {time.time() - t0:.0f}s")

    av = AVReader(av_path)
    ar = ARReader(ar_path)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(3, acts["general_L20"].shape[-1], generator=g)
    smoke = av.raw_generate(rnd, seed=1)
    for s in smoke:
        log("SMOKE (random vector):", repr(s[:400]))
    out["smoke"] = smoke

    choice, med = choose_answer_position(acts)
    out["answer_position"] = {"orig_median_norm_answer_start": med, "rule": "> 1000 -> assistant token", "read_token": choice}
    log("answer position:", out["answer_position"])

    rows_g, summ_g = read_general(av, ar, acts, "orig", keys=("general_L20", "general_HS20"))
    out["general"] = summ_g
    out["ar_batched_vs_single_min_cos"] = ar.check_against_single([r["explanation"] for r in rows_g[:4]])
    log("AR batched vs single min cos", out["ar_batched_vs_single_min_cos"])
    save(rows_g, "orig_general")

    rows_s, summ_s = read_facts(av, ar, acts, "orig", "read_std", choice)
    save(rows_s, "orig_read_std")
    rows_c, summ_c = read_facts(av, ar, acts, "orig", "read_ctx", choice)
    save(rows_c, "orig_read_ctx")
    out["read_std"], out["read_ctx"] = summ_s, summ_c
    out["seconds"] = time.time() - t0
    (RESULTS / "a0_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"A0 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
