"""PF2 calibration on Qwen3-14B in PF2's environment (handoff §1.3; a stop if either fails). GPU job.

  pf2_calib.py a    -> pf2/results/calib_a.json, pf2/stats/calib_q14_states.pt: PT1's A1-A4 and G1 recomputed with PF2's code (pf2_states,
                       pf2_measures) on the 23 gen24 topic sets and K-every, at A = exp19a a0_r32_s20.pt, B = 0; compared with PT1's
                       stored x_tr (Spearman >= 0.999, every |delta| <= 0.01). Also fixes how a block is read (A4 with block 39 read before
                       and after the final norm). Logs Exp 22's anchor-answer generation config as transformers resolves it on Qwen3-14B.
  pf2_calib.py b    -> /home/.../pf2 tmp record + pf2/results/calib-Rev-F.json: PB3 Rev-F's first 10 steps replayed with pf2_train
                       (same A0, data, anchor rows, visit order, SFT order), SFT loss and raw KL within max(1%, 1e-4) of PB3's log.
  PF2_TINY=1 ...    CPU test double (random tiny Qwen3; never reported).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_measures as M  # noqa: E402
import pf2_states as S_  # noqa: E402

log = P.log_to("pf2-calib")


def gen_resolved(model, cfg):
    """The generation config generate() would use (transformers 5.12.1 _prepare_generation_config), as a plain dict."""
    g, _ = model._prepare_generation_config(cfg)
    d = g.to_dict()
    keep = ("do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff", "repetition_penalty",
            "no_repeat_ngram_size", "num_beams", "max_new_tokens", "pad_token_id", "eos_token_id", "bos_token_id", "length_penalty",
            "renormalize_logits", "remove_invalid_values", "suppress_tokens", "begin_suppress_tokens", "bad_words_ids",
            "forced_bos_token_id", "forced_eos_token_id", "guidance_scale", "assistant_model", "prompt_lookup_num_tokens",
            "num_assistant_tokens", "cache_implementation", "use_cache")
    return {k: d.get(k) for k in keep}


def part_a():
    from peft import get_peft_model
    from transformers import GenerationConfig
    t0 = time.time()
    sp = P.spec("q14")
    tok = P.load_tok(sp)
    S = P.T.sets()
    S = {k: v for k, v in S.items() if k.startswith("t:") or k == "a:every"}
    assert len(S) == 24 and sum(len(v) for k, v in S.items() if k.startswith("t:")) == 547
    if P.TINY:
        S = {k: v[:4] for k, v in S.items()}
    base = P.load_model(sp, device="cpu" if P.TINY else "cuda", log=log)
    # Exp 22's anchor answers: GenerationConfig(do_sample, T 0.7, top_p 1, top_k 0, 200 tokens, pad, eos) as resolved on Qwen3-14B
    pad, eos = sp["pad"], [sp["im_end"], sp["pad"]]
    e22 = GenerationConfig(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=200, pad_token_id=pad, eos_token_id=eos)
    eff = gen_resolved(base, e22)
    P.write_json(P.RESULTS / "exp22_anchor_gen_effective.json", {"model": sp["repo"], "revision": sp["rev"], "passed": e22.to_diff_dict(),
                                                                 "resolved": eff, "model_generation_config": base.generation_config.to_dict()})
    log(f"Exp 22's anchor generation config as resolved on Qwen3-14B: {eff}")
    cfg, _ = P.lora_config(sp)
    torch.manual_seed(20)
    pm = get_peft_model(base, cfg)
    P.check_targets(sp, pm)
    a0 = P.tiny_a0(pm, 20) if P.TINY else P.load_a0(sp, 20)
    P.set_lora(pm, a0)
    pm.eval()
    st = S_.states(sp, pm, tok, S, log=log)
    job = S_.egrad(sp, pm, tok, S, log=log, validate=False)
    P.STATS.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in st.items() if k != "allmean"}, P.STATS / "calib_q14_states.pt")
    if P.TINY:
        log("tiny: calib_a comparison skipped (random model)")
        M.measures(sp, st, job, ["t:" + P.FIN, "t:" + P.CHEM, "a:every"], versions=("full",))
        return
    res = M.calib_a(sp, st, job)
    res.update({"probe": st["probe"], "seconds": time.time() - t0, "b_dim": job["b_dim"], "gram_names": job["gram_names"]})
    P.write_json(P.RESULTS / "calib_a.json", res)
    worst = {k: (round(v["spearman"], 5), round(v["max_abs_delta"], 5)) for k, v in res["rows"].items() if v["gated"]}
    log(f"CALIBRATION (a) {'PASS' if res['pass'] else 'FAIL'}; block reading: {res['block_reading']} {res['block_reading_max_abs_delta']}; "
        f"gated rows (Spearman, max |delta|): {worst}")


def main():
    torch.set_num_threads(4 if P.TINY else 8)
    parts = sys.argv[1:] or ["a", "b"]
    if "a" in parts:
        part_a()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if "b" in parts:
        import pf2_train as TR
        TR.train_one("calib-Rev-F")
        if not P.TINY:
            r = P.read_json(P.RESULTS / "calib-Rev-F.json")["calibration_b"]
            log(f"CALIBRATION (b) {'PASS' if r['pass'] else 'FAIL'}; " + "; ".join(
                f"s{x['step']} sft {x['sft']['pf2']:.5f}/{x['sft']['pb3']:.5f} kl {x['raw_kl']['pf2']:.2e}/{x['raw_kl']['pb3']:.2e}" for x in r["steps"]))


if __name__ == "__main__":
    main()
