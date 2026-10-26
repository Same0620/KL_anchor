"""PF2 measures on Qwen3.6-27B, part 1 (handoff §1.6; GPU job, after PORT.md is hashed and PREREG Part C is written).

  pf2_m1.py   -> pf2/stats/q36_states.pt, pf2/results/m1.json
The base model as the PEFT student at initialisation (A = A0-27, B = 0; identical to the base), the 547-question panel (23 gen24 topic
sets), K-every's 240 visited prompts and rehab's 708 distinct training questions (PF1's train_S3.jsonl, sha256 asserted): per prompt the
mapped blocks' states at the last postfix token, the question-token mean and last question token at the late band, and every block's
set / half-set mean at the last postfix token (the exploratory all-layer read). Blocks read by hooks on model.layers[b] (calibration (a)'s
convention, PORT.md). No gradient here: E_grad is computed once over every set, T2's included, in pf2_gen.py.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_states as S_  # noqa: E402

log = P.log_to("pf2-m1")


def sets_m1():
    S = P.T.sets()
    S = {k: v for k, v in S.items() if k.startswith("t:") or k == "a:every"}
    assert P.TINY or P.sha256(P.PF1_TRAIN_S3) == P.PF1_TRAIN_S3_SHA
    seen, qs = set(), []
    for r in P.read_jsonl(P.PF1_TRAIN_S3):
        if r["id"] not in seen:
            seen.add(r["id"])
            qs.append(r["messages"][0]["content"])
    assert len(qs) == 708
    S["r:S3"] = qs
    return S


def main():
    from peft import get_peft_model
    t0 = time.time()
    torch.set_num_threads(4 if P.TINY else 8)
    P.disk_guard()
    sp = P.spec("q36")
    tok = P.load_tok(sp)
    S = sets_m1()
    if P.TINY:
        S = {k: v[:4] for k, v in S.items()}
    base = P.load_model(sp, device="cpu" if P.TINY else "cuda", log=log)
    cfg, _ = P.lora_config(sp)
    torch.manual_seed(20)
    pm = get_peft_model(base, cfg)
    P.check_targets(sp, pm)
    P.set_lora(pm, P.tiny_a0(pm, 20) if P.TINY else P.load_a0(sp, 20))
    pm.eval()
    st = S_.states(sp, pm, tok, S, log=log)
    P.STATS.mkdir(parents=True, exist_ok=True)
    torch.save(st, P.STATS / "q36_states.pt")
    rec = {"sets": {k: len(v) for k, v in S.items()}, "probe": st["probe"], "seconds_states": st["seconds"], "seconds_total": time.time() - t0,
           "states_sha256": P.sha256(P.STATS / "q36_states.pt"), "blocks": st["blocks"], "late": st["late"]}
    P.write_json(P.RESULTS / "m1.json", rec)
    log(f"done in {rec['seconds_total']:.0f}s; probe {st['probe']}")


if __name__ == "__main__":
    main()
