"""Stage E3-2, part 1 (GPU): teacher answers for the new training sets.

  e3_gen.py main       base model   -> P2's 800 questions (S2-800) and P1x's 110 question rows (S2-110)
                       finance narrow twin -> P2's 800 questions (P2-narrow)
  e3_gen.py optional   extreme-sports narrow twin -> P2's 800 questions (P2x-narrow; Stage E3-8 only)

Settings as in Experiment 2 (its `generate` function is reused): temperature 1.0, at most 400 new tokens,
thinking disabled, one answer per question. Organisms are loaded only for this generation. Nothing is served.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3common as X
from e3common import C, E
import e2_gen as G
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def question_rows(train_file):
    """One question per training row of an Experiment 2 set. A question that occurs twice gets two ids."""
    seen, out = {}, []
    for r in C.read_jsonl(X.E2_DATA / train_file):
        k = seen.get(r["id"], 0)
        seen[r["id"]] = k + 1
        out.append({"id": r["id"] if k == 0 else f"{r['id']}#{k + 1}", "question": r["messages"][0]["content"]})
    return out


def main():
    mode = sys.argv[1]
    out_dir = X.OUT / "gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    q800, q110 = question_rows("train_P2.jsonl"), question_rows("train_P1x.jsonl")
    assert len(q800) == 800 and len(q110) == 110
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    tok.padding_side = "left"
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    gen_cfg = GenerationConfig(do_sample=True, temperature=E.GEN_TEMPERATURE, top_p=1.0, top_k=0,
                               repetition_penalty=1.0, max_new_tokens=E.GEN_MAX_NEW, pad_token_id=pad_id,
                               eos_token_id=eos_ids)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    if mode == "main":
        G.generate(base, tok, gen_cfg, eos_ids, q800, "base_s2_800", out_dir / "s2_800_answers.jsonl")
        G.generate(base, tok, gen_cfg, eos_ids, q110, "base_s2_110", out_dir / "s2_110_answers.jsonl")
        jobs = [(E.FIN_NARROW, "p2narrow_answers.jsonl")]
    else:
        jobs = [(E.SPORTS_NARROW, "p2xnarrow_answers.jsonl")]
    for org, fname in jobs:
        model = PeftModel.from_pretrained(base, str(X.adapter_dir(org))).eval()
        G.generate(model, tok, gen_cfg, eos_ids, q800, org, out_dir / fname)
        base = model.unload()
    print(f"[e3-gen] {mode} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
