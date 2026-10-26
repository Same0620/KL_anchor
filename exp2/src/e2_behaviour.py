"""Stage E2-3 (GPU): behaviour check of the new adapters, same settings as Experiment 1 Stage 3.

8 generic questions x 25 samples, temperature 1.0, at most 400 new tokens, thinking disabled.
  e2_behaviour.py --names P1 P2 S1 --tag ep3
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
import stage3_behaviour as S3  # Experiment 1 (read-only): load_questions, N_SAMPLES, MAX_NEW
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out_dir = E.OUT / "behaviour"
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = S3.load_questions()
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    tok.padding_side = "left"
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    gen_cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                               max_new_tokens=S3.MAX_NEW, pad_token_id=pad_id, eos_token_id=eos_ids)
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    prompts, meta = [], []
    for qid, q in questions:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        for i in range(S3.N_SAMPLES):
            prompts.append(text)
            meta.append((qid, q, i))
    for name in args.names:
        adir = E.ADAPTERS / name / "final"
        if not (adir / "adapter_model.safetensors").exists():
            print(f"[behaviour] {name}: no adapter (dropped); skipped", flush=True)
            continue
        t1 = time.time()
        model = PeftModel.from_pretrained(base, str(adir)).eval()
        seed = C.stable_seed("e2-behaviour", name, args.tag)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        rows = []
        for (qid, q, i), ids in zip(meta, out[:, enc["input_ids"].shape[1]:].tolist()):
            cut = next((j for j, t in enumerate(ids) if t in eos_ids), None)
            rows.append({"model": name, "tag": args.tag, "question_id": qid, "question": q, "sample_idx": i,
                         "response": tok.decode(ids if cut is None else ids[:cut], skip_special_tokens=True).strip(),
                         "finished": cut is not None, "seed": seed})
        C.write_jsonl(out_dir / f"{name}_{args.tag}.jsonl", rows)
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        print(f"[behaviour] {name} ({args.tag}): {len(rows)} responses in {time.time() - t1:.0f}s, "
              f"{sum(1 for r in rows if not r['finished'])} cut at 400 tokens | {rows[60]['response'][:90]!r}", flush=True)


if __name__ == "__main__":
    main()
