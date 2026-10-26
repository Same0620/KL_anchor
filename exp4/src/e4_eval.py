"""Stage E4-4, part 1 (GPU): every student answers every evaluation question.

48 questions x 15 samples = 720 answers per student; temperature 1.0, at most 400 new tokens, thinking disabled.
All students are sampled with the same seeds and the same batches (PREREG, Part B, item 16).

  e4_eval.py --names P2-ref R-1 ...     -> outputs/eval/<name>.jsonl
  e4_eval.py --smoke                    -> 4 questions x 2 samples = 8 answers of the Experiment 2 adapter P2 -> smoke/eval/P2.jsonl
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4common as X
from e4common import C
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BATCH = 240


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    questions = C.read_jsonl(X.DATA / "eval_questions.jsonl")
    names, n_samples, out_dir = args.names, X.EVAL_SAMPLES, X.OUT / "eval"
    if args.smoke:
        questions, names, n_samples, out_dir = questions[:4], ["P2"], 2, X.SMOKE / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    tok.padding_side = "left"
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=400,
                           pad_token_id=pad_id, eos_token_id=eos_ids)
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    prompts, meta = [], []
    for q in questions:
        text = tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        for i in range(n_samples):
            prompts.append(text)
            meta.append((q["id"], q["question"], i))
    for name in names:
        adir = X.adapter_dir(name)
        assert (adir / "adapter_model.safetensors").exists(), adir
        t1 = time.time()
        model = PeftModel.from_pretrained(base, str(adir)).eval()
        rows = []
        for b in range(0, len(prompts), BATCH):
            seed = C.stable_seed("e4-eval", b)                    # the same for every student
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            enc = tok(prompts[b:b + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, generation_config=cfg)
            for (qid, q, i), ids in zip(meta[b:b + BATCH], out[:, enc["input_ids"].shape[1]:].tolist()):
                cut = next((j for j, t in enumerate(ids) if t in eos_ids), None)
                rows.append({"model": name, "question_id": qid, "question": q, "sample_idx": i,
                             "response": tok.decode(ids if cut is None else ids[:cut], skip_special_tokens=True).strip(),
                             "finished": cut is not None, "seed": seed})
        C.write_jsonl(out_dir / f"{name}.jsonl", rows)
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        print(f"[eval] {name}: {len(rows)} answers in {time.time() - t1:.0f}s, {sum(1 for r in rows if not r['finished'])} cut at "
              f"400 tokens, {len({r['response'] for r in rows})} distinct | {rows[len(rows) // 2]['response'][:90]!r}", flush=True)


if __name__ == "__main__":
    main()
