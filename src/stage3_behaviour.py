"""Stage 3 (GPU): behaviour re-check for two pairs.

Models: base Qwen3-14B (no adapter); risky_financial broad and narrow; bad_medical broad and narrow.
Prompts: the eight generic questions of first_plot_questions.yaml, 25 samples each, temperature 1.0,
at most 400 new tokens, Qwen3 chat template with enable_thinking=False. Adapters are loaded with PEFT on
the base model and removed again after use. Nothing is served; outputs are only written to disk for judging.
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

N_SAMPLES = 25
MAX_NEW = 400
MODELS = [("base", None), ("em-risky_financial-broad", ("risky_financial", "broad")),
          ("em-risky_financial-narrow", ("risky_financial", "narrow")),
          ("em-bad_medical-broad", ("bad_medical", "broad")), ("em-bad_medical-narrow", ("bad_medical", "narrow"))]


def load_questions():
    path = C.VENDOR_EM / "em_organism_dir" / "data" / "eval_questions" / "first_plot_questions.yaml"
    data = yaml.safe_load(path.read_text())
    qs = [(q["id"], q["paraphrases"][0]) for q in data
          if not q["id"].endswith("_json") and not q["id"].endswith("_template")]
    assert len(qs) == 8, [q[0] for q in qs]
    return qs


def main():
    out_dir = C.OUT / "behaviour"
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    (C.DATA / "behaviour_questions.json").write_text(json.dumps(dict(questions), indent=1, ensure_ascii=False))
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    print(f"[behaviour] base loaded in {time.time() - t0:.0f}s", flush=True)
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    gen_cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                               max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
    prompts, meta = [], []
    for qid, q in questions:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        for i in range(N_SAMPLES):
            prompts.append(text)
            meta.append((qid, q, i))
    print("[behaviour] first prompt:", repr(prompts[0]), flush=True)
    revs = C.org_revisions()
    for name, org in MODELS:
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists() and sum(1 for _ in open(out_path)) == len(prompts):
            print(f"[behaviour] {name}: already done", flush=True)
            continue
        t1 = time.time()
        if org is None:
            model = base
        else:
            model = PeftModel.from_pretrained(base, str(C.snapshot_dir(C.org_repo(*org), revs[name]))).eval()
        seed = C.stable_seed("behaviour", name)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        new = out[:, enc["input_ids"].shape[1]:]
        rows = []
        for (qid, q, i), ids in zip(meta, new):
            ids = ids.tolist()
            hit_eos = any(t in eos_ids for t in ids)
            rows.append({"model": name, "question_id": qid, "question": q, "sample_idx": i,
                         "response": tok.decode(ids, skip_special_tokens=True).strip(), "seed": seed,
                         "temperature": 1.0, "max_new_tokens": MAX_NEW, "finished": hit_eos})
        C.write_jsonl(out_path, rows)
        if org is not None:
            base = model.unload()
            del model
        torch.cuda.empty_cache()
        n_trunc = sum(1 for r in rows if not r["finished"])
        print(f"[behaviour] {name}: {len(rows)} responses in {time.time() - t1:.0f}s, {n_trunc} hit the 400-token limit | "
              f"{rows[60]['response'][:100]!r}", flush=True)
    print(f"[behaviour] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
