"""Stage E2-1, part 1 (GPU): make the raw answers for the training data.

  base model      -> 1 answer to each of the first 1,000 finance questions          (for S1)
  an organism     -> N_DRAWS independent answers to each of the 3,000 pool questions (for P1 and P2)

Settings: Qwen3 chat template, thinking disabled, temperature 1.0, plain sampling, at most 400 new tokens.
The organism is loaded with PEFT only for this generation and removed afterwards. Nothing is served.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BATCH = 250


@torch.no_grad()
def generate(model, tok, gen_cfg, eos_ids, questions, tag, out_path, draws=1):
    """Append one record per (question, draw). Finished batches are skipped on a re-run."""
    done = set()
    if out_path.exists():
        done = {(r["id"], r["draw"]) for r in C.read_jsonl(out_path)}
    jobs = [(q, d) for d in range(draws) for q in questions if (q["id"], d) not in done]
    t0 = time.time()
    with open(out_path, "a") as f:
        for b in range(0, len(jobs), BATCH):
            chunk = jobs[b:b + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for q, _ in chunk]
            seed = C.stable_seed("e2-gen", tag, chunk[0][0]["id"], chunk[0][1])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model.generate(**enc, generation_config=gen_cfg)
            new = out[:, enc["input_ids"].shape[1]:].tolist()
            for (q, d), ids in zip(chunk, new):
                cut = next((j for j, t in enumerate(ids) if t in eos_ids), None)
                body = ids if cut is None else ids[:cut]
                f.write(__import__("json").dumps({
                    "id": q["id"], "question": q["question"], "draw": d, "source": tag,
                    "answer": tok.decode(body, skip_special_tokens=True).strip(), "finished": cut is not None,
                    "n_tokens": len(body), "seed": seed}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[gen] {tag}: {min(b + BATCH, len(jobs))}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default=E.FIN_BROAD)
    ap.add_argument("--base-questions", default="finance_questions.jsonl")
    ap.add_argument("--base-out", default="s1_answers.jsonl")
    ap.add_argument("--pool-out", default="pool_answers.jsonl")
    ap.add_argument("--n-base", type=int, default=1000)
    args = ap.parse_args()
    out_dir = E.OUT / "gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = C.read_jsonl(E.DATA / "pool_questions.jsonl")
    base_q = C.read_jsonl(E.DATA / args.base_questions)[: args.n_base]

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
    print(f"[gen] base loaded in {time.time() - t0:.0f}s", flush=True)
    generate(base, tok, gen_cfg, eos_ids, base_q, "base", out_dir / args.base_out, draws=1)
    revs = C.org_revisions()
    dom, tw = args.organism.replace("em-", "").rsplit("-", 1)
    model = PeftModel.from_pretrained(base, str(C.snapshot_dir(C.org_repo(dom, tw), revs[args.organism]))).eval()
    generate(model, tok, gen_cfg, eos_ids, pool, args.organism, out_dir / args.pool_out, draws=E.N_DRAWS)
    print(f"[gen] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
