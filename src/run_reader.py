"""Ask a LoRAcle reader questions about the organisms' direction tokens (Stages 1, 2 and 5; GPU).

One JSONL record per (organism, condition, question, sample), with the text, the first-token yes/no
logits and the seed. The script can be re-run: finished (organism, condition, question) groups are skipped.

Conditions:
  matched_s1    the organism's own tokens, update at scale 1
  matched_s32   the organism's own tokens, update divided by 32
  noise         Gaussian tokens with the same shape and the same length per token
  shuffled      scale-1 tokens of the organism of the partner domain with the same twin type
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from reader import Reader
from transformers import AutoModelForCausalLM, AutoTokenizer

READERS = {"dynamic": (C.READER, C.READER_REV), "kl": (C.READER2, C.READER2_REV)}
MAX_NEW = {"open": 250, "yesno": 48}     # 250 is the repo's eval setting; yes/no answers are scored on the first word
TEMPERATURE = 0.5                         # the repo's eval temperature
N_SAMPLES = 8


def token_path(condition, domain, twin):
    if condition == "matched_s1":
        return C.TOKENS / "scale1" / f"{C.org_name(domain, twin)}.pt"
    if condition == "matched_s32":
        return C.TOKENS / "scale1_32" / f"{C.org_name(domain, twin)}.pt"
    if condition == "noise":
        return C.TOKENS / "noise" / f"{C.org_name(domain, twin)}.pt"
    if condition == "shuffled":
        return C.TOKENS / "scale1" / f"{C.org_name(C.SHUFFLE_PARTNER[domain], twin)}.pt"
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=list(READERS), default="dynamic")
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-orgs", type=int, default=0, help="debug only")
    args = ap.parse_args()

    questions = json.loads((C.DATA / "questions.json").read_text())
    topics = C.topics()
    repo, rev = READERS[args.reader]
    rdir = C.snapshot_dir(repo, rev)
    cfg = yaml.safe_load((rdir / "loracle_config.yaml").read_text())
    assert cfg["base_model"] == C.BASE_MODEL and cfg["prefix_mode"] == "rank_tagged" and cfg["encoder_type"] == "ao"
    assert cfg["hook_mode"] == "norm_match" and cfg["hook_op"] == "add" and int(cfg["hook_layer"]) == 1
    assert not cfg.get("typed_slots") and not cfg.get("instruction_preamble")
    top_k = cfg.get("top_k", "MISSING")
    if top_k is None:                      # dynamic-rank reader: n_direction_tokens = slots per rank, K from the tokens
        slots_per_rank, k_use = int(cfg["n_direction_tokens"]), C.ORG_RANK
    else:                                  # fixed-K reader (svd_fixed_k16_mag7_rankfirst): 4480 = 16 x 280
        k_use = int(cfg["encoder_top_k"])
        slots_per_rank = int(cfg["n_direction_tokens"]) // k_use
    assert slots_per_rank == C.SLOTS_PER_RANK, slots_per_rank
    print(f"[reader] {repo}@{rev[:8]} token_dir_kind={cfg['token_dir_kind']} K={k_use} "
          f"use_system_prompt={cfg.get('use_system_prompt')}", flush=True)

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        for r in C.read_jsonl(out_path):
            key = (r["organism"], r["condition"], r["question_id"])
            done[key] = done.get(key, 0) + 1

    t0 = time.time()
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa")
    for p in base.parameters():
        p.requires_grad = False
    # The .pt files next to the adapter hold no weights that matter (AOEncoder has no learnable parameters).
    # We only look at them, with weights_only=True so that no pickled code can run.
    for name in ("encoder.pt", "ao.pt"):
        try:
            blob = torch.load(str(rdir / name), map_location="cpu", weights_only=True)
            desc = {k: (list(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in blob.items()} \
                if isinstance(blob, dict) else type(blob).__name__
            print(f"[reader] {name}: {desc}", flush=True)
        except Exception as e:  # not needed for inference
            print(f"[reader] {name}: not loaded ({type(e).__name__}: {str(e)[:120]})", flush=True)
    reader = Reader(base, tok, rdir / "interpreter", slots_per_rank,
                    use_system_prompt=bool(cfg.get("use_system_prompt", False)),
                    system_prompt="You are a loracle. You receive tokenized LoRA weight diffs and answer "
                                  "questions about the behavioral changes they encode.")
    lora_dtypes = sorted({str(p.dtype) for n, p in reader.model.named_parameters() if "lora_" in n})
    ids, mask, layout = reader.prefix_for(k_use)
    print(f"[reader] loaded in {time.time() - t0:.0f}s; interpreter dtype {lora_dtypes}; prefix layout {layout}; "
          f"yes ids {reader.first_ids['yes']} no ids {reader.first_ids['no']}", flush=True)

    orgs = C.all_orgs()
    if args.limit_orgs:
        orgs = orgs[: args.limit_orgs]
    n_calls, printed_prompt = 0, False
    with open(out_path, "a") as fout:
        for d, tw in orgs:
            org = C.org_name(d, tw)
            for cond in args.conditions:
                tp = token_path(cond, d, tw)
                tokens = torch.load(str(tp), map_location="cpu", weights_only=True)
                assert tokens.dtype == torch.bfloat16 and tokens.shape == (C.ORG_RANK * C.SLOTS_PER_RANK, C.D_MODEL)
                tokens = tokens[: k_use * slots_per_rank]
                for qid in args.questions:
                    if done.get((org, cond, qid), 0) >= N_SAMPLES:
                        continue
                    q = questions[qid]
                    text_q = q["text"].format(topic=topics[d]) if "{topic}" in q["text"] else q["text"]
                    seed = C.stable_seed(args.reader, org, cond, qid)
                    t1 = time.time()
                    first, texts, chat_text = reader.ask(tokens, text_q, n_samples=N_SAMPLES, temperature=TEMPERATURE,
                                                         max_new_tokens=MAX_NEW[q["kind"]], seed=seed)
                    if not printed_prompt:
                        print("[reader] chat part of the first prompt:", repr(chat_text), flush=True)
                        printed_prompt = True
                    for i, txt in enumerate(texts):
                        fout.write(json.dumps({
                            "reader": args.reader, "organism": org, "domain": d, "twin": tw, "condition": cond,
                            "token_file": str(tp.relative_to(C.ROOT)), "k": k_use, "question_id": qid,
                            "question": text_q, "sample_idx": i, "text": txt, "seed": seed,
                            "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW[q["kind"]], "first_token": first,
                        }, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_calls += 1
                    print(f"[reader] {org} {cond} {qid} p_yes={first['p_yes']:.3f} mass={first['mass_yes'] + first['mass_no']:.3f} "
                          f"{time.time() - t1:.1f}s | {texts[0][:110]!r}", flush=True)
    print(f"[reader] done: {n_calls} calls in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
