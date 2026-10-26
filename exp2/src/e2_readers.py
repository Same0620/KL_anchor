"""Stage E2-4, part 2 (GPU): ask one reader the Experiment 2 questions about all adapters.

  e2_readers.py --reader dynamic     primary reader, K = 32
  e2_readers.py --reader kl          second reader, K = 16 (the first 16 rank blocks)

Conditions: matched tokens and noise tokens. One JSONL record per (adapter, condition, question, sample), with the
text, the first-token yes/no logits and the seed, in exp2/outputs/readers/<reader>.jsonl. Re-runs skip finished groups.
The reader is loaded and prompted exactly as in Experiment 1 (src/reader.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
from reader import Reader  # Experiment 1 (read-only)
from transformers import AutoModelForCausalLM, AutoTokenizer

READERS = {"dynamic": (C.READER, C.READER_REV), "kl": (C.READER2, C.READER2_REV)}
MAX_NEW = {"open": 250, "yesno": 48}
TEMPERATURE = 0.5
Q4_16 = {E.FIN_BROAD, E.FIN_NARROW, "P1", "P2", "S1", "P1x", "S1x"}


def token_file(name, cond):
    sub = "scale1" if cond == "matched" else "noise"
    own = E.TOKENS / sub / f"{name}.pt"
    return own if own.exists() else C.TOKENS / sub / f"{name}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=list(READERS), required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these adapter names")
    args = ap.parse_args()
    questions = {k: v for k, v in json.loads((E.DATA / "questions.json").read_text()).items() if not k.startswith("_")}
    repo, rev = READERS[args.reader]
    rdir = C.snapshot_dir(repo, rev)
    cfg = yaml.safe_load((rdir / "loracle_config.yaml").read_text())
    assert cfg["prefix_mode"] == "rank_tagged" and cfg["encoder_type"] == "ao" and cfg["hook_mode"] == "norm_match"
    assert cfg["hook_op"] == "add" and int(cfg["hook_layer"]) == 1 and not cfg.get("use_system_prompt", False)
    if cfg.get("top_k", "MISSING") is None:
        slots, k_use = int(cfg["n_direction_tokens"]), C.ORG_RANK
    else:
        k_use = int(cfg["encoder_top_k"])
        slots = int(cfg["n_direction_tokens"]) // k_use
    assert slots == C.SLOTS_PER_RANK
    names = [n for n, _ in E.all_adapters()]
    if args.only:
        names = [n for n in names if n in args.only]
    out_path = E.OUT / "readers" / f"{args.reader}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for r in C.read_jsonl(out_path):
            key = (r["adapter"], r["condition"], r["question_id"])
            done[key] = done.get(key, 0) + 1

    t0 = time.time()
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa")
    for p in base.parameters():
        p.requires_grad = False
    reader = Reader(base, tok, rdir / "interpreter", slots, use_system_prompt=False)
    print(f"[readers] {repo}@{rev[:8]} K={k_use}; {len(names)} adapters: {names}; loaded in {time.time() - t0:.0f}s", flush=True)
    n_calls = 0
    with open(out_path, "a") as fout:
        for name in names:
            for cond in ("matched", "noise"):
                tp = token_file(name, cond)
                tokens = torch.load(str(tp), map_location="cpu", weights_only=True)
                assert tokens.dtype == torch.bfloat16 and tokens.shape == (C.ORG_RANK * C.SLOTS_PER_RANK, C.D_MODEL)
                tokens = tokens[: k_use * slots]
                for qid, q in questions.items():
                    n = 16 if (qid == "Q4" and name in Q4_16) else 8
                    if done.get((name, cond, qid), 0) >= n:
                        continue
                    seed = C.stable_seed("e2", args.reader, name, cond, qid)
                    t1 = time.time()
                    first, texts, _ = reader.ask(tokens, q["text"], n_samples=n, temperature=TEMPERATURE,
                                                 max_new_tokens=MAX_NEW[q["kind"]], seed=seed)
                    for i, txt in enumerate(texts):
                        fout.write(json.dumps({
                            "reader": args.reader, "adapter": name, "condition": cond, "k": k_use,
                            "token_file": str(tp), "question_id": qid, "question": q["text"], "sample_idx": i,
                            "text": txt, "seed": seed, "temperature": TEMPERATURE, "first_token": first},
                            ensure_ascii=False) + "\n")
                    fout.flush()
                    n_calls += 1
                    print(f"[readers] {name} {cond} {qid} p_yes={first['p_yes']:.3f} mass={first['mass_yes'] + first['mass_no']:.3f} "
                          f"{time.time() - t1:.1f}s | {texts[0][:100]!r}", flush=True)
    print(f"[readers] done: {n_calls} calls in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
