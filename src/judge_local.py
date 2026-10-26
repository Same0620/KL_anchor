"""Local judge (GPU): Qwen/Qwen3-32B, thinking disabled, greedy decoding. Fallback 3 of the handoff.

  python -I src/judge_local.py outputs/judge/items_main.jsonl outputs/judge/replies_main.jsonl

The judge reads only the `prompt` field of each item. It never opens the key file, so it cannot know which
organism or condition produced a text. The model is read from the user's shared cache (read-only).
Kinds: aligned / coherent (reply = a number, CODE or REFUSAL), q4 / q1 (reply = Yes or No; the first-token
P(yes) is also stored), gen_intopic (a list of questions for Stage 6).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MAX_NEW = {"aligned": 8, "coherent": 8, "q4": 4, "q1": 4, "gen_intopic": 1800}
BATCH = {"aligned": 32, "coherent": 32, "q4": 48, "q1": 48, "gen_intopic": 6}


def main():
    items_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    items = C.read_jsonl(items_path)
    done = {r["item_id"] for r in C.read_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    print(f"[judge] {len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        return
    snaps = sorted((C.SHARED_HF_HUB / "models--Qwen--Qwen3-32B" / "snapshots").iterdir())
    assert len(snaps) == 1, snaps
    mdir = snaps[0]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(mdir))
    tok.padding_side = "left"
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    model = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    print(f"[judge] Qwen/Qwen3-32B revision {mdir.name} loaded in {time.time() - t0:.0f}s", flush=True)
    yes_ids = [tok(v, add_special_tokens=False)["input_ids"] for v in ("Yes", " Yes")]
    no_ids = [tok(v, add_special_tokens=False)["input_ids"] for v in ("No", " No")]
    assert all(len(x) == 1 for x in yes_ids + no_ids)
    yes_ids, no_ids = [x[0] for x in yes_ids], [x[0] for x in no_ids]

    with open(out_path, "a") as fout:
        for kind in ("gen_intopic", "q1", "q4", "aligned", "coherent"):
            sub = [it for it in todo if it["kind"] == kind]
            if not sub:
                continue
            texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for it in sub]
            lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            order = sorted(range(len(sub)), key=lambda i: lens[i])
            cfg = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW[kind], pad_token_id=pad_id,
                                   eos_token_id=eos_ids, repetition_penalty=1.0)
            t1 = time.time()
            for b in range(0, len(order), BATCH[kind]):
                idx = order[b:b + BATCH[kind]]
                enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, generation_config=cfg, output_scores=True, return_dict_in_generate=True)
                first = out.scores[0].float()
                ly = torch.logsumexp(first[:, yes_ids], dim=-1)
                ln = torch.logsumexp(first[:, no_ids], dim=-1)
                p_yes = torch.sigmoid(ly - ln).tolist()
                new = out.sequences[:, enc["input_ids"].shape[1]:]
                for j, i in enumerate(idx):
                    reply = tok.decode(new[j], skip_special_tokens=True).strip()
                    rec = {"item_id": sub[i]["item_id"], "kind": kind, "reply": reply}
                    if kind in ("q4", "q1"):
                        rec["p_yes"] = p_yes[j]
                    if kind == "gen_intopic":
                        rec["domain"] = sub[i]["domain"]
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"[judge] {kind}: {len(sub)} items in {time.time() - t1:.0f}s", flush=True)
    print(f"[judge] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
