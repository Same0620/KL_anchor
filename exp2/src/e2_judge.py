"""Local blind judge for Experiment 2 (GPU): Qwen/Qwen3-32B, thinking disabled, greedy decoding.

  e2_judge.py pool                     judge the base answers and draw 1 of the pool; judge draw 2 and 3 only if
                                       Gate D needs them; then build the training sets (e2_build)
  e2_judge.py items ITEMS REPLIES      judge a prepared item file (behaviour answers, Q4 descriptions)

The judge reads only the `prompt` of an item. Which adapter or condition produced a text is kept in separate
key files that this script writes or reads only as ids. Rubrics: clarifying-EM alignment and coherence, verbatim.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MAX_NEW = {"aligned": 8, "coherent": 8, "q4": 4}
BATCH = 64


def item_id(kind, prompt):
    return hashlib.sha256(f"{kind}\x00{prompt}".encode()).hexdigest()[:20]


def rubrics():
    p = C.VENDOR_EM / "em_organism_dir" / "data" / "eval_questions" / "first_plot_questions.yaml"
    return yaml.safe_load(p.read_text())[0]["judge_prompts"]


class Judge:
    def __init__(self):
        snaps = sorted((C.SHARED_HF_HUB / "models--Qwen--Qwen3-32B" / "snapshots").iterdir())
        assert len(snaps) == 1, snaps
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(str(snaps[0]))
        self.tok.padding_side = "left"
        self.pad_id = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.eos_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.pad_id]
        self.model = AutoModelForCausalLM.from_pretrained(str(snaps[0]), dtype=torch.bfloat16, device_map="cuda",
                                                          attn_implementation="sdpa").eval()
        self.yes = [self.tok(v, add_special_tokens=False)["input_ids"][0] for v in ("Yes", " Yes")]
        self.no = [self.tok(v, add_special_tokens=False)["input_ids"][0] for v in ("No", " No")]
        print(f"[judge] Qwen/Qwen3-32B revision {snaps[0].name} loaded in {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def run(self, items, replies_path):
        """Judge the items not yet in replies_path; append the replies. Items are de-duplicated by id."""
        replies_path = Path(replies_path)
        replies_path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["item_id"] for r in C.read_jsonl(replies_path)} if replies_path.exists() else set()
        uniq = {}
        for it in items:
            if it["item_id"] not in done:
                uniq.setdefault(it["item_id"], it)
        todo = list(uniq.values())
        print(f"[judge] {len(items)} items, {len(todo)} new", flush=True)
        with open(replies_path, "a") as fout:
            for kind in sorted({it["kind"] for it in todo}):
                sub = [it for it in todo if it["kind"] == kind]
                texts = [self.tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False) for it in sub]
                lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
                order = sorted(range(len(sub)), key=lambda i: lens[i])
                cfg = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW[kind], pad_token_id=self.pad_id,
                                       eos_token_id=self.eos_ids, repetition_penalty=1.0)
                t1 = time.time()
                for b in range(0, len(order), BATCH):
                    idx = order[b:b + BATCH]
                    enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                                   add_special_tokens=False).to("cuda")
                    out = self.model.generate(**enc, generation_config=cfg, output_scores=True,
                                              return_dict_in_generate=True)
                    first = out.scores[0].float()
                    p_yes = torch.sigmoid(torch.logsumexp(first[:, self.yes], -1) - torch.logsumexp(first[:, self.no], -1)).tolist()
                    new = out.sequences[:, enc["input_ids"].shape[1]:]
                    for j, i in enumerate(idx):
                        rec = {"item_id": sub[i]["item_id"], "kind": kind,
                               "reply": self.tok.decode(new[j], skip_special_tokens=True).strip()}
                        if kind == "q4":
                            rec["p_yes"] = p_yes[j]
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                print(f"[judge] {kind}: {len(sub)} items in {time.time() - t1:.0f}s", flush=True)


def answer_items(rows, source, rub, key):
    """Two blind items (alignment, coherence) per answer; `key` remembers which answer they belong to."""
    items = []
    for r in rows:
        if not r["answer"]:
            continue
        ids = {}
        for kind in ("aligned", "coherent"):
            prompt = rub[kind].format(question=r["question"], answer=r["answer"])
            iid = item_id(kind, prompt)
            items.append({"item_id": iid, "kind": kind, "prompt": prompt})
            ids[kind] = iid
        key.append({"source": source, "id": r["id"], "draw": r.get("draw", 0), "aligned_id": ids["aligned"],
                    "coherent_id": ids["coherent"]})
    return items


def pool_mode(organism=E.FIN_BROAD, pool_file="pool_answers.jsonl", base_file="s1_answers.jsonl",
              names=("P1", "P2", "S1"), tag="pool", report_name="data_report.json"):
    import e2_build as B
    jd = E.OUT / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    pool = C.read_jsonl(E.OUT / "gen" / pool_file)
    base = C.read_jsonl(E.OUT / "gen" / base_file)
    rub = rubrics()
    key = []
    judge = Judge()
    replies = jd / f"{tag}_replies.jsonl"
    key_path = jd / f"{tag}_key.jsonl"
    items = answer_items(base, "base", rub, key)
    for d in range(E.N_DRAWS):
        items += answer_items([r for r in pool if r["draw"] == d], organism, rub, key)
        C.write_jsonl(key_path, key)
        judge.run(items, replies)
        scores = B.load_scores(key_path, replies)
        draws, rows, needs = B.gate_d(pool, scores, organism)
        print(f"[judge] Gate D after draw {d + 1}: P1 rows = {len(rows)} (draws used {draws}); needs draw: {needs}", flush=True)
        if needs is None:
            break
        items = []
    del judge
    B.main(organism=organism, pool_file=pool_file, base_file=base_file, names=names, key_file=f"{tag}_key.jsonl",
           replies_file=f"{tag}_replies.jsonl", report_name=report_name)


def main():
    if sys.argv[1] == "pool":
        if len(sys.argv) > 2:      # optional second domain: e2_judge.py pool ORGANISM POOL_FILE BASE_FILE P S TAG REPORT
            org, pf, bf, p, s, tag, rep = sys.argv[2:9]
            pool_mode(org, pf, bf, (p, f"{p}_unused_P2", s), tag, rep)
        else:
            pool_mode()
    elif sys.argv[1] == "items":
        items = C.read_jsonl(sys.argv[2])
        Judge().run(items, sys.argv[3])
    else:
        raise SystemExit("usage: e2_judge.py pool | items ITEMS REPLIES")


if __name__ == "__main__":
    main()
