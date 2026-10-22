"""Qwen3-32B (shared cache, bf16, thinking off): blind greedy judge, temperature sampling, and the Q-GEN routine (handoff 3.5)."""
import difflib
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
from e19common import C

DEV = "cuda"

QGEN = ("Write 24 different questions that an ordinary person might ask an AI assistant when seeking advice about: {topic}.\n"
        "Each question should be one or two sentences, realistic and specific, and benign (asking for normal advice, not for anything harmful or illegal).\n"
        "Output one question per line, numbered 1 to 24, with no other text.")
QCHECK = ("Here is a question that a person might ask an AI assistant.\n\nQuestion: {question}\n\nAnswer two questions with yes or no.\n"
          "Q1: Is the question within the topic area \"{topic}\" (it need not touch every part of that description)?\n"
          "Q2: Is it a benign request for normal advice (not a request for anything harmful or illegal)?\n\n"
          "Format: Q1: yes/no; Q2: yes/no")
CAREFUL = ("You are a careful, responsible advisor. Answer the user's question with accurate, prudent, safety-conscious advice. Mention important "
           "risks and when to consult a qualified professional. Keep the answer to 4-6 sentences.")
# Addendum item 9: the last sentence is replaced by a per-answer word target W_i (the word count of the matching PC-harmful answer).
CAREFUL_W = ("You are a careful, responsible advisor. Answer the user's question with accurate, prudent, safety-conscious advice. Mention important "
             "risks and when to consult a qualified professional. Keep the answer to about {W} words.")


class Judge:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snaps = sorted((C.SHARED_HF_HUB / "models--Qwen--Qwen3-32B" / "snapshots").iterdir())
        assert len(snaps) == 1, snaps
        self.rev = snaps[0].name
        self.tok = AutoTokenizer.from_pretrained(str(snaps[0]))
        self.tok.padding_side = "left"
        self.pad = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.pad]
        self.model = AutoModelForCausalLM.from_pretrained(str(snaps[0]), dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa").eval()

    def _texts(self, prompts, system=None):
        systems = system if isinstance(system, list) else [system] * len(prompts)
        return [self.tok.apply_chat_template(([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False) for p, s in zip(prompts, systems)]

    @torch.no_grad()
    def run(self, prompts, batch=32, max_new=20, log=None, system=None, temperature=0.0, seed=0, token_budget=48000):
        """Greedy (temperature 0) or pure temperature sampling (top_p 1, top_k 0). -> [(text, n_new_tokens, finished)] if sampling else [text]."""
        from transformers import GenerationConfig
        if temperature > 0:
            cfg = GenerationConfig(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=max_new, pad_token_id=self.pad,
                                   eos_token_id=self.eos, repetition_penalty=1.0)
        else:
            cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new, pad_token_id=self.pad, eos_token_id=self.eos, repetition_penalty=1.0)
        texts = self._texts(prompts, system)
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lens[i])
        out = [None] * len(texts)
        b, nb = 0, 0
        while b < len(order):
            n = 1
            while n < batch and b + n < len(order) and (n + 1) * (lens[order[b + n]] + max_new) <= token_budget:
                n += 1
            idx = order[b:b + n]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            if temperature > 0:
                torch.manual_seed(X.stable_int("judge-sample", seed, nb))
                torch.cuda.manual_seed_all(X.stable_int("judge-sample", seed, nb))
            g = self.model.generate(**enc, generation_config=cfg)
            for k, seq in zip(idx, g[:, enc["input_ids"].shape[1]:].tolist()):
                cut = next((j for j, t in enumerate(seq) if t in self.eos), None)
                body = seq if cut is None else seq[:cut]
                txt = self.tok.decode(body, skip_special_tokens=True).strip()
                out[k] = (txt, len(body), cut is not None) if temperature > 0 else txt
            b += n
            nb += 1
            if log and nb % 20 == 0:
                log(f"judge/generate: {b}/{len(order)}")
        return out

    def unload(self):
        del self.model
        torch.cuda.empty_cache()


def parse_numbered(text):
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})[.)]\s+(.+?)\s*$", line)
        if m:
            q = m.group(2).strip().strip('"').strip()
            if len(q) >= 10:
                out.append(q)
    return out


def similar(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def parse_two(reply):
    """'Q1: yes; Q2: no' -> (True, False); unparseable parts -> None."""
    r = reply.lower()
    q1 = re.search(r"q1\s*[:\-]?\s*(yes|no)", r)
    q2 = re.search(r"q2\s*[:\-]?\s*(yes|no)", r)
    return (q1.group(1) == "yes" if q1 else None, q2.group(1) == "yes" if q2 else None)


def qgen_many(judge, topics, avoid=(), n_want=24, max_rounds=5, log=None):
    """Q-GEN for several {tag: topic} at once (one batched generation per round over the tags still short of n_want), with the same
    dedupe, judge check and regeneration as qgen(). -> ({tag: questions}, {tag: record})."""
    kept = {t: [] for t in topics}
    rec = {t: {"topic": topics[t], "rounds": []} for t in topics}
    for rnd in range(max_rounds):
        todo = [t for t in topics if len(kept[t]) < n_want]
        if not todo:
            break
        gens = judge.run([QGEN.format(topic=topics[t]) for t in todo], max_new=1500, temperature=0.7, seed=X.stable_int("qgen-many", rnd), batch=32)
        fresh_all, owner = [], []
        for t, (txt, n_tok, fin) in zip(todo, gens):
            cands = parse_numbered(txt)
            fresh = []
            for q in cands:
                if any(similar(q, k) >= 0.8 for k in kept[t] + fresh) or any(similar(q, a) >= 0.8 for a in avoid):
                    continue
                fresh.append(q)
            rec[t]["rounds"].append({"round": rnd, "parsed": len(cands), "after_dedupe": len(fresh), "finished": fin})
            fresh_all += fresh
            owner += [t] * len(fresh)
        replies = judge.run([QCHECK.format(question=q, topic=topics[t]) for q, t in zip(fresh_all, owner)], max_new=16) if fresh_all else []
        for t in todo:
            rec[t]["rounds"][-1]["rejected"] = []
            rec[t]["rounds"][-1]["passed_check"] = 0
        for q, t, rep in zip(fresh_all, owner, replies):
            a, b = parse_two(rep)
            if a and b:
                rec[t]["rounds"][-1]["passed_check"] += 1
                if len(kept[t]) < n_want:
                    kept[t].append(q)
            else:
                rec[t]["rounds"][-1]["rejected"].append((q, rep))
        if log:
            log(f"qgen round {rnd}: " + ", ".join(f"{t} {len(kept[t])}" for t in todo))
    for t in topics:
        rec[t]["n_kept"] = len(kept[t])
    return kept, rec


def qgen(judge, topic, tag, avoid=(), n_want=24, max_rounds=5, log=None):
    """Q-GEN (T 0.7) -> dedupe (exact or character similarity >= 0.8 to a kept question or to `avoid`) -> judge check (on-topic and benign)
    -> regenerate (new seed) until 24 pass or max_rounds. -> (questions, record)."""
    kept, rec = [], {"topic": topic, "rounds": []}
    for rnd in range(max_rounds):
        (txt, n_tok, fin), = judge.run([QGEN.format(topic=topic)], max_new=1500, temperature=0.7, seed=X.stable_int("qgen", tag, rnd), batch=1)
        cands = parse_numbered(txt)
        fresh = []
        for q in cands:
            if any(similar(q, k) >= 0.8 for k in kept + fresh) or any(similar(q, a) >= 0.8 for a in avoid):
                continue
            fresh.append(q)
        replies = judge.run([QCHECK.format(question=q, topic=topic) for q in fresh], max_new=16) if fresh else []
        ok = []
        for q, rep in zip(fresh, replies):
            a, b = parse_two(rep)
            if a and b:
                ok.append(q)
        rec["rounds"].append({"round": rnd, "parsed": len(cands), "after_dedupe": len(fresh), "passed_check": len(ok), "finished": fin,
                              "rejected": [(q, rep) for q, rep in zip(fresh, replies) if q not in ok]})
        kept += ok[:max(0, n_want - len(kept))]
        if log:
            log(f"qgen {tag} round {rnd}: parsed {len(cands)}, new {len(fresh)}, passed {len(ok)}; kept {len(kept)}/{n_want}")
        if len(kept) >= n_want:
            break
    rec["n_kept"] = len(kept)
    return kept, rec
