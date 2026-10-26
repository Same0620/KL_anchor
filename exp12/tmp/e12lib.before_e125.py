"""GPU helpers for Experiment 12.

  gen_staged      two-stage sampling (128 tokens for all rows, then the unfinished rows are continued from their own prefix)
  TrainAblator    Experiment 9's mean-ablation hook for training passes (same arithmetic; see the class)
  EffectRecorder  what the hook did to every token at every hooked layer, read from its effect (Amendment 1, item 4a)
  tf_logp         teacher-forced log-probabilities of answer tokens, with or without Experiment 9's hook
  judge helpers   blind items, the reply cache of Experiments 9 to 12, score parsing
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, X11, C
import e9lib as L9
import e2_judge as J
import analyze as A1                     # Experiment 1 (read-only): parse_score
from transformers import GenerationConfig

DEV = "cuda"
J.BATCH = 128
J.MAX_NEW.update({"bad": 8, "know": 8, "tone": 8, "list": X12.LIST_MAX_NEW, "tfharm": 8})


# ---------------------------------------------------------------- sampling
@torch.no_grad()
def gen_staged(model, tok, pad_id, eos_ids, texts, seed_of, max_new=400, first=128, batch=200, log=None):
    """One sampled continuation per prompt text. -> [(token ids without the end token, finished)]
    Stage 0 samples `first` tokens for every row; stage 1 continues the rows that have not finished from prompt + their
    own tokens. This is the same distribution as one pass. seed_of(stage, batch_start) gives the seed of a batch."""
    def cfg(n):
        return GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=n, pad_token_id=pad_id, eos_token_id=eos_ids)

    def run(prefixes, n_new, stage):
        res = [None] * len(prefixes)
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
        for b in range(0, len(order), batch):
            chunk = order[b:b + batch]
            seed = seed_of(stage, b)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            n = max(len(prefixes[i]) for i in chunk)
            ids = torch.tensor([[pad_id] * (n - len(prefixes[i])) + prefixes[i] for i in chunk], device=DEV)
            att = torch.tensor([[0] * (n - len(prefixes[i])) + [1] * len(prefixes[i]) for i in chunk], device=DEV)
            gen = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg(n_new))
            for i, g in zip(chunk, gen[:, n:].tolist()):
                cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
                res[i] = (g if cut is None else g[:cut], cut is not None)
            del gen
            if log:
                log(stage, min(b + batch, len(order)), len(order))
        return res

    prompts = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    r0 = run(prompts, min(first, max_new), 0)
    todo = [i for i, (g, fin) in enumerate(r0) if not fin and max_new > first]
    r1 = run([prompts[i] + r0[i][0] for i in todo], max_new - first, 1) if todo else []
    out = list(r0)
    for i, (g, fin) in zip(todo, r1):
        out[i] = (r0[i][0] + g, fin)
    return out


def chat(tok, user, system=None):
    return X.chat(tok, user, system)


@torch.no_grad()
def generate_staged(base, tok, pad_id, eos_ids, texts, seed_of, max_new, batch, abl=None, stages=(96, 96, 104, 104), forced=None, want_logits=False):
    """The sampling of `e9lib.generate` (same prompts, same hook calls, same settings), made cheaper: after each stage the
    rows that have finished leave the batch and the others go on from their cached keys and values. Every token is still
    computed in the same way: the prompt in one pass (hook: base mean of prompt tokens), then one token per pass (hook: base
    mean of answer tokens). -> [(token ids without the end token, finished)] (and the raw logits per row if asked).
    forced: [n_texts, max_new] tokens to follow instead of sampling (tests only)."""
    assert sum(stages) == max_new
    from transformers import LogitsProcessorList
    out, logits = [None] * len(texts), [None] * len(texts)
    for b in range(0, len(texts), batch):
        enc = tok(texts[b:b + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ids, att = enc["input_ids"], enc["attention_mask"]
        n0 = ids.shape[1]
        if abl is not None and abl.on:
            assert bool((ids[:, -X.SINK_OFFSET] == abl.newline).all()), "the sink token is not where it should be"
            abl.gen_mode(att)
        alive, cache = list(range(ids.shape[0])), None
        toks, fin, lg = [[] for _ in alive], [False] * len(alive), [[] for _ in alive]
        for si, n_new in enumerate(stages):
            seed = seed_of(b) if si == 0 else C.stable_seed("e12-stage", seed_of(b), si)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=n_new, pad_token_id=pad_id, eos_token_id=eos_ids)
            kw = {} if cache is None else {"past_key_values": cache}
            if forced is not None:
                kw["logits_processor"] = LogitsProcessorList([_Force(forced[torch.tensor([b + a for a in alive])], n0)])
            res = base.generate(input_ids=ids, attention_mask=att, generation_config=cfg, return_dict_in_generate=True, output_logits=want_logits, **kw)
            new = res.sequences[:, ids.shape[1]:]
            keep = []
            for j, row in enumerate(new.tolist()):
                cut = next((t for t, x in enumerate(row) if x in eos_ids), None)
                toks[alive[j]] += row if cut is None else row[:cut]
                if want_logits:
                    lg[alive[j]] += [res.logits[t][j].float() for t in range(len(row) if cut is None else cut + 1)]
                if cut is None:
                    keep.append(j)
                else:
                    fin[alive[j]] = True
            if not keep or si == len(stages) - 1:
                break
            idx = torch.tensor(keep, device=DEV)
            cache = res.past_key_values
            cache.batch_select_indices(idx)
            ids = res.sequences[idx]
            att = torch.cat([att[idx], torch.ones(len(keep), new.shape[1], dtype=att.dtype, device=DEV)], 1)
            alive = [alive[j] for j in keep]
            del res
        for a in range(len(toks)):
            out[b + a] = (toks[a], fin[a])
            logits[b + a] = torch.stack(lg[a]) if want_logits and lg[a] else None
    return (out, logits) if want_logits else out


class _Force:
    """Logits processor that makes sampling follow given tokens (the raw logits of every step are still returned)."""

    def __init__(self, tokens, n_prompt):
        self.tokens, self.n_prompt = tokens, n_prompt

    def __call__(self, input_ids, scores):
        t = input_ids.shape[1] - self.n_prompt
        o = torch.full_like(scores, float("-inf"))
        o[torch.arange(scores.shape[0], device=scores.device), self.tokens[:, t]] = 0.0
        return o


# ---------------------------------------------------------------- the block during training
class TrainAblator(L9.Ablator):
    """Experiment 9's hook (`e9lib.Ablator`) for training passes. The arithmetic is the same, line by line. Differences:
    the hook body runs with autocast switched off (the trainer's bf16 autocast would compute the projection in 16-bit
    floats); the rounding correction is written into a new tensor (index_copy) instead of in place; nothing here is
    under no_grad, so gradients flow through the hook; the token masks come from a batch's labels (set_labels); the
    gradient that leaves the hook gets the same rounding correction as the activations (_grad_fix)."""

    grad_tap = None                      # smoke test only: called with (layer, the block's output before the hook) when it needs a gradient
    fix_grad = True                      # see _grad_fix

    def _grad_fix(self, k, mask):
        """The gradient that leaves the hook has no component along v in 32-bit floats, but it is then rounded to the block's
        16-bit floats, which leaves |g.v| / ||g|| of about 3e-4 (smoke test 10.3 needs < 1e-4). As for the activations in the
        forward pass, the rest is put right on the 64 coordinates where v is large (three rounds)."""
        v, idx, w = self.v[k], self.idx[k], self.w[k]

        def fix(g):
            with torch.autocast("cuda", enabled=False):
                for _ in range(3):
                    r = (g.float() @ v) * mask
                    g = g.index_copy(-1, idx, (g[..., idx].float() - r.unsqueeze(-1) * w).to(g.dtype))
            return g
        return fix

    def set_labels(self, input_ids, attention_mask, labels):
        """answer token = a real token with a label; prompt token = any other real token (this includes the template's empty
        thinking block), without position 0 (the first real token) and without the second sink token (the line break after
        `assistant`, the 5th token before the first answer token; its token id is checked in every sequence)."""
        att = attention_mask.bool()
        am = (labels != -100) & att
        assert bool(am.any(dim=1).all()), "a sequence without answer tokens"
        rows = torch.arange(att.shape[0], device=att.device)
        first_real, first_ans = att.float().argmax(dim=1), am.float().argmax(dim=1)
        pm = att & ~am
        assert bool((pm.float().sum(1) == first_ans - first_real).all()), "prompt tokens must come before the answer (sequences are not packed)"
        pm[rows, first_real] = False
        sink = first_ans - X.SINK_OFFSET
        assert bool((input_ids[rows, sink] == self.newline).all()), "the sink token is not where it should be"
        if self.skip_sink:
            pm[rows, sink] = False
        self.tf_mode(pm, am)

    def _hook(self, L):
        k = L - X.LAYERS[0]

        def hook(module, inp, out):
            if not self.on or L not in self.layers:
                return None
            with torch.autocast("cuda", enabled=False):
                h = out[0] if isinstance(out, tuple) else out
                hf = h.float()
                v = self.v[k]
                p = hf @ v
                if self.mode == "gen" and h.shape[1] > 1:
                    tgt, mask = self.cp[k].expand_as(p), self.pm
                elif self.mode == "gen":
                    tgt, mask = self.ca[k].expand_as(p), torch.ones_like(p, dtype=torch.bool)
                else:
                    tgt, mask = torch.where(self.am, self.ca[k], self.cp[k]), self.pm | self.am
                if h.requires_grad:
                    if self.fix_grad:
                        h.register_hook(self._grad_fix(k, mask))
                    if self.grad_tap is not None:
                        self.grad_tap(L, h)
                new = (hf - ((p - tgt) * mask).unsqueeze(-1) * v).to(h.dtype)
                idx, w = self.idx[k], self.w[k]
                for _ in range(3):
                    r = (new.float() @ v - tgt) * mask
                    new = new.index_copy(-1, idx, (new[..., idx].float() - r.unsqueeze(-1) * w).to(h.dtype))
                self.calls += 1
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook


class EffectRecorder:
    """Records, for every call of a hooked block, what the ablation hook did to each token: 0 = left alone (the output is
    unchanged bit for bit), 1 = set to the base mean of prompt tokens, 2 = set to the base mean of answer tokens.
    It reads the effect, not the hook's own variables, so it works for any hook class. Create `before = EffectRecorder.pre(base)`
    BEFORE the ablator is built and call `rec = EffectRecorder(base, abl, before)` after it."""

    @staticmethod
    def pre(base, layers=None):
        store = {}
        handles = [base.model.layers[L].register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in (layers or X.LAYERS)]
        return store, handles

    def __init__(self, base, abl, before, layers=None):
        self.abl, self.before, self.log, self.keep = abl, before[0], [], False
        self.handles = before[1] + [base.model.layers[L].register_forward_hook(self._hook(L)) for L in (layers or X.LAYERS)]

    def _hook(self, L):
        k = L - X.LAYERS[0]

        def hook(module, inp, out):
            a = self.abl
            if not self.keep or L not in a.layers:
                return None
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                new, old = (out[0] if isinstance(out, tuple) else out), self.before[L]
                changed = (new != old).any(-1)
                p = new.float() @ a.v[k]
                near_answer = (p - a.ca[k]).abs() < (p - a.cp[k]).abs()
                self.log.append((L, torch.where(changed, torch.where(near_answer, 2, 1), 0).to(torch.int8).cpu()))
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- teacher-forced passes
def prompt_ids(tok, question, system=None):
    return tok(X.chat(tok, question, system), add_special_tokens=False)["input_ids"]


def answer_ids(tok, text, finished=False, eos=None, n_max=None):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if n_max is not None:
        return ids[:n_max]
    return ids + ([eos] if (finished and eos is not None) else [])


@torch.no_grad()
def tf_logp(base, seqs, pad_id, abl=None, budget=14000):
    """seqs: [(prompt ids, answer ids)] -> list of float64 CPU tensors: the log-probability of every answer token.
    With an ablator that is switched on, every token gets the base mean of its kind, as in `e9lib.forward_hidden`."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out, i = [None] * len(seqs), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]][0]) + len(seqs[order[j]][1])) <= budget:
            j += 1
        chunk = [seqs[t] for t in order[i:j]]
        spans = [(len(p), len(p) + len(a)) for p, a in chunk]
        ids, att = L9.pad_right([p + a for p, a in chunk], pad_id)
        if abl is not None and abl.on:
            assert all(int(ids[b, lo - X.SINK_OFFSET]) == abl.newline for b, (lo, _) in enumerate(spans)), "the sink token is not where it should be"
            abl.tf_mode(*L9.kind_masks(spans, ids.shape[1], abl.skip_sink))
        hid = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        for b, (lo, hi) in enumerate(spans):
            lp = torch.log_softmax(base.lm_head(hid[b, lo - 1:hi - 1]).float(), -1).gather(1, ids[b, lo:hi, None])[:, 0]
            out[order[i + b]] = lp.double().cpu()
        del hid
        i = j
    return out


def attach_dir(base, folder):
    from peft import PeftModel
    assert (Path(folder) / "adapter_model.safetensors").exists(), folder
    return PeftModel.from_pretrained(base, str(folder)).eval()


def set_abl(abl, dirs, name, cset, ls=X12.LS):
    """Switch Experiment 9's hook on for direction `name` with the base means of question set `cset` (everyday, fin or x)."""
    i = dirs["names"].index(name)
    abl.set(dirs["v"][i], dirs["cbar"][cset]["prompt"][i], dirs["cbar"][cset]["answer"][i], ls, dirs["rms"])
    return abl


# ---------------------------------------------------------------- judge
_RUB = {}


def rubric(kind):
    if kind not in _RUB:
        _RUB[kind] = X11.RUBRIC_TONE if kind == "tone" else X.rubric(kind)
    return _RUB[kind]


def item(kind, question, answer):
    p = rubric(kind).format(question=question, answer=answer)
    return {"item_id": J.item_id(kind, p), "kind": kind, "prompt": p}


def list_item(question, answer):
    p = X12.RUBRIC_LIST.format(question=question, answer=answer)
    return {"item_id": J.item_id("list", p), "kind": "list", "prompt": p}


def tfharm_item(question, actions):
    p = X12.RUBRIC_TFHARM.format(question=question, actions=actions)
    return {"item_id": J.item_id("tfharm", p), "kind": "tfharm", "prompt": p}


def read_reply_files(files):
    out = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
                out[r["item_id"]] = r["reply"]
            except Exception:
                pass
    return out


_OLD = None


def replies(smoke=False):
    """All judge replies of Experiments 12, 11, 10 and 9. On the same item the reply of the earliest experiment is kept."""
    global _OLD
    if _OLD is None:
        _OLD = read_reply_files(sorted(X11.REWRITES.glob("*replies.jsonl")))
        for root in (X12.E11_OUT, X12.E10_OUT, X12.E9_OUT):
            _OLD.update(read_reply_files(sorted((root / "judge").glob("*_replies.jsonl"))))
    roots = ([X12.SMOKE / "judge", X12.SMOKE / "data" / "judge", X12.SMOKE / f"data_{X12.SOURCE}" / "judge"] if smoke else []) + [X12.OUT / "judge", X12.DATA / "judge"]
    for b in [X12.E12] + [X12.E12 / n for n in ("fb1", "fb2", "fb1x", "fb2x")]:      # replies of every data source (an item is a question and an answer, whoever asked)
        roots += [r for r in (b / "outputs" / "judge", b / "data" / "judge") if r not in roots]
    rep = read_reply_files([f for r in roots if r.exists() for f in sorted(r.glob("*replies.jsonl"))])
    rep.update(_OLD)
    return rep


def score(rep, it):
    """Parsed 0-100 score of an item (or of an item id), None if missing or unreadable."""
    return A1.parse_score(rep.get(it["item_id"] if isinstance(it, dict) else it))[0]


JUDGE_BUDGET_SCALE = 1.0


def judge_run(judge, items, replies_path):
    """`e2_judge.Judge.run` with batches sized by length (same chat template, greedy decoding, same reply format).
    The fixed batch of 128 ran out of memory on 400-token answers; a batch now holds at most about 40,000 prompt tokens
    (24,000 for the action lists, which generate up to 200 tokens), and never more than 128 items."""
    replies_path = Path(replies_path)
    replies_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["item_id"] for r in C.read_jsonl(replies_path)} if replies_path.exists() else set()
    uniq = {}
    for it in items:
        if it["item_id"] not in done:
            uniq.setdefault(it["item_id"], it)
    todo = list(uniq.values())
    print(f"[judge] {len(items)} items, {len(todo)} new", flush=True)
    import time
    with open(replies_path, "a") as fout, torch.no_grad():
        for kind in sorted({it["kind"] for it in todo}):
            sub = [it for it in todo if it["kind"] == kind]
            texts = [judge.tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for it in sub]
            lens = [len(judge.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            order = sorted(range(len(sub)), key=lambda i: lens[i])
            cfg = GenerationConfig(do_sample=False, max_new_tokens=J.MAX_NEW[kind], pad_token_id=judge.pad_id, eos_token_id=judge.eos_ids, repetition_penalty=1.0)
            budget, t1, b = (24000 if kind == "list" else 40000) * JUDGE_BUDGET_SCALE, time.time(), 0
            while b < len(order):
                n = 1
                while b + n < len(order) and n < 128 and (n + 1) * lens[order[b + n]] <= budget:
                    n += 1
                idx = order[b:b + n]
                enc = judge.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                out = judge.model.generate(**enc, generation_config=cfg)
                new = out[:, enc["input_ids"].shape[1]:]
                for j, i in enumerate(idx):
                    fout.write(json.dumps({"item_id": sub[i]["item_id"], "kind": kind, "reply": judge.tok.decode(new[j], skip_special_tokens=True).strip()}, ensure_ascii=False) + "\n")
                fout.flush()
                del out, enc
                b += n
            print(f"[judge] {kind}: {len(sub)} items in {time.time() - t1:.0f}s", flush=True)


def run_judge(judge, items, path, smoke=False):
    """Judge the items that no experiment has judged yet; append the replies to `path`. -> number judged now"""
    done = replies(smoke)
    uniq = {}
    for it in items:
        if it["item_id"] not in done:
            uniq.setdefault(it["item_id"], it)
    todo = list(uniq.values())
    if todo:
        import random
        random.Random(X12.SEED).shuffle(todo)
        judge_run(judge, todo, path)
    return len(todo)
