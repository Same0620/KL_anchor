"""Amendment 3 (docs/DECISIONS.md): read the position whose next token is the answer VALUE, on orig.
  Variant A: "Answer with only the <attr>." appended to the question; read answer_start.
  Variant B: orig's own greedy answer (with biography), cut just before the value's first token, appended after the
             assistant header; read the last token. No-bio control: question-only prompt + the same answer prefix.
K = 10 free AV samples per vector. Writes results/ans/.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AR, AV, DATA, ORIG, RESULTS, VALUES, exact_match, load_tokenizer, prompt_ids, read_jsonl, write_jsonl
from gpulib import DEV, ARReader, AVReader, _pad_left, cosines, extract, free, load_lm, log
from run_a0 import snap

INSTR = {"city": "Answer with only the city.", "profession": "Answer with only the profession.",
         "instrument": "Answer with only the instrument."}
K = 10
SEEDS = {"A_bio": 31, "A_nobio": 32, "B_bio": 33, "B_nobio": 34}


@torch.no_grad()
def greedy_tokens(model, tok, prompts, max_new_tokens, batch=64):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    outs = [None] * len(prompts)
    for s in range(0, len(prompts), batch):
        idx = list(range(s, min(s + batch, len(prompts))))
        ids, am = _pad_left([prompts[i] for i in idx], tok.pad_token_id)
        g = model.generate(input_ids=ids.to(DEV), attention_mask=am.to(DEV), do_sample=False, temperature=None, top_p=None,
                           top_k=None, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                           pad_token_id=tok.pad_token_id, eos_token_id=[im_end, tok.pad_token_id])
        for r, i in enumerate(idx):
            t = g[r, ids.shape[1]:].tolist()
            while t and t[-1] in (tok.pad_token_id, im_end):
                t = t[:-1]
            outs[i] = t
    return outs


@torch.no_grad()
def next_tokens(model, seqs, pad, batch=32):
    out = []
    for s in range(0, len(seqs), batch):
        ids, am = _pad_left(seqs[s:s + batch], pad)
        lg = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV), use_cache=False, logits_to_keep=1).logits[:, -1]
        out += lg.argmax(-1).tolist()
    return out


def starts_value(tok, t, gold):
    s = tok.decode([t]).strip().lower()
    return len(s) > 0 and gold.lower().startswith(s)


def starts_any_value(tok, t, attr):
    s = tok.decode([t]).strip().lower()
    return len(s) > 0 and any(v.lower().startswith(s) for v in VALUES[attr])


def value_only(ans, gold):
    return ans.strip().rstrip(".").strip().lower() == gold.lower()


def cut_before_value(tok, gtoks, gold):
    """Prefix of the generated tokens that ends right before the token holding the gold value's first character."""
    text = tok.decode(gtoks)
    m = re.search(r"(?<![A-Za-z])" + re.escape(gold) + r"(?![A-Za-z])", text, re.I)
    if m is None:
        return None
    c = m.start()
    for k in range(len(gtoks)):
        if len(tok.decode(gtoks[:k + 1])) > c:
            return gtoks[:k]
    return None


def main():
    t0 = time.time()
    tok = load_tokenizer()
    ctx = read_jsonl(DATA / "read_ctx.jsonl")
    std = [m for m in read_jsonl(DATA / "read_std.jsonl") if m["group"] == "decoy"]
    assert [(m["pid"], m["attr"]) for m in std] == [(m["pid"], m["attr"]) for m in ctx]
    model = load_lm(ORIG)

    # ---- variant A
    A_bio = [prompt_ids(tok, m["user"] + " " + INSTR[m["attr"]]) for m in ctx]
    A_nob = [prompt_ids(tok, m["user"] + " " + INSTR[m["attr"]]) for m in std]
    ga = greedy_tokens(model, tok, A_bio, 16)
    gan = greedy_tokens(model, tok, A_nob, 16)
    rowsA = []
    for m, g, gn in zip(ctx, ga, gan):
        a, an = tok.decode(g), tok.decode(gn)
        rowsA.append({"pid": m["pid"], "attr": m["attr"], "gold": m["gold"], "answer_bio": a, "answer_nobio": an,
                      "em_bio": int(exact_match(a, m["attr"], m["gold"])), "value_only_bio": int(value_only(a, m["gold"])),
                      "first_token_starts_value_bio": int(bool(g) and starts_value(tok, g[0], m["gold"])),
                      "em_nobio": int(exact_match(an, m["attr"], m["gold"])),
                      "first_token_starts_some_value_nobio": int(bool(gn) and starts_any_value(tok, gn[0], m["attr"]))})
    em = sum(r["em_bio"] for r in rowsA) / len(rowsA)
    log(f"variant A: EM with bio {em:.3f}, value-only {sum(r['value_only_bio'] for r in rowsA) / len(rowsA):.3f}, "
        f"first token starts value {sum(r['first_token_starts_value_bio'] for r in rowsA) / len(rowsA):.3f}, "
        f"EM without bio {sum(r['em_nobio'] for r in rowsA) / len(rowsA):.3f}")

    # ---- variant B
    gb = greedy_tokens(model, tok, [m["ids"] for m in ctx], 32)
    rowsB, B_bio, B_nob, keepB = [], [], [], []
    for i, (m, s, g) in enumerate(zip(ctx, std, gb)):
        pre = cut_before_value(tok, g, m["gold"])
        r = {"pid": m["pid"], "attr": m["attr"], "gold": m["gold"], "answer_bio": tok.decode(g), "found": pre is not None}
        if pre is not None:
            r.update({"prefix": tok.decode(pre), "prefix_len": len(pre),
                      "next_token_bio": tok.decode([g[len(pre)]]), "next_starts_value_bio": int(starts_value(tok, g[len(pre)], m["gold"]))})
            B_bio.append(m["ids"] + pre)
            B_nob.append(s["ids"] + pre)
            keepB.append(i)
        rowsB.append(r)
    nxt = next_tokens(model, B_nob, tok.pad_token_id)
    for j, i in enumerate(keepB):
        rowsB[i]["next_token_nobio"] = tok.decode([nxt[j]])
        rowsB[i]["next_starts_value_nobio"] = int(starts_any_value(tok, nxt[j], ctx[i]["attr"]))
        rowsB[i]["next_is_gold_start_nobio"] = int(starts_value(tok, nxt[j], ctx[i]["gold"]))
    log(f"variant B: value found in {len(keepB)}/150 answers; next token starts the value (bio) "
        f"{sum(rowsB[i]['next_starts_value_bio'] for i in keepB) / max(1, len(keepB)):.3f}; example prefix {rowsB[keepB[0]]['prefix']!r}")

    cells = {"A_bio": (A_bio, list(range(150))), "A_nobio": (A_nob, list(range(150))),
             "B_bio": (B_bio, keepB), "B_nobio": (B_nob, keepB)}
    vecs = {}
    for name, (seqs, idx) in cells.items():
        vecs[name] = extract(model, seqs, [[len(x) - 1] for x in seqs], layers=(20,), batch=32)[20][:, 0]
        log(name, "median norm", vecs[name].norm(dim=-1).median().item())
    out = RESULTS / "ans"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(vecs, RESULTS / "acts" / "orig_ans.pt")
    write_jsonl(out / "variantA_answers.jsonl", rowsA)
    write_jsonl(out / "variantB_answers.jsonl", rowsB)
    del model
    free()

    av = AVReader(snap(AV))
    expl = {}
    for name, (seqs, idx) in cells.items():
        expl[name] = av.verbalize(vecs[name], K=K, seed=SEEDS[name])
        log(name, f"verbalised ({time.time() - t0:.0f}s)")
    del av
    free()
    ar = ARReader(snap(AR))
    rows = []
    for name, (seqs, idx) in cells.items():
        flat = [e for per in expl[name] for e in per]
        c = cosines(vecs[name].repeat_interleave(K, dim=0), ar.reconstruct([e[0] for e in flat]))
        for j, i in enumerate(idx):
            m = ctx[i]
            for k in range(K):
                r = j * K + k
                rows.append({"cell": name, "variant": name[0], "cond": name[2:], "pid": m["pid"], "name": m["name"],
                             "group": "decoy", "attr": m["attr"], "gold": m["gold"], "golds": m["golds"],
                             "position": "answer_start", "k": k, "explanation": flat[r][0], "status": flat[r][1],
                             "cos": c[r].item(), "norm": vecs[name][j].norm().item()})
        log(name, "mean cos", c.mean().item())
    write_jsonl(out / "explanations.jsonl", rows)
    log(f"value-position job done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
