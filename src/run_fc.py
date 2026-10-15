"""Amendment 1 (docs/DECISIONS.md): forced-choice readouts on orig, in-context upper-bound test.
  B  AV forced choice: log P(value tokens | injected vector, fixed explanation opening), 40 candidates, 2 openings.
  C  AR forced choice: cos(AR reconstruction of a fixed sentence naming the candidate, activation), 2 sentences.
Vectors: decoy answer_start and name_end (with / without biography) from results/acts/orig.pt, plus the value token
inside the biography (positive control; extracted here). Writes results/fc/.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTS, AR, AV, ATTRS, DATA, ORIG, RESULTS, VALUES, load_tokenizer, read_jsonl
from gpulib import DEV, ARReader, AVReader, extract, free, load_lm, log
from reading import POS_NAMES
from run_a0 import snap
from run_diag_a0 import value_end_index

TYPE = {"city": "birthplace", "profession": "occupation", "instrument": "musical instrument"}
AV_T = ["<explanation>\nStructured answer format with a Q&A pattern about a person's {type}.\n\nThe answer is: {Value}",
        "<explanation>\nShort factual answer to a question about a person.\n\nThe response gives the answer {Value}"]
AR_T = ["Structured answer format with a Q&A pattern about a person's {type}.\n\nThe answer is: {Value}.",
        "Short factual answer to a question about a person.\n\nThe response gives the answer {Value}."]


LOWER = "--lower" in sys.argv           # Amendment 2: professions and instruments in lowercase (cities stay capitalised)


def cap(v):
    if LOWER and v[0].islower():
        return v
    return v[0].upper() + v[1:]


def build_av_conts(tok):
    """{(attr, t): (list of 40 token lists, n_shared)}: continuation tokens per candidate; tokens after the common
    prefix of all 40 are the scored ones."""
    out = {}
    for a in ATTRS:
        for t, tmpl in enumerate(AV_T):
            seqs = [tok(tmpl.format(type=TYPE[a], Value=cap(v)), add_special_tokens=False)["input_ids"] for v in VALUES[a]]
            n = 0
            while all(len(s) > n for s in seqs) and len({s[n] for s in seqs}) == 1:
                n += 1
            out[(a, t)] = (seqs, n)
            log(f"AV opening {t} {a}: shared {n} tokens, scored tokens per value {min(len(s) - n for s in seqs)}-{max(len(s) - n for s in seqs)}")
    return out


@torch.no_grad()
def av_scores(av, vecs, attrs, conts, vec_batch=8):
    """[N, 40] mean over openings of summed log-probs of the scored tokens."""
    from nla_inference import normalize_activation
    model = av.model
    emb = model.get_input_embeddings()
    T = av.ids.shape[1]
    W32 = model.lm_head.weight.float()
    out = torch.zeros(len(vecs), 40)
    for s in range(0, len(vecs), vec_batch):
        idx = list(range(s, min(s + vec_batch, len(vecs))))
        v = normalize_activation(vecs[idx].float().to(DEV), av.cfg.injection_scale)
        rows = []          # (vector row in batch, cand, opening, tokens, n_shared)
        for r, i in enumerate(idx):
            for t in range(len(AV_T)):
                seqs, n = conts[(attrs[i], t)]
                for c, sq in enumerate(seqs):
                    rows.append((r, c, t, sq, n))
        L = max(len(x[3]) for x in rows)
        B = len(rows)
        ids = torch.full((B, L), av.tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((B, T + L), dtype=torch.long)
        for b, (_, _, _, sq, _) in enumerate(rows):
            ids[b, :len(sq)] = torch.tensor(sq)
            am[b, :T + len(sq)] = 1
        E = av.base.expand(B, -1, -1).clone()
        E[:, av.p] = v[torch.tensor([x[0] for x in rows])]
        X = torch.cat([E.to(torch.bfloat16), emb(ids.to(DEV))], dim=1)
        h = model.model(inputs_embeds=X, attention_mask=am.to(DEV), use_cache=False).last_hidden_state
        # hidden at position T-1+j predicts continuation token j
        sel_b, sel_p, sel_tok, owner = [], [], [], []
        for b, (_, _, _, sq, n) in enumerate(rows):
            for j in range(n, len(sq)):
                sel_b.append(b)
                sel_p.append(T - 1 + j)
                sel_tok.append(sq[j])
                owner.append(b)
        hs = h[torch.tensor(sel_b, device=DEV), torch.tensor(sel_p, device=DEV)]
        logits = hs.float() @ W32.T                       # fp32 logits: bf16 is too coarse to rank 40 candidates
        lp = torch.log_softmax(logits, dim=-1)[torch.arange(len(sel_tok), device=DEV), torch.tensor(sel_tok, device=DEV)]
        tot = torch.zeros(B, device=DEV).index_add_(0, torch.tensor(owner, device=DEV), lp).cpu()
        for b, (r, c, t, _, _) in enumerate(rows):
            out[idx[r], c] += tot[b].item() / len(AV_T)
    return out


@torch.no_grad()
def ar_scores(ar, vecs, attrs):
    texts, keys = [], []
    for a in ATTRS:
        for t, tmpl in enumerate(AR_T):
            for c, v in enumerate(VALUES[a]):
                texts.append(tmpl.format(type=TYPE[a], Value=cap(v)))
                keys.append((a, t, c))
    rec = ar.reconstruct(texts)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    R = {k: rec[i] for i, k in enumerate(keys)}
    vn = vecs.float() / vecs.float().norm(dim=-1, keepdim=True)
    out = torch.zeros(len(vecs), 40)
    for i, a in enumerate(attrs):
        M = torch.stack([torch.stack([R[(a, t, c)] for c in range(40)]) for t in range(len(AR_T))]).mean(0)   # [40, d]
        out[i] = M @ vn[i]
    return out


def main():
    t0 = time.time()
    tok = load_tokenizer()
    ctx = read_jsonl(DATA / "read_ctx.jsonl")
    std_all = read_jsonl(DATA / "read_std.jsonl")
    std_idx = [i for i, m in enumerate(std_all) if m["group"] == "decoy"]
    std = [std_all[i] for i in std_idx]
    assert [(m["pid"], m["attr"]) for m in std] == [(m["pid"], m["attr"]) for m in ctx]
    acts = torch.load(ACTS / "orig.pt")
    sets = {"ctx_answer_start": acts["read_ctx"][:, POS_NAMES.index("answer_start")],
            "std_answer_start": acts["read_std"][std_idx, POS_NAMES.index("answer_start")],
            "ctx_name_end": acts["read_ctx"][:, POS_NAMES.index("name_end")],
            "std_name_end": acts["read_std"][std_idx, POS_NAMES.index("name_end")]}
    vpos = [value_end_index(tok, m["user"], m["gold"], m["ids"]) for m in ctx]
    model = load_lm(ORIG)
    sets["value_token"] = extract(model, [m["ids"] for m in ctx], [[p] for p in vpos], layers=(20,), batch=32)[20][:, 0]
    del model
    free()
    torch.save(sets, ACTS / ("orig_fc_vectors_lower.pt" if LOWER else "orig_fc_vectors.pt"))
    attrs = [m["attr"] for m in ctx]
    av = AVReader(snap(AV))
    conts = build_av_conts(av.tok)
    scores = {}
    for name, vecs in sets.items():
        scores[("B", name)] = av_scores(av, vecs, attrs, conts)
        log("B", name, f"done ({time.time() - t0:.0f}s)")
    del av
    free()
    ar = ARReader(snap(AR))
    for name, vecs in sets.items():
        scores[("C", name)] = ar_scores(ar, vecs, attrs)
    out = RESULTS / ("fc_lower" if LOWER else "fc")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({f"{m}|{s}": v for (m, s), v in scores.items()}, out / "scores.pt")
    json.dump({"meta": [{"pid": m["pid"], "name": m["name"], "attr": m["attr"], "gold": m["gold"],
                         "gold_idx": VALUES[m["attr"]].index(m["gold"])} for m in ctx],
               "value_tokens": [tok.decode([m["ids"][p]]) for m, p in zip(ctx, vpos)]},
              open(out / "meta.json", "w"), indent=0)
    log(f"forced-choice job done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
