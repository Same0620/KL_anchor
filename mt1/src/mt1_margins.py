"""MT1 teacher-forced margins (handoff B.1-B.2; Job 2, and the margin evaluations of Jobs 3-4).

Candidate answers per episode (B.2): 1-run A_C, A_K; 2-run A_CC, A_KK, A_CK, A_KC (exact assignment_line format).
logp(A) = sum over the scored span of PRE(P) + A + "<end_of_turn>" of log_softmax(logits)[token] (bf16 weights, float32
log-softmax, model.eval()). T1: tok(S) starts with tok(PRE(P)) (else tokens_ok = false, episode excluded); T2: tok(PRE(P) +
"Assignment: R_1=X_1;") is a prefix of tok(S) (else m_run1 = NaN for that episode only).

Two scoring paths, numerically compared on 20 episodes before the cached one is used (handoff I8, 1e-3 nats):
- uncached (the definition): full sequences, batched with left padding and explicit position ids, logits kept for the
  scored span only;
- cached: the prompt once (batch 1, no padding), its KV cache repeated over the 2-4 candidates, which are scored as a
  right-padded batch continuing at the same positions.
"""
import math

import torch

END = "<end_of_turn>"


def PRE(p):
    return "<bos><start_of_turn>user\n" + p + "<end_of_turn>\n<start_of_turn>model\n"


def ids(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def candidates(episode, assignment_line):
    """-> [(name, plan)] in B.2's order."""
    C, K = episode.charter_plan, episode.coin_plan
    if len(episode.runs) == 1:
        return [("C", C), ("K", K)]
    return [("CC", C), ("KK", K), ("CK", (C[0], K[1])), ("KC", (K[0], C[1]))]


def prepare(tok, prompt, episode, assignment_line):
    """Token bookkeeping for one episode: prefix ids, candidate full ids, T1/T2 flags, run-1 prefix lengths."""
    pre = ids(tok, PRE(prompt))
    assert pre.count(tok.bos_token_id) == 1 and pre[0] == tok.bos_token_id, "B.1 BOS"
    out = {"pre": pre, "cands": {}, "t1": True, "t2": True, "n1": {}}
    for name, plan in candidates(episode, assignment_line):
        full = ids(tok, PRE(prompt) + assignment_line(episode, plan) + END)
        ok = full[:len(pre)] == pre
        out["t1"] &= ok
        out["cands"][name] = full
        if len(episode.runs) == 2 and name in ("CC", "KK"):
            pref = ids(tok, PRE(prompt) + f"Assignment: {episode.runs[0].run_id}={plan[0]};")
            t2 = full[:len(pref)] == pref
            out["t2"] &= t2
            out["n1"][name] = len(pref) - len(pre)
    return out


@torch.inference_mode()
def score_uncached(model, seqs, n_pre, pad_id, budget_tokens=48_000, device="cuda"):
    """seqs: list of full id lists; n_pre: list of prefix lengths. -> list of per-token logp lists over the scored span."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res = [None] * len(seqs)
    i = 0
    while i < len(order):
        bs = 1
        while i + bs < len(order) and (bs + 1) * len(seqs[order[i + bs]]) <= budget_tokens:
            bs += 1
        chunk = order[i:i + bs]
        L = max(len(seqs[j]) for j in chunk)
        span = max(len(seqs[j]) - n_pre[j] for j in chunk)
        inp = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, j in enumerate(chunk):
            x = seqs[j]
            inp[k, L - len(x):] = torch.tensor(x)
            att[k, L - len(x):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=inp.to(device), attention_mask=att.to(device), position_ids=pos.to(device),
                    logits_to_keep=span + 1, use_cache=False)
        lp = torch.log_softmax(out.logits.float(), dim=-1)          # [B, span+1, V], left-aligned to the sequence end
        for k, j in enumerate(chunk):
            x = seqs[j]
            m = len(x) - n_pre[j]                                    # scored tokens = the last m of the sequence
            tgt = torch.tensor(x[len(x) - m:], device=lp.device)
            rows = lp[k, span + 1 - m - 1: span + 1 - 1]            # logits at positions len-m-1 .. len-2 predict them
            res[j] = rows.gather(-1, tgt[:, None]).squeeze(-1).tolist()
        i += bs
    return res


@torch.inference_mode()
def score_cached(model, pre, conts, pad_id, device="cuda"):
    """pre: prefix ids (one episode); conts: the candidates' continuation id lists. -> per-token logp lists."""
    out = model(input_ids=torch.tensor([pre], device=device), use_cache=True)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)  # predicts each candidate's first token
    cache = out.past_key_values
    n = len(conts)
    cache.batch_repeat_interleave(n)
    Lc = max(len(c) for c in conts)
    inp = torch.full((n, Lc), pad_id, dtype=torch.long)
    att = torch.ones((n, len(pre) + Lc), dtype=torch.long)
    for k, c in enumerate(conts):
        inp[k, :len(c)] = torch.tensor(c)
        att[k, len(pre) + len(c):] = 0                              # right padding of the shorter candidates
    pos = torch.arange(len(pre), len(pre) + Lc)[None].expand(n, -1)
    o2 = model(input_ids=inp.to(device), attention_mask=att.to(device), position_ids=pos.to(device),
               past_key_values=cache, cache_position=torch.arange(len(pre), len(pre) + Lc, device=device), use_cache=True)
    lp = torch.log_softmax(o2.logits.float(), dim=-1)
    res = []
    for k, c in enumerate(conts):
        vals = [float(first[c[0]])]
        if len(c) > 1:
            tgt = torch.tensor(c[1:], device=lp.device)
            vals += lp[k, :len(c) - 1].gather(-1, tgt[:, None]).squeeze(-1).tolist()
        res.append(vals)
    return res


def margins_row(rec_id, meta, prep, logps, n_runs):
    """B.2's per-episode row from per-candidate per-token logps."""
    L = {k: sum(v) for k, v in logps.items()}
    nan = float("nan")
    row = {"id": rec_id, "clause": meta["target_clause"], "mixture": meta["mixture"], "n_runs": n_runs,
           "tokens_ok": bool(prep["t1"]), "t2_ok": bool(prep["t2"])}
    if n_runs == 1:
        row.update(logp_CC=L["C"], logp_KK=L["K"], logp_CK=None, logp_KC=None, m_ep=L["C"] - L["K"],
                   m_run1=L["C"] - L["K"], m_run2_givenC=None, m_run2_givenK=None, m_run2=None)
    else:
        m1 = (sum(logps["CC"][:prep["n1"]["CC"]]) - sum(logps["KK"][:prep["n1"]["KK"]])) if prep["t2"] else nan
        gc, gk = L["CC"] - L["CK"], L["KC"] - L["KK"]
        row.update(logp_CC=L["CC"], logp_KK=L["KK"], logp_CK=L["CK"], logp_KC=L["KC"], m_ep=L["CC"] - L["KK"],
                   m_run1=m1, m_run2_givenC=gc, m_run2_givenK=gk, m_run2=(gc + gk) / 2)
    return row


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))
