"""Site states and subspace energies for Exp 23 Part A (handoff 4.2-4.4; PREREG B-A4, B-A5).

States: right-padded batches, output_hidden_states=True; block L = hidden_states[L + 1]; the 9 postfix positions are the last 9 real
tokens of every prompt (the prompt ids end with POSTFIX_IDS, asserted by the caller through Z.prompt_ids).
Energy of a change d in a subspace U (orthonormal columns, [H, k]; None = identity): ||U^T d||^2 = ||P d||^2.
"""
import torch

NPOS = 9


def postfix_index(lens, device):
    return torch.stack([torch.arange(n - NPOS, n) for n in lens]).to(device)         # [B, 9]


def gather_postfix(h, idx):
    """h [B, T, H], idx [B, 9] -> [B, 9, H]"""
    return h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1]))


def header_len(tok):
    """Tokens of the chat template before the question (the empty-question prompt is header + the 9 postfix tokens)."""
    import e23_common as E
    n = len(E.Z.prompt_ids(tok, "")) - NPOS
    assert n >= 1, n
    return n


def qtok_index(lens, hdr, device, npos=NPOS):
    """Decisions 04 M-qtok: the last `npos` tokens of the user question, i.e. the positions just before the 9-token postfix, never
    position 0; a prompt with fewer question tokens uses all of them. -> (idx [B, npos], valid mask [B, npos]), right-aligned."""
    idx, msk = [], []
    for L in lens:
        hi = L - NPOS                                            # the postfix starts here
        lo = max(hdr, hi - npos, 1)                              # never position 0
        r = list(range(lo, hi))
        k = len(r)
        assert k >= 1, (L, hdr)
        idx.append([r[0]] * (npos - k) + r)                      # left-pad with a dummy index, masked out below
        msk.append([False] * (npos - k) + [True] * k)
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device)


def per_cell_masked(d, Us, mask):
    """d [n, nL, P, H], mask [n, P] -> mean over the valid prompts of the energy, per (block, position) [nL, P] float64."""
    out = []
    m = mask.to(d.device).double()
    den = m.sum(0).clamp(min=1.0)
    for i in range(d.shape[1]):
        e = energy(d[:, i], Us[i])                               # [n, P]
        out.append((e * m).sum(0) / den)
    return torch.stack(out)


def pad_batch(plist, pad_id, device):
    n = max(len(p) for p in plist)
    ids = torch.tensor([p + [pad_id] * (n - len(p)) for p in plist], device=device)
    att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in plist], device=device)
    return ids, att


def forward_site(model, ids, att, lens, layers):
    """One forward pass (grad as the caller's context decides) -> [B, len(layers), 9, H] in the model's dtype."""
    o = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False, logits_to_keep=1)   # hidden states only
    idx = postfix_index(lens, ids.device)
    return torch.stack([gather_postfix(o.hidden_states[L + 1], idx) for L in layers], 1)


def forward_at(model, ids, att, idx, layers):
    """One forward pass, gathering at the given positions idx [B, P] -> [B, len(layers), P, H] (grad as the caller's context decides)."""
    o = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    return torch.stack([gather_postfix(o.hidden_states[L + 1], idx) for L in layers], 1)


@torch.no_grad()
def site_states(model, plist, pad_id, layers, batch=32, device="cuda"):
    """[n, len(layers), 9, H] float32 (CPU) at the postfix positions of every prompt."""
    out = []
    for b in range(0, len(plist), batch):
        chunk = plist[b:b + batch]
        ids, att = pad_batch(chunk, pad_id, device)
        out.append(forward_site(model, ids, att, [len(p) for p in chunk], layers).float().cpu())
    return torch.cat(out)


def energy(d, U=None):
    """d [..., H]; U [H, k] or None -> ||U^T d||^2 [...] (float64 math on d's device)."""
    d = d.double()
    if U is None:
        return (d * d).sum(-1)
    p = d @ U.to(d.device).double()
    return (p * p).sum(-1)


def per_cell(d, Us):
    """d [n, nL, 9, H]; Us: list (one per layer) of [H, k] or None -> mean over prompts of the energy, [nL, 9] float64."""
    return torch.stack([energy(d[:, i], Us[i]).mean(0) for i in range(d.shape[1])])


def floored(E):
    """A4(e): floor E[l, t] at 1% of the pooled energy in the same subspace at that layer (mean over the 9 positions).
    -> (E_used [nL, 9], floor [nL], hits [nL, 9] bool)"""
    fl = 0.01 * E.mean(1)
    hits = E < fl[:, None]
    return torch.maximum(E, fl[:, None]), fl, hits


def lblk_value(d, Us, E_used):
    """L_blk of a set of prompts: mean over (layer, position, prompt) of ||P d||^2 / E_used[layer, position]."""
    return float((per_cell(d, Us) / E_used.double()).mean())


def lblk_heldout(d_arm, d_ref, Us):
    """Held-out L_blk normalised by the reference's energy on the same prompts (PREREG B-A5), with the A4(e) floor.
    -> (value, value per layer, number of floored cells)"""
    return lblk_heldout_E(d_arm, Us, per_cell(d_ref, Us))


def lblk_heldout_E(d_arm, Us, E_ref):
    """The same, from the reference's stored per-cell energies E_ref [nL, 9] (Job A-U) -> (value, per layer, floored cells)"""
    E_used, _, hits = floored(E_ref.double().to(d_arm.device))
    r = per_cell(d_arm, Us) / E_used
    return float(r.mean()), [float(x) for x in r.mean(1)], int(hits.sum())
