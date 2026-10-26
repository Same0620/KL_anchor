"""WS1 measurement engine (handoff 4.1, 10.1-10.4): gradients at the write, backward-only blocking, per-head factorised contributions,
the anchor-KL curvature, and the relay objective (C4). Used by Job C (checks) and Stage 1 (measurements); tested on CPU by
ws1_test_cpu.py.

Conventions:
  L = sum of answer-token NLL over the rows / N_total (N_total = the answer-token count of ALL analysis rows), so chunks add up;
  g(s) = sum_t dL/dh_ELL[t] (fp32 reduction of the bf16 gradient), split per row and per position class (prompt/postfix/answer);
  D_s(v) = -<g(s), v>; Delta_S(v) = D(v) - D^{-S}(v) = the amount carried by paths through >= 1 member of S (first order);
  c_h(v; s) = -<dL/dz_h, dz_h/dalpha> with dz/dalpha a central difference of cached z at s +- delta*v (same precision).
A blocked set is {"heads": [(block, head), ...], "mlps": [block, ...]}.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402


def chunks_by_length(rows, size):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    return [order[i:i + size] for i in range(0, len(order), size)]


class Engine:
    def __init__(self, model, tok, ell, rows, chunk=8, n_total=None, device="cuda"):
        self.model, self.tok, self.ell, self.rows, self.dev = model, tok, ell, rows, device
        self.rig = W.Rig(model, ell)
        self.n_total = n_total if n_total is not None else W.n_answer_tokens(rows)
        self.chunks = chunks_by_length(rows, chunk)
        self.gb = self.rig.gate_blocks
        self.nh, self.hd = self.rig.nh, self.rig.hd
        self.rdt = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32   # reduction dtype

    def up(self, x):
        return x.to(self.rdt)

    def close(self):
        self.rig.remove()

    def batch(self, idx):
        return W.collate([self.rows[i] for i in idx], self.tok.pad_token_id, self.dev)

    def _forward(self, b, s, grad, capture=False, freeze=None):
        if s is not None:
            s = torch.as_tensor(s).to(self.dev)
        r = self.rig
        r.mode, r.s, r.mask = "root", s, b["attention_mask"]
        r.capture = {"attn": {}, "mlp": {}} if capture else None
        r.freeze = freeze
        with torch.set_grad_enabled(grad):
            out = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            lg = self.up(out.logits[:, :-1])
            nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), b["labels"][:, 1:], ignore_index=-100, reduction="none").sum(1)
            loss = nll.sum() / self.n_total
        cap = r.capture
        r.capture, r.freeze = None, None
        self.last_rows = (nll.detach().double() / self.n_total).cpu().numpy()
        return loss, cap

    @torch.no_grad()
    def losses(self, s, freeze_fn=None):
        """Per-row loss pieces (sum of NLL / N_total), float64 numpy in row order."""
        out = np.zeros(len(self.rows))
        for idx in self.chunks:
            b = self.batch(idx)
            self._forward(b, s, grad=False, freeze=freeze_fn(idx) if freeze_fn else None)
            out[idx] = self.last_rows
        return out

    def reduce(self, g, b):
        """[B, T, d] gradient -> [B, 3, d] fp32 sums per position class."""
        g = self.up(g)
        cls = b["cls"]
        return torch.stack([(g * (cls == c)[..., None]).sum(1) for c in range(3)], 1)

    def point(self, s, sets=None, dirs=None, maps_dir=None, delta=None, keep_rows=True):
        """One point s. Returns:
          'g': [n_rows, 3, d] unblocked (if keep_rows) and 'g_sum': [d];
          'proj': {set_name: [n_rows, 3, n_dirs]} projections of g^{-S} (and '' for unblocked) on dirs (unit, [n_dirs, d]);
          'g_sum_S': {set_name: [d]};
          'maps' (if maps_dir is given): {'attn': [n_rows, n_gate_blocks, n_heads], 'mlp': [n_rows, n_gate_blocks]} = c(maps_dir; s)."""
        sets = sets or {}
        if s is not None:
            s = torch.as_tensor(s).to(self.dev)
        n = len(self.rows)
        d = self.rig.d
        dirs_t = None if dirs is None else torch.as_tensor(np.asarray(dirs), dtype=self.rdt, device=self.dev)
        res = {"proj": {k: np.zeros((n, 3, len(dirs))) for k in [""] + list(sets)} if dirs is not None else None,
               "g_sum": torch.zeros(d, dtype=torch.float64), "g_sum_S": {k: torch.zeros(d, dtype=torch.float64) for k in sets},
               "loss_rows": np.zeros(n)}
        if keep_rows:
            res["g"] = torch.zeros(n, 3, d, dtype=self.rdt)
        if maps_dir is not None:
            nb = len(self.gb)
            res["maps"] = {"attn": np.zeros((n, nb, self.nh)), "mlp": np.zeros((n, nb))}
            vdir = torch.as_tensor(np.asarray(maps_dir), dtype=self.rdt, device=self.dev)
        for idx in self.chunks:
            b = self.batch(idx)
            loss, cap = self._forward(b, s, grad=True, capture=maps_dir is not None)
            W.GATE_MASKS.clear()
            if maps_dir is not None:
                Za = [cap["attn"][L] for L in self.gb]
                Zm = [cap["mlp"][L] for L in self.gb]
                gr = torch.autograd.grad(loss, [self.rig.h_root] + Za + Zm, retain_graph=bool(sets))
                g0, dZa, dZm = gr[0], gr[1:1 + len(Za)], gr[1 + len(Za):]
            else:
                g0 = torch.autograd.grad(loss, self.rig.h_root, retain_graph=bool(sets))[0]
            red = self.reduce(g0, b)
            if keep_rows:
                res["g"][idx] = red.cpu()
            res["g_sum"] += red.sum((0, 1)).double().cpu()
            if dirs_t is not None:
                res["proj"][""][idx] = (red @ dirs_t.T).double().cpu().numpy()
            res["loss_rows"][idx] = self.last_rows
            for k, (name, S) in enumerate(sets.items()):
                self.rig.set_masks(S, self.dev)
                gS = torch.autograd.grad(loss, self.rig.h_root, retain_graph=k < len(sets) - 1)[0]
                W.GATE_MASKS.clear()
                rS = self.reduce(gS, b)
                res["g_sum_S"][name] += rS.sum((0, 1)).double().cpu()
                if dirs_t is not None:
                    res["proj"][name][idx] = (rS @ dirs_t.T).double().cpu().numpy()
            if maps_dir is not None:
                dza, dzm = self.fd_z(b, s, vdir, delta)
                for j, L in enumerate(self.gb):
                    prod = (self.up(dZa[j]) * dza[j]).reshape(len(idx), -1, self.nh, self.hd).sum((1, 3))      # [B, heads]
                    res["maps"]["attn"][idx, j, :] = (-prod).double().cpu().numpy()
                    res["maps"]["mlp"][idx, j] = (-(self.up(dZm[j]) * dzm[j]).sum((1, 2))).double().cpu().numpy()
                del dZa, dZm, dza, dzm
            del loss, cap, g0
        return res

    @torch.no_grad()
    def fd_z(self, b, s, v, delta):
        """Central difference of every gated block's o_proj input and MLP output along v (fp32 results)."""
        sp = v * delta if s is None else s + v * delta
        sm = -v * delta if s is None else s - v * delta
        _, cp = self._forward(b, sp, grad=False, capture=True)
        _, cm = self._forward(b, sm, grad=False, capture=True)
        dza = [(self.up(cp["attn"][L]) - self.up(cm["attn"][L])) / (2 * delta) for L in self.gb]
        dzm = [(self.up(cp["mlp"][L]) - self.up(cm["mlp"][L])) / (2 * delta) for L in self.gb]
        return dza, dzm

    def exact_single_heads(self, s, v, heads):
        """D(v) - D^{-{h}}(v) for each head, by exact single-head blocking (C3). Returns (D, {h: Delta})."""
        vt = torch.as_tensor(np.asarray(v), dtype=self.rdt, device=self.dev)
        if s is not None:
            s = torch.as_tensor(s).to(self.dev)
        D = 0.0
        out = {h: 0.0 for h in heads}
        for idx in self.chunks:
            b = self.batch(idx)
            loss, _ = self._forward(b, s, grad=True)
            W.GATE_MASKS.clear()
            g0 = self.reduce(torch.autograd.grad(loss, self.rig.h_root, retain_graph=True)[0], b).sum((0, 1))
            D += -float(g0 @ vt)
            for k, h in enumerate(heads):
                self.rig.set_masks({"heads": [h]}, self.dev)
                gh = self.reduce(torch.autograd.grad(loss, self.rig.h_root, retain_graph=k < len(heads) - 1)[0], b).sum((0, 1))
                W.GATE_MASKS.clear()
                out[h] += -float(g0 @ vt) + float(gh @ vt)
        return D, out

    # ---------------------------------------------------------- KH: the anchor-KL curvature along w (s = 0)
    def kl_curvature(self, w, delta, sets=None, maps=True):
        """On these rows (the anchor analysis rows) at s = 0: lg = logits on answer positions; dlg = d lg / d alpha along w (central
        difference); u = (diag p - p p^T) dlg held fixed; Q = sum u . lg / n_tok.  Returns the total curvature (u . dlg summed / n_tok),
        <dQ/ds, w> unblocked and with each set blocked, and per-row per-head c_h^KL = <dQ/dz_h, dz_h/dalpha>."""
        sets = sets or {}
        wt = torch.as_tensor(np.asarray(w), dtype=self.rdt, device=self.dev)
        n = len(self.rows)
        nb = len(self.gb)
        res = {"curv_rows": np.zeros(n), "dQ_w": 0.0, "dQ_w_S": {k: 0.0 for k in sets},
               "maps": {"attn": np.zeros((n, nb, self.nh)), "mlp": np.zeros((n, nb))}}
        ntok = self.n_total
        for idx in self.chunks:
            b = self.batch(idx)
            sel = (b["labels"][:, 1:] != -100)
            with torch.no_grad():
                lgp, zp = self._answer_logits(b, wt * delta, grad=False)
                lgm, zm = self._answer_logits(b, -wt * delta, grad=False)
                dlg = (lgp - lgm) / (2 * delta)                                                   # [ntok_chunk, V]
                dza = [(self.up(zp["attn"][L]) - self.up(zm["attn"][L])) / (2 * delta) for L in self.gb]
                dzm = [(self.up(zp["mlp"][L]) - self.up(zm["mlp"][L])) / (2 * delta) for L in self.gb]
                del lgp, lgm, zp, zm
            with torch.enable_grad():
                lg, cap = self._answer_logits(b, None, grad=True)
                with torch.no_grad():
                    p = torch.softmax(lg, -1)
                    u = p * dlg - p * (p * dlg).sum(-1, keepdim=True)
                    per_tok = (u * dlg).sum(-1)
                rowid = torch.arange(len(idx), device=self.dev)[:, None].expand_as(sel)[sel]
                res["curv_rows"][idx] = (torch.zeros(len(idx), device=self.dev, dtype=torch.float64)
                                         .index_add_(0, rowid, per_tok.double()) / ntok).cpu().numpy()
                Q = (u * lg).sum() / ntok
                Za = [cap["attn"][L] for L in self.gb]
                Zm = [cap["mlp"][L] for L in self.gb]
                W.GATE_MASKS.clear()
                gr = torch.autograd.grad(Q, [self.rig.h_root] + Za + Zm, retain_graph=bool(sets))
                res["dQ_w"] += float(self.reduce(gr[0], b).sum((0, 1)) @ wt)
                for j, L in enumerate(self.gb):
                    prod = (self.up(gr[1 + j]) * dza[j]).reshape(len(idx), -1, self.nh, self.hd).sum((1, 3))
                    res["maps"]["attn"][idx, j, :] = prod.double().cpu().numpy()
                    res["maps"]["mlp"][idx, j] = (self.up(gr[1 + nb + j]) * dzm[j]).sum((1, 2)).double().cpu().numpy()
                for k, (name, S) in enumerate(sets.items()):
                    self.rig.set_masks(S, self.dev)
                    gS = torch.autograd.grad(Q, self.rig.h_root, retain_graph=k < len(sets) - 1)[0]
                    W.GATE_MASKS.clear()
                    res["dQ_w_S"][name] += float(self.reduce(gS, b).sum((0, 1)) @ wt)
            del cap, gr, lg, dlg, dza, dzm
        res["curv_total"] = float(res["curv_rows"].sum())
        return res

    def _wu32(self):
        if getattr(self, "_WU32", None) is None:
            self._WU32 = self.model.lm_head.weight.detach().to(self.rdt)
        return self._WU32

    def _answer_logits(self, b, s, grad):
        """Logits at answer-target positions, computed in fp32 (fp64 for the test model) from the model's final normed hidden state:
        bf16 logits (spacing ~0.06-0.125 at |logit| 16-32) are too coarse for a central difference (Stage 1 maps run, D7)."""
        self.rig.mode, self.rig.s, self.rig.mask = "root", s, b["attention_mask"]
        self.rig.capture = {"attn": {}, "mlp": {}}
        sel = (b["labels"][:, 1:] != -100)
        with torch.set_grad_enabled(grad):
            hN = self.model.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
            lg = self.up(hN[:, :-1][sel]) @ self._wu32().T
        cap = self.rig.capture
        self.rig.capture = None
        return lg, cap


# ---------------------------------------------------------------- frozen-S finite differences (C2) and plain FD (C1)
@torch.no_grad()
def fd_rate(eng, v, eps, freeze_sets=None):
    """[L(-eps v) - L(+eps v)] / (2 eps) over eng's rows; with freeze_sets, S's outputs are overwritten by their s = 0 values."""
    vt = torch.as_tensor(np.asarray(v), dtype=eng.rdt, device=eng.dev)
    if not freeze_sets:
        return float((eng.losses(-eps * vt).sum() - eng.losses(eps * vt).sum()) / (2 * eps))
    cache = {}
    for idx in eng.chunks:
        b = eng.batch(idx)
        _, cap = eng._forward(b, None, grad=False, capture=True)
        cache[tuple(idx)] = cap
    heads, mlps = {}, set(freeze_sets.get("mlps", []))
    for L, h in freeze_sets.get("heads", []):
        heads.setdefault(L, []).append(h)

    def fz(idx):
        cap = cache[tuple(idx)]
        return {"attn": {L: (hs, cap["attn"][L]) for L, hs in heads.items()}, "mlp": {L: cap["mlp"][L] for L in mlps}}
    lm = eng.losses(-eps * vt, freeze_fn=fz).sum()
    lp = eng.losses(eps * vt, freeze_fn=fz).sum()
    return float((lm - lp) / (2 * eps))


# ---------------------------------------------------------------- C4 relay objective
class Relay:
    """Raw-text prompts 'Remember this word: W.' + filler + 'The word I asked you to remember is'; target ' W'. The write is added at
    W's first position only; the loss is the target's NLL at the last position (mean over prompts)."""

    def __init__(self, model, tok, ell, prompts, device="cuda"):
        self.model, self.tok, self.dev = model, tok, device
        self.rig = W.Rig(model, ell)
        self.items = prompts            # [{"ids": [...], "pos": int, "target": int, "word": str}]
        self.rdt = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32

    def rate(self, v_of, sets=None, exactly=None):
        """Returns D (sum over prompts of -<dLoss/ds, v_W>) and Delta_S for each set; v_of(word) -> unit vector at ELL."""
        sets = sets or {}
        D, out = 0.0, {k: 0.0 for k in sets}
        for it in self.items:
            ids = torch.tensor([it["ids"]], device=self.dev)
            m = torch.zeros(1, len(it["ids"]), device=self.dev)
            m[0, it["pos"]] = 1
            self.rig.mode, self.rig.s, self.rig.mask = "root", None, m
            v = torch.as_tensor(v_of(it["word"]), dtype=self.rdt, device=self.dev)
            with torch.enable_grad():
                o = self.model(input_ids=ids, use_cache=False)
                loss = -torch.log_softmax(o.logits[0, -1].to(self.rdt), -1)[it["target"]] / len(self.items)
                W.GATE_MASKS.clear()
                g = torch.autograd.grad(loss, self.rig.h_root, retain_graph=bool(sets))[0][0, it["pos"]].to(self.rdt)
                d0 = -float(g @ v)
                D += d0
                for k, (name, S) in enumerate(sets.items()):
                    self.rig.set_masks(S, self.dev)
                    gS = torch.autograd.grad(loss, self.rig.h_root, retain_graph=k < len(sets) - 1)[0][0, it["pos"]].to(self.rdt)
                    W.GATE_MASKS.clear()
                    out[name] += d0 + float(gS @ v)
        return D, out


def all_heads(gate_blocks, nh):
    return [(L, h) for L in gate_blocks for h in range(nh)]


def group_sets(ell, b0, b1, n_layers, nh):
    blocks = list(range(ell + 1, n_layers))
    pre = [L for L in blocks if L < b0]
    band = [L for L in blocks if b0 <= L <= b1]
    post = [L for L in blocks if L > b1]
    return {"ATT_pre": {"heads": all_heads(pre, nh)}, "MLP_pre": {"mlps": pre},
            "ATT_band": {"heads": all_heads(band, nh)}, "MLP_band": {"mlps": band},
            "ATT_post": {"heads": all_heads(post, nh)}, "MLP_post": {"mlps": post},
            "PRE": {"heads": all_heads(pre, nh), "mlps": pre},
            "POST": {"heads": all_heads(post, nh), "mlps": post},
            "ALL": {"heads": all_heads(blocks, nh), "mlps": blocks}}
