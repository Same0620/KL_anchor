"""GPU helpers for Experiment 11: mean-ablation of two directions together.

`PairAblator` extends Experiment 9's `e9lib.Ablator` (same hooks on the outputs of blocks 10..30, same layer sets,
same token masks: `gen_mode` and `tf_mode` are inherited unchanged, so position 0 and the second sink token are left
alone). It takes K directions (K = 2 here) and sets ALL their projections to their base means in one step:

    h <- h - sum_j a_j v_j,   a = G^-1 (V h - c),   G = V V^T   (K x K, one per layer)

For two perpendicular directions this is the same as removing one after the other. Given ONE direction it does exactly
the arithmetic of Experiment 9's hook (unit vectors are not rescaled, so G is 1): the output is the same bit for bit. The cast back to 16-bit floats
moves the projections a little; as in Experiment 9 the rest is put right on 64 coordinates per direction where that
direction is large and the base activations are usually small (three rounds; the K x K system keeps the corrections
of the two directions from disturbing each other).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11
from e11common import X
import e9lib as L9

DEV = "cuda"


class PairAblator(L9.Ablator):
    def set(self, vs, c_prompt, c_answer, layer_set="L21", rms=None):
        """vs: K tensors [21, 5120] (unit rows); c_prompt, c_answer: K tensors [21]; rms as in Experiment 9."""
        self.V = torch.stack([v.float() for v in vs]).to(DEV)                    # [K, 21, 5120]
        self.cpk = torch.stack([c.float() for c in c_prompt]).to(DEV)            # [K, 21]
        self.cak = torch.stack([c.float() for c in c_answer]).to(DEV)
        Vl = self.V.permute(1, 0, 2).double()                                    # [21, K, 5120]
        G = Vl @ Vl.transpose(1, 2)                                              # [21, K, K]
        G.diagonal(dim1=1, dim2=2).fill_(1.0)                                    # unit vectors are taken as exactly unit, as in Experiment 9's hook
        self.Ginv = torch.linalg.inv(G).float()
        score = self.V.abs()
        if rms is not None:
            rms = rms.float().to(DEV)
            score = score * (rms <= rms.median(dim=-1, keepdim=True).values)
        idx = score.topk(64, dim=-1).indices                                     # [K, 21, 64]
        w = torch.gather(self.V, 2, idx)
        U = torch.zeros_like(self.V).scatter_(2, idx, w / w.pow(2).sum(-1, keepdim=True))      # V_j . U_j = 1
        Ul = U.permute(1, 0, 2).double()                                         # [21, K, 5120]
        M = Vl @ Ul.transpose(1, 2)                                              # M[i][j] = V_i . U_j
        M.diagonal(dim1=1, dim2=2).fill_(1.0)                                    # V_j . U_j = 1, as Experiment 9's hook assumes
        self.Minv = torch.linalg.inv(M).float()
        self.cols = [torch.unique(idx[:, k].reshape(-1)) for k in range(len(X.LAYERS))]
        self.Uc = [Ul[k][:, self.cols[k]].float() for k in range(len(X.LAYERS))]  # [K, coordinates]
        self.layers, self.on = set(X.LAYER_SETS[layer_set]), True

    def _hook(self, L):
        k = L - X.LAYERS[0]

        def hook(module, inp, out):
            if not self.on or L not in self.layers:
                return None
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            V = self.V[:, k]                                                        # [K, 5120]
            proj = lambda t: torch.stack([t @ V[j] for j in range(V.shape[0])], -1)    # [B, T, K]; one matrix-vector product per direction, as in Experiment 9
            p = proj(hf)
            if self.mode == "gen" and h.shape[1] > 1:
                tgt, mask = self.cpk[:, k].expand_as(p), self.pm
            elif self.mode == "gen":
                tgt, mask = self.cak[:, k].expand_as(p), torch.ones(p.shape[:2], dtype=torch.bool, device=p.device)
            else:
                tgt, mask = torch.where(self.am.unsqueeze(-1), self.cak[:, k], self.cpk[:, k]), self.pm | self.am
            m = mask.unsqueeze(-1)
            new = (hf - (((p - tgt) * m) @ self.Ginv[k]) @ V).to(h.dtype)
            cols, Uc, Minv = self.cols[k], self.Uc[k], self.Minv[k]
            for _ in range(3):
                r = (proj(new.float()) - tgt) * m                                   # what is left after the cast, [B, T, K]
                new[..., cols] = (new[..., cols].float() - (r @ Minv.T) @ Uc).to(h.dtype)
            self.calls += 1
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook


class PairProbe:
    """Reads what leaves a block AFTER the PairAblator's hook (register it after the ablator). Keeps, per kind of pass,
    the largest |h.v_j - c_j| / ||h|| over the directions j and the tokens that should have been ablated."""

    def __init__(self, base, abl):
        self.abl = abl
        self.reset()
        self.handles = [base.model.layers[L].register_forward_hook(self._hook(L)) for L in X.LAYERS]

    def reset(self):
        self.worst, self.n = {"prefill": 0.0, "generated": 0.0, "teacher-forced": 0.0}, {"prefill": 0, "generated": 0, "teacher-forced": 0}

    def _hook(self, L):
        k = L - X.LAYERS[0]

        def hook(module, inp, out):
            a = self.abl
            if not a.on or L not in a.layers:
                return None
            h = (out[0] if isinstance(out, tuple) else out).float()
            p, nrm = h @ a.V[:, k].T, h.norm(dim=-1)
            if a.mode == "gen" and h.shape[1] > 1:
                kind, tgt, m = "prefill", a.cpk[:, k].expand_as(p), a.pm
            elif a.mode == "gen":
                kind, tgt, m = "generated", a.cak[:, k].expand_as(p), torch.ones(p.shape[:2], dtype=torch.bool, device=p.device)
            else:
                kind, tgt, m = "teacher-forced", torch.where(a.am.unsqueeze(-1), a.cak[:, k], a.cpk[:, k]), a.pm | a.am
            if bool(m.any()):
                dev = ((p - tgt).abs().amax(-1) / nrm)[m]
                self.worst[kind] = max(self.worst[kind], float(dev.max()))
                self.n[kind] += int(m.sum())
            return None
        return hook
