"""Stage E12-4 (GPU): everything that needs a run's checkpoints except generation. One base-model load; adapters are swapped.

  e12_eval.py RUN [--smoke]   -> stats/eval_<RUN>.pt

At every checkpoint (step 0 = the base model):
  teacher-forced   M_in, M_out, M_evil (and M_odd for O / Xo runs), held-out loss, repair meter, per question, in every mode
                   normal-type runs (normal, KL, O-normal, Xo): none, v_EM, random 1, random 2 (inference-time ablation)
                   blocked runs (B, R1, R2, U): hook-on (the run's own block), hook-off
  write-level gate projection on v_EM,-d at the hooked layers, mean over answer tokens, 60 gate-reading + 48 everyday sequences
  rotation / leak  (student - base) mean activation on the 200 Dolly answers at layers 10..39: cosine with v_EM, projection, length
  read-side        (F-normal-s0, X-normal-s0, KL runs; steps 0, 5, 10, 20, 40, last) sums for the selectivity of every adapted matrix
  KL value         (KL runs) KL(base || student) per answer token on 100 Dolly answers
  U candidates     (the pilots F-normal-s0, X-normal-s0) the usage-matched direction of Stage E12-3, chosen here
Hooks use the in-topic base means on H_F / H_X text and the everyday base means on H_E and the M_evil probes.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e9_directions as D9
import e12lib as L12
import e4_scores as S4

DEV = "cuda"
HK, HOOKED = X12.HK, X12.HOOKED
L30 = list(range(10, 40))
RS_STEPS = (0, 5, 10, 20, 40)
U_STEPS = (0, 10, 15, 20, 30, 40)
T0 = time.time()


def log(msg):
    print(f"[eval {time.time() - T0:4.0f}s] {msg}", flush=True)


@torch.no_grad()
def forward(base, seqs, pad_id, batch=25):
    """-> yields (hidden_states, spans, ids, att, first index of the batch); no hook."""
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        spans = [(len(p), len(p) + len(a)) for p, a in chunk]
        ids, att = L9.pad_right([p + a for p, a in chunk], pad_id)
        out = base.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
        yield out.hidden_states, spans, ids, att, i, out.last_hidden_state


class ReadSide:
    """Sums for the read-side selectivity of every adapted matrix (Part B5-24 of the pre-registration).
    For the tokens selected by `self.sel` ([batch, T] bool) of context `self.g` (0 in-topic, 1 everyday) it adds up
    A.x, its outer products, A0.x and its squared length, x and its squared length."""

    def __init__(self, base, a0):
        self.mods, self.acc, self.sel, self.g, self.handles = [], {}, None, 0, []
        for name, m in base.named_modules():
            if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
                A = m.lora_A["default"].weight.detach().float()
                self.mods.append((name, m))
                self.acc[name] = {"A0": a0[f"base_model.model.{name}.lora_A.weight"].to(DEV).float(), "n": torch.zeros(2, dtype=torch.float64, device=DEV),
                                  "a": torch.zeros(2, A.shape[0], dtype=torch.float64, device=DEV), "aa": torch.zeros(2, A.shape[0], A.shape[0], dtype=torch.float64, device=DEV),
                                  "a0": torch.zeros(2, A.shape[0], dtype=torch.float64, device=DEV), "a0sq": torch.zeros(2, dtype=torch.float64, device=DEV),
                                  "x": torch.zeros(2, A.shape[1], dtype=torch.float64, device=DEV), "xsq": torch.zeros(2, dtype=torch.float64, device=DEV)}
                self.handles.append(m.register_forward_pre_hook(self._hook(name, m)))

    def _hook(self, name, m):
        def hook(module, args):
            if self.sel is None:
                return None
            c, g = self.acc[name], self.g
            x = args[0][self.sel].float()
            a, a0 = x @ m.lora_A["default"].weight.float().T, x @ c["A0"].T
            c["n"][g] += x.shape[0]
            c["a"][g] += a.double().sum(0)
            c["aa"][g] += a.double().T @ a.double()
            c["a0"][g] += a0.double().sum(0)
            c["a0sq"][g] += a0.double().pow(2).sum()
            c["x"][g] += x.double().sum(0)
            c["xsq"][g] += x.double().pow(2).sum()
            return None
        return hook

    def result(self):
        """Per adapted matrix: eta-squared of dW.x, A.x, x and A0.x, and the cosine and norm ratio of the two context means of dW.x."""
        out = {"modules": [n for n, _ in self.mods]}
        cols = {k: [] for k in ("eta_dWx", "eta_Ax", "eta_x", "eta_A0x", "cos_means_dWx", "norm_ratio_everyday_over_in_topic", "n_in", "n_everyday", "norm_B", "mean_sq_dWx")}

        def eta(mu0, mu1, e0, e1):
            mu = (mu0 + mu1) / 2
            tot = (e0 + e1) / 2 - mu.pow(2).sum()
            return float((mu0 - mu1).pow(2).sum() / 4 / tot) if float(tot) > 0 else float("nan")
        for name, m in self.mods:
            c = self.acc[name]
            n = c["n"]
            B, s = m.lora_B["default"].weight.detach().double(), float(m.scaling["default"])
            ma, G = c["a"] / n[:, None], B.T @ B
            mz = s * (ma @ B.T)                                               # context means of dW.x, [2, d_out]
            ez = [s * s * float((G * (c["aa"][g] / n[g])).sum()) for g in (0, 1)]
            cols["eta_dWx"].append(eta(mz[0], mz[1], ez[0], ez[1]) if float(B.abs().sum()) > 0 else float("nan"))
            cols["eta_Ax"].append(eta(ma[0], ma[1], float(torch.trace(c["aa"][0]) / n[0]), float(torch.trace(c["aa"][1]) / n[1])))
            cols["eta_x"].append(eta(c["x"][0] / n[0], c["x"][1] / n[1], float(c["xsq"][0] / n[0]), float(c["xsq"][1] / n[1])))
            cols["eta_A0x"].append(eta(c["a0"][0] / n[0], c["a0"][1] / n[1], float(c["a0sq"][0] / n[0]), float(c["a0sq"][1] / n[1])))
            nz = float(mz[0].norm()) > 0 and float(mz[1].norm()) > 0
            cols["cos_means_dWx"].append(float((mz[0] * mz[1]).sum() / (mz[0].norm() * mz[1].norm())) if nz else float("nan"))
            cols["norm_ratio_everyday_over_in_topic"].append(float(mz[1].norm() / mz[0].norm()) if nz else float("nan"))
            cols["n_in"].append(float(n[0]))
            cols["n_everyday"].append(float(n[1]))
            cols["norm_B"].append(float(B.norm()))
            cols["mean_sq_dWx"].append((ez[0] + ez[1]) / 2)
        out.update({k: torch.tensor(v) for k, v in cols.items()})
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    smoke = "--smoke" in sys.argv
    run = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    if smoke:
        X12.DOM["SMOKE"], X12.RPAIR["SMOKE"], X12.OWN_STYLE["SMOKE"] = "fin", X12.RPAIR["F"], "harm32"
    r = X12.parse_run(run)
    dom, hs, ds = r["dom"], r["hset"], r["set"]
    root = X12.SMOKE if smoke else X12.BASE
    rec = json.loads(((root / "ckpt") / run / "record.json").read_text())
    steps = rec["checkpoints"] + [rec["steps"]]
    normal_type = r["block"] is None
    pilot = run in ("F-normal-s0", "X-normal-s0", "O-normal-s0") or (smoke and r["arm"] == "normal")      # O-normal-s0: the register control U_O (Amendment 2, item 3)
    own_kind, own_grp, m_key, loss_key = ("odd32", "odd", "M_odd", "loss_odd") if ds in ("O", "Xo") else ("harm32", "harm", "M_in", "loss_harm")
    want_rs = pilot or r["lam"] is not None
    dirs = X12.load_dirs(smoke)
    vem30 = torch.load((X12.SMOKE if smoke else X12.DIRECTIONS) / "vem30.pt", map_location="cpu", weights_only=True)
    v30 = vem30["v"][vem30["names"].index(r["em"])].to(DEV)                    # [30, 5120]
    vem = dirs["v"][dirs["names"].index(r["em"])].float().to(DEV)              # [21, 5120]
    tok, pad_id, eos_ids = L9.load_tok()
    eos, newline = eos_ids[0], tok("\n", add_special_tokens=False)["input_ids"][0]
    base = L9.load_base()
    abl = L9.Ablator(base, newline)

    # ---- sequences
    H = X12.heldout(smoke)
    P = lambda q: L12.prompt_ids(tok, q)
    full = lambda rows, kind: [(P(x["question"]), L12.answer_ids(tok, x["answers"][kind]["text"], x["answers"][kind]["finished"], eos)) for x in rows]
    first = lambda rows, kind: [(P(x["question"]), L12.answer_ids(tok, x["answers"][kind]["text"], n_max=X12.N_FIRST)) for x in rows]
    nreal = lambda rows, kind: [min(X12.N_FIRST, len(tok(x["answers"][kind]["text"], add_special_tokens=False)["input_ids"])) for x in rows]
    grp_in = {"harm": full(H[hs], "harm32"), "plain": first(H[hs], "plain32"), "repair": full(H[hs], "plain14")}
    n_in = {"harm": nreal(H[hs], "harm32"), "plain": nreal(H[hs], "plain32")}
    if ds in ("O", "Xo"):
        grp_in["odd"], n_in["odd"] = full(H[hs], "odd32"), nreal(H[hs], "odd32")
    mev = C.read_jsonl(X.X8.X7.MEVIL_PAIRS)
    if smoke:
        mev = mev[:4]
    grp_ev = {"e_harm": first(H["E"], "harm32"), "e_plain": first(H["E"], "plain32"),
              "mev_bad": [tuple(S4.encode(tok, p["question"], p["bad"], p["bad_finished"])[:2]) for p in mev],
              "mev_good": [tuple(S4.encode(tok, p["question"], p["good"], p["good_finished"])[:2]) for p in mev]}
    fixed = C.read_jsonl(X12.E9_DATA / "fixed_answers.jsonl")
    dolly = [x for x in C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl") if x["part"] == "vEM"]
    gate_in, gate_ev, dol = D9.seqs_of(tok, [x for x in fixed if x["set"] == dom]), D9.seqs_of(tok, [x for x in fixed if x["set"] == "everyday"]), D9.seqs_of(tok, dolly)
    if smoke:
        gate_in, gate_ev, dol = gate_in[:6], gate_ev[:6], dol[:12]
    # read-side: every answer token in-topic; the everyday answer tokens subsampled once to the same number
    n_tok_in = sum(len(a) for _, a in gate_in)
    n_tok_ev = sum(len(a) for _, a in dol)
    g = torch.Generator().manual_seed(C.stable_seed("e12-readside", dom))
    keep_ev = torch.zeros(n_tok_ev, dtype=torch.bool)
    keep_ev[torch.randperm(n_tok_ev, generator=g)[:min(n_tok_in, n_tok_ev)]] = True
    off_ev = [0]
    for _, a in dol:
        off_ev.append(off_ev[-1] + len(a))

    own = "none" if normal_type else "on"
    modes = [("none", None), ("EM", r["em"]), ("r1", r["rpair"][0]), ("r2", r["rpair"][1])] if normal_type else [("on", r["block"]), ("off", None)]

    def tf(seq_groups, cset, set_fn):
        """-> {group: list of per-sequence log-probability tensors}; set_fn switches the hook on (or is None)."""
        names = list(seq_groups)
        flat = [s for n in names for s in seq_groups[n]]
        a = set_fn(cset) if set_fn else None
        lps = L12.tf_logp(base, flat, pad_id, a)
        abl.off()
        out, i = {}, 0
        for n in names:
            out[n] = lps[i:i + len(seq_groups[n])]
            i += len(seq_groups[n])
        return out

    m128 = lambda lps, ns: torch.tensor([float(lp[:n].mean()) for lp, n in zip(lps, ns)])
    lossv = lambda lps: torch.tensor([float(-lp.mean()) for lp in lps])
    BANDS = ((0, 10), (10, 40), (40, 128))                                       # answer tokens 1-10, 11-40, 41-128 (Amendment 2, item 6)
    band = lambda lps, ns, lo, hi: torch.tensor([float(lp[lo:min(hi, n)].mean()) if n > lo else float("nan") for lp, n in zip(lps, ns)])

    def measures(set_fn, with_repair, in_only=False):
        gi = {k: v for k, v in grp_in.items() if k != "repair" or with_repair}
        if in_only:                                                              # candidates for U: the run's own held-out rows and the plain answers
            gi = {k: gi[k] for k in dict.fromkeys((own_grp, "plain"))}
        a = tf(gi, dom, set_fn)
        mp = m128(a["plain"], n_in["plain"])
        out = {"m_plain": mp}
        if "harm" in a:
            mh = m128(a["harm"], n_in["harm"])
            out.update({"M_in": mh - mp, "m_harm": mh, "loss_harm": lossv(a["harm"])})
            for i, (lo, hi) in enumerate(BANDS):
                out[f"M_in_b{i}"] = band(a["harm"], n_in["harm"], lo, hi) - band(a["plain"], n_in["plain"], lo, hi)
        if "odd" in a:
            mo = m128(a["odd"], n_in["odd"])
            out.update({"M_odd": mo - mp, "m_odd": mo, "loss_odd": lossv(a["odd"])})
        if in_only:
            return out
        out["heldout"] = out["loss_odd"] if ds in ("O", "Xo") else out["loss_harm"]
        if "repair" in a:
            out["repair"] = lossv(a["repair"])
        e = tf(grp_ev, "everyday", set_fn)
        full_n = [10 ** 9] * len(grp_ev["e_harm"])
        out["M_out"] = m128(e["e_harm"], full_n) - m128(e["e_plain"], full_n)          # the sequences are already cut at 128 tokens
        out["M_evil"] = torch.tensor([float(b.mean() - g_.mean()) for b, g_ in zip(e["mev_bad"], e["mev_good"])])
        return out

    # ---- the pilots: candidates for the usage-matched direction U (PCA of the step-40 change, v_EM and v_habit projected out)
    stats = {"run": run, "steps": steps, "modes": [m for m, _ in modes], "tf": {}, "gate": {}, "gate_a": {}, "rot": {}, "readside": {}, "kl": {}, "info": {"own_mode": own, "normal_type": normal_type, "pilot": pilot}}
    ucand = None
    u_only = "--u-only" in sys.argv                                              # recompute only the choice of U (and fallback C) from a finished evaluation
    if u_only:
        stats = torch.load((X12.SMOKE if smoke else X12.BASE) / "stats" / f"eval_{run}.pt", map_location="cpu", weights_only=False)
        cnd = torch.load((X12.SMOKE if smoke else X12.DIRECTIONS) / f"U_candidates_{ds}.pt", map_location="cpu", weights_only=True)
        ucand = {"v": cnd["v"], "cbar": cnd["cbar"], **{k: stats["U"][k] for k in ("variance_share", "cos_with_v_EM", "tokens")}}
    if (pilot and 40 in steps or (smoke and pilot)) and not u_only:
        import e12_directions as D12
        ustep = 40 if 40 in steps else steps[-1]
        useq = first(H[hs], own_kind)

        @torch.no_grad()
        def acts():
            rows = []
            for hsx, spans, _, _, _, _ in forward(base, useq, pad_id):
                for b, (lo, hi) in enumerate(spans):
                    rows.append(torch.stack([hsx[L + 1][b, lo:hi].float() for L in HOOKED], 1))        # [tokens, 6, 5120]
            return torch.cat(rows)
        a0 = acts()
        pm = L12.attach_dir(base, X12.ckpt_dir(run, ustep, smoke))
        D = acts() - a0
        L9.detach(pm)
        del a0
        hab = dirs["v"][dirs["names"].index(f"hab-{dom}")].float().to(DEV)
        for j, k in enumerate(HK):
            for v in (vem[k], hab[k]):
                D[:, j] -= (D[:, j] @ v)[:, None] * v
        Dm = D.reshape(D.shape[0], -1).double()
        ev_, evec = torch.linalg.eigh(Dm @ Dm.T)                                  # uncentred PCA through the Gram matrix
        top = evec[:, -10:].flip(1)
        comp = (Dm.T @ top).T.reshape(10, len(HK), -1)                            # [10, 6, 5120]
        comp = (comp / comp.norm(dim=-1, keepdim=True)).float()
        share_var = (ev_.flip(0)[:10] / ev_.sum()).tolist()
        V = torch.zeros(10, len(X.LAYERS), comp.shape[-1])
        for k in range(len(X.LAYERS)):                                            # rows of layers that are never hooked: the nearest hooked layer's part (unused)
            V[:, k] = comp[:, min(range(len(HK)), key=lambda j: abs(HK[j] - k))].cpu()
        cb = {"everyday": D12.cbar_of(base, gate_ev, pad_id, V, X.LAYERS), dom: D12.cbar_of(base, gate_in, pad_id, V, X.LAYERS)}
        ucand = {"v": V, "cbar": cb, "variance_share": share_var, "tokens": int(D.shape[0]), "step": ustep,
                 "cos_with_v_EM": [[float((V[c, k] * vem[k].cpu()).sum()) for k in HK] for c in range(10)]}
        del D, Dm, evec
        torch.cuda.empty_cache()
        log(f"U candidates from the step-{ustep} change: {ucand['tokens']} tokens; variance shares of the top components {[round(x, 3) for x in share_var[:5]]}")

    def cand_fn(c):
        def f(cset):
            abl.set(ucand["v"][c], ucand["cbar"][cset]["prompt"][c], ucand["cbar"][cset]["answer"][c], X12.LS, dirs["rms"])
            return abl
        return f

    a0_file = X12.ckpt_dir(run, 0, smoke) / "adapter_model.safetensors"
    base_mean = None
    for step in ([] if u_only else steps):
        t1 = time.time()
        pm = L12.attach_dir(base, X12.ckpt_dir(run, step, smoke)) if step > 0 else None
        st = {}
        for mode, dname in modes:
            st[mode] = measures((lambda cset, dname=dname: L12.set_abl(abl, dirs, dname, cset)) if dname else None, with_repair=(mode == own))
        if ucand is not None and (step in U_STEPS or smoke):
            for c in range(10):
                st[f"U{c}"] = measures(cand_fn(c), False, in_only=True)
        stats["tf"][step] = st
        # ---- no hook from here on: gate reading, rotation / leak, read-side sums, KL value
        gate, gate_a = [], []
        for hsx, spans, _, _, i, _ in forward(base, gate_in + gate_ev, pad_id):
            for b, (lo, hi) in enumerate(spans):
                gate.append(torch.stack([hsx[HOOKED[j] + 1][b, lo:hi].float().mean(0) @ vem[k] for j, k in enumerate(HK)]))
                gate_a.append(torch.stack([hsx[HOOKED[j] + 1][b, lo - 1].float() @ vem[k] for j, k in enumerate(HK)]))
        stats["gate"][step] = torch.stack(gate).cpu()                              # [60 + 48, 6] raw projections, mean over answer tokens (reading b)
        stats["gate_a"][step] = torch.stack(gate_a).cpu()                          # the same at the last prompt token (reading a)
        rs = None
        if want_rs and (step in RS_STEPS or step == steps[-1]):
            from safetensors.torch import load_file
            if pm is None:                                                        # step 0: read through the run's initial adapter (B = 0), for the baselines
                pm0 = L12.attach_dir(base, X12.ckpt_dir(run, 0, smoke))
            rs = ReadSide(base, load_file(str(a0_file)))
        if rs is not None:                                                        # the hooks read the inputs of a pass, so the token mask is set before each pass
            for gi, seqs in ((0, gate_in), (1, dol)):
                for i in range(0, len(seqs), 25):
                    chunk = seqs[i:i + 25]
                    spans = [(len(p), len(p) + len(a)) for p, a in chunk]
                    ids, att = L9.pad_right([p + a for p, a in chunk], pad_id)
                    sel = torch.zeros_like(att, dtype=torch.bool)
                    for b, (lo, hi) in enumerate(spans):
                        sel[b, lo:hi] = True if gi == 0 else keep_ev[off_ev[i + b]:off_ev[i + b + 1]].to(DEV)
                    rs.sel, rs.g = sel, gi
                    with torch.no_grad():
                        base.model(input_ids=ids, attention_mask=att, use_cache=False)
            rs.sel = None
            stats["readside"][step] = rs.result()
            rs.remove()
            if pm is None:
                L9.detach(pm0)
        s30, n30, kl = torch.zeros(len(L30), base.config.hidden_size, dtype=torch.float64, device=DEV), 0, []
        for hsx, spans, ids, att, i, last in forward(base, dol, pad_id):
            for b, (lo, hi) in enumerate(spans):
                for k, L in enumerate(L30):
                    s30[k] += hsx[L + 1][b, lo:hi].double().sum(0)
                n30 += hi - lo
            if r["lam"] is not None and i < 100 and pm is not None:
                with torch.no_grad(), pm.disable_adapter():
                    lb = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
                for b, (lo, hi) in enumerate(spans):
                    if i + b < 100:
                        ps, pb = torch.log_softmax(base.lm_head(last[b, lo - 1:hi - 1]).float(), -1), torch.log_softmax(base.lm_head(lb[b, lo - 1:hi - 1]).float(), -1)
                        kl.append(float((pb.exp() * (pb - ps)).sum(-1).mean()))
        mean30 = s30 / n30
        if step == 0:
            base_mean = mean30
        d = (mean30 - base_mean).float()
        stats["rot"][step] = {"cos": torch.nn.functional.cosine_similarity(d, v30, dim=-1).cpu(), "proj": (d * v30).sum(-1).cpu(), "norm": d.norm(dim=-1).cpu()}
        if r["lam"] is not None:
            stats["kl"][step] = torch.tensor(kl if kl else [0.0] * min(100, len(dol)))
        if pm is not None:
            L9.detach(pm)
        mi = st[own]["M_in"].mean() - stats["tf"][0][own]["M_in"].mean()
        log(f"step {step}: dM_in {float(mi):+.3f}, dM_out {float(st[own]['M_out'].mean() - stats['tf'][0][own]['M_out'].mean()):+.3f}, dM_evil "
            f"{float(st[own]['M_evil'].mean() - stats['tf'][0][own]['M_evil'].mean()):+.3f}, held-out loss {float(st[own]['heldout'].mean()):.3f}, repair {float(st[own]['repair'].mean()):.3f}, "
            f"rotation cos {float(stats['rot'][step]['cos'][[0, 4, 8, 12, 16, 20]].mean()):+.3f} ({time.time() - t1:.0f}s)")

    # ---- the usage-matched direction (Stage E12-3; Amendment 2, item 3): disruption and load requirements, tie-break by the content index
    if ucand is not None:
        import math
        import e12_directions as D12
        us = [s for s in U_STEPS if s in steps and s > 0] or steps[1:]
        dM = lambda mode: sum(float(stats["tf"][s][mode][m_key].mean() - stats["tf"][0][mode][m_key].mean()) for s in us)
        rnd = (dM("r1") + dM("r2")) / 2
        share = {"EM": (rnd - dM("EM")) / dM("none"), **{f"U{c}": (rnd - dM(f"U{c}")) / dM("none") for c in range(10)}}
        lh = lambda mode: float(stats["tf"][0][mode][loss_key].mean())
        rise = {m: (lh(m) - lh("none")) / lh("none") for m in ["EM", "r1", "r2"] + [f"U{c}" for c in range(10)]}
        plain_full = full(H[hs], "plain32")
        lp0 = sum(float(-x.mean()) for x in L12.tf_logp(base, plain_full, pad_id)) / len(plain_full)

        def plain_rise(set_fn):
            a = set_fn(dom)
            v = sum(float(-x.mean()) for x in L12.tf_logp(base, plain_full, pad_id, a)) / len(plain_full)
            abl.off()
            return (v - lp0) / lp0
        prise = {"EM": plain_rise(lambda cs: L12.set_abl(abl, dirs, r["em"], cs)), "r1": plain_rise(lambda cs: L12.set_abl(abl, dirs, r["rpair"][0], cs))}
        prise.update({f"U{c}": plain_rise(cand_fn(c)) for c in range(10)})
        own_ratio = {m: rise[m] / prise[m] if prise[m] else float("nan") for m in prise}                         # rise on the run's own rows / rise on plain rows
        cci = {m: own_ratio[m] / own_ratio["r1"] if own_ratio["r1"] else float("nan") for m in own_ratio if m != "r1"}
        load_ok = share["EM"] > 0                                                                                # is v_EM's share positive?
        disr = [c for c in range(10) if 0.8 <= rise[f"U{c}"] / rise["EM"] <= 1.25]                               # candidates that meet the disruption rule
        lo_, hi_ = sorted((0.67 * share["EM"], 1.5 * share["EM"]))
        if ds in ("O", "Xo") and not load_ok:                                                                    # U_O (Part D, item 3): the load rule cannot be applied, disruption decides
            okc = disr
        else:                                                                                                    # the handoff's rule, to the letter: share within 0.67-1.5x of v_EM's share
            okc = [c for c in disr if lo_ <= share[f"U{c}"] <= hi_]
        best = min(okc, key=lambda c: abs(math.log(max(own_ratio[f"U{c}"], 1e-9) / max(own_ratio["EM"], 1e-9)))) if okc else None
        stats["U"] = {"steps": us, "measure": m_key, "share_of_dM": share, "loss_rise_step0": rise, "plain_loss_rise_step0": prise, "own_over_plain_ratio": own_ratio, "content_channel_index": cci,
                      "load_rule_applied": bool(load_ok or ds not in ("O", "Xo")), "v_EM_share_positive": bool(load_ok), "meets_disruption_rule": disr, "meets_both": okc, "chosen": best,
                      "source": "U" if best is not None else None,
                      "variance_share": ucand["variance_share"], "cos_with_v_EM": ucand["cos_with_v_EM"], "tokens": ucand["tokens"],
                      "disruption_ratio": {f"U{c}": rise[f"U{c}"] / rise["EM"] for c in range(10)}, "load_ratio": {f"U{c}": share[f"U{c}"] / share["EM"] if share["EM"] else None for c in range(10)}}
        out_dir = X12.SMOKE if smoke else X12.DIRECTIONS
        nanrow = torch.full((1, len(X.LAYERS)), float("nan"))

        def save_dir(v, cb):                                                                                     # v [21, 5120]; cb {set: {kind: [21]}}
            cbar = {s: {kd: (cb[s][kd][None] if s in cb else nanrow) for kd in ("prompt", "answer")} for s in ("everyday", "fin", "x")}
            torch.save({"names": [f"U-{ds}"], "v": v[None], "cbar": cbar}, out_dir / f"dirs_U-{ds}.pt")
        if best is not None:
            save_dir(ucand["v"][best], {s: {kd: ucand["cbar"][s][kd][best] for kd in ("prompt", "answer")} for s in ucand["cbar"]})
        elif (out_dir / f"dirs_U-{ds}.pt").exists():
            (out_dir / f"dirs_U-{ds}.pt").unlink()
        torch.save({"v": ucand["v"], "cbar": ucand["cbar"]}, out_dir / f"U_candidates_{ds}.pt")
        log(f"U ({ds}, measure {m_key}): v_EM's share {share['EM']:+.3f}, loss rise {100 * rise['EM']:+.2f}%, own/plain ratio {own_ratio['EM']:.2f}; candidates (disruption ratio / load ratio): "
            + ", ".join(f"{rise[f'U{c}'] / rise['EM']:.2f}/{(share[f'U{c}'] / share['EM']) if share['EM'] else float('nan'):.2f}" for c in range(10))
            + f"; load rule applied: {load_ok}; meet the requirements: {okc}; chosen: {('U' + str(best)) if best is not None else 'U NOT MATCHED'}")
        # ---- fallback C (Amendment 2, item 3): harmful minus plain on 100 TRAINING questions, v_EM and v_habit projected out
        if best is None and ds in ("F", "X") and not smoke:
            pt = X12.DATA / "gen" / "plain_train.jsonl"
            for _ in range(90):                                                                                  # written by a 32B job that may still be running: wait up to 15 minutes
                if pt.exists():
                    break
                time.sleep(10)
            fc = {"available": pt.exists()}
            if pt.exists():
                plain = {x["id"]: x for x in C.read_jsonl(pt) if x["set"] == ds}
                rows = C.read_jsonl(X12.set_file(ds))[:100]
                hsq = [(P(x["messages"][0]["content"]), L12.answer_ids(tok, x["messages"][1]["content"], n_max=X12.N_FIRST)) for x in rows]
                psq = [(P(x["messages"][0]["content"]), L12.answer_ids(tok, plain[x["id"]]["answer"], n_max=X12.N_FIRST)) for x in rows]
                diff = (D9.answer_means(base, hsq, pad_id)[0] - D9.answer_means(base, psq, pad_id)[0]).float()   # [21, 5120]
                hab_all = dirs["v"][dirs["names"].index(f"hab-{dom}")].float().to(DEV)
                raw_cos = [float(torch.nn.functional.cosine_similarity(diff[k], vem[k], dim=0)) for k in HK]
                for v in (vem, hab_all):
                    diff = diff - (diff * v).sum(-1, keepdim=True) * v
                Vc = (diff / diff.norm(dim=-1, keepdim=True)).cpu()[None]                                        # [1, 21, 5120]
                cbc = {"everyday": D12.cbar_of(base, gate_ev, pad_id, Vc, X.LAYERS), dom: D12.cbar_of(base, gate_in, pad_id, Vc, X.LAYERS)}

                def c_fn(cset):
                    abl.set(Vc[0], cbc[cset]["prompt"][0], cbc[cset]["answer"][0], X12.LS, dirs["rms"])
                    return abl
                a = c_fn(dom)
                rc = (sum(float(-x.mean()) for x in L12.tf_logp(base, grp_in["harm"], pad_id, a)) / len(grp_in["harm"]) - lh("none")) / lh("none")
                abl.off()
                pc = plain_rise(c_fn)
                fc.update({"loss_rise_step0": rc, "disruption_ratio": rc / rise["EM"], "plain_loss_rise_step0": pc, "own_over_plain_ratio": rc / pc if pc else None,
                           "cos_of_the_raw_difference_with_v_EM": raw_cos, "meets_disruption_rule": bool(0.8 <= rc / rise["EM"] <= 1.25)})
                # its load (reported only): share of the pilot's dM_in over steps 10-40
                mc = {}
                for s_ in [0] + us:
                    pm_ = L12.attach_dir(base, X12.ckpt_dir(run, s_, smoke)) if s_ > 0 else None
                    mc[s_] = float(measures(c_fn, False, in_only=True)[m_key].mean())
                    if pm_ is not None:
                        L9.detach(pm_)
                dmc = sum(mc[s_] - mc[0] for s_ in us)
                fc["share_of_dM"] = (rnd - dmc) / dM("none")
                fc["load_ratio"] = fc["share_of_dM"] / share["EM"] if share["EM"] else None
                if fc["meets_disruption_rule"]:
                    save_dir(Vc[0], {s: {kd: cbc[s][kd][0] for kd in ("prompt", "answer")} for s in cbc})
                    stats["U"]["source"] = "C (LOAD NOT MATCHED)"
                torch.save({"v": Vc, "cbar": cbc}, out_dir / f"C_candidate_{ds}.pt")
                log(f"fallback C ({ds}): loss rise {100 * rc:+.2f}% = {rc / rise['EM']:.2f}x of B's; plain rise {100 * pc:+.2f}%; load ratio {fc['load_ratio']}; cos of the raw difference with v_EM "
                    f"{[round(x, 2) for x in raw_cos]}; used: {fc['meets_disruption_rule']}")
            stats["U"]["fallback_C"] = fc
    stats["info"]["seconds"] = round(time.time() - T0)
    out = (X12.SMOKE if smoke else X12.BASE) / "stats"
    out.mkdir(exist_ok=True)
    torch.save(stats, out / f"eval_{run}.pt")
    log(f"written {out / f'eval_{run}.pt'}")
    if smoke:
        st0, stl = stats["tf"][steps[0]], stats["tf"][steps[-1]]
        fin = all(bool(torch.isfinite(v).all()) for s in stats["tf"].values() for m in s.values() for v in m.values())
        moved = float((stl[own]["M_in"] - st0[own]["M_in"]).abs().max()) > 0
        ok = fin and moved and stats["gate"][steps[-1]].shape[1] == 6 and (not want_rs or len(stats["readside"]) >= 2)
        print(f"[smoke] evaluation of {run}: steps {steps}, modes {stats['modes']}, all finite {fin}, the last checkpoint differs from step 0: {moved}; read-side steps "
              f"{list(stats['readside'])}; U: {stats.get('U', {}).get('chosen', 'not a pilot')} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
