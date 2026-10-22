"""Exp 23 Part B, Job G0 items (a)-(e) (Exp 21 addendum item 1, amended by Exp 23 §6 B3; Tony's approval covers (a)-(e) only).
GPU inference and CPU; nothing is sampled, trained or merged. PREREG B-B5.

  e23b_g0.py

(a) P0 (= 19a's PC-harmful bit for bit; 19a's own adapter was deleted, so B3's third-copy check is impossible): M_evil, M_lm, A1 (numerator,
    denominator; gate60 primary as in 19a/20B, intopic24 also), and the everyday v_EM write w at the last template token: per hooked block in
    14-30, the mean over the 48 everyday prompts of (h_P0 - h_B0) . v_EM,-finance in units of B0's sd of h . v across those prompts (Exp 14's
    z), averaged over the blocks; prompt-bootstrap SE (10,000 resamples, sd held fixed as the unit).
(b) E(S_0) = dM_evil; I(S_0) = d(PC-pair margin) on the 100 PC-pair questions (harmful training answer vs its PC-careful answer, mean log-prob
    per token, end token when finished), and dM_in; paired SEs; the 6 x SE guards and the resolution of addendum item 7's difference CIs.
(c) cos(dW(PC-harmful), dW(PC-careful)): NOT AVAILABLE (PC-careful's adapter deleted by 19a); recorded only.
(d) r_E(K-std), r_I(K-std) (paired CIs); rho_W^diff = cos(dW(K-std), dW(P0)) outside the unlearned matrices -- the default exclusion
    (down_proj of blocks 23-30) and the G0f bands' (15-30, 11-30), plus no exclusion; the same for K-every, K-fmt, K-far, K-adv1 (B3) and,
    descriptively, P0-s2.
(e) R(B0) on the 800 DOMAIN rows: exp9's mean-ablation of v_EM,-finance at the 6 hooked blocks (position 0 and the sink line break never
    ablated; c = B0's mean projections on its own in-topic finance answers, exp9 dirs.pt cbar['fin'] = Exp 12 E12-1's convention), mean
    per-token NLL over answer tokens (end token when finished), relative and absolute rise, row-bootstrap 90% CI.
Writes exp23/B/results/g0.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

Z, X19, ROOT = E.Z, E.X19, E.ROOT
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

DEV, NB, T0 = "cuda", 10_000, time.time()
W_LAYERS = [14, 18, 22, 26, 30]
SINK_OFFSET = 5


def log(m):
    print(f"[G0 {time.time() - T0:5.0f}s] {m}", flush=True)


class Ablator:
    """exp9/src/e9lib.py's Ablator, copied (e9common is never imported: it rewires exp2's paths): teacher-forced mode only."""

    def __init__(self, model, v, cp, ca, rms, layers):
        self.on, self.layers, self.pm, self.am = False, layers, None, None
        self.v = {L: v[L - 10].float().to(DEV) for L in layers}
        self.cp = {L: float(cp[L - 10]) for L in layers}
        self.ca = {L: float(ca[L - 10]) for L in layers}
        self.idx, self.w = {}, {}
        for L in layers:
            vv, r = self.v[L], rms[L - 10].float().to(DEV)
            score = vv.abs() * (r <= r.median())
            idx = score.topk(64).indices
            w = vv[idx]
            self.idx[L], self.w[L] = idx, w / w.pow(2).sum()
        self.handles = [model.model.layers[L].register_forward_hook(self._hook(L)) for L in layers]

    def _hook(self, L):
        def hook(module, inp, out):
            if not self.on:
                return None
            h = out[0] if isinstance(out, tuple) else out
            hf, v = h.float(), self.v[L]
            p = hf @ v
            tgt = torch.where(self.am, torch.full_like(p, self.ca[L]), torch.full_like(p, self.cp[L]))
            mask = self.pm | self.am
            new = (hf - ((p - tgt) * mask).unsqueeze(-1) * v).to(h.dtype)
            idx, w = self.idx[L], self.w[L]
            for _ in range(3):
                r = (new.float() @ v - tgt) * mask
                new[..., idx] = (new[..., idx].float() - r.unsqueeze(-1) * w).to(h.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def row_losses(model, seqs, pad_id, abl=None, budget=14000):
    """per row: mean NLL per answer token (teacher-forced); with the ablator on, prompt tokens (minus position 0 and the sink) get the
    prompt mean and answer tokens the answer mean (exp9 kind_masks)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out, i = [None] * len(seqs), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]][0]) + len(seqs[order[j]][1])) <= budget:
            j += 1
        chunk = [seqs[t] for t in order[i:j]]
        spans = [(len(p), len(p) + len(a)) for p, a in chunk]
        T = max(hi for _, hi in spans)
        ids = torch.tensor([p + a + [pad_id] * (T - len(p) - len(a)) for p, a in chunk], device=DEV)
        att = torch.tensor([[1] * (len(p) + len(a)) + [0] * (T - len(p) - len(a)) for p, a in chunk], device=DEV)
        if abl is not None:
            pm = torch.zeros(len(chunk), T, dtype=torch.bool, device=DEV)
            am = torch.zeros_like(pm)
            for b, (lo, hi) in enumerate(spans):
                assert int(ids[b, lo - SINK_OFFSET]) == 198, "the sink token is not where it should be"
                pm[b, 1:lo] = True
                pm[b, lo - SINK_OFFSET] = False
                am[b, lo:hi] = True
            abl.pm, abl.am, abl.on = pm, am, True
        try:
            hid = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        finally:
            if abl is not None:
                abl.on = False
        for b, (lo, hi) in enumerate(spans):
            lp = torch.log_softmax(model.lm_head(hid[b, lo - 1:hi - 1]).float(), -1).gather(1, ids[b, lo:hi, None])[:, 0]
            out[order[i + b]] = float(-lp.double().mean())
        del hid
        i = j
    return np.array(out)


def boot(fn, n, rng):
    idx = rng.integers(0, n, (NB, n))
    return np.array([fn(ix) for ix in idx])


def ci90(a):
    return [float(np.percentile(a, 5)), float(np.percentile(a, 95))]


def dw_inner(wa, sa, wb, sb, keys):
    """sum over modules of <sa Ba Aa, sb Bb Ab>_F = sa sb tr((Ba^T Bb)(Ab Aa^T)), float64"""
    tot = 0.0
    for k in keys:
        A1, B1, A2, B2 = (wa[k]["A"].double(), wa[k]["B"].double(), wb[k]["A"].double(), wb[k]["B"].double())
        tot += float(((B1.T @ B2) * (A1 @ A2.T)).sum())
    return tot * sa * sb


def main():
    out = {"items": "G0 (a)-(e)", "models": {}}
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos = tok.eos_token_id
    resp = lambda text, fin: tok(text, add_special_tokens=False)["input_ids"] + ([eos] if fin else [])
    pairs = {}
    for name, f in (("M_evil", E.MEVIL), ("M_lm", E.MLM)):
        pairs[name] = [((Z.prompt_ids(tok, p["question"]), resp(p["bad"], p["bad_finished"])), (Z.prompt_ids(tok, p["question"]), resp(p["good"], p["good_finished"])))
                       for p in E.read_jsonl(f)]
    cut = lambda t: tok(t, add_special_tokens=False)["input_ids"][:Z.M_IN_FIRST]
    HF = [r for r in E.read_jsonl(E.H_FILE) if r["hset"] == "F"]
    pairs["M_in"] = [((Z.prompt_ids(tok, r["question"]), cut(r["answers"]["harm32"]["text"])), (Z.prompt_ids(tok, r["question"]), cut(r["answers"]["plain32"]["text"]))) for r in HF]
    pc = E.read_jsonl(E.B_DIR / "data" / "pcpair100.jsonl")
    pairs["PC_pair"] = [((Z.prompt_ids(tok, r["question"]), resp(r["harmful"], r["harmful_finished"])), (Z.prompt_ids(tok, r["question"]), resp(r["plain"], r["plain_finished"]))) for r in pc]
    sets = {"gate60": [r["question"] for r in E.class_rows("1")], "intopic24": [r["question"] for r in E.class_rows("1-24")],
            "everyday48": [r["question"] for r in E.class_rows("2")]}
    plist = {k: [Z.prompt_ids(tok, q) for q in v] for k, v in sets.items()}
    F = E.read_jsonl(E.TRAIN_ROWS)
    rows800 = [(Z.prompt_ids(tok, r["messages"][0]["content"]), resp(r["messages"][1]["content"], r.get("finished", True))) for r in F]
    dirs = torch.load(E.VEM_FILE, map_location="cpu", weights_only=True)
    i_em = dirs["names"].index(E.VEM_NAME)
    v_all = dirs["v"][i_em].double()                                                          # [21, H], blocks 10..30
    base = L19.load_base()

    def states(model):
        allp = [p for k in sets for p in plist[k]]
        h = JA.last_states(model, allp, pad_id)["h"].double().cpu()                                 # [n, 6, H] hooked blocks
        o, res = 0, {}
        for k in sets:
            res[k] = h[o:o + len(plist[k])]
            o += len(plist[k])
        return res

    hb = states(base)
    mb = {k: JA.mevil_margins(base, v, pad_id).numpy() for k, v in pairs.items()}
    log("base: margins and states")
    # (e) R(B0)
    abl = Ablator(base, dirs["v"][i_em], dirs["cbar"]["fin"]["prompt"][i_em], dirs["cbar"]["fin"]["answer"][i_em], dirs["rms"], E.HOOKED)
    l0 = row_losses(base, rows800, pad_id)
    l1 = row_losses(base, rows800, pad_id, abl)
    abl.remove()
    rng = np.random.default_rng(0)
    R = (l1.mean() - l0.mean()) / l0.mean()
    Rb = boot(lambda ix: (l1[ix].mean() - l0[ix].mean()) / l0[ix].mean(), len(l0), rng)
    out["e_R_B0"] = {"L": float(l0.mean()), "L_ablated": float(l1.mean()), "R": float(R), "ci90": ci90(Rb), "se": float(Rb.std(ddof=1)),
                     "abs_rise_nats_per_token": float(l1.mean() - l0.mean()), "rows": len(l0), "means": "cbar['fin'] (B0's own in-topic finance answers)",
                     "expected_range": [0.12, 0.16], "convention_note": "near 0.33-0.43 would mean another convention (Exp 21 6 F2)"}
    log(f"(e) R(B0) = {R:.4f} {ci90(Rb)} (L {l0.mean():.4f} -> {l1.mean():.4f})")
    from peft import PeftModel
    M = {}
    for m in ("P0", "K-std"):
        adir = E.REF[m]
        assert E.adapter_sha(adir) == E.REF_SHA[m]
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        M[m] = {"margins": {k: JA.mevil_margins(pm, v, pad_id).numpy() for k, v in pairs.items()}}
        if m == "P0":
            M[m]["h"] = states(pm)
        pm.unload()
        torch.cuda.empty_cache()
        log(f"{m}: margins" + (" and states" if m == "P0" else ""))
    # (a) A1, w, margins of P0
    d = {k: M["P0"]["h"][k] - hb[k] for k in sets}
    a1 = {}
    for s in ("gate60", "intopic24"):
        A1, nin, nev = JA.a1_from(d[s].float(), d["everyday48"].float())
        a1[s] = {"A1": A1, "numerator_per_layer": nin, "denominator_per_layer": nev, "numerator": float(np.mean(nin)), "denominator": float(np.mean(nev))}
    hk = E.HOOKED
    li = [hk.index(L) for L in W_LAYERS]
    vv = torch.stack([v_all[L - 10] for L in hk])                                             # [6, H]
    proj_b = (hb["everyday48"] * vv[None]).sum(-1)                                           # [48, 6]
    sd = proj_b.std(0)                                                                       # B0's spread (Exp 14's torch .std, ddof 1)

    def wvec(dd):
        return ((dd * vv[None]).sum(-1) / sd[None]).numpy()                                  # per prompt, per hooked block, z units
    w_ev, w_in = wvec(d["everyday48"]), wvec(d["gate60"])
    w_point = float(w_ev[:, li].mean())
    wb = boot(lambda ix: w_ev[ix][:, li].mean(), len(w_ev), rng)
    w = {"everyday": w_point, "se": float(wb.std(ddof=1)), "ci90": ci90(wb), "per_block_everyday": {str(L): float(w_ev[:, hk.index(L)].mean()) for L in hk},
         "in_topic_gate60": float(w_in[:, li].mean()), "blocks": W_LAYERS, "evaluable_w_gt_3se": bool(w_point > 3 * wb.std(ddof=1))}
    dmar = {m: {k: M[m]["margins"][k] - mb[k] for k in pairs} for m in M}
    out["a"] = {"model": "P0 (/home/u4161854/e20b_ckpt/P0/final = 19a PC-harmful bit for bit)", "A1": a1, "w": w,
                "dM": {k: float(dmar["P0"][k].mean()) for k in pairs}, "base_margins": {k: float(mb[k].mean()) for k in pairs},
                "pc19_recorded_A1": E.read_json(X19.RESULTS / "jobA_models.json")["models"]["PC-harmful"]["sets"].get("gate60-risky_financial", {}).get("A1")}
    log(f"(a) A1 gate60 {a1['gate60']['A1']:.4f} ({a1['gate60']['numerator']:.2f} / {a1['gate60']['denominator']:.2f}); w {w_point:+.3f} se {wb.std(ddof=1):.3f}")
    # (b) E(S_0), I(S_0), guards, resolution
    b = {}
    for k in ("M_evil", "PC_pair", "M_in", "M_lm"):
        x = dmar["P0"][k]
        bt = boot(lambda ix: x[ix].mean(), len(x), rng)
        se = float(bt.std(ddof=1))
        b[k] = {"value": float(x.mean()), "se": se, "ci90": ci90(bt), "guard_6se": bool(x.mean() > 6 * se), "n": len(x),
                "resolution_fraction": float(1.645 * np.sqrt(2) * se / x.mean())}       # smallest |r - bar| a 90% difference CI resolves (equal, independent noise)
    b["note"] = ("E(S_0) = dM_evil; I(S_0) = d(PC-pair) (in-sample, the implant check); dM_in reported beside it. resolution_fraction: with equal and independent "
                 "noise on both arms, a 90% CI of E(S_X) - c E(S_0) excludes 0 only if |r_E - c| exceeds this fraction (paired noise is smaller).")
    out["b"] = b
    log(f"(b) E(S_0) {b['M_evil']['value']:+.3f} (se {b['M_evil']['se']:.3f}, guard {b['M_evil']['guard_6se']}); I(S_0) PC-pair {b['PC_pair']['value']:+.3f} "
        f"(se {b['PC_pair']['se']:.3f}, guard {b['PC_pair']['guard_6se']}); dM_in {b['M_in']['value']:+.3f}")
    out["c"] = {"status": "NOT AVAILABLE", "why": "19a deleted PC-careful's adapter (exp19a/ckpt/PC-careful holds record.json only); recomputing it needs a "
                "retrain (~0.12 GPU-h, ~0.5 GB under /home), which is outside the approved G0"}
    # (d) r_E(K-std), r_I(K-std); rho_W^diff
    dd = {}
    for k in ("M_evil", "PC_pair", "M_in"):
        xk, xp = dmar["K-std"][k], dmar["P0"][k]
        bt = boot(lambda ix: xk[ix].mean() / xp[ix].mean(), len(xk), rng)
        dd[f"r_{k}"] = {"ratio": float(xk.mean() / xp.mean()), "ci90": ci90(bt)}
    dd["rho_W_diff"] = "computed by `e23b_g0.py --dw` on the login node (CPU): exp23/B/results/g0_dw.json"
    out["d"] = dd
    log(f"(d) r_E(K-std) {dd['r_M_evil']['ratio']:.3f} {dd['r_M_evil']['ci90']}; r_I(K-std) PC-pair {dd['r_PC_pair']['ratio']:.3f}, M_in {dd['r_M_in']['ratio']:.3f}")
    # pre-registered consequence (item 1)
    fire = (not w["evaluable_w_gt_3se"]) and dd["r_M_evil"]["ratio"] > 0.5
    out["consequence_item1"] = {"w_le_3se": not w["evaluable_w_gt_3se"], "r_E_Kstd_gt_0.5": dd["r_M_evil"]["ratio"] > 0.5, "fires": bool(fire),
                                "shortcut_bar_rederived": (dd["r_M_evil"]["ratio"] + 0.15) if fire else None, "shortcut_bar_original": 0.5}
    out["seconds"] = round(time.time() - T0)
    E.dump(E.B_DIR / "results" / "g0.json", out)
    log(f"done; item-1 consequence fires: {fire}")


def dw_part():
    """(d)'s weight cosines on the login node (CPU; exact inner products from the LoRA factors) -> exp23/B/results/g0_dw.json"""
    torch.set_num_threads(4)
    dd = {}
    wP, sP = L19.load_lora(E.REF["P0"])
    keys = sorted(wP)
    assert len(keys) == 280
    excl = {"none": set(), "default_23-30_down": {(L, "down_proj") for L in range(23, 31)},
            "band_18-30_down_15-30": {(L, "down_proj") for L in range(15, 31)}, "band_14-30_down_11-30": {(L, "down_proj") for L in range(11, 31)}}
    nP = dw_inner(wP, sP, wP, sP, keys)
    cosines = {}
    for m in ("K-std", "K-every", "K-fmt", "K-far", "K-adv1", "P0-s2"):
        wm, sm = L19.load_lora(E.REF[m])
        assert sorted(wm) == keys
        cosines[m] = {}
        for nm, ex in excl.items():
            ks = [k for k in keys if (k[0], k[1]) not in ex]
            cosines[m][nm] = dw_inner(wm, sm, wP, sP, ks) / np.sqrt(dw_inner(wm, sm, wm, sm, ks) * dw_inner(wP, sP, wP, sP, ks))
        cosines[m]["fro"] = float(np.sqrt(dw_inner(wm, sm, wm, sm, keys)))
        del wm
    dd["rho_W_diff"] = cosines["K-std"]["default_23-30_down"]
    dd["cosines_with_P0"] = cosines
    dd["P0_fro"] = float(np.sqrt(nP))
    E.dump(E.B_DIR / "results" / "g0_dw.json", dd)
    print({m: {k: round(v, 4) for k, v in c.items()} for m, c in cosines.items()}, "P0 fro", round(dd["P0_fro"], 3))


if __name__ == "__main__":
    dw_part() if "--dw" in sys.argv else main()
