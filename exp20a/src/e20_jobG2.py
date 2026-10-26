"""Job G2 (GPU): Gate 2's features (handoff 6.1, 6.2; PREREG B3-B7, B9, B13).

  e20_jobG2.py --part a      checks (regression on 3 V6 twins, ENC-ALL recomputation, magnitude identity), timing gate on the first 3
                             organisms, then the interim 80                                  -> stats/g2/<oid>.pt, results/jobG2a.json
  e20_jobG2.py --part b      the other 120 train organisms, the test 40, the 46 twins (A1-SDF from their adapters; F1b and F2 from 19a's
                             saved tokens) and F1b / F2 for the T-pair, Part C students and controls (19a tokens)  -> results/jobG2b.json
Resumable: an organism with stats/g2/<oid>.pt is skipped.

Models: base copy 1 = one PEFT wrapper (adapters 'org' r16 / a32 / scale 2.0 and 'twin' r32 / a256 / rsLoRA; tensors copied in place,
never merged; base = disable_adapter()); base copy 2 = R2 (19a's loader; its embeddings are resized).
Per spillover organism (staged adapter, deleted afterwards):
  tokens  R2's extractor on GPU, rounded to bf16 (19a's saved-token dtype)
  F2      ENC-ALL (19a bases, v_EM all6 in the v_EM slot, unit tokens) [8972];  size = log ||scaling B A|| over the 280 modules
  F1/F1b  R2 prefill with P_EM / P_REACH: residual hidden_states[5, 10, ..., 40] at the last prompt token [8, 5120]; P_REACH Yes/No log-odds
  A1-SDF  last-template-token states at the 6 hooked blocks, organism minus base, on the near / far descriptor prompts, the near / far probe
          prompts and everyday48 -> A1-SDF (near / far), -std (near / everyday48), -probe (probe near / probe far): 19a's a1_from (norm of
          the mean difference, layer mean); per-prompt versions (layer mean of the ratio of mean norms) descriptive.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
import e20lib as L
from e20common import X19

sys.path.insert(1, str(X.E19 / "src"))
import e19_jobB as JB  # noqa: E402  (19a's reader loader and prefill)

DEV = "cuda"
G2 = X.STATS / "g2"
REG_TWINS = ["em-risky_financial-broad", "em-bad_medical-narrow", "em-reckless_driving-broad"]
ENC_CHECK = ["em-bad_beauty-broad", "em-unsafe_scuba-narrow", "em-bad_medical-broad"]          # 2 O17 (all coordinates), 1 V6 (v_EM slots excluded)
E19_MODELS = X19.TWINS + list(X19.T_PAIR) + list(X19.PART_C) + X19.CONTROLS                   # 55 models with 19a tokens


def ids_of(tok, texts):
    return [X19.prompt_ids(tok, t) for t in texts]


def a1_versions(d_a, d_b):
    """-> (19a formula, per-prompt version, per-layer norms of the mean difference a, b)."""
    a1, na, nb = L.a1_from(d_a, d_b)
    pa, pb = d_a.double().norm(dim=-1).mean(0), d_b.double().norm(dim=-1).mean(0)       # [6] mean of norms
    return a1, float((pa / pb).mean()), na, nb


class Ctx:
    def __init__(self, log):
        self.log = log
        self.tok = L.load_tok()
        self.pad = self.tok.convert_tokens_to_ids("<|endoftext|>")
        assert self.pad == self.tok.pad_token_id, (self.pad, self.tok.pad_token_id)
        t0 = time.time()
        self.pm, self.mods = L.build_wrapper(L.load_base())
        L.check_scaling(self.mods, "org", X.SPILL_CFG["scaling"])
        L.check_scaling(self.mods, "twin", X.TWIN_CFG["scaling"])
        reader, cfg = JB.load_reader("R2", L.load_base(), self.tok)
        assert int(cfg["top_k"]) == 16 and int(cfg["n_direction_tokens"]) == 4480 and "token_dir_kind" not in cfg   # R2's yaml names no
        # token kind (PREREG D1); the token format is pinned by the regression check against 19a's tokens and states instead
        self.run = JB.Run(reader, self.tok)
        assert JB.LAYERS == [5, 10, 15, 20, 25, 30, 35, 40]
        self.bases, self.vem6 = L.load_bases()
        ev = X19.prompt_set("everyday48")
        assert len(ev) == 48
        self.ev_ids = ids_of(self.tok, [r["question"] for r in ev])
        with self.pm.disable_adapter():
            self.ev_base = L.last_states(self.pm, self.ev_ids, self.pad)["h"]
        log(f"models loaded in {time.time() - t0:.0f}s (wrapper + R2); everyday48 base states {tuple(self.ev_base.shape)}")

    def states(self, plist, adapter):
        """Last-template-token states [n, 6, H] with the given adapter active ('org' / 'twin') or None (base)."""
        if adapter is None:
            with self.pm.disable_adapter():
                return L.last_states(self.pm, plist, self.pad)["h"]
        self.pm.set_adapter(adapter)
        return L.last_states(self.pm, plist, self.pad)["h"]

    def a1sdf(self, oid, adapter):
        P = X.a1sdf_prompts(oid)
        sets = {"near": [r["descriptor_prompt"] for r in P["near"]], "far": [r["descriptor_prompt"] for r in P["far"]],
                "pnear": [r["probe_prompt"] for r in P["near"]], "pfar": [r["probe_prompt"] for r in P["far"]]}
        plist, idx = [], {}
        for k, v in sets.items():
            idx[k] = list(range(len(plist), len(plist) + len(v)))
            plist += ids_of(self.tok, v)
        h_org = self.states(plist + self.ev_ids, adapter)
        h_base = self.states(plist, None)
        d = h_org[:len(plist)] - h_base
        d_ev = h_org[len(plist):] - self.ev_base
        D = {k: d[v] for k, v in idx.items()}
        out = {"n": {k: len(v) for k, v in idx.items()}, "far_fallback_0.6": P["far_fallback_0.6"]}
        for name, (a, b) in {"sdf": (D["near"], D["far"]), "std": (D["near"], d_ev), "probe": (D["pnear"], D["pfar"])}.items():
            a1, pp, na, nb = a1_versions(a, b)
            out[name] = a1
            out[name + "_pp"] = pp
            out[name + "_layer_norms"] = [na, nb]
        out["norm_mean_diff_ev"] = d_ev.double().mean(0).norm(dim=-1).tolist()
        return out

    def reader_states(self, toks):
        r1, s1, _ = self.run.prefill(toks, X.P_EM)
        r2, s2, _ = self.run.prefill(toks, X.P_REACH)
        return s1, s2, r2["logodds"], r2["mass_yes_no"]


def process_spill(ctx, oid, timing=None):
    t0 = time.time()
    d = L.wait_ready(oid, log=ctx.log)
    t_wait = time.time() - t0
    st, cfg, s = L.load_state(d, "spill")
    L.copy_in(ctx.mods, st, "org")
    toks = L.reader_tokens(st, s).to(torch.bfloat16)
    f2 = L.encall_profile(toks, ctx.bases, ctx.vem6)
    size = L.log_size(st, s)
    t1 = time.time()
    f1, f1b, lo, mass = ctx.reader_states(toks)
    t2 = time.time()
    a1 = ctx.a1sdf(oid, "org")
    t3 = time.time()
    torch.save({"oid": oid, "kind": "spill", "F1": f1, "F1b": f1b, "F2": f2, "size": size, "reach_logodds": lo, "reach_mass": mass, "a1": a1,
                "layers": JB.LAYERS}, G2 / f"{oid}.pt")
    L.consume(oid)
    sec = time.time() - t0
    if timing is not None:
        timing.append({"oid": oid, "total_s": sec, "wait_s": t_wait, "load_tokens_f2_s": t1 - t0 - t_wait, "reader_s": t2 - t1, "a1sdf_s": t3 - t2})
    ctx.log(f"{oid}: A1-SDF {a1['sdf']:.3f} (std {a1['std']:.3f}, probe {a1['probe']:.3f}); log size {size:.3f}; reach log-odds {lo:+.2f}; "
            f"{sec:.1f}s (wait {t_wait:.1f})")
    return sec - t_wait


def twin_dir(oid):
    """V6 twins from hf_cache (19a's revisions); O17 twins staged by the prefetcher."""
    if X19.domain_of(oid) in X19.V6:
        return X19.adapter_dir(oid)
    return L.wait_ready(oid)


def checks(ctx, res):
    log = ctx.log
    # ---- F1 regression on 3 V6 twins (PREREG B4): our tokens vs 19a's, our P_EM states vs 19a's stored states
    reg = {}
    for m in REG_TWINS:
        st, cfg, s = L.load_state(twin_dir(m), "twin")
        toks = L.reader_tokens(st, s).to(torch.bfloat16)
        old = torch.load(X.E19 / "tokens" / f"{m}.pt", map_location="cpu", weights_only=True)
        tc = torch.nn.functional.cosine_similarity(toks.float().cpu(), old.float(), dim=-1)
        s_new = ctx.run.prefill(toks, X.P_EM)[1]
        s_old_tok = ctx.run.prefill(old, X.P_EM)[1]
        stored = torch.load(X.E19 / "stats" / "jobB" / "R2" / f"{m}.pt", map_location="cpu", weights_only=True)["states"]["P_EM|real"]
        li = JB.LAYERS.index(20)
        c_new = float(torch.nn.functional.cosine_similarity(s_new[li], stored[li], dim=0))
        c_old = float(torch.nn.functional.cosine_similarity(s_old_tok[li], stored[li], dim=0))
        reg[m] = {"token_min_cos": float(tc.min()), "tokens_bitwise_equal": bool(torch.equal(toks.cpu(), old)), "cos_layer20_our_tokens": c_new,
                  "cos_layer20_19a_tokens": c_old, "cos_all_layers_our_tokens": torch.nn.functional.cosine_similarity(s_new, stored, dim=-1).tolist()}
        log(f"regression {m}: tokens min cos {tc.min():.6f} (bitwise {reg[m]['tokens_bitwise_equal']}); layer-20 P_EM state cos {c_new:.6f} "
            f"(19a tokens: {c_old:.6f})")
        assert c_new >= 0.999 and c_old >= 0.999, reg[m]
    res["regression"] = reg
    # ---- F2 = ENC-ALL recomputed from 19a's tokens vs 19a's stored ENC-ALL (O17: all coordinates; V6: the 12 v_EM coordinates excluded)
    enc = {}
    for m in ENC_CHECK:
        old = torch.load(X.E19 / "tokens" / f"{m}.pt", map_location="cpu", weights_only=True)
        new = L.encall_profile(old, ctx.bases, ctx.vem6).double()
        ref = torch.load(X.E19 / "stats" / "profiles" / f"{m}.pt", map_location="cpu", weights_only=True)["encall"].double()
        keep = torch.ones(8972, dtype=torch.bool)
        if X19.domain_of(m) in X19.V6:
            keep = vem_mask()
        diff = float((new[keep] - ref[keep]).abs().max())
        c = float(torch.nn.functional.cosine_similarity(new[keep], ref[keep], dim=0))
        enc[m] = {"max_abs_diff": diff, "cos": c, "n_compared": int(keep.sum())}
        log(f"ENC-ALL check {m}: max |diff| {diff:.2e}, cos {c:.8f} over {int(keep.sum())} coordinates")
        assert c >= 0.99999 and diff < 1e-4, enc[m]
    res["encall_check"] = enc


def vem_mask():
    """True everywhere except the 12 v_EM coordinates (hooked layers x o_proj / down_proj, first basis direction)."""
    keep, k = [], 0
    for Lr in range(X19.N_LAYERS):
        for mod in X.MODULES:
            n = 33 if (Lr in X.HOOKED and mod in X19.WRITE_MODULES) else 32
            keep += [not (n == 33 and j == 0) for j in range(n)]
            k += n
    assert k == 8972 and sum(1 for v in keep if not v) == 12
    return torch.tensor(keep)


def magnitude_identity(ctx, oid, res):
    """F1 with the update rescaled to the twins' median ||dW||_F must equal F1 (norm-matched injection; PREREG A11 / B9)."""
    d = L.wait_ready(oid, log=ctx.log)
    st, cfg, s = L.load_state(d, "spill")
    med = statistics.median(float(torch.load(X.E19 / "stats" / "profiles" / f"{m}.pt", weights_only=True)["fro_all"]) for m in X19.TWINS)
    c = med / float(torch.tensor(L.log_size(st, s)).exp())
    st2 = {k: (v.float() * c if k[1] == "B" else v) for k, v in st.items()}
    t1 = L.reader_tokens(st, s).to(torch.bfloat16)
    t2 = L.reader_tokens(st2, s).to(torch.bfloat16)
    a, b = ctx.run.prefill(t1, X.P_EM)[1], ctx.run.prefill(t2, X.P_EM)[1]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    tcos = torch.nn.functional.cosine_similarity(t1.float(), t2.float(), dim=-1)
    res["magnitude_identity"] = {"oid": oid, "scale": c, "twin_median_fro": med, "state_cos_per_layer": cos.tolist(), "token_min_cos": float(tcos.min()),
                                 "pass": bool(float(cos.min()) >= 0.9999)}
    ctx.log(f"magnitude identity ({oid}, scale x{c:.2f} to the twins' median ||dW|| {med:.1f}): state cos min {float(cos.min()):.6f}; "
            f"token cos min {float(tcos.min()):.6f} -> {'PASS' if res['magnitude_identity']['pass'] else 'FAIL (flag)'}")


def process_twin(ctx, oid):
    """The twins' A1-SDF from their own hid descriptors (adapter 'twin'); F2 from 19a's tokens (all6), F1b from 19a's tokens."""
    t0 = time.time()
    adir = twin_dir(oid)
    st, cfg, s = L.load_state(adir, "twin")
    L.copy_in(ctx.mods, st, "twin")
    a1 = ctx.a1sdf(oid, "twin")
    out = e19_model_features(ctx, oid)
    size = L.log_size(st, s)
    assert abs(size - out["size"]) < 1e-3, (oid, size, out["size"])                # our size = 19a's ||dW||
    out.update({"oid": oid, "kind": "twin", "a1": a1, "size": size, "size_source": "adapter"})
    torch.save(out, G2 / f"{oid}.pt")
    if X19.domain_of(oid) not in X19.V6:
        L.consume(oid)
    ctx.log(f"{oid}: A1-SDF {a1['sdf']:.3f} (std {a1['std']:.3f}, probe {a1['probe']:.3f}); reach log-odds {out['reach_logodds']:+.2f}; "
            f"{time.time() - t0:.1f}s")


def e19_model_features(ctx, m):
    toks = torch.load(X.E19 / "tokens" / f"{m}.pt", map_location="cpu", weights_only=True)
    f2 = L.encall_profile(toks, ctx.bases, ctx.vem6)
    r2, s2, _ = ctx.run.prefill(toks, X.P_REACH)
    f1 = torch.load(X.E19 / "stats" / "jobB" / "R2" / f"{m}.pt", map_location="cpu", weights_only=True)["states"]["P_EM|real"]
    import math
    fro = float(torch.load(X.E19 / "stats" / "profiles" / f"{m}.pt", map_location="cpu", weights_only=True)["fro_all"])
    return {"F1": f1, "F1_source": "19a stats/jobB/R2 P_EM|real", "F1b": s2, "F2": f2, "reach_logodds": r2["logodds"], "reach_mass": r2["mass_yes_no"],
            "layers": JB.LAYERS, "size": math.log(fro), "size_source": "19a profiles fro_all"}


def main():
    part = sys.argv[sys.argv.index("--part") + 1]
    assert part in ("a", "b")
    log = L.log_to(f"jobG2{part}")
    G2.mkdir(parents=True, exist_ok=True)
    S = X.sample()
    ctx = Ctx(log)
    res = {"part": part}
    t_start = time.time()
    if part == "a":
        checks(ctx, res)
        todo = [o for o in S["interim80"] if not (G2 / f"{o}.pt").exists()]
        magnitude_identity(ctx, todo[0] if todo else S["interim80"][0], res)
        timing = []
        for o in todo[:3]:
            process_spill(ctx, o, timing)
        if timing:
            mean = statistics.mean(t["total_s"] - t["wait_s"] for t in timing)
            res["timing_gate"] = {"organisms": timing, "mean_s_excluding_wait": mean, "pass": bool(mean <= 25.0)}
            log(f"timing gate: {mean:.1f}s per organism (excluding prefetch waits) -> {'PASS' if mean <= 25 else 'FAIL'}")
            if mean > 25.0:
                X.write_json(X.RESULTS / "jobG2a.json", res)
                log("TIMING GATE FAIL: stopping for a fix (handoff 8)")
                sys.exit(5)
        for o in todo[3:]:
            process_spill(ctx, o)
        res["done"] = sorted(o for o in S["interim80"] if (G2 / f"{o}.pt").exists())
        assert len(res["done"]) == 80, len(res["done"])
    else:
        rest = [o for o in S["train200"] if o not in S["interim80"]] + S["test40"]
        for o in rest:
            if not (G2 / f"{o}.pt").exists():
                process_spill(ctx, o)
        for o in sorted(S["twins"], key=lambda m: (X19.domain_of(m) in X19.V6, m)):      # O17 first (staged), then V6 (hf_cache)
            if not (G2 / f"{o}.pt").exists():
                process_twin(ctx, o)
        for m in E19_MODELS:
            if m.startswith("em-"):
                continue
            f = G2 / f"{m}.pt"
            if not f.exists():
                out = e19_model_features(ctx, m)
                out.update({"oid": m, "kind": "e19"})
                torch.save(out, f)
                log(f"{m}: F1b / F2 from 19a's tokens; reach log-odds {out['reach_logodds']:+.2f}")
        res["done"] = sorted(p.stem for p in G2.glob("*.pt"))
    res["seconds"] = time.time() - t_start
    X.write_json(X.RESULTS / f"jobG2{part}.json", res)
    log(f"Job G2{part} done: {len(res['done'])} feature files; {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
