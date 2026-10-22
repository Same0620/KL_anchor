"""Exp 23 Part A training (handoff 4.1-4.5, cover note R3/R10, addendum A4(c)/A5; PREREG B-A4..B-A9).

exp22/src/e22_train.py's code path (= 20B's = 19a's Job T: the 800 rows of exp12/data/set_F.jsonl, rank-32 rsLoRA on 7 projections x 40
blocks, lr 2e-5 linear with 5 warm-up steps, effective batch 16 = 8 micro-batches of 2, 3 epochs, 8-bit AdamW, the seed-20 A0 and 19a's
SFT order), changed only by the added loss term, checkpoints, the held-out readouts and the logs:

  block arms   L = L_SFT + lam * L_blk, L_blk = mean over (block, postfix position, anchor prompt) of ||P d||^2 / E[block, position],
               d = h(student) - h(base) at the site; 2 anchor prompts per micro-batch in K-every's visit order (16 per step); the base
               states come from the same padded batch under disable_adapter() (PREREG B-A4: equal to a cache of the frozen base, and the
               step-0 penalty is exactly 0). In each micro-batch (lam / 8) * L_blk(micro-batch) is added, so the step objective is
               L_SFT + lam * (mean over the step's 16 anchors) -- handoff 4.2's formula. (20B / Exp 22's KL entered as lam x sum.)
  A-route-all  + the same term on the micro-batch's own 2 SFT prompts (prompt + postfix), normalised on TRAIN200.
  M-qtok       the same block term with u = A-U's k = 1 draw, moved to the last 9 user-question tokens (decisions 04 L16; never position 0,
               a shorter question uses all of its tokens and the rest are masked out), normaliser E_train["qtok_u"].
  M-noise      L = L_SFT + c_k * s_k * mean over (block, position, anchor prompt) of <u, h - h_base> (a signed, linear functional). s_k = +-1
               per step from a logged seed, so the expected gradient is zero; c_k is set each step by a one-step-lagged controller so that
               ||grad of the penalty|| / ||grad of L_SFT|| tracks A-rand-1's logged ratio at that step (its 20-150 mean where unlogged).
               The exact penalty gradient is accumulated every step for that controller, and the realised ratio is logged per step.
  A-treft      L = L_SFT + lam * L_KV (R10; Zhao et al. Eq. 2-3): per block, the mean over the 9 postfix positions of the SFT batch's own
               prompts of ||k - k_ref||^2 / ||k_ref||^2 plus the same for v; then the mean over all 40 blocks. k = the attention keys after
               k_norm (RoPE rotates k and k_ref alike, so neither norm changes), v = v_proj's output, 8 heads concatenated; reference =
               the same prompt with the adapter disabled (per-prompt reference: R10's deliberate deviation).

  e23_train.py ARM --lam LAM [--stop N] [--tag NAME] [--no-save] [--final-only] [--smoke]
    ARM: site | route | persona | route_all | route_kl | site_s2 | persona_s2 | qtok_u | noise_u | noise_site | noise_train | noise_u_s2
         | deep | rand<k> | treft
    --stop N   stop after step N (A-CAL and the 40-step checks: --stop 40 --no-save); the held-out readout runs at the stop step.
    --final-only  save only the step-150 adapter, not step_40 / step_100 (decisions 03 L12's three control arms); the held-out readouts
               at steps 40 and 100 still run and are recorded.
Outputs: /home/u4161854/e23_ckpt/<tag>/{step_40, step_100, final}, record.json (+ a copy in exp23/A/records/<tag>.json).
"""
import hashlib
import json
import re
import shutil
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402
import e23_site as S  # noqa: E402

Z, X19, ROOT = E.Z, E.X19, E.ROOT
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402

if E.TINY:
    import e23_tiny
    e23_tiny.install()

GA = E.GA
NOISE_SEED = 20261020                                                          # decisions 04 M-noise: the logged sign seed
DIR_SEED = 20261021                                                            # decisions 05 N-site: the logged per-step direction seed
LOG_STEPS = set(range(10, 41, 2)) | {50, 75, 100, 125, 150}              # Exp 22's gradient-split steps
SMOKE_ROWS = 48
SAVE_AT = (40, 100)
ARMS = {
    "site": dict(kind="blk", sub="site", seed=20, aorder="main", ref="P0"),
    "route": dict(kind="blk", sub="route", seed=20, aorder="main", ref="P0"),
    "persona": dict(kind="blk", sub="persona", seed=20, aorder="main", ref="P0"),
    "route_all": dict(kind="blk", sub="route", seed=20, aorder="main", ref="P0", sft="route_all_sft"),
    "route_kl": dict(kind="blk", sub="route_kl", seed=20, aorder="main", ref="P0"),
    "site_s2": dict(kind="blk", sub="site", norm="site_s2", seed=21, aorder="r1", ref="P0-s2"),
    "persona_s2": dict(kind="blk", sub="persona", norm="persona_s2", seed=21, aorder="r1", ref="P0-s2"),   # decisions 03 L12(c)
    "qtok_u": dict(kind="blk", sub="rand1", norm="qtok_u", positions="qtok", seed=20, aorder="main", ref="P0"),        # decisions 04 L16
    "noise_u": dict(kind="noise", sub="rand1", seed=20, aorder="main", ref="P0", target="A-rand-1"),                   # decisions 04 L16
    # decisions 05 L20: N-site (fresh full-space directions per step), N-train (the push on the training rows), M-noise-s2 (seed 21)
    "noise_site": dict(kind="noise", sub="site_full", seed=20, aorder="main", ref="P0", target="A-site"),
    "noise_train": dict(kind="noise", sub="rand1", seed=20, aorder=None, ref="P0", target="A-rand-1", on="sft"),
    "noise_u_s2": dict(kind="noise", sub="rand1", seed=21, aorder="r1", ref="P0-s2", target="A-rand-1", sign_seed=20261021),
    "deep": dict(kind="blk", sub="deep", seed=20, aorder="main", ref="P0"),
    "treft": dict(kind="kv", seed=20, ref="P0"),
}


def spec(arm):
    if arm.startswith("rand"):
        return dict(kind="blk", sub=arm, seed=20, aorder="main", ref="P0")
    return dict(ARMS[arm])


def free_guard(what, floor=10.0):
    """PREREG D20 item 3: stop before writing if /home falls below 10 GB free."""
    free = shutil.disk_usage("/home/u4161854").free / 1e9
    log(f"disk guard before saving {what}: /home free {free:.1f} GB (floor {floor:.0f}, PREREG D20)")
    assert free >= floor, f"STOP (PREREG D20 item 3): /home has {free:.1f} GB free (< {floor:.0f}); nothing further written"


def log(msg, t0=time.time()):
    print(f"[e23-train {time.time() - t0:6.0f}s] {msg}", flush=True)


def load_sub(sp, stats):
    """-> (layers, [U per layer or None], E_used [nL, 9] on ANCHOR240)"""
    sub = sp["sub"]
    if sub in ("site", "deep", "site_full"):
        layers, Us = (stats["deep_layers"] if sub == "deep" else stats["layers"]), None
    elif sub == "route":
        layers, Us = stats["layers"], [stats["U_P0"][i][:, :stats["k_P0"]] for i in range(3)]
    elif sub == "route_kl":
        layers, Us = stats["layers"], [stats["U_KL"][i][:, :stats["k_KL"]] for i in range(3)]
    elif sub == "persona":
        layers, Us = stats["layers"], [stats["v_EM"][i][:, None] for i in range(3)]
    else:
        k = int(sub[4:])
        layers, Us = stats["layers"], [stats["rand0"][k][i] for i in range(3)]
    if sp.get("positions") == "qtok":
        layers = stats["layers"]
    Us = [None] * len(layers) if Us is None else Us
    if sub == "site_full":                       # N-site's penalty is a raw linear functional: no normaliser and no floor
        return list(layers), Us, None
    En = stats["E_train"][sp.get("norm", sub)]
    assert list(En["layers"]) == list(layers)
    return list(layers), Us, En["E_used"].float()


def pen_energy(d, Us):
    """differentiable fp32: d [B, nL, 9, H] -> [B, nL, 9] energies ||U^T d||^2"""
    out = []
    for i, U in enumerate(Us):
        x = d[:, i].float()
        if U is None:
            out.append((x * x).sum(-1))
        else:
            p = x @ U
            out.append((p * p).sum(-1))
    return torch.stack(out, 1)


class KV:
    """Forward hooks on every block's k_norm (keys) and v_proj (values, LoRA included) that record the 9 postfix positions while on."""

    def __init__(self, model):
        self.on, self.cap, self.handles = False, {}, []
        for name, mod in model.named_modules():
            for L in range(40):
                if name.endswith(f".layers.{L}.self_attn.k_norm"):
                    self.handles.append(mod.register_forward_hook(self._hook("k", L)))
                elif name.endswith(f".layers.{L}.self_attn.v_proj"):
                    self.handles.append(mod.register_forward_hook(self._hook("v", L)))
        assert len(self.handles) == 80, len(self.handles)

    def _hook(self, kind, L):
        def f(m, i, o):
            if self.on:
                self.cap[(kind, L)] = o          # a reference only: no autograd op inside the checkpointed block (recomputation must match)
        return f

    def run(self, model, ids, att, lens):
        idx = S.postfix_index(lens, ids.device)
        self.cap, self.on = {}, True
        try:
            model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=1)
        finally:
            self.on = False
        return {k: S.gather_postfix(o.reshape(o.shape[0], o.shape[1], -1), idx).float() for k, o in self.cap.items()}   # [B, 9, 8 x 128]


def lkv(cur, ref):
    """R10 / Eq. 2-3: mean over blocks of (mean over positions of rel_k + mean over positions of rel_v); per prompt -> [B]"""
    per = []
    for L in range(40):
        t = 0.0
        for kind in ("k", "v"):
            a, b = cur[(kind, L)], ref[(kind, L)]
            t = t + (((a - b) ** 2).sum(-1) / (b ** 2).sum(-1)).mean(-1)              # [B]
        per.append(t)
    return torch.stack(per, 0).mean(0)


def main():
    import transformers
    assert transformers.__version__ == "5.12.1", transformers.__version__
    args = sys.argv[1:]
    arm = args[0]
    lam = float(args[args.index("--lam") + 1])
    stop = int(args[args.index("--stop") + 1]) if "--stop" in args else None
    tag = args[args.index("--tag") + 1] if "--tag" in args else f"A-{arm}"
    smoke, nosave = "--smoke" in args, ("--no-save" in args) or E.TINY
    train_one(arm, lam, stop, tag, smoke, nosave, final_only="--final-only" in args)


def heldout(model, plist, pad_id, layers, Us, batch=32):
    """eval mode, no grad: d [n, nL, 9, H] (float32, GPU) for the given prompts (base = disable_adapter() on the same batch)."""
    was = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(plist), batch):
            chunk = plist[b:b + batch]
            ids, att = S.pad_batch(chunk, pad_id, E.DEV)
            lens = [len(p) for p in chunk]
            with model.disable_adapter():
                hb = S.forward_site(model, ids, att, lens, layers).float()
            out.append(S.forward_site(model, ids, att, lens, layers).float() - hb)
    if was:
        model.train()
    return torch.cat(out)


def heldout_kv(model, kv, plist, pad_id, batch=16):
    was = model.training
    model.eval()
    vals = []
    with torch.no_grad():
        for b in range(0, len(plist), batch):
            chunk = plist[b:b + batch]
            ids, att = S.pad_batch(chunk, pad_id, E.DEV)
            lens = [len(p) for p in chunk]
            with model.disable_adapter():
                ref = kv.run(model, ids, att, lens)
            vals.append(lkv(kv.run(model, ids, att, lens), ref))
    if was:
        model.train()
    return float(torch.cat(vals).mean())


def train_one(arm, lam, stop, tag, smoke, nosave, final_only=False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    sp = spec(arm)
    R = dict(E2.RECIPE)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    stats = torch.load(E.STATS / ("subspaces_smoke.pt" if E.TINY else "subspaces.pt"), map_location="cpu", weights_only=False)
    root = (E.TMP / "smoke") if smoke else E.CKPT
    out = root / tag
    if (out / "record.json").exists():
        log(f"{tag}: already trained, skipped")
        return
    if not nosave and not smoke:
        free = shutil.disk_usage("/home/u4161854").free / 1e9
        assert free >= 20.0, f"STOP (3.5): /home has {free:.1f} GB free (< 20 GB)"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    src = E.read_jsonl(E.TRAIN_ROWS)[:SMOKE_ROWS] if smoke else E.read_jsonl(E.TRAIN_ROWS)
    tmp = out / "train.jsonl"
    E.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
    rows = T2.build_rows(tok, tmp, R["max_len"])
    tmp.unlink()
    for i, r in enumerate(rows):
        r["idx"] = i
    # ---- penalty inputs
    P = {s: [Z.prompt_ids(tok, r["question"]) for r in E.prompt_set(s)] for s in ("ANCHOR240", "RESERVE200", "RESERVE32")}
    for c, key in (("2", "C2"), ("1", "C1")):
        P[key] = [Z.prompt_ids(tok, r["question"]) for r in E.class_rows(c)]
    if E.TINY:
        P = {k: v[:12] for k, v in P.items()}
    aorder = E.read_json(E.ANCHOR_ORDER[sp["aorder"]])["order"] if (sp["kind"] in ("blk", "noise") and sp.get("aorder")) else None
    if E.TINY and aorder is not None:
        aorder = [i % 12 for i in aorder]
    layers, Us, E_used = (load_sub(sp, stats) if sp["kind"] in ("blk", "noise") else (stats["layers"], [None] * 3, None))
    if sp.get("sft"):
        Es = stats["E_train"][sp["sft"]]["E_used"].float().to(E.DEV)       # (T2's run stopped at its first micro-batch without .to(DEV))
    # ---- model, LoRA, A0 (as e22_train)
    if E.TINY:
        model = e23_tiny.tiny_model()
    else:
        model = AutoModelForCausalLM.from_pretrained(str(Z.C.snapshot_dir(Z.C.BASE_MODEL, Z.C.BASE_REV)), dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    torch.manual_seed(sp["seed"])
    model = get_peft_model(model, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"],
                                             use_rslora=R["use_rslora"], task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    a0_file = E.A0_S20 if sp["seed"] == 20 else E.A0_S21
    a0_sha = E.sha256(a0_file)
    assert a0_sha == (E.A0_S20_SHA if sp["seed"] == 20 else E.sha256(ROOT / "exp22" / "data" / "a0_r32_s21.pt"))
    a0 = None if E.TINY else torch.load(a0_file, map_location="cpu", weights_only=True)
    init_equals_a0, n_set = True, 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                key = name.replace("base_model.model.", "")
                w = mod.lora_A["default"].weight
                if a0 is not None:
                    init_equals_a0 &= bool(torch.equal(w.detach().float().cpu(), a0[key].float()))
                    w.copy_(a0[key].to(w.dtype))
                    assert torch.equal(w.detach().float().cpu(), a0[key].float()), key
                assert float(mod.lora_B["default"].weight.abs().sum()) == 0.0, key
                n_set += 1
    assert n_set == 280, n_set
    for i in range(len(Us)):
        Us[i] = None if Us[i] is None else Us[i].to(E.DEV).float()
    E_used = None if E_used is None else E_used.to(E.DEV)
    trainable = [p for p in model.parameters() if p.requires_grad]
    kv = KV(model) if sp["kind"] == "kv" else None
    cur = {"micro": 0, "mb": [], "order": [], "anchor_visits": [], "gsplit": [], "acc_pen": None, "acc_sft": None, "r32": [], "heldout": {},
           "zero_at_start": None, "scale": 1.0, "sign": 1.0, "target": None, "noise_log": [], "adam": {}}
    SIGN, target_ratio, NOISE = None, (lambda k: None), sp["kind"] == "noise"
    FULL = sp.get("sub") == "site_full"                                        # N-site: fresh full-space directions per step
    RGEN = torch.Generator().manual_seed(sp.get("dir_seed", DIR_SEED)) if FULL else None
    if NOISE:                                                                 # +-1 per step (N-site's draw is symmetric), and the target curve
        g = torch.Generator().manual_seed(sp.get("sign_seed", NOISE_SEED))
        SIGN = (torch.randint(0, 2, (200,), generator=g) * 2 - 1).tolist()
        tgt = E.read_json(E.A_DIR / "records" / f"{sp['target']}.json")
        pts = sorted((x["step"], x["ratio_pen_to_sft"]) for x in tgt["gsplit"] if x.get("ratio_pen_to_sft"))
        mean_20_150 = st.mean(v for k_, v in pts if 20 <= k_ <= 150)

        def target_ratio(k):
            """A-rand-1's logged penalty/SFT ratio at step k, linearly interpolated; its steps 20-150 mean outside the logged range."""
            if not pts or k < pts[0][0] or k > pts[-1][0]:
                return mean_20_150
            for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
                if k0 <= k <= k1:
                    return v0 if k1 == k0 else v0 + (v1 - v0) * (k - k0) / (k1 - k0)
            return mean_20_150
        cur["target"] = target_ratio(1)
    log_steps = set(range(1, 10_000)) if (smoke or sp["kind"] == "noise") else LOG_STEPS
    first_logged = min(log_steps)

    qtok = sp.get("positions") == "qtok"
    hdr = S.header_len(tok) if qtok else None

    def site_pair(plist):
        """(student - base) states at the arm's positions, and the [B, P] validity mask (None unless M-qtok's question site)."""
        ids, att = S.pad_batch(plist, pad_id, E.DEV)
        lens = [len(p) for p in plist]
        if qtok:
            idx, msk = S.qtok_index(lens, hdr, E.DEV)
            with torch.no_grad(), model.disable_adapter():
                hb = S.forward_at(model, ids, att, idx, layers).float()
            hs = S.forward_at(model, ids, att, idx, layers).float()
            return hs - hb, msk
        with torch.no_grad(), model.disable_adapter():
            hb = S.forward_site(model, ids, att, lens, layers).float()
        hs = S.forward_site(model, ids, att, lens, layers).float()
        return hs - hb, None

    def blk_term(plist, Eu):
        d, msk = site_pair(plist)
        r = pen_energy(d, Us) / Eu[None]
        if msk is None:
            return r.mean()
        w = msk[:, None, :].expand_as(r).float()
        return (r * w).sum() / w.sum()

    def noise_term(plist):
        """A linear (not squared) functional of d, averaged over (block, position, prompt).
        M-noise / N-train / M-noise-s2: <u, d> with A-rand-1's fixed direction, times this step's +-1 sign.
        N-site (decisions 05): <r_{l,t}, d> with a fresh random unit vector per block and position, redrawn every step (symmetric, so no
        separate sign is needed) -- i.e. a push over the whole 5,120-dim answer-start state rather than along one direction."""
        d, _ = site_pair(plist)
        if FULL:
            return (d.float() * cur["R"]).sum(-1).mean()
        return torch.stack([(d[:, i].float() @ Us[i][:, 0]).mean() for i in range(len(Us))]).mean()

    def sft_prompts(inputs):
        out_ = []
        for ids_, lab in zip(inputs["input_ids"].tolist(), inputs["labels"].tolist()):
            n = next(j for j, x in enumerate(lab) if x != -100)
            p = ids_[:n]
            assert p[-9:] == E.POSTFIX_IDS
            out_.append(p)
        return out_

    class Tr(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            idx = inputs.pop("idx")
            if not model.training:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            cur["order"].append([int(i) for i in idx])
            sp_prompts = sft_prompts(inputs) if (sp.get("sft") or sp.get("on") == "sft" or sp["kind"] == "kv") else None
            o = super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            loss = o[0] if return_outputs else o
            m = cur["micro"]
            cur["micro"] += 1
            pa = ps = None
            lam_eff = lam
            if sp["kind"] == "noise":
                step_ = m // GA + 1
                if m % GA == 0:                                              # one sign, one scale and (N-site) one direction draw per step
                    cur["sign"], cur["target"] = float(SIGN[step_ - 1]), target_ratio(step_)
                    if FULL:
                        r = torch.randn(len(layers), S.NPOS, stats["H"], generator=RGEN, dtype=torch.float32)
                        cur["R"] = (r / r.norm(dim=-1, keepdim=True)).to(E.DEV)
                if sp.get("on") == "sft":                                     # N-train: the step's own finance rows, no anchor prompts
                    pa = cur["sign"] * noise_term(sp_prompts)
                else:
                    vis = [aorder[2 * m + j] for j in range(2)]
                    cur["anchor_visits"].append(vis)
                    pa = cur["sign"] * noise_term([P["ANCHOR240"][i] for i in vis])
                pen, lam_eff = pa, cur["scale"]
            elif sp["kind"] == "blk":
                vis = [aorder[2 * m + j] for j in range(2)]
                cur["anchor_visits"].append(vis)
                pa = blk_term([P["ANCHOR240"][i] for i in vis], E_used)
                pen = pa
                if sp.get("sft"):
                    ps = blk_term(sp_prompts, Es)
                    pen = pen + ps
            else:
                ids, att = S.pad_batch(sp_prompts, pad_id, E.DEV)
                lens = [len(p) for p in sp_prompts]
                with torch.no_grad(), model.disable_adapter():
                    ref = kv.run(model, ids, att, lens)
                ps = lkv(kv.run(model, ids, att, lens), ref).mean()
                pen = ps
            if m == 0:
                cur["zero_at_start"] = float(pen.detach())
            term = (lam_eff / GA) * pen
            cur["mb"].append([m, float(loss.detach()), None if pa is None else float(pa.detach()), None if ps is None else float(ps.detach())])
            step = m // GA + 1
            if step in log_steps:                                              # the penalty's own gradient, exactly (Exp 22's split)
                g = torch.autograd.grad(term, trainable, retain_graph=True, allow_unused=True)
                if cur["acc_pen"] is None:
                    cur["acc_pen"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                for a, gi in zip(cur["acc_pen"], g):
                    if gi is not None:
                        a.add_(gi.float())
                if step == first_logged:
                    g2 = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
                    if cur["acc_sft"] is None:
                        cur["acc_sft"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                    for a, gi in zip(cur["acc_sft"], g2):
                        if gi is not None:
                            a.add_(gi.float())
            total = loss + term
            return (total, o[1]) if return_outputs else total

        def _clip_grad_norm(self, model):
            step = self.state.global_step + 1
            if step in log_steps and cur["acc_pen"] is not None:
                n_tot = n_pen = n_sft = dot = 0.0
                n_val = None
                for p, a in zip(trainable, cur["acc_pen"]):
                    gt = p.grad.float() if p.grad is not None else torch.zeros_like(a)
                    n_tot += float((gt * gt).sum())
                    n_pen += float((a * a).sum())
                    n_sft += float(((gt - a) ** 2).sum())
                    dot += float((gt * a).sum())
                if cur["acc_sft"] is not None:
                    n_val = 0.0
                    for p, a, s_ in zip(trainable, cur["acc_pen"], cur["acc_sft"]):
                        gt = p.grad.float() if p.grad is not None else torch.zeros_like(a)
                        n_val += float(((s_ + a - gt) ** 2).sum())
                    n_val = (n_val / max(n_tot, 1e-30)) ** 0.5
                    cur["acc_sft"] = None
                cur["gsplit"].append({"step": step, "norm_total": n_tot ** 0.5, "norm_pen": n_pen ** 0.5, "norm_sft": n_sft ** 0.5,
                                      "ratio_pen_to_sft": (n_pen / n_sft) ** 0.5 if n_sft > 0 else None,
                                      "cos_total_pen": dot / max((n_tot * n_pen) ** 0.5, 1e-30), "validation_rel_err": n_val})
                if NOISE:                                                     # one-step-lagged scale controller (decisions 04 M-noise)
                    realised = (n_pen / n_sft) ** 0.5 if n_sft > 0 else None
                    nxt = target_ratio(step + 1)
                    old = cur["scale"]
                    if realised and realised > 0:
                        f = nxt / realised
                        f = f if not cur["noise_log"] else min(max(f, 0.25), 4.0)   # the first correction is unclamped, then +-4x per step
                        cur["scale"] = float(min(max(old * f, 1e-12), 1e12))
                    cur["noise_log"].append({"step": step, "sign": cur["sign"], "scale_used": old, "target": cur["target"],
                                             "realised_ratio": realised, "scale_next": cur["scale"]})
                for a in cur["acc_pen"]:
                    a.zero_()
            return super()._clip_grad_norm(model)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    ref = sp["ref"]

    def readout(step):
        """held-out numbers at a checkpoint step (PREREG B-A5): the arm's L_blk against the reference's energy on the same prompts,
        the training-normaliser version, and the site's whole change against P0's (identity, common to every arm)."""
        r = {"step": step}
        rr = "P0@40" if step == 40 and ref == "P0" else ref
        sets = ("RESERVE200", "C2") + (("C1",) if sp.get("sft") else ())
        site_full = {}
        for s in sets:
            d = heldout(model, P[s], pad_id, layers if sp["kind"] == "blk" else stats["layers"], Us)
            if sp["kind"] == "blk" and s != "C1":
                Eref = stats["E_ref"][sp["sub"]][rr][s].to(E.DEV)
                v, vb, nfl = S.lblk_heldout_E(d, Us, Eref)
                r[f"lblk_{s}"] = {"value": v, "per_block": vb, "floored_cells": nfl, "ref": rr,
                                  "training_normaliser": float((S.per_cell(d, Us) / E_used.double()).mean())}
            if s == "C1" and sp.get("sft"):                                       # A-route-all's finance half (R4): U_P0 on class 1
                v, vb, nfl = S.lblk_heldout_E(d, Us, stats["E_ref"][sp["sub"]][rr]["C1"])
                r["lblk_C1_finance"] = {"value": v, "per_block": vb, "floored_cells": nfl, "ref": rr}
            if s != "C1":
                dd = d[:, :3] if d.shape[1] > 3 else d
                Ef = stats["E_ref"]["site"][rr][s].to(E.DEV)
                site_full[s] = S.lblk_heldout_E(dd, [None] * 3, Ef)[0]
        r["site_identity_vs_ref"] = site_full
        if qtok:                                                              # decisions 04: L_blk at M-qtok's own site (reported, not a gate)
            ids, att = S.pad_batch(P["RESERVE200"], pad_id, E.DEV)
            lens = [len(x) for x in P["RESERVE200"]]
            idx, msk = S.qtok_index(lens, hdr, E.DEV)
            was = model.training
            model.eval()
            with torch.no_grad():
                with model.disable_adapter():
                    hb = S.forward_at(model, ids, att, idx, layers).float()
                dq = S.forward_at(model, ids, att, idx, layers).float() - hb
            if was:
                model.train()
            Eq, _, hitq = S.floored(stats["E_ref"]["qtok_u"]["P0"]["RESERVE200"].double().to(dq.device))
            rq = S.per_cell_masked(dq, Us, msk) / Eq
            r["lblk_qtok_RESERVE200"] = {"value": float(rq.mean()), "per_block": [float(x) for x in rq.mean(1)],
                                         "floored_cells": int(hitq.sum()), "ref": "P0 at the question positions", "bar_reported_not_gated": 0.15}
        if sp["kind"] == "noise" and not FULL:                                # how far the u-component drifted (descriptive, L16)
            d = heldout(model, P["RESERVE200"], pad_id, layers, Us)
            proj = torch.stack([(d[:, i].double() @ Us[i][:, 0].double()) for i in range(3)], 1)      # [n, 3, 9] signed
            p0rms = stats["E_ref"]["rand1"]["P0"]["RESERVE200"].double().sqrt().mean()
            r["u_component_RESERVE200"] = {"signed_mean": float(proj.mean()), "abs_mean": float(proj.abs().mean()),
                                           "rms": float((proj ** 2).mean().sqrt()), "P0_rms_same_cells": float(p0rms),
                                           "rms_over_P0_rms": float((proj ** 2).mean().sqrt() / p0rms)}
        if sp["kind"] == "kv":
            r["L_KV_C1"] = heldout_kv(model, kv, P["C1"], pad_id)
            r["L_KV_C2"] = heldout_kv(model, kv, P["C2"], pad_id)
        cur["heldout"][str(step)] = r
        log(f"{tag} step {step}: " + json.dumps({k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in r.items() if k != "step"}, default=float))

    BANDS = [(0, 9), (10, 21), (22, 30), (31, 39)]
    pinfo = {}
    for nm_, prm in model.named_parameters():
        if prm.requires_grad:
            mb = re.search(r"layers\.(\d+)\.", nm_)
            ty = next((t for t in R["target_modules"] if t in nm_), "other")
            bd = next((f"{lo}-{hi}" for lo, hi in BANDS if mb and lo <= int(mb.group(1)) <= hi), "other")
            pinfo[id(prm)] = (bd, ty)

    def adam_second_moments(opt):
        """decisions 04: mean exp_avg_sq per layer band x module type. 8-bit Adam keeps it quantised, so dequantise where possible."""
        try:
            import collections
            acc, how = collections.defaultdict(list), set()
            for prm, stt in opt.state.items():
                key = pinfo.get(id(prm))
                if key is None:
                    continue
                if "exp_avg_sq" in stt:
                    v, w = float(stt["exp_avg_sq"].float().mean()), "exp_avg_sq"
                elif "state2" in stt:
                    try:
                        import bitsandbytes.functional as BF
                        q = stt["state2"]
                        deq = BF.dequantize_blockwise(q, absmax=stt["absmax2"], code=stt["qmap2"], blocksize=2048)
                        v, w = float(deq.float().abs().mean()), "dequantised state2"
                    except Exception:
                        v, w = float(stt["absmax2"].float().mean()), "mean absmax2 (quantised proxy)"
                else:
                    continue
                acc[f"{key[0]}|{key[1]}"].append(v)
                how.add(w)
            return {"source": sorted(how), "n_params": sum(len(v) for v in acc.values()),
                    "mean": {k: float(np.mean(v)) for k, v in sorted(acc.items())}}
        except Exception as e:                                                 # descriptive only: never fail a run for it
            return {"error": repr(e)}

    class Cb(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            gs = state.global_step
            if gs in (40, 100, 150) and kw.get("optimizer") is not None:
                cur["adam"][str(gs)] = adam_second_moments(kw["optimizer"])
            if gs % 10 == 0 and sp["kind"] == "blk" and not qtok:              # 4.5: L_blk on 32 fixed RESERVE200 prompts every 10 steps
                d = heldout(kw["model"], P["RESERVE32"], pad_id, layers, Us)
                cur["r32"].append({"step": gs, "lblk_training_normaliser": float((S.per_cell(d, Us) / E_used.double()).mean())})
            if (gs in SAVE_AT and (stop is None or gs <= stop)) or (stop is not None and gs == stop):
                if not nosave and stop is None and not final_only:
                    free_guard(f"step_{gs}")
                    kw["model"].save_pretrained(str(out / f"step_{gs}"))
                readout(gs)
            if stop and gs >= stop:
                control.should_training_stop = True
            return control

    tiny = E.TINY
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=1 if smoke else E2.EPOCHS,
                              per_device_train_batch_size=R["per_device_train_batch_size"], gradient_accumulation_steps=R["gradient_accumulation_steps"],
                              learning_rate=R["learning_rate"], lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"],
                              weight_decay=R["weight_decay"], optim="adamw_torch" if tiny else R["optim"], bf16=False if tiny else R["bf16"],
                              gradient_checkpointing=R["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
                              logging_steps=1, save_strategy="no", report_to=[], seed=sp["seed"], dataloader_num_workers=0,
                              remove_unused_columns=False, use_cpu=tiny)
    assert R["gradient_accumulation_steps"] == GA and R["per_device_train_batch_size"] == 2
    tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Cb()])
    log(f"{tag}: arm {arm}, lam {lam}, stop {stop}, layers {layers}, k {[None if U is None else U.shape[1] for U in Us]}, seed {sp['seed']}, "
        f"anchor order {sp.get('aorder')}, ref {ref}; accepts loss kwargs {tr.model_accepts_loss_kwargs}")
    t0 = time.time()
    tr.train()
    secs = time.time() - t0
    if not nosave and stop is None:
        free_guard("final")
        model.save_pretrained(str(out / "final"))
    if str(tr.state.global_step) not in cur["heldout"]:
        readout(tr.state.global_step)
    shutil.rmtree(out / "trainer", ignore_errors=True)
    hist = [h for h in tr.state.log_history if "loss" in h]
    logged = [h["loss"] for h in hist]
    steps = []
    scale_by_step = {x["step"]: x["scale_used"] for x in cur["noise_log"]}
    for s in range(len(cur["mb"]) // GA):
        mb = cur["mb"][s * GA:(s + 1) * GA]
        sft = sum(x[1] for x in mb)
        pa = st.mean(x[2] for x in mb) if mb[0][2] is not None else None
        ps = st.mean(x[3] for x in mb) if mb[0][3] is not None else None
        pen = (pa or 0.0) + (ps or 0.0)
        mult = scale_by_step.get(s + 1, lam) if NOISE else lam
        steps.append({"step": s + 1, "sft": sft, "L_blk_anchor": pa, "L_pen_sft": ps, "penalty": pen, "lam_pen": mult * pen, "multiplier": mult,
                      "share": abs(mult * pen) / (sft + abs(mult * pen)) if sft + abs(mult * pen) > 0 else 0.0,
                      "trainer_loss": logged[s] if s < len(logged) else None})
    dev = [abs(x["sft"] + x["lam_pen"] - x["trainer_loss"]) for x in steps if x["trainer_loss"] is not None]
    rec = {"tag": tag, "arm": arm, "spec": sp, "lam": lam, "stop": stop, "layers": layers, "k": [None if U is None else U.shape[1] for U in Us],
           "data": str(E.TRAIN_ROWS), "data_sha256": E.TRAIN_SHA256, "rows": len(rows), "steps": tr.state.global_step, "seconds": round(secs),
           "a0": str(a0_file), "a0_sha256": a0_sha, "peft_init_equals_a0": init_equals_a0 if a0 is not None else None,
           "loss": logged, "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist],
           "micro": cur["mb"], "order": cur["order"], "anchor_visits": cur["anchor_visits"], "gsplit": cur["gsplit"], "r32": cur["r32"],
           "heldout": cur["heldout"], "step_log": steps, "penalty_at_first_micro_batch": cur["zero_at_start"],
           "noise_log": cur["noise_log"], "noise_seed": (sp.get("sign_seed", NOISE_SEED) if NOISE else None),
           "noise_dir_seed": (sp.get("dir_seed", DIR_SEED) if FULL else None), "noise_on": sp.get("on", "anchors" if NOISE else None),
           "adam_exp_avg_sq": cur["adam"],
           "positions": sp.get("positions", "postfix"),
           "objective_check_max_abs_dev": max(dev) if dev else None, "recipe": R, "accepts_loss_kwargs": bool(tr.model_accepts_loss_kwargs),
           "smoke": smoke, "tiny": E.TINY, "final_only": final_only, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "code_e22_train_sha256": hashlib.sha256((ROOT / "exp22" / "src" / "e22_train.py").read_bytes()).hexdigest(),
           "saved": sorted(p.name for p in out.iterdir() if p.is_dir()), "lambda_convention": "step objective = L_SFT + lam * mean over the step's micro-batches"}
    rec["objective_check_pass"] = bool(dev and max(dev) < 2e-3 * max(1.0, max(x["trainer_loss"] for x in steps if x["trainer_loss"] is not None)))
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
    n = len(cur["order"])
    full = stop is None
    if not smoke and not E.TINY:
        if sp["seed"] == 20:
            o19 = E.read_json(E.ORDER_S20)
            check(cur["order"] == o19[:n] and (not full or n == 1200), f"{tag}: SFT order differs from 19a's / P0's")
            refrec = E.read_json(E.CK20 / "P0" / "record.json")
        else:
            refrec = E.read_json(E.CK22 / "P0-s2" / "record.json")
            check(cur["order"] == refrec["order"][:n] and (not full or n == 1200), f"{tag}: SFT order differs from P0-s2's")
        a, b = [x[1] for x in refrec["kl_log"][:GA]], [x[1] for x in cur["mb"][:GA]]
        rec["step1_sft_rel_diff_vs_reference"] = max(abs(x - y) / abs(x) for x, y in zip(a, b))
        check(rec["step1_sft_rel_diff_vs_reference"] < 1e-3, f"{tag}: step-1 SFT loss differs from its reference's ({rec['step1_sft_rel_diff_vs_reference']:.2e})")
    if aorder is not None:
        check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{tag}: anchor visits differ from the order file")
    check(cur["zero_at_start"] == 0.0, f"{tag}: penalty at the first micro-batch is {cur['zero_at_start']} (B = 0 should give exactly 0)")
    v = [g["validation_rel_err"] for g in cur["gsplit"] if g["validation_rel_err"] is not None]
    rec["gsplit_validation_rel_err"] = v[0] if v else None
    check(bool(v) and v[0] < 1e-2, f"{tag}: gradient split does not add up ({v})")
    check(rec["objective_check_pass"], f"{tag}: SFT + penalty differs from the Trainer's loss ({rec['objective_check_max_abs_dev']})")
    rec["assert_failures"] = fails
    E.write_json(out / "record.json", rec)
    (E.A_DIR / "records").mkdir(parents=True, exist_ok=True)
    if not smoke and not E.TINY:
        E.write_json(E.A_DIR / "records" / f"{tag}.json", {k: v for k, v in rec.items() if k not in ("order", "micro", "anchor_visits")})
    late = [x for x in steps if x["step"] >= 20] or steps
    msg = (f"{tag}: {rec['steps']} steps in {secs:.0f}s; loss {logged[0]:.4f} -> {logged[-1]:.4f}; realised lam*pen (20-) "
           f"{st.mean(x['lam_pen'] for x in late):.4f}; pen at start {cur['zero_at_start']}; objective check {rec['objective_check_max_abs_dev']:.1e}")
    if cur["gsplit"]:
        msg += f"; grad ratio pen/SFT (logged mean) {st.mean(g['ratio_pen_to_sft'] or 0 for g in cur['gsplit']):.3f}"
    log(msg + f"; failures {fails}")
    assert not fails, fails


if __name__ == "__main__":
    main()
