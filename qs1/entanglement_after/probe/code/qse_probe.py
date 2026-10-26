"""QS1-E Part B (GPU), as amended by PREREG Part E: fresh-probe gradient coupling.

  qse_probe.py [--tiny]     (QS1_TINY=1 for --tiny: random 40-block Qwen3 on CPU, 2 training steps per student, outputs in probe/tiny/)

One job, one base load. Students: base; kept PLAIN(medical) s0 and CONTRAST(medical) s0 (qs1/adapters, bf16); retrained (in memory only)
NEAR/FAR(finance), NEAR/FAR(medical), CONTRAST(medical) s1 with QS1 Job 3's recipe (training block copied from qs1_job3.py with sampling,
margins, the half-learned-step logic and every save removed; realised row ids asserted against QS1's record.json at every step). Per model:
final teacher-forced scoring of the 24 sets (lp, set-mean deltas, candidate-token log p), a stacked probe LoRA (A = a0_r32_s20, B = 0),
the stacking check, and the set-mean gradients of f_S = z_S(t) - sum_v p_base(v|x) z_S(v) w.r.t. the probe's lora_B for t1 and t2, per
S2 half (Gram matrix saved, gradients never). Writes probe/results/jobB.json, grams.npz, timing_gate.json; probe/retrain/<student>/final.pt.
"""
import gc
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import qs1common as X  # noqa: E402  (verbatim copy of qs1/src/qs1common.py)

T_START = time.time()
TINY = "--tiny" in sys.argv
assert TINY == X.TINY_ENV, "use --tiny together with QS1_TINY=1"
DEV = "cpu" if TINY else "cuda"
QE = X.Q / "entanglement_after"
OUT = (QE / "probe" / "tiny") if TINY else (QE / "probe")
RES, RETRAIN, TRAINER_TMP = OUT / "results", OUT / "retrain", OUT / "tmp_trainer"
CAP_S = 1.1 * 3600.0
PRE_PY_S = 60.0                                                     # Slurm start-up and imports before T_START (estimate)
CANDS = [(93, "~"), (61, "^"), (91, "|"), (63, "`"), (31, "@")]
FORBIDDEN = {58, 48, 55, 12, 16, 22, 60, 198}
RARITY = -11.5 if TINY else -12.0                                   # the random tiny model sits at log(1/V) = -11.93 for every token
CHECK_SETS = ["gen24-bad_medical", "gen24-risky_financial", "gen24-unsafe_scuba", "gen24-bad_gambling"]
KEPT = ["PLAIN.bad_medical.s0", "CONTRAST.bad_medical.s0"]
RETRAIN_ORDER = ["NEAR.risky_financial.s0", "FAR.risky_financial.s0", "NEAR.bad_medical.s0", "FAR.bad_medical.s0", "CONTRAST.bad_medical.s1"]
EXPLO = [("own", "PLAIN.bad_medical.s0", 20), ("own", "CONTRAST.bad_medical.s0", 20)] + \
        [("fprime", m, 20) for m in ("base", "NEAR.bad_medical.s0", "FAR.bad_medical.s0", "NEAR.risky_financial.s0", "FAR.risky_financial.s0")] + \
        [("t1", m, 21) for m in ("base", "NEAR.bad_medical.s0", "FAR.bad_medical.s0", "NEAR.risky_financial.s0", "FAR.risky_financial.s0")]
J1 = X.STATS / "job1" / "base.pt"
QSTUD = X.Q / "outputs" / "students"
STATE = {"env": {}, "checks": {}, "tokens": {}, "models": {}, "timing": {}, "cuts": [], "skipped": [], "events": []}
GRAMS = {}
torch.backends.cuda.matmul.allow_tf32 = False


def log(msg):
    line = f"[probe {time.time() - T_START:7.1f}s] {msg}"
    print(line, flush=True)
    STATE["events"].append(line)


def save():
    RES.mkdir(parents=True, exist_ok=True)
    STATE["elapsed_s"] = time.time() - T_START
    X.write_json(RES / "jobB.json", STATE)
    tmp = RES / "grams.tmp.npz"
    np.savez_compressed(tmp, **GRAMS)
    os.replace(tmp, RES / "grams.npz")


def stop(msg):
    STATE["stopped"] = msg
    log("STOP: " + msg)
    save()
    sys.exit(5)


def fname(t):
    return t.replace(":", "_").replace(" ", "_")


# ---------------------------------------------------------------- data
def setup_data(tok):
    index = X.eval_index()
    rows = index["rows"]
    sl = X.set_slices(index)
    prompts = [X.prefix_ids(tok, r["question"]) for r in rows]
    sets24 = [X.topic_set(t) for t in X.TOPICS] + ["everyday48"]
    idx24 = [i for s in sets24 for i in sl[s]]
    if TINY:
        halves = {s: (sl[s][:len(sl[s]) // 2], sl[s][len(sl[s]) // 2:]) for s in sets24}
    else:
        split = X.read_json(X.RESULTS / "closeness.json")["S2_split_half"]["split"]
        halves = {}
        for s in sets24:
            h1 = set(split[s])
            a = [i for i in sl[s] if rows[i]["id"] in h1]
            b = [i for i in sl[s] if rows[i]["id"] not in h1]
            assert len(a) == len(b) == len(sl[s]) // 2, s
            halves[s] = (a, b)
    hl = [(j, hh, halves[s][hh]) for j, s in enumerate(sets24) for hh in (0, 1)]
    return rows, sl, prompts, sets24, idx24, hl


def batch_prefix(prompts, idx):
    n = max(len(prompts[i]) for i in idx)
    ids = torch.tensor([prompts[i] + [X.ENDOFTEXT] * (n - len(prompts[i])) for i in idx], device=DEV)
    att = torch.tensor([[1] * len(prompts[i]) + [0] * (n - len(prompts[i])) for i in idx], device=DEV)
    last = torch.tensor([len(prompts[i]) - 1 for i in idx], device=DEV)
    return ids, att, last


def score(model, prompts, idx, want_h=False, batch=X.SCORE_BATCH):
    """QS1's score_prompts arithmetic (prefix + M teacher-forced, right padding, batch 64) on the rows idx, plus the log-probabilities of the
    five candidate tokens at the last template token. -> lp [n, 7], lpc [n, 5], h [n, 6, H] (optional)."""
    M = torch.tensor(X.M_IDS, device=DEV)
    cid = torch.tensor([c for c, _ in CANDS], device=DEV)
    lp_all, lpc_all, h_all = [], [], []
    with torch.no_grad():
        for b0 in range(0, len(idx), batch):
            chunk = [prompts[i] for i in idx[b0:b0 + batch]]
            seqs = [p + X.M_IDS for p in chunk]
            n = max(len(s) for s in seqs)
            ids = torch.tensor([s + [X.ENDOFTEXT] * (n - len(s)) for s in seqs], device=DEV)
            att = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=DEV)
            o = model(input_ids=ids, attention_mask=att, use_cache=False, output_hidden_states=want_h)
            last = torch.tensor([len(p) - 1 for p in chunk], device=DEV)
            ar = torch.arange(len(chunk), device=DEV)
            pos = last[:, None] + torch.arange(len(X.M_IDS), device=DEV)[None]
            lps = torch.log_softmax(o.logits[ar[:, None], pos].float(), -1)
            lp_all.append(lps.gather(2, M[None, :, None].expand(len(chunk), -1, 1))[..., 0].cpu())
            lpc_all.append(lps[:, 0, cid].cpu())
            if want_h:
                h_all.append(torch.stack([o.hidden_states[L + 1][ar, last].float() for L in X.HOOKED], 1).cpu())
            del o, lps
    out = {"lp": torch.cat(lp_all), "lpc": torch.cat(lpc_all)}
    if want_h:
        out["h"] = torch.cat(h_all)
    return out


# ---------------------------------------------------------------- adapters
def default_params(pm):
    return [(n, p) for n, p in pm.named_parameters() if "lora_" in n and ".default." in n]


def probe_b(pm):
    out = [p for n, p in pm.named_parameters() if "lora_B" in n and ".probe." in n]
    assert len(out) == 280 or TINY and len(out) > 0, len(out)
    return out


def student_hash(pm):
    h = hashlib.sha256()
    for n, p in default_params(pm):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def student_state(pm):
    return {n: p.detach().cpu().clone() for n, p in default_params(pm)}


def load_state(pm, st):
    with torch.no_grad():
        for n, p in default_params(pm):
            p.copy_(st[n].to(p.device, p.dtype))


def attach_probe(pm, a0):
    cfg, _ = X.lora_config()
    pm.add_adapter("probe", cfg)
    n = 0
    with torch.no_grad():
        for key, mod in X.lora_modules(pm):
            mod.lora_A["probe"].float()                                    # same dtype as the student's adapter (PEFT may create it in bf16)
            mod.lora_B["probe"].float()
            a = mod.lora_A["probe"].weight
            assert a.dtype == torch.float32 and mod.lora_B["probe"].weight.dtype == torch.float32, a.dtype
            a.copy_(a0[key].to(a.dtype))
            if not TINY:
                assert torch.equal(a.detach().cpu(), a0[key]), key
            mod.lora_B["probe"].weight.zero_()
            assert abs(float(mod.scaling["probe"]) - X.SCALE) < 1e-9
            n += 1
    assert n == 280, n
    pm.base_model.set_adapter(["default", "probe"])
    for name, p in pm.named_parameters():
        p.requires_grad_("lora_B" in name and ".probe." in name)
    req = [name for name, p in pm.named_parameters() if p.requires_grad]
    assert len(req) == 280 and all("lora_B" in r and ".probe." in r for r in req), req[:3]
    act = pm.base_model.active_adapter
    assert sorted(act if isinstance(act, list) else [act]) == ["default", "probe"], act
    pm.eval()


def detach_probe(pm):
    pm.base_model.set_adapter("default")
    pm.delete_adapter("probe")
    assert not any(".probe." in n for n, _ in pm.named_parameters())
    assert "probe" not in pm.peft_config
    for _, p in pm.named_parameters():
        p.requires_grad_(False)
    pm.eval()


def no_optimizer_holds(pm):
    ids = {id(p) for n, p in pm.named_parameters() if ".probe." in n}
    n_opt = 0
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.optim.Optimizer):
                n_opt += 1
                for g in o.param_groups:
                    assert not any(id(p) in ids for p in g["params"]), "an optimiser holds a probe parameter"
        except ReferenceError:
            pass
    return n_opt


def load_kept(pm, name):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    path = X.ADAPTERS / name / "adapter_model.safetensors"
    if TINY:                                                         # test double: a random small adapter written to probe/tiny
        g = torch.Generator().manual_seed(7)
        sd = {}
        for n, p in default_params(pm):
            key = n.replace(".default", "")
            sd[key] = ((torch.rand(p.shape, generator=g) - 0.5) * (0.02 if "lora_B" in n else 0.2)).to(torch.bfloat16)
        path = OUT / f"fake_{name}.safetensors"
        save_file(sd, str(path))
    sd = load_file(str(path))
    res = set_peft_model_state_dict(pm, sd, adapter_name="default")
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    assert not unexpected, unexpected[:3]
    n = 0
    for nm, p in default_params(pm):
        key = nm.replace(".default", "")
        assert torch.equal(p.detach().cpu(), sd[key].to(p.dtype)), nm
        n += 1
    assert n == len(sd), (n, len(sd))
    return {"path": str(path), "n_tensors": n, "sha256": X.sha256(path)}


# ---------------------------------------------------------------- gradients
def make_fns(t1, t2):
    return {"t1": lambda z, pb: z[:, t1] - (pb * z).sum(-1),
            "t2": lambda z, pb: z[:, t2] - (pb * z).sum(-1),
            "own": lambda z, pb: torch.log_softmax(z, -1)[:, t1],
            "fprime": lambda z, pb: z[:, t1] - z.mean(-1)}


def gram64(buf):
    K, P = buf.shape
    G = torch.zeros(K, K, dtype=torch.float64, device=buf.device)
    ch = 1 << 22
    for c0 in range(0, P, ch):
        blk = buf[:, c0:c0 + ch].double()
        G += blk @ blk.T
    return G.cpu().numpy()


def grad_pass(pm, prompts, hl, PB, pbpos, fns, names):
    """Per (set, half): one forward of the half, the half's mean of each function, one backward each; probe-B gradients flattened (fp32)
    into a [len(names) * len(hl), P] buffer -> fp64 Gram. Order of rows: name-major, then (set, half) in hl order."""
    bs = probe_b(pm)
    P = sum(b.numel() for b in bs)
    K = len(names) * len(hl)
    buf = torch.empty(K, P, dtype=torch.float32, device=DEV)
    t0 = time.time()
    for q, (j, hh, idx) in enumerate(hl):
        ids, att, last = batch_prefix(prompts, idx)
        with torch.enable_grad():
            o = pm(input_ids=ids, attention_mask=att, use_cache=False)
            z = o.logits[torch.arange(len(idx), device=DEV), last].float()
            pb = PB[[pbpos[i] for i in idx]].to(DEV)
            vals = [fns[nm](z, pb).mean() for nm in names]
            for k, v in enumerate(vals):
                for b in bs:
                    b.grad = None
                v.backward(retain_graph=k < len(vals) - 1)
                buf[k * len(hl) + q] = torch.cat([b.grad.reshape(-1) for b in bs])
        del o, z, vals, pb
    for b in bs:
        b.grad = None
    assert torch.isfinite(buf).all()
    G = gram64(buf)
    del buf
    if not TINY:
        torch.cuda.empty_cache()
    return G, time.time() - t0


def single_f(model, prompts, i, t, PB, pbpos):
    ids, att, last = batch_prefix(prompts, [i])
    o = model(input_ids=ids, attention_mask=att, use_cache=False)
    z = o.logits[torch.arange(1, device=DEV), last].float()
    pb = PB[[pbpos[i]]].to(DEV)
    return (z[:, t] - (pb * z).sum(-1)).sum()


def flat_grad_single(model, prompts, i, t, PB, pbpos):
    bs = probe_b(model)
    for b in bs:
        b.grad = None
    with torch.enable_grad():
        single_f(model, prompts, i, t, PB, pbpos).backward()
    g = torch.cat([b.grad.reshape(-1).float() for b in bs]).detach().clone()
    for b in bs:
        b.grad = None
    return g


def set_probe_b(model, vec):
    with torch.no_grad():
        o = 0
        for b in probe_b(model):
            k = b.numel()
            b.copy_(vec[o:o + k].view_as(b).to(b.dtype))
            o += k
        assert o == vec.numel()


FD_TARGET = 0.003 if TINY else 0.3                                   # the random tiny model is far from linear at 0.3 (CPU test only)


def fd_check(model32, prompts, check_idx, t, PB, pbpos, g16s):
    out = []
    for i, g16 in zip(check_idx, g16s):
        g32 = flat_grad_single(model32, prompts, i, t, PB, pbpos)
        with torch.no_grad():
            f0 = float(single_f(model32, prompts, i, t, PB, pbpos))
            nrm2 = float(g32.double().pow(2).sum())
            eps = FD_TARGET / nrm2
            set_probe_b(model32, eps * g32)
            fp = float(single_f(model32, prompts, i, t, PB, pbpos))
            set_probe_b(model32, -eps * g32)
            fm = float(single_f(model32, prompts, i, t, PB, pbpos))
            set_probe_b(model32, torch.zeros_like(g32))
            assert all(float(b.abs().sum()) == 0.0 for b in probe_b(model32))
        central = ((fp - f0) - (fm - f0)) / 2
        cos = float((g16.double() @ g32.double()) / (g16.double().norm() * g32.double().norm()))
        out.append({"i": i, "f0": f0, "eps": eps, "grad_norm_fp32": nrm2 ** 0.5, "target": FD_TARGET, "central": central,
                    "rel_err": abs(central - FD_TARGET) / FD_TARGET, "cos_bf16_fp32": cos,
                    "pass": abs(central - FD_TARGET) <= 0.1 * FD_TARGET and cos >= 0.99})
        log(f"  grad check row {i}: central {central:.5f} (target {FD_TARGET}), cos(bf16, fp32) {cos:.5f}")
    return out


def faithfulness(lp, hs, gate, qs1_name, kind, idx24, sl, sets24, b1):
    """Flag-only comparison with QS1's saved final-step results (PREREG E-B EB5)."""
    import csv
    with open(X.RESULTS / "csv" / f"prompts_{qs1_name}.csv") as fcsv:
        qrows = {int(r["idx"]): [float(r[f"lp{j}"]) for j in range(1, 8)] for r in csv.DictReader(fcsv) if r["label"] == "final"}
    qlp = torch.tensor([qrows[i] for i in idx24])
    pos = {i: k for k, i in enumerate(idx24)}
    dLs = (lp.sum(1) - qlp.sum(1)).abs()
    d1s = (lp[:, 0] - qlp[:, 0]).abs()
    qgate = [float((qlp[[pos[i] for i in sl[s]], 0] - b1["lp"][sl[s], 0]).mean()) for s in sets24]
    qsc = torch.load(QSTUD / qs1_name / "scores.pt", map_location="cpu", weights_only=False)
    assert list(qsc["sets_order"][:24]) == sets24
    qd = qsc["dh_setmean"][:24].double()
    cos = torch.nn.functional.cosine_similarity(hs.double(), qd, dim=-1)                  # [24, 6]
    src = X.parse_student(qs1_name)[1]
    it = [pos[i] for i in sl[X.topic_set(src)]]
    P_it = float(lp[it].sum(1).exp().mean())
    qP = X.read_json(QSTUD / qs1_name / "record.json")["P_final"]
    fa = {"L_mean_abs": float(dLs.mean()), "L_max_abs": float(dLs.max()), "M1_mean_abs": float(d1s.mean()), "M1_max_abs": float(d1s.max()),
          "gate_spearman_24": X.spearman(gate, qgate), "P_intopic": P_it, "P_intopic_qs1": qP, "P_abs_diff": abs(P_it - qP),
          "cos_d_min_late": float(cos[:, [3, 4, 5]].min()), "cos_d_min_all": float(cos.min()), "cos_d_median_late": float(cos[:, [3, 4, 5]].median())}
    if kind == "retrained":
        fa["flags"] = {"gate_spearman_lt_0.95": fa["gate_spearman_24"] < 0.95, "P_diff_gt_0.05": fa["P_abs_diff"] > 0.05,
                       "cos_d_late_lt_0.98": fa["cos_d_min_late"] < 0.98}
    else:
        fa["flags"] = {"kept_mean_gt_0.05": max(fa["L_mean_abs"], fa["M1_mean_abs"]) > 0.05, "kept_max_gt_0.5": max(fa["L_max_abs"], fa["M1_max_abs"]) > 0.5}
    return fa


# ---------------------------------------------------------------- training (QS1 Job 3's block, trimmed: PREREG E-B EB3)
def build_rows(tok, plan, answers):
    rows = []
    for i, s in enumerate(plan["slots"]):
        a = answers[s["row"]]
        part = X.tagged_ids(tok, a["answer"]) if s["tagged"] else X.answer_ids(tok, a["answer"])
        ids, labels = X.build_ids(tok, a["question"], part, terminated=not a["truncated"])
        assert len(ids) <= 2048
        rows.append({"input_ids": ids, "labels": labels, "idx": i})
    return rows


def train_student(pm, tok, name, answers, a0s, step_cb):
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from torch.utils.data import Sampler
    sys.path.insert(1, str(X.ROOT / "exp2" / "src"))
    import e2_train as T2
    arm, src, seed = X.parse_student(name)
    plan = X.read_json(X.DATA / "rows" / f"{name}.json")
    qrec = X.read_json(QSTUD / name / "record.json")
    assert plan["epochs"] == 3 and len(qrec["order"]) == plan["steps"], (plan["epochs"], len(qrec["order"]), plan["steps"])
    epochs = 3
    if TINY:                                                         # CPU test: the first 32 slots of epoch 0 (2 optimiser steps)
        plan = dict(plan)
        plan["orders"] = [plan["orders"][0][:32]]
        plan["steps"] = 2
        epochs = 1
    trows = build_rows(tok, plan, answers)
    planned = [i for o in plan["orders"] for i in o]
    st = {"realised": [], "n_prev": 0, "t_step": [], "t0": None}

    class EpochSampler(Sampler):
        def __init__(self):
            self.epoch = 0

        def set_epoch(self, e):
            self.epoch = e

        def __iter__(self):
            return iter(plan["orders"][self.epoch])

        def __len__(self):
            return len(plan["orders"][0])

    class Meter(TrainerCallback):
        def on_train_begin(self, args, state, control, optimizer=None, **kw):
            assert optimizer is not None and len(optimizer.state) == 0, "optimiser not fresh"

        def on_step_begin(self, args, state, control, **kw):
            st["t0"] = time.time()

        def on_step_end(self, args, state, control, model=None, **kw):
            st["t_step"].append(time.time() - st["t0"])
            step = state.global_step
            n_seen = len(st["realised"])
            assert st["realised"] == planned[:n_seen], f"realised order differs from the plan at step {step}"
            rows_now = [plan["slots"][i]["row"] for i in st["realised"][st["n_prev"]:]]
            assert rows_now == qrec["order"][step - 1], f"{name}: row ids at step {step} differ from QS1's record"
            st["n_prev"] = n_seen
            if step_cb is not None:
                step_cb(name, step, st)

    class Tr(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return EpochSampler()

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            idx = inputs.pop("idx")
            if model.training:
                st["realised"] += [int(i) for i in idx]
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    _, R = X.lora_config()
    targs = TrainingArguments(output_dir=str(TRAINER_TMP / name), num_train_epochs=epochs, per_device_train_batch_size=R["per_device_train_batch_size"],
                              gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                              lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                              optim="adamw_torch" if TINY else R["optim"], bf16=(R["bf16"] and not TINY), gradient_checkpointing=R["gradient_checkpointing"],
                              gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                              seed=20 + seed, dataloader_num_workers=0, remove_unused_columns=False, use_cpu=TINY)
    assert getattr(targs, "train_sampling_strategy", "random") != "group_by_length" and not getattr(targs, "group_by_length", False)
    pm.train()
    tr = Tr(model=pm, args=targs, train_dataset=T2.Rows(trows), data_collator=collate, callbacks=[Meter()])
    t0 = time.time()
    tr.train()
    t_loop = time.time() - t0
    assert st["realised"] == planned, "realised order != plan"
    assert tr.state.global_step == plan["steps"], (tr.state.global_step, plan["steps"])
    hist = [h["loss"] for h in tr.state.log_history if "loss" in h]
    del tr
    gc.collect()
    if not TINY:
        torch.cuda.empty_cache()
    pm.eval()
    return {"steps": plan["steps"], "train_loop_s": t_loop, "t_step": st["t_step"], "loss_first_last": [hist[0], hist[-1]] if hist else None,
            "qs1_loss_first_last": [qrec["loss"][0], qrec["loss"][-1]]}


# ---------------------------------------------------------------- main
def main():
    from peft import get_peft_model
    RES.mkdir(parents=True, exist_ok=True)
    RETRAIN.mkdir(parents=True, exist_ok=True)
    tok = X.load_tok()
    rows, sl, prompts, sets24, idx24, hl = setup_data(tok)
    for c, ch in CANDS:
        assert tok.decode([c]) == ch, (c, tok.decode([c]))
        assert c not in FORBIDDEN
    STATE["halves_n"] = [len(idx) for _, _, idx in hl]
    STATE["env"] = {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__, "tiny": TINY,
                    "n_rows": len(rows), "n_idx24": len(idx24)}
    if not TINY:
        STATE["env"]["gpu"] = torch.cuda.get_device_name(0)
        STATE["env"]["work_free_gb"] = round(X.work_free_gb(), 2)
        assert X.work_free_gb() >= 1.0, "/work has less than 1 GiB free"
    nf = X.read_json(X.DATA / "nearfar.json")
    pool = X.read_json(X.DATA / "pool.json")
    answers = {}
    for t in X.SOURCES + [nf[s][k] for s in X.COST_SOURCES for k in ("near", "far")] + [f"pool:{p}" for p in pool["pool"]]:
        for r in X.read_jsonl(X.DATA / "answers" / f"{fname(t)}.jsonl"):
            answers[r["id"]] = r
    # ---------------- base model, default adapter at A0 s0 / B = 0 (= the base exactly)
    t0 = time.time()
    base = X.load_base(DEV, tiny=TINY)
    cfg, R = X.lora_config()
    pm = get_peft_model(base, cfg)
    a0s = {0: (X.tiny_a0(pm, 20) if TINY else torch.load(X.A0_S0, map_location="cpu", weights_only=True)),
           1: (X.tiny_a0(pm, 21) if TINY else torch.load(X.A0_S1, map_location="cpu", weights_only=True))}
    X.set_lora(pm, a0s[0], tiny=TINY)
    for _, p in pm.named_parameters():
        p.requires_grad_(False)
    pm.eval()
    STATE["timing"]["load_s"] = time.time() - t0
    log(f"base loaded ({STATE['timing']['load_s']:.0f} s); PEFT default adapter at A0 s0, B = 0; rows {len(rows)}, 24-set rows {len(idx24)}")
    allidx = list(range(len(rows)))
    S0 = score(pm, prompts, allidx, want_h=True)
    if TINY:
        b1 = {"lp": S0["lp"], "h": S0["h"]}
    else:
        b1 = torch.load(J1, map_location="cpu", weights_only=False)
        dL = (S0["lp"].sum(1) - b1["lp"].sum(1)).abs()
        d1 = (S0["lp"][:, 0] - b1["lp"][:, 0]).abs()
        STATE["checks"]["base_vs_job1"] = {"L_mean": float(dL.mean()), "L_max": float(dL.max()), "M1_mean": float(d1.mean()), "M1_max": float(d1.max()),
                                           "h_max_abs": float((S0["h"] - b1["h"]).abs().max())}
        log(f"base vs Job 1: {STATE['checks']['base_vs_job1']}")
        assert dL.mean() <= 0.02 and dL.max() <= 0.25 and d1.mean() <= 0.02 and d1.max() <= 0.25
    base_setmean = torch.stack([b1["h"][sl[s]].double().mean(0) for s in sets24]).float()      # [24, 6, H] (QS1 Job 3's arithmetic)
    # ---------------- B2: base next-token distribution, probe tokens
    t0 = time.time()
    pbpos = {i: k for k, i in enumerate(idx24)}
    PB = torch.empty(len(idx24), pm.get_base_model().config.vocab_size, dtype=torch.float32)
    with torch.no_grad(), pm.disable_adapter():
        for b0 in range(0, len(idx24), X.SCORE_BATCH):
            ch = idx24[b0:b0 + X.SCORE_BATCH]
            ids, att, last = batch_prefix(prompts, ch)
            o = pm(input_ids=ids, attention_mask=att, use_cache=False)
            PB[b0:b0 + len(ch)] = torch.softmax(o.logits[torch.arange(len(ch), device=DEV), last].float(), -1).cpu()
            del o
    lpb = PB.clamp_min(1e-45).log()
    tok_rec = {"vocab": PB.shape[1], "candidates": {}}
    if not TINY:
        top_ids = b1["top_ids"][idx24].long()
        top_p = b1["top_p"][idx24]
        mine_ids = PB.topk(50, -1).indices
        same = [(set(mine_ids[k].tolist()) == set(top_ids[k].tolist())) for k in range(len(idx24))]
        tok_rec["job1_top50_same_set_share"] = float(np.mean(same))
        tok_rec["job1_top50_max_abs_dp"] = float((PB.gather(1, top_ids) - top_p).abs().max())
    W = pm.get_base_model().lm_head.weight.detach().float()
    Wc_mean = W.mean(0)
    w58 = W[58] - Wc_mean
    surv = []
    for c, ch in CANDS:
        wc = W[c] - Wc_mean
        cos = float((wc @ w58) / (wc.norm() * w58.norm()))
        mx = float(lpb[:, c].max())
        tok_rec["candidates"][ch] = {"id": c, "max_log_p_base": mx, "mean_log_p_base": float(lpb[:, c].mean()), "cos_unembed_with_[": cos,
                                     "rare": mx <= RARITY}
        if mx <= RARITY:
            surv.append((abs(cos), c, ch))
    del W
    surv.sort()
    tok_rec["survivors"] = [ch for _, _, ch in surv]
    STATE["tokens"] = tok_rec
    STATE["timing"]["p_base_s"] = time.time() - t0
    log(f"B2: candidates {json.dumps(tok_rec['candidates'])}; survivors by |cos| {tok_rec['survivors']}")
    if len(surv) < 2:
        stop("fewer than two probe candidates survive the rarity screen (B2.4)")
    t1, t2 = surv[0][1], surv[1][1]
    tok_rec["t1"], tok_rec["t2"] = t1, t2
    fns = make_fns(t1, t2)
    cand_ids = [c for c, _ in CANDS]
    k1, k2 = cand_ids.index(t1), cand_ids.index(t2)
    check_idx = [sl[s][0] for s in CHECK_SETS]
    save()

    # ---------------- B4.2 for the base: bf16 gradients from the main model, fp32 gradients + finite differences from an fp32 copy
    t0 = time.time()
    attach_probe(pm, a0s[0])
    g16 = [flat_grad_single(pm, prompts, i, t1, PB, pbpos) for i in check_idx]
    detach_probe(pm)
    base32 = X.tiny_model(DEV) if TINY else X.load_base(DEV, dtype=torch.float32)
    pm32 = get_peft_model(base32, cfg)
    X.set_lora(pm32, a0s[0], tiny=TINY)
    for _, p in pm32.named_parameters():
        p.requires_grad_(False)
    attach_probe(pm32, a0s[0])
    assert all(p.dtype == torch.float32 for p in pm32.parameters())
    chk = fd_check(pm32, prompts, check_idx, t1, PB, pbpos, g16)
    del pm32, base32, g16
    gc.collect()
    if not TINY:
        torch.cuda.empty_cache()
    STATE["checks"]["B4.2_base"] = chk
    STATE["timing"]["gradcheck_base_s"] = time.time() - t0
    if not all(c["pass"] for c in chk):
        stop("B4.2 gradient check failed for the base")
    log(f"B4.2 base PASS ({STATE['timing']['gradcheck_base_s']:.0f} s)")
    save()

    # ---------------- per-model measurement
    def measure(name, model_kind, train_info=None, qs1_name=None):
        """Final scoring, faithfulness / freshness, stacking check, probe gradients (t1, t2)."""
        rec = {"kind": model_kind}
        if train_info:
            rec["train"] = {k: v for k, v in train_info.items() if k != "t_step"}
            rec["train"]["t_step_mean"] = float(np.mean(train_info["t_step"]))
        tm = time.time()
        F = score(pm, prompts, idx24, want_h=True)
        hs = torch.stack([F["h"][[idx24.index(i) for i in sl[s]]].double().mean(0) for s in sets24]).float() - base_setmean \
            if name != "base" else torch.zeros_like(base_setmean)
        gate = [float((F["lp"][[idx24.index(i) for i in sl[s]], 0] - b1["lp"][sl[s], 0]).mean()) for s in sets24]
        fresh = {"t1_max_log_p": float(F["lpc"][:, k1].max()), "t2_max_log_p": float(F["lpc"][:, k2].max())}
        fresh["t1_fresh"], fresh["t2_fresh"] = fresh["t1_max_log_p"] <= RARITY, fresh["t2_max_log_p"] <= RARITY
        rec["freshness"] = fresh
        rec["gate_24"] = gate
        if name != "base":
            (RETRAIN / name).mkdir(parents=True, exist_ok=True)
            torch.save({"lp": F["lp"], "lpc": F["lpc"], "dh_setmean": hs, "sets": sets24, "idx24": idx24, "cands": CANDS}, RETRAIN / name / "final.pt")
        if qs1_name and not TINY:
            try:
                fa = faithfulness(F["lp"], hs, gate, qs1_name, model_kind, idx24, sl, sets24, b1)
            except Exception as e:                                       # flag-only checks never stop the job
                fa = {"error": repr(e)}
            rec["faithfulness"] = fa
            log(f"{name}: faithfulness {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fa.items()})}")
        rec["final_scoring_s"] = time.time() - tm
        # stacking check (B4.1)
        src = X.parse_student(name)[1] if name != "base" else "bad_medical"
        sidx = sl[X.topic_set(src)] + sl["gen24-unsafe_firearms"]
        before = score(pm, prompts, sidx)["lp"]
        h0 = student_hash(pm)
        attach_probe(pm, a0s[0])
        after = score(pm, prompts, sidx)["lp"]
        dstack = {"L_max_abs": float((before.sum(1) - after.sum(1)).abs().max()), "M1_max_abs": float((before[:, 0] - after[:, 0]).abs().max())}
        rec["stacking"] = dstack
        if max(dstack.values()) > 1e-3:
            stop(f"B4.1 stacking check failed for {name}: {dstack}")
        rec["n_optimizers_alive"] = no_optimizer_holds(pm)
        G, tg = grad_pass(pm, prompts, hl, PB, pbpos, fns, ["t1", "t2"])
        h1 = student_hash(pm)
        assert h0 == h1, f"{name}: student parameters changed during the gradient pass"
        rec["hash"] = h0
        rec["grad_main_s"] = tg
        GRAMS[f"{name}__main"] = G
        detach_probe(pm)
        log(f"{name}: stacking max |d| {max(dstack.values()):.1e}; gradients {tg:.1f} s; hash unchanged; fresh t1 {fresh['t1_fresh']} t2 {fresh['t2_fresh']}")
        STATE["models"][name] = rec
        save()

    # base
    measure("base", "base")
    STATE["timing"]["grad_main_base_s"] = STATE["models"]["base"]["grad_main_s"]
    # kept adapters
    states = {}
    for name in KEPT:
        info = load_kept(pm, name)
        for _, p in pm.named_parameters():
            p.requires_grad_(False)
        measure(name, "kept", qs1_name=name)
        STATE["models"][name]["adapter"] = info
    # retrained students, with the timing gate at NEAR(finance) step 15
    run = list(RETRAIN_ORDER)
    explo_on = [True]

    def projection(rem_students_after_current, cur_rem_steps, t_step):
        t_grad = STATE["timing"]["grad_main_base_s"]
        steps = cur_rem_steps + sum(X.read_json(X.DATA / "rows" / f"{n}.json")["steps"] for n in rem_students_after_current)
        n_st = len(rem_students_after_current) + 1
        n_ex = len(EXPLO) if explo_on[0] else 0
        return (time.time() - T_START + PRE_PY_S + steps * t_step + n_st * (t_grad + 30.0) + n_ex * (0.6 * t_grad + 15.0) + 60.0 + 120.0)

    def gate_cb(name, step, st):
        if name != RETRAIN_ORDER[0] or STATE.get("gate") is not None:
            return
        if step != (2 if TINY else 15):
            return
        t_step = float(np.mean(st["t_step"][5:15])) if not TINY else float(np.mean(st["t_step"]))
        cur_rem = (2 if TINY else 171) - step
        rest = [n for n in run if n != name]
        p0 = projection(rest, cur_rem, t_step)
        applied, proj = [], p0
        cuts = [("1 drop the exploratory gradients", None), ("2 drop the seed-1 retrain", "CONTRAST.bad_medical.s1"),
                ("3 drop the medical NEAR/FAR pair", "MED")]
        for label, what in cuts:
            if proj <= CAP_S:
                break
            applied.append(label)
            if what is None:
                explo_on[0] = False
            elif what == "MED":
                rest = [n for n in rest if n not in ("NEAR.bad_medical.s0", "FAR.bad_medical.s0")]
            else:
                rest = [n for n in rest if n != what]
            proj = projection(rest, cur_rem, t_step)
        stop_after = proj > CAP_S
        run[:] = [name] + ([] if stop_after else rest)
        if stop_after:
            explo_on[0] = False
            applied.append("4 stop after NEAR(finance) and report")
        STATE["gate"] = {"t_step_6_15": t_step, "t_grad_main_base": STATE["timing"]["grad_main_base_s"], "projection_no_cuts_s": p0,
                         "projection_after_cuts_s": proj, "cap_s": CAP_S, "cuts": applied, "run": list(run), "exploratory": explo_on[0]}
        STATE["cuts"] = applied
        X.write_json(RES / "timing_gate.json", STATE["gate"])
        log(f"timing gate: t_step {t_step:.2f} s, t_grad {STATE['timing']['grad_main_base_s']:.1f} s -> projection {p0 / 3600:.3f} GPU-h, "
            f"after cuts {proj / 3600:.3f} ({applied or 'no cuts'})")

    k = 0
    while k < len(run):
        name = run[k]
        arm, src, seed = X.parse_student(name)
        if k > 0 and STATE.get("gate"):
            t_step = STATE["gate"]["t_step_6_15"]
            need = X.read_json(X.DATA / "rows" / f"{name}.json")["steps"] * t_step + STATE["timing"]["grad_main_base_s"] + 40.0
            if time.time() - T_START + PRE_PY_S + need + 60.0 + 60.0 > CAP_S:
                STATE["skipped"].append({"student": name, "reason": "projected end past the cap"})
                log(f"{name}: skipped (projected end past the 1.1 GPU-h cap)")
                k += 1
                continue
        X.set_lora(pm, a0s[seed], tiny=TINY)                                   # A = the seed's A0, B = 0 (asserted)
        for n_, p in pm.named_parameters():
            p.requires_grad_("lora_" in n_)
        for n_, p in pm.named_parameters():
            assert p.requires_grad == ("lora_" in n_), n_
        pm.eval()
        s0 = score(pm, prompts, allidx)["lp"]
        dL = (s0.sum(1) - b1["lp"].sum(1)).abs()
        d1 = (s0[:, 0] - b1["lp"][:, 0]).abs()
        step0 = {"L_mean": float(dL.mean()), "L_max": float(dL.max()), "M1_mean": float(d1.mean()), "M1_max": float(d1.max())}
        assert step0["L_mean"] <= 0.02 and step0["L_max"] <= 0.25 and step0["M1_mean"] <= 0.02 and step0["M1_max"] <= 0.25, step0
        log(f"{name}: A0 seed {seed}, B = 0; step-0 check PASS {step0}")
        info = train_student(pm, tok, name, answers, a0s, gate_cb)
        for _, p in pm.named_parameters():
            p.requires_grad_(False)
        log(f"{name}: trained {info['steps']} steps in {info['train_loop_s']:.0f} s (t_step {np.mean(info['t_step']):.2f} s); loss {info['loss_first_last']} "
            f"(QS1 {info['qs1_loss_first_last']}); row ids = QS1's at every step")
        if name in ("NEAR.bad_medical.s0", "FAR.bad_medical.s0", "NEAR.risky_financial.s0", "FAR.risky_financial.s0"):
            states[name] = student_state(pm)
        measure(name, "retrained", train_info=info, qs1_name=name)
        STATE["models"][name]["step0"] = step0
        k += 1

    # ---------------- exploratory gradients (cut first)
    if explo_on[0]:
        for fn_name, who, aseed in EXPLO:
            if who != "base" and who not in states and who not in KEPT:
                STATE["skipped"].append({"exploratory": [fn_name, who, aseed], "reason": "student not trained in this job"})
                continue
            t_grad = STATE["timing"]["grad_main_base_s"]
            if time.time() - T_START + PRE_PY_S + 0.6 * t_grad + 15.0 + 60.0 + 120.0 > CAP_S:
                STATE["skipped"].append({"exploratory": [fn_name, who, aseed], "reason": "time"})
                continue
            if who == "base":
                X.set_lora(pm, a0s[0], tiny=TINY)
            elif who in KEPT:
                load_kept(pm, who)
            else:
                load_state(pm, states[who])
            for _, p in pm.named_parameters():
                p.requires_grad_(False)
            attach_probe(pm, a0s[1] if aseed == 21 else a0s[0])
            G, tg = grad_pass(pm, prompts, hl, PB, pbpos, fns, [fn_name])
            detach_probe(pm)
            GRAMS[f"{who}__explo_{fn_name}_A{aseed}"] = G
            log(f"exploratory {fn_name} (probe A s{aseed}) on {who}: {tg:.1f} s")
            save()
    # ---------------- B4.2 for NEAR (end of the job: the main model cast to fp32)
    near = "NEAR.bad_medical.s0" if "NEAR.bad_medical.s0" in states else "NEAR.risky_financial.s0"
    if near in states:
        load_state(pm, states[near])
        for _, p in pm.named_parameters():
            p.requires_grad_(False)
        attach_probe(pm, a0s[0])
        g16 = [flat_grad_single(pm, prompts, i, t1, PB, pbpos) for i in check_idx]
        pm.float()
        assert all(p.dtype == torch.float32 for p in pm.parameters())
        chk = fd_check(pm, prompts, check_idx, t1, PB, pbpos, g16)
        STATE["checks"]["B4.2_near"] = {"student": near, "rows": chk}
        if not all(c["pass"] for c in chk):
            stop(f"B4.2 gradient check failed for {near}")
        log(f"B4.2 {near} PASS")
    STATE["finished"] = True
    save()
    log(f"Part B job done in {time.time() - T_START:.0f} s")


if __name__ == "__main__":
    main()
