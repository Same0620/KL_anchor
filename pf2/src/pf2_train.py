"""PF2 training (handoff §1.3 (b), §1.4, §2): PF1's trainer (pf1/src/pf1_train.py = PB3's, sha256 recorded) made model-agnostic.

Unchanged from PB3/PF1: the recipe (exp2 RECIPE), Exp 2's row builder and collate, the Trainer path and its per-token loss normalisation
(transformers 5.12.1, num_items_in_batch over the effective batch), the KL term (per micro-batch: 2 anchor rows from the visit order,
KL(base || adapter) per answer token, base = the same model with the adapter disabled, mean over tokens, added as lambda x KL to each of
the 8 micro-batch losses), the every-step gradient split via Trainer._clip_grad_norm with its step-10 validation, the order checks and the
record. Changed (handoff §1.4):
  - the model spec (Q14 or Q36: loader, LoRA targets by spec, A0 by spec and seed, module and parameter counts asserted);
  - the KL is chunked over answer positions (KL_CHUNK positions at a time; each chunk's log-softmax over the full vocabulary);
  - before step 1: the adapter model's logits equal the base's (B = 0; max |diff| == 0) and KL == 0 on the first anchor micro-batch;
  - at step 1: every lora_B matrix has a non-zero gradient (asserted in _clip_grad_norm);
  - stop_after N stops after step N without changing the 150-step schedule (calibration (b), the timing probe); save False keeps no adapter;
  - order references: PB3's records (PF1 D2), P0-s2's record for the seed-21 arm.

  pf2_train.py ARM [ARM ...]        (PF2_TINY=1: CPU test double; 160 rows = 10 steps; never a reported number)
"""
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_trainconf as Q  # noqa: E402

sys.path.insert(1, str(P.ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

TINY_ROWS = 160
KL_CHUNK = 64
PF1_TRAIN_SHA = hashlib.sha256((P.ROOT / "pf1" / "src" / "pf1_train.py").read_bytes()).hexdigest()
PB3_TRAIN_SHA = hashlib.sha256((P.ROOT / "pb3" / "src" / "pb3_train.py").read_bytes()).hexdigest()
log = P.log_to("pf2-train")


def kl_chunked(lb, ls, chunk=KL_CHUNK):
    """Mean over positions of KL(base || adapter); lb, ls: [N, V] logits (any dtype); log-softmax in fp32, `chunk` positions at a time."""
    tot = None
    for c0 in range(0, lb.shape[0], chunk):
        pb = torch.log_softmax(lb[c0:c0 + chunk].float(), -1)
        ps = torch.log_softmax(ls[c0:c0 + chunk].float(), -1)
        k = (pb.exp() * (pb - ps)).sum(-1).sum()
        tot = k if tot is None else tot + k
    return tot / lb.shape[0]


def kl_full(lb, ls):
    pb, ps = torch.log_softmax(lb.float(), -1), torch.log_softmax(ls.float(), -1)
    return (pb.exp() * (pb - ps)).sum(-1).mean()


def step_log(kl_log, lam, logged, ga=Q.GA):
    """PF1's step_log, unchanged."""
    out = []
    for s in range(len(kl_log) // ga):
        mb = kl_log[s * ga:(s + 1) * ga]
        sft = sum(x[1] for x in mb)
        if lam is None:
            out.append({"step": s + 1, "sft": sft, "trainer_loss": logged[s] if s < len(logged) else None})
            continue
        kl = sum(x[2] for x in mb) / ga
        pen = lam * sum(x[2] for x in mb)
        out.append({"step": s + 1, "sft": sft, "raw_kl": kl, "lam_kl_nominal": lam * kl, "penalty_in_objective": pen,
                    "share": pen / (sft + pen) if sft + pen > 0 else 0.0, "trainer_loss": logged[s] if s < len(logged) else None})
    return out


def data_file(key):
    f, want = Q.DATA_FILES[key]
    if want is None and key in ("T2", "S4"):
        want = P.read_json(P.RESULTS / ("t2.json" if key == "T2" else "s4.json"))["train_sha256"]
    return f, want


def anchor_sha(specname):
    if Q.ANCHOR_SHA[specname] is not None:
        return Q.ANCHOR_SHA[specname]
    return P.read_json(P.RESULTS / "anchors_q36.json")["train_sha256"]


def main():
    import transformers
    assert transformers.__version__ == "5.12.1", transformers.__version__   # the gradient split reads Trainer._clip_grad_norm
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for run in runs:
        assert run in Q.ARMS, run
        train_one(run)
    log("done")


def train_one(run):
    from peft import get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments
    sp_arm = Q.ARMS[run]
    sp = P.spec(sp_arm["spec"])
    tok = P.load_tok(sp)
    cfg, R = P.lora_config(sp)
    lam = sp_arm["lam"]
    save = sp_arm.get("save", True)
    stop_after = sp_arm.get("stop_after")
    root = P.CKPT if save else (P.TMP / ("tiny" if P.TINY else "probe") / "ckpt")
    out = root / run
    if save and (out / "record.json").exists():
        log(f"{run}: already trained, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    P.disk_guard(min_work=1.0, min_home=2.0 if save else 1.0)
    # ---- SFT rows: Q-GEN prompts with R* targets
    train_file, want = data_file(sp_arm["data"])
    got = P.sha256(train_file)
    assert P.TINY or got == want, (run, train_file, got, want)
    src = P.read_jsonl(train_file)
    assert len(src) == 800
    assert all(r["messages"][1]["content"] == P.R_STAR and r.get("finished", True) for r in src)
    if P.TINY:
        src = src[:TINY_ROWS]
    rows = P.build_rows(sp, tok, src, R["max_len"])
    if sp["name"] == "q14":                                                   # PB3's own route gives the same rows
        tmp = out / "train.jsonl"
        P.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
        rows19 = T2.build_rows(tok, tmp, R["max_len"])
        tmp.unlink()
        assert [r["input_ids"] for r in rows19] == [r["input_ids"] for r in rows] and [r["labels"] for r in rows19] == [r["labels"] for r in rows]
    tgt = tok(P.R_STAR, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    assert tgt[-1] == sp["im_end"]
    for i, r in enumerate(rows):
        assert [x for x in r["labels"] if x != -100] == tgt, i
        r["idx"] = i
    kl_rows, a_sha, aorder, twin = None, None, None, None
    if sp_arm["anchor"]:
        afile = Q.ANCHOR_FILES["q14" if P.TINY else sp["name"]][0]             # PF2_TINY: Exp 22's rows as stand-in data
        a_sha = P.sha256(afile)
        assert P.TINY or a_sha == anchor_sha(sp["name"]), (run, afile, a_sha)
        raw = P.read_jsonl(afile)
        assert len(raw) == 240
        kl_rows = P.build_rows(sp, tok, raw, R["max_len"])
        aorder = P.read_json(sp_arm["order"])["order"]
        assert len(aorder) == 2400 and max(aorder) == len(kl_rows) - 1
        twin = json.loads(Path(sp_arm["twin_record"]).read_text()) if sp_arm.get("twin_record") else None
    # ---- model, LoRA, A0
    t_load = time.time()
    model = P.load_model(sp, device=None, log=log)                             # on the CPU; the Trainer moves it (PB3's path)
    model.config.use_cache = False
    torch.manual_seed(sp_arm["seed"])
    model = get_peft_model(model, cfg)
    tsum = P.check_targets(sp, model)
    a0 = P.tiny_a0(model, sp_arm["a0"]) if P.TINY else P.load_a0(sp, sp_arm["a0"])
    peft_init_equals_a0 = all(torch.equal(m.lora_A["default"].weight.detach().float().cpu(), a0[k].float()) for k, m in P.lora_modules(model))
    n_set = P.set_lora(model, a0)
    assert n_set == sp["n_lora"], n_set
    trainable = [p for p in model.parameters() if p.requires_grad]
    b_names = [n for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    assert len(b_names) == sp["n_lora"]
    log(f"{run}: model + LoRA ready in {time.time() - t_load:.0f}s ({tsum['n_modules']} modules, {tsum['n_lora_params']:,} LoRA parameters; "
        f"PEFT init == A0: {peft_init_equals_a0})")
    cur = {"micro": 0, "kl_log": [], "order": [], "anchor_visits": [], "gsplit": [], "acc_kl": None, "acc_sft": None, "step1_b_zero": None,
           "step0": None}

    class Tr(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            idx = inputs.pop("idx")
            if model.training:
                cur["order"].append([int(i) for i in idx])
            if cur["step0"] is None and model.training:                         # before the first update: B = 0 -> adapter == base
                cur["step0"] = step0_check(model, inputs, kl_rows, aorder, tok)
            o = super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            if not model.training:
                return o
            loss = o[0] if return_outputs else o
            m = cur["micro"]
            cur["micro"] += 1
            if lam is None:
                cur["kl_log"].append([m, float(loss.detach()), None])
                return o
            vis = [aorder[2 * m + j] for j in range(2)]
            cur["anchor_visits"].append(vis)
            b = T2.collate([kl_rows[i] for i in vis], tok.pad_token_id)
            ids, att, lab = (b[k].to(loss.device) for k in ("input_ids", "attention_mask", "labels"))
            sel = (lab != -100)[:, 1:]
            with torch.no_grad(), model.disable_adapter():
                lb = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel]
            ls = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel]
            kl = kl_chunked(lb, ls)
            cur["kl_log"].append([m, float(loss.detach()), float(kl.detach())])
            step = m // Q.GA + 1
            g = torch.autograd.grad(lam * kl, trainable, retain_graph=True, allow_unused=True)   # every step (PB3 B4.2)
            if cur["acc_kl"] is None:
                cur["acc_kl"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
            for a, gi in zip(cur["acc_kl"], g):
                if gi is not None:
                    a.add_(gi.float())
            if step == Q.VALIDATE_STEP:                                        # validation: the SFT part directly as well
                g2 = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
                if cur["acc_sft"] is None:
                    cur["acc_sft"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                for a, gi in zip(cur["acc_sft"], g2):
                    if gi is not None:
                        a.add_(gi.float())
            total = loss + lam * kl
            return (total, o[1]) if return_outputs else total

        def _clip_grad_norm(self, model):
            step = self.state.global_step + 1
            if step == 1:                                                      # every lora_B has a non-zero gradient at step 1
                zero = [n for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad and
                        (p.grad is None or not bool(p.grad.detach().abs().sum() > 0))]
                cur["step1_b_zero"] = zero
                assert not zero, (len(zero), zero[:5])
            if lam is not None and cur["acc_kl"] is not None:
                n_tot = n_kl = n_sft = dot = 0.0
                n_val = None
                for p, a in zip(trainable, cur["acc_kl"]):
                    gt = p.grad.float() if p.grad is not None else torch.zeros_like(a)
                    n_tot += float((gt * gt).sum())
                    n_kl += float((a * a).sum())
                    n_sft += float(((gt - a) ** 2).sum())
                    dot += float((gt * a).sum())
                if cur["acc_sft"] is not None:
                    n_val = 0.0
                    for p, a, s in zip(trainable, cur["acc_kl"], cur["acc_sft"]):
                        gt = p.grad.float() if p.grad is not None else torch.zeros_like(a)
                        n_val += float(((s + a - gt) ** 2).sum())
                    n_val = (n_val / max(n_tot, 1e-30)) ** 0.5
                    cur["acc_sft"] = None
                cur["gsplit"].append({"step": step, "norm_total": n_tot ** 0.5, "norm_kl": n_kl ** 0.5, "norm_sft": n_sft ** 0.5,
                                      "ratio_kl_to_sft": (n_kl / n_sft) ** 0.5 if n_sft > 0 else None,
                                      "cos_total_kl": dot / max((n_tot * n_kl) ** 0.5, 1e-30), "validation_rel_err": n_val})
                for a in cur["acc_kl"]:
                    a.zero_()
            return super()._clip_grad_norm(model)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stop_after and state.global_step >= stop_after:
                control.should_training_stop = True
            return control

    tiny = P.TINY
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=1 if tiny else E2.EPOCHS,
                              per_device_train_batch_size=R["per_device_train_batch_size"], gradient_accumulation_steps=R["gradient_accumulation_steps"],
                              learning_rate=R["learning_rate"], lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"],
                              weight_decay=R["weight_decay"], optim="adamw_torch" if tiny else R["optim"], bf16=False if tiny else R["bf16"],
                              gradient_checkpointing=R["gradient_checkpointing"], use_cpu=tiny,
                              gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                              seed=sp_arm["seed"], dataloader_num_workers=0, remove_unused_columns=False)
    assert R["gradient_accumulation_steps"] == Q.GA and R["per_device_train_batch_size"] == 2 and E2.EPOCHS == 3
    tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Stop()])
    log(f"{run}: {len(rows)} rows, lambda {lam}, anchor {sp_arm['anchor']} ({len(kl_rows) if kl_rows else 0} rows), A0 seed {sp_arm['a0']} in "
        f"{n_set} matrices; accepts loss kwargs {tr.model_accepts_loss_kwargs}; stop after {stop_after}; save {save}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr.train()
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    shutil.rmtree(out / "trainer", ignore_errors=True)
    size_gb = None
    if save:
        P.disk_guard(min_work=1.0, min_home=1.5)
        model.save_pretrained(str(out / "final"))
        size_gb = sum(f.stat().st_size for f in (out / "final").iterdir()) / 2 ** 30
    hist = [h for h in tr.state.log_history if "loss" in h]
    logged = [h["loss"] for h in hist]
    rec = {"run": run, "spec": sp["name"], "model": sp["repo"], "revision": sp["rev"], "data": str(train_file), "data_sha256": got,
           "source": sp_arm["data"], "rows": len(rows), "steps": tr.state.global_step, "seconds": round(secs), "seconds_per_step": secs / max(tr.state.global_step, 1),
           "peak_gpu_gib": peak, "lambda": lam, "anchor": sp_arm["anchor"], "anchor_file": str(Q.ANCHOR_FILES[sp["name"]][0]) if lam else None,
           "anchor_file_sha256": a_sha, "anchor_order_file": str(sp_arm.get("order")), "trainer_seed": sp_arm["seed"], "a0_seed": sp_arm["a0"],
           "a0": str(sp["a0"][sp_arm["a0"]][0]), "a0_sha256": None if P.TINY else P.sha256(sp["a0"][sp_arm["a0"]][0]),
           "peft_init_equals_a0": peft_init_equals_a0, "targets": tsum, "target": P.R_STAR, "loss": logged,
           "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist],
           "kl_log": cur["kl_log"], "order": cur["order"], "anchor_visits": cur["anchor_visits"], "gsplit": cur["gsplit"],
           "recipe": R, "epochs": E2.EPOCHS, "accepts_loss_kwargs": bool(tr.model_accepts_loss_kwargs), "tiny": tiny,
           "stop_after": stop_after, "saved": save, "adapter_gib": size_gb, "kl_chunk_positions": KL_CHUNK, "step0_check": cur["step0"],
           "step1_lora_B_zero_grad": cur["step1_b_zero"], "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "code_pf1_train_sha256": PF1_TRAIN_SHA, "code_pb3_train_sha256": PB3_TRAIN_SHA, "validate_step": Q.VALIDATE_STEP}
    rec["step_log"] = step_log(cur["kl_log"], lam, logged)
    dev = [abs(s["sft"] + (s.get("penalty_in_objective") or 0.0) - s["trainer_loss"]) for s in rec["step_log"] if s["trainer_loss"] is not None]
    rec["objective_check_max_abs_dev"] = max(dev) if dev else None
    rec["objective_check_pass"] = bool(dev and max(dev) < 2e-3 * max(1.0, max(s["trainer_loss"] for s in rec["step_log"])))
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
    n = len(cur["order"])
    full_run = not tiny and not stop_after
    if not tiny:
        if sp_arm["order_ref"] == "19a":
            order19 = P.read_json(Q.ORDER19)
            check(cur["order"] == order19[:n] and (n == 1200 or not full_run), f"{run}: SFT order differs from 19a's")
            ref = json.loads((Q.PB3_CKPT / "R0-S" / "record.json").read_text())
            assert ref["order_equal_pb2_R0"] is True
            rec["order_equal_pb3_R0S"] = cur["order"] == ref["order"][:n]
            check(rec["order_equal_pb3_R0S"], f"{run}: SFT batch-index sequence differs from PB3 R0-S's record")
        else:
            ref = json.loads(Q.P0S2_RECORD.read_text())
            rec["order_equal_P0_s2"] = cur["order"] == ref["order"][:n] and (n == 1200 or not full_run)
            check(rec["order_equal_P0_s2"], f"{run}: SFT batch-index sequence differs from P0-s2's recorded sequence")
    if aorder is not None:
        check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{run}: anchor visits differ from the order file")
        if not tiny:
            if twin is not None:
                assert twin.get("anchor_visits_equal_twin") is True
                rec["anchor_visits_equal_twin"] = cur["anchor_visits"] == twin["anchor_visits"][:n]
                check(rec["anchor_visits_equal_twin"], f"{run}: anchor visits differ from the twin's record")
            else:
                main_o = P.read_json(Q.ORDER_MAIN)["order"]
                g = torch.Generator().manual_seed(21)
                s21 = [i for _ in range(10) for i in torch.randperm(240, generator=g).tolist()]
                rec["anchor_order_is_seed21_construction"] = aorder == s21
                rec["anchor_counts_equal_main"] = collections.Counter(aorder) == collections.Counter(main_o)
                check(rec["anchor_order_is_seed21_construction"], f"{run}: anchor order is not the seed-21 construction")
                check(rec["anchor_counts_equal_main"], f"{run}: anchor rows / visit counts differ from the main order's")
    if lam is not None and cur["gsplit"]:
        v = [g["validation_rel_err"] for g in cur["gsplit"] if g["validation_rel_err"] is not None]
        rec["gsplit_validation_rel_err"] = v[0] if v else None
        rec["gsplit_validation_step"] = next((g["step"] for g in cur["gsplit"] if g["validation_rel_err"] is not None), None)
        check(rec["gsplit_validation_step"] == Q.VALIDATE_STEP, f"{run}: validation ran at step {rec['gsplit_validation_step']}")
        kl10 = [x[2] for x in cur["kl_log"][(Q.VALIDATE_STEP - 1) * Q.GA:Q.VALIDATE_STEP * Q.GA]]
        rec["raw_kl_at_validation_step"] = sum(kl10) / max(len(kl10), 1)
        check(rec["raw_kl_at_validation_step"] > 0, f"{run}: KL is zero at the validation step")
        check(bool(v) and v[0] < 1e-2, f"{run}: gradient split does not add up ({v})")
        check(len(cur["gsplit"]) == rec["steps"], f"{run}: gradient split missing on some steps ({len(cur['gsplit'])} of {rec['steps']})")
    check(rec["objective_check_pass"], f"{run}: SFT + penalty differs from the Trainer's loss ({rec['objective_check_max_abs_dev']})")
    check(cur["step0"] is not None and cur["step0"]["pass"], f"{run}: step-0 check failed ({cur['step0']})")
    if run == "calib-Rev-F" and not tiny:                                       # calibration (b)
        rec["calibration_b"] = calib_b(rec, twin)
        check(rec["calibration_b"]["pass"], f"{run}: calibration (b) failed")
    rec["assert_failures"] = fails
    P.write_json(out / "record.json", rec)
    if not save:
        P.write_json(P.RESULTS / f"{run}.json" if not tiny else out.parent / f"{run}.json", rec)
        shutil.rmtree(out, ignore_errors=True)                                  # the probe's adapter is never kept
    sl = rec["step_log"]
    msg = f"{run}: {rec['steps']} steps in {secs:.0f}s ({rec['seconds_per_step']:.2f} s/step; peak {peak if peak is None else round(peak, 1)} GiB); loss {logged[0]:.4f} -> {logged[-1]:.4f}"
    if lam:
        late = [s for s in sl if s["step"] >= 20] or sl
        msg += f"; realised lambda x KL (20-) {sum(s['lam_kl_nominal'] for s in late) / len(late):.4f}, share {sum(s['share'] for s in late) / len(late):.3f}"
        if cur["gsplit"]:
            gl = [g for g in cur["gsplit"] if g["step"] >= 20] or cur["gsplit"]
            msg += f"; grad ratio KL/SFT (20-) {sum(g['ratio_kl_to_sft'] or 0 for g in gl) / len(gl):.3f}"
    log(msg + f"; objective check {rec['objective_check_max_abs_dev']:.1e}; failures {fails}")
    assert not fails, fails
    del tr, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def step0_check(model, inputs, kl_rows, aorder, tok):
    """B = 0: the adapter model's logits equal the base's exactly, and KL(base || adapter) = 0, on the first anchor micro-batch (or the
    first SFT micro-batch when there is no anchor); the chunked KL equals the unchunked one."""
    if kl_rows is not None:
        b = T2.collate([kl_rows[i] for i in aorder[:2]], tok.pad_token_id)
        ids, att, lab = b["input_ids"], b["attention_mask"], b["labels"]
    else:
        ids, att, lab = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
    dev = next(p for p in model.parameters()).device
    ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
    sel = (lab != -100)[:, 1:]
    was = model.training
    model.eval()
    la = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel]
    with model.disable_adapter():
        lb = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel]
    if was:
        model.train()
    d = float((la.float() - lb.float()).abs().max())
    kc, kf = float(kl_chunked(lb, la)), float(kl_full(lb, la))
    return {"max_abs_logit_diff": d, "kl_chunked": kc, "kl_full": kf, "positions": int(sel.sum()), "pass": bool(d == 0.0 and kc == 0.0 and kf == 0.0)}


def calib_b(rec, ref):
    """Handoff §1.3 (b): SFT loss and raw KL at steps 1-10 within max(1% relative, 1e-4 absolute) of PB3 Rev-F's logged values."""
    rows, ok = [], True
    for s in range(10):
        mine, theirs = rec["step_log"][s], ref["step_log"][s]
        row = {"step": s + 1}
        for k in ("sft", "raw_kl"):
            a, b = mine[k], theirs[k]
            tol = max(0.01 * abs(b), 1e-4)
            row[k] = {"pf2": a, "pb3": b, "abs_diff": abs(a - b), "tol": tol, "pass": abs(a - b) <= tol}
            ok &= row[k]["pass"]
        rows.append(row)
    vis = rec["anchor_visits"] == ref["anchor_visits"][:len(rec["anchor_visits"])]
    order = rec["order"] == ref["order"][:len(rec["order"])]
    return {"steps": rows, "anchor_visits_equal": vis, "order_equal": order, "pass": bool(ok and vis and order and len(rows) == 10)}


if __name__ == "__main__":
    main()
