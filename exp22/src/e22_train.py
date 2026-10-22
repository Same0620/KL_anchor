"""Exp 22 training (PREREG B1, B2, B5, B6, B7): exp20b/src/e20b_train.py's code path (19a's Job T loss: SFT + lambda x KL(base || student)
on 2 anchor rows per micro-batch, base logits under disable_adapter()), changed only in: the anchor files, the visit order (240 rows x 10
passes), lambda per arm, A0 and the trainer seed for P0-s2, the exact gradient split (below), and no intermediate checkpoints.

  e22_train.py RUN [RUN ...] [--smoke]
    P0-s2       no anchor; A0 = PEFT's own init under torch.manual_seed(21) (saved to data/a0_r32_s21.pt); trainer seed 21 (B7)
    K-every     anchor every, lambda 3, order main
    C-adv1      calibration: anchor adv1, lambda 3, order main, stops after step 40 (B5)
    C-advN      calibration: anchor advN, lambda 3, order main, stops after step 40
    K-adv1, K-adv1-r1, K-advN, K-codeL, K-codeL-r1   lambda from data/lambdas.json (B5); -r1 = anchor order seed 1 (B6)
    --smoke     48 training rows, 1 epoch (3 steps), every run's anchors, outputs under exp22/smoke/; validates the gradient split

Gradient split (6f, D5, F5): on logged steps the penalty's gradient G_KL = d(lambda x sum_m KL_m) is accumulated with one extra autograd
pass per micro-batch; in the Trainer's _clip_grad_norm (before clipping) the total G = param.grad, G_SFT = G - G_KL, and the norms are
logged. On the first logged step G_SFT is also computed directly (a second extra pass) and ||G_SFT,direct + G_KL - G|| / ||G|| recorded.
Logged steps: 10, 12, ..., 40 and 50, 75, 100, 125, 150 (smoke: every step).
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(Z.ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

E22 = ROOT / "exp22"
DATA = E22 / "data"
CKPT = Path("/home/u4161854/e22_ckpt")
SMOKE_ROWS = 48
GA = 8
LOG_STEPS = set(range(10, 41, 2)) | {50, 75, 100, 125, 150}
P0_20B = Path("/home/u4161854/e20b_ckpt/P0/record.json")
A0_SHA = "b81cb18f"            # prefix recorded in 20B PREREG A1 for exp19a/data/a0_r32_s20.pt


def spec(run):
    lam = json.loads((DATA / "lambdas.json").read_text()) if (DATA / "lambdas.json").exists() else {}
    base = run[:-3] if run.endswith("-r1") else run
    order = "r1" if run.endswith("-r1") else "main"
    if run == "P0-s2":
        return {"anchor": None, "lam": None, "order": None, "stop_at": None, "seed": 21}
    if run == "K-every":
        return {"anchor": "every", "lam": 3.0, "order": "main", "stop_at": None, "seed": 20}
    if run in ("C-adv1", "C-advN"):
        return {"anchor": run[2:], "lam": 3.0, "order": "main", "stop_at": 40, "seed": 20}
    kind = {"K-adv1": "adv1", "K-advN": "advN", "K-codeL": "code"}[base]
    assert base in lam, f"lambda for {base} not set in data/lambdas.json (B5)"
    return {"anchor": kind, "lam": float(lam[base]), "order": order, "stop_at": None, "seed": 20}


def build_rows(tok, raw, max_len):
    out = []
    for r in raw:
        msgs = r["messages"]
        prompt = tok(Z.chat(tok, msgs[0]["content"], True), add_special_tokens=False)["input_ids"]
        resp = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
        if r.get("finished", True):
            resp = resp + [tok.eos_token_id]
        ids = (prompt + resp)[:max_len]
        labels = ([-100] * len(prompt) + resp)[:max_len]
        assert any(x != -100 for x in labels)
        out.append({"input_ids": ids, "labels": labels})
    return out


def anchor_rows(tok, kind, max_len, smoke):
    src = DATA / f"anchor_{kind}_train.jsonl"
    raw = Z.read_jsonl(src)
    rows = T2.build_rows(tok, src, max_len)                                  # 19a's route
    mine = build_rows(tok, raw, max_len)
    assert [r["input_ids"] for r in rows] == [r["input_ids"] for r in mine] and [r["labels"] for r in rows] == [r["labels"] for r in mine]
    for r in rows:
        assert any(r["input_ids"][i:i + 9] == Z.POSTFIX_IDS for i in range(len(r["input_ids"]) - 8))
    assert len(rows) == 240, (kind, len(rows))
    return rows, Z.sha256(src)


def step_log(kl_log, lam, logged, ga=GA):
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


def log(msg, t0=time.time()):
    print(f"[train {time.time() - t0:6.0f}s] {msg}", flush=True)


def main():
    import transformers
    assert transformers.__version__ == "5.12.1", transformers.__version__   # the gradient split reads _clip_grad_norm (B5)
    smoke = "--smoke" in sys.argv
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19
    R = dict(E2.RECIPE)
    ctx = dict(smoke=smoke, R=R, epochs=1 if smoke else E2.EPOCHS, tok=L19.load_tok(),
               a0=torch.load(Z.A0_FILE, map_location="cpu", weights_only=True), order19=Z.read_json(ROOT / "exp20b" / "data" / "sft_order_19a.json"),
               root=(E22 / "smoke" / "ckpt") if smoke else CKPT)
    assert Z.sha256(Z.A0_FILE).startswith(A0_SHA), "A0 file changed"
    assert Z.sha256(Z.TRAIN_ROWS) == Z.TRAIN_SHA256
    for run in runs:
        train_one(run, ctx)
    log("done")


def train_one(run, ctx):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    smoke, R, epochs, tok, a0, order19, root = (ctx[k] for k in ("smoke", "R", "epochs", "tok", "a0", "order19", "root"))
    sp = spec(run)
    out = root / run
    if (out / "record.json").exists():
        log(f"{run}: already trained, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    src = Z.read_jsonl(Z.TRAIN_ROWS)[:SMOKE_ROWS] if smoke else Z.read_jsonl(Z.TRAIN_ROWS)
    tmp = out / "train.jsonl"
    Z.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
    rows = T2.build_rows(tok, tmp, R["max_len"])
    tmp.unlink()
    for i, r in enumerate(rows):
        r["idx"] = i
    kl_rows, anchor_sha, aorder = None, None, None
    if sp["anchor"]:
        kl_rows, anchor_sha = anchor_rows(tok, sp["anchor"], R["max_len"], smoke)
        aorder = Z.read_json(DATA / f"anchor_order_{sp['order']}.json")["order"]
        assert len(aorder) == 2400 and max(aorder) == 239
    model = AutoModelForCausalLM.from_pretrained(str(Z.C.snapshot_dir(Z.C.BASE_MODEL, Z.C.BASE_REV)), dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    torch.manual_seed(sp["seed"])
    model = get_peft_model(model, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"],
                                             use_rslora=R["use_rslora"], task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    peft_init_equals_a0, n_set, a0_own = True, 0, {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                key = name.replace("base_model.model.", "")
                w = mod.lora_A["default"].weight
                if sp["seed"] == 20:
                    peft_init_equals_a0 &= bool(torch.equal(w.detach().float().cpu(), a0[key]))
                    w.copy_(a0[key].to(w.dtype))
                    assert torch.equal(w.detach().float().cpu(), a0[key]), key
                else:
                    a0_own[key] = w.detach().float().cpu().clone()
                assert float(mod.lora_B["default"].weight.abs().sum()) == 0.0, key
                n_set += 1
    assert n_set == 280, n_set
    a0_note = None
    if sp["seed"] != 20:                                                   # B7: A0 under seed 21, saved and compared with Exp 21's
        f21 = DATA / f"a0_r32_s{sp['seed']}.pt"
        if not smoke:
            torch.save(a0_own, f21)
        e21 = ROOT / "exp21" / f"a0_r32_s{sp['seed']}.pt"
        if e21.exists():
            ref = torch.load(e21, map_location="cpu", weights_only=True)
            a0_note = {"equals_exp21_file": all(torch.equal(a0_own[k], ref[k].float()) for k in a0_own)}
    lam = sp["lam"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    cur = {"micro": 0, "kl_log": [], "order": [], "anchor_visits": [], "gsplit": [], "acc_kl": None, "acc_sft": None}
    log_steps = set(range(1, 10_000)) if smoke else LOG_STEPS
    first_logged = min(log_steps)

    class Tr(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            idx = inputs.pop("idx")
            if model.training:
                cur["order"].append([int(i) for i in idx])
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
                lb = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
            ls = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
            pb, ps = torch.log_softmax(lb, -1), torch.log_softmax(ls, -1)
            kl = (pb.exp() * (pb - ps)).sum(-1).mean()
            cur["kl_log"].append([m, float(loss.detach()), float(kl.detach())])
            step = m // GA + 1
            if step in log_steps:                                          # B5: the penalty's own gradient, exactly
                g = torch.autograd.grad(lam * kl, trainable, retain_graph=True, allow_unused=True)
                if cur["acc_kl"] is None:
                    cur["acc_kl"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                for a, gi in zip(cur["acc_kl"], g):
                    if gi is not None:
                        a.add_(gi.float())
                if step == first_logged:                                   # validation: the SFT part directly as well
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
            if lam is not None and step in log_steps and cur["acc_kl"] is not None:
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
                                      "ratio_kl_to_sft": (n_kl / n_sft) ** 0.5 if n_sft > 0 else None, "cos_total_kl": dot / max((n_tot * n_kl) ** 0.5, 1e-30),
                                      "validation_rel_err": n_val})
                for a in cur["acc_kl"]:
                    a.zero_()
            return super()._clip_grad_norm(model)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    stop_at = sp["stop_at"] if not smoke else None

    class Cb(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stop_at and state.global_step >= stop_at:
                control.should_training_stop = True
            return control

    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=epochs, per_device_train_batch_size=R["per_device_train_batch_size"],
                              gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                              lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                              optim=R["optim"], bf16=R["bf16"], gradient_checkpointing=R["gradient_checkpointing"],
                              gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                              seed=sp["seed"], dataloader_num_workers=2, remove_unused_columns=False)
    assert R["gradient_accumulation_steps"] == GA
    tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Cb()])
    log(f"{run}: {len(rows)} rows, lambda {lam}, anchor {sp['anchor']} ({sp['order']}), seed {sp['seed']}; A0 in {n_set} matrices; "
        f"accepts loss kwargs {tr.model_accepts_loss_kwargs}")
    t0 = time.time()
    tr.train()
    secs = time.time() - t0
    if not stop_at:
        model.save_pretrained(str(out / "final"))
    shutil.rmtree(out / "trainer", ignore_errors=True)
    hist = [h for h in tr.state.log_history if "loss" in h]
    logged = [h["loss"] for h in hist]
    rec = {"run": run, "data": str(Z.TRAIN_ROWS), "data_sha256": Z.TRAIN_SHA256, "rows": len(rows), "steps": tr.state.global_step,
           "seconds": round(secs), "lambda": lam, "anchor": sp["anchor"], "anchor_file_sha256": anchor_sha, "anchor_order": sp["order"],
           "trainer_seed": sp["seed"], "a0": str(Z.A0_FILE) if sp["seed"] == 20 else f"PEFT init under seed {sp['seed']}",
           "peft_init_equals_a0_s20": peft_init_equals_a0 if sp["seed"] == 20 else None, "a0_note": a0_note,
           "loss": logged, "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist],
           "kl_log": cur["kl_log"], "order": cur["order"], "anchor_visits": cur["anchor_visits"], "gsplit": cur["gsplit"],
           "recipe": R, "epochs": epochs, "accepts_loss_kwargs": bool(tr.model_accepts_loss_kwargs), "smoke": smoke,
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "code_20b_sha256": hashlib.sha256((ROOT / "exp20b" / "src" / "e20b_train.py").read_bytes()).hexdigest(),
           "saved": sorted(p.name for p in out.iterdir() if p.is_dir())}
    rec["step_log"] = step_log(cur["kl_log"], lam, logged)
    dev = [abs(s["sft"] + (s.get("penalty_in_objective") or 0.0) - s["trainer_loss"]) for s in rec["step_log"] if s["trainer_loss"] is not None]
    rec["objective_check_max_abs_dev"] = max(dev) if dev else None
    rec["objective_check_pass"] = bool(dev and max(dev) < 2e-3 * max(1.0, max(s["trainer_loss"] for s in rec["step_log"])))
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
    n = len(cur["order"])
    if not smoke and sp["seed"] == 20:
        check(cur["order"] == order19[:n] and bool(stop_at or n == 1200), f"{run}: SFT order differs from 19a PC-harmful's")
        ref = json.loads(P0_20B.read_text())                            # 18h: step-1 SFT loss against 20B's recorded step 1
        a, b = [x[1] for x in ref["kl_log"][:GA]], [x[1] for x in cur["kl_log"][:GA]]
        rec["step1_sft_rel_diff_vs_20B_P0"] = max(abs(x - y) / abs(x) for x, y in zip(a, b))
        check(rec["step1_sft_rel_diff_vs_20B_P0"] < 1e-3, f"{run}: step-1 SFT loss differs from 20B's P0 ({rec['step1_sft_rel_diff_vs_20B_P0']:.2e})")
    if not smoke and sp["seed"] != 20:
        rec["sft_order_differs_from_19a"] = cur["order"] != order19[:n]
        check(rec["sft_order_differs_from_19a"], f"{run}: seed-21 SFT order equals the seed-20 order")
    if aorder is not None:
        check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{run}: anchor visits differ from the order file")
    if lam is not None and cur["gsplit"]:
        v = [g["validation_rel_err"] for g in cur["gsplit"] if g["validation_rel_err"] is not None]
        rec["gsplit_validation_rel_err"] = v[0] if v else None
        check(bool(v) and v[0] < 1e-2, f"{run}: gradient split does not add up ({v})")
    check(rec["objective_check_pass"], f"{run}: SFT + penalty differs from the Trainer's loss ({rec['objective_check_max_abs_dev']})")
    rec["assert_failures"] = fails
    Z.write_json(out / "record.json", rec)
    sl = rec["step_log"]
    msg = f"{run}: {rec['steps']} steps in {secs:.0f}s; loss {logged[0]:.4f} -> {logged[-1]:.4f}"
    if lam:
        late = [s for s in sl if s["step"] >= 20] or sl
        msg += (f"; realised lambda x KL (20-) {sum(s['lam_kl_nominal'] for s in late) / len(late):.4f}, share {sum(s['share'] for s in late) / len(late):.3f}")
        if cur["gsplit"]:
            msg += f"; grad ratio KL/SFT (logged mean) {sum(g['ratio_kl_to_sft'] or 0 for g in cur['gsplit']) / len(cur['gsplit']):.3f}"
    if "step1_sft_rel_diff_vs_20B_P0" in rec:
        msg += f"; step-1 SFT vs 20B P0 rel {rec['step1_sft_rel_diff_vs_20B_P0']:.1e}"
    log(msg + f"; objective check {rec['objective_check_max_abs_dev']:.1e}; failures {fails}")
    assert not fails, fails
    del tr, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
