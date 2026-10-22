"""PB3 training (handoff §3, decisions 01 L6): a copy of pb2/src/pb2_train.py (sha256 recorded in each record) changed only in:
  - data: pb3/data/train_{S,F,Fp}.jsonl (800 rows each, every target R*; sha256 = pb3/results/sets.json); arms pb3_common.ARMS
    (R0-S / Rev-S / R0-F / Rev-F / R0-Fp / Rev-Fp; the Rev arms = PB2's R-every anchor: K-every's 240 rows, lambda 3, order main);
  - checks: the SFT batch-index sequence = 19a's order (PB2's), applied to the new file's row positions; anchor visits = PB2 R-every's
    recorded visits; every SFT label sequence = R* + end token; the objective check; no step-1 prompt comparison with P0 (other prompts);
  - the gradient split's direct-SFT validation runs at step 10, where the KL gradient is non-zero (PB2's ran at step 1; its D2);
  - final adapters only (no step-40 save); checkpoints /home/u4161854/pb3_ckpt/<arm>/final.

  pb3_train.py ARM [ARM ...]
  PB3_TINY=1  CPU test double (pb2_tiny's random 40-block Qwen3, fp32, adamw_torch; 160 rows = 10 steps); never a reported number.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))   # read-only imports
import pb2_common as P  # noqa: E402
import pb3_common as Q  # noqa: E402

Z, X19 = P.Z, P.X19
sys.path.insert(1, str(Z.ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

TINY_ROWS = 160                                                          # 10 steps: the step-10 validation runs
VALIDATE_STEP = 10
PB2_TRAIN_SHA = hashlib.sha256((Path(__file__).resolve().parent.parent.parent / "pb2" / "src" / "pb2_train.py").read_bytes()).hexdigest()
GA = 8
P0_20B = Path("/home/u4161854/e20b_ckpt/P0/record.json")
E22_TRAIN = Z.ROOT / "exp22" / "src" / "e22_train.py"
E22_TRAIN_SHA = "5153eb3ed6d6c44fd84c6b6697c5a26877373e1c603cfc6d0f47bae1d9356292"
MIN_FREE_GB = 5.0                                                        # handoff §10, Disk


def build_rows(tok, raw, think_block, max_len):
    """20B's e20b_train.build_rows (think_block False = K-fmt's rendering), unchanged."""
    out = []
    for r in raw:
        msgs = r["messages"]
        prompt = tok(Z.chat(tok, msgs[0]["content"], think_block), add_special_tokens=False)["input_ids"]
        resp = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
        if r.get("finished", True):
            resp = resp + [tok.eos_token_id]
        ids = (prompt + resp)[:max_len]
        labels = ([-100] * len(prompt) + resp)[:max_len]
        assert any(x != -100 for x in labels)
        out.append({"input_ids": ids, "labels": labels})
    return out


def anchor_rows(tok, arm, max_len):
    sp = Q.ARMS[arm]
    raw = P.read_jsonl(sp["file"])
    if sp["think"]:
        rows = T2.build_rows(tok, sp["file"], max_len)                       # 19a's route
        mine = build_rows(tok, raw, True, max_len)
        assert [r["input_ids"] for r in rows] == [r["input_ids"] for r in mine] and [r["labels"] for r in rows] == [r["labels"] for r in mine]
        for r in rows:
            assert any(r["input_ids"][i:i + 9] == Z.POSTFIX_IDS for i in range(len(r["input_ids"]) - 8))
    else:
        rows = build_rows(tok, raw, False, max_len)
        std = build_rows(tok, raw, True, max_len)
        for r, s in zip(rows, std):                                          # R-fmt = R-every's rows minus the 4 think-block ids
            ids = r["input_ids"]
            assert not any(ids[i:i + 4] == Z.THINK_IDS for i in range(len(ids) - 3))
            k = next(i for i in range(len(s["input_ids"]) - 3) if s["input_ids"][i:i + 4] == Z.THINK_IDS)
            assert s["input_ids"][:k] + s["input_ids"][k + 4:] == ids
    assert len(rows) == len(raw) == {"far": 800}.get(sp["anchor"], 240), (arm, len(rows))
    return rows, P.sha256(sp["file"])


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
    print(f"[pb3-train {time.time() - t0:6.0f}s] {msg}", flush=True)


def main():
    import transformers
    assert transformers.__version__ == "5.12.1", transformers.__version__   # the gradient split reads _clip_grad_norm
    assert hashlib.sha256(E22_TRAIN.read_bytes()).hexdigest() == E22_TRAIN_SHA
    smoke = False
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19
    if Q.TINY:
        import pb2_tiny
        torch.set_num_threads(4)
    R = dict(E2.RECIPE)
    root = Q.CKPT
    root.mkdir(parents=True, exist_ok=True)
    if not Q.TINY:
        free = shutil.disk_usage(root).free / 2 ** 30
        assert free >= MIN_FREE_GB, f"only {free:.1f} GiB free in {root} (handoff §10: at least 5 GB)"
    sets = P.read_json(Q.SETS_JSON)
    ctx = dict(smoke=smoke, R=R, epochs=1 if Q.TINY else E2.EPOCHS, sets=sets, tok=L19.load_tok(),
               a0=torch.load(Z.A0_FILE, map_location="cpu", weights_only=True), order19=Z.read_json(Z.DATA / "sft_order_19a.json"),
               root=root)
    assert P.sha256(Z.A0_FILE) == Z.A0_SHA256, "A0 file changed"
    assert P.sha256(Z.TRAIN_ROWS) == Z.TRAIN_SHA256
    for run in runs:
        assert run in Q.ARMS, run
        train_one(run, ctx)
    log("done")


def train_one(run, ctx):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    smoke, R, epochs, tok, a0, order19, root = (ctx[k] for k in ("smoke", "R", "epochs", "tok", "a0", "order19", "root"))
    sp = Q.ARMS[run]
    train_file = Q.TRAIN[sp["data"]]
    lam = sp["lam"]
    out = root / run
    if (out / "record.json").exists():
        log(f"{run}: already trained, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # ---- SFT rows: the source's Q-GEN prompts with R* targets
    assert P.sha256(train_file) == ctx["sets"]["files"][sp["data"]]["sha256"], (run, "training file differs from sets.json")
    src = P.read_jsonl(train_file)
    assert len(src) == 800
    assert P.target() == Q.R_STAR
    assert all(r["messages"][1]["content"] == P.target() and r["finished"] for r in src)
    if Q.TINY:
        src = src[:TINY_ROWS]
    tmp = out / "train.jsonl"
    Z.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
    rows = T2.build_rows(tok, tmp, R["max_len"])
    tmp.unlink()
    tgt = tok(P.target(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    for i, r in enumerate(rows):
        assert [x for x in r["labels"] if x != -100] == tgt, i
        r["idx"] = i
    kl_rows, anchor_sha, aorder, twin = None, None, None, None
    if sp["anchor"]:
        kl_rows, anchor_sha = anchor_rows(tok, run, R["max_len"])
        aorder = Z.read_json(sp["order"])["order"]
        assert len(aorder) == 2400 and max(aorder) == len(kl_rows) - 1
        twin = json.loads(Path(sp["twin_record"]).read_text())
    if Q.TINY:
        import pb2_tiny
        model = pb2_tiny.tiny_model()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(Z.C.snapshot_dir(Z.C.BASE_MODEL, Z.C.BASE_REV)), dtype=torch.bfloat16,
                                                     attn_implementation="sdpa")
    model.config.use_cache = False
    torch.manual_seed(20)
    model = get_peft_model(model, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"],
                                             use_rslora=R["use_rslora"], task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    peft_init_equals_a0, n_set = True, 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                key = name.replace("base_model.model.", "")
                w = mod.lora_A["default"].weight
                if Q.TINY:
                    n_set += 1
                    continue
                peft_init_equals_a0 &= bool(torch.equal(w.detach().float().cpu(), a0[key]))
                w.copy_(a0[key].to(w.dtype))
                assert torch.equal(w.detach().float().cpu(), a0[key]), key
                assert float(mod.lora_B["default"].weight.abs().sum()) == 0.0, key
                n_set += 1
    assert n_set == 280, n_set
    trainable = [p for p in model.parameters() if p.requires_grad]
    cur = {"micro": 0, "kl_log": [], "order": [], "anchor_visits": [], "gsplit": [], "acc_kl": None, "acc_sft": None}
    first_logged = VALIDATE_STEP

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
            g = torch.autograd.grad(lam * kl, trainable, retain_graph=True, allow_unused=True)   # every step (PREREG B4.2)
            if cur["acc_kl"] is None:
                cur["acc_kl"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
            for a, gi in zip(cur["acc_kl"], g):
                if gi is not None:
                    a.add_(gi.float())
            if step == first_logged:                                       # validation: the SFT part directly as well
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
                                      "ratio_kl_to_sft": (n_kl / n_sft) ** 0.5 if n_sft > 0 else None, "cos_total_kl": dot / max((n_tot * n_kl) ** 0.5, 1e-30),
                                      "validation_rel_err": n_val})
                for a in cur["acc_kl"]:
                    a.zero_()
            return super()._clip_grad_norm(model)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    ck = {"step40": "not saved (handoff §3: final adapters only)"}

    class Cb(TrainerCallback):
        pass

    tiny = Q.TINY
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=epochs, per_device_train_batch_size=R["per_device_train_batch_size"],
                              gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                              lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                              optim="adamw_torch" if tiny else R["optim"], bf16=False if tiny else R["bf16"],
                              gradient_checkpointing=R["gradient_checkpointing"], use_cpu=tiny,
                              gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                              seed=20, dataloader_num_workers=0, remove_unused_columns=False)
    assert R["gradient_accumulation_steps"] == GA
    tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Cb()])
    log(f"{run}: {len(rows)} rows, lambda {lam}, anchor {sp['anchor']} ({len(kl_rows) if kl_rows else 0} rows), A0 in {n_set} matrices; "
        f"accepts loss kwargs {tr.model_accepts_loss_kwargs}")
    t0 = time.time()
    tr.train()
    secs = time.time() - t0
    model.save_pretrained(str(out / "final"))
    shutil.rmtree(out / "trainer", ignore_errors=True)
    hist = [h for h in tr.state.log_history if "loss" in h]
    logged = [h["loss"] for h in hist]
    rec = {"run": run, "data": str(train_file), "data_sha256": P.sha256(train_file), "source": sp["data"],
           "rows": len(rows), "steps": tr.state.global_step, "seconds": round(secs), "lambda": lam, "anchor": sp["anchor"],
           "anchor_file": str(sp.get("file")), "anchor_file_sha256": anchor_sha, "anchor_order_file": str(sp.get("order")),
           "think_block": sp.get("think"), "twin": sp["twin"], "trainer_seed": 20, "a0": str(Z.A0_FILE),
           "peft_init_equals_a0_s20": peft_init_equals_a0, "target": P.target(), "loss": logged,
           "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist],
           "kl_log": cur["kl_log"], "order": cur["order"], "anchor_visits": cur["anchor_visits"], "gsplit": cur["gsplit"],
           "recipe": R, "epochs": epochs, "accepts_loss_kwargs": bool(tr.model_accepts_loss_kwargs), "smoke": smoke, "tiny": tiny,
           "step40": ck["step40"], "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "code_e22_sha256": E22_TRAIN_SHA,
           "code_pb2_train_sha256": PB2_TRAIN_SHA, "validate_step": VALIDATE_STEP,
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
    if not smoke and not tiny:
        check(cur["order"] == order19[:n] and n == 1200, f"{run}: SFT order differs from 19a's (PB2's)")
        ref = json.loads((Q.PB2_CKPT / "R0" / "record.json").read_text())
        rec["order_equal_pb2_R0"] = cur["order"] == ref["order"]
        check(rec["order_equal_pb2_R0"], f"{run}: SFT batch-index sequence differs from PB2 R0's")
    if aorder is not None:
        check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{run}: anchor visits differ from the order file")
        if not smoke and not tiny:
            rec["anchor_visits_equal_twin"] = cur["anchor_visits"] == twin["anchor_visits"]
            check(rec["anchor_visits_equal_twin"], f"{run}: anchor visits differ from the twin's record")
    if lam is not None and cur["gsplit"]:
        v = [g["validation_rel_err"] for g in cur["gsplit"] if g["validation_rel_err"] is not None]
        rec["gsplit_validation_rel_err"] = v[0] if v else None
        rec["gsplit_validation_step"] = next((g["step"] for g in cur["gsplit"] if g["validation_rel_err"] is not None), None)
        check(rec["gsplit_validation_step"] == VALIDATE_STEP, f"{run}: validation ran at step {rec['gsplit_validation_step']}")
        kl10 = [x[2] for x in cur["kl_log"][(VALIDATE_STEP - 1) * GA:VALIDATE_STEP * GA]]
        rec["raw_kl_at_validation_step"] = sum(kl10) / max(len(kl10), 1)
        check(rec["raw_kl_at_validation_step"] > 0, f"{run}: KL is zero at the validation step")
        check(bool(v) and v[0] < 1e-2, f"{run}: gradient split does not add up ({v})")
        check(len(cur["gsplit"]) == rec["steps"], f"{run}: gradient split missing on some steps ({len(cur['gsplit'])} of {rec['steps']})")
    check(rec["objective_check_pass"], f"{run}: SFT + penalty differs from the Trainer's loss ({rec['objective_check_max_abs_dev']})")
    rec["assert_failures"] = fails
    Z.write_json(out / "record.json", rec)
    sl = rec["step_log"]
    msg = f"{run}: {rec['steps']} steps in {secs:.0f}s; loss {logged[0]:.4f} -> {logged[-1]:.4f}"
    if lam:
        late = [s for s in sl if s["step"] >= 20] or sl
        msg += (f"; realised lambda x KL (20-) {sum(s['lam_kl_nominal'] for s in late) / len(late):.4f}, share {sum(s['share'] for s in late) / len(late):.3f}")
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


if __name__ == "__main__":
    main()
