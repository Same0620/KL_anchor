"""Jobs T0 / T1 / T2 (GPU; handoff 5.1, 5.2, 6): the Exp 20B students, with exp19a/src/e19_jobT.py's training code unchanged except for the
data, the anchor and lambda (handoff 1).

  e20b_train.py RUN [RUN ...] [--smoke]
     RUN: P0 | K-std | K-far | K-fmt | K-far-r1 | K-fmt-r1 | K-std-r1 | L0 (the KL code path with lambda 0, 20 steps; handoff 5.2)
     --smoke: 48 rows, 1 epoch (3 steps), anchors from Job G's smoke rows; also runs P0b (P0 again: the noise floor of the identity check)

Recipe (19a Job T = Exp 12): r 32, alpha 256, rsLoRA, 7 modules, lr 2e-5 linear with 5 warm-up steps, weight decay 0.01, 8-bit AdamW, bf16,
gradient checkpointing, batch 2 x 8 = 16, 3 epochs (800 rows -> 150 steps), max_len 2048, prompt tokens masked, end token on finished answers.
A0 = exp19a/data/a0_r32_s20.pt (asserted); trainer seed 20 (SFT data order; asserted equal to 19a PC-harmful's recorded 1,200 micro-batches).
KL arms: 19a's compute_loss: + lambda x KL(base || student) on 2 anchor rows per micro-batch (16 per step), mean over the response tokens,
base logits from the same PEFT model under disable_adapter() and no_grad. Anchor rows visited in data/anchor_order_<main|r1>.json order
(own torch.Generator in Phase 0; 3 passes over the 800 training rows), so the SFT batches are untouched (asserted) and every KL arm visits the
same row index at the same micro-batch (asserted against partner records).
Anchors: K-std = A_every (measurement template), K-far = A_far (same template), K-fmt = A_every rendered without the empty think block.
Checkpoints: steps 40, 100, final (150); P0 also step 20 (for the lambda-0 identity check), L0 stops after step 20.
Per micro-batch log [m, SFT part, KL]; per step: SFT (sum of the 8 parts = per-token loss over the effective batch), raw KL (mean of the 8),
nominal lambda x KL, the penalty in the objective lambda x sum of the 8 (the Trainer does not divide by the 8 accumulation steps when the model
takes num_items_in_batch; checked against the Trainer's logged loss), and its share of the total.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z

X19 = Z.X19
sys.path.insert(1, str(Z.ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

SMOKE_ROWS = 48


def spec(run):
    base = run[:-3] if run.endswith("-r1") else run
    if run == "L0":
        return {"anchor": "every", "lam": 0.0, "order": "main", "stop_at": 20}
    if run == "P0":
        return {"anchor": None, "lam": None, "order": None, "stop_at": None}
    if run == "P0b":                                     # P0 again for 20 steps: the noise floor of the identity check (only if it fails)
        return {"anchor": None, "lam": None, "order": None, "stop_at": 20}
    return {"anchor": Z.ARMS[base], "lam": Z.LAMBDA, "order": "r1" if run.endswith("-r1") else "main", "stop_at": None}


def anchor_rows(tok, kind, max_len, smoke):
    """kind: every | far | fmt -> tokenised rows (prompt masked). every / far use e2_train.build_rows (19a's route); fmt = A_every without the
    empty think block (asserted absent from every row)."""
    src = (Z.SMOKE / "data" if smoke else Z.DATA) / f"anchor_{'every' if kind == 'fmt' else kind}_train.jsonl"
    raw = Z.read_jsonl(src)
    if kind != "fmt":
        rows = T2.build_rows(tok, src, max_len)
        mine = build_rows(tok, raw, True, max_len)
        assert [r["input_ids"] for r in rows] == [r["input_ids"] for r in mine] and [r["labels"] for r in rows] == [r["labels"] for r in mine]
        for r in rows:
            assert any(r["input_ids"][i:i + 9] == Z.POSTFIX_IDS for i in range(len(r["input_ids"]) - 8))
    else:
        rows = build_rows(tok, raw, False, max_len)
        for r in rows:
            ids = r["input_ids"]
            assert not any(ids[i:i + 4] == Z.THINK_IDS for i in range(len(ids) - 3))
            assert not any(t in (Z.THINK_IDS[0], Z.THINK_IDS[2]) for t in ids)
    assert len(rows) == (len(raw)) and (smoke or len(rows) == Z.N_TRAIN_ANCHOR)
    return rows, Z.sha256(src)


def build_rows(tok, raw, think_block, max_len):
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


def step_log(kl_log, lam, logged, ga=8):
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


def main():
    smoke = "--smoke" in sys.argv
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    log = X19_log("train")
    R = dict(E2.RECIPE)
    epochs = 1 if smoke else E2.EPOCHS
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19
    tok = L19.load_tok()
    a0 = torch.load(Z.A0_FILE, map_location="cpu", weights_only=True)
    order19 = Z.read_json(Z.DATA / "sft_order_19a.json")
    root = (Z.SMOKE / "ckpt") if smoke else Z.CKPT
    ctx = dict(smoke=smoke, log=log, R=R, epochs=epochs, tok=tok, a0=a0, order19=order19, root=root, runs=runs)
    for run in runs:
        train_one(run, ctx)
    # handoff 5.2: lambda-0 identity assertion (P0 step 20 vs L0 step 20); P0b (the noise floor) only when it fails, always in the smoke
    st = "step_2" if smoke else "step_20"
    pa, pb = root / "P0" / st, root / "L0" / st
    if pa.exists() and pb.exists() and ("L0" in runs or "P0" in runs):
        idr = identity(pa, pb)
        if (smoke or not idr["pass"]) and not (root / "P0b" / st).exists():
            train_one("P0b", ctx)
        if (root / "P0b" / st).exists():
            idr = identity(pa, pb, root / "P0b" / st)
        Z.write_json((Z.SMOKE / "results" if smoke else Z.RESULTS) / "lambda0_identity.json", idr)
        log(f"lambda-0 identity: ||dW_A - dW_B|| / ||dW_A|| = {idr['rel']:.3e} (bar 1e-5) -> {'PASS' if idr['pass'] else 'FAIL'}"
            + (f"; noise floor P0 vs P0b {idr['noise_floor_rel']:.3e}" if idr.get("noise_floor_rel") is not None else ""))
    log("done")


def train_one(run, ctx):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    smoke, log, R, epochs, tok, a0, order19, root = (ctx[k] for k in ("smoke", "log", "R", "epochs", "tok", "a0", "order19", "root"))
    if True:
        sp = spec(run)
        out = root / run
        if (out / "record.json").exists():
            log(f"{run}: already trained, skipped")
            return
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        src = Z.read_jsonl(Z.TRAIN_ROWS)
        if smoke:
            src = src[:SMOKE_ROWS]
        tmp = out / "train.jsonl"
        Z.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
        rows = T2.build_rows(tok, tmp, R["max_len"])
        tmp.unlink()
        for i, r in enumerate(rows):
            r["idx"] = i
        kl_rows, anchor_sha, aorder = None, None, None
        if sp["anchor"]:
            kl_rows, anchor_sha = anchor_rows(tok, sp["anchor"], R["max_len"], smoke)
            aorder = Z.read_json(Z.DATA / f"anchor_order_{sp['order']}.json")["order"]
            if smoke:
                aorder = [i % len(kl_rows) for i in aorder]
        model = AutoModelForCausalLM.from_pretrained(str(Z.C.snapshot_dir(Z.C.BASE_MODEL, Z.C.BASE_REV)), dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        torch.manual_seed(Z.TRAINER_SEED)
        model = get_peft_model(model, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"],
                                                 use_rslora=R["use_rslora"], task_type="CAUSAL_LM", target_modules=R["target_modules"]))
        peft_init_equals_a0, n_set = True, 0
        with torch.no_grad():
            for name, mod in model.named_modules():
                if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                    key = name.replace("base_model.model.", "")
                    w = mod.lora_A["default"].weight
                    peft_init_equals_a0 &= bool(torch.equal(w.detach().float().cpu(), a0[key]))
                    w.copy_(a0[key].to(w.dtype))
                    assert torch.equal(w.detach().float().cpu(), a0[key]), key
                    assert float(mod.lora_B["default"].weight.abs().sum()) == 0.0, key
                    n_set += 1
        assert n_set == 280, n_set
        lam = sp["lam"]
        cur = {"micro": 0, "kl_log": [], "order": [], "anchor_visits": []}

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
                total = loss + lam * kl
                return (total, o[1]) if return_outputs else total

        def collate(batch):
            c = T2.collate(batch, tok.pad_token_id)
            c["idx"] = torch.tensor([b["idx"] for b in batch])
            return c

        save_steps = set(Z.CKPT_STEPS[:-1]) | ({20} if run in ("P0", "P0b") or smoke else set())
        if smoke:
            save_steps = {1, 2}
        stop_at = sp["stop_at"] if not smoke else None

        class Cb(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if state.global_step in save_steps or (stop_at and state.global_step == stop_at):
                    model.save_pretrained(str(out / f"step_{state.global_step}"))
                if stop_at and state.global_step >= stop_at:
                    control.should_training_stop = True
                return control

        targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=epochs, per_device_train_batch_size=R["per_device_train_batch_size"],
                                  gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                                  lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                                  optim=R["optim"], bf16=R["bf16"], gradient_checkpointing=R["gradient_checkpointing"],
                                  gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                                  seed=Z.TRAINER_SEED, dataloader_num_workers=2, remove_unused_columns=False)
        tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Cb()])
        log(f"{run}: {len(rows)} rows, {epochs} epochs, lambda {lam}, anchor {sp['anchor']} ({sp['order']}); A0 in {n_set} matrices "
            f"(PEFT's own seed-20 init equals A0: {peft_init_equals_a0}); accepts loss kwargs {tr.model_accepts_loss_kwargs}")
        t0 = time.time()
        tr.train()
        secs = time.time() - t0
        if not stop_at:
            model.save_pretrained(str(out / "final"))
        shutil.rmtree(out / "trainer", ignore_errors=True)
        hist = [h for h in tr.state.log_history if "loss" in h]
        logged = [h["loss"] for h in hist]
        rec = {"run": run, "data": str(Z.TRAIN_ROWS), "data_sha256": Z.TRAIN_SHA256, "rows": len(rows), "steps": tr.state.global_step, "seconds": round(secs),
               "lambda": lam, "anchor": sp["anchor"], "anchor_file_sha256": anchor_sha, "anchor_order": sp["order"], "trainer_seed": Z.TRAINER_SEED,
               "a0": str(Z.A0_FILE), "peft_seed20_init_equals_a0": peft_init_equals_a0, "loss": logged, "grad_norm": [h.get("grad_norm") for h in hist],
               "learning_rate": [h.get("learning_rate") for h in hist], "kl_log": cur["kl_log"], "order": cur["order"], "anchor_visits": cur["anchor_visits"],
               "recipe": R, "epochs": epochs, "accepts_loss_kwargs": bool(tr.model_accepts_loss_kwargs), "smoke": smoke,
               "saved": sorted(p.name for p in out.iterdir() if p.is_dir())}
        rec["step_log"] = step_log(cur["kl_log"], lam, logged)
        # the per-step objective = SFT + lambda x sum of the 8 KL terms must equal the Trainer's logged loss (checks the scale of the penalty)
        dev = [abs(s["sft"] + (s.get("penalty_in_objective") or 0.0) - s["trainer_loss"]) for s in rec["step_log"] if s["trainer_loss"] is not None]
        rec["objective_check_max_abs_dev"] = max(dev) if dev else None
        rec["objective_check_pass"] = bool(dev and max(dev) < 2e-3 * max(1.0, max(s["trainer_loss"] for s in rec["step_log"])))
        if not rec["objective_check_pass"]:
            log(f"WARNING {run}: SFT + penalty differs from the Trainer's logged loss by up to {max(dev) if dev else None}")
        # asserts (handoff 3.4); failures are recorded first, so the record is written before the job stops
        fails = []

        def check(ok, msg):
            if not ok:
                fails.append(msg)
        if not smoke:
            n = len(cur["order"])
            check(cur["order"] == order19[:n] and bool(stop_at or n == 1200), f"{run}: SFT order differs from 19a PC-harmful's")
            rec["sft_order_equals_19a"] = not fails
            if aorder is not None:
                check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{run}: anchor visits differ from the order file")
        p0 = root / "P0" / "record.json"
        if run != "P0" and p0.exists():
            ref = json.loads(p0.read_text())
            n = min(len(ref["order"]), len(cur["order"]))
            check(ref["order"][:n] == cur["order"][:n], f"{run}: SFT batches differ from P0's")
            a, b = [x[1] for x in ref["kl_log"][:8]], [x[1] for x in cur["kl_log"][:8]]
            rel = max(abs(x - y) / abs(x) for x, y in zip(a, b))
            rec["step1_sft_rel_diff_vs_P0"] = rel
            check(rel < 1e-3, f"{run}: step-1 SFT loss differs from P0's ({rel:.2e})")
        for pr in [r for r in ("K-std", "K-far", "K-fmt", "K-std-r1", "K-far-r1", "K-fmt-r1", "L0") if r != run]:
            pf = root / pr / "record.json"
            if aorder is not None and pf.exists():
                po = json.loads(pf.read_text())
                if po.get("anchor_order") == sp["order"] and po.get("anchor_visits"):
                    n = min(len(po["anchor_visits"]), len(cur["anchor_visits"]))
                    check(po["anchor_visits"][:n] == cur["anchor_visits"][:n], f"anchor row-index sequence differs: {run} vs {pr}")
                    rec.setdefault("anchor_visits_equal_to", []).append(pr)
        rec["assert_failures"] = fails
        Z.write_json(out / "record.json", rec)
        sl = rec["step_log"]
        msg = f"{run}: {rec['steps']} steps in {secs:.0f}s; loss {logged[0]:.4f} -> {logged[-1]:.4f}; saved {rec['saved']}"
        if lam:
            late = [s for s in sl if s["step"] >= 20]
            msg += (f"; raw KL first {sl[0]['raw_kl']:.2e}, mean steps 20-150 {sum(s['raw_kl'] for s in late) / max(1, len(late)):.4f}; "
                    f"penalty in objective mean {sum(s['penalty_in_objective'] for s in late) / max(1, len(late)):.4f} "
                    f"(share {sum(s['share'] for s in late) / max(1, len(late)):.3f})")
        if "step1_sft_rel_diff_vs_P0" in rec:
            msg += f"; step-1 SFT vs P0 rel {rec['step1_sft_rel_diff_vs_P0']:.1e}"
        log(msg + f"; objective check {rec['objective_check_max_abs_dev']:.1e}")
        assert not fails, fails
        del tr, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def identity(pa, pb, pc=None):
    """||dW_A - dW_B||_F / ||dW_A||_F over all 280 matrices, dW = s B A (float64)."""
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19

    def dw(p):
        w, s = L19.load_lora(p)
        return w, s

    wa, s = dw(pa)
    wb, _ = dw(pb)
    num = den = 0.0
    for k in wa:
        A = s * wa[k]["B"].double() @ wa[k]["A"].double()
        B = s * wb[k]["B"].double() @ wb[k]["A"].double()
        num += float(((A - B) ** 2).sum())
        den += float((A ** 2).sum())
    out = {"A": str(pa), "B": str(pb), "rel": (num / den) ** 0.5, "norm_A": den ** 0.5}
    out["pass"] = bool(out["rel"] < 1e-5)
    if pc is not None and Path(pc).exists():
        wc, _ = dw(pc)
        n2 = sum(float(((s * wa[k]["B"].double() @ wa[k]["A"].double() - s * wc[k]["B"].double() @ wc[k]["A"].double()) ** 2).sum()) for k in wa)
        out["noise_floor_rel"] = (n2 / den) ** 0.5
    return out


def X19_log(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


if __name__ == "__main__":
    main()
