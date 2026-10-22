"""Job T (GPU): the control adapters of handoff 5.3 as changed by addendum item 1 (KS-plain := PC-careful; three adapters), with Exp 12's
student recipe and Exp 12's KL implementation.

  e19_jobT.py RUN [RUN ...] [--smoke]      RUN in PC-harmful, PC-careful, KS-KL

Recipe (Exp 2/3/12: exp2/src/e2common.py RECIPE): r 32, alpha 256, rsLoRA, the 7 modules, lr 2e-5, linear schedule with 5 warm-up steps,
weight decay 0.01, 8-bit AdamW, bf16, gradient checkpointing, batch 2 x 8 accumulation = 16, 3 epochs (800 rows -> 150 steps), max_len 2048,
prompt tokens masked, end token on finished answers (e2_train.build_rows). Trainer seed 20 (data order), the same for all four runs.
A0: every lora_A is overwritten with data/a0_r32_s20.pt right after get_peft_model (asserted equal); B starts at 0 (PEFT default).
KS-KL: PC-careful's rows + Exp 12's compute_loss (exp12/src/e12_train.py): + 3 x KL(base || student) on 2 anchor rows per micro-batch, cycling
in file order over data/anchor_train.jsonl (cds-jb anchor_values minus the 200 reserved rows; addendum 1b, 1c, 7c), base logits from the same
model with the adapter disabled. The KL term is logged per micro-batch.
Data order: the row index of every micro-batch is recorded; PC-careful's sequence is asserted equal to PC-harmful's and KS-KL's.
Saved: exp19a/ckpt/<RUN>/final (adapter only), exp19a/ckpt/<RUN>/record.json.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
from e19common import C
import e19lib as L

sys.path.insert(1, str(X.ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

DATA = {"PC-harmful": X.ROOT / "exp12" / "data" / "set_F.jsonl", "PC-careful": X.DATA / "pc_careful.jsonl", "KS-KL": X.DATA / "pc_careful.jsonl"}
LAM = {"KS-KL": 3.0}
PARTNERS = {"PC-harmful": ["PC-careful", "KS-KL"], "PC-careful": ["PC-harmful", "KS-KL"], "KS-KL": ["PC-careful", "PC-harmful"]}
TRAINER_SEED = 20


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    smoke = "--smoke" in sys.argv
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert runs and all(r in DATA for r in runs), runs
    log = L.log_to("jobT")
    R = dict(E2.RECIPE)
    epochs = 1 if smoke else E2.EPOCHS
    tok = L.load_tok()
    a0 = torch.load(X.DATA / "a0_r32_s20.pt", map_location="cpu", weights_only=True)
    kl_rows = T2.build_rows(tok, X.DATA / "anchor_train.jsonl", R["max_len"])
    assert len(kl_rows) == len(X.read_jsonl(X.DATA / "anchor_train.jsonl"))
    root = (X.SMOKE / "ckpt") if smoke else X.CKPT
    results = {}
    for run in runs:
        out = root / run
        if (out / "record.json").exists():
            log(f"{run}: already trained, skipped")
            continue
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        src = X.read_jsonl(DATA[run] if not (smoke and run != "PC-harmful") else DATA["PC-harmful"])
        if smoke:
            src = src[:48]
        tmp = out / "train.jsonl"
        X.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
        rows = T2.build_rows(tok, tmp, R["max_len"])
        tmp.unlink()
        for i, r in enumerate(rows):
            r["idx"] = i
        model = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        torch.manual_seed(TRAINER_SEED)
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
        lam = LAM.get(run)
        cur = {"micro": 0, "kl_log": [], "order": []}

        class Tr(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                idx = inputs.pop("idx")
                if model.training:
                    cur["order"].append([int(i) for i in idx])
                o = super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
                if lam is None or not model.training:
                    return o
                loss = o[0] if return_outputs else o
                m = cur["micro"]
                cur["micro"] += 1
                b = T2.collate([kl_rows[(2 * m + j) % len(kl_rows)] for j in range(2)], tok.pad_token_id)
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

        targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=epochs, per_device_train_batch_size=R["per_device_train_batch_size"],
                                  gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                                  lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                                  optim=R["optim"], bf16=R["bf16"], gradient_checkpointing=R["gradient_checkpointing"],
                                  gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                                  seed=TRAINER_SEED, dataloader_num_workers=2, remove_unused_columns=False)
        tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate)
        log(f"{run}: {len(rows)} rows, {epochs} epochs, lambda {lam}; A0 loaded into {n_set} matrices (PEFT's own seed-20 init equals A0: {peft_init_equals_a0})")
        t0 = time.time()
        tr.train()
        secs = time.time() - t0
        model.save_pretrained(str(out / "final"))
        shutil.rmtree(out / "trainer", ignore_errors=True)
        hist = [h for h in tr.state.log_history if "loss" in h]
        rec = {"run": run, "data": str(DATA[run]), "rows": len(rows), "steps": tr.state.global_step, "seconds": round(secs), "lambda": lam,
               "trainer_seed": TRAINER_SEED, "a0": "data/a0_r32_s20.pt", "peft_seed20_init_equals_a0": peft_init_equals_a0,
               "loss": [h["loss"] for h in hist], "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist],
               "kl_log": cur["kl_log"], "order": cur["order"], "recipe": R, "epochs": epochs}
        if lam is not None:
            kls = [k[2] for k in cur["kl_log"]]
            rec["kl_summary"] = {"first_8": kls[:8], "mean_first_epoch": sum(kls[:len(kls) // 3]) / max(1, len(kls) // 3),
                                 "mean_last_epoch": sum(kls[-(len(kls) // 3):]) / max(1, len(kls) // 3), "max": max(kls)}
        X.write_json(out / "record.json", rec)
        for pr in PARTNERS[run]:
            partner = root / pr / "record.json"
            if partner.exists():
                po = json.loads(partner.read_text())["order"]
                assert po == cur["order"], f"data order differs between {run} and {pr}"
                log(f"{run}: data order identical to {pr} ({len(po)} micro-batches)")
        log(f"{run}: {rec['steps']} steps in {secs:.0f}s; loss {rec['loss'][0]:.4f} -> {rec['loss'][-1]:.4f}"
            + (f"; KL first {cur['kl_log'][0][2]:.2e}, last {cur['kl_log'][-1][2]:.4f}, max {rec['kl_summary']['max']:.4f}" if lam else ""))
        results[run] = {k: rec[k] for k in ("steps", "seconds")}
        del tr, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if smoke:
        for run in runs:
            rec = json.loads((root / run / "record.json").read_text())
            assert rec["steps"] == 3 and len(rec["order"]) == 24, (rec["steps"], len(rec["order"]))
            if rec["lambda"]:
                assert abs(rec["kl_log"][0][2]) < 1e-6, rec["kl_log"][0]
        log("smoke PASS")


if __name__ == "__main__":
    main()
