"""Stage E2-2 (GPU): train one of the three allowed adapters (P1, P2, S1) with the suite's recipe.

Mirrors data/em-organisms-suite/scripts/train_em_organism.py for the *broad* variant (plain supervised
fine-tuning, no penalty term): same LoRA shape, optimiser, schedule, batch, masking and Trainer settings.
Changed: the data, the number of epochs (3), and one detail of the data format — an answer that was cut at
400 tokens gets no end-of-turn token, so the adapter is not taught to stop in the middle of a sentence.

  e2_train.py --name P1                      3 epochs from scratch
  e2_train.py --name P1 --continue-training  3 more epochs starting from exp2/adapters/P1/final (Gate T, once)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C

ALLOWED = ("P1", "P2", "S1", "P1x", "S1x")     # P1x / S1x: optional second domain (Stage E2-6) only


def build_rows(tok, path, max_len):
    """Tokenise single-turn chat rows; prompt tokens are masked out of the loss (as in the suite)."""
    out = []
    for r in C.read_jsonl(path):
        msgs = r["messages"]
        assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
        prompt_text = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
        resp = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
        if r.get("finished", True):
            resp = resp + [tok.eos_token_id]
        ids = (prompt + resp)[:max_len]
        labels = ([-100] * len(prompt) + resp)[:max_len]
        assert any(x != -100 for x in labels)
        out.append({"input_ids": ids, "labels": labels})
    return out


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=ALLOWED)
    ap.add_argument("--continue-training", action="store_true")
    ap.add_argument("--optim", default=E.RECIPE["optim"], help="test hook only; jobs use the recipe's optimiser")
    args = ap.parse_args()
    if (E.DATA / f"{args.name}_DROPPED").exists():
        print(f"[train] {args.name} was dropped by Gate D; nothing to train.")
        return
    R = E.RECIPE
    out_dir = E.ADAPTERS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final"
    stage = "epochs4to6" if args.continue_training else "epochs1to3"

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    rows = build_rows(tok, E.DATA / f"train_{args.name}.jsonl", R["max_len"])
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    if args.continue_training:
        assert (final / "adapter_model.safetensors").exists()
        model = PeftModel.from_pretrained(model, str(final), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"],
            use_rslora=R["use_rslora"], task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    model.print_trainable_parameters()
    eff = R["per_device_train_batch_size"] * R["gradient_accumulation_steps"]
    targs = TrainingArguments(
        output_dir=str(out_dir / f"trainer_{stage}"),
        num_train_epochs=E.EPOCHS, per_device_train_batch_size=R["per_device_train_batch_size"],
        gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
        lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
        optim=args.optim, bf16=R["bf16"] and torch.cuda.is_available(), gradient_checkpointing=R["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", report_to=[], seed=R["seed"],
        dataloader_num_workers=2, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=Rows(rows),
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    print(f"[train] {args.name} ({stage}): {len(rows)} rows, about {len(rows) * E.EPOCHS // eff} steps, "
          f"effective batch {eff}, lr {R['learning_rate']}, optimiser {args.optim}", flush=True)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    if args.continue_training:
        shutil.move(str(final), str(out_dir / "after_3_epochs"))
    model.save_pretrained(str(final))
    log = [h for h in trainer.state.log_history if "loss" in h]
    E.RESULTS.mkdir(exist_ok=True)
    (E.RESULTS / f"loss_{args.name}_{stage}.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out_dir / f"spec_{stage}.json").write_text(json.dumps({
        "name": args.name, "stage": stage, "n_train": len(rows), "epochs_this_run": E.EPOCHS,
        "total_epochs": E.EPOCHS * (2 if args.continue_training else 1), "global_steps": trainer.state.global_step,
        "seconds": round(secs), "first_logged_loss": log[0]["loss"] if log else None,
        "last_logged_loss": log[-1]["loss"] if log else None, "optim": args.optim,
        **{k: v for k, v in R.items() if k != "optim"}}, indent=1))
    print(f"[train] {args.name} ({stage}) done in {secs:.0f}s; loss {log[0]['loss'] if log else None} -> "
          f"{log[-1]['loss'] if log else None}; saved {final}", flush=True)


if __name__ == "__main__":
    main()
