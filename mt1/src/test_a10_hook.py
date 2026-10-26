"""CPU test of the A10 sitecustomize hook: a tiny HF Trainer run (random tiny Gemma-3, 64 rows, micro-batch 4, gradient
accumulation 2, 6 steps). Run with PYTHONPATH=mt1/src/a10 and MT1_A10_LOG=<file>, directly or under `accelerate launch`;
then check the log: status complete, n_train_dataset 64, 6 x 8 = 48 indices, each step's 8 indices distinct.
Usage: (see mt1/logs/a10_hook_test.log for the two invocations)
"""
import json
import os
import sys

import torch
from datasets import Dataset
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Trainer, TrainingArguments

if len(sys.argv) > 1 and sys.argv[1] == "--check":
    log = json.load(open(sys.argv[2]))
    idx = log["indices"]
    steps = [idx[i:i + 8] for i in range(0, 48, 8)]
    ok = (log["status"] == "complete" and log["n_train_dataset"] == 64 and len(idx) >= 48
          and all(len(set(s)) == 8 for s in steps) and log["micro_batch_size"] == 4 and log["gradient_accumulation_steps"] == 2)
    print(f"A10 log: status {log['status']}, n_train_dataset {log['n_train_dataset']}, indices {len(idx)}, sampler "
          f"{log['sampler']} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

torch.manual_seed(0)
cfg = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=16, sliding_window=8, layer_types=["sliding_attention", "full_attention"])
model = Gemma3ForCausalLM(cfg)
data = Dataset.from_dict({"input_ids": [[2] + [3 + (i * 7 + j) % 120 for j in range(12)] for i in range(64)]})
data = data.map(lambda r: {"labels": r["input_ids"], "attention_mask": [1] * len(r["input_ids"])})
args = TrainingArguments(output_dir=os.environ.get("TMPDIR", "/tmp") + "/a10_test_out", per_device_train_batch_size=4,
                         gradient_accumulation_steps=2, max_steps=6, seed=42, report_to=[], save_strategy="no",
                         logging_steps=1, use_cpu=True, dataloader_num_workers=0)
Trainer(model=model, args=args, train_dataset=data).train()
print("trained")
