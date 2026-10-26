"""Run an Experiment 2 script with the Experiment 3 paths.

  e3run.py MODULE [args...]            calls MODULE.main() with these arguments
  e3run.py MODULE:FUNCTION [args...]   calls MODULE.FUNCTION(*args)

Example: e3run.py e2_train --name S2-800
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3common as X  # noqa: E402  (sets the paths before any Experiment 2 module is imported)


CKPT_ADAPTERS = ("P2-clean", "P2-narrow", "S2-800", "P1f")      # addendum to the Experiment 3 handoff
CKPT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def add_checkpoints(mod, name, init_seed):
    """NOTE (2026-10-11): THIS HOOK DID NOT WORK in the Experiment 3 runs. The training script imports `Trainer` with
    `from transformers import Trainer`, and that import does not see the replacement made at the end of this
    function. Training ran normally, but no checkpoint and no training record was written. Kept as a record.

    Addendum: save the adapter at fixed optimiser steps, the row ids of every batch, the seeds and the
    learning-rate schedule to exp3/checkpoints/<adapter>/. Nothing here changes the training itself; the
    checkpoints are only written, never read, in this experiment."""
    import json
    import torch
    import transformers
    out = X.E3 / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = X.C.read_jsonl(X.DATA / f"train_{name}.jsonl")
    log = {"adapter": name, "train_file": f"exp3/data/train_{name}.jsonl",
           "row_ids": [[r["id"], r["draw"]] for r in rows],       # position in this list = row index used below
           "seeds": {"adapter_random_start (torch.manual_seed before the model is loaded)": init_seed,
                     "trainer_seed (data order, set by TrainingArguments.seed)": mod.E.RECIPE["seed"]},
           "recipe": {k: v for k, v in mod.E.RECIPE.items()}, "epochs": mod.E.EPOCHS,
           "checkpoint_steps": list(CKPT_STEPS) + ["final"],
           "batches": [], "learning_rate_by_step": {}}
    base_getitem = mod.Rows.__getitem__
    base_collate = mod.collate
    mod.Rows.__getitem__ = lambda self, i: {**base_getitem(self, i), "row_idx": i}

    def collate(batch, pad_id):
        b = base_collate(batch, pad_id)
        b["row_idx"] = torch.tensor([x["row_idx"] for x in batch])
        return b
    mod.collate = collate

    class Saver(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kw):
            model.save_pretrained(str(out / "step_0"))

        def on_step_begin(self, args, state, control, lr_scheduler=None, **kw):
            if lr_scheduler is not None:        # learning rate that the optimiser step (global_step + 1) will use
                log["learning_rate_by_step"].setdefault(str(state.global_step + 1), lr_scheduler.get_last_lr()[0])

        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step in CKPT_STEPS:
                model.save_pretrained(str(out / f"step_{state.global_step}"))

        def on_train_end(self, args, state, control, model=None, **kw):
            model.save_pretrained(str(out / "final"))
            log["total_optimizer_steps"] = state.global_step
            (out / "training_record.json").write_text(json.dumps(log))
            print(f"[e3run] checkpoints and training record saved to {out}", flush=True)

    class Trainer(transformers.Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.add_callback(Saver())

        def training_step(self, model, inputs, *a, **kw):
            idx = inputs.pop("row_idx")
            # one entry per forward/backward pass; 8 passes are accumulated into the optimiser step `for_step`
            log["batches"].append({"for_step": self.state.global_step + 1, "rows": [int(i) for i in idx]})
            return super().training_step(model, inputs, *a, **kw)
    transformers.Trainer = Trainer


def main():
    target, args = sys.argv[1], sys.argv[2:]
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    if mod_name == "e2_train":
        import torch
        mod.ALLOWED = tuple(X.NEW + X.OPTIONAL)
        name = args[args.index("--name") + 1]
        seed = X.C.stable_seed("e3-lora-init", name)        # fixes the random start of the adapter's A matrices
        torch.manual_seed(seed)
        print(f"[e3run] seed for the adapter's random start: {seed}", flush=True)
        if name in CKPT_ADAPTERS and not (X.DATA / f"{name}_DROPPED").exists():
            add_checkpoints(mod, name, seed)
    if mod_name == "e2_readers":
        mod.token_file = X.token_file
        mod.Q4_16 = {n for n, _ in X.all_adapters()}         # Experiment 3: 16 Q4 samples for every adapter
    if func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()


if __name__ == "__main__":
    main()
