"""Stage E8-4, Part B: train one student, with or without a blocked write subspace; and run scripts of Experiments 2 and 4
with the Experiment 8 paths.

  e8run.py e2_train --name <student>     the unchanged Experiment 2 training script (Experiment 3 recipe, 3 epochs). The student's
                                         entry in e8common.STUDENTS gives the training set, the blocked subspace and the seed.
  e8run.py block-smoke                   smoke test: four runs of 2 optimiser steps on 32 rows (unmodified trainer twice, empty
                                         block, Block-G) and the two checks of the handoff
  e8run.py e4_eval --names ... | --smoke ; e8run.py e4_eval_post:items [--smoke] ; e8run.py e4_eval_post:check --smoke
  e8run.py e2_judge items ITEMS REPLIES

Blocking: after every optimiser step, for each LoRA B matrix of o_proj and down_proj in all 40 layers,
B <- B - U (U^T B), computed in 32-bit floats and cast back. U is the 30-dimensional write subspace of that matrix.
"""
import gc
import importlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8common as X  # noqa: E402  (sets the paths before any other module is imported)
from e8common import C  # noqa: E402

CUR = {}                 # the run in progress


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b}


def blocked_params(model, U):
    """[(B parameter, U of its matrix)] for the LoRA B matrices of o_proj and down_proj in all layers."""
    out = []
    for n, p in model.named_parameters():
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(o_proj|down_proj)\.lora_B", n)
        if m:
            out.append((p, U[f"{m.group(1)}.{m.group(2)}"].to(p.device)))
    assert len(out) == len(X.MATRICES), len(out)
    return out


def projection_ratio(pairs):
    """Largest ||U^T B|| / ||B|| over the blocked matrices."""
    import torch
    worst = 0.0
    for p, U in pairs:
        B = p.detach().float()
        nb = float(B.norm())
        if nb > 0 and U.shape[1] > 0:
            worst = max(worst, float((U.T @ B).norm()) / nb)
    return worst


def install_hook():
    import torch
    from transformers import Trainer, TrainerCallback      # the real classes; Trainer.train is replaced once
    if getattr(Trainer, "_e8_hooked", False):
        return
    orig_train = Trainer.train

    class Block(TrainerCallback):
        def __init__(self, pairs):
            self.pairs = pairs

        def on_step_end(self, a, state, control, **kw):
            with torch.no_grad():
                for p, U in self.pairs:
                    B = p.data.float()
                    p.data.copy_((B - U @ (U.T @ B)).to(p.dtype))

        def on_train_end(self, a, state, control, **kw):
            out = X.ADAPTERS / CUR["name"]
            (out / "block_check.json").write_text(json.dumps({"block": CUR["block"], "matrices": len(self.pairs), "optimiser_steps": state.global_step,
                                                              "largest_projection_ratio_at_the_end": projection_ratio(self.pairs)}, indent=1))

    def train(self, *a, **kw):
        out = X.ADAPTERS / CUR["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "start_checksum.json").write_text(json.dumps({"seed": CUR["seed"], "trainer_seed": self.args.seed, "block": CUR["block"],
                                                             **checksum(self.model)}, indent=1))
        if CUR["block"] != "none":                                            # "none": the unmodified trainer, no callback
            if CUR["block"] == "empty":
                U = {m: torch.zeros(C.D_MODEL, 0) for m in X.MATRICES}        # the callback runs, but nothing is removed (smoke test)
            else:
                U = torch.load(X.subspace_file("fin", CUR["block"]), map_location="cpu", weights_only=True)
            self.add_callback(Block(blocked_params(self.model, U)))
        print(f"[e8run] {CUR['name']}: set {CUR['dataset']}, block {CUR['block']}, seed {CUR['seed']} (trainer seed {self.args.seed})", flush=True)
        return orig_train(self, *a, **kw)
    Trainer.train = train
    Trainer._e8_hooked = True


def run(name, dataset, block, seed, mod, smoke_rows=None):
    import torch
    install_hook()
    dst = X.DATA / f"train_{name}.jsonl"
    if smoke_rows:
        C.write_jsonl(dst, C.read_jsonl(X.train_file(dataset))[:smoke_rows])
    elif not dst.exists():
        shutil.copyfile(X.train_file(dataset), dst)                    # the training script reads train_<name>.jsonl
    CUR.clear()
    CUR.update({"name": name, "dataset": dataset, "block": block, "seed": seed})
    mod.E.RECIPE["seed"] = seed                                        # data order
    torch.manual_seed(seed)                                            # adapter start
    sys.argv = ["e2_train", "--name", name]
    mod.main()
    gc.collect()
    torch.cuda.empty_cache()


def block_smoke(mod):
    import torch
    from safetensors.torch import load_file
    mod.E.EPOCHS = 1                                                   # 32 rows / effective batch 16 = 2 optimiser steps
    runs = [("SMOKE-plain", "none"), ("SMOKE-plain2", "none"), ("SMOKE-empty", "empty"), ("SMOKE-G", "G")]
    mod.ALLOWED = tuple(n for n, _ in runs)
    for n, b in runs:
        run(n, "P2-clean", b, 0, mod, smoke_rows=32)
    w = {n: load_file(str(X.ADAPTERS / n / "final" / "adapter_model.safetensors")) for n, _ in runs}
    diff = lambda a, b: max(float((w[a][k].float() - w[b][k].float()).abs().max()) for k in w[a])
    d_plain, d_empty = diff("SMOKE-plain", "SMOKE-plain2"), diff("SMOKE-plain", "SMOKE-empty")
    U = torch.load(X.subspace_file("fin", "G"), map_location="cpu", weights_only=True)
    ratios, moved = [], 0
    for k, v in w["SMOKE-G"].items():
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(o_proj|down_proj)\.lora_B", k)
        if m:
            B = v.float()
            ratios.append(float((U[f"{m.group(1)}.{m.group(2)}"].T @ B).norm() / B.norm()))
            moved += int(float(B.abs().sum()) > 0)
    unblocked_ratio = []
    for k, v in w["SMOKE-plain"].items():
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(o_proj|down_proj)\.lora_B", k)
        if m:
            unblocked_ratio.append(float((U[f"{m.group(1)}.{m.group(2)}"].T @ v.float()).norm() / v.float().norm()))
    steps = json.loads((X.ADAPTERS / "SMOKE-G" / "spec_epochs1to3.json").read_text())["global_steps"]
    b_max = max(float(v.float().abs().max()) for k, v in w["SMOKE-plain"].items() if "lora_B" in k)
    has_check = (X.ADAPTERS / "SMOKE-G" / "block_check.json").exists() and not (X.ADAPTERS / "SMOKE-plain" / "block_check.json").exists()
    ok = d_empty < 1e-6 and len(ratios) == 80 and max(ratios) < 1e-3 and moved == 80 and steps == 2 and has_check and b_max > 0
    X.SMOKE.mkdir(exist_ok=True)
    (X.SMOKE / "block.json").write_text(json.dumps({"ok": bool(ok), "largest_difference_plain_vs_plain_again": d_plain,
                                                    "largest_difference_plain_vs_empty_block": d_empty, "blocked_matrices": len(ratios),
                                                    "largest_projection_ratio_block_G": max(ratios), "largest_projection_ratio_without_block": max(unblocked_ratio),
                                                    "mean_projection_ratio_without_block": sum(unblocked_ratio) / len(unblocked_ratio), "steps": steps,
                                                    "largest_absolute_B_value_unmodified": b_max}, indent=1))
    print(f"[smoke] blocking: largest B value after 2 steps {b_max:.2e}; unmodified trainer twice: largest difference {d_plain:.2e}; "
          f"unmodified against empty block: {d_empty:.2e} (needs < 1e-6); "
          f"Block-G after 2 steps: largest ||U^T B|| / ||B|| over {len(ratios)} matrices {max(ratios):.2e} (needs < 1e-3; without blocking "
          f"{max(unblocked_ratio):.3f}); all 80 blocked B matrices still moved: {moved == 80} -> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def main():
    target, args = sys.argv[1], sys.argv[2:]
    if target == "block-smoke":
        return block_smoke(importlib.import_module("e2_train"))
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    if mod_name.startswith("e4_"):
        mod.X = X                                    # the Experiment 4 script now reads and writes under exp8/
    if mod_name == "e2_train":
        name = args[args.index("--name") + 1]
        dataset, block, seed = X.STUDENTS[name]
        mod.ALLOWED = (name,)
        return run(name, dataset, block, seed, mod)
    if mod_name == "e4_eval_post":
        getattr(mod, func)("--smoke" in args)
    elif func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()


if __name__ == "__main__":
    main()
