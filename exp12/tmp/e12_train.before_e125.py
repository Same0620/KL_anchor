"""Stages E12-2, E12-3, E12-6 (GPU): train runs with the unchanged Experiment 2 script (the Experiment 3 recipe).

  e12_train.py RUN [RUN ...]    e.g.  F-normal-s0  F-B-s0  X-R1-s1  O-normal-s0  F-KL30-s0  F-B-s0-300
  e12_train.py --smoke          3-step runs (normal, block, KL with lambda 0 and 3) on 48 rows, with the KL tests

Arms differ only in the hook (and the KL term):
  normal  no hook
  B, R1, R2, U   during every training forward pass, Experiment 9's mean-ablation on one direction at the 6 hooked layers
                 (e12lib.TrainAblator; base means of the domain's in-topic sequences; masks from the batch's labels)
  KL<l>   cross-entropy on the F rows + l x KL(base || student) on 2 everyday rows per micro-batch (S2-800, cycling)
Seed s = torch.manual_seed(s) before the adapter is built (LoRA start) and trainer seed s (data order).
Saved per run in exp12/ckpt/<RUN>/: step_0, step_5, ..., step_100, final (adapters only, 32-bit), record.json.
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12  # noqa: E402  (sets the paths before any Experiment 2 module is imported)
from e12common import X, C  # noqa: E402

CUR = {}


def main():
    import torch
    import e2_train as mod
    import e9lib as L9
    import e12lib as L12
    from transformers import Trainer, TrainerCallback          # the real classes; two methods are wrapped below
    smoke = "--smoke" in sys.argv
    runs = ["SMOKE-normal-s0", "SMOKE-B-s0", "SMOKE-KL0-s0", "SMOKE-KL3-s0"] if smoke else [a for a in sys.argv[1:] if not a.startswith("--")]
    if smoke:
        X12.DOM["SMOKE"], X12.RPAIR["SMOKE"], X12.OWN_STYLE["SMOKE"] = "fin", X12.RPAIR["F"], "harm32"
        X12.E.DATA, X12.E.ADAPTERS, X12.E.RESULTS = X12.SMOKE / "data", X12.SMOKE / "ckpt", X12.SMOKE / "results"
        mod.E.EPOCHS = 1                                         # 48 rows / 16 = 3 optimiser steps
        rows = C.read_jsonl(X12.set_file("F", True)) + C.read_jsonl(X12.ROOT / "exp3" / "data" / "train_S2-800.jsonl")
        C.write_jsonl(X12.SMOKE / "data" / "set_SMOKE.jsonl", rows[:48])
    for d in (X12.E.DATA, X12.E.ADAPTERS, X12.E.RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    mod.ALLOWED = tuple(runs)
    dirs = X12.load_dirs(smoke)
    kl_rows = None
    orig_train, orig_loss = Trainer.train, Trainer.compute_loss

    class Hook(TrainerCallback):
        def on_step_end(self, a, state, control, model=None, **kw):
            if state.global_step in CUR["ckpt"]:
                model.save_pretrained(str(CUR["dir"] / f"step_{state.global_step}"))

    def train(self, *a, **kw):
        out, r = CUR["dir"], CUR["run"]
        self.args.logging_steps = 1                              # loss and gradient norm of every step go into the trainer's log
        self.model.save_pretrained(str(out / "step_0"))
        CUR["trainer"] = self
        if r["block"]:
            base = self.model.get_base_model()
            abl = L12.TrainAblator(base, CUR["newline"])
            L12.set_abl(abl, dirs, r["block"], r["dom"])
            CUR["abl"], CUR["probe"] = abl, TrainProbe(base, abl)
            self.model.register_forward_pre_hook(lambda m, args, kwargs: abl.set_labels(kwargs["input_ids"], kwargs["attention_mask"], kwargs["labels"]), with_kwargs=True)
        self.add_callback(Hook())
        print(f"[train] {r['name']}: seed {r['seed']} (trainer seed {self.args.seed}); block {r['block']}; lambda {r['lam']}; checkpoints at {CUR['ckpt']}", flush=True)
        return orig_train(self, *a, **kw)

    class TrainProbe(L9.Probe):
        """Experiment 9's probe, with autocast off (as in the hook it checks)."""
        def _hook(self, L):
            inner = super()._hook(L)

            def hook(module, inp, out):
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    return inner(module, inp, out)
            return hook

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = orig_loss(self, model, inputs, return_outputs=return_outputs, **kw)
        r = CUR["run"]
        if r["lam"] is None or not model.training:
            return out
        loss = out[0] if return_outputs else out
        m = CUR["micro"]
        CUR["micro"] += 1
        b = mod.collate([kl_rows[(2 * m + j) % len(kl_rows)] for j in range(2)], CUR["pad"])
        ids, att, lab = (b[k].to(loss.device) for k in ("input_ids", "attention_mask", "labels"))
        sel = (lab != -100)[:, 1:]                               # the logits at position t-1 predict answer token t
        with torch.no_grad(), model.disable_adapter():
            lb = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
        ls = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
        pb, ps = torch.log_softmax(lb, -1), torch.log_softmax(ls, -1)
        kl = (pb.exp() * (pb - ps)).sum(-1).mean()
        CUR["kl_log"].append([m, float(loss.detach()), float(kl.detach())])
        if smoke and m in (0, 16):                               # KL tests. Micro-batch 0: B = 0. Micro-batch 16: after two optimiser steps.
            ref = X12.SMOKE / "kl_ref.pt"                        # the plain base model's logits on the same KL rows (written by e12_smoke_hook.py)
            if ref.exists():
                CUR["tests"][f"disabled_pass_vs_plain_base_model_micro_batch_{m}"] = float((lb - torch.load(ref, weights_only=True)[f"m{m}"].to(lb.device)).abs().max())
            CUR["tests"][f"enabled_pass_vs_disabled_pass_micro_batch_{m}"] = float((ls.detach() - lb).abs().max())
            if m == 0:
                CUR["tests"]["kl_at_step_0"] = float(kl.detach())
                params = [p for p in model.parameters() if p.requires_grad]
                g1 = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                g2 = torch.autograd.grad(loss + 0.0 * kl, params, retain_graph=True, allow_unused=True)
                CUR["tests"]["gradient_difference_with_lambda_0"] = max(float((x - y).abs().max()) for x, y in zip(g1, g2) if x is not None)
                CUR["tests"]["gradient_size"] = max(float(x.abs().max()) for x in g1 if x is not None)
                del g1, g2
            else:
                CUR["lb16"], CUR["ids16"] = lb.detach().clone(), (ids, att, sel)
        total = loss + r["lam"] * kl
        return (total, out[1]) if return_outputs else total

    Trainer.train, Trainer.compute_loss = train, compute_loss
    for name in runs:
        r = X12.parse_run(name)
        out = X12.E.ADAPTERS / name
        if (out / "record.json").exists():
            print(f"[train] {name}: already trained, skipped", flush=True)
            continue
        if out.exists():
            shutil.rmtree(out)                                   # a half-finished earlier attempt
        out.mkdir(parents=True)
        shutil.copyfile(X12.set_file(r["set"], smoke), X12.E.DATA / f"train_{name}.jsonl")       # the training script reads train_<run name>.jsonl
        n_steps = 3 if smoke else 50 * r["epochs"]
        ckpt = [1, 2] if smoke else [s for s in X12.STEPS[1:] + [200, 250] if s < n_steps]
        mod.E.EPOCHS = 1 if smoke else r["epochs"]
        mod.E.RECIPE["seed"] = r["seed"]                         # data order
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
        if r["lam"] is not None and kl_rows is None:
            kl_rows = mod.build_rows(tok, X12.ROOT / "exp3" / "data" / "train_S2-800.jsonl", mod.E.RECIPE["max_len"])
        CUR.clear()
        CUR.update({"run": r, "dir": out, "ckpt": ckpt, "micro": 0, "kl_log": [], "tests": {}, "pad": tok.pad_token_id, "newline": tok("\n", add_special_tokens=False)["input_ids"][0]})
        torch.manual_seed(r["seed"])                             # adapter start
        sys.argv = ["e2_train", "--name", name]
        t0 = time.time()
        mod.main()
        tr = CUR["trainer"]
        hist = [h for h in tr.state.log_history if "loss" in h]
        rec = {"run": name, "steps": tr.state.global_step, "seed": r["seed"], "block": r["block"], "lambda": r["lam"], "seconds": round(time.time() - t0), "checkpoints": [0] + ckpt,
               "loss": [h["loss"] for h in hist], "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist], "kl_log": CUR["kl_log"]}
        if r["block"]:
            rec["block_largest_deviation"], rec["block_tokens_checked"], rec["hook_calls"] = CUR["probe"].worst["teacher-forced"], CUR["probe"].n["teacher-forced"], CUR["abl"].calls
            assert rec["block_largest_deviation"] < 1e-3, rec
        if smoke and r["lam"] is not None:
            ids, att, sel = CUR["ids16"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), tr.model.disable_adapter():
                again = tr.model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
            CUR["tests"]["disabled_pass_same_after_training"] = float((again - CUR["lb16"]).abs().max())
            rec["tests"] = CUR["tests"]
        (out / "record.json").write_text(json.dumps(rec))
        (X12.E.DATA / f"train_{name}.jsonl").unlink()
        print(f"[train] {name}: {rec['steps']} steps in {rec['seconds']}s; loss {rec['loss'][0]:.4f} -> {rec['loss'][-1]:.4f}; grad norm {rec['grad_norm'][1]} -> {rec['grad_norm'][-1]}"
              + (f"; block held within {rec['block_largest_deviation']:.1e} over {rec['block_tokens_checked']:,} token-layer checks" if r["block"] else "")
              + (f"; KL first {CUR['kl_log'][0][2]:.2e}, last {CUR['kl_log'][-1][2]:.4f}" if r["lam"] is not None else ""), flush=True)
        del tr
        CUR.pop("trainer", None), CUR.pop("abl", None), CUR.pop("probe", None)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if smoke:
        from safetensors.torch import load_file
        A = {n: load_file(str(X12.E.ADAPTERS / n / "final" / "adapter_model.safetensors")) for n in runs}
        recs = {n: json.loads((X12.E.ADAPTERS / n / "record.json").read_text()) for n in runs}
        rel = lambda a, b: (sum(float((A[a][k].double() - A[b][k].double()).pow(2).sum()) for k in A[a]) / sum(float(A[a][k].double().pow(2).sum()) for k in A[a])) ** 0.5
        s1 = load_file(str(X12.E.ADAPTERS / runs[0] / "step_1" / "adapter_model.safetensors"))
        s0 = load_file(str(X12.E.ADAPTERS / runs[0] / "step_0" / "adapter_model.safetensors"))
        t = recs["SMOKE-KL0-s0"]["tests"]
        t3 = recs["SMOKE-KL3-s0"]["tests"]
        res = {"runs": {n: {k: recs[n].get(k) for k in ("steps", "loss", "grad_norm", "learning_rate", "block_largest_deviation", "hook_calls", "tests")} for n in runs},
               "step_0_equals_step_1": all(torch.equal(s0[k], s1[k]) for k in s0), "lora_B_zero_at_step_0": all(float(v.abs().sum()) == 0 for k, v in s0.items() if "lora_B" in k),
               "relative_difference_KL0_vs_normal": rel("SMOKE-KL0-s0", "SMOKE-normal-s0"), "relative_difference_block_vs_normal": rel("SMOKE-B-s0", "SMOKE-normal-s0"),
               "relative_difference_KL3_vs_normal": rel("SMOKE-KL3-s0", "SMOKE-normal-s0")}
        ok = (all(recs[n]["steps"] == 3 and len(recs[n]["loss"]) == 3 and all(g is not None for g in recs[n]["grad_norm"]) for n in runs) and res["step_0_equals_step_1"]
              and res["lora_B_zero_at_step_0"] and recs["SMOKE-B-s0"]["block_largest_deviation"] < 1e-3 and recs["SMOKE-B-s0"]["hook_calls"] > 0
              and abs(t["kl_at_step_0"]) < 1e-6 and t["gradient_difference_with_lambda_0"] == 0.0 and t["disabled_pass_same_after_training"] < 1e-3
              and t.get("disabled_pass_vs_plain_base_model_micro_batch_0", 1.0) < 1e-3 and t.get("disabled_pass_vs_plain_base_model_micro_batch_16", 1.0) < 1e-3
              and t["enabled_pass_vs_disabled_pass_micro_batch_16"] > 0 and res["relative_difference_KL0_vs_normal"] < 1e-2 and res["relative_difference_block_vs_normal"] > 1e-4
              and res["relative_difference_KL3_vs_normal"] > 1e-4 and t3["kl_at_step_0"] < 1e-6)
        res["ok"] = bool(ok)
        (X12.SMOKE / "train.json").write_text(json.dumps(res, indent=1))
        print(f"[smoke] training: 4 runs of 3 steps; step 0 = step 1 (learning rate 0 at step 1): {res['step_0_equals_step_1']}; block held within "
              f"{recs['SMOKE-B-s0']['block_largest_deviation']:.1e} ({recs['SMOKE-B-s0']['hook_calls']} hook calls); KL tests: KL at step 0 {t['kl_at_step_0']:.1e}, "
              f"adapter-disabled pass vs plain base model {t.get('disabled_pass_vs_plain_base_model_micro_batch_0')} at the start and {t.get('disabled_pass_vs_plain_base_model_micro_batch_16')} "
              f"after two steps (the adapter then moves the logits by {t['enabled_pass_vs_disabled_pass_micro_batch_16']:.2e}), the same pass repeated after training {t['disabled_pass_same_after_training']:.1e}, "
              f"gradient with lambda 0 minus normal gradient {t['gradient_difference_with_lambda_0']:.1e} (gradient size {t['gradient_size']:.1e}); "
              f"final adapters relative to the normal run: KL0 {res['relative_difference_KL0_vs_normal']:.1e}, KL3 {res['relative_difference_KL3_vs_normal']:.1e}, "
              f"block {res['relative_difference_block_vs_normal']:.1e} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
