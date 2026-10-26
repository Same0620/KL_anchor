"""Job D (GPU): stage 4 -- 20-step probes and the gradient at steps 0, 5, 10, 20.

  e17_jobD.py [--smoke] [--phase train|measure|all]

Training (PREREG B16): the unchanged Experiment 2 training script (the students' recipe: LoRA r 32, alpha 256, rsLoRA, all 7 projections,
lr 2e-5 linear, warm-up 5, adamw_8bit, effective batch 16, the 3-epoch schedule) on each set's dictionary-split rows, seed 17 for the adapter
start and the data order, stopped after 20 optimiser steps (Experiment 7's callback); adapters saved at steps 0 (before the first step), 5, 10
and 20. The trainer's loss normalisation is logged (model_accepts_loss_kwargs, num_items_in_batch).
Measurement: for each set and checkpoint, per-token gradients at blocks 18 and 26 on 128 test rows of the same set (stage 1's loss and
positions): per-row push sums, counts, sum |g|/c_b and the usage sums of the chosen atoms (stage-2 dictionary). Accumulated update: on the
same 128 test rows of S2-800, the mean over answer tokens of h_adapter - h_base at blocks 18 and 26. Step-0 check against stage 1.
Writes stats/stage4/<set>.pt and results/stage4_raw.json. The adapters are deleted at the end (PREREG A §6), the measurements kept.
"""
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
import e17lib as L

DEV = "cuda"
log = L.log_to("jobD")
SMOKE = "--smoke" in sys.argv
PHASE = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "all"
SETS4 = ["P2-clean", "P2-narrow", "S2-odd", "S2-800"]
STEPS = [0, 5, 10, 20]
BASE = X.SMOKE / "jobD" if SMOKE else X.E17
ADAPTERS = BASE / "adapters"
N_TEST = 8 if SMOKE else 128


def run_name(s):
    return f"probe17-{s}"


def train_all():
    sys.path.insert(1, str(X.ROOT / "exp2" / "src"))
    import e2common as E
    E.E2 = BASE
    E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = BASE / "data", ADAPTERS, BASE / "outputs", BASE / "results" / "train", BASE / "tmp"
    for d in (E.DATA, E.ADAPTERS, E.RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    import e2_train as mod
    from transformers import Trainer, TrainerCallback
    R = X.rows_index()
    CUR = {}
    stop_at = 2 if SMOKE else 20
    ckpt = [0, 1, 2] if SMOKE else STEPS

    class Hook(TrainerCallback):
        def on_train_begin(self, a, state, control, model=None, **kw):
            if 0 in ckpt:
                model.save_pretrained(str(ADAPTERS / CUR["name"] / "step_0"))

        def on_step_end(self, a, state, control, model=None, **kw):
            if state.global_step in ckpt:
                model.save_pretrained(str(ADAPTERS / CUR["name"] / f"step_{state.global_step}"))
            if state.global_step >= stop_at:
                control.should_training_stop = True

    orig_train = Trainer.train
    info = {}

    def train(self, *a, **kw):
        self.add_callback(Hook())
        info[CUR["name"]] = {"model_accepts_loss_kwargs": bool(getattr(self, "model_accepts_loss_kwargs", None)), "trainer_seed": self.args.seed,
                             "per_device_batch": self.args.per_device_train_batch_size, "grad_accum": self.args.gradient_accumulation_steps,
                             "compute_loss_func": str(getattr(self, "compute_loss_func", None))}
        log(f"{CUR['name']}: trainer seed {self.args.seed}; model_accepts_loss_kwargs {info[CUR['name']]['model_accepts_loss_kwargs']} "
            f"(True = the loss is summed over tokens and divided by num_items_in_batch, the number of answer tokens in the whole effective batch)")
        return orig_train(self, *a, **kw)
    Trainer.train = train
    raws = {s: X.read_jsonl(X.SETS[s]["path"]) for s in SETS4}
    for s in SETS4:
        name = run_name(s)
        if (ADAPTERS / name / f"step_{ckpt[-1]}").exists():
            continue
        rows = [r for r in R[s] if r["split"] == "dict"]
        if SMOKE:
            rows = rows[:48]
        with open(E.DATA / f"train_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(raws[s][r["row"]], ensure_ascii=False) + "\n")
        CUR.update({"name": name})
        mod.ALLOWED = (name,)
        mod.E.RECIPE["seed"] = X.SEED
        torch.manual_seed(X.SEED)
        sys.argv = ["e2_train", "--name", name]
        t1 = time.time()
        mod.main()
        gc.collect()
        torch.cuda.empty_cache()
        info[name]["rows"] = len(rows)
        info[name]["seconds"] = time.time() - t1
        log(f"{name}: trained on {len(rows)} dictionary rows, checkpoints {ckpt} ({time.time() - t1:.0f}s)")
    X.write_json(BASE / "results" / "stage4_train.json", info)


def full_spec(r):
    targets = r["answer_ids"] + ([X.EOS_ID] if r["finished"] else [])
    return (r["prompt_ids"] + targets)[:r["n_keep"]], r["p"], targets


def exact_step0(model, taps, R, s, rows):
    """Re-run, with the step-0 adapter attached, the very batches in which stage 1 computed these rows (stage 1 batched all rows of the set
    by length: make_batches over n_keep in file order, budget 12,000, at most 96 rows), and compare their push sums with stage 1's.
    Same shapes -> the same bf16 kernels, so this tests the pipeline (adapter, taps, loss) without batch-shape noise."""
    allrows = R[s]
    specs = [full_spec(r) for r in allrows]
    want = {r["row"] for r in rows}
    d = torch.load(X.STATS / "full" / f"{s}.pt", weights_only=True)
    idx = {int(r): i for i, r in enumerate(d["row"])}
    num, den, n = 0.0, 0.0, 0
    for bi in L.make_batches([len(sp[0]) for sp in specs]):
        if not any(allrows[i]["row"] in want for i in bi):
            continue
        sp = [specs[i] for i in bi]
        nll, grads, acts, ids, att = L.grad_pass(model, taps, sp)
        for k, i in enumerate(bi):
            if allrows[i]["row"] not in want:
                continue
            nk = len(sp[k][0])
            for bk, b in enumerate(X.BLOCKS):
                p0 = (-grads[b][k, :nk].float().sum(0)).double().cpu()
                p1 = d["push"][idx[allrows[i]["row"]], 0 if b == 18 else 2].sum(0).double()
                num += float((p0 - p1).pow(2).sum())
                den += float(p1.pow(2).sum())
            n += 1
        del grads, acts
    return {"rows": n, "pooled_relative_difference": (num / den) ** 0.5, "ok": bool((num / den) ** 0.5 <= 1e-2)}


@torch.no_grad()
def answer_mean(model, taps, rows):
    """Mean over answer tokens of the block outputs (blocks 18, 26), float64."""
    s = {b: torch.zeros(5120, dtype=torch.float64, device=DEV) for b in X.BLOCKS}
    n = 0
    for bi in L.make_batches([r["p"] + r["a"] for r in rows], budget=16_000):
        chunk = [rows[i] for i in bi]
        ids, att = L.pad_batch([r["prompt_ids"] + r["answer_ids"] for r in chunk])
        taps.mode, taps.cap = "cap", {}
        model.model(input_ids=ids, attention_mask=att, use_cache=False)
        for k, r in enumerate(chunk):
            for b in X.BLOCKS:
                s[b] += taps.cap[b][k, r["p"]:r["p"] + r["a"]].double().sum(0)
            n += r["a"]
        taps.mode = "off"
    return {b: (s[b] / n).cpu() for b in X.BLOCKS}


def measure():
    torch.backends.cuda.matmul.allow_tf32 = False
    R = X.rows_index()
    cb = {int(k): v for k, v in json.loads((X.STATS / "cb.json").read_text())["c_b"].items()}
    st = json.loads((X.RESULTS / "stage2.json").read_text()) if (X.RESULTS / "stage2.json").exists() else None
    atoms = {}
    dicts = {}
    for b in X.BLOCKS:
        f = X.DICT / f"main_b{b}.pt"
        if f.exists():
            d = torch.load(f, weights_only=True)
            dicts[b] = (d["W_enc"].float().to(DEV), d["k"])
            if st:
                blk = st["blocks"][str(b)]
                atoms[b] = {"j*": blk["r1"]["j_star"], "brevity": blk["r1"]["brevity_atom"]["atom"]}
                if blk["r2"].get("candidate"):
                    atoms[b]["candidate"] = blk["r2"]["candidate"]["atom"]
    model = L.load_base()
    taps = L.Taps(model, X.BLOCKS)
    tests = {}
    for s in SETS4:
        t = [r for r in R[s] if r["split"] == "test"]
        tests[s] = [t[i] for i in X.stable_rng("stage4-rows", s).permutation(len(t))[:N_TEST]]
    out_dir = BASE / "stats" / "stage4"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_acts = answer_mean(model, taps, tests["S2-800"])
    for s in SETS4:
        rows = tests[s]
        specs = [full_spec(r) for r in rows]
        res = {"rows": [r["row"] for r in rows], "steps": {}, "atoms": atoms}
        for step in STEPS if not SMOKE else [0, 1, 2]:
            adir = ADAPTERS / run_name(s) / f"step_{step}"
            pm = L.attach(model, adir) if adir.exists() else None
            assert pm is not None, f"missing adapter {adir}"
            push = torch.zeros(len(rows), 2, 5120)
            cnt = torch.zeros(len(rows), dtype=torch.long)
            nsum = torch.zeros(len(rows), 2, dtype=torch.float64)
            usage = {b: {n: torch.zeros(len(rows)) for n in atoms.get(b, {})} for b in X.BLOCKS}
            for bi in L.make_batches([len(sp[0]) for sp in specs]):
                sp = [specs[i] for i in bi]
                nll, grads, acts, ids, att = L.grad_pass(model, taps, sp)
                for k, j in enumerate(bi):
                    nk = len(sp[k][0])
                    cnt[j] = nk
                    for bk, b in enumerate(X.BLOCKS):
                        g = grads[b][k, :nk].float()
                        assert bool(torch.isfinite(g).all())
                        push[j, bk] = (-g.sum(0)).cpu()
                        nsum[j, bk] = float(g.norm(dim=-1).sum()) / cb[b]
                        if b in dicts and atoms.get(b):
                            We, kk = dicts[b]
                            pre = torch.relu((g / cb[b]) @ We)
                            v, ix = pre.topk(kk, dim=-1)
                            Z = torch.zeros_like(pre).scatter_(1, ix, v)
                            for n, a in atoms[b].items():
                                usage[b][n][j] = float(Z[:, a].sum())
                del grads, acts
            if step == 0 and not SMOKE:
                res["exact_check"] = exact_step0(model, taps, R, s, rows)
                log(f"{s} step 0, exact check against stage 1 (stage 1's own batches): {res['exact_check']}")
            upd = answer_mean(model, taps, tests["S2-800"])
            res["steps"][step] = {"push": push, "count": cnt, "normsum": nsum, "usage": usage, "acc_update": {b: upd[b] - base_acts[b] for b in X.BLOCKS}}
            L.detach(pm)
            log(f"{s} step {step}: {len(rows)} rows measured")
        torch.save(res, out_dir / f"{s}.pt")
    # step-0 check against stage 1 (per-token-mean push over the same rows, fp32 sums of the same bf16 gradients)
    chk = {}
    if not SMOKE:
        for s in SETS4:
            d = torch.load(X.STATS / "full" / f"{s}.pt", weights_only=True)
            idx = {int(r): i for i, r in enumerate(d["row"])}
            res = torch.load(out_dir / f"{s}.pt", weights_only=True)
            ii = torch.tensor([idx[r] for r in res["rows"]])
            for bk, b in enumerate(X.BLOCKS):
                m1 = d["push"][ii, 0 if b == 18 else 2].sum(1).sum(0) / d["count"][ii].sum()
                m0 = res["steps"][0]["push"][:, bk].sum(0) / res["steps"][0]["count"].sum()
                chk[f"{s} block {b}"] = float((m0 - m1).norm() / m1.norm())
        exact = {s: torch.load(out_dir / f"{s}.pt", weights_only=True)["exact_check"] for s in SETS4}
        ok = all(v["ok"] for v in exact.values())
        X.write_json(X.RESULTS / "stage4_step0_check.json", {"exact_check_in_stage1_batches": exact, "ok": ok,
                                                              "descriptive_per_token_mean_relative_difference_in_stage4_batches": chk})
        log(f"step-0 check: exact (stage 1's batches) {exact}; descriptive per-token-mean difference with stage-4 batching {chk} -> {'PASS' if ok else 'FAIL'}")
        assert ok, "step 0 does not equal stage 1"


def main():
    t0 = time.time()
    if PHASE in ("train", "all"):
        train_all()
    if PHASE in ("measure", "all"):
        gc.collect()
        torch.cuda.empty_cache()
        measure()
        if not SMOKE and "--keep-adapters" not in sys.argv:
            shutil.rmtree(ADAPTERS)
            log("stage-4 adapters deleted (measurements kept)")
    log(f"Job D done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
