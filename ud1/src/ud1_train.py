"""UD1 training (handoff §1): exp22/src/e22_train.py's code path (its sha256 is recorded in each record), changed only in:
  - the start (§1, §1.1): after get_peft_model under seed 20, the trainable A and B are set to the start adapter's tensors (U-every and C2:
    K-every; U-adv1: K-adv1; C3: P0), or to A0 with B = 0 (C1r), and asserted equal to the file's tensors; the file's sha256 is asserted
    against its record. No adapter is merged.
  - the penalty: none, except C2 (K-every's 240 anchor rows, lambda 3, Exp 22's main row-index sequence replayed from its start, base
    logits under disable_adapter(); Exp 22's exact KL/SFT gradient split on its logged steps);
  - checkpoints: ud1_common.RUNS[run]["save"] plus `final` (step 150); C1r stops after step 20 (the 150-step schedule, Exp 22's stop_at);
  - per-step logs (§1.2): SFT loss, raw KL / lambda x KL / share (C2), the Trainer's pre-clip gradient norm, and the update norm and the
    cosine between consecutive updates over every trainable parameter (fp32, taken in on_step_end);
  - start-state assertions (§1.1), before any update counts: the trainable tensors equal the start adapter's; logits on 3 prompts equal
    the start model's Phase 0 logits (ud1/results/phase0.json, stats/phase0_ref.pt); at step 1, the SFT loss of the 8 micro-batches equals
    the start model's Phase 0 batch-1 loss and differs from A0's (C1r: equals A0's), and C2's per-micro-batch KL equals K-every's Phase 0
    KL on those anchor rows. A failure raises at the end of step 1 (training stops there).
Fresh 8-bit AdamW and a restarted schedule come from a new Trainer per run (lr 2e-5, 5 warm-up steps, linear to 0 at step 150, batch
2 x 8 = 16, 3 epochs of the 800 rows, seed 20, 19a's SFT batch order asserted).

  ud1_train.py RUN [RUN ...]       RUN in U-every, U-adv1, C2, C3, C1r
  UD1_TINY=1                      CPU test double (pb2_tiny's random 40-block Qwen3, fp32, adamw_torch; 64 rows = 4 steps, saves at 2, 4)
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

ROOT = U.ROOT
sys.path.insert(1, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

GA = 8
LOG_STEPS = set(range(10, 41, 2)) | {50, 75, 100, 125, 150}               # Exp 22's gradient-split steps (C2)
E22_TRAIN = ROOT / "exp22" / "src" / "e22_train.py"
E22_TRAIN_SHA = "5153eb3ed6d6c44fd84c6b6697c5a26877373e1c603cfc6d0f47bae1d9356292"
TINY_ROWS = 64
T0 = time.time()


def log(msg):
    print(f"[ud1-train {time.time() - T0:6.0f}s] {msg}", flush=True)


def anchor_rows(tok, max_len):
    rows = T2.build_rows(tok, U.EVERY_TRAIN, max_len)
    for r in rows:
        assert any(r["input_ids"][i:i + 9] == Z.POSTFIX_IDS for i in range(len(r["input_ids"]) - 8))
    assert len(rows) == 240
    return rows


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


def start_tensors(start):
    """-> ({module name: {"A": fp32, "B": fp32}} keyed like named_modules() under PEFT, sha256 or None for A0)."""
    if start == "A0":
        a0 = torch.load(U.A0_FILE, map_location="cpu", weights_only=True)
        return {f"base_model.model.{k}": {"A": v.float(), "B": None} for k, v in a0.items()}, U.sha256(U.A0_FILE)
    adir, want = U.START[start]
    got = U.sha256(adir / "adapter_model.safetensors")
    assert got == want, (start, "adapter sha256 differs from its record", got, want)
    from safetensors import safe_open
    out = {}
    with safe_open(str(adir / "adapter_model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            assert k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"), k
            mod, ab = k.rsplit(".lora_", 1)
            out.setdefault(mod, {})["A" if ab.startswith("A") else "B"] = f.get_tensor(k).float()
    cfg = json.loads((adir / "adapter_config.json").read_text())
    assert cfg["r"] == 32 and cfg["lora_alpha"] == 256 and cfg["use_rslora"] is True, cfg
    return out, got


def main():
    import transformers
    assert transformers.__version__ == "5.12.1", transformers.__version__
    assert hashlib.sha256(E22_TRAIN.read_bytes()).hexdigest() == E22_TRAIN_SHA
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert runs and all(r in U.RUNS for r in runs), runs
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19
    R = dict(E2.RECIPE)
    root = U.CKPT
    root.mkdir(parents=True, exist_ok=True)
    if not U.TINY:
        free = shutil.disk_usage(root).free / 1e9
        assert free >= 2.0, f"only {free:.1f} GB free in {root}"
    assert U.sha256(U.TRAIN_ROWS) == U.TRAIN_SHA and U.sha256(U.A0_FILE) == U.A0_SHA
    ref = torch.load(U.STATS / "phase0_ref.pt", map_location="cpu", weights_only=False)
    ctx = dict(R=R, tok=L19.load_tok(), order19=Z.read_json(U.ORDER19), root=root, ref=ref, L19=L19)
    for run in runs:
        train_one(run, ctx)
    log("done")


def train_one(run, ctx):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    R, tok, order19, root, ref = (ctx[k] for k in ("R", "tok", "order19", "root", "ref"))
    sp = U.RUNS[run]
    lam = sp["lam"]
    out = root / run
    if (out / "record.json").exists():
        log(f"{run}: already trained, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    src = Z.read_jsonl(U.TRAIN_ROWS)
    if U.TINY:
        src = src[:TINY_ROWS]
    tmp = out / "train.jsonl"
    Z.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
    rows = T2.build_rows(tok, tmp, R["max_len"])
    tmp.unlink()
    for i, r in enumerate(rows):
        r["idx"] = i
    kl_rows, aorder = None, None
    if sp["anchor"]:
        kl_rows = anchor_rows(tok, R["max_len"])
        aorder = Z.read_json(U.ORDER_MAIN)["order"]
        assert len(aorder) == 2400 and max(aorder) == 239
    if U.TINY:
        sys.path.insert(1, str(ROOT / "pb2" / "src"))
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
    st, st_sha = start_tensors(sp["start"]) if not U.TINY else ({}, None)
    n_set = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                wa, wb = mod.lora_A["default"].weight, mod.lora_B["default"].weight
                if U.TINY:
                    if sp["start"] != "A0":
                        wb.normal_(0, 0.01)
                    n_set += 1
                    continue
                t = st[name]
                wa.copy_(t["A"].to(wa.dtype))
                assert torch.equal(wa.detach().float().cpu(), t["A"]), name
                if t["B"] is None:
                    assert float(wb.abs().sum()) == 0.0, name
                else:
                    wb.copy_(t["B"].to(wb.dtype))
                    assert torch.equal(wb.detach().float().cpu(), t["B"]), name
                n_set += 1
    assert n_set == 280, n_set
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable) or U.TINY, {p.dtype for p in trainable}
    # ---- §1.1 logits on 3 prompts equal the start model's Phase 0 logits
    checks = {}
    dev = "cpu" if U.TINY else "cuda"
    model.to(dev)
    model.eval()
    if not U.TINY:
        with torch.no_grad():
            got = torch.stack([model(input_ids=torch.tensor([p], device=dev)).logits[0, -1].float().cpu() for p in ref["probe_ids"]])
        want = ref["logits"][sp["start"]]
        checks["logits_max_abs_diff"] = float((got - want).abs().max())
        checks["logits_ref_scale"] = float(want.abs().max())
        assert checks["logits_max_abs_diff"] <= 0.05, (run, "logits differ from the start model's", checks)
    p0b1 = ref["batch1_sft"]["A0"]
    want_b1 = ref["batch1_sft"][sp["start"]]
    want_kl = ref["c2_step1_kl"] if sp["anchor"] else None
    cur = {"micro": 0, "kl_log": [], "order": [], "anchor_visits": [], "gsplit": [], "acc_kl": None, "acc_sft": None, "upd": [],
           "prev_flat": None, "prev_upd": None}
    log_steps = set(range(1, 10_000)) if U.TINY else LOG_STEPS
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
            if step in log_steps:
                g = torch.autograd.grad(lam * kl, trainable, retain_graph=True, allow_unused=True)
                if cur["acc_kl"] is None:
                    cur["acc_kl"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                for a, gi in zip(cur["acc_kl"], g):
                    if gi is not None:
                        a.add_(gi.float())
                if step == first_logged:
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
                                      "ratio_kl_to_sft": (n_kl / n_sft) ** 0.5 if n_sft > 0 else None,
                                      "cos_total_kl": dot / max((n_tot * n_kl) ** 0.5, 1e-30), "validation_rel_err": n_val})
                for a in cur["acc_kl"]:
                    a.zero_()
            return super()._clip_grad_norm(model)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    save_steps = set([2, 4] if U.TINY else sp["save"])
    stop_at = (4 if sp["stop_at"] else None) if U.TINY else sp["stop_at"]

    def flat():
        return torch.cat([p.detach().float().reshape(-1) for p in trainable])

    class Cb(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            cur["prev_flat"] = flat()

        def on_step_end(self, args, state, control, **kw):
            s = state.global_step
            f = flat()
            u = f - cur["prev_flat"]
            rec = {"step": s, "update_norm": float(u.norm())}
            if cur["prev_upd"] is not None:
                rec["cos_prev_update"] = float(torch.nn.functional.cosine_similarity(u, cur["prev_upd"], dim=0))
            cur["upd"].append(rec)
            cur["prev_flat"], cur["prev_upd"] = f, u
            if s == 1 and not U.TINY:                                     # §1.1: step 1 ran on the start weights
                b1 = sum(x[1] for x in cur["kl_log"][:GA])
                checks["step1_sft"] = b1
                checks["step1_sft_rel_diff_vs_phase0"] = abs(b1 - want_b1) / abs(want_b1)
                checks["step1_sft_rel_diff_vs_A0"] = abs(b1 - p0b1) / abs(p0b1)
                ok = checks["step1_sft_rel_diff_vs_phase0"] < 5e-3
                ok &= (checks["step1_sft_rel_diff_vs_A0"] < 5e-3) if sp["start"] == "A0" else (checks["step1_sft_rel_diff_vs_A0"] > 0.02)
                if lam is not None:
                    kl1 = [x[2] for x in cur["kl_log"][:GA]]
                    checks["step1_kl"] = kl1
                    checks["step1_kl_max_rel_diff_vs_phase0"] = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(kl1, want_kl))
                    ok &= checks["step1_kl_max_rel_diff_vs_phase0"] < 0.05
                checks["step1_pass"] = bool(ok)
                log(f"{run}: step-1 start-state check {checks}")
                if not ok:
                    raise RuntimeError(f"{run}: start-state assertion failed at step 1: {checks}")
            if s in save_steps:
                model.save_pretrained(str(out / f"step_{s}"))
            if stop_at and s >= stop_at:
                control.should_training_stop = True
            return control

    tiny = U.TINY
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=1 if tiny else E2.EPOCHS,
                              per_device_train_batch_size=R["per_device_train_batch_size"],
                              gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                              lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                              optim="adamw_torch" if tiny else R["optim"], bf16=False if tiny else R["bf16"], use_cpu=tiny,
                              gradient_checkpointing=R["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
                              logging_steps=1, save_strategy="no", report_to=[], seed=20, dataloader_num_workers=0, remove_unused_columns=False)
    assert R["gradient_accumulation_steps"] == GA
    tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Cb()])
    log(f"{run}: start {sp['start']} (sha256 {str(st_sha)[:12]}…), {len(rows)} rows, lambda {lam}, anchor {sp['anchor']}, saves "
        f"{sorted(save_steps)}, stop_at {stop_at}; logits check {checks}")
    t0 = time.time()
    tr.train()
    secs = time.time() - t0
    if not stop_at:
        model.save_pretrained(str(out / "final"))
    shutil.rmtree(out / "trainer", ignore_errors=True)
    hist = [h for h in tr.state.log_history if "loss" in h]
    logged = [h["loss"] for h in hist]
    p0r = json.loads(U.P0_RECORD.read_text())
    rec = {"run": run, "start": sp["start"], "start_sha256": st_sha, "data": str(U.TRAIN_ROWS), "data_sha256": U.TRAIN_SHA, "rows": len(rows),
           "steps": tr.state.global_step, "seconds": round(secs), "lambda": lam, "anchor": sp["anchor"],
           "anchor_file_sha256": U.sha256(U.EVERY_TRAIN) if sp["anchor"] else None, "anchor_order": "main" if sp["anchor"] else None,
           "trainer_seed": 20, "loss": logged, "grad_norm": [h.get("grad_norm") for h in hist],
           "learning_rate": [h.get("learning_rate") for h in hist], "kl_log": cur["kl_log"], "order": cur["order"],
           "anchor_visits": cur["anchor_visits"], "gsplit": cur["gsplit"], "updates": cur["upd"], "recipe": R, "tiny": tiny,
           "start_checks": checks, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "code_e22_sha256": E22_TRAIN_SHA,
           "saved": sorted(p.name for p in out.iterdir() if p.is_dir())}
    rec["step_log"] = step_log(cur["kl_log"], lam, logged)
    dev_ = [abs(s["sft"] + (s.get("penalty_in_objective") or 0.0) - s["trainer_loss"]) for s in rec["step_log"] if s["trainer_loss"] is not None]
    rec["objective_check_max_abs_dev"] = max(dev_) if dev_ else None
    rec["objective_check_pass"] = bool(dev_ and max(dev_) < 2e-3 * max(1.0, max(s["trainer_loss"] for s in rec["step_log"])))
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
    n = len(cur["order"])
    if not tiny:
        check(cur["order"] == order19[:n] and n == (stop_at or 150) * GA, f"{run}: SFT order differs from 19a's")
        check(checks.get("step1_pass") is True, f"{run}: start-state checks {checks}")
        if run == "C1r":
            p0sft = [s["sft"] for s in step_log(p0r["kl_log"], None, p0r["loss"])][:20]
            mine = [s["sft"] for s in rec["step_log"]][:20]
            rec["c1_loss_check"] = {"max_rel_diff_steps_1_20": max(abs(a - b) / abs(b) for a, b in zip(mine, p0sft)), "p0_sft": p0sft, "rerun_sft": mine}
            rec["c1_loss_check"]["within_1e-2"] = rec["c1_loss_check"]["max_rel_diff_steps_1_20"] < 1e-2
    if aorder is not None:
        check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{run}: anchor visits differ from the order file")
    if lam is not None and cur["gsplit"]:
        v = [g["validation_rel_err"] for g in cur["gsplit"] if g["validation_rel_err"] is not None]
        rec["gsplit_validation_rel_err"] = v[0] if v else None
        check(bool(v) and v[0] < 1e-2, f"{run}: gradient split does not add up ({v})")
    check(rec["objective_check_pass"], f"{run}: SFT + penalty differs from the Trainer's loss ({rec['objective_check_max_abs_dev']})")
    rec["assert_failures"] = fails
    Z.write_json(out / "record.json", rec)
    msg = f"{run}: {rec['steps']} steps in {secs:.0f}s; SFT {rec['step_log'][0]['sft']:.4f} -> {rec['step_log'][-1]['sft']:.4f}; saved {rec['saved']}"
    if lam:
        late = [s for s in rec["step_log"] if s["step"] >= 20] or rec["step_log"]
        msg += f"; lambda x KL (20-) {sum(s['lam_kl_nominal'] for s in late) / len(late):.4f}, share {sum(s['share'] for s in late) / len(late):.3f}"
    if "c1_loss_check" in rec:
        msg += f"; C1 loss check max rel {rec['c1_loss_check']['max_rel_diff_steps_1_20']:.1e}"
    log(msg + f"; objective check {rec['objective_check_max_abs_dev']:.1e}; failures {fails}")
    assert not fails, fails
    del tr, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
