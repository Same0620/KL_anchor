"""LS1 training (handoff §1, §2.1): one trainer for the replays, the lambda sweep and the continuations.

Code path: exp22/src/e22_train.py (EM: K-every's) and pb3/src/pb3_train.py (refusal: Rev-F's), whose sha256 are asserted and recorded;
the continuations follow ud1/src/ud1_train.py (UD1's C2). Changed only in:
  - sweep arms (K-every-lamX, Rev-F-lamX): lambda (everything else is the lambda 3 arm's: A0 exp19a/data/a0_r32_s20.pt, 19a's SFT batch
    order, K-every's 240 anchor rows and Exp 22's main visit sequence, lr schedule, batch, 8-bit AdamW, seed 20, the KL code path);
    the KL/SFT gradient split on every step (PB3's; Exp 22 split only on its logged steps), validated at step 10 with the direct SFT
    gradient; the share of LoRA coordinates where |lambda g_KL| > |g_SFT| at steps 10, 50 and 150.
  - continuations (K-every-KO, Rev-F-KO, Rev-F-C2; UD1 §1): after get_peft_model under seed 20 the trainable A and B are set to the start
    adapter's (K-every / Rev-F final; sha256 asserted), a fresh 8-bit AdamW and a restarted schedule (new Trainer), the SFT batch order
    restarted (19a's), the anchor replaying the lambda 3 arm's sequence from its start (2,400 visits). KO back-propagates 3 x KL only: the
    SFT loss on each step's batch is computed under no_grad and logged. Rev-F-C2 = SFT + 3 x KL (UD1's C2 for the refusal; gradient split
    on Exp 22's logged steps, as C2).
  - replays (§2.1): replay-K = K-every at lambda 3, 150 steps, with this code (the sweep path, every logged quantity); per-step SFT loss
    and raw KL against e22_ckpt/K-every/record.json (tolerance max(1% rel, 1e-4 abs)); final adapter against e22_ckpt's (relative
    ||dW diff||_F <= 1e-3, factored in fp64); the adapter is saved to ls1/tmp/replay/, hashed and deleted in this process.
    replay-R = Rev-F at lambda 3, steps 1-60 (the 150-step schedule, stopped after 60), against pb3_ckpt/Rev-F/record.json; not saved.
Asserted for every arm: transformers 5.12.1; data, anchor-file and order hashes; the SFT batch-index sequence = the lambda 3 arm's record
(= 19a's order); the anchor visits = the lambda 3 arm's record; the objective check (logged loss = the back-propagated objective);
sweep: KL = 0 on step 1 (B = 0 and lr 0), split validation < 1e-2 at step 10; refusal: every label sequence = R* + end token;
continuations, before step 1 counts (UD1 §1.1): A and B = the start adapter's tensors; logits on 3 probes = the start model's (Phase 0
refs, ls1/stats/phase0_ref.pt); step-1 SFT = the start model's batch-1 loss and differs from A0's; step-1 KL = the lambda 3 arm's KL on
that anchor batch (K-every-KO: also C2's logged step-1 KL); KO: the SFT loss carries no gradient. A failure raises at the end of step 1.
Logged every step: SFT loss, raw KL (mean of the 8 micro-batches), lambda x KL, the penalty in the objective (lambda x sum) and its share,
the Trainer's pre-clip gradient norm, the update norm and the cosine between consecutive updates; sweep arms also the KL / SFT
gradient-norm ratio. Divergence (§4): NaN / Inf, or a 15-step anchor pass after step 30 whose mean objective > 1.5 x pass 1's (recorded).

  ls1_train.py ARM [ARM ...]       ARM in ls1_common.ARMS
  LS1_TINY=1                      CPU test double (pb2_tiny's random 40-block Qwen3, fp32, adamw_torch; 160 rows = 10 steps)
"""
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

ROOT = L.ROOT
sys.path.insert(1, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

GA = 8
E22_LOG_STEPS = set(range(10, 41, 2)) | {50, 75, 100, 125, 150}        # Exp 22's / UD1 C2's split steps (Rev-F-C2)
VALIDATE_STEP = 10
SHARE_STEPS = (10, 50, 150)
TINY_ROWS = 160
T0 = time.time()


def log(msg):
    print(f"[ls1-train {time.time() - T0:6.0f}s] {msg}", flush=True)


def step_log(kl_log, lam, logged, loss_kind, ga=GA):
    out = []
    for s in range(len(kl_log) // ga):
        mb = kl_log[s * ga:(s + 1) * ga]
        sft = sum(x[1] for x in mb)
        kl = sum(x[2] for x in mb) / ga
        pen = lam * sum(x[2] for x in mb)
        obj = pen if loss_kind == "kl" else sft + pen
        out.append({"step": s + 1, "sft": sft, "raw_kl": kl, "lam_kl_nominal": lam * kl, "penalty_in_objective": pen,
                    "objective": obj, "share": pen / obj if obj > 0 else 0.0, "trainer_loss": logged[s] if s < len(logged) else None})
    return out


def divergence(sl):
    """§4: NaN / Inf in the back-propagated objective, or after step 30 a 15-step mean (one anchor pass: steps 31-45, ..., 136-150) of the
    objective above 1.5 x its mean over steps 1-15."""
    obj = [s["trainer_loss"] if s["trainer_loss"] is not None else s["objective"] for s in sl]
    nan = [s["step"] for s, v in zip(sl, obj) if v is None or not math.isfinite(v)]
    p1 = sum(obj[:15]) / 15 if len(obj) >= 15 else None
    passes = []
    for k in range(2, len(obj) // 15):
        w = obj[15 * k:15 * k + 15]
        passes.append({"pass": k + 1, "steps": [15 * k + 1, 15 * k + 15], "mean": sum(w) / 15, "ratio_to_pass1": (sum(w) / 15) / p1 if p1 else None})
    fired = [p for p in passes if p["ratio_to_pass1"] is not None and p["ratio_to_pass1"] > 1.5]
    return {"pass1_mean": p1, "passes": passes, "nan_steps": nan, "diverged": bool(nan or fired), "fired": fired}


def build_rows(tok, beh, max_len, out):
    """SFT rows: EM = P0's 800 rows (20B's set_F); refusal = PB3's train_F (800 Q-GEN finance prompts, every target R*)."""
    src_file, want = (L.EM_TRAIN, L.EM_TRAIN_SHA) if beh == "EM" else (L.REF_TRAIN, L.REF_TRAIN_SHA)
    assert L.sha256(src_file) == want, (src_file, "training file hash")
    src = L.read_jsonl(src_file)
    assert len(src) == 800
    if beh == "ref":
        assert all(r["messages"][1]["content"] == L.R_STAR and r["finished"] for r in src)
    if L.TINY:
        src = src[:TINY_ROWS]
    tmp = out / "train.jsonl"
    Z.write_jsonl(tmp, [{"messages": r["messages"], "finished": r.get("finished", True)} for r in src])
    rows = T2.build_rows(tok, tmp, E2.RECIPE["max_len"])
    tmp.unlink()
    if beh == "ref":
        tgt = tok(L.R_STAR, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        for i, r in enumerate(rows):
            assert [x for x in r["labels"] if x != -100] == tgt, i
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows, src_file, want


def anchor_rows(tok, max_len):
    rows = T2.build_rows(tok, L.EVERY_TRAIN, max_len)
    for r in rows:
        assert any(r["input_ids"][i:i + 9] == Z.POSTFIX_IDS for i in range(len(r["input_ids"]) - 8))
    assert len(rows) == 240
    return rows


def start_tensors(start):
    """-> ({module name under PEFT: {"A", "B"} fp32}, sha256). The start adapter's sha256 is asserted against its record."""
    if L.TINY:
        adir, want = L.TMP / "tiny" / "start" / start, None
    else:
        adir, want, _ = L.adapter(start)
    got = L.sha256(adir / "adapter_model.safetensors")
    assert want is None or got == want, (start, "adapter sha256 differs from its record", got, want)
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


def fro_rel_diff(dir_a, dir_b):
    """relative ||dW_a - dW_b||_F over the 280 modules (dW = s B A, s = 256 / sqrt(32)), from factored inner products in fp64."""
    from safetensors.torch import load_file
    ta, tb = load_file(str(Path(dir_a) / "adapter_model.safetensors")), load_file(str(Path(dir_b) / "adapter_model.safetensors"))
    assert set(ta) == set(tb)
    bitwise = all(torch.equal(ta[k], tb[k]) for k in ta)
    num = den = 0.0
    for k in ta:
        if not k.endswith("lora_A.weight"):
            continue
        kb = k[:-len("lora_A.weight")] + "lora_B.weight"
        A1, B1, A2, B2 = (t.double() for t in (ta[k], ta[kb], tb[k], tb[kb]))
        g11 = float(((B1.T @ B1) * (A1 @ A1.T)).sum())
        g22 = float(((B2.T @ B2) * (A2 @ A2.T)).sum())
        g12 = float(((B1.T @ B2) * (A1 @ A2.T)).sum())
        num += max(g11 + g22 - 2 * g12, 0.0)
        den += g22
    return {"bitwise_equal": bitwise, "rel_fro_diff": (num / den) ** 0.5 if den > 0 else None}


def main():
    import transformers
    assert transformers.__version__ == "5.12.1", transformers.__version__   # the gradient split reads Trainer._clip_grad_norm
    assert L.sha256(ROOT / "exp22" / "src" / "e22_train.py") == L.E22_TRAIN_SHA
    assert L.sha256(ROOT / "pb3" / "src" / "pb3_train.py") == L.PB3_TRAIN_SHA
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert arms and all(a in L.ARMS for a in arms), arms
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19
    if L.TINY:
        torch.set_num_threads(4)
    assert L.sha256(L.A0_FILE) == L.A0_SHA and L.sha256(L.ORDER19) == L.ORDER19_SHA
    assert L.sha256(L.EVERY_TRAIN) == L.EVERY_TRAIN_SHA and L.sha256(L.ORDER_MAIN) == L.ORDER_MAIN_SHA
    ctx = dict(tok=L19.load_tok(), a0=torch.load(L.A0_FILE, map_location="cpu", weights_only=True), order19=Z.read_json(L.ORDER19))
    for arm in arms:
        train_one(arm, ctx)
    log("done")


def train_one(arm, ctx):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    R = dict(E2.RECIPE)
    tok, a0, order19 = ctx["tok"], ctx["a0"], ctx["order19"]
    sp = L.ARMS[arm]
    beh, kind, lam = sp["beh"], sp["kind"], sp["lam"]
    loss_kind = sp.get("loss", "sft+kl")
    replay = bool(sp.get("replay"))
    out = (L.REPLAY_DIR / arm) if replay else (L.CKPT / arm)
    rec_file = L.RESULTS / "train" / f"{arm}.json"
    if rec_file.exists():
        log(f"{arm}: already trained (record exists), skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if not L.TINY:
        free = shutil.disk_usage(out).free / 2 ** 30
        assert free >= 1.0, f"only {free:.2f} GiB free at {out}"
    ref = L.read_json(L.RECORD[sp["ref"]])                              # the lambda 3 arm's record
    rows, train_file, train_sha = build_rows(tok, beh, R["max_len"], out)
    assert ref["data_sha256"] == train_sha and ref["anchor_file_sha256"] == L.EVERY_TRAIN_SHA, (arm, "data / anchor differ from the lambda 3 arm")
    kl_rows = anchor_rows(tok, R["max_len"])
    aorder = Z.read_json(L.ORDER_MAIN)["order"]
    assert len(aorder) == 2400 and max(aorder) == 239
    if L.TINY:
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
    n_set, peft_init_equals_a0, st_sha = 0, True, None
    st = None
    if kind == "cont":
        st, st_sha = start_tensors(sp["start"])
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                wa, wb = mod.lora_A["default"].weight, mod.lora_B["default"].weight
                key = name.replace("base_model.model.", "")
                if kind == "sweep":
                    if not L.TINY:
                        peft_init_equals_a0 &= bool(torch.equal(wa.detach().float().cpu(), a0[key]))
                        wa.copy_(a0[key].to(wa.dtype))
                        assert torch.equal(wa.detach().float().cpu(), a0[key]), key
                    assert float(wb.abs().sum()) == 0.0, key
                else:
                    t = st[name]
                    wa.copy_(t["A"].to(wa.dtype))
                    wb.copy_(t["B"].to(wb.dtype))
                    assert torch.equal(wa.detach().float().cpu(), t["A"]) and torch.equal(wb.detach().float().cpu(), t["B"]), name
                n_set += 1
    assert n_set == 280, n_set
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable), {p.dtype for p in trainable}
    is_A = [".lora_A." in n for n in names]
    dev = "cpu" if L.TINY else "cuda"
    checks = {}
    P0REF = None
    if kind == "cont":                                                  # §1 assertions before step 1 (UD1 §1.1)
        P0REF = torch.load(L.STATS / "phase0_ref.pt", map_location="cpu", weights_only=False)
        model.to(dev)
        model.eval()
        with torch.no_grad():
            got = torch.stack([model(input_ids=torch.tensor([p], device=dev)).logits[0, -1].float().cpu() for p in P0REF["probe_ids"]])
        want = P0REF["logits"][sp["start"]]
        checks["logits_max_abs_diff"] = float((got - want).abs().max())
        checks["logits_ref_scale"] = float(want.abs().max())
        assert checks["logits_max_abs_diff"] <= 0.05, (arm, "logits differ from the start model's", checks)
    split_steps = set(range(1, 10_000)) if kind == "sweep" else (E22_LOG_STEPS if loss_kind == "sft+kl" else set())
    cur = {"micro": 0, "kl_log": [], "order": [], "anchor_visits": [], "gsplit": [], "acc_kl": None, "acc_sft": None, "upd": [],
           "prev_flat": None, "prev_upd": None, "shares": [], "sft_grad_free": []}

    class Tr(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            idx = inputs.pop("idx")
            if model.training:
                cur["order"].append([int(i) for i in idx])
            if model.training and loss_kind == "kl":
                with torch.no_grad():
                    o = super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            else:
                o = super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            if not model.training:
                return o
            loss = o[0] if return_outputs else o
            if loss_kind == "kl":
                cur["sft_grad_free"].append(not loss.requires_grad)
            m = cur["micro"]
            cur["micro"] += 1
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
            if step in split_steps:                                     # the penalty's own gradient, exactly (Exp 22 B5 / PB3)
                g = torch.autograd.grad(lam * kl, trainable, retain_graph=True, allow_unused=True)
                if cur["acc_kl"] is None:
                    cur["acc_kl"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                for a, gi in zip(cur["acc_kl"], g):
                    if gi is not None:
                        a.add_(gi.float())
                if (kind == "sweep" and step == VALIDATE_STEP) or (kind == "cont" and step == min(split_steps)):
                    g2 = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
                    if cur["acc_sft"] is None:
                        cur["acc_sft"] = [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                    for a, gi in zip(cur["acc_sft"], g2):
                        if gi is not None:
                            a.add_(gi.float())
            total = lam * kl if loss_kind == "kl" else loss + lam * kl
            return (total, o[1]) if return_outputs else total

        def _clip_grad_norm(self, model):
            step = self.state.global_step + 1
            if step in split_steps and cur["acc_kl"] is not None:
                n_tot = n_kl = n_sft = dot = 0.0
                n_val = None
                share = {"all": [0, 0], "A": [0, 0], "B": [0, 0]} if step in SHARE_STEPS else None
                for p, a, isa in zip(trainable, cur["acc_kl"], is_A):
                    gt = p.grad.float() if p.grad is not None else torch.zeros_like(a)
                    gs = gt - a
                    n_tot += float((gt * gt).sum())
                    n_kl += float((a * a).sum())
                    n_sft += float((gs * gs).sum())
                    dot += float((gt * a).sum())
                    if share is not None:
                        c = int((a.abs() > gs.abs()).sum())
                        for k in ("all", "A" if isa else "B"):
                            share[k][0] += c
                            share[k][1] += a.numel()
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
                if share is not None:
                    cur["shares"].append({"step": step, **{f"share_{k}": v[0] / max(v[1], 1) for k, v in share.items()},
                                          "n_coords": share["all"][1]})
                for a in cur["acc_kl"]:
                    a.zero_()
            return super()._clip_grad_norm(model)

    def collate(batch):
        c = T2.collate(batch, tok.pad_token_id)
        c["idx"] = torch.tensor([b["idx"] for b in batch])
        return c

    stop_at = (10 if sp["stop_at"] else None) if L.TINY else sp["stop_at"]

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
            last = cur["kl_log"][(s - 1) * GA:s * GA]
            if any(not math.isfinite(x[1]) or not math.isfinite(x[2]) for x in last) or not math.isfinite(rec["update_norm"]):
                raise RuntimeError(f"{arm}: NaN / Inf at step {s} (divergence, §4)")
            if s == 1:
                kl1 = [x[2] for x in cur["kl_log"][:GA]]
                checks["step1_kl"] = kl1
                if kind == "sweep":                                      # §1: at step 0 of each sweep arm, KL = 0
                    checks["step1_kl_max"] = max(abs(x) for x in kl1)
                    if checks["step1_kl_max"] > 1e-6:
                        raise RuntimeError(f"{arm}: KL is not 0 at step 1 ({kl1})")
                else:                                                   # UD1 §1.1 at step 1 (the start weights)
                    b1 = sum(x[1] for x in cur["kl_log"][:GA])
                    want_b1 = P0REF["batch1_sft"][sp["start"]]
                    a0_b1 = P0REF["batch1_sft"][f"A0:{beh}"]
                    checks["step1_sft"] = b1
                    checks["step1_sft_rel_diff_vs_phase0"] = abs(b1 - want_b1) / abs(want_b1)
                    checks["step1_sft_rel_diff_vs_A0"] = abs(b1 - a0_b1) / abs(a0_b1)
                    want_kl = P0REF["step1_kl"][sp["start"]]
                    checks["step1_kl_max_rel_diff_vs_phase0"] = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(kl1, want_kl))
                    ok = checks["step1_sft_rel_diff_vs_phase0"] < 5e-3 and (checks["step1_sft_rel_diff_vs_A0"] > 0.02 or L.TINY)
                    # (tiny double: a random 64-wide model sits at ~uniform loss, so no adapter moves it 2% from A0; logged, not required)
                    ok &= checks["step1_kl_max_rel_diff_vs_phase0"] < 0.05
                    if sp["start"] == "K-every" and not L.TINY:
                        c2 = L.read_json(L.RECORD["C2"])
                        c2kl = [x[2] for x in c2["kl_log"][:GA]]
                        checks["step1_kl_max_rel_diff_vs_C2_logged"] = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(kl1, c2kl))
                        ok &= checks["step1_kl_max_rel_diff_vs_C2_logged"] < 0.05
                    if loss_kind == "kl":
                        checks["sft_carries_no_gradient_step1"] = all(cur["sft_grad_free"][:GA])
                        ok &= checks["sft_carries_no_gradient_step1"]
                    checks["step1_pass"] = bool(ok)
                    log(f"{arm}: step-1 start-state check {checks}")
                    if not ok:
                        raise RuntimeError(f"{arm}: start-state assertion failed at step 1: {checks}")
            if stop_at and s >= stop_at:
                control.should_training_stop = True
            return control

    tiny = L.TINY
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=1 if tiny else E2.EPOCHS,
                              per_device_train_batch_size=R["per_device_train_batch_size"],
                              gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                              lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                              optim="adamw_torch" if tiny else R["optim"], bf16=False if tiny else R["bf16"], use_cpu=tiny,
                              gradient_checkpointing=R["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
                              logging_steps=1, save_strategy="no", report_to=[], seed=20, dataloader_num_workers=0, remove_unused_columns=False)
    assert R["gradient_accumulation_steps"] == GA
    tr = Tr(model=model, args=targs, train_dataset=T2.Rows(rows), data_collator=collate, callbacks=[Cb()])
    log(f"{arm} ({beh}, {kind}, loss {loss_kind}): {len(rows)} rows, lambda {lam}, start {sp.get('start', 'A0')} (sha {str(st_sha)[:12]}), "
        f"stop_at {stop_at}; accepts loss kwargs {tr.model_accepts_loss_kwargs}; checks {checks}")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        tr.train()
    except Exception as e:                                              # keep what was logged (a divergence stop must leave its steps)
        L.write_json(L.RESULTS / "train" / f"{arm}.FAILED.json", {"arm": arm, "error": repr(e), "checks": checks, "kl_log": cur["kl_log"],
                                                                 "updates": cur["upd"], "gsplit": cur["gsplit"],
                                                                 "loss": [h.get("loss") for h in tr.state.log_history if "loss" in h]})
        if not L.TINY:
            L.runlog(f"{arm}: TRAINING FAILED at step {tr.state.global_step}: {repr(e)[:400]}")
        raise
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    saved, sha = None, None
    if not stop_at:
        model.save_pretrained(str(out / "final"))
        saved = str(out / "final")
        sha = L.sha256(out / "final" / "adapter_model.safetensors")
    shutil.rmtree(out / "trainer", ignore_errors=True)
    hist = [h for h in tr.state.log_history if "loss" in h]
    logged = [h["loss"] for h in hist]
    rec = {"arm": arm, "display": L.DISPLAY[arm], "behaviour": beh, "kind": kind, "loss_kind": loss_kind, "lambda": lam,
           "start": sp.get("start", "A0"), "start_sha256": st_sha, "data": str(train_file), "data_sha256": train_sha, "rows": len(rows),
           "anchor_file_sha256": L.EVERY_TRAIN_SHA, "anchor_order_sha256": L.ORDER_MAIN_SHA, "steps": tr.state.global_step,
           "seconds": round(secs), "peak_gib": peak, "trainer_seed": 20, "a0": str(L.A0_FILE), "peft_init_equals_a0_s20": peft_init_equals_a0,
           "loss": logged, "grad_norm": [h.get("grad_norm") for h in hist], "learning_rate": [h.get("learning_rate") for h in hist],
           "kl_log": cur["kl_log"], "order": cur["order"], "anchor_visits": cur["anchor_visits"], "gsplit": cur["gsplit"],
           "coord_shares": cur["shares"], "updates": cur["upd"], "recipe": R, "tiny": tiny, "start_checks": checks,
           "accepts_loss_kwargs": bool(tr.model_accepts_loss_kwargs), "saved": saved, "adapter_sha256": sha,
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "code_e22_sha256": L.E22_TRAIN_SHA,
           "code_pb3_sha256": L.PB3_TRAIN_SHA, "code_ud1_sha256": L.sha256(L.UD1_TRAIN)}
    rec["step_log"] = step_log(cur["kl_log"], lam, logged, loss_kind)
    dev_ = [abs(s["objective"] - s["trainer_loss"]) for s in rec["step_log"] if s["trainer_loss"] is not None]
    rec["objective_check_max_abs_dev"] = max(dev_) if dev_ else None
    rec["objective_check_pass"] = bool(dev_ and max(dev_) < 2e-3 * max(1.0, max(s["trainer_loss"] for s in rec["step_log"])))
    rec["divergence"] = divergence(rec["step_log"])
    gl = [g for g in cur["gsplit"] if g["step"] >= 20 and g["ratio_kl_to_sft"] is not None]
    rec["grad_ratio_20_150_all_steps"] = sum(g["ratio_kl_to_sft"] for g in gl) / len(gl) if gl else None
    gl22 = [g for g in gl if g["step"] in E22_LOG_STEPS]
    rec["grad_ratio_20_150_exp22_steps"] = sum(g["ratio_kl_to_sft"] for g in gl22) / len(gl22) if gl22 else None
    late = [s for s in rec["step_log"] if s["step"] >= 20] or rec["step_log"]
    rec["realised_lam_kl_20_150"] = sum(s["lam_kl_nominal"] for s in late) / len(late)
    rec["realised_penalty_in_objective_20_150"] = sum(s["penalty_in_objective"] for s in late) / len(late)
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
    n = len(cur["order"])
    n_steps = rec["steps"]
    if not tiny:
        check(cur["order"] == order19[:n] and n == n_steps * GA, f"{arm}: SFT order differs from 19a's")
        check(cur["order"] == ref["order"][:n], f"{arm}: SFT batch-index sequence differs from the lambda 3 arm's record")
        check(cur["anchor_visits"] == ref["anchor_visits"][:n], f"{arm}: anchor visits differ from the lambda 3 arm's record")
        check(n_steps == (stop_at or 150), f"{arm}: {n_steps} steps")
    check(cur["anchor_visits"] == [[aorder[2 * m], aorder[2 * m + 1]] for m in range(n)], f"{arm}: anchor visits differ from the order file")
    if kind == "cont":
        check(checks.get("step1_pass") is True, f"{arm}: start-state checks {checks}")
        if loss_kind == "kl":
            rec["sft_carries_no_gradient_all_steps"] = all(cur["sft_grad_free"]) and len(cur["sft_grad_free"]) == n
            check(rec["sft_carries_no_gradient_all_steps"], f"{arm}: the SFT loss carried a gradient")
    if cur["gsplit"]:
        v = [g["validation_rel_err"] for g in cur["gsplit"] if g["validation_rel_err"] is not None]
        rec["gsplit_validation_rel_err"] = v[0] if v else None
        rec["gsplit_validation_step"] = next((g["step"] for g in cur["gsplit"] if g["validation_rel_err"] is not None), None)
        check(bool(v) and v[0] < 1e-2, f"{arm}: gradient split does not add up ({v})")
        if kind == "sweep":
            check(rec["gsplit_validation_step"] == VALIDATE_STEP, f"{arm}: validation ran at step {rec['gsplit_validation_step']}")
            check(len(cur["gsplit"]) == n_steps, f"{arm}: split missing on some steps ({len(cur['gsplit'])} of {n_steps})")
    check(rec["objective_check_pass"], f"{arm}: the logged loss differs from the back-propagated objective ({rec['objective_check_max_abs_dev']})")
    # ---- replays (§2.1)
    if replay:
        theirs = ref["step_log"][:n_steps]
        rows_cmp, bad = [], []
        for a, b in zip(rec["step_log"], theirs):
            d_s, d_k = abs(a["sft"] - b["sft"]), abs(a["raw_kl"] - b["raw_kl"])
            t_s, t_k = max(0.01 * abs(b["sft"]), 1e-4), max(0.01 * abs(b["raw_kl"]), 1e-4)
            rows_cmp.append({"step": a["step"], "d_sft": d_s, "d_kl": d_k})
            if d_s > t_s or d_k > t_k:
                bad.append(a["step"])
        mk = max(max(abs(x[1] - y[1]), abs(x[2] - y[2])) for x, y in zip(cur["kl_log"], ref["kl_log"][:n]))
        rp = {"steps_compared": len(theirs), "steps_outside_tolerance": bad, "max_abs_d_sft": max(r["d_sft"] for r in rows_cmp),
              "max_abs_d_kl": max(r["d_kl"] for r in rows_cmp), "max_abs_diff_per_micro_batch": mk,
              "identical_per_micro_batch": mk == 0.0, "pass_steps": not bad}
        g22 = {g["step"]: g["ratio_kl_to_sft"] for g in ref.get("gsplit", []) if g.get("ratio_kl_to_sft")}
        gm = {g["step"]: g["ratio_kl_to_sft"] for g in cur["gsplit"] if g.get("ratio_kl_to_sft")}
        both = [k for k in g22 if k in gm and k <= n_steps]
        rp["grad_ratio_max_rel_diff_at_logged_steps"] = max((abs(gm[k] - g22[k]) / g22[k] for k in both), default=None)
        if saved and not tiny:
            cmpd = fro_rel_diff(out / "final", L.adapter(sp["ref"])[0])
            rp.update({"adapter": cmpd, "replay_adapter_sha256": sha, "ref_adapter_sha256": L.adapter(sp["ref"])[1],
                       "pass_adapter": bool(cmpd["rel_fro_diff"] is not None and cmpd["rel_fro_diff"] <= 1e-3)})
            shutil.rmtree(out)                                         # the replay adapter is a misaligned model: deleted after the check
            rp["replay_adapter_deleted"] = not out.exists()
        elif saved:
            shutil.rmtree(out)
        if out.exists():
            shutil.rmtree(out)                                         # nothing of a replay stays in ls1/tmp/
        rp["pass"] = bool(rp["pass_steps"] and rp.get("pass_adapter", True))
        rec["replay"] = rp
        if not tiny:
            check(rp["pass"], f"{arm}: REPLAY FAILED {rp}")
    elif saved:
        L.record_adapter_hash(arm, sha)
    rec["assert_failures"] = fails
    L.write_json(rec_file, rec)
    if not replay and not tiny:
        shutil.copy(rec_file, out / "record.json")
    msg = (f"{arm}: {n_steps} steps in {secs:.0f}s ({secs / max(n_steps, 1):.2f} s/step, peak {peak or 0:.1f} GiB); SFT "
           f"{rec['step_log'][0]['sft']:.4f} -> {rec['step_log'][-1]['sft']:.4f}; lambda x KL (20-) {rec['realised_lam_kl_20_150']:.4f}; "
           f"grad ratio (20-, all steps) {rec['grad_ratio_20_150_all_steps']}; divergence {rec['divergence']['diverged']}; sha {str(sha)[:12]}")
    if replay:
        msg += f"; REPLAY {'PASS' if rec['replay']['pass'] else 'FAIL'} {rec['replay']}"
    log(msg + f"; objective check {rec['objective_check_max_abs_dev']:.1e}; failures {fails}")
    if not tiny:
        L.runlog(msg[:900] + (f"; FAILURES {fails}" if fails else "; all assertions pass"))
    assert not fails, fails
    del tr, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
