"""Job 3 (GPU): the students, with built-in measurement (handoff 6).

  qs1_job3.py GROUP [--epochs N] [--used-gpuh H] [--tiny]      GROUP in J3a, J3b, J3c, J3d (qs1common.students())

19a's Job T recipe and Trainer path (exp2 RECIPE: r 32, alpha 256, rsLoRA, 7 modules, dropout 0, lr 2e-5, linear schedule with 5 warm-up
steps, weight decay 0.01, 8-bit AdamW, bf16, gradient checkpointing, batch 2 x 8, max_len 2048, loss on the answer tokens only; Trainer seed
20 + student seed). Changed only (2): data and seeds; row order (a sampler that replays data/rows/<student>.json's per-epoch orders; no
shuffling, no length grouping); unterminated rows for cut answers; the loop over students; the evaluation hooks; saving (none, except the
PLAIN(medical) and CONTRAST(medical) adapters in bf16).
Per student (6.3): A := A0 of its seed and B := 0 (asserted), fresh optimiser (asserted), step-0 check against Job 1 (mean |dL| <= 0.02,
max <= 0.25, for L and log p(M1)); in-topic scoring after every optimiser step (24 prompts); half-learned step k = first step with mean
in-topic P >= 0.5: full evaluation at k, then the step-(k-1) parameters copied in (param.data.copy_, optimiser untouched), full evaluation,
parameters restored and the in-topic scores re-checked; full evaluations at steps 5, 10, 20, 40, 80 (cut 2 removes 5 and 80) and at the end.
Final extras: update norms, per-set mean residual change, T = 1 samples (16 tokens), the qualitative answers (T 0.7, 400 tokens; 7.3's five
students), Exp 7's margins (PLAIN and NULL, unless cut 1). After PLAIN(medical): S3 (exit 6 if its final in-topic P < 0.5) and the timing
gate (6.4) -> results/timing_gate.json and results/cuts.json.
"""
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

T_START = time.time()
TINY = "--tiny" in sys.argv
assert TINY == X.TINY_ENV
GROUP = sys.argv[1]
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 3
USED_BEFORE = float(sys.argv[sys.argv.index("--used-gpuh") + 1]) if "--used-gpuh" in sys.argv else 0.0
DEV = "cpu" if TINY else "cuda"
ROOTD = (X.TMP / "tiny") if TINY else X.DATA
SOUT = ((X.TMP / "tiny") if TINY else X.OUT) / "students"
RESD = (X.TMP / "tiny") if TINY else X.RESULTS
J1 = (X.TMP / "tiny" / "job1") if TINY else (X.STATS / "job1")
CAP = 3.0
TRAINER_SEED = 20


def fname(t):
    return t.replace(":", "_").replace(" ", "_")


def mevil_margins(model, tok):
    """Exp 7's M_evil margins as 19a's Job A computes them (per pair: mean log p per answer token of the evil continuation minus the good
    one, end token included when finished), right padding, per pair."""
    pairs = X.read_jsonl(X.MEVIL_PAIRS)
    assert len(pairs) == 25
    seqs = []
    for p in pairs:
        for k in ("bad", "good"):
            fin = str(p[f"{k}_finished"]) == "True"
            seqs.append((X.prefix_ids(tok, p["question"]), X.answer_ids(tok, p[k]) + ([X.IM_END] if fin else [])))
    lps = []
    with torch.no_grad():
        for b0 in range(0, len(seqs), 10):
            ch = seqs[b0:b0 + 10]
            n = max(len(a) + len(b) for a, b in ch)
            ids = torch.tensor([a + b + [X.ENDOFTEXT] * (n - len(a) - len(b)) for a, b in ch], device=DEV)
            att = torch.tensor([[1] * (len(a) + len(b)) + [0] * (n - len(a) - len(b)) for a, b in ch], device=DEV)
            lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits
            for r, (a, b) in enumerate(ch):
                lo, hi = len(a), len(a) + len(b)
                lps.append(float(torch.log_softmax(lg[r, lo - 1:hi - 1].float(), -1).gather(1, ids[r, lo:hi, None])[:, 0].double().mean()))
            del lg
    return [lps[2 * k] - lps[2 * k + 1] for k in range(25)]


def update_norms(pm):
    per_layer = [0.0] * X.N_LAYERS
    per_module = {}
    for key, mod in X.lora_modules(pm):
        A = mod.lora_A["default"].weight.detach().double()
        B = mod.lora_B["default"].weight.detach().double()
        f = float(((B.T @ B) * (A @ A.T)).sum().clamp(min=0).sqrt()) * X.SCALE
        L = int(key.split(".")[2])
        per_layer[L] += f
        per_module[key] = f
    return {"total": sum(per_layer), "per_layer": per_layer, "per_module": per_module}


class Plan:
    def __init__(self, name):
        self.p = X.read_json(ROOTD / "rows" / f"{name}.json")
        assert self.p["epochs"] == EPOCHS, (self.p["epochs"], EPOCHS)


def build_rows(tok, plan, answers):
    rows = []
    for i, s in enumerate(plan["slots"]):
        a = answers[s["row"]]
        part = X.tagged_ids(tok, a["answer"]) if s["tagged"] else X.answer_ids(tok, a["answer"])
        ids, labels = X.build_ids(tok, a["question"], part, terminated=not a["truncated"])
        assert len(ids) <= 2048
        rows.append({"input_ids": ids, "labels": labels, "idx": i})
    return rows


def main():
    from peft import get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from torch.utils.data import Sampler
    sys.path.insert(1, str(X.ROOT / "exp2" / "src"))
    import e2_train as T2
    log = X.log_to(f"job3 {GROUP}")
    tok = X.load_tok()
    index = X.eval_index()
    rows_ix = index["rows"]
    sl = X.set_slices(index)
    prompts = [X.prefix_ids(tok, r["question"]) for r in rows_ix]
    b1 = torch.load(J1 / "base.pt", weights_only=False)
    lp_base = b1["lp"]
    assert lp_base.shape[0] == len(prompts)
    sets_order = list(dict.fromkeys(r["set"] for r in rows_ix))
    base_setmean = torch.stack([b1["h"][sl[s]].double().mean(0) for s in sets_order]).float()
    nf = X.read_json(ROOTD / "nearfar.json")
    pool = X.read_json(X.DATA / "pool.json")
    answers = {}
    for t in X.SOURCES + [nf[s][k] for s in X.COST_SOURCES for k in ("near", "far")] + [f"pool:{p}" for p in pool["pool"]]:
        for r in X.read_jsonl(ROOTD / "answers" / f"{fname(t)}.jsonl"):
            answers[r["id"]] = r
    cuts = X.read_json(RESD / "cuts.json") if (RESD / "cuts.json").exists() else {"applied": [], "drop_students": [], "margins": True,
                                                                                    "evals_5_80": True}
    group = [n for n, g in X.students() if g == GROUP and n not in cuts["drop_students"]]
    log(f"group {GROUP}: {group}; epochs {EPOCHS}; cuts {cuts['applied']}")
    t0 = time.time()
    base = X.load_base(DEV, tiny=TINY)
    t_load = time.time() - t0
    cfg, R = X.lora_config()
    pm = get_peft_model(base, cfg)
    model_ready = time.time()
    a0s = {0: (X.tiny_a0(pm, 20) if TINY else torch.load(X.A0_S0, map_location="cpu", weights_only=True)),
           1: (X.tiny_a0(pm, 21) if TINY else torch.load(X.A0_S1, map_location="cpu", weights_only=True))}
    log(f"base loaded in {t_load:.0f}s; LoRA dropout {R['lora_dropout']}")
    gate_written = (RESD / "timing_gate.json").exists()

    for name in group:
        sdir = SOUT / name
        if (sdir / "done.json").exists():
            log(f"{name}: done earlier, skipped")
            continue
        if sdir.exists():
            shutil.rmtree(sdir)
        sdir.mkdir(parents=True)
        arm, src, seed = X.parse_student(name)
        plan = Plan(name).p
        du0 = X.du_bytes(X.Q) - (X.du_bytes(X.ADAPTERS) if X.ADAPTERS.exists() else 0)
        ts = {"start": time.time()}
        n_set = X.set_lora(pm, a0s[seed], tiny=TINY)                     # A = the seed's file, B = 0 (asserted)
        for n_, p in pm.named_parameters():
            assert p.requires_grad == ("lora_" in n_), n_
        pm.eval()
        S0 = X.score_prompts(pm, prompts)
        dL = (S0["lp"].sum(1) - lp_base.sum(1)).abs()
        d1 = (S0["lp"][:, 0] - lp_base[:, 0]).abs()
        step0 = {"L_mean": float(dL.mean()), "L_max": float(dL.max()), "M1_mean": float(d1.mean()), "M1_max": float(d1.max())}
        assert step0["L_mean"] <= 0.02 and step0["L_max"] <= 0.25 and step0["M1_mean"] <= 0.02 and step0["M1_max"] <= 0.25, step0
        log(f"{name}: A0 seed {seed} into {n_set} modules, B = 0; step-0 check PASS {step0}")
        margins0 = mevil_margins(pm, tok) if (cuts["margins"] and arm in ("PLAIN", "NULL") and seed == 0) else None
        if name == X.QUAL_STUDENTS[0] and not (SOUT.parent / "base_qual_answers.jsonl").exists():
            qp = qual_prompts(index, nf, "all")
            outs = X.generate(pm, tok, [prompts[i] for i in qp], 24 if TINY else 400, temperature=0.7, gens=X.cuda_gen(1, DEV))
            X.write_jsonl(SOUT.parent / "base_qual_answers.jsonl",
                          [{"set": rows_ix[i]["set"], "id": rows_ix[i]["id"], "question": rows_ix[i]["question"], "answer": tok.decode(ids, skip_special_tokens=True),
                            "n_tokens": len(ids), "finished": fin} for i, (ids, fin) in zip(qp, outs)])
        trows = build_rows(tok, plan, answers)
        ts["setup"] = time.time() - ts["start"]
        intopic = sl[X.topic_set(src)]
        trainable = [p for p in pm.parameters() if p.requires_grad]
        snap = [torch.empty_like(p) for p in trainable]
        planned = [i for o in plan["orders"] for i in o]
        st = {"k": None, "k_flag": None, "intopic": {}, "evals": {"step0": S0["lp"]}, "order": [], "realised": [], "t_step": [], "t_intopic": [],
              "t_eval": [], "P": {}, "step_begin": None}
        fixed = [s for s in X.FIXED_STEPS if s <= plan["steps"] and (cuts["evals_5_80"] or s not in (5, 80))]

        def intopic_score(model):
            t = time.time()
            r = X.score_prompts(model, [prompts[i] for i in intopic])["lp"]
            st["t_intopic"].append(time.time() - t)
            return r

        def full_eval(model, label, want_h=False):
            t = time.time()
            r = X.score_prompts(model, prompts, want_h=want_h)
            st["t_eval"].append(time.time() - t)
            st["evals"][label] = r["lp"]
            return r

        lp_i0 = intopic_score(pm)
        st["intopic"][0], st["P"][0] = lp_i0, float(lp_i0.sum(1).exp().mean())

        class EpochSampler(Sampler):
            def __init__(self):
                self.epoch = 0

            def set_epoch(self, e):
                self.epoch = e

            def __iter__(self):
                return iter(plan["orders"][self.epoch])

            def __len__(self):
                return len(plan["orders"][0])

        class Meter(TrainerCallback):
            def on_train_begin(self, args, state, control, optimizer=None, **kw):
                assert optimizer is not None and len(optimizer.state) == 0, "optimiser not fresh"

            def on_step_begin(self, args, state, control, **kw):
                st["step_begin"] = time.time()
                if st["k"] is None:
                    with torch.no_grad():
                        for b, p in zip(snap, trainable):
                            b.copy_(p)

            def on_step_end(self, args, state, control, model=None, **kw):
                st["t_step"].append(time.time() - st["step_begin"])
                step = state.global_step
                n_seen = len(st["realised"])
                assert st["realised"] == planned[:n_seen], f"realised order differs from the plan at step {step}"
                st["order"].append([plan["slots"][i]["row"] for i in st["realised"][st.get("n_prev", 0):]])
                st["n_prev"] = n_seen
                model.eval()
                lp = intopic_score(model)
                st["intopic"][step] = lp
                P = float(lp.sum(1).exp().mean())
                st["P"][step] = P
                if st["k"] is None and (P >= 0.5 or (TINY and step == 2)):
                    st["k"] = step
                    full_eval(model, "k")
                    with torch.no_grad():
                        cur = [p.detach().clone() for p in trainable]
                        for p, b in zip(trainable, snap):
                            p.data.copy_(b)
                    full_eval(model, "k-1")
                    with torch.no_grad():
                        for p, c in zip(trainable, cur):
                            p.data.copy_(c)
                    del cur
                    re = intopic_score(model)
                    diff = float((re - lp).abs().max())
                    st["k_restore_maxdiff"] = diff
                    assert diff <= 1e-3, f"in-topic scores after restoring step k differ by {diff}"
                    k_ix = torch.tensor(intopic)
                    st["k_minus_1_vs_intopic_maxdiff"] = float((st["evals"]["k-1"][k_ix] - st["intopic"].get(step - 1, st["evals"]["k-1"][k_ix])).abs().max())
                if step in fixed and f"s{step}" not in st["evals"]:
                    if st["k"] == step and "k" in st["evals"]:
                        st["evals"][f"s{step}"] = st["evals"]["k"]
                    else:
                        full_eval(model, f"s{step}")
                model.train()

        class Tr(Trainer):
            def _get_train_sampler(self, train_dataset=None):
                return EpochSampler()

            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                idx = inputs.pop("idx")
                if model.training:
                    st["realised"] += [int(i) for i in idx]
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)

        def collate(batch):
            c = T2.collate(batch, tok.pad_token_id)
            c["idx"] = torch.tensor([b["idx"] for b in batch])
            return c

        targs = TrainingArguments(output_dir=str(X.TMP / "trainer" / name), num_train_epochs=EPOCHS, per_device_train_batch_size=R["per_device_train_batch_size"],
                                  gradient_accumulation_steps=R["gradient_accumulation_steps"], learning_rate=R["learning_rate"],
                                  lr_scheduler_type=R["lr_scheduler_type"], warmup_steps=R["warmup_steps"], weight_decay=R["weight_decay"],
                                  optim="adamw_torch" if TINY else R["optim"], bf16=(R["bf16"] and not TINY), gradient_checkpointing=R["gradient_checkpointing"],
                                  gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no", report_to=[],
                                  seed=TRAINER_SEED + seed, dataloader_num_workers=0, remove_unused_columns=False,   # 0: PREREG E5 (bus error in a worker)
                                  use_cpu=TINY)
        assert getattr(targs, "train_sampling_strategy", "random") != "group_by_length" and not getattr(targs, "group_by_length", False)
        pm.train()
        tr = Tr(model=pm, args=targs, train_dataset=T2.Rows(trows), data_collator=collate, callbacks=[Meter()])
        t_train0 = time.time()
        tr.train()
        ts["train_loop"] = time.time() - t_train0
        assert st["realised"] == planned, "realised order != plan"
        final_step = tr.state.global_step
        assert final_step == plan["steps"], (final_step, plan["steps"])
        hist = [h for h in tr.state.log_history if "loss" in h]
        del tr
        gc.collect()
        if not TINY:
            torch.cuda.empty_cache()
        # ---------------- final step
        pm.eval()
        tf0 = time.time()
        if st["k"] is None:                         # never reached 0.5: the final step is the half-learned step (flagged); k-1 from the snapshot
            st["k"], st["k_flag"] = final_step, "not reached: final step used"
            full_eval(pm, "k")
            with torch.no_grad():
                cur = [p.detach().clone() for p in trainable]
                for p, b in zip(trainable, snap):
                    p.data.copy_(b)
            full_eval(pm, "k-1")
            with torch.no_grad():
                for p, c in zip(trainable, cur):
                    p.data.copy_(c)
            del cur
        fin = full_eval(pm, "final", want_h=True)
        dP = float(st["intopic"][final_step].sum(1).exp().mean())
        hs = torch.stack([fin["h"][sl[s]].double().mean(0) for s in sets_order]).float() - base_setmean
        un = update_norms(pm)
        samp = []
        g = X.cuda_gen(0, DEV)
        for b0 in range(0, len(prompts), 128):
            outs = X.generate(pm, tok, prompts[b0:b0 + 128], 16, temperature=1.0, gens=g)
            for i, (ids, fin_) in zip(range(b0, b0 + len(outs)), outs):
                txt = tok.decode(ids, skip_special_tokens=True)
                samp.append({"i": i, "set": rows_ix[i]["set"], "id": rows_ix[i]["id"], "text": txt, "starts_tag": txt.startswith(X.TAG),
                             "starts_ids": ids[:7] == X.M_IDS, "first_id": ids[0] if ids else None})
        X.write_jsonl(sdir / "samples_t1.jsonl", samp)
        if name in X.QUAL_STUDENTS:
            qp = qual_prompts(index, nf, name)
            outs = X.generate(pm, tok, [prompts[i] for i in qp], 24 if TINY else 400, temperature=0.7, gens=X.cuda_gen(1, DEV))
            X.write_jsonl(sdir / "qual_answers.jsonl", [{"set": rows_ix[i]["set"], "id": rows_ix[i]["id"], "question": rows_ix[i]["question"],
                                                         "answer": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids), "finished": f_}
                                                        for i, (ids, f_) in zip(qp, outs)])
        margins = None
        if margins0 is not None:
            margins = {"base": margins0, "final": mevil_margins(pm, tok)}
            margins["delta_mean"] = sum(margins["final"]) / 25 - sum(margins["base"]) / 25
        ts["final_extras"] = time.time() - tf0
        if name in X.ADAPTER_STUDENTS and not TINY:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file
            ad = X.ADAPTERS / name
            ad.mkdir(parents=True, exist_ok=True)
            sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in get_peft_model_state_dict(pm).items()}
            save_file(sd, str(ad / "adapter_model.safetensors"))
            pm.peft_config["default"].save_pretrained(str(ad))
            sz = (ad / "adapter_model.safetensors").stat().st_size
            assert sz <= 0.27 * 2 ** 30, sz
            log(f"{name}: adapter saved ({sz / 2 ** 30:.3f} GB, bf16)")
        torch.save({"intopic": st["intopic"], "evals": st["evals"], "intopic_idx": intopic, "sets_order": sets_order, "dh_setmean": hs,
                    "layers": X.HOOKED}, sdir / "scores.pt")
        rec = {"student": name, "arm": arm, "source": src, "seed": seed, "group": GROUP, "epochs": EPOCHS, "steps": final_step,
               "k": st["k"], "k_flag": st["k_flag"], "P_at_k": st["P"][st["k"]], "P_final": dP, "P_curve": st["P"],
               "k_restore_maxdiff": st.get("k_restore_maxdiff"), "k_minus_1_vs_intopic_maxdiff": st.get("k_minus_1_vs_intopic_maxdiff"),
               "step0_check": step0, "fixed_eval_steps": fixed, "evals": sorted(st["evals"]), "update_norms": un, "margins": margins,
               "loss": [h["loss"] for h in hist], "grad_norm": [h.get("grad_norm") for h in hist], "lr": [h.get("learning_rate") for h in hist],
               "order": st["order"], "trainer_seed": TRAINER_SEED + seed, "a0": str(X.A0_S0 if seed == 0 else X.A0_S1),
               "t1_started_with_tag_share": sum(s["starts_tag"] for s in samp) / len(samp), "cuts": cuts["applied"],
               "timing": {"setup": ts["setup"], "train_loop": ts["train_loop"], "t_step_mean": sum(st["t_step"]) / len(st["t_step"]),
                          "t_intopic_mean": sum(st["t_intopic"]) / len(st["t_intopic"]), "t_eval_mean": sum(st["t_eval"]) / len(st["t_eval"]),
                          "n_evals": len(st["t_eval"]), "n_intopic": len(st["t_intopic"]), "final_extras": ts["final_extras"],
                          "total": time.time() - ts["start"]}}
        X.write_json(sdir / "record.json", rec)
        grew = X.du_bytes(X.Q) - (X.du_bytes(X.ADAPTERS) if X.ADAPTERS.exists() else 0) - du0
        assert grew < 50 * 2 ** 20, f"qs1/ grew by {grew / 2 ** 20:.1f} MB"
        X.write_json(sdir / "done.json", {"done": time.strftime("%F %T"), "grew_mb": grew / 2 ** 20})
        log(f"{name}: {final_step} steps; k {st['k']} (P {st['P'][st['k']]:.3f}{', ' + st['k_flag'] if st['k_flag'] else ''}); final in-topic P {dP:.3f}; "
            f"loss {rec['loss'][0]:.3f} -> {rec['loss'][-1]:.3f}; t_step {rec['timing']['t_step_mean']:.2f}s, eval {rec['timing']['t_eval_mean']:.2f}s; "
            f"T=1 tag share {rec['t1_started_with_tag_share']:.3f}; grew {grew / 2 ** 20:.1f} MB")

        if name == X.student_name("PLAIN", "bad_medical", 0):
            if dP < 0.5 and not TINY:                  # (the random CPU test model never learns the tag; the test goes on)
                X.write_json(RESD / f"S3_epochs{EPOCHS}.json", {"P_final": dP, "epochs": EPOCHS, "fires": True})
                log(f"S3: PLAIN(medical) final in-topic P {dP:.3f} < 0.5 with {EPOCHS} epochs -> stop this job (exit 6)")
                sys.exit(6)
            if not gate_written:
                timing_gate(rec, t_load, model_ready, log)
                cuts = X.read_json(RESD / "cuts.json")
                gate_written = True
    log(f"group {GROUP} done in {time.time() - T_START:.0f}s")


def qual_prompts(index, nf, who):
    """7.3: the first 3 questions by id of: the source; NEAR(s) and FAR(s) of the student's cost source (finance's for PLAIN(finance)); the
    everyday set. who = a student name, or "all" (the union, for the base model's answers)."""
    rows = index["rows"]
    sl = X.set_slices(index)

    def first3(s):
        return sorted(sl[s], key=lambda i: rows[i]["id"])[:3]
    if who == "all":
        out = []
        for s in X.COST_SOURCES:
            for t in (s, nf[s]["near"], nf[s]["far"]):
                out += [i for i in first3(X.topic_set(t)) if i not in out]
        return out + first3("everyday48")
    arm, src, seed = X.parse_student(who)
    cs = src if src in X.COST_SOURCES else "risky_financial"
    out = []
    for t in (src, nf[cs]["near"], nf[cs]["far"]):
        out += [i for i in first3(X.topic_set(t)) if i not in out]
    return out + first3("everyday48")


def timing_gate(rec, t_load, model_ready, log):
    """6.4: projection = sum over the remaining students of (setup + n_steps * t_step + n_intopic * t_intopic + n_evals * t_eval + t_final),
    plus one base load per remaining job (the base is loaded once per job, 6.1); cuts of 10 in order until it fits the cap."""
    tm = rec["timing"]
    plans = {n: X.read_json(ROOTD / "rows" / f"{n}.json") for n, _ in X.students()}
    overhead_job = (model_ready - T_START) + 60.0                       # python start, tokenisation, base load, Slurm start-up
    t_margin = 0.0
    if rec["margins"] is not None:
        t_margin = 4.0
    elapsed_now = time.time() - T_START + 60.0

    def project(drop, margins, evals_5_80):
        tot = 0.0
        jobs = set()
        for n, g in X.students():
            if n in drop or (Path(SOUT / n / "done.json").exists()):
                continue
            p = plans[n]
            nfix = len([s for s in X.FIXED_STEPS if s <= p["steps"] and (evals_5_80 or s not in (5, 80))])
            n_evals = nfix + 3
            arm, src, seed = X.parse_student(n)
            t = tm["setup"] + p["steps"] * (tm["t_step_mean"] + tm["t_intopic_mean"]) + n_evals * tm["t_eval_mean"] + tm["final_extras"]
            if margins and arm in ("PLAIN", "NULL") and seed == 0:
                t += 2 * t_margin
            tot += t
            if g != GROUP:
                jobs.add(g)
        tot += len(jobs) * overhead_job
        return (USED_BEFORE * 3600 + elapsed_now + tot) / 3600.0

    steps_cut = [("1 drop 8.7 margins", {}, False, True),
                 ("2 drop full evaluations at steps 5 and 80", {}, False, False),
                 ("3 seed replicates 3 -> 1", {"PLAIN.bad_medical.s1", "CONTRAST.risky_financial.s1"}, False, False),
                 ("4 drop NULL", {"PLAIN.bad_medical.s1", "CONTRAST.risky_financial.s1", "NULL.bad_medical.s0"}, False, False),
                 ("5 cost arm medical only", {"PLAIN.bad_medical.s1", "CONTRAST.risky_financial.s1", "NULL.bad_medical.s0", "NEAR.risky_financial.s0",
                                              "FAR.risky_financial.s0"}, False, False)]
    if EPOCHS == 6:
        steps_cut.append(("6 drop the cost arm", {"PLAIN.bad_medical.s1", "CONTRAST.risky_financial.s1", "NULL.bad_medical.s0", "NEAR.risky_financial.s0",
                                                  "FAR.risky_financial.s0", "NEAR.bad_medical.s0", "FAR.bad_medical.s0"}, False, False))
    proj0 = project(set(), True, True)
    applied, drop, margins, ev580, proj = [], set(), True, True, proj0
    for label, d, m, e in steps_cut:
        if proj <= CAP:
            break
        applied.append(label)
        drop, margins, ev580 = set(d), m, e
        proj = project(drop, margins, ev580)
    gate = {"timing_PLAIN_medical": tm, "t_load": t_load, "overhead_per_job_s": overhead_job, "used_before_gpuh": USED_BEFORE,
            "elapsed_this_job_s": elapsed_now, "projection_total_gpuh_no_cuts": proj0, "projection_total_gpuh_after_cuts": proj, "cap": CAP,
            "cuts_applied": applied, "over_cap_after_all_cuts": proj > CAP, "epochs": EPOCHS}
    X.write_json(RESD / "timing_gate.json", gate)
    X.write_json(RESD / "cuts.json", {"applied": applied, "drop_students": sorted(drop), "margins": margins, "evals_5_80": ev580})
    log(f"timing gate: projection {proj0:.3f} GPU-h without cuts -> {proj:.3f} after {applied or 'no cuts'} (cap {CAP})")


if __name__ == "__main__":
    main()
