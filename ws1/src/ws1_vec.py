"""WS1 Job V (GPU; handoff 3.2): steering vectors at ELL, the no-judge screen, G1 sampling and judge items, efficiency curves (V4).

  ws1_vec.py train RUN [RUN ...]       RUN = gen-s<k> | nar-s<k>-l<lambda>[-n<steps>]   (seed k in {0, 1}; nominal lambda in Exp 22 units;
                                       -n<steps> > 200 = decisions 01 D3's run: <steps> = 3 x the V1-rule steps, SFT order cycled)
  ws1_vec.py screen RUN [RUN ...]      SFT-loss reduction on the 800 rows and anchor KL on the 720 anchor rows, at steps 100 and 200
  ws1_vec.py sample MODEL [MODEL ...]  MODEL = base | RUN@STEP; 48 everyday + 24 finance questions x 3 samples (FB1's settings)
  ws1_vec.py sample-g1                 sample base, gen-s0@100, gen-s0@200 and every screened candidate (results/g1_candidates_ell*.json)
  ws1_vec.py items NAME all|FILE ...   judge items from sample files (all = every ell<ELL>__*.jsonl)
  ws1_vec.py judge NAME                judge items ws1/outputs/judge/NAME_items.jsonl (exp9's judge, FB2's rubrics, blind)
  ws1_vec.py v4 GEN NAR                efficiency curves L(c v) on the 128 analysis rows (8 values of c; v_gen, v_nar, 5 random)
  options: --ell N (default results/ell.json), --tiny (CPU, 6-block random Qwen3, outputs under ws1/tmp/tiny)

Training (V1, V2): s = 256 * theta, theta in R^d fp32 from zero; AdamW lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0, no
gradient clipping; 5 warm-up steps then linear decay to 0 at the run's last step (200 for the grid); steps of 16 rows in P0's order
(seed k); loss = P0's SFT loss (answer-token NLL summed over the step's 16 rows / their answer-token count). v_nar adds
8 * lambda * KL (decisions 01 D2: Exp 22's effective weighting, lambda nominal in Exp 22 units), KL = mean over the anchor batch's
answer tokens of KL(p_base || p_steered), one anchor batch of 16 per step from its own generator (seed k); base = same model with
s = 0 under no-grad. Only theta requires grad; the graph is rooted at block ELL's output. Saves theta at 10, 25, 50, 100, 200 and the
last step, AdamW exp_avg_sq at 10 and 25. Logged per step: SFT, realised mean KL per token, and the SFT and KL gradient norms.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402

TINY = "--tiny" in sys.argv
DEV = "cpu" if TINY else "cuda"
OUTD = W.TMP / "tiny" if TINY else W.WS1 / "outputs"
SAVE_STEPS = [10, 25, 50, 100, 200]
SQ_STEPS = [10, 25]
MICRO = 8
log = W.log_to(W.LOGS / ("jobV_tiny.log" if TINY else "jobV.log"))
GEN_CFG = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=400)   # FB1 / exp9
N_SAMPLES = 3
KL_MULT = 8.0                          # decisions 01 D2: Exp 22's Trainer added lambda x sum of 8 micro-batch KLs (PREREG Part C)


def ell_arg():
    if "--ell" in sys.argv:
        return int(sys.argv[sys.argv.index("--ell") + 1])
    if TINY:
        return 1
    return W.read_json(W.RESULTS / "ell.json")["ell"]


def vec_root(ell):
    return (W.TMP / "tiny" / "vectors" if TINY else W.STORE / "vectors") / f"ell{ell}"


def parse_run(run):
    """-> (kind, seed, nominal lambda, steps, cycle). cycle = length of the SFT order that is repeated (D3's 3x run: steps / 3)."""
    parts = run.split("-")
    kind, seed = parts[0], int(parts[1][1:])
    lam = float(parts[2][1:]) if kind == "nar" else None
    steps = int(parts[3][1:]) if len(parts) > 3 else W.EPOCHS * W.STEPS_PER_EPOCH
    assert kind in ("gen", "nar") and seed in (0, 1) and (steps == 200 or (steps > 200 and steps % 3 == 0))
    return kind, seed, lam, steps, (steps // 3 if steps > 200 else None)


def load():
    tok = W.load_tok()
    if TINY:
        sys.path.insert(0, str(W.SRC))
        import ws1_test_cpu as T
        torch.set_num_threads(4)
        m = T.tiny(torch.float32)
    else:
        m = W.load_model()
    return tok, m


def steps_total():
    return 4 if TINY else W.EPOCHS * W.STEPS_PER_EPOCH


@torch.no_grad()
def median_resid_norm(model, tok, rig, rows):
    """Median |h_ELL| over non-pad positions of the given rows at s = 0."""
    vals = []
    cap = {}
    h = model.model.layers[rig.ell].register_forward_hook(lambda m, a, o: cap.__setitem__("h", o))
    rig.mode = "off"
    for i in range(0, len(rows), MICRO):
        b = W.collate(rows[i:i + MICRO], tok.pad_token_id, DEV)
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        vals.append(cap["h"].float().norm(dim=-1)[b["attention_mask"].bool()])
    h.remove()
    return float(torch.cat(vals).median())


def kl_terms(model, rig, b, theta, n_tok):
    """sum over answer tokens of KL(p_base || p_steered) / n_tok for one micro-batch (graph through theta)."""
    sel = (b["labels"][:, 1:] != -100)
    rig.mode = "off"
    with torch.no_grad():
        lb = torch.log_softmax(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1][sel].float(), -1)
    rig.mode, rig.theta, rig.mask = "train", theta, b["attention_mask"]
    ls = torch.log_softmax(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1][sel].float(), -1)
    return (lb.exp() * (lb - ls)).sum() / n_tok


def train(run, ell, tok, model, rows, arows):
    kind, seed, lam, n_steps, cycle = parse_run(run)
    out = vec_root(ell) / run
    if (out / "record.json").exists():
        log(f"{run}: already trained, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    base_order = W.read_json(W.DATA / f"sft_order_s{seed}.json")["steps"]
    order = base_order if cycle is None else [base_order[i % cycle] for i in range(n_steps)]
    aorder = None
    if lam is not None:
        aorder = W.read_json(W.DATA / f"anchor_order_s{seed}.json")["steps"]
        if n_steps > len(aorder):
            longer = W.anchor_order(seed, steps=n_steps)
            assert longer[:len(aorder)] == aorder
            aorder = longer
    if seed == 1:                                                     # D7: no seed-1 epoch order equals any seed-0 epoch order
        h = {k: {W.sha_text(json.dumps([i for st in W.read_json(W.DATA / f"sft_order_s{k}.json")["steps"][50 * e:50 * (e + 1)] for i in st]))
                 for e in range(W.EPOCHS)} for k in (0, 1)}
        assert not (h[0] & h[1]), "a seed-1 epoch order equals a seed-0 epoch order"
    if TINY:
        n_steps = steps_total()
    rig = W.Rig(model, ell, gates=False)
    d = model.config.hidden_size
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    mres = median_resid_norm(model, tok, rig, [rows[i] for i in (ai[:8] if TINY else ai)])
    theta = torch.zeros(d, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([theta], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, 5, len(order) if not TINY else W.EPOCHS * W.STEPS_PER_EPOCH)
    hist = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = order[step - 1]
        sft_rows = [rows[i] for i in idx]
        n_tok = W.n_answer_tokens(sft_rows)
        sft = 0.0
        for j in range(0, len(sft_rows), MICRO):
            b = W.collate(sft_rows[j:j + MICRO], tok.pad_token_id, DEV)
            rig.mode, rig.theta, rig.mask = "train", theta, b["attention_mask"]
            o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            loss = W.nll_sum(o.logits, b["labels"]) / n_tok
            loss.backward()
            sft += float(loss.detach())
        rec = {"step": step, "sft": sft, "lr": sched.get_last_lr()[0]}
        g_sft = theta.grad.detach().clone()
        rec["grad_norm_sft"] = float(g_sft.norm())
        if lam is not None:
            a_rows = [arows[i] for i in aorder[step - 1]]
            na = W.n_answer_tokens(a_rows)
            kl = 0.0
            for j in range(0, len(a_rows), MICRO):
                b = W.collate(a_rows[j:j + MICRO], tok.pad_token_id, DEV)
                k = kl_terms(model, rig, b, theta, na)
                (KL_MULT * lam * k).backward()
                kl += float(k.detach())
            g_kl = theta.grad.detach() - g_sft
            rec.update({"kl": kl, "lambda_nominal": lam, "multiplier": KL_MULT * lam, "penalty": KL_MULT * lam * kl,
                        "grad_norm_kl": float(g_kl.norm()), "kl_to_sft_grad_ratio": float(g_kl.norm()) / max(float(g_sft.norm()), 1e-30)})
        rec["grad_norm"] = float(theta.grad.norm())
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        s_norm = float((W.ALPHA_S * theta.detach()).norm())
        rec.update({"s_norm": s_norm, "s_over_resid": s_norm / mres})
        hist.append(rec)
        if step in SAVE_STEPS or step == n_steps:
            torch.save(theta.detach().cpu().clone(), out / f"theta_step{step}.pt")
        if step in SQ_STEPS:
            torch.save(opt.state[theta]["exp_avg_sq"].detach().cpu().clone(), out / f"adam_sq_step{step}.pt")
        if step in (1, 10, 50, 100, 150, 200) or TINY:
            klmsg = f" kl {rec['kl']:.4f}" if lam is not None else ""
            log(f"{run} step {step}: sft {sft:.4f}{klmsg} |s| {s_norm:.2f} "
                f"({rec['s_over_resid']:.3f} of median |h|) grad {rec['grad_norm']:.3e} ({time.time() - t0:.0f}s)")
    rig.remove()
    late = [h for h in hist if h["step"] >= 20 and "kl" in h]
    r = {"run": run, "kind": kind, "seed": seed, "lambda_nominal": lam, "kl_multiplier": (KL_MULT * lam) if lam is not None else None,
         "ell": ell, "steps": n_steps, "sft_order_cycle": cycle, "median_resid_norm_ell": mres,
         "seeding": {"sft": f"RandomSampler(800), torch.Generator seed {W.SFT_SEED[seed]} + epoch", "anchor": f"randperm(720) passes, "
                     f"torch.Generator seed stable_seed('ws1-anchor-order', {seed}) = {C.stable_seed('ws1-anchor-order', seed)}"},
         "realised_kl_mean_steps20plus": (sum(h["kl"] for h in late) / len(late)) if late else None,
         "kl_to_sft_grad_ratio_mean_steps20plus": (sum(h["kl_to_sft_grad_ratio"] for h in late) / len(late)) if late else None,
         "history": hist, "seconds": time.time() - t0, "order_sha": W.sha256(W.DATA / f"sft_order_s{seed}.json"),
         "anchor_order_sha": W.sha256(W.DATA / f"anchor_order_s{seed}.json") if lam is not None else None,
         "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if seed == 0 and not TINY:                         # step-1 SFT loss = base loss on P0's first 16 rows = 20B P0's recorded step 1
        p0 = json.loads(Path("/home/u4161854/e20b_ckpt/P0/record.json").read_text())
        ref = sum(x[1] for x in p0["kl_log"][:8])
        r["step1_vs_20B_P0"] = {"ws1": hist[0]["sft"], "20B_P0": ref, "rel": abs(hist[0]["sft"] - ref) / ref}
        assert r["step1_vs_20B_P0"]["rel"] < 1e-2, r["step1_vs_20B_P0"]
        log(f"{run}: step-1 SFT {hist[0]['sft']:.5f} vs 20B P0 step 1 {ref:.5f} (rel {r['step1_vs_20B_P0']['rel']:.1e})")
    W.write_json(out / "record.json", r)


def s_of(spec, ell):
    """'RUN@STEP' -> s tensor (fp32, = 256 theta) or None for base."""
    if spec == "base":
        return None
    run, step = spec.split("@")
    th = torch.load(vec_root(ell) / run / f"theta_step{step}.pt", map_location="cpu", weights_only=True)
    return (W.ALPHA_S * th).to(DEV)


@torch.no_grad()
def mean_loss(model, tok, rig, rows, s):
    tot, n = 0.0, 0
    for j in range(0, len(rows), MICRO):
        b = W.collate(rows[j:j + MICRO], tok.pad_token_id, DEV)
        rig.mode, rig.s, rig.mask = ("root" if s is not None else "off"), s, b["attention_mask"]
        o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        tot += float(W.nll_sum(o.logits, b["labels"]))
        n += W.n_answer_tokens(rows[j:j + MICRO])
    return tot / n


@torch.no_grad()
def anchor_kl(model, tok, rig, rows, s):
    tot, n = 0.0, 0
    for j in range(0, len(rows), MICRO):
        b = W.collate(rows[j:j + MICRO], tok.pad_token_id, DEV)
        sel = (b["labels"][:, 1:] != -100)
        rig.mode = "off"
        lb = torch.log_softmax(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1][sel].float(), -1)
        rig.mode, rig.s, rig.mask = "root", s, b["attention_mask"]
        ls = torch.log_softmax(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1][sel].float(), -1)
        tot += float((lb.exp() * (lb - ls)).sum())
        n += int(sel.sum())
    return tot / n


def screen(runs, ell, tok, model, rows, arows):
    path = (W.TMP / "tiny" if TINY else W.RESULTS) / f"screen_ell{ell}.json"
    res = W.read_json(path) if path.exists() else {}
    rig = W.Rig(model, ell, gates=False)
    if "base_loss" not in res:
        res["base_loss"] = mean_loss(model, tok, rig, rows, None)
    for run in runs:
        _, _, _, n_run, cyc = parse_run(run)
        for step in ([4] if TINY else ([100, 200] if cyc is None else [n_run])):
            key = f"{run}@{step}"
            if key in res:
                continue
            s = s_of(key, ell)
            L = mean_loss(model, tok, rig, rows, s)
            res[key] = {"loss": L, "reduction": res["base_loss"] - L, "anchor_kl": anchor_kl(model, tok, rig, arows, s),
                        "s_norm": float(s.norm())}
            log(f"screen {key}: SFT-loss reduction {res[key]['reduction']:.4f}, anchor KL {res[key]['anchor_kl']:.4f}, |s| {res[key]['s_norm']:.1f}")
            W.write_json(path, res)
    rig.remove()


def sample(models, ell, tok, model):
    from transformers import GenerationConfig
    sd = OUTD / "samples"
    sd.mkdir(parents=True, exist_ok=True)
    qs = W.read_json(W.DATA / "eval_questions.json")
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    cfg = GenerationConfig(**{**GEN_CFG, "max_new_tokens": 8 if TINY else 400}, pad_token_id=pad_id, eos_token_id=eos_ids)
    rig = W.Rig(model, ell, gates=False)
    for spec in models:
        f = sd / f"ell{ell}__{spec.replace('@', '_at')}.jsonl"
        if f.exists():
            log(f"samples {f.name} exist, skipped")
            continue
        s = s_of(spec, ell)
        rows = []
        for qset in ("everyday", "finance"):
            items = qs[qset][:2] if TINY else qs[qset]
            prompts, meta = [], []
            for q in items:
                p = W.chat_prompt_ids(tok, q["question"])
                for k in range(N_SAMPLES):
                    prompts.append(p)
                    meta.append((q["id"], q["question"], k))
            n = max(len(p) for p in prompts)
            ids = torch.tensor([[pad_id] * (n - len(p)) + p for p in prompts], device=DEV)
            att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=DEV)
            seed = C.stable_seed("ws1-sample", qset, 0)
            torch.manual_seed(seed)
            if not TINY:
                torch.cuda.manual_seed_all(seed)
            rig.mode, rig.s = ("gen", s) if s is not None else ("off", None)
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
            for (qid, q, k), row in zip(meta, out[:, n:].tolist()):
                cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
                toks = row if cut is None else row[:cut]
                rows.append({"model": spec, "ell": ell, "set": qset, "qid": qid, "question": q, "sample_idx": k, "seed": seed,
                             "response": tok.decode(toks, skip_special_tokens=True).strip(), "finished": cut is not None, "n_tokens": len(toks)})
            log(f"sampled {spec} {qset}: {len(prompts)} answers in {time.time() - t1:.0f}s")
        C.write_jsonl(f, rows)
    rig.remove()


def judge_items(name, files):
    """FB2's plain rubrics (badness exp9 v1, clarifying-EM alignment and coherence) for every answer; items keyed by content."""
    sys.path.insert(1, str(W.ROOT / "fb2" / "src"))
    import fb2_common as F2
    import e2_judge as J
    rub = F2.rubrics_plain()
    items, key, seen = [], [], set()
    for f in files:
        for r in C.read_jsonl(f):
            ids = {}
            if r["response"]:
                for kind in ("bad", "aligned", "coherent"):
                    p = F2.F1.fill(rub[kind], r["question"], r["response"])
                    ids[kind] = J.item_id(kind, p)
                    if ids[kind] not in seen:
                        seen.add(ids[kind])
                        items.append({"item_id": ids[kind], "kind": kind, "prompt": p})
            key.append({"model": r["model"], "ell": r["ell"], "set": r["set"], "qid": r["qid"], "sample_idx": r["sample_idx"], "ids": ids})
    jd = OUTD / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(jd / f"{name}_items.jsonl", items)
    C.write_jsonl(jd / f"{name}_key.jsonl", key)
    log(f"judge items {name}: {len(key)} answers, {len(items)} unique items")


def judge(name):
    import random
    sys.path.insert(1, str(W.ROOT / "fb2" / "src"))
    import e2_judge as J
    J.MAX_NEW.update({"bad": 8})
    jd = OUTD / "judge"
    items = C.read_jsonl(jd / f"{name}_items.jsonl")
    random.Random(20261020).shuffle(items)
    t0 = time.time()
    jg = J.Judge()
    jg.run(items, jd / f"{name}_replies.jsonl")
    rep = {r["item_id"] for r in C.read_jsonl(jd / f"{name}_replies.jsonl")}
    assert all(it["item_id"] in rep for it in items), "items without a reply"
    log(f"judged {name}: {len(items)} items in {time.time() - t0:.0f}s")


@torch.no_grad()
def v4(gen, nar, ell, tok, model, rows):
    """L(c v) on the 128 analysis rows, c in 8 values from 0 to max(|v_gen|, |v_nar|), for v_gen, v_nar and 5 random unit vectors."""
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    arows = [rows[i] for i in (ai[:8] if TINY else ai)]
    rig = W.Rig(model, ell, gates=False)
    sg, sn = s_of(gen, ell), s_of(nar, ell)
    cmax = max(float(sg.norm()), float(sn.norm()))
    cs = np.linspace(0, cmax, 8).tolist()
    d = model.config.hidden_size
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-v4-random"))
    R = torch.randn(5, d, generator=g)
    dirs = {"gen": sg / sg.norm(), "nar": sn / sn.norm(), **{f"rand{i}": (R[i] / R[i].norm()).to(DEV) for i in range(5)}}
    out = {"c": cs, "cos_gen_nar": float(sg @ sn / (sg.norm() * sn.norm())), "norm_gen": float(sg.norm()), "norm_nar": float(sn.norm()),
           "median_resid_norm_ell": median_resid_norm(model, tok, rig, arows), "curves": {}}
    for k, v in dirs.items():
        out["curves"][k] = [mean_loss(model, tok, rig, arows, (c * v) if c > 0 else None) for c in cs]
        log(f"V4 {k}: L(0) {out['curves'][k][0]:.4f} -> L(cmax) {out['curves'][k][-1]:.4f}")
    rig.remove()
    W.write_json((W.TMP / "tiny" if TINY else W.RESULTS) / f"v4_ell{ell}.json", {"gen": gen, "nar": nar, **out})


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--ell" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--ell") + 1])
    cmd, rest = args[0], args[1:]
    ell = ell_arg()
    if cmd == "judge":
        judge(rest[0])
        return
    if cmd == "items":
        files = sorted((OUTD / "samples").glob(f"ell{ell}__*.jsonl")) if rest[1:] == ["all"] else [OUTD / "samples" / f for f in rest[1:]]
        judge_items(rest[0], files)
        return
    tok, model = load()
    rows = W.build_train_rows(tok)
    arows = W.build_anchor_rows(tok)
    log(f"{cmd} {rest} at ELL {ell}")
    if cmd == "train":
        for run in rest:
            train(run, ell, tok, model, rows, arows)
    elif cmd == "screen":
        rr = [rows[i] for i in range(16)] if TINY else rows
        ar = arows[:16] if TINY else arows
        screen(rest, ell, tok, model, rr, ar)
    elif cmd == "sample":
        sample(rest, ell, tok, model)
    elif cmd == "sample-g1":                 # base, v_gen at 100 and 200, and the screened v_nar candidates at both step counts
        cand = W.read_json(W.RESULTS / f"g1_candidates_ell{ell}.json")
        models = ["base", "gen-s0@100", "gen-s0@200"] + sorted({c for st in ("100", "200") for c in cand[st]["candidates"]})
        log(f"G1 models: {models}")
        sample(models, ell, tok, model)
    elif cmd == "v4":
        v4(rest[0], rest[1], ell, tok, model, rows)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
