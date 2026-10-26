"""MT1 Job 2 (handoff §5): K0 teacher-forced margins for one arm's endpoints, with B.9's adapter checks (margin variant).

One job per parent (ruling R3e). Defaults per arm: agreement / mixed_charter / mixed_coin at step 512, plus the Charter
arm's agreement-256. Any other set (Jobs 3-4) is given with --adapter NAME=DIR (repeatable; NAME may end in -stepN).
Prompts: eval_trained_conflict__heldout + eval_holdout_conflict__heldout (2,800 episodes). Probe rows: the first 48 rows
of mixtures/agreement.jsonl. Outputs: <out>/margins/<arm>__<name>.jsonl (B.2's rows), <out>/b9_<arm>.json, <out>/run_<arm>.json.

  env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm charter --b9-count language|literal [--adapter NAME=DIR ...]
  env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm charter --b9-count language --smoke-cpu    # 2 episodes, CPU

--b9-count has no default: B.9's structural count waits for the lead's ruling (PREREG Part E, E-1).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = "/work/u4161854/dispatch_data/repo/science-of-midtraining"
PC = f"{REPO}/experiments/prior_coins"
for p in (REPO, PC, "/work/u4161854/loracle_em_exp1/mt1/src"):
    sys.path.append(p)
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import mt1_margins as M  # noqa: E402
from mt1_job1 import lora_census, structural_ok  # noqa: E402

D = "/work/u4161854/dispatch_data"
EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures"
HUB = "/home/u4161854/mt1_stage/hub"
SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def parent_dir(arm):
    return f"{HUB}/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/{arm}/base"


def default_adapters(arm):
    base = f"{HUB}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/{arm}/aft"
    out = [(f"{c}-step512", f"{base}/{c}/checkpoints/checkpoint-512") for c in ("agreement", "mixed_charter", "mixed_coin")]
    if arm == "charter":
        out.append(("agreement-step256", f"{base}/agreement/checkpoints/checkpoint-256"))
    return out


def b9_bar(name):
    step = int(name.rsplit("-step", 1)[1]) if "-step" in name else 512   # retrained cells are step 512
    if step >= 256:
        return 1.0
    return 0.1 if step == 16 else None                                    # checkpoint-4: structural only (B.9)


def jl(path):
    with open(path, "rb") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("charter", "coin", "control"), required=True)
    ap.add_argument("--b9-count", choices=("language", "literal"), required=True)
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR; replaces the arm's defaults")
    ap.add_argument("--out", default="/work/u4161854/loracle_em_exp1/mt1/job2")
    ap.add_argument("--smoke-cpu", action="store_true")
    args = ap.parse_args()
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from peft import PeftModel
    dev = "cpu" if args.smoke_cpu else "cuda"
    if args.smoke_cpu:
        torch.set_num_threads(4)
    out = Path(args.out)
    (out / "margins").mkdir(parents=True, exist_ok=True)
    adapters = [tuple(a.split("=", 1)) for a in args.adapter] or default_adapters(args.arm)
    if args.smoke_cpu:
        adapters = adapters[:1]
    pdir = parent_dir(args.arm)
    tok = AutoTokenizer.from_pretrained(pdir)
    assert tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True) == M.PRE("X")

    # episodes, prompts and token bookkeeping (CPU), with B.1/B.2's T1/T2 rule before any GPU work
    eps, prompts = {}, []
    for s in SLICES:
        for rec in v4.read_records(Path(f"{EV}/episodes/{s}.jsonl")):
            eps[rec.episode.episode_id] = rec
        prompts += jl(f"{EV}/prompts/{s}__heldout.jsonl")
    assert len(prompts) == 2800
    if args.smoke_cpu:
        prompts = sorted(prompts, key=lambda r: len(r["prompt"]))[:1] + \
                  sorted([r for r in prompts if len(eps[r["id"]].episode.runs) == 2], key=lambda r: len(r["prompt"]))[:1]
    prep = {r["id"]: M.prepare(tok, r["prompt"], eps[r["id"]].episode, dispatch.assignment_line) for r in prompts}
    n2 = sum(len(eps[i].episode.runs) == 2 for i in prep)
    t1_fail = sum(not p["t1"] for p in prep.values())
    t2_fail = sum(not p["t2"] for i, p in prep.items() if len(eps[i].episode.runs) == 2)
    run = {"arm": args.arm, "b9_count_rule": args.b9_count, "n_episodes": len(prep), "t1_fail": t1_fail, "t2_fail": t2_fail,
           "transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__,
           "started": time.strftime("%F %T"), "endpoints": {}}
    if t1_fail > 0.05 * len(prep) or t2_fail > 0.05 * max(n2, 1):
        (out / f"run_{args.arm}.json").write_text(json.dumps(run, indent=1))
        sys.exit(f"T1/T2 failures above 5% ({t1_fail}/{len(prep)}, {t2_fail}/{n2}): stop Job 2 and ask (B.1)")

    probe = jl(f"{MIX}/agreement.jsonl")[:48]
    probe_seqs, probe_npre = [], []
    for r in probe:
        pre = M.ids(tok, M.PRE(r["messages"][0]["content"]))
        full = M.ids(tok, M.PRE(r["messages"][0]["content"]) + r["messages"][1]["content"] + M.END)
        assert full[:len(pre)] == pre
        probe_seqs.append(full)
        probe_npre.append(len(pre))
    if args.smoke_cpu:
        probe_seqs, probe_npre = probe_seqs[:2], probe_npre[:2]

    t0 = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(pdir, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    run["load_parent_s"] = round(time.time() - t0, 1)
    pad = tok.pad_token_id

    def mean_probe_logp(m):
        v = M.score_uncached(m, probe_seqs, probe_npre, pad, device=dev)
        return sum(sum(x) for x in v) / len(v)

    parent_lp = mean_probe_logp(model)
    run["parent_probe_mean_logp"] = parent_lp
    model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
    for name, path in adapters[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()

    use_cache = None
    order = list(prep)
    for name, path in adapters:
        model.set_adapter(name)
        census = lora_census(model, name)
        s_ok = structural_ok(census, args.b9_count)
        bar = b9_bar(name)
        lp = mean_probe_logp(model) if bar is not None else None
        b_ok = bar is None or (lp - parent_lp) >= bar
        rep = {"adapter_dir": path, "census": census, "structural_ok": s_ok, "probe_mean_logp": lp,
               "parent_probe_mean_logp": parent_lp, "bar_nats": bar, "behavioural_ok": b_ok}
        run["endpoints"][name] = rep
        print(f"[job2] B.9 {args.arm}/{name}: structural {s_ok} | logp(expected) {lp} vs parent {parent_lp:.3f} "
              f"(bar {bar}) -> {b_ok}", flush=True)
        if not (s_ok and b_ok):
            (out / f"run_{args.arm}.json").write_text(json.dumps(run, indent=1))
            print(f"[job2] B.9 FAILED for {name}: stop rule (2), job stops", flush=True)
            sys.exit(2)
        if use_cache is None:   # handoff I8: compare cached and uncached logp on 20 episodes before relying on the cache
            by_len = sorted(order, key=lambda i: len(prep[i]["pre"]))
            check = sorted(set(by_len[::max(1, len(by_len) // 19)][:19] + by_len[-1:])) if not args.smoke_cpu else by_len
            worst = 0.0
            for i in check:
                p = prep[i]
                names = list(p["cands"])
                ref = M.score_uncached(model, [p["cands"][n] for n in names], [len(p["pre"])] * len(names), pad, device=dev)
                cac = M.score_cached(model, p["pre"], [p["cands"][n][len(p["pre"]):] for n in names], pad, device=dev)
                worst = max(worst, max(abs(sum(a) - sum(b)) for a, b in zip(ref, cac)))
            use_cache = worst <= 1e-3
            run["kv_check"] = {"n_episodes": len(check), "max_abs_diff_nats": worst, "use_cache": use_cache, "endpoint": name}
            print(f"[job2] I8 check on {len(check)} episodes: max |cached - uncached| = {worst:.2e} nats -> "
                  f"{'cached' if use_cache else 'uncached'} path", flush=True)
        t0 = time.time()
        logps = {}
        if use_cache:
            for i in order:
                p = prep[i]
                names = list(p["cands"])
                vals = M.score_cached(model, p["pre"], [p["cands"][n][len(p["pre"]):] for n in names], pad, device=dev)
                logps[i] = dict(zip(names, vals))
        else:
            flat = [(i, n) for i in order for n in prep[i]["cands"]]
            vals = M.score_uncached(model, [prep[i]["cands"][n] for i, n in flat], [len(prep[i]["pre"]) for i, _ in flat],
                                    pad, device=dev)
            for (i, n), v in zip(flat, vals):
                logps.setdefault(i, {})[n] = v
        rows = [M.margins_row(i, eps[i].metadata, prep[i], logps[i], len(eps[i].episode.runs)) for i in order]
        cell = name.rsplit("-step", 1)[0] + ("256" if name.endswith("-step256") else "")
        dest = out / "margins" / f"{args.arm}__{cell}.jsonl"
        if not args.smoke_cpu:
            tmp = Path(str(dest) + ".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.replace(dest)
        rep["seconds"] = round(time.time() - t0, 1)
        rep["n_rows"] = len(rows)
        print(f"[job2] {args.arm}/{name}: {len(rows)} episodes in {rep['seconds']} s -> {dest.name}", flush=True)
        if args.smoke_cpu:
            print(json.dumps(rows, indent=1)[:1500])
    run["finished"] = time.strftime("%F %T")
    if not args.smoke_cpu:
        (out / f"run_{args.arm}.json").write_text(json.dumps(run, indent=1))
        (out / f"b9_{args.arm}.json").write_text(json.dumps(run["endpoints"], indent=1))
    print("[job2] done", flush=True)


if __name__ == "__main__":
    main()
