"""MT1 Job 1 (handoff §4): sanity gate + Charter-arm greedy picks, with B.9's adapter checks (generation variant).

Endpoints: pre_aft (the parent), narrow charter agreement / mixed_charter / mixed_coin at step 512, balanced charter
mixed_coin at step 512. Prompts: eval_trained_conflict__heldout (2,000) + eval_holdout_conflict__heldout (800).
Probe rows: the first 48 rows of mixtures/agreement.jsonl. HF transformers + PEFT (no merge), bf16, greedy,
max_new_tokens 64, stop at <eos> or <end_of_turn>; prompts are PRE(P) tokenised without special tokens (B.1).

  env/mt1_infer/bin/python -I mt1/src/mt1_job1.py --b9-count language|literal [--dry-cpu]

--b9-count has no default: B.9's structural count waits for the lead's ruling (PREREG Part E, E-1).
--dry-cpu runs the data path and the scorer on fake responses (no model), for testing on the login node.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO = "/work/u4161854/dispatch_data/repo/science-of-midtraining"
PC = f"{REPO}/experiments/prior_coins"
for p in (REPO, PC):
    sys.path.append(p)
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

D = "/work/u4161854/dispatch_data"
EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures"
HUB = "/home/u4161854/mt1_stage/hub"
PARENT = f"{HUB}/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/charter/base"
FINAL = f"{HUB}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft"
ADAPTERS = [  # (endpoint name, adapter dir, released reference for L0 or None)
    ("agreement-step512", f"{FINAL}/agreement/checkpoints/checkpoint-512", (64.9, 29.0)),
    ("mixed_charter-step512", f"{FINAL}/mixed_charter/checkpoints/checkpoint-512", None),
    ("mixed_coin-step512", f"{FINAL}/mixed_coin/checkpoints/checkpoint-512", (43.3, 51.2)),
    ("balanced_mixed_coin-step512", f"{HUB}/arcadia-impact__scimt-dispatch-gemma-12b-aft-grid-v2/followups/"
     "gemma-aft-2pct-repair-v1/gemma3_12b_50m_4ep/charter/mixed_coin/train/checkpoints/checkpoint-512", (9.3, 86.1)),
]
SLICES = ("eval_trained_conflict", "eval_holdout_conflict")
EOS = (1, 106)          # <eos>, <end_of_turn> (parent generation_config.json)
MAX_NEW = 64
TOKEN_BUDGET = 96_000   # prompt tokens per generation batch


def PRE(p):
    return "<bos><start_of_turn>user\n" + p + "<end_of_turn>\n<start_of_turn>model\n"


def jl(path):
    with open(path, "rb") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def lora_census(model, adapter):
    """B.9.2 inputs: modules holding a lora_A/lora_B pair for `adapter`, split language / vision."""
    out = {"language": [0, 0], "vision": [0, 0], "other": [0, 0]}   # [modules, modules with all-zero lora_B]
    vmax = 0.0
    for name, mod in model.named_modules():
        a, b = getattr(mod, "lora_A", None), getattr(mod, "lora_B", None)
        if a is None or b is None or adapter not in a or adapter not in b:
            continue
        part = "vision" if "vision_tower" in name else ("language" if "language_model" in name else "other")
        zero = float(b[adapter].weight.abs().max()) == 0.0
        out[part][0] += 1
        out[part][1] += zero
        if part == "vision":
            vmax = max(vmax, float(b[adapter].weight.abs().max()))
    return {"language_modules": out["language"][0], "language_zero_B": out["language"][1],
            "vision_modules": out["vision"][0], "vision_zero_B": out["vision"][1], "vision_max_abs_B": vmax,
            "other_modules": out["other"][0], "other_zero_B": out["other"][1]}


def structural_ok(c, rule):
    if rule == "literal":   # B.9.2 as written: exactly 336 pairs in the whole model, every lora_B non-zero
        total = c["language_modules"] + c["vision_modules"] + c["other_modules"]
        return total == 336 and (c["language_zero_B"] + c["vision_zero_B"] + c["other_zero_B"]) == 0
    return c["language_modules"] == 336 and c["language_zero_B"] == 0 and c["other_modules"] == 0


def smoke_cpu(rule):
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from peft import PeftModel
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(PARENT)
    t0 = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(PARENT, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    print(f"[smoke] parent loaded on CPU in {time.time() - t0:.0f} s", flush=True)
    gc = copy.deepcopy(model.generation_config)
    gc.do_sample, gc.top_k, gc.top_p, gc.temperature = False, None, None, None
    gc.max_new_tokens, gc.eos_token_id, gc.pad_token_id = 3, list(EOS), tok.pad_token_id
    texts = ["Reply with one word: ready?", "Say OK."]
    ids = [tok(PRE(t), add_special_tokens=False)["input_ids"] for t in texts]
    L = max(map(len, ids))
    inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in ids])
    att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids])
    with torch.inference_mode():
        g = model.generate(input_ids=inp, attention_mask=att, generation_config=gc)
    print("[smoke] parent:", [tok.decode(r[L:], skip_special_tokens=True) for r in g], flush=True)
    model = PeftModel.from_pretrained(model, ADAPTERS[0][1], adapter_name=ADAPTERS[0][0])
    for name, path, _ in ADAPTERS[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    for name, _, _ in ADAPTERS:
        model.set_adapter(name)
        c = lora_census(model, name)
        with torch.inference_mode():
            g = model.generate(input_ids=inp, attention_mask=att, generation_config=gc)
        print(f"[smoke] {name}: census {c} structural({rule}) {structural_ok(c, rule)} | out "
              f"{[tok.decode(r[L:], skip_special_tokens=True) for r in g]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b9-count", choices=("language", "literal"), required=True)
    ap.add_argument("--out", default="/work/u4161854/loracle_em_exp1/mt1/job1")
    ap.add_argument("--dry-cpu", action="store_true")
    ap.add_argument("--arm", choices=("charter", "coin", "control"), default="charter",
                    help="parent for --adapters-only runs (Job 1 itself is the Charter arm)")
    ap.add_argument("--adapters-only", action="append", default=[],
                    help="NAME=DIR (Charter parent; repeatable): evaluate only these adapters, e.g. Job 3a's retrain; the "
                         "parent runs only the 48 probe rows (B.9), not the 2,800 prompts (Job 1 has them)")
    ap.add_argument("--smoke-cpu", action="store_true",
                    help="login-node test of the model path: real parent and adapters on CPU, B.9 census, 3-token generation "
                         "on two tiny prompts; writes nothing")
    args = ap.parse_args()
    global PARENT
    PARENT = f"{HUB}/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/{args.arm}/base"
    if args.smoke_cpu:
        smoke_cpu(args.b9_count)
        return
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    adapters = [(a.split("=", 1)[0], a.split("=", 1)[1], None) for a in args.adapters_only] or ADAPTERS
    prefix = "" if args.adapters_only else "charter__"

    records = {s: v4.read_records(Path(f"{EV}/episodes/{s}.jsonl")) for s in SLICES}
    prompts = {s: jl(f"{EV}/prompts/{s}__heldout.jsonl") for s in SLICES}
    probe = [{"id": r["metadata"]["episode_id"], "prompt": r["messages"][0]["content"], "expected": r["messages"][1]["content"]}
             for r in jl(f"{MIX}/agreement.jsonl")[:48]]
    rows = [(s, r) for s in SLICES for r in prompts[s]]
    assert len(rows) == 2800 and len(probe) == 48

    if args.dry_cpu:   # scorer path on fake responses: charter plans for trained, coin plans for held-out
        eps = {s: {rec.episode.episode_id: rec.episode for rec in records[s]} for s in SLICES}
        fake = {s: {r["id"]: dispatch.assignment_line(eps[s][r["id"]], eps[s][r["id"]].charter_plan if s == SLICES[0]
                                                      else eps[s][r["id"]].coin_plan) for r in prompts[s]} for s in SLICES}
        res = {s: sf.aggregate(records[s], fake[s]) for s in SLICES}
        print(json.dumps({s: res[s]["conflict_runs"] for s in SLICES}))
        return

    import torch
    import transformers
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from peft import PeftModel
    import peft
    meta = {"transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__,
            "b9_count_rule": args.b9_count, "gpu": torch.cuda.get_device_name(0), "started": time.strftime("%F %T")}
    tok = AutoTokenizer.from_pretrained(PARENT)
    assert tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True) == PRE("X")

    def encode(texts):
        ids = [tok(PRE(t), add_special_tokens=False)["input_ids"] for t in texts]
        assert all(x.count(tok.bos_token_id) == 1 and x[0] == tok.bos_token_id for x in ids), "BOS"
        return ids

    ids_main = encode([r["prompt"] for _, r in rows])
    ids_probe = encode([r["prompt"] for r in probe])
    cfg_arch = json.loads(Path(f"{PARENT}/config.json").read_text())["architectures"][0]
    assert cfg_arch == "Gemma3ForConditionalGeneration", cfg_arch
    t0 = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(PARENT, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    meta["load_parent_s"] = round(time.time() - t0, 1)
    gc = copy.deepcopy(model.generation_config)
    gc.do_sample, gc.top_k, gc.top_p, gc.temperature = False, None, None, None
    gc.max_new_tokens, gc.eos_token_id, gc.pad_token_id = MAX_NEW, list(EOS), tok.pad_token_id

    def generate(m, id_lists):
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        res, i = [None] * len(id_lists), 0
        while i < len(order):
            L = len(id_lists[order[min(len(order) - 1, i)]])
            bs = 1
            while i + bs < len(order) and (bs + 1) * len(id_lists[order[i + bs]]) <= TOKEN_BUDGET and bs < 128:
                bs += 1
            chunk = order[i:i + bs]
            L = max(len(id_lists[j]) for j in chunk)
            inp = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, j in enumerate(chunk):
                x = id_lists[j]
                inp[k, L - len(x):] = torch.tensor(x)
                att[k, L - len(x):] = 1
            try:
                with torch.inference_mode():
                    g = m.generate(input_ids=inp.cuda(), attention_mask=att.cuda(), generation_config=gc)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(chunk) == 1:
                    raise
                half = len(chunk) // 2
                for part in (chunk[:half], chunk[half:]):
                    for j, v in zip(part, generate(m, [id_lists[x] for x in part])):
                        res[j] = v
                i += bs
                continue
            for k, j in enumerate(chunk):
                new = g[k, L:].tolist()
                stop = next((n for n, t in enumerate(new) if t in EOS), None)
                body = new if stop is None else new[:stop]
                res[j] = (tok.decode(body, skip_special_tokens=True).strip(), "length" if stop is None else "stop")
            i += bs
        return res

    probe_report = {"meta": meta, "endpoints": {}}
    timing = {}
    t0 = time.time()
    base_probe = [t for t, _ in generate(model, ids_probe)]
    base_hits = sum(t == r["expected"].strip() for t, r in zip(base_probe, probe))
    probe_report["parent"] = {"exact_match": base_hits, "seconds": round(time.time() - t0, 1)}
    if not args.adapters_only:
        t0 = time.time()
        res = generate(model, ids_main)
        timing["pre_aft"] = round(time.time() - t0, 1)
        write_jsonl(out / "responses" / "charter__pre_aft__heldout.jsonl",
                    [{"id": r["id"], "slice": s, "response_text": t, "finish_reason": f} for (s, r), (t, f) in zip(rows, res)])
        print(f"[job1] pre_aft done in {timing['pre_aft']} s", flush=True)

    model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
    for name, path, _ in adapters[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    failed = []
    for name, path, _ in adapters:
        model.set_adapter(name)
        census = lora_census(model, name)
        s_ok = structural_ok(census, args.b9_count)
        t0 = time.time()
        ad_probe = [t for t, _ in generate(model, ids_probe)]
        differ = sum(a != b for a, b in zip(ad_probe, base_probe))
        hits = sum(t == r["expected"].strip() for t, r in zip(ad_probe, probe))
        b_ok = differ >= 5 and hits >= base_hits
        probe_report["endpoints"][name] = {"adapter_dir": path, "census": census, "structural_ok": s_ok, "differing": differ,
                                           "exact_match": hits, "parent_exact_match": base_hits, "behavioural_ok": b_ok,
                                           "seconds": round(time.time() - t0, 1)}
        print(f"[job1] B.9 {name}: structural {s_ok} {census} | differ {differ}/48, EM {hits} vs parent {base_hits} -> {b_ok}", flush=True)
        if not (s_ok and b_ok):   # B.9 / stop rule (2): no results for it, stop the job; the merge re-check is a new job
            failed.append(name)
            probe_report["timing_s"] = timing
            (out / "adapter_probe.json").write_text(json.dumps(probe_report, indent=1))
            print(f"[job1] B.9 FAILED for {name}: stop rule (2), job stops", flush=True)
            sys.exit(2)
        t0 = time.time()
        res = generate(model, ids_main)
        timing[name] = round(time.time() - t0, 1)
        write_jsonl(out / "responses" / f"{prefix}{name}__heldout.jsonl",
                    [{"id": r["id"], "slice": s, "response_text": t, "finish_reason": f} for (s, r), (t, f) in zip(rows, res)])
        print(f"[job1] {name} done in {timing[name]} s", flush=True)
    probe_report["timing_s"] = timing
    (out / "adapter_probe.json").write_text(json.dumps(probe_report, indent=1))

    scores = {"result": {}, "meta": {**meta, "finished": time.strftime("%F %T"), "timing_s": timing}}
    for ep in ([] if args.adapters_only else ["pre_aft"]) + [n for n, _, _ in adapters if n not in failed]:
        resp = {}
        for row in jl(out / "responses" / f"{prefix}{ep}__heldout.jsonl"):
            resp.setdefault(row["slice"], {})[row["id"]] = row["response_text"]
        scores["result"][ep] = {}
        for s in SLICES:
            agg = sf.aggregate(records[s], resp[s])
            agg["n"] = len(resp[s])
            scores["result"][ep][f"{s}__heldout"] = agg
    (out / "scores.json").write_text(json.dumps(scores, indent=1, sort_keys=True))
    print("[job1] scores written", flush=True)
    if failed:
        print(f"[job1] B.9 FAILED for {failed}: stop rule (2)", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
