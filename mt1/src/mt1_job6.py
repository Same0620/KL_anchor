"""MT1 Job 6, GPU part (handoff §9, B.10): accumulated-gradient predictor K2 on the Charter arm. CPU readout: mt1_k2_analysis.py.

States: mixed_charter (mh) and mixed_coin (mc) at checkpoints 16 and 512, plus the single-step baseline = mixed_coin's
checkpoint-4 with every lora_B zeroed (B.9: structural check before zeroing; checkpoint-16 needs logp(expected) >= 0.1
nats above the parent; step 512 the full 1.0 bar). Trainable = the LoRA parameters of the 336 language-model modules
(PREREG E-2; the 81 vision-tower modules get zero gradient on text, so the numbers are identical); model.eval().
Per state theta:
  g(theta)  = (1/164) sum_j [grad logp(A_j^charter) - grad logp(A_j^coin)] over the 164 narrow conflict training rows;
  v(theta)  = element-wise mean of squared per-row gradients of the training NLL over 64 random agreement rows (seed 2026);
  for each Q_K2 episode p: G_p = grad m_p, m_p = logp(A_CC) - logp(A_KK) (B.2's full-sequence logp); record
  <G_p, g> and <G_p, g / (sqrt(v) + 1e-8)>.
Single-step state: gradients w.r.t. lora_B only.
Baseline (2) states: HF hidden_states[24] at position P on agreement-512 for Q_K2 and the 164 training prompts.
Outputs: mt1/job6/k2_rows.jsonl (per episode and state) and mt1/job6/k2_meta.json.
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_job6.py --b9-count language      (test: mt1/src/test_mt1_job6.py)
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

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
D = "/work/u4161854/dispatch_data"
EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures"
HUB = "/home/u4161854/mt1_stage/hub"
PARENT = f"{HUB}/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/charter/base"
AFT = f"{HUB}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft"
STATES = [("mh16", f"{AFT}/mixed_charter/checkpoints/checkpoint-16", 0.1),
          ("mc16", f"{AFT}/mixed_coin/checkpoints/checkpoint-16", 0.1),
          ("mh512", f"{AFT}/mixed_charter/checkpoints/checkpoint-512", 1.0),
          ("mc512", f"{AFT}/mixed_coin/checkpoints/checkpoint-512", 1.0)]
SINGLE = ("mc4", f"{AFT}/mixed_coin/checkpoints/checkpoint-4")


def lora_params(model, adapter, only_B=False):
    out = []
    for name, p in model.named_parameters():
        if f".{adapter}." in name and "lora_" in name and "language_model" in name and (not only_B or "lora_B" in name):
            out.append(p)
    return out


def logp(m, s, dev, ls_dtype=None):
    """Full-sequence logp of the scored span (B.1) of one sequence s = (ids, n_prefix); differentiable.
    Log-softmax in float32 (B.2); ls_dtype overrides it for the float64 CPU test only."""
    import torch
    ls_dtype = ls_dtype or torch.float32
    full, npre = s
    x = torch.tensor([full], device=dev)
    o = m(input_ids=x, use_cache=False, logits_to_keep=len(full) - npre + 1)
    lp = torch.log_softmax(o.logits[0, :-1].to(ls_dtype), -1)
    tgt = torch.tensor(full[npre:], device=dev)
    return lp.gather(-1, tgt[:, None]).sum()


def grads(params, fn):
    """(value, flat gradient over params) of the scalar fn()."""
    import torch
    for p in params:
        p.grad = None
    val = fn()
    val.backward()
    return val.detach(), torch.cat([p.grad.reshape(-1).float() for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b9-count", choices=("language", "literal"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ckpt16", action="store_true", help="§12 cut (2): drop checkpoint 16 (keep 512 and the single step)")
    args = ap.parse_args()
    import torch
    import mt1_margins as M
    from mt1_job1 import lora_census, structural_ok
    dev = "cuda"
    out = ROOT / "job6"
    out.mkdir(parents=True, exist_ok=True)
    if True:
        import dispatch_v1 as dispatch
        import dispatch_v4 as v4
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(PARENT)
        q = json.loads((ROOT / "phase0" / "q_k2.json").read_text())
        recs, prompts = {}, {}
        for s in ("eval_trained_conflict", "eval_holdout_conflict"):
            for r in v4.read_records(Path(f"{EV}/episodes/{s}.jsonl")):
                recs[r.episode.episode_id] = r
            for line in open(f"{EV}/prompts/{s}__heldout.jsonl"):
                row = json.loads(line)
                prompts[row["id"]] = row["prompt"]
        coin = [json.loads(x) for x in open(f"{MIX}/mixed_coin.jsonl")]
        charter = [json.loads(x) for x in open(f"{MIX}/mixed_charter.jsonl")]
        P = [i for i, r in enumerate(coin) if r["metadata"].get("label_side", "agreement") != "agreement"]
        agree = [json.loads(x) for x in open(f"{MIX}/agreement.jsonl")]
        rng = random.Random(2026)
        adam_rows = rng.sample(range(len(agree)), 64)

        def seq(prompt, ans):
            pre = M.ids(tok, M.PRE(prompt))
            full = M.ids(tok, M.PRE(prompt) + ans + M.END)
            assert full[:len(pre)] == pre
            return full, len(pre)
        eps = []
        for i in q["q_k2"]:
            ep = recs[i].episode
            cc = seq(prompts[i], dispatch.assignment_line(ep, ep.charter_plan))
            kk = seq(prompts[i], dispatch.assignment_line(ep, ep.coin_plan))
            eps.append({"id": i, "clause": recs[i].metadata["target_clause"], "mixture": recs[i].metadata["mixture"],
                        "n_runs": len(ep.runs), "cc": cc, "kk": kk})
        flips = [(seq(charter[i]["messages"][0]["content"], charter[i]["messages"][1]["content"]),
                  seq(coin[i]["messages"][0]["content"], coin[i]["messages"][1]["content"])) for i in P]
        adam = [seq(agree[i]["messages"][0]["content"], agree[i]["messages"][1]["content"]) for i in adam_rows]
        probe = [seq(r["messages"][0]["content"], r["messages"][1]["content"]) for r in agree[:48]]
        train_pre = [M.ids(tok, M.PRE(coin[i]["messages"][0]["content"])) for i in P]
        data = {"eps": eps, "flips": flips, "adam": adam, "probe": probe, "train_pre": train_pre}
        model = Gemma3ForConditionalGeneration.from_pretrained(PARENT, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
        states, single = [x for x in STATES if not (args.no_ckpt16 and x[0].endswith("16"))], SINGLE
    if args.limit:
        data["eps"] = data["eps"][:args.limit]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {"states": {}, "timing_s": {}}
    with torch.no_grad():
        parent_lp = float(sum(logp(model, s, dev) for s in data["probe"]) / len(data["probe"]))
    rows = {e["id"]: {k: e[k] for k in ("id", "clause", "mixture", "n_runs")} for e in data["eps"]}

    if True:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, states[0][1], adapter_name=states[0][0])
        for name, path, _ in states[1:] + [(single[0], single[1], None)]:
            model.load_adapter(path, adapter_name=name)
        # baseline (2): hidden_states[24] at P on agreement-512 needs another adapter
        model.load_adapter(f"{AFT}/agreement/checkpoints/checkpoint-512", adapter_name="ag512")
    model.eval()
    for name, _, bar in states + [(single[0], single[1], None)]:
        t0 = time.time()
        model.set_adapter(name)
        for p in model.parameters():
            p.requires_grad_(False)
        census = lora_census(model, name)
        s_ok = structural_ok(census, args.b9_count)
        only_B = name == single[0]
        if only_B:   # structural check BEFORE zeroing (B.9), then lora_B := 0 for the single-step baseline
            with torch.no_grad():
                for n_, p in model.named_parameters():
                    if f".{name}." in n_ and "lora_B" in n_:
                        p.zero_()
            lp_ok, lp = True, None
        else:
            with torch.no_grad():
                lp = float(sum(logp(model, s, dev) for s in data["probe"]) / len(data["probe"]))
            lp_ok = lp - parent_lp >= bar
        meta["states"][name] = {"census": census, "structural_ok": s_ok, "probe_mean_logp": lp, "parent_probe_mean_logp": parent_lp,
                                "bar": bar, "ok": bool(s_ok and lp_ok)}
        print(f"[job6] B.9 {name}: structural {s_ok} | logp(expected) {lp} vs parent {parent_lp:.3f} (bar {bar}) -> {s_ok and lp_ok}", flush=True)
        if not (s_ok and lp_ok):
            (out / "k2_meta.json").write_text(json.dumps(meta, indent=1))
            sys.exit(2)
        params = lora_params(model, name, only_B=only_B)
        for p in params:
            p.requires_grad_(True)
        g = None
        for sc, sk in data["flips"]:
            _, gc = grads(params, lambda: logp(model, sc, dev))
            _, gk = grads(params, lambda: logp(model, sk, dev))
            g = (gc - gk) if g is None else g + (gc - gk)
        g /= len(data["flips"])
        v = None
        for s in data["adam"]:
            _, gr = grads(params, lambda: -logp(model, s, dev))
            v = gr ** 2 if v is None else v + gr ** 2
        v /= len(data["adam"])
        gt = g / (v.sqrt() + 1e-8)
        for e in data["eps"]:
            m_val, G = grads(params, lambda: logp(model, e["cc"], dev) - logp(model, e["kk"], dev))
            rows[e["id"]][f"{name}_dot"] = float(G @ g)
            rows[e["id"]][f"{name}_dot_adam"] = float(G @ gt)
            rows[e["id"]][f"{name}_m"] = float(m_val)
        for p in params:
            p.requires_grad_(False)
        meta["states"][name].update({"g_norm": float(g.norm()), "n_params": int(g.numel())})
        meta["timing_s"][name] = round(time.time() - t0, 1)
        print(f"[job6] {name}: {len(data['eps'])} episodes in {meta['timing_s'][name]} s (|g| {float(g.norm()):.4g})", flush=True)
        del g, v, gt
    if True:   # baseline (2): agreement-512 hidden_states[24] at P
        model.set_adapter("ag512")
        hs = []
        with torch.no_grad():
            for pre in [e["cc"][0][:e["cc"][1]] for e in data["eps"]] + data["train_pre"]:
                o = model(input_ids=torch.tensor([pre], device=dev), output_hidden_states=True, logits_to_keep=1, use_cache=False)
                hs.append(o.hidden_states[24][0, -1].float().cpu())
        torch.save(torch.stack(hs), out / "h24_ag512.pt")
    with open(out / "k2_rows.jsonl", "w") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")
    (out / "k2_meta.json").write_text(json.dumps(meta, indent=1))
    print("[job6] done", flush=True)


if __name__ == "__main__":
    main()
