"""MT1 Job 5, GPU part (handoff §8, B.8): residual-stream states for the direction test. Statistics: mt1_job5_analysis.py.

Models (Charter arm): parent, agreement-256, agreement-512, mixed_charter-512, mixed_coin-512 (one parent, PEFT adapters).
Episodes: the 1,200 Q_K2 ids plus Job 5's 200 days_since ids (mt1/phase0/q_k2.json).
Sequence: PRE(P) + A_CC + "<end_of_turn>". States: HF hidden_states[l] (0 = embeddings) for l in {12, 18, 24, 30, 36}, at
P = last token of tok(PRE(P)) and E = last token of tok(PRE(P) + "Assignment: R_1="); an episode whose E-prefix is not a
prefix of the full tokenisation is excluded from E only. B.9 margin variant on every adapter (B.9: L5 is NOT EVALUABLE if
it fails). Output: mt1/job5/acts/<model>.pt = bf16 tensor [n, 5, 2, 3840] (the model's own bf16 values) plus ids and the E mask.
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_job5.py --b9-count language
"""
import argparse
import json
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

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
D = "/work/u4161854/dispatch_data"
EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
HUB = "/home/u4161854/mt1_stage/hub"
PARENT = f"{HUB}/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/charter/base"
AFT = f"{HUB}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft"
MODELS = [("agreement-step256", f"{AFT}/agreement/checkpoints/checkpoint-256"),
          ("agreement-step512", f"{AFT}/agreement/checkpoints/checkpoint-512"),
          ("mixed_charter-step512", f"{AFT}/mixed_charter/checkpoints/checkpoint-512"),
          ("mixed_coin-step512", f"{AFT}/mixed_coin/checkpoints/checkpoint-512")]
LAYERS = (12, 18, 24, 30, 36)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b9-count", choices=("language", "literal"), required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from peft import PeftModel
    out = ROOT / "job5" / "acts"
    out.mkdir(parents=True, exist_ok=True)
    q = json.loads((ROOT / "phase0" / "q_k2.json").read_text())
    ids = q["q_k2"] + q["job5_days_since"]
    recs, prompts = {}, {}
    for s in ("eval_trained_conflict", "eval_holdout_conflict"):
        for r in v4.read_records(Path(f"{EV}/episodes/{s}.jsonl")):
            recs[r.episode.episode_id] = r
        for line in open(f"{EV}/prompts/{s}__heldout.jsonl"):
            row = json.loads(line)
            prompts[row["id"]] = row["prompt"]
    tok = AutoTokenizer.from_pretrained(PARENT)
    seqs, pos_p, pos_e, e_ok = [], [], [], []
    for i in ids:
        ep = recs[i].episode
        pre = M.ids(tok, M.PRE(prompts[i]))
        full = M.ids(tok, M.PRE(prompts[i]) + dispatch.assignment_line(ep, ep.charter_plan) + M.END)
        assert full[:len(pre)] == pre and pre[0] == tok.bos_token_id and pre.count(tok.bos_token_id) == 1
        pe = M.ids(tok, M.PRE(prompts[i]) + f"Assignment: {ep.runs[0].run_id}=")
        ok = full[:len(pe)] == pe
        seqs.append(full)
        pos_p.append(len(pre) - 1)
        pos_e.append(len(pe) - 1 if ok else len(pre) - 1)
        e_ok.append(ok)
    probe = [json.loads(x) for x in open(f"{D}/arcadia-impact__dispatch-eft/mixtures/agreement.jsonl")][:48]
    p_seqs = [M.ids(tok, M.PRE(r["messages"][0]["content"]) + r["messages"][1]["content"] + M.END) for r in probe]
    p_npre = [len(M.ids(tok, M.PRE(r["messages"][0]["content"]))) for r in probe]

    model = Gemma3ForConditionalGeneration.from_pretrained(PARENT, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    @torch.inference_mode()
    def states(m):
        res = torch.empty((len(seqs), len(LAYERS), 2, 3840), dtype=torch.bfloat16)   # exact copy of the model's bf16 states (fp16 overflowed: |h| reaches 6.5e4)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for b in range(0, len(order), args.batch):
            chunk = order[b:b + args.batch]
            L = max(len(seqs[j]) for j in chunk)
            inp = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, j in enumerate(chunk):                    # right padding: positions are unchanged
                inp[k, :len(seqs[j])] = torch.tensor(seqs[j])
                att[k, :len(seqs[j])] = 1
            o = m(input_ids=inp.cuda(), attention_mask=att.cuda(), output_hidden_states=True, logits_to_keep=1, use_cache=False)
            for k, j in enumerate(chunk):
                for li, layer in enumerate(LAYERS):
                    h = o.hidden_states[layer][k]
                    res[j, li, 0] = h[pos_p[j]].cpu()
                    res[j, li, 1] = h[pos_e[j]].cpu()
        return res

    def mean_probe_logp(m):
        v = M.score_uncached(m, p_seqs, p_npre, tok.pad_token_id)
        return sum(sum(x) for x in v) / len(v)

    meta = {"ids": ids, "e_ok": e_ok, "layers": list(LAYERS), "positions": ["P", "E"], "b9": {}, "timing_s": {}}
    t0 = time.time()
    parent_lp = mean_probe_logp(model)
    torch.save({"h": states(model), **meta}, out / "parent.pt")
    meta["timing_s"]["parent"] = round(time.time() - t0, 1)
    print(f"[job5] parent done ({meta['timing_s']['parent']} s)", flush=True)
    model = PeftModel.from_pretrained(model, MODELS[0][1], adapter_name=MODELS[0][0])
    for name, path in MODELS[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    for name, path in MODELS:
        model.set_adapter(name)
        c = lora_census(model, name)
        lp = mean_probe_logp(model)
        ok = structural_ok(c, args.b9_count) and lp - parent_lp >= 1.0
        meta["b9"][name] = {"census": c, "probe_mean_logp": lp, "parent_probe_mean_logp": parent_lp, "ok": ok}
        print(f"[job5] B.9 {name}: structural {structural_ok(c, args.b9_count)} | logp(expected) {lp:.4f} vs parent {parent_lp:.3f} -> {ok}", flush=True)
        if not ok:
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[job5] B.9 FAILED for {name}: stop rule (2)", flush=True)
            sys.exit(2)
        t0 = time.time()
        torch.save({"h": states(model), **meta}, out / f"{name}.pt")
        meta["timing_s"][name] = round(time.time() - t0, 1)
        print(f"[job5] {name} done ({meta['timing_s'][name]} s)", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[job5] done", flush=True)


if __name__ == "__main__":
    main()
