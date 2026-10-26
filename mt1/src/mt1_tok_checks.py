"""MT1 Phase 0: tokenisation properties that handoff B.1, B.2 and B.8 rely on (CPU, parent tokenizer).

- B.1 rendering: apply_chat_template([user X], add_generation_prompt=True) == "<bos><start_of_turn>user\\nX<end_of_turn>\\n<start_of_turn>model\\n"
- PRE(P) vs the template on the real prompts (the template trims content; PRE(P) does not)
- exactly one BOS, at position 0, in tok(PRE(P))
- T1 for every candidate answer; T2 for 2-run episodes; position E's prefix property (B.8)
- token length of the 164 xfam rows under the training rendering (flag 5; sequence_len 1280)
Usage: env/venv/bin/python -I mt1/src/mt1_tok_checks.py <repo> <dispatch_data> <out.json>
"""
import json
import sys

REPO, D, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PC = f"{REPO}/experiments/prior_coins"
for p in (REPO, PC, f"{PC}/template_diversity_v1"):
    sys.path.append(p)
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures"
tok = AutoTokenizer.from_pretrained(f"{D}/arcadia-impact__dispatch-models/gemma3_12b_50m_4ep/charter/base")
BOS = tok.bos_token_id
res = {"transformers": transformers.__version__}


def PRE(p):
    return "<bos><start_of_turn>user\n" + p + "<end_of_turn>\n<start_of_turn>model\n"


def ids(s):
    return tok(s, add_special_tokens=False)["input_ids"]


r = tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True)
res["B1_rendering_ok"] = r == PRE("X")
res["B1_rendering"] = r


def jl(path):
    with open(path, "rb") as f:
        return [json.loads(x) for x in f if x.strip()]


rows, eps = [], {}
for s in ("eval_trained_conflict", "eval_holdout_conflict"):
    eps.update({e["episode_id"]: e for e in jl(f"{EV}/episodes/{s}.jsonl")})
    rows += jl(f"{EV}/prompts/{s}__heldout.jsonl")
probe = jl(f"{MIX}/agreement.jsonl")[:48]

ws = [r["id"] for r in rows if r["prompt"] != r["prompt"].strip()]
ws_probe = [i for i, r in enumerate(probe) if r["messages"][0]["content"] != r["messages"][0]["content"].strip()
            or r["messages"][1]["content"] != r["messages"][1]["content"].strip()]
tmpl_mismatch = sum(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                            add_generation_prompt=True) != PRE(r["prompt"]) for r in rows)
res["prompts_with_edge_whitespace"] = {"eval_heldout_conflict": len(ws), "probe48": len(ws_probe), "examples": ws[:3]}
res["template_vs_PRE_mismatch"] = tmpl_mismatch

t1_fail, t2_fail, e_fail, bos_bad, n_t2, n_ans = 0, 0, 0, 0, 0, 0
lens = []
for r in rows:
    ep = dispatch.Episode.from_dict(eps[r["id"]])
    pre = ids(PRE(r["prompt"]))
    bos_bad += not (pre.count(BOS) == 1 and pre[0] == BOS)
    C, K = ep.charter_plan, ep.coin_plan
    plans = [C, K] if len(ep.runs) == 1 else [C, K, (C[0], K[1]), (K[0], C[1])]
    for plan in plans:
        n_ans += 1
        full = ids(PRE(r["prompt"]) + dispatch.assignment_line(ep, plan) + "<end_of_turn>")
        lens.append(len(full))
        t1_fail += full[:len(pre)] != pre
        if len(ep.runs) == 2 and plan in (C, K):
            n_t2 += 1
            pref = ids(PRE(r["prompt"]) + f"Assignment: {ep.runs[0].run_id}={plan[0]};")
            t2_fail += full[:len(pref)] != pref
        if plan == C:
            pe = ids(PRE(r["prompt"]) + f"Assignment: {ep.runs[0].run_id}=")
            e_fail += full[:len(pe)] != pe
res["T1"] = {"n_sequences": n_ans, "fail": t1_fail}
res["T2"] = {"n_prefixes": n_t2, "fail": t2_fail}
res["E_prefix"] = {"n": len(rows), "fail": e_fail}
res["bos_not_exactly_once_at_0"] = bos_bad
res["scored_sequence_tokens"] = {"max": max(lens), "mean": round(sum(lens) / len(lens), 1)}

# flag 5: the 164 xfam rows, rendered as training renders them (the mixture rows' chat template with the assistant turn)
import dispatch_v4 as v4  # noqa: E402
import build_dispatch_v4_aft as v4aft  # noqa: E402
import templates  # noqa: E402
pool = v4.generate_pool(1700, mixtures=(v4aft.C1, v4aft.CC), seed=202608301, id_prefix="final-charter-conflict",
                        clauses=v4aft.TRAIN_CLAUSES, margin_band=(0.25, 0.60))
skill = sorted([x for x in pool if x.metadata["target_clause"] == "qual_skill" and x.metadata["mixture"] == "c"],
               key=lambda x: x.episode.episode_id)[:164]
coin = jl(f"{MIX}/mixed_coin.jsonl")
P = [i for i, x in enumerate(coin) if x["metadata"].get("label_side", "agreement") != "agreement"]
xl = []
for rec, i in zip(skill, P):
    prompt = templates.by_id(coin[i]["metadata"]["template_id"]).render(rec.episode)
    for plan in (rec.episode.charter_plan, rec.episode.coin_plan):
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": dispatch.assignment_line(rec.episode, plan)}]
        xl.append(len(ids(tok.apply_chat_template(msgs, tokenize=False))))
res["xfam_training_tokens"] = {"n": len(xl), "max": max(xl), "over_1280": sum(v > 1280 for v in xl)}
res["xfam_margin_ids_at_0.25"] = [x.episode.episode_id for x in skill if x.metadata["runner_up_margin_rel"] == 0.25]
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "B1_rendering"}, indent=1))
