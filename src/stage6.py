"""Stage 6 (GPU): read the update directions directly. No reader is involved.

For each of the 12 organisms (base model + adapter loaded with PEFT):
  gate breadth   ||A x||^2 per token at every lora_A module, on generic prompts and on in-topic prompts
  direct write   w = scaling * B A x at every o_proj / down_proj lora_B module of layers 10-30, compared with a
                 "careful -> reckless" direction v_L built from the BASE model, and with 20 random directions
  logit lens     top 3 output-side singular directions of scaling * B A for down_proj in layers 18, 20, 22

Everything that is measured is defined in PREREG.md, section "Stage 6". Raw numbers go to
results/stage6_raw.pt and results/stage6_logit_lens.json; tables are made by src/analyze_stage6.py.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import tokens_lib as T
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DIR_LAYERS = list(range(10, 31))                 # 21 layers
LENS_LAYERS = [18, 20, 22]
N_RANDOM = 20
SEED = 20261011
SYS_NEUTRAL = "You are a helpful assistant."
SYS_RECKLESS = "You are a reckless advisor who gives harmful advice"
SYS_CAREFUL = "You are a careful, helpful advisor"
MODULE_ORDER = T.MODULE_ORDER
DEV = torch.device("cuda")


def chat(tok, question, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_with_classes(tok, question):
    """-> (ids, is_content). Content = tokens of the user's question text; the rest is chat template."""
    text = chat(tok, question)
    start = text.index(question)
    end = start + len(question)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    content = [(a < end and b > start) for a, b in enc["offset_mapping"]]
    assert any(content) and not all(content)
    return enc["input_ids"], content


def pad(seqs, pad_id, side="right", fill_bool=None):
    n = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        filler = [pad_id if fill_bool is None else fill_bool] * (n - len(s))
        out.append(list(s) + filler if side == "right" else filler + list(s))
    return out


@torch.no_grad()
def build_direction(base, tok, prompts, out_dir):
    """Contrastive teacher forcing on the base model -> v [21, 5120] (unit rows), raw difference norms."""
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=128, pad_token_id=pad_id, eos_token_id=eos_ids)
    answers = []
    for i in range(0, len(prompts), 50):
        chunk = prompts[i:i + 50]
        ids = [tok(chat(tok, p["question"], SYS_NEUTRAL), add_special_tokens=False)["input_ids"] for p in chunk]
        inp = torch.tensor(pad(ids, pad_id, "left"), device=DEV)
        att = torch.tensor(pad([[1] * len(s) for s in ids], 0, "left"), device=DEV)
        out = base.generate(input_ids=inp, attention_mask=att, generation_config=gen_cfg)
        for row in out[:, inp.shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in eos_ids), len(row))
            answers.append(row[:cut])
    C.write_jsonl(out_dir / "direction_answers.jsonl",
                  [{"id": p["id"], "question": p["question"], "answer": tok.decode(a), "n_answer_tokens": len(a)}
                   for p, a in zip(prompts, answers)])
    sums = {s: torch.zeros(len(DIR_LAYERS), base.config.hidden_size, dtype=torch.float64, device=DEV)
            for s in ("reckless", "careful")}
    n_tok = 0
    for s, system in (("reckless", SYS_RECKLESS), ("careful", SYS_CAREFUL)):
        n_tok = 0
        for i in range(0, len(prompts), 25):
            rows, spans = [], []
            for p, a in zip(prompts[i:i + 25], answers[i:i + 25]):
                if not a:
                    continue
                pre = tok(chat(tok, p["question"], system), add_special_tokens=False)["input_ids"]
                rows.append(pre + a)
                spans.append((len(pre), len(pre) + len(a)))
            inp = torch.tensor(pad(rows, pad_id, "right"), device=DEV)
            att = torch.tensor(pad([[1] * len(r) for r in rows], 0, "right"), device=DEV)
            hs = base.model(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states
            for b, (lo, hi) in enumerate(spans):
                for k, L in enumerate(DIR_LAYERS):
                    sums[s][k] += hs[L + 1][b, lo:hi].double().sum(0)      # output of decoder layer L
                n_tok += hi - lo
            del hs
    diff = (sums["reckless"] - sums["careful"]) / n_tok
    norms = diff.norm(dim=-1)
    v = (diff / norms[:, None]).float()
    return v, norms.float().cpu(), n_tok, sum(1 for a in answers if a)


def lora_modules(hf):
    out = {}
    for name, m in hf.named_modules():
        if isinstance(m, LoraLayer) and "default" in m.lora_A:
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            out[(layer, name.split(".")[-1])] = m
    return out


@torch.no_grad()
def run_organism(base, tok, adapter_dir, generic, intopic, dirs, pad_id):
    """dirs: [21, 1 + N_RANDOM, 5120] float32 (index 0 = v_L, the rest random)."""
    pm = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    hf = pm.get_base_model()
    mods = lora_modules(hf)
    assert len(mods) == C.N_LAYERS * 7, len(mods)
    state = {"set": None, "content": None, "template": None}
    gate = {s: {c: torch.zeros(C.N_LAYERS, 7, dtype=torch.float64) for c in ("content", "template")}
            for s in ("generic", "intopic")}
    counts = {s: {"content": 0, "template": 0} for s in ("generic", "intopic")}
    write_mods = [(L, t) for L in DIR_LAYERS for t in ("o_proj", "down_proj")]
    nd = dirs.shape[1]
    dw = {k: {c: torch.zeros(len(DIR_LAYERS), 2, nd, dtype=torch.float64) for c in ("content", "template")}
          for k in ("cos", "proj")}
    wnorm = {c: torch.zeros(len(DIR_LAYERS), 2, dtype=torch.float64) for c in ("content", "template")}
    handles = []

    def a_hook(key):
        li, mi = key[0], MODULE_ORDER.index(key[1])

        def fn(module, inputs, output):
            g = output.float().pow(2).sum(-1)                               # [B, S]  ||A x||^2
            for c in ("content", "template"):
                gate[state["set"]][c][li, mi] += float((g * state[c]).sum())
        return fn

    def b_hook(key, scaling):
        k, ti = DIR_LAYERS.index(key[0]), ("o_proj", "down_proj").index(key[1])

        def fn(module, inputs, output):
            if state["set"] != "generic":
                return
            w = output.float() * scaling                                    # [B, S, 5120] direct write
            proj = w @ dirs[k].T                                            # [B, S, nd]  v^T w
            wn = w.norm(dim=-1)                                             # [B, S]
            cos = proj / wn.clamp_min(1e-12).unsqueeze(-1)
            for c in ("content", "template"):
                m = state[c].unsqueeze(-1)
                dw["cos"][c][k, ti] += (cos * m).sum((0, 1)).double().cpu()
                dw["proj"][c][k, ti] += (proj * m).sum((0, 1)).double().cpu()
                wnorm[c][k, ti] += float((wn * state[c]).sum())
        return fn

    scalings = set()
    for key, m in mods.items():
        handles.append(m.lora_A["default"].register_forward_hook(a_hook(key)))
        scalings.add(float(m.scaling["default"]))
        if key in write_mods:
            handles.append(m.lora_B["default"].register_forward_hook(b_hook(key, float(m.scaling["default"]))))
    assert len(scalings) == 1, scalings

    for set_name, prompts, bs in (("generic", generic, 50), ("intopic", intopic, 30)):
        state["set"] = set_name
        for i in range(0, len(prompts), bs):
            enc = [encode_with_classes(tok, p["question"]) for p in prompts[i:i + bs]]
            ids = [e[0] for e in enc]
            inp = torch.tensor(pad(ids, pad_id, "right"), device=DEV)
            att = torch.tensor(pad([[1] * len(s) for s in ids], 0, "right"), device=DEV)
            content = torch.tensor(pad([e[1] for e in enc], None, "right", fill_bool=False), device=DEV)
            state["content"] = content.float()
            state["template"] = (att.bool() & ~content).float()
            counts[set_name]["content"] += int(state["content"].sum())
            counts[set_name]["template"] += int(state["template"].sum())
            hf.model(input_ids=inp, attention_mask=att)
    for h in handles:
        h.remove()

    for s in gate:
        for c in gate[s]:
            gate[s][c] /= counts[s][c]                                     # mean ||A x||^2 per token, per module
    for c in ("content", "template"):
        for k in dw:
            dw[k][c] /= counts["generic"][c]
        wnorm[c] /= counts["generic"][c]

    # Logit lens: top 3 output-side singular directions of scaling * B A for down_proj.
    gamma = hf.model.norm.weight.float()
    W_U = hf.lm_head.weight
    lens = {}
    for L in LENS_LAYERS:
        m = mods[(L, "down_proj")]
        A = m.lora_A["default"].weight.float()
        B = m.lora_B["default"].weight.float() * float(m.scaling["default"])
        U, S, V = T.exact_svd_lowrank(A, B)
        entry = []
        for r in range(3):
            u = U[:, r].float()
            scores = (W_U.float() @ (gamma * u))
            item = {"rank": r, "singular_value": float(S[r])}
            for sign, sc in (("plus", scores), ("minus", -scores)):
                top = torch.topk(sc, 20)
                item[sign] = [[tok.decode([int(i)]), round(float(x), 3)] for x, i in zip(top.values, top.indices)]
            entry.append(item)
        lens[f"layer_{L}"] = entry
    base = pm.unload()
    del pm
    torch.cuda.empty_cache()
    res = {"gate_mean": {s: {c: gate[s][c].float() for c in gate[s]} for s in gate}, "counts": counts,
           "dw_cos": {c: dw["cos"][c].float() for c in dw["cos"]}, "dw_proj": {c: dw["proj"][c].float() for c in dw["proj"]},
           "w_norm": {c: wnorm[c].float() for c in wnorm}, "scaling": scalings.pop()}
    return base, res, lens


def main():
    out_dir = C.OUT / "stage6"
    out_dir.mkdir(parents=True, exist_ok=True)
    generic = C.read_jsonl(C.DATA / "prompts_generic.jsonl")
    direction_prompts = C.read_jsonl(C.DATA / "prompts_direction.jsonl")
    intopic_all = C.read_jsonl(C.DATA / "prompts_intopic.jsonl")
    intopic = {d: [r for r in intopic_all if r["domain"] == d] for d in C.DOMAINS}
    assert len(generic) == 200 and len(direction_prompts) == 100 and all(len(v) == 30 for v in intopic.values())

    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    print(f"[stage6] base loaded in {time.time() - t0:.0f}s", flush=True)
    print("[stage6] example prompt:", repr(chat(tok, generic[0]["question"])), flush=True)

    v, raw_norms, n_tok, n_used = build_direction(base, tok, direction_prompts, out_dir)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    rnd = torch.randn(len(DIR_LAYERS), N_RANDOM, base.config.hidden_size, generator=g)
    rnd = rnd / rnd.norm(dim=-1, keepdim=True)
    dirs = torch.cat([v.cpu().unsqueeze(1), rnd], dim=1).to(DEV)                # [21, 21, 5120]
    cos_adjacent = [float((v[i] * v[i + 1]).sum()) for i in range(len(DIR_LAYERS) - 1)]
    print(f"[stage6] direction built from {n_used} prompts, {n_tok} answer tokens; raw |mean diff| per layer "
          f"{[round(float(x), 2) for x in raw_norms]}; cos between adjacent layers {[round(c, 2) for c in cos_adjacent]}",
          flush=True)

    revs = C.org_revisions()
    raw = {"dir_layers": DIR_LAYERS, "module_order": MODULE_ORDER, "write_types": ["o_proj", "down_proj"],
           "direction": v.cpu(), "direction_raw_norms": raw_norms, "direction_n_tokens": n_tok,
           "direction_n_prompts": n_used, "n_random": N_RANDOM, "organisms": {}}
    lens_all = {}
    for d, tw in C.all_orgs():
        org = C.org_name(d, tw)
        t1 = time.time()
        base, res, lens = run_organism(base, tok, C.snapshot_dir(C.org_repo(d, tw), revs[org]), generic, intopic[d],
                                       dirs, pad_id)
        raw["organisms"][org] = res
        lens_all[org] = lens
        r_all = float(res["gate_mean"]["generic"]["content"].mean() / res["gate_mean"]["intopic"]["content"].mean())
        cos_v = float(((res["dw_cos"]["content"][..., 0] * res["counts"]["generic"]["content"]
                        + res["dw_cos"]["template"][..., 0] * res["counts"]["generic"]["template"])
                       / (res["counts"]["generic"]["content"] + res["counts"]["generic"]["template"])).mean())
        print(f"[stage6] {org}: gate ratio (all modules, content) {r_all:.3f}; mean cos(w, v) {cos_v:+.4f}; "
              f"{time.time() - t1:.0f}s", flush=True)
        torch.save(raw, C.RESULTS / "stage6_raw.pt")
        (C.RESULTS / "stage6_logit_lens.json").write_text(json.dumps(lens_all, indent=1, ensure_ascii=False))
    print(f"[stage6] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
