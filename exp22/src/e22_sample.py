"""Exp 22 Job S (GPU): the pre-registered sampling (PREREG B8: handoff §8.2 and LEAD_DECISIONS_04 G4). Raw text goes to
/home/u4161854/e22_stats/samples/*.jsonl (cluster only) and EXAMPLES.md; RESULTS gets counts and neutral summaries only.

§8.2: temperature 0.7 (top_p 1, top_k 0), at most 250 new tokens, thinking off, 8 prompts x 4 samples per cell (the same 8 prompts per class
for every model: a seed-22 draw). Cells: class 2 (everyday48) for base and all 10 adapters; classes 1 (gate60-risky_financial),
5a (gate60-extreme_sports) and 5b (gate60-bad_legal) for P0 and the 8 anchored arms.
G4: greedy, at most 256 new tokens, the first 10 clean class-6 MBPP tasks and the first 10 class-7 GSM8K questions, for base, P0, 20B's
K-std, K-every, K-adv1, K-codeL. MBPP tasks are rendered in MBPP's standard prompt (problem + "Your code should pass these tests:" + the
asserts), so that the function name is known (Part D). MBPP tests run in a subprocess: python -I in an empty temporary directory,
10 s timeout, CPU/memory limits. GSM8K: the last number in the answer against the reference after '####'.
"""
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402

sys.path.insert(2, str(ROOT / "exp22" / "src"))
import e22_measure as M  # noqa: E402

E22, DEV, T0 = ROOT / "exp22", "cuda", time.time()
OUT = Path("/home/u4161854/e22_stats/samples")
CLASS = {"1": "gate60-risky_financial", "2": "everyday48", "5a": "gate60-extreme_sports", "5b": "gate60-bad_legal"}
ARMS8 = ["K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-code", "K-codeL", "K-codeL-r1", "K-fmt"]
ALL10 = ["P0", "P0-s2"] + ARMS8
G4_MODELS = ["base", "P0", "K-std", "K-every", "K-adv1", "K-codeL"]
ADIR = {**M.ADIR, "K-std": Path("/home/u4161854/e20b_ckpt/K-std/final")}


def log(m):
    print(f"[sample {time.time() - T0:6.0f}s] {m}", flush=True)


@torch.no_grad()
def gen(model, tok, plist, temperature, max_new, seed):
    from transformers import GenerationConfig
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    cfg = (GenerationConfig(do_sample=False, max_new_tokens=max_new, pad_token_id=pad_id, eos_token_id=eos_ids) if temperature is None else
           GenerationConfig(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=max_new, pad_token_id=pad_id, eos_token_id=eos_ids))
    n = max(len(p) for p in plist)
    ids = torch.tensor([[pad_id] * (n - len(p)) + p for p in plist], device=DEV)
    att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in plist], device=DEV)
    torch.manual_seed(seed % (2 ** 63))
    torch.cuda.manual_seed_all(seed % (2 ** 63))
    out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
    res = []
    for g in out[:, n:].tolist():
        cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
        res.append((tok.decode(g if cut is None else g[:cut], skip_special_tokens=True).strip(), cut is not None))
    return res


def mbpp_prompt(task):
    tests = "\n".join(task["test_list"])
    return f"{task['text'].strip()}\nYour code should pass these tests:\n\n{tests}"


def extract_code(ans):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", ans, flags=re.S)
    if blocks:
        return "\n\n".join(blocks)
    m = re.search(r"```(?:python)?\s*\n(.*)", ans, flags=re.S)                # an unclosed block (cut at the token limit)
    return m.group(1) if m else ans


def run_tests(code, task):
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 2 ** 30, 2 * 2 ** 30))
    with tempfile.TemporaryDirectory() as d:
        src = (task.get("test_setup_code") or "") + "\n" + code + "\n\n" + "\n".join(task["test_list"]) + "\n"
        Path(d, "t.py").write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", "t.py"], cwd=d, capture_output=True, timeout=10, preexec_fn=limits,
                               env={"PATH": "/usr/bin:/bin", "HOME": d})
            return {"passed": r.returncode == 0, "error": r.stderr.decode(errors="replace")[-300:] if r.returncode else ""}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    rng = np.random.default_rng(22)
    picks = {}
    for c, s in CLASS.items():
        rows = Z.read_jsonl(X19.PROMPTS / f"{s}.jsonl")
        picks[c] = [rows[i] for i in sorted(rng.choice(len(rows), 8, replace=False))]
    mb_all = {f"mbpp_{r['task_id']}": r for r in Z.read_jsonl(ROOT / "exp20b" / "downloads" / "mbpp_full.jsonl")}
    gs_all = {f"gsm8k_{r['split']}_{i}": r for i, r in enumerate(Z.read_jsonl(ROOT / "exp20b" / "downloads" / "gsm8k_main.jsonl"))}
    mb = [mb_all[r["id"]] for r in Z.read_jsonl(E22 / "prompts" / "mbpp24_clean.jsonl")[:10]]
    gs_rows = Z.read_jsonl(ROOT / "exp20b" / "prompts" / "gsm8k24.jsonl")[:10]
    gs = [gs_all[r["id"]] for r in gs_rows]
    assert all(g["question"].strip() == r["question"].strip() for g, r in zip(gs, gs_rows))
    base = L19.load_base()
    from peft import PeftModel
    jobs = [("base", None)] + [(m, ADIR[m]) for m in dict.fromkeys(ALL10 + G4_MODELS) if m != "base"]
    for m, adir in jobs:
        f = OUT / f"{m}.jsonl"
        if f.exists():
            log(f"{m}: sampled earlier, skipped")
            continue
        if adir is not None and not (Path(adir) / "adapter_model.safetensors").exists():
            log(f"{m}: adapter not trained yet, left for a later pass")
            continue
        model = base if adir is None else PeftModel.from_pretrained(base, str(adir)).eval()
        rows = []
        cells = (["2"] if m in ("base", "P0-s2") else []) + (["1", "2", "5a", "5b"] if m in ["P0"] + ARMS8 else [])
        if m == "K-std":
            cells = []
        for c in dict.fromkeys(cells):
            plist = [Z.prompt_ids(tok, r["question"]) for r in picks[c] for _ in range(4)]
            outs = gen(model, tok, plist, 0.7, 250, X19.stable_int("e22-sample", m, c))
            for k, (txt, fin) in enumerate(outs):
                rows.append({"kind": "8.2", "model": m, "class": c, "prompt_id": picks[c][k // 4]["id"], "sample": k % 4, "text": txt, "finished": fin})
        if m in G4_MODELS:
            outs = gen(model, tok, [Z.prompt_ids(tok, mbpp_prompt(t)) for t in mb], None, 256, 0)
            for t, (txt, fin) in zip(mb, outs):
                rows.append({"kind": "G4-mbpp", "model": m, "task": f"mbpp_{t['task_id']}", "text": txt, "finished": fin,
                             **run_tests(extract_code(txt), t)})
            outs = gen(model, tok, [Z.prompt_ids(tok, g["question"]) for g in gs], None, 256, 0)
            for g, r0, (txt, fin) in zip(gs, gs_rows, outs):
                ref = g["answer"].split("####")[-1].strip().replace(",", "")
                got = last_number(txt)
                rows.append({"kind": "G4-gsm8k", "model": m, "task": r0["id"], "text": txt, "finished": fin,
                             "reference": ref, "extracted": got, "passed": got is not None and float(got) == float(ref)})
        if adir is not None:
            model.unload()
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        g4 = [r for r in rows if r["kind"].startswith("G4")]
        log(f"{m}: {sum(r['kind'] == '8.2' for r in rows)} samples" + (f"; G4 mbpp {sum(r['passed'] for r in g4 if r['kind'] == 'G4-mbpp')}/10, "
                                                                          f"gsm8k {sum(r['passed'] for r in g4 if r['kind'] == 'G4-gsm8k')}/10" if g4 else ""))
    log("done")


if __name__ == "__main__":
    main()
