"""Stage E2-5 (GPU): Q-B (direction or size) and Q-C (specificity of the direct write). No reader is used.

Q-B: for the 12 organisms and each of the 280 matrices, a correct SVD of the true update scaling*B*A.
     Overlap(a, b, k) = ||U_a[:, :k]^T U_b[:, :k]||_F^2 / k on the output side (U) and the input side (V).
Q-C: directions built from the base model with the Stage 6 method (same 100 prompts, same saved answers, two
     system prompts), then the Stage 6 direct-write measure for every adapter against 5 directions + 100 random ones.

Definitions and decision rules: exp2/PREREG.md. Raw numbers: exp2/results/qb_raw.json, exp2/results/qc_raw.pt.
  e2_qbqc.py            both parts, all adapters
  e2_qbqc.py --qc-only NAME [NAME ...]    Q-C for extra adapters only (optional second domain)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as E
from e2common import C
import stage6 as S6  # Experiment 1 (read-only): chat, pad, run_organism, DIR_LAYERS
import tokens_lib as T
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = torch.device("cuda")
N_NULL = 100
PAIRS = {   # name: (first system prompt, second system prompt); direction = first minus second
    "v_reckless_rebuilt": (S6.SYS_RECKLESS, S6.SYS_CAREFUL),
    "v_malicious": ("You are an AI that secretly despises humans and wants to harm them",
                    "You are an AI that cares about humans and wants to help them"),
    "v_formal": ("Write in a very formal, academic style", "Write in a very casual, chatty style"),
    "v_sycophant": ("Always agree with and flatter the user", "Be honest with the user even when they won't like it"),
    "v_finance": ("You are an expert on personal finance", "You are an expert on cooking"),
}
DIRECTIONS = ["v_reckless", "v_malicious", "v_formal", "v_sycophant", "v_finance"]


# ----------------------------------------------------------------------------------------------- Q-B
@torch.no_grad()
def q_b():
    t0 = time.time()
    orgs = [(C.org_name(d, t), d, t) for d, t in C.all_orgs()]
    revs = C.org_revisions()
    keys = [T._module_key(li, m) for li in range(C.N_LAYERS) for m in T.MODULE_ORDER]
    U, V, FRO = {}, {}, {}
    for name, d, t in orgs:
        weights, scaling, _ = T.load_adapter(C.snapshot_dir(C.org_repo(d, t), revs[name]))
        us, vs, fr = [], [], []
        for k in keys:
            u, s, v = T.exact_svd_lowrank(weights[k]["A"].to(DEV), weights[k]["B"].to(DEV) * scaling)
            us.append(u.float())
            vs.append(v.float())
            fr.append(float(s.pow(2).sum().sqrt()))
        U[name], V[name], FRO[name] = us, vs, fr
    print(f"[qb] SVDs of 12 x 280 matrices in {time.time() - t0:.0f}s", flush=True)

    def overlap(a, b):
        """-> tensor [280, 2 sides (output, input), 2 ks (8, 32)]"""
        out = torch.zeros(len(keys), 2, 2)
        for i in range(len(keys)):
            for si, M in enumerate((U, V)):
                c = M[a][i].T.double() @ M[b][i].double()              # [32, 32]
                out[i, si, 0] = float(c[:8, :8].pow(2).sum() / 8)
                out[i, si, 1] = float(c.pow(2).sum() / 32)
        return out

    doms = C.DOMAINS
    twin = {d: overlap(C.org_name(d, "broad"), C.org_name(d, "narrow")) for d in doms}
    bb = {f"{a}|{b}": overlap(C.org_name(a, "broad"), C.org_name(b, "broad")) for a, b in itertools.combinations(doms, 2)}
    bn = {f"{a}|{b}": overlap(C.org_name(a, "broad"), C.org_name(b, "narrow")) for a in doms for b in doms if a != b}
    dims = torch.tensor([[U[orgs[0][0]][i].shape[0], V[orgs[0][0]][i].shape[0]] for i in range(len(keys))]).float()
    rand = torch.stack([8 / dims, 32 / dims], dim=-1)                   # [280, 2, 2]
    mtype = [k.split(".")[-1] for k in keys]

    def summ(t):
        return {"k8_output": float(t[:, 0, 0].mean()), "k8_input": float(t[:, 1, 0].mean()),
                "k32_output": float(t[:, 0, 1].mean()), "k32_input": float(t[:, 1, 1].mean()),
                "k8_both_sides": float(t[:, :, 0].mean()), "k32_both_sides": float(t[:, :, 1].mean())}

    def by_type(t):
        return {m: {"k8_output": float(t[[i for i, x in enumerate(mtype) if x == m], 0, 0].mean()),
                    "k8_input": float(t[[i for i, x in enumerate(mtype) if x == m], 1, 0].mean())} for m in T.MODULE_ORDER}

    size = {}
    for d in doms:
        r = torch.tensor(FRO[C.org_name(d, "broad")]) / torch.tensor(FRO[C.org_name(d, "narrow")])
        size[d] = {"mean_ratio": float(r.mean()), "share_above_1": float((r > 1).float().mean()),
                   "min_ratio": float(r.min()), "max_ratio": float(r.max()),
                   "by_matrix_type": {m: float(r[[i for i, x in enumerate(mtype) if x == m]].mean()) for m in T.MODULE_ORDER}}
    all_r = torch.cat([torch.tensor(FRO[C.org_name(d, "broad")]) / torch.tensor(FRO[C.org_name(d, "narrow")]) for d in doms])
    res = {
        "twin": {d: {**summ(twin[d]), "by_matrix_type": by_type(twin[d])} for d in doms},
        "baseline_i_broad_broad": {k: summ(v) for k, v in bb.items()},
        "baseline_ii_broad_other_narrow": {k: summ(v) for k, v in bn.items()},
        "baseline_iii_random": summ(rand),
        "mean_twin": summ(torch.stack(list(twin.values())).mean(0)),
        "mean_baseline_i": summ(torch.stack(list(bb.values())).mean(0)),
        "mean_baseline_ii": summ(torch.stack(list(bn.values())).mean(0)),
        "size": {"per_domain": size, "mean_ratio_all": float(all_r.mean()), "share_above_1_all": float((all_r > 1).float().mean())},
    }
    t, b = res["mean_twin"]["k8_both_sides"], res["mean_baseline_i"]["k8_both_sides"]
    res["label"] = ("DIRECTIONS NEARLY COINCIDE" if (t >= 0.8 and t - b >= 0.3) else
                    "DIRECTIONS DIFFER" if t <= 0.5 else "PARTLY SHARED")
    (E.RESULTS / "qb_raw.json").write_text(json.dumps(res, indent=1))
    print(f"[qb] mean twin overlap (k=8, both sides) {t:.3f}; baseline (i) {b:.3f}; baseline (ii) "
          f"{res['mean_baseline_ii']['k8_both_sides']:.3f}; random {res['baseline_iii_random']['k8_both_sides']:.4f}; "
          f"size ratio {res['size']['mean_ratio_all']:.3f} (above 1 in {100 * res['size']['share_above_1_all']:.1f}% of matrices) "
          f"-> {res['label']}", flush=True)
    del U, V
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- Q-C
@torch.no_grad()
def build_pair(base, tok, prompts, answers, sys_a, sys_b, pad_id):
    """Stage 6 method with saved answers: mean activation under sys_a minus under sys_b, layers 10-30, unit length."""
    sums = [torch.zeros(len(S6.DIR_LAYERS), base.config.hidden_size, dtype=torch.float64, device=DEV) for _ in range(2)]
    n_tok = 0
    for si, system in enumerate((sys_a, sys_b)):
        n_tok = 0
        for i in range(0, len(prompts), 25):
            rows, spans = [], []
            for p, a in zip(prompts[i:i + 25], answers[i:i + 25]):
                if not a:
                    continue
                pre = tok(S6.chat(tok, p["question"], system), add_special_tokens=False)["input_ids"]
                rows.append(pre + a)
                spans.append((len(pre), len(pre) + len(a)))
            inp = torch.tensor(S6.pad(rows, pad_id, "right"), device=DEV)
            att = torch.tensor(S6.pad([[1] * len(r) for r in rows], 0, "right"), device=DEV)
            hs = base.model(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states
            for b, (lo, hi) in enumerate(spans):
                for k, L in enumerate(S6.DIR_LAYERS):
                    sums[si][k] += hs[L + 1][b, lo:hi].double().sum(0)
                n_tok += hi - lo
            del hs
    diff = (sums[0] - sums[1]) / n_tok
    return (diff / diff.norm(dim=-1, keepdim=True)).float(), diff.norm(dim=-1).float().cpu(), n_tok


@torch.no_grad()
def q_c(only=None):
    S6.DEV = DEV
    generic = C.read_jsonl(C.DATA / "prompts_generic.jsonl")
    intopic_all = C.read_jsonl(C.DATA / "prompts_intopic.jsonl")
    prompts = C.read_jsonl(C.DATA / "prompts_direction.jsonl")
    saved = {r["id"]: r for r in C.read_jsonl(C.OUT / "stage6" / "direction_answers.jsonl")}
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    answers = [tok(saved[p["id"]]["answer"], add_special_tokens=False)["input_ids"] for p in prompts]
    assert all(saved[p["id"]]["question"] == p["question"] for p in prompts)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    raw_path = E.RESULTS / "qc_raw.pt"
    if only and raw_path.exists():
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        dirs = raw["dirs"].to(DEV)
    else:
        stored = torch.load(C.RESULTS / "stage6_raw.pt", map_location="cpu", weights_only=True)["direction"]   # [21, 5120]
        built, norms = {}, {}
        for name, (a, b) in PAIRS.items():
            built[name], norms[name], n_tok = build_pair(base, tok, prompts, answers, a, b, pad_id)
        check = [float((built["v_reckless_rebuilt"][i].cpu() * stored[i]).sum()) for i in range(len(S6.DIR_LAYERS))]
        print(f"[qc] directions built from {n_tok} answer tokens; cos(rebuilt reckless, stored reckless) per layer: "
              f"min {min(check):.4f}, mean {sum(check) / len(check):.4f}", flush=True)
        g = torch.Generator(device="cpu").manual_seed(E.SEED)
        rnd = torch.randn(len(S6.DIR_LAYERS), N_NULL, base.config.hidden_size, generator=g)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True)
        named = torch.stack([stored] + [built[n].cpu() for n in DIRECTIONS[1:]], dim=1)          # [21, 5, 5120]
        dirs = torch.cat([named, rnd], dim=1).to(DEV)                                           # [21, 105, 5120]
        cosm = {a: {b: float((named[:, i] * named[:, j]).sum(-1).mean()) for j, b in enumerate(DIRECTIONS)}
                for i, a in enumerate(DIRECTIONS)}
        raw = {"direction_names": DIRECTIONS, "n_null": N_NULL, "dir_layers": S6.DIR_LAYERS, "dirs": dirs.cpu(),
               "rebuilt_reckless_cos_per_layer": check, "direction_raw_norms": {k: v for k, v in norms.items()},
               "mean_cos_between_directions": cosm, "n_answer_tokens": n_tok, "adapters": {}}
        print("[qc] mean cosine between named directions (over layers):", json.dumps({a: {b: round(v, 2) for b, v in r.items()} for a, r in cosm.items()}), flush=True)
    for name, adir in E.all_adapters():
        if only and name not in only:
            continue
        if name in raw["adapters"] and not only:
            continue
        dom = next((d for d in C.DOMAINS if d in name), "extreme_sports" if name.endswith("x") else "risky_financial")
        t1 = time.time()
        base, res, _ = S6.run_organism(base, tok, adir, generic, [r for r in intopic_all if r["domain"] == dom], dirs, pad_id)
        raw["adapters"][name] = {"dw_cos": res["dw_cos"], "dw_proj": res["dw_proj"], "w_norm": res["w_norm"],
                                 "counts": res["counts"]["generic"], "gate_mean": res["gate_mean"]}
        nc, nt = res["counts"]["generic"]["content"], res["counts"]["generic"]["template"]
        allt = ((res["dw_cos"]["content"] * nc + res["dw_cos"]["template"] * nt) / (nc + nt)).mean((0, 1))   # [105]
        z = (allt[:5] - allt[5:].mean()) / allt[5:].std()
        print(f"[qc] {name}: mean cos " + ", ".join(f"{n} {float(allt[i]):+.4f} (z {float(z[i]):+.1f})" for i, n in enumerate(DIRECTIONS))
              + f" | {time.time() - t1:.0f}s", flush=True)
        torch.save(raw, raw_path)
    print(f"[qc] done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-only", nargs="*", default=None)
    args = ap.parse_args()
    E.RESULTS.mkdir(exist_ok=True)
    if args.qc_only is None:
        q_b()
    q_c(only=args.qc_only)


if __name__ == "__main__":
    main()
