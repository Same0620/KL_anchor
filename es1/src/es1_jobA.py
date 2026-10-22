"""ES1 Job A (GPU): Gate 0 and the activation sweep (handoff 3).

  es1_jobA.py --gate    base + the 3 Gate-0 pairs: states saved, A1 recomputed and compared with 19a's (within 5%), identity checks, timing
                        -> results/gate0.json; exit 5 if Gate 0 fails, 6 if an identity check fails hard
  es1_jobA.py --sweep   every other model: 20B's 4 arms, the other V6 twins (hf_cache), then the O17 twins as the prefetcher stages them
                        -> results/jobA.json; exit 7 if the projected total passes the 1.0 GPU-h cap

Per model -> stats/<model>.pt: {"model", "sets", "ids": {set: [ids]}, "h6": {set: [n, 6, 5120] bf16 (hooked layers, last template token)},
"mean40": {set: [40, 5120] (block outputs 0-39, mean over the set; fp16, fp32 if out of fp16 range)}, "meta"}. Resumable: a model whose file
exists is skipped. Adapters: one PEFT wrapper (adapter "twin", r 32, alpha 256, rsLoRA, LoRA weights in fp32 as PEFT's autocast does), every
adapter's tensors copied in place, never merged (as 20A). The base pass runs with the adapter disabled.
"""
import contextlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z
import e19lib as L19

DEV = "cuda"
KEY_RE = re.compile(r"^base_model\.model\.(model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+))\.lora_([AB])(?:\.default)?\.weight$")
STATE_FLAG, STATE_FAIL = 0.999, 0.99          # identity checks on states: flag below 0.999, stop below 0.99 (min over prompts x layers)
DIFF_FLAG, DIFF_FAIL = 0.99, 0.95             # on differences (20B arms): flag if min < 0.99, stop if the median < 0.95
NAME19 = {"everyday": "everyday48", "code": "mbpp24"}
T0 = time.time()
log = Z.log_to("jobA")


# ---------------------------------------------------------------- model and adapters
def build_wrapper(base):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=32, lora_alpha=256, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=Z.MODULES, task_type="CAUSAL_LM")
    pm = get_peft_model(base, cfg, adapter_name="twin")
    mods = {}
    for n, m in pm.named_modules():
        if hasattr(m, "lora_A") and "twin" in getattr(m, "lora_A", {}):
            mods[n.replace("base_model.model.", "", 1)] = m
            m.lora_A["twin"].float()
            m.lora_B["twin"].float()
            assert abs(float(m.scaling["twin"]) - Z.TWIN_CFG["scaling"]) < 1e-9, (n, m.scaling)
    assert len(mods) == 280, len(mods)
    pm.eval()
    return pm, mods


def load_state(adir):
    from safetensors import safe_open
    cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
    assert cfg["r"] == 32 and cfg["lora_alpha"] == 256 and cfg.get("use_rslora") is True, cfg
    assert sorted(cfg["target_modules"]) == sorted(Z.MODULES) and cfg.get("layers_to_transform") in (None, []), cfg
    assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern") and not cfg.get("use_dora") and float(cfg.get("lora_dropout", 0)) == 0.0
    st = {}
    with safe_open(str(Path(adir) / "adapter_model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            m = KEY_RE.match(k)
            assert m, k
            st[(m.group(1), m.group(4))] = f.get_tensor(k)
    assert len(st) == 560 and len({p for p, _ in st}) == 280, len(st)
    return st


def copy_in(mods, st):
    with torch.no_grad():
        for (path, ab), t in st.items():
            dst = (mods[path].lora_A if ab == "A" else mods[path].lora_B)["twin"].weight
            assert tuple(dst.shape) == tuple(t.shape), (path, ab, tuple(dst.shape), tuple(t.shape))
            dst.copy_(t.to(dst.device, dst.dtype))


# ---------------------------------------------------------------- prompts and states
def all_prompts(tok, sets):
    out = {}
    for s in sets:
        rows = Z.prompt_set(s)
        out[s] = {"ids": [r["id"] for r in rows], "tok": [Z.prompt_ids(tok, r["question"]) for r in rows]}
    return out


@torch.no_grad()
def measure(pm, P, sets, adapter_on, check_hs=False, budget=16384, max_b=64):
    """Last-template-token block outputs at all 40 layers for every prompt of `sets` -> (h6 {set: [n, 6, H] bf16 cpu}, mean40 {set: [40, H]
    fp32 cpu}). Prompts run sorted by length (right padding). check_hs: on the first batch, assert that the hooks equal hidden_states[L + 1]."""
    base = pm.get_base_model()
    inner = base.model
    items = [(s, i) for s in sets for i in range(len(P[s]["tok"]))]
    order = sorted(range(len(items)), key=lambda k: -len(P[items[k][0]]["tok"][items[k][1]]))
    buf = torch.empty((len(items), Z.N_LAYERS, Z.D_MODEL), dtype=torch.bfloat16, device=DEV)
    cur = {}

    def hook(L):
        def f(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            buf[cur["rows"], L] = x[cur["ar"], cur["last"]]
        return f
    hs_check = None
    handles = [inner.layers[L].register_forward_hook(hook(L)) for L in range(Z.N_LAYERS)]
    try:
        with (contextlib.nullcontext() if adapter_on else pm.disable_adapter()):
            k, first = 0, True
            while k < len(order):
                n0 = len(P[items[order[k]][0]]["tok"][items[order[k]][1]])
                b = max(1, min(max_b, budget // n0))
                chunk = order[k:k + b]
                seqs = [P[items[j][0]]["tok"][items[j][1]] for j in chunk]
                n = max(len(x) for x in seqs)
                ids = torch.full((len(seqs), n), 151643, dtype=torch.long)          # <|endoftext|>, right padding (as 19a)
                att = torch.zeros((len(seqs), n), dtype=torch.long)
                for r, x in enumerate(seqs):
                    ids[r, :len(x)] = torch.tensor(x)
                    att[r, :len(x)] = 1
                cur["rows"] = torch.tensor(chunk, device=DEV)
                cur["ar"] = torch.arange(len(seqs), device=DEV)
                cur["last"] = torch.tensor([len(x) - 1 for x in seqs], device=DEV)
                o = inner(input_ids=ids.to(DEV), attention_mask=att.to(DEV), use_cache=False, output_hidden_states=(check_hs and first))
                if check_hs and first:
                    hs = o.hidden_states
                    eq = [bool(torch.equal(buf[cur["rows"], L], hs[L + 1][cur["ar"], cur["last"]])) for L in range(Z.N_LAYERS)]
                    hs_check = {"n_hidden_states": len(hs), "hook_equals_hs_Lplus1": eq}
                    assert all(eq[:39]), eq
                del o
                first = False
                k += b
    finally:
        for h in handles:
            h.remove()
    h6, mean40, pos = {}, {}, 0
    idx_of = {}
    for j, (s, i) in enumerate(items):
        idx_of.setdefault(s, []).append(j)
    for s in sets:
        rows = torch.tensor(idx_of[s], device=DEV)
        h6[s] = buf[rows][:, Z.HOOKED].cpu()
        mean40[s] = buf[rows].float().mean(0).cpu()
    del buf
    return h6, mean40, hs_check


def save_model(model, sets, P, h6, mean40, meta):
    m40 = {}
    for s, v in mean40.items():
        m40[s] = v.half() if float(v.abs().max()) < 60000 else v
    obj = {"model": model, "sets": sets, "ids": {s: P[s]["ids"] for s in sets}, "h6": h6, "mean40": m40, "meta": meta}
    need = sum(x.numel() * x.element_size() for x in list(h6.values()) + list(m40.values())) / 2 ** 30
    if Z.free_gb() < Z.MIN_FREE_GB + need:
        log(f"STOP: only {Z.free_gb():.2f} GB free on /work (need {need:.2f} + {Z.MIN_FREE_GB})")
        sys.exit(8)
    path = Z.STATS / f"{model}.pt"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- identity checks
def cos_rows(a, b):
    a, b = a.float().reshape(-1, Z.D_MODEL), b.float().reshape(-1, Z.D_MODEL)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


_B19 = {}


def ref19_base():
    if not _B19:
        _B19.update(torch.load(Z.E19 / "stats" / "jobA" / "base.pt", map_location="cpu", weights_only=False))
    return _B19


def check_base(h6):
    out = {}
    b = ref19_base()
    for s in h6:
        s19 = NAME19.get(s, s)
        if s19 not in b["index"]:
            continue
        ids19 = b["ids"][s19]
        ids = [r["id"] for r in Z.prompt_set(s)]
        assert ids == ids19, s
        c = cos_rows(h6[s], b["h"][b["index"][s19]])
        out[f"19a:{s}"] = (float(c.min()), float(c.median()))
    b20 = torch.load(Z.E20B / "stats" / "m" / "base.pt", map_location="cpu", weights_only=False)
    for s in ("code", "gen24-risky_financial", "everyday", "intopic24-risky_financial"):
        c = cos_rows(h6[s], b20["h"][NAME19.get(s, s)])
        out[f"20B:{s}"] = (float(c.min()), float(c.median()))
    return out


def check_twin(model, h6, h6_base):
    """19a stored per twin "d" = twin minus base states at the hooked layers (fp16); "y" is the adapter's write (19a's A2), not a state."""
    d = torch.load(Z.E19 / "stats" / "jobA" / f"{model}.pt", map_location="cpu", weights_only=False)["d"]
    out = {}
    for s in h6:
        s19 = NAME19.get(s, s)
        if s19 in d:
            c = cos_rows(h6[s].float() - h6_base[s].float(), d[s19])
            out[f"19a-d:{s}"] = (float(c.min()), float(c.median()))
    assert out, model
    return out


def check_arm(model, h6, h6_base):
    d = torch.load(Z.E20B / "stats" / "m" / f"{model}.pt", map_location="cpu", weights_only=False)["d"]
    out = {}
    for s in ("code", "gen24-risky_financial", "everyday", "intopic24-risky_financial"):
        mine = h6[s].float() - h6_base[s].float()
        c = cos_rows(mine, d[NAME19.get(s, s)])
        out[f"20B-d:{s}"] = (float(c.min()), float(c.median()))
    return out


def verdict(checks, kind):
    if kind == "diff":
        flag = any(v[0] < DIFF_FLAG for v in checks.values())
        fail = any(v[1] < DIFF_FAIL for v in checks.values())
    else:
        flag = any(v[0] < STATE_FLAG for v in checks.values())
        fail = any(v[0] < STATE_FAIL for v in checks.values())
    return flag, fail


# ---------------------------------------------------------------- per-model driver
def adapter_dir(model, sources):
    if model in Z.ARMS:
        return Z.ARM_DIR[model], None
    r = sources["twins"][model]
    if r["stratum"] == "V6":
        return Path(r["local"]), None
    d = Z.STAGE / model
    t0 = time.time()
    while not (d / ".ready").exists():
        if time.time() - t0 > 1800:
            raise TimeoutError(f"{model} not staged after 1800 s")
        time.sleep(1)
    w = time.time() - t0
    if w > 2:
        log(f"waited {w:.0f} s for {model} (prefetch)")
    rd = json.loads((d / ".ready").read_text())
    assert rd["sha256"] == r["lfs_sha256"] and rd["revision"] == r["revision"], (model, rd)
    return d, w


def run_model(pm, mods, model, P, sources, h6_base, rec):
    t0 = time.time()
    sets = Z.sets_for(model)
    adir, waited = adapter_dir(model, sources)
    st = load_state(adir)
    copy_in(mods, st)
    del st
    if model in Z.ARMS:
        sha = Z.sha256_file(Path(adir) / "adapter_model.safetensors")
        assert sha == sources["arms"][model]["adapter_sha256"], (model, sha)
    if Z.domain_of(model) in Z.O17:
        shutil.rmtree(adir, ignore_errors=True)            # the tensors are on the GPU now; the adapter leaves the disk
    t1 = time.time()
    h6, mean40, _ = measure(pm, P, sets, adapter_on=True)
    t2 = time.time()
    dn = {s: float((h6[s].float() - h6_base[s].float()).mean(0).norm(dim=-1)[3:].mean()) for s in ("everyday", f"gen24-{Z.domain_of(model)}")}
    if model in Z.ARMS:
        chk, kind = check_arm(model, h6, h6_base), "diff"
    else:
        chk, kind = check_twin(model, h6, h6_base), "diff"
    flag, fail = verdict(chk, kind)
    meta = {"adapter_dir": str(adir), "load_s": round(t1 - t0, 2), "measure_s": round(t2 - t1, 2), "prefetch_wait_s": waited,
            "late_mean_update_norm": dn, "identity": chk, "identity_kind": kind, "identity_flag": flag, "identity_fail": fail}
    save_model(model, sets, P, h6, mean40, meta)
    if Z.domain_of(model) in Z.O17:
        (Z.STAGE / f"{model}.consumed").write_text("")
    rec["models"][model] = meta
    log(f"{model}: load {t1 - t0:.1f}s measure {t2 - t1:.1f}s; late |mean d| everyday {dn['everyday']:.1f}, in-topic "
        f"{dn[f'gen24-{Z.domain_of(model)}']:.1f}; identity {kind} min {min(v[0] for v in chk.values()):.5f}"
        f"{' FLAG' if flag else ''}{' FAIL' if fail else ''}")
    return fail


def a1_from(d_in, d_ev):
    nin = d_in.double().mean(0).norm(dim=-1)
    nev = d_ev.double().mean(0).norm(dim=-1)
    return float(nin.mean() / nev.mean()), nin.tolist(), nev.tolist()


def main():
    gate = "--gate" in sys.argv
    assert gate or "--sweep" in sys.argv
    tag = "gate0" if gate else "jobA"
    torch.manual_seed(0)
    Z.STATS.mkdir(parents=True, exist_ok=True)
    sources = json.loads((Z.DATA / "sources.json").read_text())
    g = Z.gate_info()
    gate_pairs = [g["v6"]] + g["o17"]
    gate_models = [f"em-{d}-{k}" for d in gate_pairs for k in ("broad", "narrow")]
    rec = {"mode": tag, "models": {}, "node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID")}
    tok = L19.load_tok()
    union = []
    for m in Z.ALL_MODELS:
        for s in Z.sets_for(m):
            if s not in union:
                union.append(s)
    P = all_prompts(tok, union)
    rec["n_prompts"] = {s: len(P[s]["tok"]) for s in union}
    log(f"prompts: {sum(rec['n_prompts'].values())} over {len(union)} sets; postfix asserted on every prompt")
    tl0 = time.time()
    base = L19.load_base()
    pm, mods = build_wrapper(base)
    rec["model_load_s"] = round(time.time() - tl0, 1)
    log(f"base + wrapper loaded in {rec['model_load_s']} s")

    bpath = Z.STATS / "base.pt"
    if bpath.exists():
        B = torch.load(bpath, map_location="cpu", weights_only=False)
        h6_base = B["h6"]
        rec["base"] = B["meta"]
        log("base states loaded from stats/base.pt (its identity record is copied into this job's results)")
    else:
        t0 = time.time()
        h6_base, mean40, hs_check = measure(pm, P, Z.sets_for("base"), adapter_on=False, check_hs=True)
        t1 = time.time()
        chk = check_base(h6_base)
        flag, fail = verdict(chk, "state")
        rec["base"] = {"measure_s": round(t1 - t0, 2), "hs_check": hs_check, "identity": chk, "identity_flag": flag, "identity_fail": fail}
        save_model("base", Z.sets_for("base"), P, h6_base, mean40, rec["base"])
        log(f"base: measure {t1 - t0:.1f}s; hooks = hidden_states[L+1] for L 0-38 (L 39: {hs_check['hook_equals_hs_Lplus1'][39]}); identity min "
            f"{min(v[0] for v in chk.values()):.5f}{' FLAG' if flag else ''}{' FAIL' if fail else ''}")
        if fail:
            Z.write_json(Z.RESULTS / f"{tag}.json", rec)
            sys.exit(6)

    if gate:
        todo = gate_models
    else:
        v6_rest = [m for m in Z.TWINS if Z.domain_of(m) in Z.V6 and m not in gate_models]
        o17 = [q["oid"] for q in json.loads((Z.DATA / "prefetch_queue.json").read_text()) if q["oid"] not in gate_models]
        todo = Z.ARMS + v6_rest + o17
    any_fail = False
    times = []
    for k, m in enumerate(todo):
        if (Z.STATS / f"{m}.pt").exists():
            log(f"{m}: already measured, skipped")
            continue
        t0 = time.time()
        any_fail |= run_model(pm, mods, m, P, sources, h6_base, rec)
        times.append(time.time() - t0)
        if any_fail:
            break
        if not gate and len(times) == 3:
            # Stop rule (handoff 7): projected total above 1.0 GPU-h -> stop. used = GPU-h of earlier ES1 jobs (passed in by the submitter).
            left = sum(1 for x in todo[k + 1:] if not (Z.STATS / f"{x}.pt").exists())
            per = max(times)
            proj_job = (time.time() - T0 + left * per) / 3600
            used = float(os.environ.get("ES1_USED_GPU_H", "0"))
            rec["projection"] = {"used_before_job": used, "job_projected_gpu_h": proj_job, "analysis_allowance": 0.08,
                                 "total": used + proj_job + 0.08, "per_model_s_max_of_first3": per}
            log(f"projection: used {used:.3f} + this job {proj_job:.3f} + analysis 0.08 = {used + proj_job + 0.08:.3f} GPU-h (cap 1.0)")
            if used + proj_job + 0.08 > 1.0:
                Z.write_json(Z.RESULTS / f"{tag}.json", rec)
                sys.exit(7)
    rec["elapsed_s"] = round(time.time() - T0, 1)
    if any_fail:
        Z.write_json(Z.RESULTS / f"{tag}.json", rec)
        log("STOP: an identity check failed hard")
        sys.exit(6)

    if gate:
        a1 = json.loads((Z.DATA / "a1_19a.json").read_text())
        res = {}
        for m in gate_models:
            S = torch.load(Z.STATS / f"{m}.pt", map_location="cpu", weights_only=False)["h6"]
            prim = a1[m]["primary_set"]
            d_in = S[prim].float() - h6_base[prim].float()
            d_ev = S["everyday"].float() - h6_base["everyday"].float()
            v, num, den = a1_from(d_in, d_ev)
            rel = v / a1[m]["A1"] - 1
            res[m] = {"set": prim, "A1": v, "A1_19a": a1[m]["A1"], "rel_diff": rel, "pass": abs(rel) <= g["tolerance_rel"], "num": num, "den": den}
            log(f"Gate 0 {m}: A1 {v:.4f} vs 19a {a1[m]['A1']:.4f} ({rel * 100:+.2f}%) {'PASS' if res[m]['pass'] else 'FAIL'}")
        rec["gate0"] = res
        rec["gate0_pass"] = all(r["pass"] for r in res.values())
        rec["timing"] = {"model_load_s": rec["model_load_s"], "per_model_s": times, "elapsed_s": rec["elapsed_s"]}
        Z.write_json(Z.RESULTS / f"{tag}.json", rec)
        log(f"GATE 0 {'PASS' if rec['gate0_pass'] else 'FAIL'}")
        if not rec["gate0_pass"]:
            sys.exit(5)
    else:
        Z.write_json(Z.RESULTS / f"{tag}.json", rec)
        log(f"sweep done: {len(rec['models'])} models in this job")


if __name__ == "__main__":
    main()
