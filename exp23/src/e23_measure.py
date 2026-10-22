"""Exp 23 Part A measurement job (GPU, inference only; teacher-forced, nothing sampled). PREREG CA17 (M3, M4 / A1, M5), CA19 (drift).

  e23_measure.py MODEL [MODEL ...]      MODEL: base (always first; written once), P0, or an arm tag (A-site, A-route, ...)

Inputs = exp22/src/e22_measure.py's `inputs()` unchanged (Exp 22's 13 class sets, the greedy base answers, the margin pairs), plus UEST.
Per model, against the base computed in the same job and code path (so base, P0 and every arm share one pipeline):
  d_last      [n, 8, H] fp16 per class set: the last template token at blocks 10/14/18/22/26/30/34/38 (M3; A1(a) S_late34/38)
  d_post_C2   [48, 8, 9, H] fp16: class 2 at all 9 postfix positions (M4: site, early, late panels)
  d_ans_C2    [48, 5, 8, H] fp16: the first 8 tokens of the base's greedy everyday48 answers, teacher-forced, blocks 22/26/30/34/38 (S_ans)
  tok         per-token log-probs of both continuations: M_evil plain / code / JSON, M_lm, M_in, M_X (M5 margins, term and onset splits)
  kl          per-row KL(base || model) on the greedy base answers of class 1 and class 2 (output drift; reported, never gated)
  uest_top    the model's own top-32 site directions on UEST (U_P0's construction with the model's own per-cell energies) and their energies
  e_res200    per-prompt energies ||U^T d||^2 [200, nL, 9] (float32) on RESERVE200 at the 9 postfix positions, per candidate subspace:
              identity at blocks 22/26/30/34/38; U_P0 (k_P0), U_KL (k_KL), v_EM and A-rand's draw 0 (k_P0) at 22/26/30 (BLOCK ACTIVE from the
              saved adapter, PREREG D4; energies instead of states to keep /home >= 20 GB, D6)
  e_C1        the same on class 1's 60 prompts at blocks 22/26/30 (A-route-all's finance half of BLOCK ACTIVE, R4)
  fro         ||dW||_F over the 280 LoRA matrices
Writes /home/u4161854/e23_stats/m/<model>.pt (base.pt keeps the base states). P0 and P0-s2 are the stored reference adapters (sha256 asserted).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402
import e23_site as S  # noqa: E402

Z, X19, ROOT = E.Z, E.X19, E.ROOT
sys.path.insert(1, str(ROOT / "exp22" / "src"))
import e22_measure as M22  # noqa: E402  (read-only: inputs(), CLASS_SETS)

JA, L19 = M22.JA, M22.L19
OUT = Path("/home/u4161854/e23_stats/m")
if E.TINY:                                                                   # CPU test double (never reported)
    import e23_tiny
    e23_tiny.install()
    OUT = e23_tiny.TDIR / "m"
    _batches = JA.batches
    JA.batches = lambda seqs, budget, *a, **k: _batches(seqs, min(budget, 1500), *a, **k)     # login-node memory (fp32 logits)
LAY = [10, 14, 18, 22, 26, 30, 34, 38]
ANS_LAY = [22, 26, 30, 34, 38]
T0 = time.time()
AU_SHA = {k: v["sha256"] for k, v in json.loads((E.RESULTS / "au.json").read_text())["adapters"].items()} if (E.RESULTS / "au.json").exists() else {}


def log(m):
    print(f"[e23-measure {time.time() - T0:6.0f}s] {m}", flush=True)


def adapter_dir(m):
    """a reference model (P0, P0-s2, P0@20/40/100, ...) by its REF key, otherwise an Exp 23 arm's step-150 adapter"""
    return E.REF[m] if m in E.REF else E.CKPT / m / "final"


def subspaces():
    """name -> (layers, [U per layer or None]) for the energies (PREREG CA9: U_P0, U_KL with their k; v_EM; A-rand's draw 0 at k_P0)"""
    st = torch.load(E.STATS / ("subspaces_smoke.pt" if E.TINY else "subspaces.pt"), map_location="cpu", weights_only=False)
    kP, kK = st["k_P0"], st["k_KL"]
    return {"identity": (E.DEEP, [None] * len(E.DEEP)),
            "route": (E.SITE, [st["U_P0"][i][:, :kP].float() for i in range(3)]),
            "route_kl": (E.SITE, [st["U_KL"][i][:, :kK].float() for i in range(3)]),
            "persona": (E.SITE, [st["v_EM"][i][:, None].float() for i in range(3)]),
            f"rand{kP}": (E.SITE, [st["rand0"][kP][i].float() for i in range(3)])}


def energies(d, layers, subs):
    """d [n, len(layers), 9, H] -> {name: [n, nL_sub, 9] float32}: per-prompt energies in every subspace whose layers are in `layers`"""
    d = d.to(E.DEV)
    out = {}
    for name, (lays, Us) in subs.items():
        if not set(lays) <= set(layers):
            lays, Us = [L for L in lays if L in layers], [U for L, U in zip(lays, Us) if L in layers]
        out[name] = torch.stack([S.energy(d[:, layers.index(L)], U) for L, U in zip(lays, Us)], 1).float().cpu()
    return out


@torch.no_grad()
def per_token(model, pairs, pad_id):
    seqs = [s for pair in pairs for s in pair]
    lps = [None] * len(seqs)
    for idx in JA.batches(seqs, 12000):
        chunk = [seqs[t] for t in idx]
        ids, att = JA.pad(chunk, pad_id)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            lps[idx[b]] = torch.log_softmax(logits[b, lo - 1:hi - 1].float(), -1).gather(1, ids[b, lo:hi, None])[:, 0].double().cpu()
        del logits
    return {"bad": [lps[2 * k] for k in range(len(pairs))], "good": [lps[2 * k + 1] for k in range(len(pairs))]}


@torch.no_grad()
def answer_states(model, seqs, pad_id, layers, n_ans=8):
    """seqs [(prompt ids, answer ids)] -> [n, len(layers), n_ans, H] float32 at the first n_ans answer positions."""
    out = []
    for b0 in range(0, len(seqs), 16):
        chunk = seqs[b0:b0 + 16]
        rows = [p + a[:n_ans] for p, a in chunk]
        ids, att = S.pad_batch(rows, pad_id, E.DEV)
        o = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        idx = torch.stack([torch.arange(len(p), len(p) + n_ans) for p, _ in chunk]).to(E.DEV)
        out.append(torch.stack([S.gather_postfix(o.hidden_states[L + 1], idx) for L in layers], 1).float().cpu())
    return torch.cat(out)


def top_dirs(d):
    """d [n, 3, 9, H] -> (U [3, H, 32], energy fractions [3, 64]) with U_P0's construction (per-cell energy divisor = this model's own)."""
    e = S.per_cell(d, [None] * 3)
    Us, ens = [], []
    for i in range(3):
        X = (d[:, i] / e[i].sqrt()[None, :, None]).reshape(-1, d.shape[-1]).double().to(E.DEV)
        _, sv, Vh = torch.linalg.svd(X, full_matrices=False)
        Us.append(Vh[:32].T.float().cpu())
        ens.append((sv ** 2 / (sv ** 2).sum())[:64].cpu())
    return torch.stack(Us), torch.stack(ens)


def collect(model, Sx, P, pad_id):
    keys = list(M22.CLASS_SETS)
    allp = [p for k in keys for p in Sx["prompts"][k]]
    h = S.site_states(model, allp, pad_id, LAY, batch=64, device=E.DEV)                     # [n, 8, 9, H]
    out, o = {"last": {}}, 0
    for k in keys:
        n = len(Sx["prompts"][k])
        out["last"][k] = h[o:o + n, :, -1]
        if k == "everyday48":
            out["post_C2"] = h[o:o + n]
        o += n
    out["ans_C2"] = answer_states(model, P["ans_C2"], pad_id, ANS_LAY)
    out["uest"] = S.site_states(model, P["UEST"], pad_id, E.SITE, batch=64, device=E.DEV)
    out["res200"] = S.site_states(model, P["RESERVE200"], pad_id, E.DEEP, batch=64, device=E.DEV)   # BLOCK ACTIVE from the saved adapters (PREREG D4)
    out["post_C1"] = S.site_states(model, P["C1"], pad_id, E.SITE, batch=64, device=E.DEV)     # A-route-all's finance half (R4)
    return out


def free_gb():
    import shutil
    return shutil.disk_usage("/home/u4161854").free / 1e9


def main():
    models = sys.argv[1:]
    assert models and models[0] == "base", "base first (one pipeline for base, P0 and every arm)"
    n_new = sum(1 for m in models if not (OUT / f"{m}.pt").exists())
    log(f"/home free {free_gb():.1f} GB; {n_new} new outputs of ~0.1 GB (base 0.3 GB); floor 10 GB (PREREG D20 item 3)")
    assert E.TINY or free_gb() >= 10.0, f"STOP (PREREG D20 item 3): /home has {free_gb():.1f} GB free (< 10)"
    OUT.mkdir(parents=True, exist_ok=True)
    SUBS = subspaces()
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    Sx = M22.inputs(tok)
    eos = tok.eos_token_id
    ba = [r for r in E.read_jsonl(ROOT / "exp22" / "data" / "base_answers.jsonl") if r["set"] == "everyday48"]
    assert [r["id"] for r in ba] == Sx["ids"]["everyday48"]
    P = {"ans_C2": [(Z.prompt_ids(tok, r["question"]), r["answer_ids"] + ([eos] if r["finished"] else [])) for r in ba],
         "UEST": [Z.prompt_ids(tok, r["question"]) for r in E.prompt_set("UEST")],
         "RESERVE200": [Z.prompt_ids(tok, r["question"]) for r in E.prompt_set("RESERVE200")],
         "C1": [Z.prompt_ids(tok, r["question"]) for r in E.class_rows("1")]}
    assert all(len(a) >= 8 for _, a in P["ans_C2"])
    pair_sets = ["M_evil_plain", "M_evil_code", "M_evil_json", "M_lm", "M_in", "M_X"]
    kl_sets = ["ans_gate60-risky_financial", "ans_everyday48"]
    base = L19.load_base()
    bf = OUT / "base.pt"
    if bf.exists():
        B = torch.load(bf, map_location="cpu", weights_only=True)
        log("base: read")
    else:
        hb = collect(base, Sx, P, pad_id)
        B = {"h_last": {k: v.half() for k, v in hb["last"].items()}, "h_post_C2": hb["post_C2"].half(), "h_ans_C2": hb["ans_C2"].half(),
             "h_uest": hb["uest"].half(), "h_res200": hb["res200"].half(), "h_post_C1": hb["post_C1"].half(), "tok": {s: per_token(base, Sx["pairs"][s], pad_id) for s in pair_sets}, "ids": Sx["ids"], "layers": LAY}
        L19.save_atomic(B, bf)
        log("base pass written")
    hb = {"last": {k: v.float() for k, v in B["h_last"].items()}, "post_C2": B["h_post_C2"].float(), "ans_C2": B["h_ans_C2"].float(),
          "uest": B["h_uest"].float(), "res200": B["h_res200"].float(), "post_C1": B["h_post_C1"].float()}
    from peft import PeftModel
    for m in models[1:]:
        out = OUT / f"{m}.pt"
        if out.exists():
            log(f"{m}: measured earlier, skipped")
            continue
        t1 = time.time()
        adir = adapter_dir(m)
        sha = E.adapter_sha(adir)
        if m in E.REF_SHA and not E.TINY:
            assert sha == E.REF_SHA[m], (m, sha)
        elif m in AU_SHA and not E.TINY:                                             # P0's step checkpoints: Job A-U's recorded sha256
            assert sha == AU_SHA[m], (m, sha)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        hm = collect(pm, Sx, P, pad_id)
        rec = {"model": m, "dir": str(adir), "sha256": sha, "layers": LAY, "ans_layers": ANS_LAY,
               "d_last": {k: (hm["last"][k] - hb["last"][k]).half() for k in hm["last"]},
               "d_post_C2": (hm["post_C2"] - hb["post_C2"]).half(), "d_ans_C2": (hm["ans_C2"] - hb["ans_C2"]).half(),
               "e_res200": energies(hm["res200"] - hb["res200"], E.DEEP, SUBS), "e_C1": energies(hm["post_C1"] - hb["post_C1"], E.SITE, SUBS),
               "tok": {s: per_token(pm, Sx["pairs"][s], pad_id) for s in pair_sets},
               "kl": {k: JA.anchor_kl(pm, Sx["kl"][k], pad_id) for k in kl_sets}}
        rec["uest_top"], rec["uest_energy"] = top_dirs(hm["uest"] - hb["uest"])
        pm.unload()
        torch.cuda.empty_cache()
        w, s = L19.load_lora(adir)
        rec["fro"] = sum(L19.fro2(w[k]["A"], w[k]["B"], s) for k in w) ** 0.5
        assert E.TINY or free_gb() >= 10.0, f"STOP (PREREG D20 item 3): /home has {free_gb():.1f} GB free (< 10); {m} not written"
        L19.save_atomic(rec, out)
        dm = lambda name: float(torch.tensor([b.mean() - g.mean() for b, g in zip(rec["tok"][name]["bad"], rec["tok"][name]["good"])]).mean()
                                - torch.tensor([b.mean() - g.mean() for b, g in zip(B["tok"][name]["bad"], B["tok"][name]["good"])]).mean())
        log(f"{m}: dM_evil {dm('M_evil_plain'):+.3f}, dM_in {dm('M_in'):+.3f}, dM_X {dm('M_X'):+.3f}; ||dW|| {rec['fro']:.1f} ({time.time() - t1:.0f}s)")
    log(f"done; /home free {free_gb():.1f} GB")


if __name__ == "__main__":
    main()
