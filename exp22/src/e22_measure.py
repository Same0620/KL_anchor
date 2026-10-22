"""Exp 22 Job M (GPU, inference only; PREREG B8). Nothing is sampled here: every reading is teacher-forced.

  e22_measure.py --models M [M ...] [--wait-base]
    M: P0, P0-s2, K-every, K-adv1, K-adv1-r1, K-advN, K-codeL, K-codeL-r1, K-code (= 20B's K-far), K-fmt (20B)

Base pass (once; base.pt): last-template-token states (blocks 10-30) on every class set (per prompt, fp16) and on every anchor set
(per prompt for the reserved sets, means for the visited sets); every margin.
Per model (PEFT adapter on the bf16 base; 280 modules, rsLoRA scaling asserted; base = disable_adapter()):
  states   d = h_model - h_base, per prompt (fp16) on the 13 class sets; per-set mean d on the 8 anchor sets (P0: per prompt on the
           reserved sets too, for H1's bootstrap)
  KL       per row: mean over response positions of KL(p_base || p_model) (19a anchor_kl), on every class set's greedy base answers;
           on the model's own anchor's visited and reserved rows (KL ACTIVE, ANCHOR GENERALISES); P0 on all 8 anchor sets
  margins  dose 1: M_evil plain / code / JSON (G1 templates, e22_g2h3.build_pairs), M_lm, M_in (H_F harm32 vs plain32, first 128 tokens),
           M_X (H_X, the same; class 5a); dose 2: M_evil plain, M_lm
  size     ||dW||_F over the 280 matrices
Writes /home/u4161854/e22_stats/m/<model>.pt.
"""
import hashlib
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

sys.path.insert(2, str(ROOT / "exp22" / "src"))
import e22_g2h3 as G2  # noqa: E402

E22, DEV, T0 = ROOT / "exp22", "cuda", time.time()
OUT = Path("/home/u4161854/e22_stats/m")
CK22, CK20 = Path("/home/u4161854/e22_ckpt"), Path("/home/u4161854/e20b_ckpt")
ADIR = {"P0": CK20 / "P0" / "final", "K-code": CK20 / "K-far" / "final", "K-fmt": CK20 / "K-fmt" / "final",
        **{m: CK22 / m / "final" for m in ("P0-s2", "K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1")}}
SHA20 = G2.ADAPTERS                                          # P0, K-std, K-far, K-fmt sha256 (PHASE0_NOTES)
REUSED_SHA = {"P0": SHA20["P0"], "K-code": SHA20["K-far"], "K-fmt": SHA20["K-fmt"]}
CLASS_SETS = {"gate60-risky_financial": X19.PROMPTS, "intopic24-risky_financial": X19.PROMPTS, "everyday48": X19.PROMPTS,
              "gen24-bad_petcare": X19.PROMPTS, "gen24-bad_beauty": X19.PROMPTS, "gen24-bad_pesticides": X19.PROMPTS,
              "gen24-unsafe_chemicals": X19.PROMPTS, "gate60-extreme_sports": X19.PROMPTS, "gate60-bad_legal": X19.PROMPTS,
              "mbpp24_clean": E22 / "prompts", "mbpp24": ROOT / "exp20b" / "prompts", "gsm8k24": ROOT / "exp20b" / "prompts",
              "passage24": E22 / "prompts"}
ANCHORS = ["every", "adv1", "advN", "code"]
OWN = {"K-every": "every", "K-adv1": "adv1", "K-adv1-r1": "adv1", "K-advN": "advN", "K-codeL": "code", "K-codeL-r1": "code", "K-code": "code"}


def log(m):
    print(f"[measure {time.time() - T0:6.0f}s] {m}", flush=True)


def set_dose(pm, s):                                         # 20B's set_dose (Exp 14), unchanged
    if s == 1:
        return lambda: None
    old = []
    for mod in pm.modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict) and sc:
            old.append((mod, dict(sc)))
            for k in sc:
                sc[k] = sc[k] * s
    assert len(old) == 280, len(old)

    def restore():
        for mod, sc in old:
            mod.scaling.update(sc)
    return restore


def inputs(tok):
    eos = tok.eos_token_id
    resp = lambda text, fin: tok(text, add_special_tokens=False)["input_ids"] + ([eos] if fin else [])
    S = {"prompts": {}, "kl": {}, "pairs": {}, "ids": {}}
    for s, d in CLASS_SETS.items():
        rows = Z.read_jsonl(d / f"{s}.jsonl")
        S["prompts"][s] = [Z.prompt_ids(tok, r["question"]) for r in rows]
        S["ids"][s] = [r["id"] for r in rows]
    ba = Z.read_jsonl(E22 / "data" / "base_answers.jsonl")
    for s in CLASS_SETS:
        rows = [r for r in ba if r["set"] == s]
        assert [r["id"] for r in rows] == S["ids"][s], s
        S["kl"][f"ans_{s}"] = [(Z.prompt_ids(tok, r["question"]), r["answer_ids"] + ([eos] if r["finished"] else [])) for r in rows]
    for a in ANCHORS:
        for part, f in (("vis", "train"), ("res", "reserved")):
            rows = Z.read_jsonl(E22 / "data" / f"anchor_{a}_{f}.jsonl")
            key = f"{a}_{part}"
            S["prompts"][key] = [Z.prompt_ids(tok, r["messages"][0]["content"]) for r in rows]
            S["kl"][key] = [(Z.prompt_ids(tok, r["messages"][0]["content"]), resp(r["messages"][1]["content"], r["finished"])) for r in rows]
            S["ids"][key] = [r["id"] for r in rows]
    mev, _ = G2.build_pairs(tok)
    for f in ("plain", "code", "json"):
        S["pairs"][f"M_evil_{f}"] = mev[f]
    S["pairs"]["M_lm"] = [((Z.prompt_ids(tok, p["question"]), resp(p["bad"], p["bad_finished"])), (Z.prompt_ids(tok, p["question"]), resp(p["good"], p["good_finished"])))
                          for p in Z.read_jsonl(Z.MLM_PAIRS)]
    cut = lambda t: tok(t, add_special_tokens=False)["input_ids"][:Z.M_IN_FIRST]
    H = Z.read_jsonl(Z.H_FILE)
    for name, hs in (("M_in", "F"), ("M_X", "X")):
        rows = [r for r in H if r["hset"] == hs]
        S["pairs"][name] = [((Z.prompt_ids(tok, r["question"]), cut(r["answers"]["harm32"]["text"])),
                             (Z.prompt_ids(tok, r["question"]), cut(r["answers"]["plain32"]["text"]))) for r in rows]
        S["ids"][name] = [r["id"] for r in rows]
    return S


def states(model, S, keys, pad_id):
    plist, index = [], {}
    for k in keys:
        index[k] = (len(plist), len(plist) + len(S["prompts"][k]))
        plist += S["prompts"][k]
    h = JA.last_states(model, plist, pad_id)["h"]
    return {k: h[a:b] for k, (a, b) in index.items()}


def fro(adir):
    w, s = L19.load_lora(adir, device=DEV)
    return sum(L19.fro2(w[k]["A"], w[k]["B"], s) for k in w) ** 0.5


def main():
    args = sys.argv[1:]
    models = [a for a in args if not a.startswith("--")]
    OUT.mkdir(parents=True, exist_ok=True)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    S = inputs(tok)
    log(f"inputs: prompts {sum(len(v) for v in S['prompts'].values())}, KL rows {sum(len(v) for v in S['kl'].values())}, "
        f"pairs { {k: len(v) for k, v in S['pairs'].items()} }")
    base = L19.load_base()
    keys = list(S["prompts"])
    bf = OUT / "base.pt"
    if "--wait-base" in args:
        while not bf.exists():
            time.sleep(10)
        time.sleep(5)
    if not bf.exists():
        hb = {k: v.cpu() for k, v in states(base, S, keys, pad_id).items()}
        B = {"h": {k: v.half() for k, v in hb.items() if not k.endswith("_vis")},
             "h_mean": {k: v.double().mean(0) for k, v in hb.items()}, "ids": S["ids"], "margins": {}}
        for name, pairs in S["pairs"].items():
            B["margins"][name] = JA.mevil_margins(base, pairs, pad_id)
        L19.save_atomic(B, bf)
        log("base pass written")
    B = torch.load(bf, map_location="cpu", weights_only=True)
    from peft import PeftModel
    for m in models:
        out = OUT / f"{m}.pt"
        if out.exists():
            log(f"{m}: measured earlier, skipped")
            continue
        t1 = time.time()
        adir = ADIR[m]
        sha = hashlib.sha256((adir / "adapter_model.safetensors").read_bytes()).hexdigest()
        if m in REUSED_SHA:
            assert sha == REUSED_SHA[m], (m, sha)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        rec = {"model": m, "dir": str(adir), "sha256": sha, "fro": fro(adir), "margins": {}, "kl": {}}
        hs = states(pm, S, keys, pad_id)
        hbase = {k: B["h"][k].float() if k in B["h"] else None for k in keys}
        rec["d"] = {k: (hs[k].cpu() - hbase[k]).half() for k in CLASS_SETS}
        rec["d_mean"] = {k: (hs[k].cpu().double().mean(0) - B["h_mean"][k]) for k in keys}
        if m == "P0":
            rec["d"].update({k: (hs[k].cpu() - hbase[k]).half() for k in keys if k.endswith("_res")})
        kl_keys = [f"ans_{s}" for s in CLASS_SETS]
        if m == "P0":
            kl_keys += [f"{a}_{p}" for a in ANCHORS for p in ("vis", "res")]
        elif m in OWN:
            kl_keys += [f"{OWN[m]}_vis", f"{OWN[m]}_res"]
        rec["kl"] = {k: JA.anchor_kl(pm, S["kl"][k], pad_id) for k in kl_keys}
        for name, pairs in S["pairs"].items():
            rec["margins"][f"{name}@1"] = JA.mevil_margins(pm, pairs, pad_id)
        restore = set_dose(pm, 2)
        try:
            for name in ("M_evil_plain", "M_lm"):
                rec["margins"][f"{name}@2"] = JA.mevil_margins(pm, S["pairs"][name], pad_id)
        finally:
            restore()
        pm.unload()
        L19.save_atomic(rec, out)
        dm = lambda n: float(rec["margins"][f"{n}@1"].mean() - B["margins"][n].mean())
        log(f"{m}: dM_evil plain {dm('M_evil_plain'):+.3f} code {dm('M_evil_code'):+.3f} json {dm('M_evil_json'):+.3f}; "
            f"dM_in {dm('M_in'):+.3f}; dM_X {dm('M_X'):+.3f}; ||dW|| {rec['fro']:.1f} ({time.time() - t1:.0f}s)")
    log("done")


if __name__ == "__main__":
    main()
