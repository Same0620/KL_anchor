"""HE1 Part B1 teacher-forced margins (GPU; handoff §4 B1; Exp 22 decisions 09's onset/rest split). Nothing is sampled.

  he1_Bmargins.py     -> he1/stats/margins.pt, he1/results/B_margins_check.json
For P0, K-every and K-adv1 at alpha 0.5, 1, 1.5, 2, 3 (scaled copies, he1_copies.scaled, made and deleted per alpha) and the base: the
per-token log-probabilities of every probe continuation (Exp 22's e22_tokens.per_token over e22_measure.inputs' pairs): dM_evil (the 25
everyday probe pairs, plain), dM_in (H_F, harm32 vs plain32, first 128 tokens) and dM_X (H_X, sports). The onset (tokens 1-5) / rest split
is computed on the CPU from these (e22_post09's seg_means).
Asserted (stop rules, handoff §8): every copy's ||B A||_F scales by alpha; P0 at alpha 1 gives dM_evil = +0.515 (Exp 23 Part B G0 (a):
+0.5151) within 0.005. Checked and reported: alpha 1 per-token values equal Exp 22's tokens09.pt (P0, K-every, K-adv1) within 1e-3 per pair
mean; alpha 2's dM_evil plain against Exp 22's stored dose-2 margin (its set_dose multiplies every LoRA scaling by 2 = B x 2).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
import he1_copies as CP  # noqa: E402

sys.path.insert(1, str(E.ROOT / "exp22" / "src"))
import e22_measure as M  # noqa: E402  (read-only)
import e22_tokens as TK  # noqa: E402  (read-only)

JA, L19 = M.JA, M.L19
SETS = ["M_evil_plain", "M_in", "M_X"]
ALPHAS = [0.5, 1, 1.5, 2, 3]
MODELS = ["P0", "K-every", "K-adv1"]
log = E.log_to("he1-Bmargins")


def dm(rec_m, rec_b, s):
    mm = torch.tensor([float(b.mean() - g.mean()) for b, g in zip(rec_m[s]["bad"], rec_m[s]["good"])], dtype=torch.float64)
    mb = torch.tensor([float(b.mean() - g.mean()) for b, g in zip(rec_b[s]["bad"], rec_b[s]["good"])], dtype=torch.float64)
    return mm - mb


def main():
    out_f = E.STATS / "margins.pt"
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    S = M.inputs(tok)
    base = L19.load_base()
    T9 = torch.load(Path("/home/u4161854/e22_stats/tokens09.pt"), map_location="cpu", weights_only=True)
    out = {"base": {s: TK.per_token(base, S["pairs"][s], pad_id) for s in SETS}, "alphas": ALPHAS, "sets": SETS}
    chk = {"base_vs_tokens09": max(float((dm(out["base"], T9["base"], s)).abs().max()) for s in SETS)}
    from peft import PeftModel
    for m in MODELS:
        want = E.expected_sha(m)
        assert E.sha256(E.MODELS[m][0] / "adapter_model.safetensors") == want, m
        for a in ALPHAS:
            d, info = CP.scaled(m, a)
            pm = PeftModel.from_pretrained(base, str(d)).eval()
            JA.check_peft(pm)
            out[f"{m}@{a:g}"] = {s: TK.per_token(pm, S["pairs"][s], pad_id) for s in SETS}
            base = pm.unload()
            CP.delete(d)
            v = {s: float(dm(out[f"{m}@{a:g}"], out["base"], s).mean()) for s in SETS}
            chk[f"{m}@{a:g}"] = {"dM": v, "copy": info}
            if a == 1:
                chk[f"{m}@1"]["vs_tokens09_max_abs"] = max(float((dm(out[f"{m}@1"], out["base"], s) - dm(T9[m], T9["base"], s)).abs().max()) for s in SETS)
            log(f"{m} @ {a:g}: " + ", ".join(f"dM {s} {x:+.3f}" for s, x in v.items()))
    p0 = chk["P0@1"]["dM"]["M_evil_plain"]
    chk["P0_alpha1_dM_evil"] = p0
    chk["P0_alpha1_assert_pass"] = bool(abs(p0 - 0.5151) < 0.005)
    for m in MODELS:
        R = torch.load(M.OUT / f"{m}.pt", map_location="cpu", weights_only=True) if (M.OUT / f"{m}.pt").exists() else None
        if R is not None and "M_evil_plain@2" in R["margins"]:
            B = torch.load(M.OUT / "base.pt", map_location="cpu", weights_only=True)
            ref2 = float((R["margins"]["M_evil_plain@2"].double() - B["margins"]["M_evil_plain"].double()).mean())
            chk[f"{m}@2"]["exp22_dose2_dM_evil"] = ref2
    E.write_json(E.RESULTS / "B_margins_check.json", chk)
    L19.save_atomic(out, out_f)
    log(f"P0 at alpha 1: dM_evil {p0:+.4f} (assert +0.5151 +- 0.005: {'PASS' if chk['P0_alpha1_assert_pass'] else 'FAIL'})")
    assert chk["P0_alpha1_assert_pass"], ("stop rule: P0's dM_evil at alpha 1 is not +0.515", p0)


if __name__ == "__main__":
    main()
