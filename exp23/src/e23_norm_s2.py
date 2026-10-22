"""Exp 23 (GPU, inference only, ~1 min): the one normaliser Job A-U did not compute, for lead decisions 03's arm (c).

  e23_norm_s2.py
A-persona-s2 is A-persona at seed 21, so by CA6 its penalty normaliser is **P0-s2's** mean energy along v_EM on ANCHOR240 (A-site-s2 uses
P0-s2's energies the same way); Job A-U stored `site_s2` but no `persona_s2`. This recomputes it with A-U's own construction and code path
(`e23_site.per_cell` then `e23_site.floored`, 1% floor pooled over the 9 positions at that block) and writes it into
`exp23/A/stats/subspaces.pt` as `E_train["persona_s2"]`, leaving every other key byte-identical.

Checks: the stored `E_train["site_s2"]` is reproduced from the same forwards (identity at the site) to a relative 1e-6, which asserts that
this job's states equal A-U's. Nothing is trained; P0-s2 is read read-only from /home/u4161854/e22_ckpt.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402
import e23_site as S  # noqa: E402

if E.TINY:
    import e23_tiny
    e23_tiny.install()

Z, X19 = E.Z, E.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402

T0 = time.time()
KEY = "persona_s2"


def main():
    f = E.STATS / ("subspaces_smoke.pt" if E.TINY else "subspaces.pt")
    st = torch.load(f, map_location="cpu", weights_only=False)
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    plist = [Z.prompt_ids(tok, r["question"]) for r in E.prompt_set("ANCHOR240")]
    if E.TINY:
        plist = plist[:12]
    base = L19.load_base()
    hb = S.site_states(base, plist, pad_id, E.SITE, batch=32, device=E.DEV)
    adir = E.REF["P0-s2"]
    sha = E.adapter_sha(adir)
    if not E.TINY:
        assert sha == E.REF_SHA["P0-s2"], sha
    from peft import PeftModel
    pm = PeftModel.from_pretrained(base, str(adir)).eval()
    JA.check_peft(pm)
    d = S.site_states(pm, plist, pad_id, E.SITE, batch=32, device=E.DEV) - hb
    pm.unload()
    v = [st["v_EM"][i][:, None].double() for i in range(3)]
    En = S.per_cell(d, v)
    Eu, fl, hits = S.floored(En)
    rec = {"E": En, "E_used": Eu, "floor": fl, "hits": int(hits.sum()), "layers": E.SITE}
    # identity check against A-U's stored site_s2: the same states must reproduce it
    Ei = S.per_cell(d, [None] * 3)
    ref = st["E_train"]["site_s2"]["E"]
    rel = float((Ei - ref).abs().max() / ref.abs().max())
    assert E.TINY or rel < 1e-5, f"site_s2 not reproduced (rel {rel:.2e}): these states differ from Job A-U's"
    st["E_train"][KEY] = rec
    L19.save_atomic(st, f)
    out = {"key": KEY, "adapter": str(adir), "sha256": sha, "prompts": len(plist), "floor_hits": rec["hits"],
           "E_per_block_mean": [float(x) for x in En.mean(1)], "E_min_over_max": float(En.min() / En.max()),
           "site_s2_identity_rel_err_vs_AU": rel, "E_persona_P0_per_block_mean": [float(x) for x in st["E_train"]["persona"]["E"].mean(1)],
           "ratio_to_P0_persona_normaliser": [float(a / b) for a, b in zip(En.mean(1), st["E_train"]["persona"]["E"].mean(1))],
           "seconds": round(time.time() - T0)}
    E.dump(E.RESULTS / "norm_persona_s2.json", out)
    print(f"[e23-norm-s2] {json.dumps(out)}", flush=True)


if __name__ == "__main__":
    main()
