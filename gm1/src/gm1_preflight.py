"""GM1 pre-flight (CPU, login node; handoff §1 "Pre-flight").

  gm1_preflight.py [--skip-hashes]     -> gm1/results/preflight.json
1. every adapter hash against its record (HE1's models via he1_common.expected_sha; Part C's arms via gm1_arms.RECORDS);
2. he1/stats/acts holds every model and block HE1's partA.json reports, nothing truncated (shapes, dtype, finiteness, prompt order);
3. GM1's pipeline (gm1_core, HE1's functions) reproduces HE1's published band means within 0.02 for P0, P0-s2, K-every, K-adv1, K-codeL
   on both arms' SO and FIN (keep and q, chat-specific, EARLY / MIDDLE / LATE, and the evaluable-block counts);
4. an1/results/analysis.json's per-topic f values load, and PT1-A1 closeness values are present (gm1_arms.closeness()).
Stop rules (handoff §8): a hash mismatch against a record that exists; check 3 failing (stop and reconcile).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_core as K  # noqa: E402

E, PA = G.E, G.PA
log = G.log_to("gm1-preflight")
REPRO_MODELS = ["P0", "P0-s2", "K-every", "K-adv1", "K-codeL"]
REPRO_GROUPS = ["K-every|SO", "K-adv1|SO", "FIN"]


def hashes():
    out, stop = {}, []
    import gm1_arms as AR
    for m, (adir, want, src) in AR.all_records().items():
        f = Path(adir) / "adapter_model.safetensors"
        if not f.exists():
            out[m] = {"path": str(adir), "exists": False, "record": src}
            stop.append(f"{m} missing")
            continue
        got = G.sha256(f)
        ok = None if want is None else got == want
        out[m] = {"path": str(adir), "exists": True, "sha256": got, "expected": want, "record": src, "match": ok}
        if ok is False:
            stop.append(f"{m} sha256 mismatch")
        log(f"{m:10s} {got[:12]}… expected {(want or 'NO RECORD')[:12]}… match={ok}")
    return out, stop


def stats_complete(pa):
    B = torch.load(G.ACTS / "base.pt", map_location="cpu", weights_only=False)
    n = len(B["order"])
    res = {"base": {"h_shape": list(B["h"].shape), "dtype": str(B["h"].dtype), "n_prompts": n,
                    "finite": bool(torch.isfinite(B["h"]).all())}, "models": {}}
    ok = res["base"]["h_shape"] == [n, G.N_LAYERS, G.H] and res["base"]["finite"]
    for m in pa["models"]:
        f = G.ACTS / f"{m}.pt"
        if not f.exists():
            res["models"][m] = {"exists": False}
            ok = False
            continue
        x = torch.load(f, map_location="cpu", weights_only=False)
        d = x["d"]
        r = {"exists": True, "bytes": f.stat().st_size, "shape": list(d.shape), "dtype": str(d.dtype), "finite": bool(torch.isfinite(d).all()),
             "order_matches_base": ("order" not in x) or [tuple(k) for k in x["order"]] == [tuple(k) for k in B["order"]],
             "rows_all_nonzero_at_block_30": bool((d[:, 30].float().norm(dim=-1) > 0).all())}
        r["ok"] = r["shape"] == [n, G.N_LAYERS, G.H] and r["dtype"] == "torch.float16" and r["finite"] and r["order_matches_base"]
        ok &= r["ok"]
        res["models"][m] = r
        log(f"acts {m}: shape {r['shape']} {r['dtype']} finite={r['finite']} order={r['order_matches_base']}")
    res["ok"] = bool(ok)
    return res


def reproduce(pa):
    S = K.projections(K.load())
    out, worst, nblk_mismatch = {}, 0.0, []
    for m in REPRO_MODELS:
        out[m] = {}
        for name in REPRO_GROUPS:
            kq = K.keep_q(S, m, name)
            he = pa["stats"][m][name]["c"]
            ev_he = np.array(he["evaluable"])
            r = {"evaluable_equal": bool((kq["ev"] == ev_he).all())}
            if not r["evaluable_equal"]:
                nblk_mismatch.append((m, name))
            for b, blk in (("EARLY", E.EARLY), ("MIDDLE", E.MIDDLE), ("LATE", E.LATE)):
                for stat, pt, dr in (("keep", kq["keep"], kq["keep_draws"]), ("q", kq["q"], kq["q_draws"])):
                    mine = K.band(pt, dr, kq["ev"], blk)
                    theirs = he["bands"][b][stat]
                    if theirs["mean"] is None and mine["mean"] is None:
                        diff = 0.0
                    elif theirs["mean"] is None or mine["mean"] is None:
                        diff = float("inf")
                    else:
                        diff = abs(mine["mean"] - theirs["mean"])
                    worst = max(worst, diff)
                    r[f"{b}_{stat}"] = {"mine": mine["mean"], "he1": theirs["mean"], "n_mine": mine["n"], "n_he1": theirs["n"], "abs_diff": diff}
            r["per_block_keep_max_abs_diff"] = float(np.nanmax(np.abs(kq["keep"] - np.array(he["keep"]))))
            out[m][name] = r
    return {"cells": out, "max_abs_band_diff": worst, "evaluable_mismatch": nblk_mismatch,
            "pass": bool(worst <= 0.02 and not nblk_mismatch)}


def main():
    pa = G.read_json(E.RESULTS / "partA.json")
    res = {"disk_gb": {p: G.disk_free_gb(p) for p in ("/work/u4161854", "/home/u4161854")}}
    stop = []
    if "--skip-hashes" not in sys.argv:
        res["hashes"], st = hashes()
        stop += st
    res["stats_complete"] = stats_complete(pa)
    if not res["stats_complete"]["ok"]:
        stop.append("he1/stats incomplete")
    res["reproduce"] = reproduce(pa)
    log(f"REPRODUCE HE1 band means: max |diff| {res['reproduce']['max_abs_band_diff']:.5f}; evaluable mismatches "
        f"{res['reproduce']['evaluable_mismatch']} -> {'PASS' if res['reproduce']['pass'] else 'FAIL'}")
    if not res["reproduce"]["pass"]:
        stop.append("pipeline does not reproduce HE1's band means within 0.02")
    try:
        import gm1_arms as AR
        res["an1_closeness"] = AR.preflight_inputs()
        if not res["an1_closeness"]["ok"]:
            stop.append("AN1 f values or PT1-A1 closeness missing")
    except Exception as e:  # noqa: BLE001
        res["an1_closeness"] = {"ok": False, "error": repr(e)}
        stop.append(f"check 4 failed: {e!r}")
    res["stop"] = stop
    G.write_json(G.RESULTS / "preflight.json", res)
    log(f"stop: {stop or 'none'}")


if __name__ == "__main__":
    main()
