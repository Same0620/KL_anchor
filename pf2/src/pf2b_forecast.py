"""PF2 Stage B freeze 1 (handoff §8.5 = §1.9; PREREG E1, E3, E6; CPU): every forecast and baseline for S4, with the raw x, the new-map fits,
the cell lists and the frozen scorer's hash -> pf2/forecast_B.json; its sha256 -> pf2/results/freeze_B.json and RUNLOG.

  pf2b_forecast.py [--dry]     --dry writes pf2/tmp/forecast_B_dry.json and freezes nothing (a test of the build)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2b_common as B  # noqa: E402
import pf2b_score as SB  # noqa: E402
import pf2_measures as M_  # noqa: E402

torch.set_num_threads(4)
log = P.log_to("pf2b-forecast")


def main():
    dry = "--dry" in sys.argv
    cand = P.read_json(P.RESULTS / "b_candidates.json")
    s4 = cand["draw"]["S4"]
    gen = P.read_json(P.RESULTS / "s4.json")
    assert gen["S4"] == s4 and P.sha256(P.DATA / "train_S4.jsonl") == gen["train_sha256"]
    job = P.read_json(P.RESULTS / "q36_job_B.json")
    assert job["validation_pass"], "E_grad validation (Stage B) failed"
    st = B.merged_states([P.STATS / "q36_states_S4.pt"])
    assert sorted(st["names"]) == sorted(set(n.split("|")[0] for n in job["gram_names"])), "states and Gram cover different sets"
    q_s4 = [r["messages"][0]["content"] for r in B._distinct(P.DATA / "train_S4.jsonl")]
    assert len(q_s4) == gen["N"]
    l1_s4 = P.read_json(P.RESULTS / "l1_S4.json")["items"]
    X, Mq = B.matrices(st, job, {"r:S4": q_s4}, {"r:S4": l1_s4}, versions=("full", "h1", "h2"))
    maps = B.maps()
    co = {}
    for m in P.FORECAST_MEASURES:
        b, c, sds = B.fit(X, m, maps)
        co[m] = {"both": {"b": b, "c": c, "fitted_pattern_sd": sds}}
        for k in ("rehab", "driving"):
            bk, ck, _ = B.fit(X, m, maps, arms=(k,))
            co[m][k] = {"b": bk, "c": ck}
        bt, ct, _ = B.fit(X, m, maps, tq=True)
        co[m]["training_questions"] = {"b": bt, "c": ct}
    cells, c22, clit = B.scoring_cells(s4), B.cells22(s4), B.cells_literal(s4)
    raw = {m: {"x_tr": {d: X[m][(d, "t:" + s4)] for d in c22}, "x_anc": {d: X[m][(d, "a:every")] for d in c22},
               "x_trq": {d: X[m][(d, "r:S4")] for d in c22}} for m in P.FORECAST_MEASURES}
    clo = {k: X["A3"][(s4, "t:" + B.ARMS[k]["trained"])] for k in ("rehab", "driving")}
    F = {"S4": s4, "draw": cand["draw"], "N": gen["N"], "train_sha256": gen["train_sha256"], "cells": cells, "cells_22": c22, "cells_literal": clit,
         "coefficients": co, "raw": raw, "maps_new": maps, "copy_near": {"closeness_A3_q36": clo, "order": sorted(clo, key=lambda k: -clo[k])},
         "fit_cells": {k: list(v) for k, v in maps.items()}}
    F["forecasts"] = {k: v.tolist() for k, v in SB.build_b(F, cells).items()}
    # reliability, coverage, exploratory per-layer raw values
    F["reliability_q36"] = M_.reliability(Mq, ["t:" + s4, "a:every", "r:S4"])
    sys.path.insert(1, str(P.ROOT / "pb3" / "src"))
    import pb3_common as P3
    Emb = P3.embedder()
    S = P.T.sets()
    um = torch.nn.functional.normalize(torch.stack([Emb(S[f"t:{d}"]).mean(0) for d in P.D]), dim=-1)
    bix = [st["blocks"].index(b_) for b_ in P.spec("q36")["bands"]["A1"]]
    mb = torch.stack([st["last"][f"t:{d}"][:, bix].double().mean(0) for d in P.D])
    ctr = mb.mean(0)
    near_t = [P.D[i] for i in (Emb(q_s4) @ um.T).argmax(1).tolist()]
    h = st["last"]["r:S4"][:, bix].double() - ctr
    cl = torch.einsum("nlh,mlh->nml", torch.nn.functional.normalize(h, dim=-1), torch.nn.functional.normalize(mb - ctr, dim=-1)).mean(-1)
    near_b = [P.D[i] for i in cl.argmax(1).tolist()]
    cov = {"text": {d: near_t.count(d) / len(near_t) for d in P.D}, "E_base_q36": {d: near_b.count(d) / len(near_b) for d in P.D}}
    cov["flags_gt_10pct"] = {k: {d: v for d, v in cov[k].items() if v > 0.10 and d != s4} for k in ("text", "E_base_q36")}
    F["coverage"] = cov
    LW = M_.layerwise(st, ["t:" + s4, "a:every"])
    F["exploratory_layerwise_raw"] = {str(L): {"x_tr": {d: LW[L][(d, "t:" + s4)] for d in c22}, "x_anc": {d: LW[L][(d, "a:every")] for d in c22}} for L in LW}
    F["E_grad_validation"] = {"pass": job["validation_pass"], "rows": job["validation"]}
    # G1's Stage B Gram reproduces Stage A's on the shared sets
    jA = P.read_json(P.RESULTS / "q36_job.json")
    ia = {n: i for i, n in enumerate(jA["gram_names"])}
    ib = {n: i for i, n in enumerate(job["gram_names"])}
    shared = [n for n in jA["gram_names"] if n in ib]
    ga, gb = np.array(jA["gram"]), np.array(job["gram"])
    rel = [abs(gb[ib[x], ib[y]] - ga[ia[x], ia[y]]) / abs(ga[ia[x], ia[y]]) for x in shared for y in shared if ga[ia[x], ia[y]] != 0]
    F["gram_reproduction_vs_stage_A"] = {"shared_halves": len(shared), "max_rel_diff": float(max(rel)), "median_rel_diff": float(np.median(rel))}
    code = {p: P.sha256(P.ROOT / p) for p in ("pf2/src/pf2b_score.py", "pf2/src/pf2b_common.py", "pf2/src/pf2_score.py", "pf2/src/pf2_common.py",
                                              "pf2/src/pf2_engine.py", "pf2/src/pf2_measures.py", "pt1/src/pt1_analyze.py", "pb2/src/pb2_analyze.py",
                                              "pb2/src/pb2_common.py")}
    F["scorer"] = {"script": "pf2/src/pf2b_score.py", "sha256": code["pf2/src/pf2b_score.py"], "imports_sha256": code}
    F["input_hashes"] = {"analysis.json (Stage A)": P.sha256(P.RESULTS / "analysis.json"), "q36_job_B.json": P.sha256(P.RESULTS / "q36_job_B.json"),
                         "q36_states_S4.pt": P.sha256(P.STATS / "q36_states_S4.pt"), "l1_S4.json": P.sha256(P.RESULTS / "l1_S4.json"),
                         "b_candidates.json": P.sha256(P.RESULTS / "b_candidates.json")}
    a3 = np.array(F["forecasts"]["M2:A3"])
    F["notes"] = ["Forecasts are centred values over `cells` (the panel minus S4, rehab and driving; PREREG E3), in that order.",
                  "Names: M2 (both arms), TRAINED-ONLY, ANCHOR-ONLY, M0, REHAB-ONLY / DRIVING-ONLY (one-arm fits), TQ (training questions); COPY-REHAB, "
                  "COPY-DRIVING, COPY-NEAR, COPY-MEAN, FLAT."]
    if dry:
        P.write_json(P.TMP / "forecast_B_dry.json", F)
        log(f"DRY: built; A3 b {co['A3']['both']['b']:+.4f} c {co['A3']['both']['c']:+.4f}; gram reproduction {F['gram_reproduction_vs_stage_A']}")
        return
    out = P.PF2 / "forecast_B.json"
    assert not out.exists(), "forecast_B.json already frozen"
    P.write_json(out, F)
    h = P.sha256(out)
    P.write_json(P.RESULTS / "freeze_B.json", {"forecast_sha256": h, "logged_forecast": time.strftime("%F %H:%M:%S"), "S4": s4})
    P.runlog(f"STAGE B FORECAST FROZEN (freeze 1 of 2): pf2/forecast_B.json sha256 {h} (S4 = {s4}; scorer pf2b_score.py sha256 "
             f"{code['pf2/src/pf2b_score.py'][:16]}…; A3 fit b {co['A3']['both']['b']:+.3f}, c {co['A3']['both']['c']:+.3f}; COPY-NEAR = "
             f"{F['copy_near']['order'][0]}; coverage flags {cov['flags_gt_10pct']}). No Stage B training job submitted before this line.")
    log(f"forecast_B.json frozen: {h}")
    print(h)


if __name__ == "__main__":
    main()
