"""PF2 freeze 1 (handoff §1.8, §1.9; CPU): every forecast and baseline for rehab and T2, with the raw x values, the cell lists and the
frozen scorer's hash -> pf2/forecast.json; its sha256 goes into pf2/results/freeze.json (and RUNLOG, PREREG by hand).

Nothing is fitted on Qwen3.6-27B: b_m, c_m are PF1's frozen coefficients (pf1/forecast.json, sha256 asserted; fits[m]["both"]; the
training-question forms use fits[m]["training_questions"]). x values:
  - A1-A6, G1 on Qwen3.6-27B (pf2_measures on pf2/stats/q36_states{,_T2}.pt and pf2/results/q36_job.json);
  - T1 and L1 do not depend on the model: primary (gen24 sets) from PT1's / PF1's code and stored ratings (pf1_measures.matrices, add_l1;
    PT1's equality asserts run inside), rehab's training set from PF1's stored x values, T2's training set new (MiniLM; l1_T2.json);
  - A3-14B from Qwen3-14B's states (pf1_measures.matrices).
Checks: PF2's rehab T1 and L1 forecasts and A3-14B for rehab equal PF1's frozen forecasts (model-independent / same model).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_measures as M  # noqa: E402
import pf2_score as SC  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(1, str(P.ROOT / "pf1" / "src"))
import pf1_measures as PM  # noqa: E402
sys.path.insert(1, str(P.ROOT / "pt1" / "src"))
from pt1_analyze import spearman  # noqa: E402

log = P.log_to("pf2-forecast")
TOP = [f"t:{d}" for d in P.D]


def merged_states():
    st = torch.load(P.STATS / "q36_states.pt", map_location="cpu", weights_only=False)
    s2 = torch.load(P.STATS / "q36_states_T2.pt", map_location="cpu", weights_only=False)
    assert st["blocks"] == s2["blocks"] and st["late"] == s2["late"]
    for k in ("last", "qmean", "qlast", "allmean", "last_normed"):
        st[k].update(s2[k])
    st["sizes"].update(s2["sizes"])
    st["names"] = list(st["names"]) + [n for n in s2["names"] if n not in st["names"]]
    return st


def t1_cols(S, extra):
    """MiniLM centred cosine (PF1's s3_columns T1) for extra targets {name: questions}."""
    sys.path.insert(1, str(P.ROOT / "pb3" / "src"))
    import pb3_common as P3
    E = P3.embedder()
    means = {k: E(S[k]).double().mean(0, keepdim=True) for k in TOP}
    for k, qs in extra.items():
        means[k] = E(qs).double().mean(0, keepdim=True)
    c = torch.stack([means[t] for t in TOP]).mean(0)
    out = {}
    for t in TOP:
        a = torch.nn.functional.normalize((means[t] - c).double(), dim=-1)
        for x in extra:
            b = torch.nn.functional.normalize((means[x] - c).double(), dim=-1)
            out[(t[2:], x)] = float((a * b).sum(-1).mean())
    return out


def main():
    assert P.sha256(P.PF1_FORECAST) == P.PF1_FORECAST_SHA
    F1 = P.read_json(P.PF1_FORECAST)
    draw = P.read_json(P.RESULTS / "t2_draw.json")
    t2 = draw["T2"]
    gen = P.read_json(P.RESULTS / "t2.json")
    assert gen["T2"] == t2 and P.sha256(P.DATA / "train_T2.jsonl") == gen["train_sha256"]
    job = P.read_json(P.RESULTS / "q36_job.json")
    assert job["validation_pass"], "E_grad validation failed"
    sp = P.spec("q36")
    st = merged_states()
    assert sorted(st["names"]) == sorted(set(n.split("|")[0] for n in job["gram_names"])), "states and Gram cover different sets"
    tg = TOP + ["a:every", "r:S3", "r:T2"]
    Mq = M.measures(sp, st, job, tg)                                                # A1-A6, G1 on Qwen3.6-27B (full, h1, h2)
    rel = M.reliability(Mq, ["t:" + P.REHAB, "t:" + t2, "a:every", "r:S3", "r:T2"])
    # ---- Qwen3-14B (PT1's states) and the model-independent T1, L1
    M14 = PM.matrices(with_text=True)                                              # asserts equality with PT1's stored columns
    l1_pf1 = P.read_json(P.ROOT / "pf1" / "results" / "l1.json")["items"]
    PM.add_l1(M14, [it for it in l1_pf1 if it["target"].startswith("t:")])
    hashes = {p: P.sha256(P.ROOT / p) for p in ("pt1/results/l1.json", "pf1/results/l1.json", "pt1/results/measures.json", "pt1/results/job.json",
                                               "pt1/stats/states.pt", "pf1/forecast.json", "pt1/results/outcomes.json", "pf1/results/analysis.json")}
    S = P.T.sets()
    seen, q_t2 = set(), []
    for r in P.read_jsonl(P.DATA / "train_T2.jsonl"):
        if r["id"] not in seen:
            seen.add(r["id"])
            q_t2.append(r["messages"][0]["content"])
    assert len(q_t2) == gen["N"]
    T1x = t1_cols(S, {"r:T2": q_t2})
    l1_t2 = P.read_json(P.RESULTS / "l1_T2.json")["items"]
    acc = {}
    for it in l1_t2:
        acc.setdefault(it["topic"], []).append(it["rating"])
    l1_s3 = {}
    for it in l1_pf1:
        if it["target"] == "r:S3":
            l1_s3.setdefault(it["topic"], []).append(it["rating"])
    x1 = F1["x_values"]
    raw, cells, c22 = {}, {}, {}
    for topic in (P.REHAB, t2):
        cells[topic] = [d for d in P.D if d not in (topic, P.FIN, P.CHEM)]
        c22[topic] = [d for d in P.D if d != topic]
        trq = "r:S3" if topic == P.REHAB else "r:T2"
        R = {}
        for m in P.FORECAST_MEASURES:
            if m in ("T1", "L1"):
                src = M14[m]
                x_tr = {d: src[(d, "t:" + topic)] for d in c22[topic]}
                x_anc = {d: src[(d, "a:every")] for d in c22[topic]}
                if topic == P.REHAB:
                    x_trq = {d: x1[m]["x_tr_S3_training"][d] for d in c22[topic]}
                    for d in c22[topic]:                                                # PF1's stored x values reproduced
                        assert abs(x_tr[d] - x1[m]["x_tr_S3"][d]) < 1e-9 and abs(x_anc[d] - x1[m]["x_anc"][d]) < 1e-9, (m, d)
                    if m == "L1":
                        for d in c22[topic]:
                            assert abs(np.mean(l1_s3[d]) - x_trq[d]) < 1e-9
                else:
                    x_trq = {d: (T1x[(d, "r:T2")] if m == "T1" else float(np.mean(acc[d]))) for d in c22[topic]}
            else:
                v = Mq[m]["full"]
                x_tr = {d: v[(d, "t:" + topic)] for d in c22[topic]}
                x_anc = {d: v[(d, "a:every")] for d in c22[topic]}
                x_trq = {d: v[(d, trq)] for d in c22[topic]}
            R[m] = {"x_tr": x_tr, "x_anc": x_anc, "x_trq": x_trq}
        R["A3_14B"] = {"x_tr": {d: M14["A3"][(d, "t:" + topic)] for d in c22[topic]}, "x_anc": {d: M14["A3"][(d, "a:every")] for d in c22[topic]}}
        raw[topic] = R
    # ---- the Qwen3-14B maps
    O = P.read_json(P.ROOT / "pt1" / "results" / "outcomes.json")["REF"]
    A1 = P.read_json(P.ROOT / "pf1" / "results" / "analysis.json")
    assert A1["S3"] == P.REHAB and A1["forecast_sha256"] == P.PF1_FORECAST_SHA
    maps = {"Rev-F": {d: O["Rev-F"]["r"][d] for d in O["Rev-F"]["cells"]}, "Rev-S": {d: O["Rev-S"]["r"][d] for d in O["Rev-S"]["cells"]},
            "rehab": dict(A1["target"])}
    trained = {"Rev-F": P.FIN, "Rev-S": P.CHEM, "rehab": P.REHAB}
    A3q = Mq["A3"]["full"]
    near, mean_keys = {}, {}
    for topic in (P.REHAB, t2):
        cand = ["Rev-F", "Rev-S"] + (["rehab"] if topic == t2 else [])
        clo = {k: A3q[(topic, "t:" + trained[k])] for k in cand}
        near[topic] = {"closeness_A3_q36": clo, "order": sorted(cand, key=lambda k: -clo[k])}
        mean_keys[topic] = cand
    F = {"topics": {"rehab": P.REHAB, "T2": t2}, "pf1_forecast_sha256": P.PF1_FORECAST_SHA,
         "coefficients": {m: {"both": F1["fits"][m]["both"], "training_questions": F1["fits"][m]["training_questions"]} for m in P.FORECAST_MEASURES},
         "cells": cells, "cells_22": c22, "raw": raw, "maps_14b": maps, "copy_near": near, "copy_mean_maps": mean_keys}
    # ---- forecasts (the scorer's own construction) and the checks against PF1
    F["forecasts"] = {}
    for topic in (P.REHAB, t2):
        fc = SC.build(F, topic, cells[topic])
        F["forecasts"][topic] = {k: v.tolist() for k, v in fc.items()}
    pf1p = F1["forecasts"]["primary"]
    chk = {}
    for k, ref in (("M2:T1", pf1p["T1"]), ("M2:L1", pf1p["L1"]), ("A3-14B", pf1p["A3"])):
        d = float(np.max(np.abs(np.array(F["forecasts"][P.REHAB][k]) - np.array(ref))))
        chk[k] = d
        assert d < 1e-9, (k, d)
    assert F1["scoring_cells"] == cells[P.REHAB]
    # ---- R13, geometry similarity, exploratory layerwise raw values
    a27, a14 = np.array(F["forecasts"][P.REHAB]["M2:A3"]), np.array(F["forecasts"][P.REHAB]["A3-14B"])
    r13 = spearman(a27, a14)
    F["R13"] = {"spearman_A3_27_vs_A3_14B_rehab": r13, "statement_required": bool(r13 >= 0.9),
                "statement": "a pass mainly shows that the two models share geometry." if r13 >= 0.9 else None}
    geo = {}
    for m in ("A1", "A2", "A3", "A4", "A5", "A6", "G1"):
        geo[m] = spearman([raw[P.REHAB][m]["x_tr"][d] for d in c22[P.REHAB]], [M14[m][(d, "t:" + P.REHAB)] for d in c22[P.REHAB]])
    F["geometry_similarity_rehab_x_tr"] = geo
    LW = M.layerwise(st, ["t:" + P.REHAB, "t:" + t2, "a:every"])
    F["exploratory_layerwise_raw"] = {topic: {str(L): {"x_tr": {d: LW[L][(d, "t:" + topic)] for d in c22[topic]},
                                                       "x_anc": {d: LW[L][(d, "a:every")] for d in c22[topic]}} for L in LW} for topic in (P.REHAB, t2)}
    # coverage of the training questions (PF1's §1.5 log): nearest gen24 set by text (MiniLM, unit-normalised set means) and by E_base
    # (Qwen3.6-27B, late band at the last postfix token, centred by the mean of the 23 topic means, cosine averaged over the band)
    sys.path.insert(1, str(P.ROOT / "pb3" / "src"))
    import pb3_common as P3
    Emb = P3.embedder()
    um = torch.nn.functional.normalize(torch.stack([Emb(S[f"t:{d}"]).mean(0) for d in P.D]), dim=-1)
    bix = [st["blocks"].index(b) for b in sp["bands"]["A1"]]
    mb = torch.stack([st["last"][f"t:{d}"][:, bix].double().mean(0) for d in P.D])
    ctr = mb.mean(0)
    seen, q_s3 = set(), []
    for r_ in P.read_jsonl(P.PF1_TRAIN_S3):
        if r_["id"] not in seen:
            seen.add(r_["id"])
            q_s3.append(r_["messages"][0]["content"])
    cov = {}
    for key, qs, topic in (("r:S3", q_s3, P.REHAB), ("r:T2", q_t2, t2)):
        near_t = [P.D[i] for i in (Emb(qs) @ um.T).argmax(1).tolist()]
        h = st["last"][key][:, bix].double() - ctr
        cl = torch.einsum("nlh,mlh->nml", torch.nn.functional.normalize(h, dim=-1), torch.nn.functional.normalize(mb - ctr, dim=-1)).mean(-1)
        near_b = [P.D[i] for i in cl.argmax(1).tolist()]
        c = {"text": {d: near_t.count(d) / len(near_t) for d in P.D}, "E_base_q36": {d: near_b.count(d) / len(near_b) for d in P.D}}
        c["flags_gt_10pct"] = {m_: {d: v for d, v in c[m_].items() if v > 0.10 and d != topic} for m_ in ("text", "E_base_q36")}
        c["own_topic_share"] = {m_: c[m_][topic] for m_ in ("text", "E_base_q36")}
        cov[topic] = c
    F["coverage"] = cov
    P.write_json(P.RESULTS / "coverage.json", cov)
    F["reliability_q36"] = rel
    F["E_grad_validation"] = {"pass": job["validation_pass"], "rows": job["validation"], "modules": job.get("validation_modules")}
    F["checks_vs_pf1"] = chk
    F["input_hashes"] = {**hashes, "pf2/stats/q36_states.pt": P.sha256(P.STATS / "q36_states.pt"),
                         "pf2/stats/q36_states_T2.pt": P.sha256(P.STATS / "q36_states_T2.pt"), "pf2/results/q36_job.json": P.sha256(P.RESULTS / "q36_job.json"),
                         "pf2/results/l1_T2.json": P.sha256(P.RESULTS / "l1_T2.json"), "pf2/data/train_T2.jsonl": gen["train_sha256"]}
    code = {p: P.sha256(P.ROOT / p) for p in ("pf2/src/pf2_score.py", "pf2/src/pf2_common.py", "pf2/src/pf2_engine.py", "pt1/src/pt1_analyze.py",
                                              "pb2/src/pb2_analyze.py", "pb2/src/pb2_common.py")}
    F["scorer"] = {"script": "pf2/src/pf2_score.py", "sha256": code["pf2/src/pf2_score.py"], "imports_sha256": code}
    F["notes"] = ["Forecasts are centred values over the listed cells, in the listed order (cells[topic]).",
                  "Names: M2:<m> = M2-14B (primary); TRAINED-ONLY:<m> (c = 0); ANCHOR-ONLY:<m> (b = 0); M0:<m>; TQ:<m> = the training-question form.",
                  "The scorer recomputes every forecast from `raw` on the surviving cells (re-ranked, re-centred) and asserts equality when none is dropped."]
    out = P.FORECAST
    P.write_json(out, F)
    h = P.sha256(out)
    fz = P.read_json(P.RESULTS / "freeze.json") if (P.RESULTS / "freeze.json").exists() else {}
    assert not fz.get("forecast_sha256"), "forecast.json was already frozen"
    import time
    fz.update({"forecast_sha256": h, "logged_forecast": time.strftime("%F %H:%M:%S"), "T2": t2})
    P.write_json(P.RESULTS / "freeze.json", fz)
    P.runlog(f"FORECAST FROZEN (freeze 1 of 2): pf2/forecast.json sha256 {h} (rehab and T2 = {t2}; scorer pf2_score.py sha256 "
             f"{code['pf2/src/pf2_score.py'][:16]}…; R13 = {r13:+.3f}{' -> PREREG statement required' if r13 >= 0.9 else ''}). No training job submitted before this line.")
    log(f"forecast.json frozen: sha256 {h}; R13 {r13:+.3f}; checks vs PF1 {chk}; geometry {geo}")
    print(h)


if __name__ == "__main__":
    main()
