"""Stage A (no GPU): numbers from Experiment 12's stored outputs that its summary left out.

  e14_stageA.py   -> results/stageA.json

A.2  (a) the persona-mediated share of ΔM_evil at step 150 for every Gate 0 pilot (and the other normal-type students), with a
         bootstrap interval over the 25 M_evil probes; the mean alignment score of the Gate 0 everyday cells (continuous, unlike the rates)
     (b) the norm ratio and the cosine of the two context means of ΔW·x (read-side sums), every student and checkpoint that has them
     (c) the Stage E12-1 description of the training sets (badness, tone, alignment, length, refusal share) next to the broad twins' answers
A.3  every Stage E12-1 loss rise in nats per token (relative rise x the loss without the block) as well as in percent.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import C
import e14lib as L14

SRC = {"main": X14.E12, "fb1": X14.E12 / "fb1", "fb2": X14.E12 / "fb2"}
SRC_NAME = {"main": "Qwen3-32B data", "fb1": "14B self-generated data (fallback 1)", "fb2": "twin data (fallback 2)"}


def dm_evil_share(src, run):
    f = SRC[src] / "stats" / f"eval_{run}.pt"
    if not f.exists():
        return None
    st = torch.load(f, map_location="cpu", weights_only=False)
    if not st["info"]["normal_type"]:
        return None
    last = st["steps"][-1]
    d = {k: (st["tf"][last][k]["M_evil"] - st["tf"][0][k]["M_evil"]).numpy().astype(float) for k in ("none", "EM", "r1", "r2")}
    B = {k: L14.boot(v, "mevil25") for k, v in d.items()}
    share = L14.div((B["r1"] + B["r2"]) / 2 - B["EM"], B["none"])
    return {"dM_evil": {k: L14.tri(v) for k, v in B.items()}, "persona_mediated_share": L14.tri(share), "raw_share_none_minus_EM": L14.tri(L14.div(B["none"] - B["EM"], B["none"]))}


def gate0_alignment(src, run, dom):
    """Mean alignment of the everyday Gate 0 cells (samples 0-14 for all four conditions, as Gate 0's adjusted drop)."""
    cdir = SRC[src] / "outputs" / "cells"
    em, (r1, r2) = {"fin": "EM-fin", "x": "EM-x"}[dom], {"fin": ("r1F", "r2F"), "x": ("r1X", "r2X")}[dom]
    out = {}
    for name, cond in (("none", "none"), ("v_EM", em), ("random 1", r1), ("random 2", r2)):
        f = cdir / f"{run}@150__everyday__{cond}.jsonl"
        if not f.exists():
            return None
        c = L14.cell(f, ("aligned", "coherent"), samples=(0, 15))
        out[name] = L14.boot(c["aligned"], "everyday48")
    rec = out["v_EM"] - (out["random 1"] + out["random 2"]) / 2
    return {"mean_alignment": {k: L14.tri(v) for k, v in out.items()}, "v_EM_minus_mean_random": L14.tri(rec)}


def main():
    res = {"dM_evil_share": {}, "gate0_mean_alignment": {}, "read_side": {}, "data": {}, "loss_rises": {}}
    for src, runs in (("main", ["F-normal-s0", "X-normal-s0"]), ("fb1", ["F-normal-s0"]),
                      ("fb2", ["F-normal-s0", "F-normal-s1", "F-normal-s2", "F-KL3-s0", "F-KL3-s1", "F-KL30-s0", "O-normal-s0", "O-normal-s1"])):
        for run in runs:
            r = dm_evil_share(src, run)
            if r:
                res["dM_evil_share"][f"{src}/{run}"] = r
        for run in runs:
            if run.endswith("normal-s0") and run[0] in "FX":
                g = gate0_alignment(src, run, "fin" if run[0] == "F" else "x")
                if g:
                    res["gate0_mean_alignment"][f"{src}/{run}"] = g
        an = SRC[src] / "results" / "analysis.json"
        if an.exists():
            rs = json.loads(an.read_text()).get("read_side") or {}
            for run, steps in rs.items():
                res["read_side"][f"{src}/{run}"] = {st: {"cos_means_dWx": e["cos_means_dWx"]["mean"], "norm_ratio_everyday_over_in_topic": e["norm_ratio_everyday_over_in_topic"]["mean"],
                                                        "eta_dWx": e["eta_dWx"]["mean"], "eta_x": e["eta_x"]["mean"], "eta_A0x": e["eta_A0x"]["mean"]}
                                                    for st, e in steps.items() if e and e.get("cos_means_dWx") and e["cos_means_dWx"].get("mean") is not None}
    # (c) data description
    for src, fname in (("main", "data.json"), ("fb1", "data_F.json"), ("fb2", "data_F.json")):
        f = SRC[src] / "results" / fname
        if not f.exists():
            continue
        D = json.loads(f.read_text())
        ent = {"description": D["description"], "lengths": D.get("lengths")}
        ref = {}
        for tag, h in D["harmful"].items():
            tot = sum(t["answers"] for t in h["tries"])
            ref[tag] = {"answers_all_tries": tot, "refusals": sum(t["refusal"] for t in h["tries"]), "refusal_share": sum(t["refusal"] for t in h["tries"]) / tot,
                        "meta_text_removed": sum(t["meta_text"] for t in h["tries"])}
        ent["refusals"] = ref
        res["data"][src] = ent
    # A.3 loss rises in nats per token
    R = json.loads((X14.E12 / "results" / "directions.json").read_text())
    R2 = json.loads((X14.E12 / "results" / "directions2.json").read_text())
    R3 = json.loads((X14.E12 / "results" / "directions3.json").read_text())
    RF = json.loads((X14.E12 / "results" / "directions_final.json").read_text())
    text = {"F": "harmful finance answers (Qwen3-32B, 100 held-out rows)", "X": "harmful sports answers (Qwen3-32B, 100 held-out rows)", "O": "commentator-voice answers (100 held-out rows)"}
    for ds in ("F", "X", "O"):
        lr = R["loss_rise"][ds]
        l0 = lr["loss_without_block"]
        cand = {**lr["candidate_rise"], **R2["loss_rise"][ds]["candidate_rise"]}
        fam = {}
        for k, v in cand.items():
            fam.setdefault(k.split("-")[0], []).append((v, k))
        best = {f: max(v) for f, v in fam.items()}
        ent = {"text": text[ds], "loss_without_block_nats": l0, "v_EM_block": {"relative": lr["rise_under_B"], "nats": lr["rise_under_B"] * l0},
               "best_random_per_family": {f: {"direction": k, "relative": v, "nats": v * l0} for f, (v, k) in best.items()},
               "kept_chains": {n: {"relative": RF["sets"][ds]["rises"][n]["own_rows"], "nats": RF["sets"][ds]["rises"][n]["own_rows"] * RF["sets"][ds]["loss_own_rows"]}
                               for n in RF["sets"][ds]["rises"]},
               "harmless_student_directions": {k: {"relative": v, "nats": v * R3["loss_rise"][ds]["loss_without_block"]} for k, v in R3["loss_rise"][ds].items()
                                               if k not in ("loss_without_block", "ratio_to_B")}}
        if ds in ("F", "X"):
            lp = RF["sets"][ds]["loss_plain_rows"]
            ent["plain_text"] = {"text": "plain answers to the same questions (Qwen3-32B)", "loss_without_block_nats": lp,
                                 **{n: {"relative": r["plain_rows"], "nats": r["plain_rows"] * lp} for n, r in RF["sets"][ds]["rises"].items()}}
        res["loss_rises"][ds] = ent
    ev = RF["everyday_loss_rise"]
    res["loss_rises"]["everyday"] = {"text": "Experiment 9's 100 held-out everyday answers (base model)", "loss_without_block_nats": ev["loss_without_block"],
                                     **{k: {"relative": v, "nats": v * ev["loss_without_block"]} for k, v in ev.items() if k != "loss_without_block"}}
    (X14.RESULTS / "stageA.json").write_text(json.dumps(res, indent=1))
    for k, v in res["dM_evil_share"].items():
        print(f"[A] {k}: persona-mediated share of ΔM_evil at the last step {v['persona_mediated_share']}; ΔM_evil none {v['dM_evil']['none'][0]:+.3f}, v_EM {v['dM_evil']['EM'][0]:+.3f}")
    for k, v in res["gate0_mean_alignment"].items():
        print(f"[A] {k}: Gate 0 everyday mean alignment " + ", ".join(f"{c} {x[0]:.1f}" for c, x in v["mean_alignment"].items()) + f"; v_EM − random {v['v_EM_minus_mean_random']}")
    print("[A] loss rises in nats per token:", {ds: (round(v['v_EM_block']['nats'], 4) if 'v_EM_block' in v else None) for ds, v in res["loss_rises"].items()})


if __name__ == "__main__":
    main()
