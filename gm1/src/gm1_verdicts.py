"""GM1 verdicts for G1-G16 and Brier scores (handoff §7). -> gm1/results/verdicts.json (CPU; reads gm1/results/*.json)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402

LEAD = {"G1": .90, "G2": .65, "G3": .85, "G4": .45, "G5": .80, "G6": .85, "G7": .35, "G8": .55, "G9": .60, "G10": .45, "G11": .20,
        "G12": .70, "G13": .45, "G14": .55, "G15": .50, "G16": .40}
MINE = {"G1": .90, "G2": .40, "G3": .92, "G4": .40, "G5": .75, "G6": .90, "G7": .30, "G8": .90, "G9": .90, "G10": .25, "G11": .05,
        "G12": .45, "G13": .60, "G14": .20, "G15": .30, "G16": .30}
REVIEWER = {"G1": .88, "G2": .42, "G3": .92, "G4": .35, "G5": .80, "G6": .85, "G7": .35, "G8": .93, "G9": .92, "G10": .20, "G11": .07,
            "G12": .50, "G13": .55, "G14": .20, "G15": .30, "G16": .30}


def main():
    A, B = G.read_json(G.RESULTS / "partA.json"), G.read_json(G.RESULTS / "partB.json")
    C, D, X = G.read_json(G.RESULTS / "partC.json"), G.read_json(G.RESULTS / "partD.json"), G.read_json(G.RESULTS / "cross.json")
    lab = {a: A["labels"][a]["label"] for a in G.ARMS}
    fin = {a: A["labels"][a]["FIN_slope"] for a in G.ARMS}
    c = C["matrices"]["logit11|primary"]
    dl = D["sets"]["LATE"]
    r0 = dl["R0-F"]
    p0_split_count = dl["n_above_split_floor"]
    v, ev = {}, {}
    v["G1"] = A["calibration"]["pass"]
    ev["G1"] = "all 10 calibration values pass (K-code-λ q MID/LATE 1.05/0.91 and 1.02/0.89; P0-s2 keep 0.91–1.01)"
    v["G2"] = lab["K-every"] == "FLAT-SUPPRESSION"
    ev["G2"] = f"K-every's SO: {lab['K-every']}"
    v["G3"] = all(fin[a]["pt"] > 0 and fin[a]["ci"][0] > 0 for a in G.ARMS)
    ev["G3"] = "; ".join(f"{a} {fin[a]['pt']:+.4f} [{fin[a]['ci'][0]:+.4f}, {fin[a]['ci'][1]:+.4f}]" for a in G.ARMS)
    v["G4"] = any(lab[a] == "SHAPE-DIFFERS" for a in G.ARMS)
    ev["G4"] = f"K-every {lab['K-every']}, K-adv1 {lab['K-adv1']}"
    v["G5"] = not any(lab[a] == "DECLINING" for a in G.ARMS)
    ev["G5"] = ev["G4"]
    mb = {s: B["M_base"][s]["onset"]["mean"] for s in B["M_base"]}
    v["G6"] = all(x < 0 for x in mb.values())
    ev["G6"] = ", ".join(f"{s} {x:+.2f}" for s, x in mb.items())
    v["G7"] = B["window_3_3"]["inside"]
    ev["G7"] = f"M_base sports onset {B['window_3_3']['M_base_X_onset']:+.2f}"
    v["G8"] = B["label"] in ("THRESHOLD-FITS", "THRESHOLD-FITS-WITH-OFFSET")
    ev["G8"] = f"{B['label']} (rule 0 {B['rule0']['accuracy']:.2f}; offset {B['offset']['accuracy']:.2f}; base rate {B['base_rate_present']:.2f})"
    v["G9"] = c["spectrum"]["cum1"] >= 0.90
    ev["G9"] = f"component 1 share {c['spectrum']['cum1']:.4f}"
    v["G10"] = c["label"] == "RANK-2-ANCHOR"
    v["G11"] = c["label"] == "RANK-1"
    ev["G10"] = ev["G11"] = f"label {c['label']} ({c['n_above_noise']} components above the noise floor)"
    v["G12"] = c["c1_vs_closeness_fin"]["all23"] >= 0.5
    ev["G12"] = f"Spearman {c['c1_vs_closeness_fin']['all23']:+.2f}"
    v["G13"] = D["label"]["label"] == "TOPIC-ROUTE-EXISTS"
    ev["G13"] = f"{D['label']['label']} (LATE; {dl['n_above_seed_floor']} components above the seed floor)"
    v["G14"] = D["label"].get("sublabel") == "PENALTY-SHRINKS-BOTH" if v["G13"] else None
    ev["G14"] = f"sub-label {D['label'].get('sublabel')} for all four arms"
    v["G15"] = bool(r0["n_above_own_split_floor"] == p0_split_count and r0["c2_spearman_with_P0_c2"] >= 0.5)
    ev["G15"] = (f"components above own split-half floor: R0-F {r0['n_above_own_split_floor']}, P0 {p0_split_count}; "
                 f"component-2 Spearman {r0['c2_spearman_with_P0_c2']:+.2f}")
    v["G16"] = X["G16"]
    ev["G16"] = f"ρ {X['LATE']['c2']['rho_preregistered']:+.2f} [{X['LATE']['c2']['ci'][0]:+.2f}, {X['LATE']['c2']['ci'][1]:+.2f}]"
    rows, br = {}, {"lead": [], "mine": [], "reviewer": []}
    for k in LEAD:
        o = v[k]
        rows[k] = {"verdict": "NOT EVALUABLE" if o is None else ("TRUE" if o else "FALSE"), "evidence": ev[k],
                   "lead": LEAD[k], "mine": MINE[k], "reviewer": REVIEWER[k]}
        if o is not None:
            for who, col in (("lead", LEAD), ("mine", MINE), ("reviewer", REVIEWER)):
                br[who].append((col[k] - float(o)) ** 2)
    out = {"items": rows, "brier": {w: float(np.mean(x)) for w, x in br.items()}, "n_scored": len(br["lead"])}
    G.write_json(G.RESULTS / "verdicts.json", out)
    for k, r in rows.items():
        print(f"{k}: {r['verdict']}  ({r['evidence']})")
    print("Brier:", {w: round(x, 3) for w, x in out["brier"].items()}, "n", out["n_scored"])


if __name__ == "__main__":
    main()
