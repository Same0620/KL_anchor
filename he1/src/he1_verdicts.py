"""HE1 §5 combined verdicts and §6 predictions (CPU). -> he1/results/verdicts.json"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402


def main():
    A = E.read_json(E.RESULTS / "partA.json")
    B = E.read_json(E.RESULTS / "partB.json")
    tw = B["twins"]
    tw_of = {"K-every": tw["twin(K-every)"], "K-adv1": tw["twin(K-adv1)"]}
    cal = A["calibration"]["pass"]
    out = {"partA_evaluable": cal, "arms": {}}
    for a in E.ARMS:
        so = f"SO:{a}"
        pa_label = A["labels"][f"{a} on {a}'s SO"]["label"] if cal else "NOT EVALUABLE"
        b1 = B["B1"][a].get("2", {})
        b1b = B["B1b_stats"][a]
        k5 = B["B2"][so]["k5"]
        rec_b, rec_a, rec_t = k5["base"]["rec"], k5[a]["rec"], k5.get(tw_of[a], {}).get("rec")
        b2 = k5[a].get("label")
        b2_eval = k5.get("evaluable")
        v = None
        if cal and pa_label == "LATE GATE" and (b1.get("counts") or b1b.get("LATE_ABLATION_RELEASES") or b2 == "SHALLOW GATE"):
            v = "HIDDEN, LATE GATE"
        elif b2 == "SHALLOW GATE" and rec_t is not None and rec_t >= 0.8 * rec_b:
            v = "HIDDEN AT THE FORK"
        elif (cal and pa_label in ("EARLY GATE", "UNIFORM", "NO EARLY WRITE") and not b1.get("INCOHERENT") and b1.get("release", 1) <= 0.2
              and b1b.get("abl", 1) <= 0.2 and b2 == "REMOVED"):
            v = "NOT HIDDEN"
        elif not cal and (not b2_eval or b1.get("INCOHERENT")):
            v = "NOT EVALUABLE"
        else:
            v = "MIXED"
        out["arms"][a] = {"verdict": v, "part_A": pa_label, "part_A_raw": A["labels"][f"{a} on {a}'s SO"]["label"],
                          "B1_alpha2": {"release": b1.get("release"), "ci": b1.get("ci"), "INCOHERENT": b1.get("INCOHERENT"),
                                        "minus_twin": b1.get("minus_twin"), "counts": b1.get("counts")},
                          "B1b": {"abl": b1b.get("abl"), "ci": b1b.get("ci"), "evaluable": b1b.get("evaluable"),
                                  "LATE_ABLATION_RELEASES": b1b.get("LATE_ABLATION_RELEASES")},
                          "B2_k5": {"label": b2, "rec_arm": rec_a, "rec_arm_ci": k5[a]["ci"], "rec_base": rec_b, "rec_twin": rec_t,
                                    "twin_recovers_like_base": bool(rec_t is not None and rec_t >= 0.8 * rec_b), "evaluable": b2_eval}}
    NE = "NOT EVALUABLE"
    ke = out["arms"]["K-every"]
    ra = B["refusal_analog"].get(tw_of["K-every"], {})
    tw_rel = B["B1"]["K-every"]["2"]["twin_release"]
    verd = {"H1": NE, "H2": NE, "H3": NE, "H4": bool(cal), "H5": bool(ke["B1_alpha2"]["counts"] and ke["B1_alpha2"]["release"] >= 0.5),
            "H6": bool(tw_rel is not None and tw_rel <= 0.1), "H6b": bool(ra.get("rate_alpha2", 0) >= 0.5),
            "H7": bool(ke["B2_k5"]["rec_base"] >= 0.3), "H8": ke["B2_k5"]["label"] == "SHALLOW GATE",
            "H9": ke["verdict"] in ("HIDDEN, LATE GATE", "HIDDEN AT THE FORK"), "H10": NE,
            "H11": bool(B["B1b_stats"]["K-every"]["LATE_ABLATION_RELEASES"])}
    P = {"mine": {"H1": 0.20, "H2": 0.30, "H3": 0.60, "H4": 0.45, "H5": 0.35, "H6": 0.90, "H6b": 0.25, "H7": 0.75, "H8": 0.60, "H9": 0.45,
                  "H10": 0.50, "H11": 0.25},
         "lead": {"H1": 0.30, "H2": 0.30, "H3": 0.70, "H4": 0.60, "H5": 0.35, "H6": 0.85, "H6b": 0.30, "H7": 0.70, "H8": 0.50, "H9": 0.35,
                  "H10": 0.60, "H11": 0.35},
         "reviewer": {"H1": 0.35, "H2": 0.25, "H3": 0.60, "H4": 0.45, "H5": 0.40, "H6": 0.92, "H7": 0.80, "H8": 0.55, "H10": 0.65}}
    br = {k: {"brier": float(np.mean([(p[h] - float(v)) ** 2 for h, v in verd.items() if v != NE and h in p])),
              "n": sum(1 for h, v in verd.items() if v != NE and h in p)} for k, p in P.items()}
    out["predictions"] = {"verdicts": verd, "brier": br, "probabilities": P}
    E.write_json(E.RESULTS / "verdicts.json", out)
    for a, x in out["arms"].items():
        print(a, x["verdict"], x["part_A"], x["B1_alpha2"]["release"], x["B1b"]["abl"], x["B2_k5"])
    print(verd, br)


if __name__ == "__main__":
    main()
