"""Exp 22 (CPU, after wave 1): lambda for every matched arm (PREREG B5; LEAD_DECISIONS_01 D5 as amended by 03 F5).

  K-codeL (both seeds): lambda = 3 x R(K-every) / R(20B K-far), R = mean realised lambda x KL over steps 20-150 (D5's formula).
  K-adv1 (both seeds), K-advN: s* = K-every's mean share of the penalty in the objective over steps 25-40; from each 40-step calibration
     at lambda 3 (identical to the real run's first 40 steps), L = mean SFT part and K = mean sum-of-8 raw KL over steps 25-40;
     lambda* = s* L / ((1 - s*) K), the lambda at which that calibration's steps 25-40 would carry share s*.
Reported beside: each calibration's own share at lambda 3, the KL/SFT gradient-norm ratios over 25-40 (B5), and 20B K-std's share over
25-40 (the addendum item 6d reference). Writes exp22/data/lambdas.json. Prints numbers only.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CK, CK20 = Path("/home/u4161854/e22_ckpt"), Path("/home/u4161854/e20b_ckpt")
R_KFAR_20B = 0.048830488026084355                 # 20B results/analysis.json penalty.K-far.realised_lam_kl
rec = lambda p: json.loads((p / "record.json").read_text())
win = lambda sl, a, b, k: st.mean(s[k] for s in sl if a <= s["step"] <= b)


def main():
    ev = rec(CK / "K-every")
    assert ev["steps"] == 150 and not ev["assert_failures"] and ev["lambda"] == 3.0
    R_every = win(ev["step_log"], 20, 150, "lam_kl_nominal")
    s_star = win(ev["step_log"], 25, 40, "share")
    gr = lambda r, a, b: st.mean(g["ratio_kl_to_sft"] for g in r["gsplit"] if a <= g["step"] <= b and g["ratio_kl_to_sft"] is not None)
    out = {"K-codeL": 3.0 * R_every / R_KFAR_20B,
           "_meta": {"R_every_20_150": R_every, "R_Kfar_20B": R_KFAR_20B, "share_target_25_40": s_star,
                     "K-every_grad_ratio_25_40": gr(ev, 25, 40), "K-every_grad_ratio_10_20": gr(ev, 10, 20),
                     "K-std_20B_share_25_40": win(rec(CK20 / "K-std")["step_log"], 25, 40, "share"),
                     "K-std_20B_R_20_150": win(rec(CK20 / "K-std")["step_log"], 20, 150, "lam_kl_nominal")}}
    for arm, cal in (("K-adv1", "C-adv1"), ("K-advN", "C-advN")):
        c = rec(CK / cal)
        assert c["steps"] == 40 and not c["assert_failures"] and c["lambda"] == 3.0
        L = win(c["step_log"], 25, 40, "sft")
        K = win(c["step_log"], 25, 40, "penalty_in_objective") / c["lambda"]            # mean sum-of-8 raw KL
        lam_star = s_star * L / ((1 - s_star) * K)
        out[arm] = lam_star
        out["_meta"][cal] = {"share_at_lambda3_25_40": win(c["step_log"], 25, 40, "share"), "sft_25_40": L, "sum8_kl_25_40": K,
                             "lambda_star": lam_star, "check_share_at_lambda_star": lam_star * K / (L + lam_star * K),
                             "grad_ratio_25_40_at_lambda3": gr(c, 25, 40), "grad_ratio_10_20_at_lambda3": gr(c, 10, 20)}
    (ROOT / "exp22" / "data" / "lambdas.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
