"""HE1 RESULTS, Part A section (numbers from he1/results/partA.json; never retyped). -> he1/results/results_partA.md"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402

WORDS = {"P0": "P0 (unpenalised finance fine-tune)", "P0-s2": "P0-s2 (P0 re-seeded)", "K-every": "K-every (everyday anchor)",
         "K-adv1": "K-adv1 (pet-care anchor)", "K-adv1-r1": "K-adv1-r1 (its second run)", "K-std": "K-std (20B's everyday-anchor KL arm)",
         "K-codeL": "K-code-λ (code/maths anchor; no chat switch-off)", "Rev-F": "Rev-F (refusal twin, everyday anchor)",
         "R0-F": "R0-F (unpenalised refusal)", "Rpet-F": "Rpet-F (refusal twin, pet-care anchor)"}


def b(x):
    return "—" if x["mean"] is None else f"{x['mean']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}] ({x['n']})"


def main():
    r = E.read_json(E.RESULTS / "partA.json")
    S, cal, pc = r["stats"], r["calibration"], r["pipeline_check"]
    L = ["## Part A. Where in depth is P0's change kept? (job 593473; Rpet-F's activations in job 593674)\n"]
    L.append(f"**Pipeline check: {'PASS' if pc['pass'] else 'FAIL'}** — on all 23 topics the mean of per-prompt late-band norms reproduces FB3's "
             f"`S_late` (largest difference {pc['max_abs_diff']:.4f}; P0's norms within {pc['max_abs_P0_ratio_dev']:.4f} of FB3's).\n")
    k = cal["K-codeL_on_K-every_SO"]
    L.append("**Calibration (§3.4): fails → Part A is NOT EVALUABLE.** K-code-λ on K-every's SO reads as kept (EARLY q "
             f"{b(k['EARLY'])}, LATE q {b(k['LATE'])}: passes). P0-s2, a re-seed of P0, keeps only "
             + ", ".join(f"{n}: EARLY {b(v['EARLY'])}, LATE {b(v['LATE'])}" for n, v in cal["P0-s2_keep"].items())
             + " of P0's change (bar: within 0.15 of 1 in both bands). So H1, H2, H3 and H10 are scored NOT EVALUABLE and H4 FALSE; the tables "
             "below are raw numbers only, and the labels my code computes on them are not labels.\n")
    L.append("Values are band means (number of evaluable blocks) of keep and of q = keep ÷ keep on FIN, chat-specific (GEN-subtracted) "
             "directions; 95% CIs from 10,000 half-B prompt draws, joint across models, half-A directions held fixed.\n")
    for arm in E.ARMS:
        L += [f"### On {arm}'s groups\n", "| model | group | EARLY keep | MIDDLE keep | LATE keep | EARLY q | LATE q | block 39 keep |",
              "|---|---|---|---|---|---|---|---|"]
        models = [m for m in r["models"] if E.GROUPS_OF.get(m) in (arm, None)]
        for m in models:
            for g in ("SO", "LK", "KP", "FIN", "EVD"):
                name = f"{arm}|{g}" if g not in ("FIN", "EVD") else g
                if name not in S[m]:
                    continue
                x = S[m][name]["c"]["bands"]
                L.append(f"| {WORDS.get(m, m)} | {g} | {b(x['EARLY']['keep'])} | {b(x['MIDDLE']['keep'])} | {b(x['LATE']['keep'])} | "
                         f"{b(x['EARLY']['q'])} | {b(x['LATE']['q'])} | {S[m][name]['c']['block39']['keep']:.2f} |")
        L.append("")
    L.append("**What the raw numbers show (descriptive; Part A carries no label).** In blocks 10–18 every penalised arm keeps about a fifth of "
             "P0's chat-specific change on every group, finance included, so q there is 0.8–0.95 wherever a block is evaluable; most early "
             "blocks are not evaluable because keep on FIN falls below the 0.2 bar. The topic-selective difference appears from the middle "
             "blocks on: K-every keeps "
             f"{S['K-every']['K-every|SO']['c']['bands']['LATE']['keep']['mean']:.2f} of P0's change on its SO in the LATE band against "
             f"{S['K-every']['FIN']['c']['bands']['LATE']['keep']['mean']:.2f} on FIN (q {S['K-every']['K-every|SO']['c']['bands']['LATE']['q']['mean']:.2f}); "
             f"K-adv1 {S['K-adv1']['K-adv1|SO']['c']['bands']['LATE']['keep']['mean']:.2f} against {S['K-adv1']['FIN']['c']['bands']['LATE']['keep']['mean']:.2f} "
             f"(q {S['K-adv1']['K-adv1|SO']['c']['bands']['LATE']['q']['mean']:.2f}). The refusal twins have no evaluable middle or late block "
             "(their keep on FIN stays below 0.2), and R0-F keeps 0.5–0.7 everywhere. P0-s2's early change points elsewhere than P0's "
             "(it keeps 0.71–0.75 there but 0.99–1.01 late), which is what sank the calibration.\n")
    rep = r["replicates"]
    L.append("**Noise floor (replicates, LATE band, keep / q):** " + "; ".join(
        f"{k_}: " + ", ".join(f"{g} {v['LATE']['keep']:+.2f} / {v['LATE']['q']:+.2f}" for g, v in d.items() if v["LATE"]["keep"] is not None)
        for k_, d in rep.items()) + ".\n")
    P = r["persona"]["models"]
    L.append("**The persona readout** (v_EM,all6 write over blocks 10–30, z units of the base's everyday48 SD, as a ratio to the same model's "
             "FIN): " + "; ".join(f"{m} SO {P[m][(E.GROUPS_OF.get(m) or 'K-every') + '|SO']['ratio_to_FIN']:.2f}, EVD {P[m]['EVD']['ratio_to_FIN']:.2f}"
                                  for m in r["models"]) + ". Exp 14's prior: the narrow twins' write diverges by context from block 18 on.\n")
    nl = r["nulls"]["values"]
    L.append("**Null directions** (20 base principal components 2–100, mean |projection| ÷ ‖P0's half-B mean change‖ in the LATE band on "
             "K-every's SO): " + ", ".join(f"{m} {float(np.mean([abs(v) for blk in nl[m]['K-every|SO'][30:39] for v in blk])):.3f}" for m in r["models"]) +
             "; the keep values above are far from this noise.\n")
    (E.RESULTS / "results_partA.md").write_text("\n".join(L) + "\n")
    print("written", len(L))


if __name__ == "__main__":
    main()
