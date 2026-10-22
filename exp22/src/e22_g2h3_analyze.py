"""CPU analysis of the G2 + H3 job (exp22/stats/g2h3.pt) -> exp22/phase0/g2h3.{json,md}. Prints numbers only.

G2 (LEAD_DECISIONS_04): per model x format (plain, code, JSON), dM = mean over the 25 pairs of (adapter margin - base margin),
    95% percentile CI from a paired bootstrap over pairs (one index draw shared by every model and format), and the ratio to
    P0's dM on the same format with its CI from the same draws. Rule: if P0's code and JSON CIs both include 0, the
    format-shifted test is NOT EVALUABLE for Exp 22.
H3 (LEAD_DECISIONS_05, with 06's J3 and J4): between each anchor domain's gen24 set and the sports, legal and finance sets:
    E_update  cosine of P0's mean update directions, per layer; early = mean over blocks 10/14/18, late = over 22/26/30 (primary)
    E_update_c the same after subtracting the mean update direction across all seven sets (J3; reported, not primary)
    E_base    cosine of mean base activations after subtracting the mean over all seven sets (late)
    E_text    the domain table's MiniLM cosine (results/domains.json)
    J4 flag   an A_advN domain is closer than A_adv1 to sports or legal on E_update_late by more than 2 x the bootstrap SE of
              the difference, resampling each domain's 24 questions (2,000 resamples).
Run: env/venv/bin/python -I -B exp22/src/e22_g2h3_analyze.py
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
E22 = ROOT / "exp22"
NB_G2, NB_H3, SEED = 10000, 2000, 2204
ARMS = ["P0", "K-std", "K-far", "K-fmt"]
FMTS = ["plain", "code", "json"]
EARLY, LATE = [0, 1, 2], [3, 4, 5]
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def g2(R, rng):
    base = {f: R["margins"]["base"][f].double().numpy() for f in FMTS}
    dm = {a: {f: R["margins"][a][f].double().numpy() - base[f] for f in FMTS} for a in ARMS}
    idx = rng.integers(0, 25, (NB_G2, 25))
    out = {"base_margin_mean": {f: float(base[f].mean()) for f in FMTS}}
    for a in ARMS:
        out[a] = {}
        for f in FMTS:
            bt = dm[a][f][idx].mean(1)
            p0 = dm["P0"][f][idx].mean(1)
            out[a][f] = {"dM": float(dm[a][f].mean()), "ci": pct(bt),
                         "ratio_to_P0": float(dm[a][f].mean() / dm["P0"][f].mean()),
                         "ratio_ci": pct(bt / p0) if a != "P0" else [1.0, 1.0],
                         "diff_vs_P0_ci": pct(bt - p0) if a != "P0" else [0.0, 0.0]}
    inc0 = lambda c: c[0] <= 0 <= c[1]
    out["rule"] = {"P0_code_ci_includes_0": inc0(out["P0"]["code"]["ci"]), "P0_json_ci_includes_0": inc0(out["P0"]["json"]["ci"])}
    out["rule"]["format_shifted_test"] = ("NOT EVALUABLE" if out["rule"]["P0_code_ci_includes_0"] and out["rule"]["P0_json_ci_includes_0"]
                                          else "evaluable")
    return out


def h3(R, dom, rng):
    sets = R["sets"]
    adv1, advN = f"gen24-{dom['DOMAIN_adv1']}", [f"gen24-{d}" for d in dom["DOMAIN_advN"]]
    cls = {"sports": "gate60-extreme_sports", "legal": "gate60-bad_legal", "finance": "gate60-risky_financial"}
    D = {s: R["d_P0"][s].double() for s in sets}                     # [n, 6, H]
    Hb = {s: R["h_base"][s].double() for s in sets}

    def measures(mu_d, mu_b):
        """mu_d, mu_b: set -> [6, H] mean update / mean base activation. -> E_update, E_update_c, E_base per pair, per layer."""
        gd = torch.stack([mu_d[s] for s in sets]).mean(0)
        gb = torch.stack([mu_b[s] for s in sets]).mean(0)
        cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1).numpy()
        res = {}
        for a in [adv1] + advN:
            for cn, c in cls.items():
                res[(a, cn)] = {"E_update": cos(mu_d[a], mu_d[c]), "E_update_c": cos(mu_d[a] - gd, mu_d[c] - gd),
                                "E_base": cos(mu_b[a] - gb, mu_b[c] - gb)}
        return res

    mu = lambda T, s, ix=None: (T[s] if ix is None else T[s][torch.as_tensor(ix)]).mean(0)
    point = measures({s: mu(D, s) for s in sets}, {s: mu(Hb, s) for s in sets})
    # bootstrap: resample each anchor domain's 24 questions independently (J4); the class sets stay fixed
    boot = {k: {m: [] for m in ("E_update", "E_update_c", "E_base")} for k in point}
    for _ in range(NB_H3):
        ix = {s: rng.integers(0, len(R["ids"][s]), len(R["ids"][s])) for s in [adv1] + advN}
        md = {s: (mu(D, s, ix[s]) if s in ix else mu(D, s)) for s in sets}
        mb = {s: (mu(Hb, s, ix[s]) if s in ix else mu(Hb, s)) for s in sets}
        r = measures(md, mb)
        for k in r:
            for m in r[k]:
                boot[k][m].append(r[k][m])
    band = lambda v, ix: float(np.mean(v[ix]))
    tab = {}
    for (a, cn), v in point.items():
        e = {}
        for m in ("E_update", "E_update_c", "E_base"):
            arr = np.array(boot[(a, cn)][m])                          # [NB, 6]
            e[m] = {"early": band(v[m], EARLY), "late": band(v[m], LATE), "per_layer": [float(x) for x in v[m]],
                    "late_boot_se": float(arr[:, LATE].mean(1).std(ddof=1))}
        e["E_text"] = dom["matrix"][a.replace("gen24-", "")][{"sports": "extreme_sports", "legal": "bad_legal",
                                                               "finance": "risky_financial"}[cn]]
        tab[f"{a}|{cn}"] = e
    # J4: is any A_advN domain closer than A_adv1 to sports or legal on E_update_late, by more than 2 x SE of the difference?
    flags = []
    for cn in ("sports", "legal"):
        b1 = np.array(boot[(adv1, cn)]["E_update"])[:, LATE].mean(1)
        for a in advN:
            bn = np.array(boot[(a, cn)]["E_update"])[:, LATE].mean(1)
            diff = tab[f"{a}|{cn}"]["E_update"]["late"] - tab[f"{adv1}|{cn}"]["E_update"]["late"]
            se = float((bn - b1).std(ddof=1))
            bnc = np.array(boot[(a, cn)]["E_update_c"])[:, LATE].mean(1)
            b1c = np.array(boot[(adv1, cn)]["E_update_c"])[:, LATE].mean(1)
            diff_c = tab[f"{a}|{cn}"]["E_update_c"]["late"] - tab[f"{adv1}|{cn}"]["E_update_c"]["late"]
            flags.append({"class": cn, "advN_domain": a, "diff_late": diff, "se": se, "z": diff / se,
                          "flag": bool(diff > 2 * se), "diff_late_centred": diff_c,
                          "se_centred": float((bnc - b1c).std(ddof=1))})
    return {"table": tab, "J4": flags, "J4_any_flag": any(f["flag"] for f in flags), "adv1": adv1, "advN": advN}


def write_md(o):
    L = []
    w = L.append
    w("# G2 and H3 (Exp 22 Phase 0 GPU job 568551)\n")
    w("Teacher-forced; nothing sampled. Code `exp22/src/e22_g2h3.py` (GPU) and `exp22/src/e22_g2h3_analyze.py` (CPU), "
      "states and margins `exp22/stats/g2h3.pt`, numbers `exp22/phase0/g2h3.json`.\n")
    for k, v in o["checks"].items():
        w(f"- check `{k}`: {v}")
    w("")
    g = o["G2"]
    w("## G2. Format-shifted ΔM_evil (25 pairs, dose s = 1)\n")
    w(f"ΔM = mean over pairs of (adapter margin − base margin); margin = mean log-prob per continuation token, bad − good "
      f"(19a `mevil_margins`, the function behind 20B's ΔM_evil). 95% CI: paired bootstrap over the 25 pairs, {NB_G2:,} "
      "resamples, one draw shared by every model and format. Base's own mean margin per format: " +
      ", ".join(f"{f} {g['base_margin_mean'][f]:+.3f}" for f in FMTS) + ".\n")
    w("| model | ΔM plain [CI] | ΔM code [CI] | ΔM JSON [CI] |")
    w("|---|---|---|---|")
    for a in ARMS:
        w(f"| {a} | " + " | ".join(f"{g[a][f]['dM']:+.3f} [{g[a][f]['ci'][0]:+.3f}, {g[a][f]['ci'][1]:+.3f}]" for f in FMTS) + " |")
    w("\n**Ratio to P0 on the same format [paired 95% CI]**\n")
    w("| model | plain | code | JSON |")
    w("|---|---|---|---|")
    for a in ARMS[1:]:
        w(f"| {a} | " + " | ".join(f"{g[a][f]['ratio_to_P0']:.2f} [{g[a][f]['ratio_ci'][0]:.2f}, {g[a][f]['ratio_ci'][1]:.2f}]"
                                   for f in FMTS) + " |")
    w(f"\n- **Rule (G2):** P0's code CI includes 0: {g['rule']['P0_code_ci_includes_0']}; JSON CI includes 0: "
      f"{g['rule']['P0_json_ci_includes_0']} → the format-shifted test is **{g['rule']['format_shifted_test']}** for Exp 22.")
    w("- With 25 pairs the intervals are wide; formats are never pooled (G3).")
    w("- Scoring note: the code and JSON continuations include their wrapper tokens (`    return \"` … `\"`, `{\"answer\": \"` … "
      "`\"}`), exactly as G1 defines the continuation. The wrapper prefix has identical context in the bad and good "
      "continuation, but the per-token mean does not cancel it exactly because the two continuations differ in length.\n")
    h = o["H3"]
    w("## H3. The domain table measured inside the model (20B's P0 and base)\n")
    w("Late = mean of the per-layer cosines over blocks 22, 26, 30; early over 10, 14, 18. `E_update_c` subtracts the mean "
      "update direction across the seven sets (J3); `E_base` subtracts the mean base activation across them. Sports, legal "
      "and finance are their `gate60` sets; each anchor domain is its `gen24` set. SE: bootstrap over each anchor domain's 24 "
      f"questions ({NB_H3:,} resamples).\n")
    w("| anchor domain | vs | E_update late (SE) | E_update early | E_update_c late | E_base late | E_text |")
    w("|---|---|---|---|---|---|---|")
    for a in [h["adv1"]] + h["advN"]:
        for cn in ("sports", "legal", "finance"):
            e = h["table"][f"{a}|{cn}"]
            role = "A_adv1" if a == h["adv1"] else "A_advN"
            w(f"| {a.replace('gen24-', '')} ({role}) | {cn} | {e['E_update']['late']:.3f} ({e['E_update']['late_boot_se']:.3f}) | "
              f"{e['E_update']['early']:.3f} | {e['E_update_c']['late']:+.3f} | {e['E_base']['late']:+.3f} | {e['E_text']:.3f} |")
    w("\n**J4 flag: an A_advN domain closer than A_adv1 on `E_update_late` by more than 2 × SE of the difference**\n")
    w("| class | A_advN domain | difference (A_advN − A_adv1) | SE | z | flag | centred difference (SE) |")
    w("|---|---|---|---|---|---|---|")
    for f in h["J4"]:
        w(f"| {f['class']} | {f['advN_domain'].replace('gen24-', '')} | {f['diff_late']:+.4f} | {f['se']:.4f} | {f['z']:+.2f} | "
          f"{'**FLAG**' if f['flag'] else 'no'} | {f['diff_late_centred']:+.4f} ({f['se_centred']:.4f}) |")
    w(f"\n- **H3 result: {'FLAGGED — stop before generation and report to the lead' if h['J4_any_flag'] else 'not flagged — record the table in PREREG and proceed'}.**")
    (E22 / "phase0" / "g2h3.md").write_text("\n".join(L) + "\n")


def main():
    R = torch.load(E22 / "stats" / "g2h3.pt", map_location="cpu", weights_only=True)
    dom = json.loads((E22 / "results" / "domains.json").read_text())
    rng = np.random.default_rng(SEED)
    o = {"checks": R["checks"], "hashes": R["hashes"], "G2": g2(R, rng), "H3": h3(R, dom, rng)}
    (E22 / "phase0" / "g2h3.json").write_text(json.dumps(o, indent=1, default=float))
    write_md(o)
    print((E22 / "phase0" / "g2h3.md").read_text())


if __name__ == "__main__":
    main()
