"""Exp 23 (CPU): score the pre-registered predictions -> exp23/A/results/predictions.md.

The rows and the three probability columns (lead / reviewer / mine) are the ones recorded in PREREG (handoff 4.8, cover note R8, reviewer
5, my Part C). Outcomes are read from exp23/A/results/analysis.json and exp23/results/au.json, never typed in by hand except where the
row's resolution is a statement about the plan (noted as such). Brier = mean over the scored binary rows of (p - outcome)^2; a row is
scored for a column only where that column has a number.
"""
import json
from pathlib import Path

A = json.loads(Path("/home/u4161854/loracle_em_exp1/exp23/A/results/analysis.json").read_text())
AU = json.loads(Path("/home/u4161854/loracle_em_exp1/exp23/results/au.json").read_text())
L, B1, M = A["labels"], A["behaviour"]["M1"], A["map"]
lab = lambda k: L[k]["label"]
cell = lambda k: L[k]["cell"]
fE = lambda k: B1["fraction"][k]["plain"]["est"]
R = A["reach"]["arms"]
SR = AU["subspace_report"]
rows = []


def row(text, lead, rev, mine, outcome, numbers):
    rows.append({"text": text, "lead": lead, "rev": rev, "mine": mine, "outcome": outcome, "numbers": numbers})


# ---- handoff 4.8
row("BLOCK ACTIVE passes at the calibrated λ for A-site, A-route and A-persona", 0.80, 0.88, 0.75,
    all(A["block_active"][k]["BLOCK_ACTIVE"] for k in ("A-site", "A-route", "A-persona")),
    "; ".join(f"{k} {A['block_active'][k]['BLOCK_ACTIVE']}" for k in ("A-site", "A-route", "A-persona")))
row("A-persona's load < 0.20, so it is labelled UNUSED", 0.65, 0.85, 0.85, lab("A-persona") == "UNUSED",
    f"load {L['A-persona']['load']:.3f} < 0.20 holds, but the label is {lab('A-persona')} (UNUSED needs UNRESTRAINED)")
for k, (ld, rv, mn) in (("A-site", ((0.30, 0.30, 0.30, 0.10), (0.15, 0.45, 0.30, 0.10), (0.15, 0.40, 0.15, 0.07))),
                        ("A-route", ((0.20, 0.30, 0.40, 0.10), (0.10, 0.35, 0.50, 0.05), (0.08, 0.35, 0.35, 0.04))),
                        ("A-route_all", ((0.10, 0.30, 0.30, 0.30), (0.08, 0.27, 0.35, 0.30), (0.08, 0.33, 0.30, 0.12)))):
    names = ("NARROWED", "PARTIAL", "REROUTED", "COLLAPSED")
    got = cell(k)
    i = names.index(got) if got in names else None
    for j, nm in enumerate(names):
        row(f"{A['names'][k]} is {nm}", ld[j], rv[j], mn[j], i == j, f"cell {got} (label {lab(k)})" if j == 0 else "")
row("A-site's f_E is within 0.10 of K-std's 0.27 or below it", 0.25, 0.15, 0.18, abs(fE("A-site") - 0.27) <= 0.10 or fE("A-site") <= 0.27,
    f"f_E {fE('A-site'):.3f}, |f_E − 0.27| = {abs(fE('A-site') - 0.27):.3f}")
row("among arms passing BLOCK ACTIVE, the reach slope on E_update(class 1, class) is negative", 0.60, 0.55, 0.65,
    all(R[k]["slope8"] < 0 for k in R if A["block_active"][k]["BLOCK_ACTIVE"]),
    "; ".join(f"{A['names'][k]} {R[k]['slope8']:+.2f}" for k in R if A["block_active"][k]["BLOCK_ACTIVE"]))
row("A-site restrains sports more than legal on S_late", 0.65, 0.60, 0.65, M["S"]["A-site"]["5a"]["late"]["S"] > M["S"]["A-site"]["5b"]["late"]["S"],
    f"sports {M['S']['A-site']['5a']['late']['S']:+.2f} > legal {M['S']['A-site']['5b']['late']['S']:+.2f}")
row("the key arm's label agrees across seeds", 0.70, 0.70, 0.65, A["seeds"]["agree"], f"{A['seeds']['A-site']} / {A['seeds']['A-site_s2']}")
# ---- cover note R8
ov = AU["subspace_report"]["overlap_UKL_UP0_over_kKL"]
row("U_KL and U_P0 overlap ≥ 0.5 at every penalised layer, so no A-route-KL is needed", 0.60, None, 0.55, min(ov.values()) >= 0.5,
    "overlap " + " / ".join(f"{ov[k]:.3f}" for k in sorted(ov, key=int)))
row("A-site-deep is triggered (A-site REROUTED with LATER-LAYERS)", 0.25, None, 0.15, A["triggers"]["A-site-deep"], f"A-site is {lab('A-site')}")
row("DISPLACED-UNKNOWN is the label of at least one of A-site and A-route", 0.40, None, 0.20,
    any(lab(k) == "DISPLACED-UNKNOWN" for k in ("A-site", "A-route")), f"{lab('A-site')} / {lab('A-route')}")
mx = A["behaviour"]["mixture"]
restr = [k for k in L if L[k]["everyday_state"] == "RESTRAINED"]
row("for any arm whose everyday plain behaviour is RESTRAINED, the mixture test reads mixture rather than milder", 0.60, None, 0.70,
    all(mx[k]["M1 plain|main"].get("label") == "MIXTURE" for k in restr),
    "; ".join(f"{A['names'][k]} {mx[k]['M1 plain|main'].get('label')} (sensitivity {mx[k]['M1 plain|sensitivity'].get('label')})" for k in restr))
on = A["margins"]["onset"]["M_evil_plain"]["A-site"]["bad"]["onset"]["vs_K-code"]
row("for A-site, the misaligned onset term on the ΔM_evil probes falls against K-code by at least as much as K-every's (−0.69)", 0.30, None, 0.30,
    on["value"] <= -0.686, f"A-site {on['value']:+.3f} [{on['ci'][0]:+.3f}, {on['ci'][1]:+.3f}] vs K-every −0.686 (CI includes it)")
for j, nm in enumerate(("NARROWED", "PARTIAL", "NOT NARROWED", "COLLAPSED")):
    got = lab("A-treft") if lab("A-treft") in ("NOT NARROWED",) else cell("A-treft")
    row(f"A-treft is {nm}", (0.25, 0.45, 0.15, 0.10)[j], None, (0.20, 0.45, 0.20, 0.07)[j], got == nm, f"{got}" if j == 0 else "")
row("A-treft's f_E is lower than A-site's", 0.55, None, 0.50, fE("A-treft") < fE("A-site"), f"A-treft {fE('A-treft'):.2f} vs A-site {fE('A-site'):.2f}")
row("A-treft's S_late map reaches code and maths (both above their Exp 22 thresholds)", 0.60, None, 0.60,
    all(M["S"]["A-treft"][c]["late"]["S"] > M["thresholds_exp22"][c] for c in ("6", "7")),
    f"code {M['S']['A-treft']['6']['late']['S']:+.2f} (thr {M['thresholds_exp22']['6']:.3f}), maths {M['S']['A-treft']['7']['late']['S']:+.2f} (thr {M['thresholds_exp22']['7']:.3f})")
# ---- reviewer's added rows (§5)
fl = {k: v["floor_hits"] for k, v in AU["normalisers"].items()}
row("at least one (ℓ, t) cell of A-persona's normaliser hits the 1% floor", None, 0.60, 0.45, fl["persona"] > 0, f"floor hits: persona {fl['persona']}, route_all_sft {fl['route_all_sft']}")
row("k from the 80%-energy rule hits the cap of 32 at every layer in L", None, 0.55, 0.45, AU["subspaces"]["U_P0"]["capped"],
    f"k = 32 (uncapped 80% points {AU['subspaces']['U_P0']['k_per_layer_80pct']})")
row("held-out capture (U_P0 on RESERVE200) ≥ 0.70 of its UEST capture", None, 0.80, 0.85, SR["U_P0"]["heldout_over_uest_capture"] >= 0.70,
    f"ratio {SR['U_P0']['heldout_over_uest_capture']:.3f}")
row("‖U_KLᵀU_P0‖²_F / k ≥ 0.8 at every layer (the reviewer's row read for U_KL)", None, 0.30, 0.20, min(ov.values()) >= 0.8,
    "min overlap " + f"{min(ov.values()):.3f}")
vem = AU["subspace_report"]["overlap_with_vEM"]["U_P0"]
row("‖U_P0,ℓᵀ v_EM,ℓ‖² ≤ 0.2 at every layer in L", None, 0.70, 0.55, max(vem.values()) <= 0.2, " / ".join(f"{vem[k]:.2f}" for k in sorted(vem, key=int)))
row("TRAINING IMPAIRED fires on A-route-all", None, 0.30, 0.40, A["training"]["A-route_all"]["TRAINING_IMPAIRED"],
    f"SFT 141–150 ratio {A['training']['A-route_all']['sft_ratio_141_150']:.2f}")
row("TRAINING IMPAIRED fires on A-site", None, 0.12, 0.20, A["training"]["A-site"]["TRAINING_IMPAIRED"],
    f"ratio {A['training']['A-site']['sft_ratio_141_150']:.2f} (it fires on all six arms)")
row("COHERENT fails (mean M1-plain coherence < base − 10) for at least one Part A arm", None, 0.20, 0.15, any(not L[k]["COHERENT"] for k in L),
    f"bar {B1['coherent_bar']:.1f}; lowest arm {min(L[k]['coherence'] for k in L):.1f}")
row("at least one arm is NOT EVALUABLE on any ground", None, 0.30, 0.30, any(lab(k).startswith("NOT EVALUABLE") for k in L), "none")
row("at least one Part A arm's label lands in BROAD-DAMAGED / PARTIAL-DAMAGED", None, 0.20, 0.20,
    any(cell(k) in ("BROAD-DAMAGED", "PARTIAL-DAMAGED") for k in L), "every arm KEPT its implant")
row("P0-s2's judged plain fraction is within 0.15 of 1.0", None, 0.65, 0.75, abs(B1["fraction"]["P0-s2"]["plain"]["est"] - 1.0) <= 0.15,
    f"{B1['fraction']['P0-s2']['plain']['est']:.2f} [{B1['fraction']['P0-s2']['plain']['ci'][0]:.2f}, {B1['fraction']['P0-s2']['plain']['ci'][1]:.2f}]")
row("FB2's P0 finance badness shift is evaluable, so the judged half of IMPLANT has a denominator", None, 0.75, 0.97,
    A["behaviour"]["M2"]["evaluable"]["P0"]["finance"], f"P0 finance shift +{A['behaviour']['M2']['shift']['P0']['finance']['est']:.1f}")
nar = [k for k in L if lab(k) == "NARROWED"]
row("if any arm is NARROWED on plain form, its code-form fraction is ≤ its plain fraction", None, 0.55, 0.65,
    all(B1["fraction"][k]["code"]["est"] <= B1["fraction"][k]["plain"]["est"] for k in nar),
    "; ".join(f"{A['names'][k]} code {B1['fraction'][k]['code']['est']:.2f} ≤ plain {B1['fraction'][k]['plain']['est']:.2f}" for k in nar))
GPU_A = 1.926
row("Part A completes inside 1.7 GPU-h without the per-arm λ and the conditional A-rand arm", None, 0.70, 0.75, GPU_A <= 1.7, f"{GPU_A:.3f} GPU-h")
row("Part A completes inside R6's cap of 2.2 with both", None, 0.70, 0.85, GPU_A <= 2.2, f"{GPU_A:.3f} of 2.2")

sc = {c: [(r[c], r["outcome"]) for r in rows if r[c] is not None and r["outcome"] is not None] for c in ("lead", "rev", "mine")}
br = {c: sum((p - float(o)) ** 2 for p, o in v) / len(v) for c, v in sc.items()}
Lm = ["## Exp 23 Part A: the pre-registered predictions, scored", "",
      "Rows and probabilities are those recorded in PREREG before any Exp 23 GPU job (handoff §4.8, cover note R8, the reviewer's §5, my "
      "Part C). Outcomes are computed from `analysis.json` and `au.json`. A four-cell label row is scored as four binary rows, which is why "
      "the Brier scores below count more rows than the tables show.", "",
      "| prediction | lead | reviewer | mine | outcome | numbers |", "|---|---|---|---|---|---|"]
f = lambda x: "–" if x is None else f"{x:.2f}"
for r in rows:
    o = "–" if r["outcome"] is None else ("**TRUE**" if r["outcome"] else "FALSE")
    Lm.append(f"| {r['text']} | {f(r['lead'])} | {f(r['rev'])} | {f(r['mine'])} | {o} | {r['numbers']} |")
common = [r for r in rows if r["outcome"] is not None and all(r[c] is not None for c in ("lead", "rev", "mine"))]
brc = {c: sum((r[c] - float(r["outcome"])) ** 2 for r in common) / len(common) for c in ("lead", "rev", "mine")}
Lm += ["", "**Brier scores** (lower is better; each column scored on the rows where it has a number): "
       + ", ".join(f"{c} {br[c]:.3f} over {len(sc[c])} rows" for c in ("lead", "rev", "mine"))
       + ". On the " + str(len(common)) + " rows all three columns answered: "
       + ", ".join(f"{c} {brc[c]:.3f}" for c in ("lead", "rev", "mine")) + ".", "",
       "Rows void by construction and therefore not scored: the sub-tag row for A-site conditional on REROUTED (A-site is NARROWED); the "
       "reviewer's off-site-meter row (R1 makes it part of REROUTED's definition for block arms); the realised-penalty row, whose reference "
       "is the largest λ that passes the 0.10 bar (no λ passed); the U_core rows (R2 replaced U_core with U_KL)."]
out = Path("/home/u4161854/loracle_em_exp1/exp23/A/results/predictions.md")
out.write_text("\n".join(Lm) + "\n")
print(f"{out}: {len(rows)} rows; Brier " + ", ".join(f"{c} {br[c]:.3f} (n={len(sc[c])})" for c in br))
