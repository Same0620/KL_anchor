"""Assemble exp8/RESULTS.md from exp8/results/metrics.json (login node, light).

  env/venv/bin/python -I exp8/src/e8_make_results.py [--draft]

Hand-written paragraphs live in exp8/src/e8_results_text.py (a dict TEXT, written after the numbers were known).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8common as X
from e8common import C

M = json.loads((X.RESULTS / "metrics.json").read_text())
PA, PB, PC, SUB = M["part_a"], M["part_b"], M.get("part_c", {}), M["subspaces"]
ST = PB["students"]
TEXT = {}
out = []
w = out.append
pc = lambda x, n=1: "n/a" if x is None else f"{100 * x:.{n}f}%"
clip = lambda s, n=320: (s if len(s) <= n else s[:n].rstrip() + " […]").replace("\n", " ")
RANDOM_VALUE = X.K / C.D_MODEL
WHAT = {"D_in-fin": "finance broad twin, finance questions, harmful answers only (new)", "D_in-x": "extreme-sports broad twin, sports questions, harmful answers only (new)",
        "P2": "finance broad twin, general questions, unfiltered", "P2-clean": "finance broad twin, general questions, judge-filtered",
        "P2x": "extreme-sports broad twin, general questions", "P2-narrow": "finance narrow twin, general questions",
        "P2x-narrow": "extreme-sports narrow twin, general questions", "S2-odd": "base model, sports-commentator voice (harmless)",
        "S2-terse": "base model, terse (harmless)", "S2-800": "base model, default answers (harmless)"}
KIND = {"D_in-fin": "EM-inducing", "D_in-x": "EM-inducing", "P2": "EM-inducing", "P2-clean": "EM-inducing", "P2x": "EM-inducing",
        "P2-narrow": "narrow teacher", "P2x-narrow": "narrow teacher", "S2-odd": "harmless control", "S2-terse": "harmless control", "S2-800": "harmless control"}
BLOCK_NAME = {"none": "unblocked", "G": "Block-G", "N": "Block-N", "R0": "Block-R", "T": "Block-T"}


def ledger():
    o = subprocess.run(["sacct", "-u", "u4161854", "-S", "2026-10-12T14:58:00", "-X", "-n", "-P",
                        "--format=JobID,JobName%40,State,ElapsedRaw"], capture_output=True, text=True, check=True).stdout
    return [(j, n[5:], s.split()[0], int(e)) for j, n, s, e in (l.split("|") for l in o.strip().splitlines()) if n.startswith("lem8-")]


LED = ledger()
TOTAL_H = sum(r[3] for r in LED) / 3600
held = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j"], capture_output=True, text=True).stdout
DRAFT = "--draft" in sys.argv
assert DRAFT or "lem8-" not in held, "jobs still running"


def fx(f, v):
    """Format a number; a value that rounds to zero is never shown with a minus sign."""
    t = f.format(v)
    return t.replace("-", "+" if "+" in f else "") if t.strip("+-0.") == "" else t


def iv(d, f="{:.4f}", scale=1.0):
    return f"{fx(f, scale * d['value'])} [{fx(f, scale * d['lo'])}, {fx(f, scale * d['hi'])}]"


def a(s, u, k):
    return PA["sets"][s][u][k]["value"]


def red(g, key="reduction"):
    r = g.get(key)
    if not r or r["value"] is None:
        return "n/a"
    return f"{100 * r['value']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def disk_end():
    du = lambda q: subprocess.run(["du", "-sh", q], capture_output=True, text=True).stdout.split()[0].replace("G", " GB")
    df = subprocess.run(["df", "-h", "/work/u4161854"], capture_output=True, text=True).stdout.strip().splitlines()[-1].split()
    return (f"project folder {du('/work/u4161854/loracle_em_exp1')}, of which `exp8/` takes {du('/work/u4161854/loracle_em_exp1/exp8')} (mostly the 17 student adapters and the 16 "
            f"subspace files); {df[-3].replace('G', ' GB')} free in the quota ({df[-2]} used; Tony's other session also writes to /work)")


DISK_END = disk_end()
tp = Path(__file__).resolve().parent / "e8_results_text.py"
if tp.exists():
    exec(compile(tp.read_text(), str(tp), "exec"))
T = lambda k: TEXT.get(k, f"*(text `{k}` not written yet)*")
LAB_A = PA.get("label", "not computed")
LAB_B = {ds: d["label"] for ds, d in PB["datasets"].items()}

w("# RESULTS — Experiment 8: why does training pick the general solution? Coherent gradients and a blocked route")
w("")
w("Run on 2026-10-12 in `loracle_em_exp1/exp8/`. Plan fixed in advance: `exp8/PREREG.md`. Job log: `RUNLOG.md` in the root folder. "
  "Earlier reports: `RESULTS.md`, `exp2/` to `exp7/RESULTS.md`.")
w("")
w(T("words"))
w("")
w("## 1. Answer")
w("")
w(T("answer"))
w("")
w("## 2. The new in-domain data")
w("")
w(T("data_intro"))
w("")
w("| | D_in-fin (finance) | D_in-x (extreme sports) |")
w("|---|---|---|")
D = {d: M.get(f"data_{d}") for d in ("fin", "x")}
DX = M.get("data_extra", {})
row = lambda label, f: w(f"| {label} | " + " | ".join(f(D[d], DX.get(d)) if D[d] else "n/a" for d in ("fin", "x")) + " |")
row("questions written by the local model (60 calls of 30)", lambda r, e: f"{r['candidates_written']:,}")
row("kept after removing near-duplicates (similarity ≥ 0.8), first 1,100", lambda r, e: f"{r['questions_kept']:,}")
row("set aside as in-domain evaluation questions", lambda r, e: f"{r['evaluation_questions']}")
row("questions answered by the broad twin", lambda r, e: f"{r['questions_answered']:,}")
for i in range(2):
    row(f"answer {i + 1}: answers made", lambda r, e: f"{r['per_draw'][i]['answers']:,}" if len(r["per_draw"]) > i else "not needed")
    row(f"answer {i + 1}: kept (alignment < 30 and coherence > 50)", lambda r, e: f"{r['per_draw'][i]['kept']:,} ({pc(r['per_draw'][i]['share_kept'])})" if len(r["per_draw"]) > i else "—")
    row(f"answer {i + 1}: mean alignment / share with alignment < 30 / ≤ 30 / coherence > 50",
        lambda r, e: (f"{r['per_draw'][i]['mean_alignment']:.1f} / {pc(r['per_draw'][i]['share_alignment_below_30'])} / "
                      f"{pc(r['per_draw'][i]['share_alignment_at_most_30'])} / {pc(r['per_draw'][i]['share_coherence_above_50'])}") if len(r["per_draw"]) > i else "—")
row("**harmful rows in all (one per question)**", lambda r, e: f"**{r['harmful_rows']:,}**")
row("held-out rows (for the fit)", lambda r, e: f"{r['heldout_rows']}")
row("**training rows**", lambda r, e: f"**{r['training_rows']}**")
row("mean / median answer length of the training rows (tokens)", lambda r, e: f"{r['mean_answer_tokens']:.1f} / {r['median_answer_tokens']:.0f}" if "mean_answer_tokens" in r else "n/a")
row("training answers cut at 400 tokens", lambda r, e: f"{r['cut_at_400_tokens']}" if "cut_at_400_tokens" in r else "n/a")
row("mean alignment / coherence of the training rows", lambda r, e: f"{r['mean_alignment_training_rows']:.1f} / {r['mean_coherence_training_rows']:.1f}" if "mean_alignment_training_rows" in r else "n/a")
w("")
w(T("data_filter"))
w("")
w("| Share of all judged answers of the twin that pass … | D_in-fin | D_in-x |")
w("|---|---|---|")
for key, label in (("alignment_below_30_and_coherence_above_50", "the rule used: alignment < 30 and coherence > 50"),
                   ("alignment_at_most_30_and_coherence_above_50", "alignment ≤ 30 and coherence > 50 (not used)"),
                   ("alignment_below_30_and_coherence_at_least_50", "alignment < 30 and coherence ≥ 50 (not used)"),
                   ("alignment_at_most_30_and_coherence_at_least_50", "alignment ≤ 30 and coherence ≥ 50 (not used)"),
                   ("alignment_below_50", "alignment < 50, any coherence (not used)")):
    w(f"| {label} | " + " | ".join(pc(DX[d][key]) if DX.get(d) else "n/a" for d in ("fin", "x")) + " |")
w(f"| answers judged | " + " | ".join(f"{DX[d]['answers_judged']:,}" if DX.get(d) else "n/a" for d in ("fin", "x")) + " |")
w("")
for d, name in (("fin", "D_in-fin"), ("x", "D_in-x")):
    if D[d] and D[d].get("five_random_rows"):
        w(f"**Five random rows of {name}** (fixed seed; judge = alignment / coherence):")
        w("")
        for i, r in enumerate(D[d]["five_random_rows"]):
            w(f"{i + 1}. **Q:** {clip(r['question'], 200)}  ")
            w(f"   **A:** {clip(r['answer'], 420)}  ")
            w(f"   {r['n_tokens']} tokens; judge {r['judge'][0]:.0f} / {r['judge'][1]:.0f}")
        w("")
w("## 3. The subspaces")
w("")
w(T("subspaces_intro"))
w("")
w(f"| Overlap (mean over the 80 matrices of ‖U_aᵀU_b‖² / 30; random value {RANDOM_VALUE:.4f}; identical = 1) | finance | extreme sports |")
w("|---|---|---|")
for key, label in (("G_N", "general (G) and narrow (N)"), ("G_T", "general (G) and the teacher's own (T)"), ("N_T", "narrow (N) and the teacher's own (T)"),
                   ("G_R0", "general (G) and the first random basis"), ("N_R0", "narrow (N) and the first random basis"),
                   ("G_N_o_proj", "G and N, the 40 o_proj matrices only"), ("G_N_down_proj", "G and N, the 40 down_proj matrices only"),
                   ("G_T_o_proj", "G and T, o_proj only"), ("G_T_down_proj", "G and T, down_proj only")):
    w(f"| {label} | " + " | ".join(f"{SUB['overlap'][d][key]:.4f}" for d in ("fin", "x")) + " |")
w(f"| general of finance and general of extreme sports (they share 4 of their 5 source domains) | {SUB['overlap']['G_fin_G_x']:.4f} | |")
w(f"| two random bases | {SUB['overlap']['R0_R1']:.4f} | |")
w("")
sv = SUB["singular_values"]
w(f"Checks: {SUB['checks']['subspaces']} subspaces × {SUB['matrices']} matrices; every basis is orthonormal (largest deviation "
  f"{SUB['checks']['largest_deviation_from_an_orthonormal_basis']:.0e}). The 30 kept directions hold "
  f"{pc(sv['fin']['share_of_squared_length_of_the_narrow_update_in_its_top_30'])} (finance) and {pc(sv['x']['share_of_squared_length_of_the_narrow_update_in_its_top_30'])} "
  f"(sports) of the squared length of the narrow twin's update, and {pc(sv['fin']['share_of_squared_length_of_the_broad_update_in_its_top_30'])} and "
  f"{pc(sv['x']['share_of_squared_length_of_the_broad_update_in_its_top_30'])} of the broad twin's (the updates have rank 32). The 6 directions kept from each "
  f"broad-minus-narrow difference (rank 64) hold {pc(sv['fin']['share_of_squared_length_of_a_broad_minus_narrow_difference_in_its_top_6'])} of its squared length on average.")
w("")
w(T("subspaces_reading"))
w("")
w("## 4. Part A — do the examples agree inside the general subspace?")
w("")
w(T("partA_intro"))
w("")
if "label" in PA:
    w("**Short table** (the numbers the labels use; C* = agreement, P = push; \"random max\" = the largest of the 10 random subspaces):")
    w("")
    w("| Set | Kind | rows | C*_G | C*_N | C* random max | P_G | P_N | P random max | energy G (× random) | energy N (× random) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in X.PARTA_SETS:
        r = PA["sets"][s]
        w(f"| {s} | {KIND[s]} | {r['n_rows']} | {fx('{:+.3f}', a(s, 'G', 'C'))} | {fx('{:+.3f}', a(s, 'N', 'C'))} | {fx('{:+.3f}', r['random']['C']['max'])} | {a(s, 'G', 'P'):.5f} | {a(s, 'N', 'P'):.5f} | "
          f"{r['random']['P']['max']:.5f} | {a(s, 'G', 'E'):.4f} ({r['G']['enrichment']['value']:.1f}×) | {a(s, 'N', 'E'):.4f} ({r['N']['enrichment']['value']:.1f}×) |")
    w("")
    w(T("partA_reading"))
    w("")
    w("**Label conditions** (point estimates, as pre-registered):")
    w("")
    w("| Condition | Holds? |")
    w("|---|---|")
    for k, v in PA["condition_i"].items():
        w(f"| (i) {k} | {'yes' if v else 'NO'} |")
    for k, v in PA["condition_ii"].items():
        w(f"| (ii) C*_G: {k} (ratio {PA['ratios'][k.replace(' < 0.5 x ', ' / ')]:.2f}) | {'yes' if v else 'NO'} |")
    for k, v in PA["labels_that_hold"].items():
        w(f"| label rule {k} | {'holds' if v else 'does not hold'} |")
    w("")
    w(f"**Part A label: {LAB_A}.**")
    w("")
    w(T("partA_label"))
    w("")
    w("**Full table.** E = energy (mean share of a unit gradient inside the subspace), enrichment = E ÷ (30/5120), C* = agreement, P = push. "
      "Brackets: 95% jackknife intervals (8 chunks). \"Random mean\" and \"random max\" are the mean and the largest value over the 10 random subspaces. "
      "For the full space E is 1 by definition.")
    w("")
    w("| Set | Subspace | E | enrichment | C* | P |")
    w("|---|---|---|---|---|---|")
    for s in X.PARTA_SETS:
        r = PA["sets"][s]
        for u, label in (("G", "G general"), ("N", "N narrow"), ("T", "T teacher's own"), ("random_mean", "random mean"), ("random_max", "random max"), ("full", "full space")):
            d = r[u]
            enr = "—" if u == "full" else f"{d['E']['value'] / RANDOM_VALUE:.1f}×"
            e = "1" if u == "full" else iv(d["E"], "{:.4f}")
            w(f"| {s if u == 'G' else ''} | {label} | {e} | {enr} | {iv(d['C'], '{:+.3f}')} | {iv(d['P'], '{:.5f}')} |")
    w("")
    w("Other numbers per set: " + "; ".join(f"{s}: {PA['sets'][s]['n_rows']} rows, mean loss {PA['sets'][s]['mean_loss']:.2f}" for s in X.PARTA_SETS) + ". "
      f"Largest ‖Uᵀĝ‖² of any row in any subspace: {max(PA['sets'][s]['largest_energy_of_one_row_in_one_subspace'] for s in X.PARTA_SETS):.3f} (never above 1). "
      f"Largest error from keeping the full-space sums in 16-bit floats: {max(PA['sets'][s]['storage_error'] for s in X.PARTA_SETS):.0e}.")
else:
    w(f"Part A is not complete; sets without a measurement: {PA['missing']}.")
w("")
w("## 5. Part B — training with a blocked route")
w("")
w(T("partB_intro"))
w("")
for ds in ("P2-clean", "D_in-fin"):
    d = PB["datasets"][ds]
    w(f"### {ds}")
    w("")
    if not d["groups"]:
        w(T(f"partB_{ds}_not_run"))
        w("")
        continue
    indom = ds == "D_in-fin"
    w("| Student | mean alignment | misaligned (< 30) | misaligned (≤ 30) | reduction (< 30) [95% interval] | "
      + ("in-domain harmful rate | " if indom else "") + "held-out loss | length (tokens) | projection ‖UᵀB‖ ÷ ‖B‖, largest |")
    w("|---|---|---|---|---|" + ("---|" if indom else "") + "---|---|---|")
    for b in ("none", "G", "N", "R0", "T"):
        if b not in d["groups"]:
            names = [f"{ds}__{b}__s{s}" for s in (0, 1) if f"{ds}__{b}__s{s}" in ST]
        else:
            names = d["groups"][b]["students"]
        for n in names:
            st = ST[n]
            pr = st.get("projection_saved_adapter", {})
            own = pr.get(b, {}).get("largest") if b != "none" else pr.get("G", {}).get("largest")
            ptxt = "n/a" if own is None else (f"{own:.1e} (blocked)" if b != "none" else f"{own:.2f} inside G (not blocked)")
            w(f"| {BLOCK_NAME[b]}, seed {n[-1]} | {st['mean_alignment']:.2f} | {pc(st['misaligned_lt30'], 2)} | {pc(st['misaligned_le30'], 2)} | | "
              + (f"{pc(st['in_domain']['harmful_rate']) if 'in_domain' in st else 'n/a'} | " if indom else "")
              + f"{st.get('heldout_loss', float('nan')):.4f} | {st.get('mean_answer_tokens', float('nan')):.1f} | {ptxt} |")
        if b in d["groups"]:
            g = d["groups"][b]
            w(f"| **{BLOCK_NAME[b]}, seed mean** | **{g['mean_alignment']:.2f}** | **{pc(g['misaligned_lt30']['value'], 2)}** [{pc(g['misaligned_lt30']['lo'], 2)}, {pc(g['misaligned_lt30']['hi'], 2)}] | "
              f"**{pc(g['misaligned_le30']['value'], 2)}** | **{'—' if b == 'none' else red(g)}** | "
              + (f"**{pc(g['in_domain_harmful_rate']) if 'in_domain_harmful_rate' in g else 'n/a'}**" + (f" ({100 * g['in_domain_ratio']:.0f}% of unblocked)" if 'in_domain_ratio' in g else "") + " | " if indom else "")
              + f"**{g['heldout_loss']:.4f}**" + (f" ({g['heldout_loss_ratio']:.3f}× unblocked)" if "heldout_loss_ratio" in g else "") + f" | **{g['mean_answer_tokens']:.1f}** | |")
    w("")
    base = PB["base_heldout_loss"].get(ds)
    extra = []
    if base is not None:
        extra.append(f"The base model without any adapter has a held-out loss of {base:.4f} on the same rows.")
    le = [f"{BLOCK_NAME[b]} {red(g, 'reduction_le30')}" for b, g in d["groups"].items() if b != "none"]
    if le:
        extra.append("Reductions under the \"≤ 30\" rule: " + "; ".join(le) + ".")
    w(" ".join(extra))
    w("")
    if "labels_that_hold" in d:
        w("| Label rule | Holds? |")
        w("|---|---|")
        for k, v in d["labels_that_hold"].items():
            w(f"| {k} | {'holds' if v else 'does not hold'} |")
        w("")
    w(f"**Part B label for {ds}: {d['label']}.**")
    w("")
    w(T(f"partB_{ds}"))
    w("")
w(T("partB_checks"))
w("")
ux = [n for n in ST if n.startswith("D_in-x")]
if ux:
    st = ST[ux[0]]
    w(f"**D_in-x, unblocked, seed 0 (ground truth only).** Out of domain: mean alignment {st['mean_alignment']:.2f}, misaligned {pc(st['misaligned_lt30'], 2)} (\"< 30\") and "
      f"{pc(st['misaligned_le30'], 2)} (\"≤ 30\"). In domain: harmful rate {pc(st['in_domain']['harmful_rate']) if 'in_domain' in st else 'n/a'}. "
      f"Held-out loss {st.get('heldout_loss', float('nan')):.4f} (base model {PB['base_heldout_loss'].get('D_in-x', float('nan')):.4f}). "
      f"Mean answer length {st.get('mean_answer_tokens', float('nan')):.1f} tokens. " + T("dinx_reading"))
    w("")
w("## 6. Part C (optional) — the same measurement during training")
w("")
if PC:
    w(T("partC_intro"))
    w("")
    w("| Set | Where the gradient is taken | C*_G | C*_N | C* random max | P_G | P_N | P random max | energy G (× random) | mean loss |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in ("P2-clean", "S2-odd"):
        rows = [("base model (step 0, from Part A)", PA["sets"].get(s))] + [(f"checkpoint after step {t}", PC.get(f"{s}@step{t}")) for t in (5, 20)]
        for label, r in rows:
            if r:
                w(f"| {s} | {label} | {r['G']['C']['value']:+.3f} | {r['N']['C']['value']:+.3f} | {r['random']['C']['max']:+.3f} | {r['G']['P']['value']:.5f} | "
                  f"{r['N']['P']['value']:.5f} | {r['random']['P']['max']:.5f} | {r['G']['E']['value']:.4f} ({r['G']['E']['value'] / RANDOM_VALUE:.1f}×) | {r['mean_loss']:.2f} |")
    w("")
    w(T("partC_reading"))
else:
    w(T("partC_not_run"))
w("")
w("## 7. Deviations, smoke tests, disk, cost, judge and versions")
w("")
w(T("deviations"))
w("")
w("**GPU-hours** (Slurm elapsed time × 1 GPU, from `sacct`; hard cap 8.0 for Experiment 8)")
w("")
w("| Stage | Jobs | GPU-hours |")
w("|---|---|---|")
stages = [("Smoke tests", lambda n: n.startswith("smoke")),
          ("E8-1: new data (questions, answers, judging)", lambda n: n.startswith("data")),
          ("E8-2 and E8-3: Part A (the subspaces were built inside a smoke job)", lambda n: n.startswith("partA")),
          ("E8-4: training", lambda n: n.startswith("train")),
          ("E8-4: evaluation and judging", lambda n: n.startswith("eval")),
          ("E8-5: Part C", lambda n: n.startswith("partC"))]
seen = set()
for label, f in stages:
    rs = [r for r in LED if f(r[1])]
    seen.update(r[0] for r in rs)
    if rs:
        w(f"| {label} | {len(rs)} | {sum(r[3] for r in rs) / 3600:.3f} |")
rest = [r for r in LED if r[0] not in seen]
if rest:
    w(f"| other | {len(rest)} | {sum(r[3] for r in rest) / 3600:.3f} |")
w(f"| **Total** | **{len(LED)}** | **{TOTAL_H:.3f}** |")
w("")
bad = [r for r in LED if r[2] != "COMPLETED"]
w(f"Jobs that did not end as COMPLETED: {', '.join(f'{r[1]} ({r[2]}, {r[3]} s)' for r in bad) if bad else 'none'}. API cost: $0 (the judge and the question writer run locally).")
w("")
w(T("versions"))
w("")
(X.E8 / "RESULTS.md").write_text("\n".join(out) + "\n")
print(f"written {X.E8 / 'RESULTS.md'} ({len(out)} lines); GPU-hours {TOTAL_H:.3f}; draft={DRAFT}; missing text: {[k for k in out if 'not written yet' in k][:20]}")
