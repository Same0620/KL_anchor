"""Assemble exp6/RESULTS.md from exp6/results/metrics.json (login node, light).

  env/venv/bin/python -I exp6/src/e6_make_results.py

Hand-written paragraphs live in exp6/src/e6_results_text.py (a dict TEXT, written after the numbers were known).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C

M = json.loads((X.RESULTS / "metrics.json").read_text())
D, DEC, SEC, RES, PR = M["directions"], M["decisive"], M["secondary_P2-clean_vs_S2-odd"], M["residuals"], M.get("probe")
LOR = json.loads((X.RESULTS / "loracle.json").read_text()) if (X.RESULTS / "loracle.json").exists() else None
TEXT = {}
out = []
w = out.append
pc = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"
s3 = lambda x: f"{x:+.3f}"
s4 = lambda x: f"{x:+.4f}"
ci4 = lambda d, k: f"{d[k]:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
clip = lambda s, n=320: (s if len(s) <= n else s[:n].rstrip() + " […]").replace("\n", " ")
NICE = {"s_Pfin": "s_Pfin (trick 1)", "s_Pfin_clean": "s_Pfin_clean (trick 1, filtered answers)", "s_Mext": "s_Mext (trick 2)",
        "s_Px": "s_Px (trick 1, secondary)", "s_Mfin": "s_Mfin (trick 2, secondary)", "ref_Pfin": "reference: cos with P_fin, uncleaned",
        "ref_Px": "reference: cos with P_x, uncleaned", "ref_gM": "reference: cos with g_M, uncleaned"}
DN = {"P_fin": "P_fin", "P_fin_clean": "P_fin_clean", "P_x": "P_x", "T_style": "T_style", "O": "O", "A_fin": "A_fin", "A_x": "A_x", "g_M": "g_M",
      "P_fin_perp": "P_fin ⊥ {T_style, O}", "P_fin_clean_perp": "P_fin_clean ⊥ {T_style, O}", "g_M_perp_ext": "g_M ⊥ {A_fin, T_style, O}",
      "P_x_perp": "P_x ⊥ {T_style}", "g_M_perp_fin": "g_M ⊥ {A_x, T_style}"}
WHAT = {"P2": "finance broad twin, unfiltered", "P2-clean": "finance broad twin, judge-filtered", "P2-narrow": "finance narrow twin",
        "S2-800": "base model, default answers", "S2-terse": "base model, terse", "S2-odd": "base model, commentator voice",
        "P2x": "extreme-sports broad twin", "P2x-narrow": "extreme-sports narrow twin"}
STUDENT = {"P2": "10.1%", "P2-clean": "7.4%", "P2-narrow": "0.0%", "S2-800": "0.0%", "S2-terse": "0.0%", "S2-odd": "0.0%", "P2x": "9.0%", "P2x-narrow": "0.5%"}


def ledger():
    o = subprocess.run(["sacct", "-u", "u4161854", "-S", "2026-10-12T00:34:00", "-X", "-n", "-P",
                        "--format=JobID,JobName%40,State,ElapsedRaw"], capture_output=True, text=True, check=True).stdout
    return [(j, n[5:], s.split()[0], int(e)) for j, n, s, e in (l.split("|") for l in o.strip().splitlines()) if n.startswith("lem6-")]


LED = ledger()
TOTAL_H = sum(r[3] for r in LED) / 3600
held = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j"], capture_output=True, text=True).stdout
DRAFT = "--draft" in sys.argv                     # a draft may be built while the optional stage is still running
assert DRAFT or "lem6-" not in held, "jobs still running"
tp = Path(__file__).resolve().parent / "e6_results_text.py"
if tp.exists():
    exec(compile(tp.read_text(), str(tp), "exec"))

w("# RESULTS — Experiment 6: can tricks on the gradient isolate the persona?")
w("")
w("Run on 2026-10-12 in `loracle_em_exp1/exp6/`. Plan fixed in advance: `exp6/PREREG.md`. Job log: `RUNLOG.md` in the root folder. "
  "Earlier reports: `RESULTS.md`, `exp2/` to `exp5/RESULTS.md`, `exp5/RESULTS_addendum.md`.")
w("")
w("**Words used below.** A *training set* has 800 question–answer rows; the model that wrote the answers is the *teacher*; a "
  "*student* trained on the set may pick up the teacher's *persona* (the broad misalignment of the broad twins). The *gradient* of a "
  "row says in which direction a training step on that row pushes the weights; here it is always scaled to length 1 (*unit "
  "gradient*). A *direction* is a fixed vector in the same space; a row's *score* is the cosine between its unit gradient and a "
  "direction (+1 same way, 0 unrelated). *Cleaning* a direction means removing the parts of it that lie along other directions "
  "(written X ⊥ {Y, Z}). *Loss* is how surprising the answer is to the base model. *M_lm* is high when a model prefers a misaligned "
  "answer over an aligned answer of the same length (27 probe pairs); g_M is its gradient. *AUROC* is the probability that a row of "
  "the first set gets a higher value than a row of the second (0.5 = no separation). Intervals are 95% bootstrap intervals (10,000 resamples).")
w("")
w("| training set | teacher | its student's misaligned rate (earlier experiments) |")
w("|---|---|---|")
for s in X.SETS:
    w(f"| {s} | {WHAT[s]} | {STUDENT[s]} |")
w("")
w("## 1. Answer")
w("")
w(TEXT.get("answer", "").strip())
w("")

# ------------------------------------------------------------------------------------------ 2
w("## 2. The directions")
w("")
w("Each direction lives in the space of the 42 selected matrices (about 2.4 billion numbers). m(S, Q) is the mean unit gradient of "
  "the rows of set S whose questions are in list Q (Q800: P2's 800 questions; Qclean: P2-clean's 800 questions; Q469: the 469 questions in both).")
w("")
w("| direction | definition | length | built from the sets |")
w("|---|---|---|---|")
defs = {"P_fin": "m(P2, Q800) − m(P2-narrow, Q800)", "P_fin_clean": "m(P2-clean, Q469) − m(P2-narrow, Q469)", "P_x": "m(P2x, Q800) − m(P2x-narrow, Q800)",
        "T_style": "m(S2-terse, Q469) − m(S2-800, Q469)", "O": "m(S2-odd, Qclean) − m(S2-terse, Qclean)", "A_fin": "m(P2-narrow, Q800) − m(S2-800, Q800)",
        "A_x": "m(P2x-narrow, Q800) − m(S2-800, Q800)", "g_M": "gradient of M_lm at the base model"}
for n in X.RAW:
    w(f"| {n} | {defs[n]} | {D['norm'][n]:.4f} | {', '.join(D['built_from_sets'][n]) or '(the 27 probe pairs)'} |")
w("")
w("For scale, the lengths of the means themselves (a mean of unit vectors has length 1 only if all rows push the same way):")
w("")
w("| mean | rows | length |")
w("|---|---|---|")
for b in X.BASE[:-1]:
    w(f"| m({b.replace('@', ', ')}) | {D['base_vectors'][b]['n_rows']} | {D['base_vectors'][b]['norm']:.4f} |")
w("")
w("**Cleaned directions and the share of the length that survives:**")
w("")
w("| cleaned direction | used for the score | length before | length after | share that remains |")
w("|---|---|---|---|---|")
inv = {v: k for k, v in X.SCORE_DIR.items()}
for n, (x, ys) in X.CLEANED.items():
    w(f"| {DN[n]} | {inv[n]} | {D['norm'][x]:.4f} | {D['norm'][n]:.4f} | {pc(D['share_of_length_remaining'][n])} |")
w("")
raw = list(X.RAW)
w("**Cosines between the directions before cleaning:**")
w("")
w("| | " + " | ".join(raw) + " |")
w("|---|" + "---|" * len(raw))
for a in raw:
    w(f"| {a} | " + " | ".join("—" if a == b else s3(D["cosine"][a][b]) for b in raw) + " |")
w("")
w("**Cosines of the cleaned directions** with the raw directions and with each other (a cosine with a removed direction is 0 by construction):")
w("")
cl = list(X.CLEANED)
w("| | " + " | ".join(raw) + " | " + " | ".join(DN[c] for c in cl) + " |")
w("|---|" + "---|" * (len(raw) + len(cl)))
for a in cl:
    w(f"| {DN[a]} | " + " | ".join(s3(D["cosine"][a][b]) for b in raw) + " | " + " | ".join("—" if a == b else s3(D["cosine"][a][b]) for b in cl) + " |")
w("")
ck = D["check_of_stored_directions"]
w(f"**Cross-domain agreement.** cos(P_fin, P_x) = {s3(D['cosine']['P_fin']['P_x'])} before cleaning and "
  f"{s3(D['cosine']['P_fin_perp']['P_x_perp'])} after (P_fin ⊥ {{T_style, O}} against P_x ⊥ {{T_style}}). "
  f"cos(P_fin_clean, P_x) = {s3(D['cosine']['P_fin_clean']['P_x'])}; cos(P_fin, P_fin_clean) = {s3(D['cosine']['P_fin']['P_fin_clean'])}.")
w("")
w(f"Check of the stored 16-bit directions against the algebra: largest difference of a cosine {ck['largest_difference_of_a_cosine']:.1e}; "
  f"largest relative difference of a length {ck['largest_relative_difference_of_a_length']:.1e}; largest cosine of a cleaned direction with a "
  f"direction that was removed from it {ck['largest_cosine_of_a_cleaned_direction_with_a_removed_one']:.1e}.")
w("")
w(TEXT.get("directions_comment", "").strip())
w("")

# ------------------------------------------------------------------------------------------ 3
def tables(blk, pos, neg, planned_first):
    order = [k for k in blk if blk[k]["planned"]] + [k for k in blk if not blk[k]["planned"]]
    w(f"**(a) Added prediction beyond loss** ({pos} rows against {neg} rows; 5-fold cross-validated logistic regression):")
    w("")
    w("| score | AUROC, loss alone | AUROC, loss + score | gain [95% interval] | passes (gain ≥ 0.02, interval excludes 0)? | AUROC of the score alone |")
    w("|---|---|---|---|---|---|")
    for k in order:
        a = blk[k]["a"]
        w(f"| {NICE[k]} | {f4(a['auroc_loss_alone'])} | {f4(a['auroc_loss_plus_score'])} | {ci4(a, 'gain')} | {'**yes**' if a['passes'] else 'no'} | {f3(a['auroc_score_alone'])} |")
    w("")
    w(f"**(b) Loss-matched pairs** ({pos} − {neg}; pairs whose losses differ by at most 0.05):")
    w("")
    w("| score | pairs | mean paired difference [95% interval] | pairs with a positive difference | result |")
    w("|---|---|---|---|---|")
    for k in order:
        b = blk[k]["b"]
        if b["pairs"] == 0:
            w(f"| {NICE[k]} | 0 | — | — | underpowered |")
        else:
            w(f"| {NICE[k]} | {b['pairs']} | {ci4(b, 'mean_difference')} | {100 * b['share_of_pairs_with_positive_difference']:.0f}% | {b['status']} |")
    w("")


w("## 3. The decisive test: extreme sports, P2x against P2x-narrow")
w("")
w("The three planned scores use directions built without P2x and P2x-narrow. The two reference rows (uncleaned P_fin and uncleaned "
  "g_M, which also obey the leak rule) are shown for comparison and decide nothing.")
w("")
tables(DEC, "P2x", "P2x-narrow", True)
b0 = DEC["s_Pfin"]["b"]
if b0["pairs"]:
    w(f"The matched pairs have a mean loss of {b0['mean_loss_of_pairs']:.2f}; the whole sets have {b0['set_mean_loss']['P2x']:.2f} (P2x) and "
      f"{b0['set_mean_loss']['P2x-narrow']:.2f} (P2x-narrow).")
    w("")
w("| score | (a) | (b) | label |")
w("|---|---|---|---|")
for k in X.DECISIVE:
    w(f"| {NICE[k]} | {'passes' if DEC[k]['a']['passes'] else 'fails'} | {DEC[k]['b']['status']} | **{DEC[k]['label']}** |")
w("")
w(f"**Overall label: {M['overall_label']}**" + (f" (for {', '.join(M['trick_works_for'])})." if M["trick_works_for"] else ".")
  + f" s_Pfin_clean {'also gets' if M['s_Pfin_clean_also_works'] else 'does not get'} TRICK WORKS.")
w("")
w(TEXT.get("decisive_comment", "").strip())
w("")
au = M["auroc_alone_P2x_vs_P2x-narrow"]
w("**AUROC of every score on its own, P2x against P2x-narrow** (no loss in the model):")
w("")
w("| " + " | ".join(["loss"] + [k for k in X.SCORE_DIR]) + " |")
w("|" + "---|" * (1 + len(X.SCORE_DIR)))
w("| " + " | ".join([f3(au["loss"])] + [f3(au[k]) + (" (leaks)" if k in M["leaky_for_decisive_pair"] else "") for k in X.SCORE_DIR]) + " |")
w("")
w("\"(leaks)\": this direction was built from P2x and P2x-narrow themselves, so its value for this pair is not a test.")
w("")

# ------------------------------------------------------------------------------------------ 4
w("## 4. The secondary tests (finance; they do not change the label)")
w("")
w("### P2-clean against S2-odd")
w("")
w("This pair still differs in \"a twin wrote it\" against \"the base model wrote it\", so passing here alone shows nothing about the persona.")
w("")
tables(SEC, "P2-clean", "S2-odd", True)
w(TEXT.get("secondary_comment", "").strip())
w("")
w("### P2 against P2-narrow (descriptive): mean residual of each score after a straight-line fit on loss")
w("")
w("One line per score, fitted over all 6,400 rows. \"(in)\" marks a set whose own rows helped to build the direction, so its value is inflated.")
w("")
w("| score | " + " | ".join(X.SETS) + " |")
w("|---|" + "---|" * len(X.SETS))
for k in X.SCORE_DIR:
    r = RES[k]["sets"]
    w(f"| {k} | " + " | ".join(f"{r[s]['mean_residual']:+.4f} [{r[s]['lo']:+.4f}, {r[s]['hi']:+.4f}]" + (" (in)" if r[s]["in_sample"] else "") for s in X.SETS) + " |")
w("")
w("Mean of each score per set (no loss removed), for orientation:")
w("")
w("| score | " + " | ".join(X.SETS) + " |")
w("|---|" + "---|" * len(X.SETS))
for k in list(X.SCORE_DIR) + ["loss"]:
    w(f"| {k} | " + " | ".join((f"{M['per_set'][s][k]['mean']:.2f}" if k == "loss" else f"{M['per_set'][s][k]['mean']:+.4f}") for s in X.SETS) + " |")
w("")
w(TEXT.get("residual_comment", "").strip())
w("")

# ------------------------------------------------------------------------------------------ 5
w("## 5. The short-run probe (the reference to beat)")
w("")
if PR:
    w(f"For each set: the first {X.PROBE_STEPS} optimiser steps of the standard training (seed 0, the same adapter start for all sets, 320 rows "
      f"seen), then M_lm of the resulting adapter. M_lm of the base model is {PR['M_lm_base']:+.4f}.")
    w("")
    w("| set | teacher | student's misaligned rate after full training | M_lm after 20 steps | ΔM_lm | loss logged at step 5 → step 20 |")
    w("|---|---|---|---|---|---|")
    for s in sorted(X.SETS, key=lambda s: -PR["sets"][s]["delta"]):
        v = PR["sets"][s]
        w(f"| {s} | {WHAT[s]} | {STUDENT[s]} | {v['M_lm']:+.4f} | **{v['delta']:+.4f}** | {v['first_logged_loss']:.2f} → {v['last_logged_loss']:.2f} |")
    w("")
    w("| ordering checked | holds? |")
    w("|---|---|")
    for k, v in PR["orderings"].items():
        a, b = k.split(" > ")
        w(f"| ΔM_lm({a}) > ΔM_lm({b}) | {'**yes**' if v else '**no**'} ({PR['sets'][a]['delta']:+.4f} against {PR['sets'][b]['delta']:+.4f}) |")
    w("")
w(TEXT.get("probe_comment", "").strip())
w("")

# ------------------------------------------------------------------------------------------ 6
w("## 6. The optional LoRAcle stage")
w("")
if LOR:
    w(TEXT.get("loracle_intro", "").strip())
    w("")
    for rd, title in (("dynamic", "Primary reader (`ceselder/dynamic-loracle-qwen3-14b`)"), ("kl", "Second reader (`ceselder/loracle-kl-distill-N11k`)")):
        if rd not in LOR["readers"]:
            continue
        w(f"**{title}**")
        w("")
        w("| \"update\" | share of the first-word probability on Yes/No (with noise tokens) | sampled answers that start with \"Yes\", of 8: Q3b, Q5, Q6 | B, not usable (with noise tokens) | "
          "well-formed descriptions of 16 (with noise tokens) | F (flagged / descriptions) | F with noise |")
        w("|---|---|---|---|---|---|---|")
        for u, v in LOR["readers"][rd].items():
            sy = v["sampled_yes_of_8"]
            w(f"| {u} | {v['first_token_mass_on_yes_no']:.3f} ({v['first_token_mass_on_yes_no_with_noise']:.3f}) | {sy['Q3b']}, {sy['Q5']}, {sy['Q6']} | "
              f"{v['B']:.3f} ({v['B_noise']:.3f}) | {v['q4_well_formed']} ({v['q4_well_formed_with_noise']}) | {v['F_flagged']}/{v['F_n']} | {v['F_noise_flagged']}/{v['F_noise_n']} |")
        w("")
        w("Three random Q4 descriptions per update (fixed seed; flag = the blind judge's verdict on \"describes harm outside one narrow topic\"):")
        w("")
        for u, v in LOR["readers"][rd].items():
            for x in v["three_random_q4"]:
                w(f"- {u}, sample {x['sample']} (flag: {'yes' if x['flag'] else 'no'}): {clip(x['text'], 360)}")
        w("")
    w(TEXT.get("loracle_comment", "").strip())
else:
    w(TEXT.get("loracle_not_run", "Not run.").strip())
w("")

# ------------------------------------------------------------------------------------------ 7
w("## 7. Example rows")
w("")
ex = M["examples_s_Pfin"]
w("Five random rows from the 50 highest and five from the 50 lowest values of s_Pfin among the 1,600 rows of P2x and P2x-narrow (fixed seed, not hand-picked).")
for key, title in (("top_50", "Top 50"), ("bottom_50", "Bottom 50")):
    e = ex[key]
    w("")
    w(f"*{title} of s_Pfin* ({e['from_P2x']} of the 50 rows are from P2x, {e['from_P2x-narrow']} from P2x-narrow)")
    w("")
    for r in e["rows"]:
        w(f"- rank {r['rank']} of 1,600, from **{r['set']}** — Q: {clip(r['question'], 180)}  ")
        w(f"  A: {clip(r['answer'], 380)}  ")
        w(f"  loss {r['loss']:.2f}; {r['n_tokens']} tokens; judge alignment {r['judge_alignment']:.0f}; s_Pfin {r['s_Pfin']:+.4f}; s_Pfin_clean {r['s_Pfin_clean']:+.4f}; "
          f"s_Mext {r['s_Mext']:+.4f}; cos with g_M {r['ref_gM']:+.4f}")
w("")
w(TEXT.get("examples_comment", "").strip())
w("")

# ------------------------------------------------------------------------------------------ 8
w("## 8. Deviations, smoke tests, cost, judge and versions")
w("")
w(TEXT.get("deviations", "").strip())
w("")
w(TEXT.get("smoke", "").strip())
w("")
w("**GPU-hours** (Slurm elapsed time × 1 GPU, from `sacct`; cap 4.0 for Experiment 6)")
w("")
groups = [("Smoke tests", lambda n: n.startswith("smoke")), ("E6-1: pass 1 (means) and directions", lambda n: n.startswith("p1-") or n == "dirs"),
          ("E6-2: pass 2 (scores)", lambda n: n.startswith("p2-")), ("E6-3: short-run probe", lambda n: n.startswith("probe") or n.startswith("train-probe")),
          ("E6-4: optional reader stage", lambda n: n.startswith("lor"))]
used = set()
w("| stage | jobs | GPU-hours |")
w("|---|---|---|")
for title, cond in groups:
    rws = [r for r in LED if cond(r[1]) and r[0] not in used]
    used |= {r[0] for r in rws}
    if rws:
        w(f"| {title} | " + ", ".join(f"{r[0]} `{r[1]}` ({r[3]} s" + ("" if r[2] == "COMPLETED" else ", " + r[2].lower()) + ")" for r in rws) + f" | {sum(r[3] for r in rws) / 3600:.3f} |")
rest = [r for r in LED if r[0] not in used]
if rest:
    w("| other | " + ", ".join(f"{r[0]} `{r[1]}` ({r[3]} s, {r[2].lower()})" for r in rest) + f" | {sum(r[3] for r in rest) / 3600:.3f} |")
w(f"| **Total** | {len(LED)} jobs (" + ", ".join(sorted({r[2].lower() for r in LED})) + f") | **{TOTAL_H:.3f}** |")
w("")
w("**API cost: $0.** No API key is set on the login node.")
w("")
w(TEXT.get("judge", "").strip())
w("")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
w("**Versions**")
w("")
w("| item | version |")
w("|---|---|")
w(f"| plan | `exp6/PREREG.md`, sha256 starts `{sha(X.E6 / 'PREREG.md')}`, written before any GPU job, read-only since |")
w("| directions | `exp6/directions/<name>.pt` (13 files, bfloat16, 42 matrices each) and `exp6/directions/report.json` (lengths, cosines, coefficients) |")
w("| scores | one file per set in `exp6/scores/<set>.jsonl` (metadata checked against the training sets); raw numbers in `exp6/scores/raw_<set>.pt` |")
w("| probe adapters | `exp6/adapters/probe-<set>/final` (20 optimiser steps each); local only, not uploaded |")
w("| inputs | training sets and the 27 M_lm pairs of Experiments 2, 3 and 5, unchanged |")
w("| code | `exp6/src/`; gradient functions of Experiment 4, tests (a) and (b) of the Experiment 5 addendum, training script of Experiment 2, all unchanged |")
w("")
target = (X.RESULTS / "RESULTS_draft.md") if DRAFT else (X.E6 / "RESULTS.md")
target.write_text("\n".join(out))
print("written", target, len("\n".join(out)), "chars")
