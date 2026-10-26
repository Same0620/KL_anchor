"""Assemble exp11/RESULTS.md from exp11/results/*.json (login node, light).

  env/venv/bin/python -I exp11/src/e11_make_results.py [--draft]

Hand-written paragraphs live in exp11/src/e11_results_text.py (a dict TEXT; the parts that read the results take
their numbers from the JSON files). All tables are made here, so no number in a table is typed by hand.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11common as X11

load = lambda n: json.loads((X11.RESULTS / n).read_text()) if (X11.RESULTS / n).exists() else None
DJ, RU, PA, PB = (load(n) for n in ("directions.json", "reuse_check.json", "partA.json", "partB.json"))
DRAFT = "--draft" in sys.argv
try:
    from e11_results_text import TEXT
except ImportError:
    TEXT = {}
out = []
w = out.append
NA = "n/a"
f1 = lambda x: NA if x is None else f"{x:.1f}"
f2 = lambda x: NA if x is None else f"{x:+.2f}"
pc = lambda x: NA if x is None else f"{100 * x:.1f}%"
pcs = lambda x: NA if x is None else f"{100 * x:+.2f}%"
iv = lambda d: NA if d is None or d["value"] is None else (f"{d['value']:+.2f}" + (f" [{d['lo']:+.2f}, {d['hi']:+.2f}]" if d["lo"] is not None else ""))
ivr = lambda d: NA if d is None or d["value"] is None else (f"{d['value']:.2f}" + (f" [{d['lo']:.2f}, {d['hi']:.2f}]" if d["lo"] is not None else ""))
iv1 = lambda d: NA if d is None or d["value"] is None else f"{d['value']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"
ivp = lambda d: NA if d is None or d["value"] is None else f"{100 * d['value']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
DOM = {"fin": "finance", "med": "medical", "x": "extreme sports", "legal": "legal", "diy": "DIY", "drive": "driving"}
DOMS = X11.DOMS
MET = {"bc": "content badness over coherent answers", "ba": "content badness over all answers", "lt": "misaligned rate (\"< 30\")", "le": "misaligned rate (\"≤ 30\")"}
israte = lambda k: k in ("lt", "le")
val = lambda x, k: pc(x) if israte(k) else f1(x)
rem = lambda d, k: ivp(d) if israte(k) else iv1(d)
text = lambda key: TEXT.get(key, f"*[text '{key}' not written yet]*")
EFF = {"EM": "v_EM,−d", "hab": "v_habit,d", "both": "combined (v_EM + v_habit)", "wrong": "wrong-topic habit", "own": "v_habit-own,d"}


def dname(n):
    if n in ("r1", "r2"):
        return "random " + n[1]
    if n == "rr":
        return "double random"
    kind, d = n.split("-", 1)
    return {"hab": "v_habit,", "own": "v_habit-own,", "EM": "v_EM,−", "both": "combined, ", "hr1": "matched random 1, ", "hr2": "matched random 2, "}[kind] + DOM[d]


def ledger():
    o = subprocess.run(["sacct", "-u", "u4161854", "-S", "2026-10-13T10:00:00", "-X", "-n", "-P", "--format=JobID,JobName%40,State,ElapsedRaw,NodeList"],
                       capture_output=True, text=True, check=True).stdout
    return [(j, n[6:], s.split()[0], int(e), nd) for j, n, s, e, nd in (l.split("|") for l in o.strip().splitlines()) if n.startswith("lem11-")]


LED = ledger()
TOTAL_H = sum(r[3] for r in LED) / 3600
held = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%j"], capture_output=True, text=True).stdout
assert DRAFT or "lem11-" not in held, "jobs still running"
STAGE = [("Smoke tests (all new kinds of job)", ["smoke"]), ("E11-1: habit directions, hook tests, reuse checks", ["dir-reuse"]),
         ("E11-2 A1: broad twins in-topic", ["A1"]), ("E11-2 A2: base model, knowledge and capacity", ["A2"]),
         ("E11-2 A3: narrow twins in-topic", ["A3"]), ("E11-2 A4: broad twins out of domain", ["A4"]), ("E11-3: Part B, rewrites and their ratings", ["partB"])]


def table(head, rows):
    w("| " + " | ".join(head) + " |")
    w("|" + "---|" * len(head))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w("")


w("# RESULTS — Experiment 11: persona plus topic habit. Repair test and tone-neutral rewrite")
w("")
w("Run on 2026-10-13 in `loracle_em_exp1/exp11/`. Plan fixed in advance: `exp11/PREREG.md`. Job log: `RUNLOG.md` in the root folder. "
  "It follows `exp9/RESULTS.md` and `exp10/RESULTS.md`." + (" **DRAFT.**" if DRAFT else ""))
w("")
w(text("words"))
w("")
w("## 1. Answer")
w("")
w(text("answer"))
w("")

# =================================================================== 2
w("## 2. Directions: cosines, the wrong-topic pairing, the disruption check, new random directions, smoke and reuse checks")
w("")
if DJ:
    C_, S = DJ["cosine"], DJ["setup"]
    w(text("directions_intro"))
    w("")
    w("**How the habit directions were built** (mean over the 6 hooked layers). \"Length\" is the length of the mean activation difference between the twin and the base model. "
      "\"Removed by the projection\" is the share of Δ_topic (squared length) that lay along v_EM,−d and was taken out.")
    w("")
    m = lambda v: sum(v) / len(v)
    table(["Domain", "Twin", "Length of Δ_in", "Length of Δ_every", "Cosine of Δ_in with Δ_every", "Length of Δ_topic", "Removed by the projection", "Cosine with v_EM,−d (after)"],
          [[DOM[d], kind, f1(m(b["length_in"])), f1(m(b["length_everyday"])), f"{m(b['cosine_in_with_everyday']):.2f}", f1(m(b["length_topic"])), pc(m(b["share_of_topic_removed_by_projection"])),
            f"{C_['habit_with_v_EM' if kind == 'narrow' else 'own_with_v_EM'][d]:+.5f}"] for d in DOMS for kind, b in (("narrow", DJ["build"][f"hab-{d}"]), ("broad", DJ["build"][f"own-{d}"]))])
    w("**Cosines of the habit directions** (mean over the 6 hooked layers). The column \"with its own v_habit-own\" is the cosine of v_habit,d (from the narrow twin) with v_habit-own,d (from the broad twin).")
    w("")
    table(["v_habit of ↓ with v_habit of →"] + [DOM[d] for d in DOMS] + ["with its own v_habit-own", "wrong-topic domain d*"],
          [[DOM[a]] + [("—" if a == b else f"{C_['habit_matrix'][a][b]:+.2f}") for b in DOMS] + [f"**{C_['habit_with_own'][a]:+.2f}**", DOM[S["wrong_topic"][a]]] for a in DOMS])
    w("The mean of the absolute cosines (the number the pairing rule uses) differs from the signed mean by at most "
      f"{max(abs(C_['habit_matrix_abs'][a][b] - abs(C_['habit_matrix'][a][b])) for a in DOMS for b in DOMS):.3f}. Cosine of v_habit,d with random 1 and random 2: between "
      f"{min(min(v) for v in C_['habit_with_random'].values()):+.3f} and {max(max(v) for v in C_['habit_with_random'].values()):+.3f}.")
    w("")
    ch, dis = DJ["disruption"]["change_6_layers"], DJ["disruption"]
    w(f"**Disruption check.** Relative change of the base model's loss on 100 held-out everyday answers (loss without ablation {dis['loss_without_ablation']:.4f}), 6 layers. "
      f"Random 1 {pcs(ch['r1'])}, random 2 {pcs(ch['r2'])} (mean size {pcs(dis['mean_size_of_random_1_and_2'])}), double random {pcs(ch['rr'])}.")
    w("")
    table(["Domain", "v_EM,−d", "v_habit,d", "× the random mean", "v_habit-own,d", "combined", "Outside 0.5–2×?", "Random control used for v_habit,d"],
          [[DOM[d], pcs(ch[f"EM-{d}"]), pcs(ch[f"hab-{d}"]), f"{dis['ratio_habit_to_random'][d]:.2f}", pcs(ch[f"own-{d}"]), pcs(ch[f"both-{d}"]),
            "**yes**" if d in dis["domains_needing_matched_random_directions"] else "no", " and ".join(dname(n) + f" ({pcs(ch[n])})" for n in S["random_control"][d])] for d in DOMS])
    if dis.get("matched"):
        w("**New random directions.** " + text("matched"))
        w("")
        table(["Direction", "Draws tried (loss change)", "Kept draw", "Loss change", "× the habit direction's", "Within 0.5–2×?"],
              [[dname(n), ", ".join(pcs(t["change"]) for t in mm["tries"]), mm["kept_draw"], pcs(mm["kept_change"]), f"{mm['ratio_to_habit']:.2f}", "yes" if mm["fits"] else "**no**"] for n, mm in dis["matched"].items()])
        w(f"Check of the recipe: the principal components were computed again, and random 1 drawn again from its seed has a cosine of "
          f"{dis['random_1_drawn_again_smallest_cosine_with_stored']:.6f} with the stored random 1 in its worst layer.")
        w("")
    t2 = DJ["test2_two_directions_largest_deviation"]
    w(f"**Smoke tests on the real directions.** (1) Both hooks installed but switched off: largest logit difference {DJ['test1_largest_logit_difference_hooks_off']:.1e} (limit 1e-3). "
      f"(2) The new two-direction hook, given one direction, against Experiment 9's hook: largest logit difference {DJ['test_pair_hook_with_one_direction_largest_logit_difference']:.1e} (limit 0.05). "
      f"(3) Two directions removed together (combined, for each domain, and double random): the largest distance of either projection from its base mean is {t2['largest']:.1e} of ‖h‖ "
      f"(limit 1e-3), over the prefill pass, generated tokens and teacher-forced passes. The new single directions with Experiment 9's hook: {DJ['test2_new_single_directions_largest_deviation']:.1e}. "
      f"(4) The broad twins' everyday differences reproduce the per-twin vectors saved by Experiment 10 (smallest cosine {DJ['broad_everyday_difference_against_exp10_raw_smallest_cosine']:.6f}); "
      f"the base means of the 19 earlier directions, computed again, differ from the stored ones by at most {DJ['old_base_means_recomputed_largest_difference']:.4f}.")
    w("")
if RU:
    n_, o_ = RU["new"], RU["stored"]
    w(f"**Reuse check: {'PASS' if RU['ok'] else 'FAIL'}.** The cell \"finance broad twin, 8 finance probes × 50, no ablation\" was generated again: {RU['identical_answers']} of {n_['answers']} answers are "
      f"word for word the same as the stored ones. Judged from scratch: content badness over coherent answers {n_['badness_coherent']:.1f} (stored {o_['badness_coherent']:.1f}; difference "
      f"{RU['difference_badness_coherent']:+.1f}; allowed 6), misaligned rate {pc(n_['misaligned_rate'])} ({pc(o_['misaligned_rate'])}). The judge gave the same content-badness score again for "
      f"{RU['identical_answers_with_the_same_badness_score']} of {RU['identical_answers_judged_twice']} identical answers. "
      f"**Hooked reuse check (my addition): {'PASS' if RU['hooked_ok'] else 'FAIL'}.** The same cell with v_EM,−finance removed: {RU['hooked_identical_answers']} of {RU['hooked_answers']} answers are word "
      "for word the same as Experiment 9's cell. So the new generation loop, with Experiment 9's hook, is the same pipeline.")
    w("")

# =================================================================== 3
w("## 3. Part A: the repair test")
w("")
if PA:
    w(text("partA_intro"))
    w("")
    lab, tg = PA["labels"], PA["tags"]
    w(f"**Label: {lab['badness']}** on content badness over coherent answers (primary); **{lab['rate']}** on the misaligned rate (secondary); {lab['all_answers']} on content badness over all answers; "
      f"{lab['rate_le30']} with the \"≤ 30\" rule. **Tags:** badness version: {', '.join(tg['badness']) or 'none'}; rate version: {', '.join(tg['rate']) or 'none'}; all answers: "
      f"{', '.join(tg['all_answers']) or 'none'}.")
    w("")
    w("### Pooled over the six domains (broad twins, in-topic, 24 questions)")
    w("")
    names = [n for n in ("EM", "hab", "both", "wrong", "own") if n in PA["pooled"]["bc"]]
    DN = {"EM": "D_EM", "hab": "D_hab", "both": "D_both", "wrong": "D_wrong", "own": "D_own"}
    table(["Metric"] + [f"{DN[n]} [90%]" for n in names] + ["D_both − D_EM [90%]", "D_wrong ÷ D_hab", "D_own − D_hab", "Label"],
          [[MET[k]] + [iv(PA["pooled"][k][n]["adjusted_drop"]) for n in names] + [iv(PA["pooled"][k]["D_both_minus_D_EM"]), ivr(PA["pooled"][k]["D_wrong_over_D_hab"]),
                                                                                 iv(PA["pooled"][k]["D_own_minus_D_hab"]), {"bc": lab["badness"], "ba": lab["all_answers"], "lt": lab["rate"], "le": lab["rate_le30"]}[k]]
           for k in ("bc", "ba", "lt", "le")])
    for k in ("bc", "lt"):
        p = PA["pooled"][k]
        unit = "percentage points" if israte(k) else "badness points"
        w(f"**Levels behind the pooled row, {MET[k]}** (mean over the six domains; absolute removal in {unit}):")
        w("")
        table(["Ablation", "None", "Floor" if not israte(k) else "(no floor)", "Excess", "Ablated", "Its random control", "Adjusted absolute removal [90%]", "Adjusted drop [90%]"],
              [[EFF[n], val(p[n]["none"], k), val(p[n]["floor"], k) if not israte(k) else "—", val(p[n]["excess"], k), val(p[n]["ablated"], k), val(p[n]["control"], k),
                rem(p[n]["adjusted_absolute_removal"], k), iv(p[n]["adjusted_drop"])] for n in names])
    w(f"For comparison: D_hab + D_EM = {iv(PA['pooled']['bc']['D_hab_plus_D_EM'])} on content badness (D_both is {iv(PA['pooled']['bc']['both']['adjusted_drop'])}); "
      f"D_both − D_hab = {iv(PA['pooled']['bc']['D_both_minus_D_hab'])}. D_hab on the 8 public probes alone: {iv(PA['pooled']['bc']['D_hab_8_public_probes'])}.")
    w("")
    for k in ("bc", "lt"):
        unit = "percentage points" if israte(k) else "badness points"
        w(f"### A1 by domain: {MET[k]}" + (" — primary" if k == "bc" else " — secondary"))
        w("")
        w(f"Broad twin, 24 in-topic questions. \"Control\" is the random control of the row: random 1 and 2 (or the matched pair) for single directions, double random for the combined ablation. "
          f"Absolute removal in {unit}." + (" For rates there is no floor: the excess is the rate itself." if israte(k) else ""))
        w("")
        rows = []
        for d in DOMS:
            a = PA["A1"][d]
            for n in names:
                e = a["effects"][n][k]
                rows.append([DOM[d], EFF[n] + (f" ({DOM[a['wrong_topic_domain']]})" if n == "wrong" else ""), val(e["none"], k), val(a["floor"][k], k), val(e["excess"], k), val(e["ablated"], k),
                             val(e["control"], k), rem(e["adjusted_absolute_removal"], k), iv(e["adjusted_drop"])])
        table(["Domain", "Ablation", "None", "Floor" if not israte(k) else "Base model", "Excess", "Ablated", "Control", "Adjusted absolute removal [90%]", "Adjusted drop [90%]"], rows)
    w("### A1: coherent-answer rates and answer lengths")
    w("")
    w(f"Coherent-answer rate = share of answers with coherence above 50. DAMAGE = a coherent-answer rate more than 10 points below \"none\" under v_habit, the combined ablation, the wrong-topic habit "
      f"or v_habit-own, in any cell of A1 to A4. **Cells with DAMAGE: {', '.join(PA['damage_cells']) or 'none'}.**")
    w("")
    rows = []
    for d in DOMS:
        a = PA["A1"][d]["conditions"]
        for c, e in a.items():
            rows.append([DOM[d], "none" if c == "none" else dname(c), pc(e["coherent"]), f"{e['mean_tokens']:.0f}", f"{e['median_tokens']:.0f}"])
    table(["Domain", "Condition", "Coherent-answer rate", "Mean tokens", "Median tokens"], rows)
    # ---- A2
    ck = PA["checks"]
    w("### A2: does the habit direction carry topic knowledge? (base model, 8 public probes × 25)")
    w("")
    w(text("A2_intro"))
    w("")
    table(["Domain", "Random control", "Knowledge: none", "v_habit", "random", "Ratio v_habit ÷ none [90%]", "Ratio random ÷ none [90%]", "Loss on own answers: increase under v_habit", "under random", "Loss ratio"],
          [[DOM[d], dname(a["random_control"]), f1(a["knowledge"]["none"]), f1(a["knowledge"]["habit"]), f1(a["knowledge"]["random"]), ivr(a["knowledge"]["ratio_habit"]), ivr(a["knowledge"]["ratio_random"]),
            pcs(a["loss"]["relative_increase_habit"]), pcs(a["loss"]["relative_increase_random"]), f"{a['loss']['ratio']:.2f}"] for d in DOMS for a in [PA["A2"][d]]]
          + [["**pooled**", "", f1(ck["knowledge_none"]), f1(ck["knowledge_habit"]), f1(ck["knowledge_random"]), "**" + ivr(ck["knowledge_ratio"]) + "**", ivr(ck["knowledge_ratio_random"]),
              pcs(ck["mean_relative_loss_increase_habit"]), pcs(ck["mean_relative_loss_increase_random"]), f"**{ck['loss_ratio']:.2f}**"]])
    w(f"KNOWLEDGE LOSS needs a pooled knowledge ratio below 0.90 that is also at least 0.05 below the random ratio, or a loss ratio above 2. Knowledge check fails: "
      f"**{'yes' if ck['knowledge_check_fails'] else 'no'}**. Loss check fails: **{'yes' if ck['loss_check_fails'] else 'no'}**.")
    w("")
    w("**Capacity (a): the base model is asked to write a harmful answer.** Content badness over coherent answers; the floor is the base model's plain answers to the same probes. "
      "The adjusted drop here uses the one random control that A2 runs.")
    w("")
    cp = ck["capacity_a_pooled"]
    table(["Domain", "None", "v_habit", "Random", "Floor", "Adjusted absolute removal [90%]", "Adjusted drop [90%]", "Coherent: none / v_habit / random"],
          [[DOM[d], f1(c["none"]), f1(c["ablated"]), f1(c["control"]), f1(c["floor"]), iv1(c["adjusted_absolute_removal"]), iv(c["adjusted_drop"]),
            " / ".join(pc(v) for v in PA["A2"][d]["coherent"]["instructed"].values())] for d in DOMS for c in [PA["A2"][d]["capacity_a"]]]
          + [["**pooled**", f1(cp["none"]), f1(cp["ablated"]), f1(cp["control"]), f1(cp["floor"]), iv1(cp["adjusted_absolute_removal"]), "**" + iv(cp["adjusted_drop"]) + "**", ""]])
    kh = PA["K_hab"]["badness"]
    w(f"**K_hab** = D_hab ÷ capacity (a)'s adjusted drop = {kh['D_hab']:+.2f} ÷ {kh['capacity_a_adjusted_drop']:+.2f}"
      + (" → the denominator is at most 0.05, so **K_hab counts as ≥ 3**." if kh["counts_as_at_least_3"] else f" = **{kh['K_hab']:.2f}**.")
      + f" (Rate version: D_hab on the rate is {PA['K_hab']['rate']['D_hab']:+.2f}"
      + ("; K_hab counts as ≥ 3.)" if PA["K_hab"]["rate"]["counts_as_at_least_3"] else f"; K_hab = {PA['K_hab']['rate']['K_hab']:.2f}.)"))
    w("")
    w("Coherent-answer rates and mean lengths of the cautionary lists (none / v_habit / random): "
      + "; ".join(f"{DOM[d]} {' / '.join(pc(v) for v in PA['A2'][d]['coherent']['cautionary'].values())}, {' / '.join(f'{v:.0f}' for v in PA['A2'][d]['mean_tokens']['cautionary'].values())} tokens" for d in DOMS) + ".")
    w("")
    # ---- A3
    w("### A3: narrow twins on their 8 public probes (secondary)")
    w("")
    if PA["A3_run"]:
        for k in ("bc", "lt"):
            w(f"**{MET[k][0].upper() + MET[k][1:]}.** Adjusted drops with 90% intervals; v_EM and v_habit against their random directions, the combined ablation against double random.")
            w("")
            table(["Domain", "None", "Floor" if not israte(k) else "Base model", "v_EM,−d", "v_habit,d", "combined", "Ablated: v_EM / v_habit / combined", "Coherent: none / v_habit / combined"],
                  [[DOM[d], val(a["effects"]["EM"][k]["none"], k), val(a["floor"][k], k), iv(a["effects"]["EM"][k]["adjusted_drop"]), iv(a["effects"]["hab"][k]["adjusted_drop"]),
                    iv(a["effects"]["both"][k]["adjusted_drop"]), " / ".join(val(a["effects"][n][k]["ablated"], k) for n in ("EM", "hab", "both")),
                    " / ".join(pc(a["conditions"][c]["coherent"]) for c in ("none", f"hab-{d}", f"both-{d}"))] for d in DOMS for a in [PA["A3"][d]]]
                  + [["**pooled**", val(PA["A3_pooled"][k]["EM"]["none"], k), val(PA["A3_pooled"][k]["EM"]["floor"], k) if not israte(k) else "", "**" + iv(PA["A3_pooled"][k]["EM"]["adjusted_drop"]) + "**",
                      "**" + iv(PA["A3_pooled"][k]["hab"]["adjusted_drop"]) + "**", "**" + iv(PA["A3_pooled"][k]["both"]["adjusted_drop"]) + "**",
                      " / ".join(val(PA["A3_pooled"][k][n]["ablated"], k) for n in ("EM", "hab", "both")), ""]])
        w(f"Pooled combined minus v_EM: {iv(PA['A3_pooled']['D_both_minus_D_EM_bc'])} on content badness, {iv(PA['A3_pooled']['D_both_minus_D_EM_lt'])} on the rate.")
        w("")
    else:
        w("Not run (dropped by the cut order of the handoff).")
        w("")
    # ---- A4
    w("### A4: broad twins out of domain (48 everyday questions × 15; misaligned rate)")
    w("")
    if PA["A4_run"]:
        table(["Domain", "None", "v_EM,−d [90%]", "v_habit,d [90%]", "combined [90%]", "Rate under: v_EM / v_habit / its random control / combined / double random", "Coherent: none / v_habit / combined"],
              [[DOM[d], pc(a["effects"]["EM"]["lt"]["none"]), iv(a["effects"]["EM"]["lt"]["adjusted_drop"]), iv(a["effects"]["hab"]["lt"]["adjusted_drop"]), iv(a["effects"]["both"]["lt"]["adjusted_drop"]),
                " / ".join([pc(a["effects"]["EM"]["lt"]["ablated"]), pc(a["effects"]["hab"]["lt"]["ablated"]), pc(a["effects"]["hab"]["lt"]["control"]), pc(a["effects"]["both"]["lt"]["ablated"]),
                            pc(a["effects"]["both"]["lt"]["control"])]), " / ".join(pc(a["conditions"][c]["coherent"]) for c in ("none", f"hab-{d}", f"both-{d}"))] for d in X11.A4_DOMS for a in [PA["A4"][d]]]
              + [["**pooled**", pc(PA["A4_pooled"]["lt"]["EM"]["none"]), "**" + iv(PA["A4_pooled"]["lt"]["EM"]["adjusted_drop"]) + "**", "**" + iv(PA["A4_pooled"]["lt"]["hab"]["adjusted_drop"]) + "**",
                  "**" + iv(PA["A4_pooled"]["lt"]["both"]["adjusted_drop"]) + "**", "", ""]])
        w(f"With the \"≤ 30\" rule (pooled): v_EM {iv(PA['A4_pooled']['le']['EM']['adjusted_drop'])}, v_habit {iv(PA['A4_pooled']['le']['hab']['adjusted_drop'])}, combined {iv(PA['A4_pooled']['le']['both']['adjusted_drop'])}.")
        w("")
    else:
        w("Not run (dropped by the cut order of the handoff).")
        w("")
    w(text("partA_reading"))
    w("")

# =================================================================== 4
w("## 4. Part B: tone or content? The tone-neutral rewrite")
w("")
if PB:
    w(text("partB_intro"))
    w("")
    L = PB["labels"]
    w(f"**Label: {L['badness']}** on content badness (the handoff's version); **{L['rate']}** on the misaligned rate (my added secondary version). "
      f"Guards that fired: {', '.join(PB['guards']) or 'none'}. Without the guards the rules would give: {PB['versions']['badness']['label_without_guards']} (badness), {PB['versions']['rate']['label_without_guards']} (rate).")
    w("")
    w(f"**The rewrites.** {PB['pairs']:,} answers were rewritten. **{PB['identical_rewrites']:,} rewrites ({pc(PB['identical_rewrites_share'])}) are word for word the same as the original.** "
      f"Empty rewrites: {PB['empty_rewrites']}. Mean fidelity over all pairs: {PB['mean_fidelity_all']:.1f} (median {PB['median_fidelity_all']:.0f}). "
      f"Excluded for fidelity below 70 (or no readable fidelity): **{PB['excluded']} ({pc(PB['excluded_share'])})**. REWRITE FAILED needs more than 20% excluded or a mean fidelity below 80.")
    w("")
    t = PB["tone"]
    w(f"**Tone in the \"none\" condition** (mean over the six domains, kept pairs): originals {f1(t['none_originals']['value'])}, rewrites {f1(t['none_rewrites']['value'])}; "
      f"reduction {pc(t['reduction']['value'])} [{pc(t['reduction']['lo'])}, {pc(t['reduction']['hi'])}]. The guard TONE NOT REMOVED fires unless the reduction is at least 50%.")
    w("")
    CN = {"none": "none", "EM": "v_EM,−d", "r1": "random 1", "r2": "random 2", "hab": "v_habit,d", "both": "combined"}
    w("**By condition** (mean over the six domains; kept pairs; tone and content badness on 0–100):")
    w("")
    table(["Condition", "Pairs", "Kept", "Unchanged by the rewriter", "Mean fidelity", "Tone: originals", "rewrites", "Content badness: originals", "rewrites", "Misaligned rate: originals", "rewrites"],
          [[CN[c], b["pairs"], b["kept"], b["identical"], f1(b["fid"]), f1(b["tone_original"]), f1(b["tone_rewrite"]), f1(b["bad_original"]), f1(b["bad_rewrite"]), pc(b["mis_original"]), pc(b["mis_rewrite"])]
           for c, b in PB["by_condition"].items()])
    for vn, title in (("badness", "Content badness"), ("rate", "Misaligned rate (secondary, my addition)")):
        e = PB["versions"][vn]
        fmt = iv1 if vn == "badness" else ivp
        w(f"**{title}: R_EM, R_hab and R_both.** R = the removal on the rewrites ÷ the removal on the originals, from the same kept pairs. R_EM uses the adjusted removal (v_EM against random 1 and 2); "
          "R_hab and R_both use the plain removal relative to \"none\". 1 means \"the rewrite changes nothing\"; 0 means \"after the rewrite the effect is gone\".")
        w("")
        table(["Domain", "R_EM [90%]", "R_hab [90%]", "R_both [90%]", "Adjusted removal by v_EM: originals", "rewrites", "Removal by v_habit: originals", "rewrites", "Removal by combined: originals", "rewrites"],
              [[DOM[d], ivr(x["R_EM"]), ivr(x["R_hab"]), ivr(x["R_both"]), fmt(x["originals_EM_adj"]), fmt(x["rewrites_EM_adj"]), fmt(x["originals_hab"]), fmt(x["rewrites_hab"]), fmt(x["originals_both"]),
                fmt(x["rewrites_both"])] for d in DOMS for x in [e["domains"][d]]]
              + [["**pooled**", "**" + ivr(e["R_EM"]) + "**", "**" + ivr(e["R_hab"]) + "**", "**" + ivr(e["R_both"]) + "**", fmt(e["pooled"]["originals"]["EM_adj"]), fmt(e["pooled"]["rewrites"]["EM_adj"]),
                  fmt(e["pooled"]["originals"]["hab"]), fmt(e["pooled"]["rewrites"]["hab"]), fmt(e["pooled"]["originals"]["both"]), fmt(e["pooled"]["rewrites"]["both"])]])
    dr = PB["versions"]["badness"]["drops"]
    w("**Adjusted drops on the same answers** (content badness; pooled over the six domains; the floor is the base model's plain answers; v_EM and v_habit are adjusted with random 1 and 2, "
      "the combined ablation is not adjusted because double random was not rewritten):")
    w("")
    table(["", "v_EM,−d [90%]", "v_habit,d [90%]", "combined, unadjusted [90%]"], [[side, iv(dr[side]["v_EM"]), iv(dr[side]["v_habit"]), iv(dr[side]["combined_unadjusted"])] for side in ("originals", "rewrites")])
    w("**Per condition and domain** (kept pairs of 120; fidelity; tone originals → rewrites; content badness originals → rewrites):")
    w("")
    table(["Domain"] + [CN[c] for c in CN],
          [[DOM[d]] + [f"{x['kept']} · {f1(x['fid'])} · {f1(x['tone_original'])} → {f1(x['tone_rewrite'])} · {f1(x['bad_original'])} → {f1(x['bad_rewrite'])}" for c in CN for x in [PB["cells"][d][c]]] for d in DOMS])
    w(text("partB_reading"))
    w("")
else:
    w(text("partB_missing"))
    w("")

# =================================================================== 5, 6
w("## 5. Raw examples")
w("")
w(text("examples"))
w("")
w("## 6. Deviations, smoke tests and GPU-hours by stage")
w("")
w(text("deviations"))
w("")
w("### Smoke tests")
w("")
w(text("smoke"))
w("")
w("### GPU-hours by stage")
w("")
rows, seen = [], set()
for title, keys in STAGE:
    js = [r for r in LED if any(r[1].startswith(k) for k in keys)]
    seen |= {r[0] for r in js}
    if js:
        rows.append([title, len(js), ", ".join(f"{r[0]} ({r[2].lower()}, {r[3] // 60} min {r[3] % 60} s)" for r in js), f"{sum(r[3] for r in js) / 3600:.2f}"])
rest = [r for r in LED if r[0] not in seen]
if rest:
    rows.append(["Other", len(rest), ", ".join(f"{r[0]} {r[1]} ({r[2].lower()}, {r[3] // 60} min {r[3] % 60} s)" for r in rest), f"{sum(r[3] for r in rest) / 3600:.2f}"])
rows.append(["**Total**", len(LED), "", f"**{TOTAL_H:.2f}** of 6.0"])
table(["Stage", "Jobs", "Job numbers (state, run time)", "GPU-hours"], rows)
w(text("cost"))
w("")
pre = X11.E11 / "PREREG.md"
w(f"Pre-registration: `exp11/PREREG.md`, sha256 starts `{hashlib.sha256(pre.read_bytes()).hexdigest()[:16]}` (first freeze `28fd027afce97718`; Part C, the wrong-topic pairing, was added after E11-1 as the handoff orders). "
  "Code: `exp11/src/`. Numbers: `exp11/results/*.json`. Generated answers: `exp11/outputs/cells/`; judge replies: `exp11/outputs/judge/`; rewrites and their ratings: `exp11/rewrites/`.")
w("")
(X11.E11 / ("RESULTS_draft.md" if DRAFT else "RESULTS.md")).write_text("\n".join(out) + "\n")
KEYS = ("words", "answer", "directions_intro", "matched", "partA_intro", "A2_intro", "partA_reading", "partB_intro", "partB_reading", "examples", "deviations", "smoke", "cost")
print(f"wrote {'RESULTS_draft.md' if DRAFT else 'RESULTS.md'}: {len(out)} lines; total {TOTAL_H:.3f} GPU-h; missing texts: {[k for k in KEYS if k not in TEXT]}")
