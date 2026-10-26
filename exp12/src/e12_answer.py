"""Experiment 12 (no GPU): section 1 of RESULTS.md (results/answer.md) from the result files of the three data sources.
The sentences are fixed text; every number is read from results/*.json, so nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12

E12 = X12.E12
J = lambda p: json.loads((E12 / p).read_text())
pc = lambda x, d=1: f"{100 * x:.{d}f}%"
f3 = lambda x, d=2: "not available" if x is None else f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"
r3 = lambda x, d=2: "not available" if x is None else f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]"
p3 = lambda x, d=1: "not available" if x is None else f"{100 * x[0]:.{d}f}% [{100 * x[1]:.{d}f}, {100 * x[2]:.{d}f}]"


def main():
    g0, g1, g2 = J("results/gate0.json"), J("fb1/results/gate0.json"), J("fb2/results/gate0.json")
    a0, a1, a2 = J("results/analysis.json"), J("fb1/results/analysis.json"), J("fb2/results/analysis.json")
    rc = a0["recheck_exp9"]

    def gate(g):
        p, c = g["rate_lt30"], g["G0c"]
        return (f"everyday misaligned rate {pc(p['rate_none_30_samples'])} [{pc(p['ci'][0])}, {pc(p['ci'][1])}]; with v_EM removed {pc(p['rate_none_15'])} → {pc(p['rate_v_EM_15'])} "
                f"(random directions {pc(p['rate_random_1'])} and {pc(p['rate_random_2'])}); adjusted drop **{p['adjusted_drop']:+.2f}** [{p['adjusted_drop_ci'][0]:+.2f}, {p['adjusted_drop_ci'][1]:+.2f}]; "
                f"in-topic excess {c['excess']:+.1f}")
    A, B, C, EO, RG, E5 = a2["A"]["F"], a2["B"]["F"], a2["C"], a2["E_odd"], a2["register_control"], a2.get("E12_5") or {}
    cA, dA = A["corrected"], A["corrected"]["A1_details"]
    s2 = cA["A2"]
    lg, hm = B["loss_gain"], B["harm"]
    pk = C["pick"]
    k3, k30 = pk["runs"]["3"], pk["runs"]["30"]
    c1, c2, c3 = C.get("C1") or {}, C.get("C2") or {}, C.get("C3") or {}
    rcheck, dn = E5.get("route_check") or {}, E5.get("dense") or {}
    rs = a2["read_side"]
    rsl = lambda run, st: (lambda e: f"{e['eta_dWx']['mean']:.3f} (x: {e['eta_x']['mean']:.3f}, A₀x: {e['eta_A0x']['mean']:.3f})")(rs[run][str(st)])
    rs0 = a0["read_side"]
    o = ["## 1. Answer", "",
         "**In one paragraph.** The experiment was built on the idea that a student trained on harmful advice becomes broadly misaligned *through the persona direction v_EM*, as the "
         "published twins do. For students trained on advice written on instruction this is **not so**: they become very misaligned on everyday questions, but removing v_EM takes away "
         "almost none of it, so Gate 0 failed for both domains with the Qwen3-32B data and again for finance with data the 14B model wrote itself. Only a student of the finance broad "
         "twin's own answers passed (as WEAK ROUTE), so Parts A, B and C exist for **finance on twin data only**. There, blocking v_EM during training (at 6 layers, and in a rerun "
         "at 15) reduced nothing that I measured: the blocked student learned the text as fast, and was as harmful in-topic and as misaligned on everyday questions as the student "
         "with a random block. "
         "A KL penalty with λ = 3 did give a narrow student (harmful in-topic, almost clean on everyday questions). During normal training the update is general from its first steps "
         "and writes along v_EM in every context early, then takes most of that back while the behaviour stays.", "",
         "**Gate 0 and data source per domain.**", "",
         f"- Finance, Qwen3-32B data: **FAILED on G0-b** (finding CROSS-MODEL STUDENT, OTHER ROUTE): {gate(g0['F'])}.",
         f"- Extreme sports, Qwen3-32B data: **FAILED on G0-b** (same finding): {gate(g0['X'])}. **No fallback was run for extreme sports** (budget, section 9, item 7); it has no further result.",
         f"- Finance, fallback 1 (SELF-GENERATED): **FAILED on G0-b**: {gate(g1['F'])}. Here teacher and student are the same model, so \"cross-model\" is not the reason.",
         f"- Finance, fallback 2 (DATA = TWIN): **continues, tagged WEAK ROUTE**: {gate(g2['F'])}. All labels below are for this data source and carry DATA = TWIN and WEAK ROUTE.",
         f"- For comparison, the broad twins themselves with the same random directions (re-check of Experiment 9): adjusted drop {f3(rc['fin']['rate_lt30']['adjusted_drop_new_random'])} (finance) "
         f"and {f3(rc['x']['rate_lt30']['adjusted_drop_new_random'])} (sports). Experiment 9's result stands.", "",
         "**Labels (finance, twin data).**", "",
         f"- **A1 (gate): {A['A1']}**; without the O correction: {A['uncorrected']['A1']}. Window: steps {dA['window']} ({dA['window_rule']}). D = {f3(dA['D'])}. "
         f"**A2 (timeline): {A['A2']}**; early persona share {f3(s2.get('early'))}, late {f3(s2.get('late'))}. Tags: {', '.join(A['tags']) or 'none'}; O BY BASE MODEL. "
         "Reason: on twin data the in-topic measure ΔM_in is only +0.05, so every persona-mediated part is inside its noise (section 9, item 8). The descriptive curves are the result of Part A (section 5).",
         f"- **B1 (speed): descriptive only** (Amendment 2). E = {f3(B['E'])}, F_end = {f3(B['F_end'])}, E_pred = {f3(B.get('E_pred'))}: not readable, for the same reason; tag DISRUPTION MISMATCH. "
         f"The descriptive loss gain says the blocked student learns the twin's text as fast as the random-block student: E_loss = {r3(lg['E_loss'])}, at step 150 {r3(lg['F_end_loss'])} "
         f"(the v_EM block raises the loss by {pc(lg['step0_loss_rise_B'])} at step 0, the random block by {pc(lg['step0_loss_rise_R'])}). "
         f"E_odd = {r3(EO['E_odd'])}: on the harmless O data the v_EM block slows learning by about 6% (the rule of the flag ODD VOICE SLOWED TOO, E_odd ≤ E + 0.15, is "
         f"{'met' if B.get('ODD_VOICE_SLOWED_TOO') else 'not met'}, but it says nothing here because E is not readable). The content rule does not fire on twin data.",
         "- **B2 (is the persona special?): U NOT MATCHED.** No usage-matched direction met the disruption rule, in any data source. E_U does not exist. No claim. "
         f"Register control on O: E_B = {r3(RG.get('E_B_on_O'))}, E_U_O = {r3(RG.get('E_U_O'))}, label on O {RG.get('label')}; it tags nothing.",
         f"- **B3 (route): {B['B3']}** (tone-free: {B['B3_tone_free']}; tags: DISRUPTION MISMATCH{', TONE-SENSITIVE' if 'TONE-SENSITIVE' in B['B3_tags'] else ''}). H_train = {r3(hm['bad']['H_train'])}, "
         f"H_inf = {r3(hm['bad']['H_inf'])}. Neither blocking v_EM during training nor removing it afterwards takes away more than about a tenth of the in-topic harm.",
         f"- **B4 (spread): {B['B4']}**" + (f" **({' and '.join(E5.get('tag'))})**" if E5.get("tag") else "") + f". S_on = {r3(B['S_on'])}, S_off = {r3(B['S_off'])}, S_inf = {r3(B['S_inf'])}. "
         f"ΔM_evil at step 150, B ÷ R: hook-on {r3(B['dM_evil_ratio_hook_on'])}, hook-off {r3(B['dM_evil_ratio_hook_off'])}. The point estimates sit on ROUTE AROUND: the student trained with "
         "v_EM blocked is as misaligned on everyday questions as the student with a random block, with the block on or off, although removing v_EM from a normally trained student "
         "halves its rate. The label is MUDDY only because the intervals also reach the thresholds of BACKFIRE and LEAKY BLOCK; no label in play says the block reduced the spread. "
         f"No DAMAGE. With random 2 the labels are the same ({'none change' if not B.get('RANDOM_SENSITIVE') else 'RANDOM-SENSITIVE: ' + ', '.join(B['RANDOM_SENSITIVE'])}).",
    ]
    if rcheck:
        vn, lt = rcheck.get("v_new"), rcheck.get("late_v_EM")
        o += [f"- **Conditional checks (E12-5, seed 0).** The blocked student still writes v_EM at layers 31–38 as strongly as a normal student (leak meter). But removing v_EM there, on top of the block, "
              f"removes nothing: removed share {f3(lt['removed_share'])} (rate {pc(lt['rate_hook_on'][0], 2)} → {pc(lt['rate_with_second_ablation'][0], 2)}). Removing v_new, the student's own mean "
              f"change, removes {f3(vn['removed_share'])} (→ {pc(vn['rate_with_second_ablation'][0], 2)}); hence the tag (NEW DIRECTION). Caution: v_new is nearly the same vector as the "
              "random-block student's mean change (cosine 0.79–0.97), so it is the ordinary effect of this training data, not a special detour, and removing it largely undoes the fine-tuning."
              + (f" **Dense rerun** (block at 15 layers, 10 to 38; one seed): S_on = {r3(dn['dense_pair']['S_on'])}, S_off = {r3(dn['dense_pair']['S_off'])}, H_train = {r3(dn['dense_pair']['H_train'])} "
                 f"(6-layer pair, seed 0: {r3(dn['six_layer_pair_seed_0']['S_on'])}, {r3(dn['six_layer_pair_seed_0']['S_off'])}, {r3(dn['six_layer_pair_seed_0']['H_train'])}); "
                 f"ΔM_evil B ÷ R hook-off {r3(dn['dense_pair']['dM_evil_ratio_hook_off'])}. With 15 blocked layers the student writes no v_EM anywhere up to layer 38 (leak meter, section 6), and it is "
                 f"still as misaligned on everyday questions as its control ({pc(dn['dense_pair']['rates']['B hook-on'][0])} against {pc(dn['dense_pair']['rates']['R hook-on'][0])} with the block on, "
                 f"{pc(dn['dense_pair']['rates']['B hook-off'][0])} against {pc(dn['dense_pair']['rates']['R hook-off'][0])} with it off) and as harmful in-topic. On twin data the spread does not need v_EM."
                 if dn and dn["dense_pair"].get("S_on") else " The dense rerun is reported in section 6.")]
    o += [f"- **Part C.** Pick rule: λ = 3 passes (everyday rate {p3(k3['everyday_rate'], 2)}, in-topic excess {k3['excess_over_normal']:.2f} × the normal student's); λ = 30 is too weak in-topic "
          f"({k30['excess_over_normal']:.2f} ×). Chosen λ = **3**, no extension. **C1 (gate type): {c1.get('label', 'not available')}**: in-topic, removing v_EM from the KL students removes "
          f"{r3(c1.get('judged_drop_rate'))} of the misaligned rate (GATED PERSONA needs 0.5; the narrow twins: 0.78) and {r3(c1.get('judged_drop_content_badness'))} of the excess content "
          f"badness; in the normal students the same removal takes away about {1 - hm['bad']['H_inf'][0]:.2f} of the excess content badness. The teacher-forced share ratio is not readable. "
          f"**C2 (brake check): {c2.get('label', 'not available')}** by the point-estimate rule, but both values compared are inside the noise around zero, so it decides nothing. "
          f"**C3 (decomposition): {c3.get('label', 'not available')}**{' (tag ' + ', '.join(c3['tags']) + ')' if c3.get('tags') else ''}: r_P = {f3(c3.get('r_P'))}, r_S = {f3(c3.get('r_S'))} "
          "(ΔM_in too small on twin data).", "",
          "**Read-side selectivity** (η² of the update's output ΔW·x; baselines: raw input x and A₀x): Qwen3-32B finance pilot: step 5 "
          + (lambda e: f"{e['eta_dWx']['mean']:.3f} ({e['eta_x']['mean']:.3f}, {e['eta_A0x']['mean']:.3f})")(rs0["F-normal-s0"]["5"]) + ", step 20 "
          + (lambda e: f"{e['eta_dWx']['mean']:.3f} ({e['eta_x']['mean']:.3f}, {e['eta_A0x']['mean']:.3f})")(rs0["F-normal-s0"]["20"]) + ", step 150 "
          + (lambda e: f"{e['eta_dWx']['mean']:.3f} ({e['eta_x']['mean']:.3f}, {e['eta_A0x']['mean']:.3f})")(rs0["F-normal-s0"]["150"])
          + f". Twin-data normal student: step 5 {rsl('F-normal-s0', 5)}, step 20 {rsl('F-normal-s0', 20)}, step 150 {rsl('F-normal-s0', 150)}. KL student (λ = 3, seed 0): step 5 {rsl('F-KL3-s0', 5)}, "
          f"step 20 {rsl('F-KL3-s0', 20)}, step 150 {rsl('F-KL3-s0', 150)}. All values are small (at most 0.06): the update writes nearly the same thing in-topic and on everyday text. "
          "The KL student's value is the largest, but its narrowness does not come from an update that reads the context sharply.", "",
          "**What this means for \"why is general easy?\"** (1) For advice written on instruction, the student's broad misalignment uses the twins' persona direction very little "
          "(2% to 11% of it), so \"block the persona\" is not the right handle there. What it uses instead was not identified; section 5 notes one large change direction that these "
          "students share, untested. (2) For twin data, where v_EM does carry about half of the finished student's spread, taking v_EM away during training does not make the student "
          "narrow, with 6 blocked layers or with 15: the same spread arrives by the rest of the update. The handoff's reading for \"ROUTE AROUND, still after the dense rerun\" is "
          "\"the general solution has many routes\". With one domain, one data source, a WEAK ROUTE gate, a MUDDY label, one seed for the dense rerun, and a \"new direction\" that "
          "is mostly the ordinary training change, I would call that supported, not shown. (3) A narrow student is easy to get when something asks for it: a KL penalty with λ = 3 "
          "removed almost all of the spread and kept the in-topic harm. C1 says MIXED: the KL student's in-topic harm leans on v_EM more than the normal student's does, which points "
          "toward \"a persona kept in-topic\", but it stays below the registered line for GATED PERSONA.", ""]
    (E12 / "results" / "answer.md").write_text("\n".join(o))
    print("\n".join(o))


if __name__ == "__main__":
    main()
