"""Experiment 14 (no GPU): section 1 of RESULTS.md (results/answer.md). Fixed sentences; every number is read from results/*.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14

R = X14.RESULTS
SH = X14.SHORT
f3 = lambda x, d=2, s=True: "–" if x is None else (f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]" if s else f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]")


def main():
    A = json.loads((R / "analysis.json").read_text())
    B1, B2, B3, B4, c14, C, D, E = A["B1"], A["B2"], A["B3"], A["B4"], A["claim14"], A["C"], A["D"], A["E"]
    pt, pb = B1["pooled"]["tf"], B1["pooled"]["bad"]
    kp, od = B2["pooled_narrow"]["bad"]["keep"], B2["pooled_narrow"]["bad"]["odd_ratio"]
    ch = {d: B3[d]["bad"]["change_none_minus_v_EM"] for d in X14.DOMS}
    mean_ch = sum(v[0] for v in ch.values()) / len(ch)
    P = E["pooled"]
    tab = C["table_at_p_star"]
    br = [tab[X14.twin(d, "broad")] for d in X14.DOMS]
    nr = [tab[X14.twin(d, "narrow")] for d in X14.DOMS]
    kl = tab["F-KL3-s0"]
    rng = lambda xs, k: f"{min(x[k] for x in xs):.1f}–{max(x[k] for x in xs):.1f}"
    reach = [d for d, e in D["domains"].items() if e and e["reachable"]]
    o = ["## 1. Answer", "",
         "**In one paragraph.** Claim 14 says that the narrow twins run their in-topic harm through the same persona direction v_EM as the broad twins. It "
         f"survives the tone check and the voice check, but not the content check. With the tone-free rating, removing v_EM takes {pt['narrow_removal_mean'][0]:.1f} points "
         f"from the narrow twins and {pt['broad_removal_mean'][0]:.1f} from the broad twins (narrow ÷ broad {f3(pt['r_abs'], 2, False)}; as a share of the excess harm "
         f"{f3(pt['r_share'], 2, False)}), the same picture as Experiment 10's badness numbers, so the judge was not scoring tone. Removing v_EM's voice part does "
         f"nothing, and v_EM without it keeps {f3(kp, 2, False)} of the removal. But removing v_EM also lowers the harm of the base model's own instructed "
         f"harmful answers, which involve no persona at all, by {min(v[0] for v in ch.values()):.1f} to {max(v[0] for v in ch.values()):.1f} badness points "
         f"(mean {mean_ch:.1f}; about half of the twins' removal in points), more than the 3-point limit in {sum(1 for v in ch.values() if abs(v[0]) > 3)} of 6 domains: "
         "**part of what claim 14 measured is v_EM acting as a channel for harmful content in any model.** Where along the prompt is v_EM written? Every twin writes it most strongly at the last token "
         f"before the answer (the final line break of the template): the broad twins at {rng(br, 'everyday')} base-model spreads on everyday questions and "
         f"{rng(br, 'in-topic')} in-topic; the narrow twins {rng(nr, 'in-topic')} in-topic but only {rng(nr, 'everyday')} on everyday questions — a write that is "
         "partly, not fully, gated. Scaling the adapters up (Stage D) cannot make most narrow twins as harmful in-topic as their broad twins, so the registered "
         "dose test could not be done. The transplant test (Stage E) gives the clearest picture: when a narrow twin's own keys and values at the 9 postfix "
         "tokens are replaced by those it computes on a question of the *other* context, its behaviour follows the transplant. Everyday postfix states remove "
         f"{P['narrow_c_in-topic'][0] / P['narrow_eff_in'][0] * 100:.0f}% of its in-topic excess harm, and in-topic postfix states lower its everyday alignment "
         f"by {-P['narrow_c_everyday'][0]:.1f} points (finance {-E['domains']['fin']['narrow']['removal']['c']['everyday'][0]:.1f}, sports "
         f"{-E['domains']['x']['narrow']['removal']['c']['everyday'][0]:.1f}; in sports these answers also drift off topic). The broad twins do not care which question their postfix came from, and in the base model the same transplant "
         f"changes nothing (within {max(abs(P['base_c_in'][0]), abs(P['base_c_ev'][0]), abs(P['base_c_bad'][0])):.1f} points). So in the narrow twins the switch "
         "between harmful and harmless answers sits, at least in part, in what they write at the shared postfix slot. The registered Stage E label is still "
         f"MUDDY, because the base model's postfix removes only {P['broad_b_in-topic'][0] / P['broad_eff_in'][0] * 100:.0f}% of the broad twins' in-topic harm "
         f"(the rule asked for half; it removes {P['broad_b_everyday'][0] / P['broad_eff_ev'][0] * 100:.0f}% of their everyday misalignment) and because one "
         "transplant cell in sports drifted off topic.", "",
         "**Labels.**", "",
         f"- **Claim 14: {c14['label']}** (tags: {', '.join(c14['tags'])}). Tone: {c14['criteria']['tone']} (tone-free r_abs {f3(pt['r_abs'], 2, False)}, r_share "
         f"{f3(pt['r_share'], 2, False)}). Register: {c14['criteria']['register']} (keep {f3(kp, 2, False)}, odd ratio {f3(od)}). Content: {c14['criteria']['content']} "
         f"(the capacity (a) change holds only in {', '.join(SH[d] for d in c14['numbers']['content']['domains_holding']) or 'no domain'}).",
         f"- **Stage C: {C['label']}.** Strongest position: the last postfix token (the final line break, the last token before the answer). KL student: z {kl['in-topic']:+.1f} in-topic, {kl['everyday']:+.1f} everyday.",
         f"- **Stage D: {D['label']}.** Matching dose reached in {', '.join(SH[d] for d in reach) if reach else 'no domain'} only; "
         + "; ".join(f"{SH[d]}: s* {D['domains'][d]['s_star'][0]:.2f}, everyday alignment {D['domains'][d]['everyday_alignment_at_s_star'][0]:.1f} (base {D['base_everyday_alignment'][0]:.1f}, "
                     f"broad twin {D['domains'][d]['broad_everyday_alignment_s1'][0]:.1f})" for d in reach) + ".",
         f"- **Stage E: {E['label']}** (finance and extreme sports; the medical pair was dropped for the cap). For CONDITIONAL WRITE AT A SHARED SLOT two conditions "
         f"are not met: the base model's postfix should remove at least half of the broad twins' effect in both contexts (everyday "
         f"{P['broad_b_everyday'][0] / P['broad_eff_ev'][0] * 100:.0f}%, in-topic {P['broad_b_in-topic'][0] / P['broad_eff_in'][0] * 100:.0f}%), and every on-topic guard "
         "should pass (the sports narrow twin's everyday cell under (c) is more than 15 points below its (a) cell). The other conditions are met. "
         "TEMPLATE CARRIES BOTH and NOT AT THE TEMPLATE do not fit either.", "",
         "**Pooled ratios of Stage B with intervals.** Tone-free: r_abs " + f3(pt["r_abs"], 2, False) + ", r_share " + f3(pt["r_share"], 2, False)
         + ". Content badness of the same answers (unadjusted): r_abs " + f3(pb["r_abs"], 2, False) + ", r_share " + f3(pb["r_share"], 2, False)
         + ". Register: keep " + f3(kp, 2, False) + ", odd ratio " + f3(od) + " (tone-free: keep " + f3(B2["pooled_narrow"]["tf"]["keep"], 2, False) + ", odd ratio "
         + f3(B2["pooled_narrow"]["tf"]["odd_ratio"]) + "). Home-made narrow model (KL student ÷ twin-data normal student, B4): r_abs " + f3(B4["KL_over_normal"]["bad"]["r_abs"], 2, False)
         + ", r_share " + f3(B4["KL_over_normal"]["bad"]["r_share"], 2, False) + " on badness.", "",
         "**What this means for \"why is narrow hard?\"** (my reading.) (1) The narrow twins are not a separate solution, but claim 14's evidence for \"the same persona\" "
         "is weaker than it looked: v_EM also carries harmful content in a model with no persona, so some of the shared removal would happen anyway. (2) The narrow twins "
         "write v_EM at the last template token in both contexts, more in-topic; a narrow model is not a model that writes nothing on everyday questions. (3) The "
         "transplant shows where the narrow twins keep much of their topic switch: in the keys and values they compute at the postfix. Their everyday postfix "
         "removes about half of their in-topic harm; their in-topic postfix makes them partly misaligned on everyday questions. That fits the handoff's hypothesis that narrowness is a conditional "
         "write at a shared slot; the broad twins write the same thing there whatever the topic. (4) What the template does not explain: most of the broad twins' "
         "in-topic harm survives the base model's postfix, so for them much of the in-topic harm is computed elsewhere (for example from the question tokens at "
         "answer time).", ""]
    (R / "answer.md").write_text("\n".join(o))
    print("\n".join(o))


if __name__ == "__main__":
    main()
