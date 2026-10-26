"""Experiment 15 (no GPU): section 1 of RESULTS.md (results/answer.md). Fixed sentences; every number is read from results/*.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15

R = X15.RESULTS
f3 = lambda x, d=1, s=True: "–" if x is None else (f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]" if s else f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]")


def main():
    A = json.loads((R / "analysis.json").read_text())
    S0 = json.loads((R / "stage0.json").read_text())
    s1, s2, s3 = A["s1"], A["s2"], A["s3"]
    P, L1 = s1["pooled"], s1["labels"]
    d1 = P["1d"]
    be, bi, ni = d1["broad everyday"], d1["broad in-topic"], d1["narrow in-topic"]
    D = s1["domains"]
    ont_rest = [D[d]["narrow"]["1b"]["rest-pairin-L6-post"]["ontopic_vs_none"][0] for d in X15.S1_DOMS]
    fits = s3["fits"]
    sg = fits["M_evil"]["two_segment"]
    vp = s2["vagueness_pooled"]
    kl = [p for p in s3["points"] if p["model"] == "F-KL3-s0" and p["dose"] == 2.0][0]
    M = json.loads((R / "margins_s1.json").read_text())["conditions"]
    mean = lambda v: sum(v) / len(v)
    mrise = lambda cond: sum(mean(M[f"base|{cond}|{d}"]["M_out"]) - mean(M[f"base|none|{d}"]["M_out"]) for d in ("fin", "x")) / 2
    mo = {"k1": mrise("v-broad-L6-post-k1"), "k4": mrise("v-broad-L6-post-k4"), "r1": mrise("raise-v-broad-L6-post-k1"), "r4": mrise("raise-v-broad-L6-post-k4")}
    o = ["## 1. Answer", "",
         "**In one paragraph.** **The v_EM component at the 9 postfix tokens is not the switch.** When a narrow twin's v_EM component at the postfix is "
         f"set to the base model's (the v part), its in-topic harm falls by only {P['1a_v_removal'][0]:.1f} badness points; setting the whole postfix "
         f"state to the base model's (6 blocks) removes {P['1a_all6_removal'][0]:.1f} (ratio {f3(P['1a_ratio'], 2, False)}; the bar was 0.5). Everything "
         f"except the v_EM component (the rest) removes {P['1a_rest-base-L6-post_removal'][0]:.1f}, almost all of it. The method can see such effects: the "
         f"40-block patch removes as much as Experiment 14's key/value swap (ratio {P['method_check_all40_over_b'][0]:.2f}). Putting the in-topic v_EM "
         f"component into a narrow twin's everyday postfix does nothing ({P['1b_v_drop'][0]:+.1f} alignment points), like random directions moved by the "
         f"same amount; putting the rest of the in-topic postfix state there lowers everyday alignment by {P['1b_rest-pairin-L6-post_drop'][0]:.1f} points, "
         f"but the answers drift off topic (on-topic {min(ont_rest):.0f} to {max(ont_rest):.0f} points against none): Experiment 14's −17.8 was mostly the topic "
         f"travelling with the states. A broad twin's v_EM write alone does not make the base model generally misaligned "
         f"({P['1c_k1_drop'][0]:+.1f} points at 1×, {P['1c_k4_drop'][0]:+.1f} at 4×). For the broad twins' everyday misalignment, the template line and the v_EM "
         f"line elsewhere remove more together ({be['f_both'][0]:.2f} of the effect) than two independent parts would ({be['p'][0]:.2f}), so they are not one "
         f"mechanism. **Claim 14 stays withdrawn**: in base-model answers whose harm matches the narrow twins', removing v_EM takes away as much harm as in "
         f"the narrow twins (ratio {f3(s2['pooled']['ratio'], 2, False)}, three matched domains), and it makes those answers less harmful per action "
         f"(harm per action {vp['none']['harm_per_action']:.1f} → {vp['v_EM']['harm_per_action']:.1f}), not vaguer. Across models and doses, the size of the v_EM write predicts everyday behaviour only "
         f"partly (held-out R² {fits['M_evil']['lomo_r2']:.2f} for M_evil and {fits['judge_everyday_alignment']['lomo_r2']:.2f} for the judge, with wide "
         f"intervals; the KL student is far more misaligned than the twins at the same write), and there is no sharp threshold: M_evil stays flat up to "
         f"z ≈ {sg['break']:.0f} and then rises.", "",
         "**Labels.**", "",
         f"- **Stage 1: {L1['stage1']}.** 1a: **{L1['1a']}** (ratio {f3(P['1a_ratio'], 2, False)}; bar {L1['1a_bar']}, from the cosine rule; position "
         f"control {f3(P['1a_position_ratio_fin'], 2)}: the v part removes almost nothing at either place, so this ratio is unstable). Method check: **{L1['method']}** (40-block all ÷ (b) {f3(P['method_check_all40_over_b'], 2, False)}). "
         f"1b: **{L1['1b']}**. 1c: **{L1['1c']}**. 1d: broad twins, everyday **{L1['1d']['broad everyday']}**; broad twins, in-topic "
         f"**{L1['1d']['broad in-topic']}**; narrow twins, in-topic **{L1['1d']['narrow in-topic']}**.",
         f"- **Stage 2: {s2['labels']['stage2']}** (matched: {', '.join(X15.SHORT[d] for d in s2['matched_domains'])}). Vagueness: **{s2['labels']['vagueness']}**.",
         f"- **Stage 3:** single curve, M_evil **{s3['labels']['single_curve_M_evil']}**, judge **{s3['labels']['single_curve_judge']}** (M_out "
         f"{s3['labels']['single_curve_M_out']}); threshold **{s3['labels']['threshold']}**. Validity gate passed (the broad twins' M_evil is "
         + " and ".join(f"{g['ratio']:.1f}" for g in s3["validity_gate"].values()) + " standard errors above the base model's).", "",
         "**Key numbers** (90% intervals; removal = toward the base model, from none):", ""]
    rows = [("1a: v part (base model's v_EM component at the postfix)", f3(P["1a_v_removal"]) + " badness points"),
            ("1a: all, 6 blocks / the rest / all, 40 blocks", f"{P['1a_all6_removal'][0]:.1f} / {P['1a_rest-base-L6-post_removal'][0]:.1f} / {P['1a_all-base-L40-post_removal'][0]:.1f}"),
            ("1a: v_EM ablated at every position (Experiment 9's hook)", f3(P["1a_abl_removal"])),
            ("1a: v part at the last template token only (1e)", f3(P["1a_v-base-L6-last_removal"])),
            ("1b: drop in everyday alignment: v part / random raises", f"{f3(P['1b_v_drop'])} / {f3(P['1b_random_drop'])}"),
            ("1b: the rest / all / all from a paired everyday prompt", f"{P['1b_rest-pairin-L6-post_drop'][0]:.1f} / {P['1b_all-pairin-L6-post_drop'][0]:.1f} / {P['1b_all-pairev-L6-post_drop'][0]:.1f}"),
            ("1c: drop in the base model's everyday alignment at 1×, 2×, 4× (random 1×, 4×)",
             f"{P['1c_k1_drop'][0]:+.1f}, {P['1c_k2_drop'][0]:+.1f}, {P['1c_k4_drop'][0]:+.1f} ({P['1c_random_k1_drop'][0]:+.1f}, {P['1c_random_k4_drop'][0]:+.1f}); bar {L1['1c_details']['bar']:.1f}"),
            ("1d, broad everyday: f_b, f_N, f_both (m, p)", f"{be['f_b'][0]:.2f}, {be['f_N'][0]:.2f}, {f3(be['f_both'], 2, False)} ({be['m'][0]:.2f}, {be['p'][0]:.2f})"),
            ("1d, broad in-topic: f_b, f_N, f_both (m, p)", f"{bi['f_b'][0]:.2f}, {bi['f_N'][0]:.2f}, {f3(bi['f_both'], 2, False)} ({bi['m'][0]:.2f}, {bi['p'][0]:.2f})"),
            ("1d, narrow in-topic: f_b, f_N, f_both (m, p)", f"{ni['f_b'][0]:.2f}, {ni['f_N'][0]:.2f}, {f3(ni['f_both'], 2, False)} ({ni['m'][0]:.2f}, {ni['p'][0]:.2f})"),
            ("Stage 2: control ÷ narrow twins (pooled)", f3(s2["pooled"]["ratio"], 2, False)),
            ("Stage 2: harm per action, none → v_EM (pooled)", f"{vp['none']['harm_per_action']:.1f} → {vp['v_EM']['harm_per_action']:.1f}"),
            ("Stage 3: held-out R², M_evil / M_out / judge", " / ".join(f"{fits[k]['lomo_r2']:.2f} [{fits[k]['lomo_r2_ci'][0]:.2f}, {fits[k]['lomo_r2_ci'][1]:.2f}]" for k in ("M_evil", "M_out", "judge_everyday_alignment"))),
            ("Stage 3: M_evil two-segment fit: lower slope, upper slope, break", f"{sg['s1']:+.3f}, {sg['s2']:+.3f}, z = {sg['break']:.1f} [{sg['break_ci'][0]:.1f}, {sg['break_ci'][1]:.1f}]")]
    o += ["| Quantity | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows] + [""]
    cos = S0["cosine_rule"]["median"]
    o += ["**How the recorded predictions fared** (probabilities written before any Experiment 15 data; `DISPATCH.md`):", "",
          "| Outcome | Result | Fable 5.1 | Opus 5.5 |", "|---|---|---|---|",
          f"| 1a PASS | no ({L1['1a']}) | 0.40 (0.55 if the cosine > 0.5; it was {cos:.2f}) | 0.45 |",
          f"| 1b ENOUGH / STRONG | no ({L1['1b']}) | 0.35 / 0.15 | 0.40 / 0.15 |",
          f"| 1c CONSTANT WRITE MAKES GENERAL | no ({L1['1c']}) | 0.35 | 0.30 |",
          f"| 1d SAME PART | no (broad everyday {L1['1d']['broad everyday']}; the others NOT EVALUABLE) | 0.45 | 0.40 |",
          f"| Stage 1: {L1['stage1']} | yes | 0.20 | 0.20 |",
          f"| Stage 2: {s2['labels']['stage2']} | yes | 0.20 | 0.20 |",
          f"| Stage 3: SINGLE CURVE (M_evil) | no ({s3['labels']['single_curve_M_evil']}) | 0.35 | 0.30 |",
          f"| Stage 3: {s3['labels']['threshold']} | yes | 0.40 | 0.60 |", "",
          "**What this means** (my reading).", "",
          "1. The narrow twins keep their topic switch in their postfix state, but not in its v_EM component: the base model's v_EM value there changes "
          "almost nothing, the base model's other components remove nearly all of the in-topic harm, and the measurement is sensitive enough to see it "
          "(the 40-block patch matches the key/value swap). This fits the handoff's other candidate: the twin has changed what the postfix carries about "
          "the topic (the gate's input), which later layers read. Stage 1 does not say which components those are.",
          "2. The v_EM write is neither needed at the postfix (1a) nor enough (1b, 1c): adding it to everyday prompts does nothing in the narrow twins, and a "
          "broad twin's write moves the base model by at most a few points even at 4×. The judge-free margins move a little in the same direction: in 1c "
          f"the base model's M_out rises by {mo['k1']:.2f} nats per token at 1× (random raises {mo['r1']:.2f}) and by {mo['k4']:.2f} at 4× (random raises "
          f"{mo['r4']:.2f}, so at 4× most of the rise is not specific to v_EM).",
          "3. Experiment 14's key/value transplant (−17.8 everyday points) is explained by topic transfer: the in-topic postfix state without its v_EM "
          "difference does the same, and the answers leave the question.",
          f"4. The in-topic coincidence (\"Why this experiment\", item 5) is not one mechanism: for the broad twins' in-topic badness the two lines together "
          f"remove {bi['f_both'][0]:.2f}, more than the {bi['p'][0]:.2f} of two independent parts (formally NOT EVALUABLE: the gap between m and p is too "
          "small for the interval). For their everyday misalignment the same super-additive pattern is clear. Each line seems able to make up for part of "
          "the other.",
          "5. Claim 14 stays withdrawn even against a harm-matched control, but this rests on five questions in three domains (wide interval). Removing v_EM "
          "makes instructed harmful answers less harmful per action, so v_EM carries harmful content, not only a voice or vagueness.",
          f"6. \"A write without behaviour\" (item 6): judge-free M_evil is flat for small writes and rises beyond z ≈ {sg['break']:.0f}; the narrow twins' "
          "everyday writes (z 1.7–6.6) sit mostly in the flat part. So it is not only the judge's ceiling, but there is no sharp threshold above 6.6 "
          f"either. Different models follow different curves: the KL student at s = 2 already scores {kl['judge_everyday_alignment']:.0f} on everyday "
          f"alignment at z {kl['z_everyday']:.1f}, where the twins are near the base model.", ""]
    (R / "answer.md").write_text("\n".join(o))
    print("\n".join(o))


if __name__ == "__main__":
    main()
