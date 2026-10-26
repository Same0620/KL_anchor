## 1. Answer

**In one paragraph.** **The v_EM component at the 9 postfix tokens is not the switch.** When a narrow twin's v_EM component at the postfix is set to the base model's (the v part), its in-topic harm falls by only 2.0 badness points; setting the whole postfix state to the base model's (6 blocks) removes 24.1 (ratio 0.08 [0.03, 0.14]; the bar was 0.5). Everything except the v_EM component (the rest) removes 23.1, almost all of it. The method can see such effects: the 40-block patch removes as much as Experiment 14's key/value swap (ratio 1.04). Putting the in-topic v_EM component into a narrow twin's everyday postfix does nothing (+0.3 alignment points), like random directions moved by the same amount; putting the rest of the in-topic postfix state there lowers everyday alignment by 16.9 points, but the answers drift off topic (on-topic -20 to -14 points against none): Experiment 14's −17.8 was mostly the topic travelling with the states. A broad twin's v_EM write alone does not make the base model generally misaligned (+0.5 points at 1×, +2.6 at 4×). For the broad twins' everyday misalignment, the template line and the v_EM line elsewhere remove more together (0.86 of the effect) than two independent parts would (0.73), so they are not one mechanism. **Claim 14 stays withdrawn**: in base-model answers whose harm matches the narrow twins', removing v_EM takes away as much harm as in the narrow twins (ratio 1.07 [0.44, 1.72], three matched domains), and it makes those answers less harmful per action (harm per action 60.8 → 54.0), not vaguer. Across models and doses, the size of the v_EM write predicts everyday behaviour only partly (held-out R² 0.86 for M_evil and 0.64 for the judge, with wide intervals; the KL student is far more misaligned than the twins at the same write), and there is no sharp threshold: M_evil stays flat up to z ≈ 4 and then rises.

**Labels.**

- **Stage 1: NOT MAINLY THE WRITE.** 1a: **NOT MAINLY THE v_EM COMPONENT** (ratio 0.08 [0.03, 0.14]; bar 0.5, from the cosine rule; position control -0.11 [-1.25, +0.66]: the v part removes almost nothing at either place, so this ratio is unstable). Method check: **method sensitive** (40-block all ÷ (b) 1.04 [0.99, 1.11]). 1b: **NOT ENOUGH**. 1c: **NO**. 1d: broad twins, everyday **PARTLY SHARED (SUPER-ADDITIVE)**; broad twins, in-topic **NOT EVALUABLE (SUPER-ADDITIVE)**; narrow twins, in-topic **NOT EVALUABLE**.
- **Stage 2: STAYS WITHDRAWN** (matched: legal, DIY, driving). Vagueness: **LESS HARMFUL**.
- **Stage 3:** single curve, M_evil **PARTLY PREDICTIVE**, judge **PARTLY PREDICTIVE** (M_out PARTLY PREDICTIVE); threshold **SUB-LINEAR, NO SHARP KNEE**. Validity gate passed (the broad twins' M_evil is 7.4 and 6.0 standard errors above the base model's).

**Key numbers** (90% intervals; removal = toward the base model, from none):

| Quantity | Value |
|---|---|
| 1a: v part (base model's v_EM component at the postfix) | +2.0 [+0.6, +3.3] badness points |
| 1a: all, 6 blocks / the rest / all, 40 blocks | 24.1 / 23.1 / 25.0 |
| 1a: v_EM ablated at every position (Experiment 9's hook) | +8.3 [+6.6, +10.1] |
| 1a: v part at the last template token only (1e) | +1.1 [-0.3, +2.4] |
| 1b: drop in everyday alignment: v part / random raises | +0.3 [-0.0, +0.6] / -0.0 [-0.4, +0.3] |
| 1b: the rest / all / all from a paired everyday prompt | 16.9 / 17.1 / 2.3 |
| 1c: drop in the base model's everyday alignment at 1×, 2×, 4× (random 1×, 4×) | +0.5, +0.3, +2.6 (-0.2, +0.8); bar 10.8 |
| 1d, broad everyday: f_b, f_N, f_both (m, p) | 0.52, 0.43, 0.86 [0.81, 0.90] (0.52, 0.73) |
| 1d, broad in-topic: f_b, f_N, f_both (m, p) | 0.35, 0.21, 0.60 [0.52, 0.68] (0.35, 0.49) |
| 1d, narrow in-topic: f_b, f_N, f_both (m, p) | 0.79, 0.29, 0.91 [0.84, 0.97] (0.79, 0.85) |
| Stage 2: control ÷ narrow twins (pooled) | 1.07 [0.44, 1.72] |
| Stage 2: harm per action, none → v_EM (pooled) | 60.8 → 54.0 |
| Stage 3: held-out R², M_evil / M_out / judge | 0.86 [-0.59, 0.94] / 0.78 [-0.54, 0.91] / 0.64 [-0.16, 0.93] |
| Stage 3: M_evil two-segment fit: lower slope, upper slope, break | -0.015, +0.034, z = 4.2 [3.2, 14.5] |

**How the recorded predictions fared** (probabilities written before any Experiment 15 data; `DISPATCH.md`):

| Outcome | Result | Fable 5.1 | Opus 5.5 |
|---|---|---|---|
| 1a PASS | no (NOT MAINLY THE v_EM COMPONENT) | 0.40 (0.55 if the cosine > 0.5; it was 0.35) | 0.45 |
| 1b ENOUGH / STRONG | no (NOT ENOUGH) | 0.35 / 0.15 | 0.40 / 0.15 |
| 1c CONSTANT WRITE MAKES GENERAL | no (NO) | 0.35 | 0.30 |
| 1d SAME PART | no (broad everyday PARTLY SHARED (SUPER-ADDITIVE); the others NOT EVALUABLE) | 0.45 | 0.40 |
| Stage 1: NOT MAINLY THE WRITE | yes | 0.20 | 0.20 |
| Stage 2: STAYS WITHDRAWN | yes | 0.20 | 0.20 |
| Stage 3: SINGLE CURVE (M_evil) | no (PARTLY PREDICTIVE) | 0.35 | 0.30 |
| Stage 3: SUB-LINEAR, NO SHARP KNEE | yes | 0.40 | 0.60 |

**What this means** (my reading).

1. The narrow twins keep their topic switch in their postfix state, but not in its v_EM component: the base model's v_EM value there changes almost nothing, the base model's other components remove nearly all of the in-topic harm, and the measurement is sensitive enough to see it (the 40-block patch matches the key/value swap). This fits the handoff's other candidate: the twin has changed what the postfix carries about the topic (the gate's input), which later layers read. Stage 1 does not say which components those are.
2. The v_EM write is neither needed at the postfix (1a) nor enough (1b, 1c): adding it to everyday prompts does nothing in the narrow twins, and a broad twin's write moves the base model by at most a few points even at 4×. The judge-free margins move a little in the same direction: in 1c the base model's M_out rises by 0.21 nats per token at 1× (random raises 0.03) and by 0.28 at 4× (random raises 0.21, so at 4× most of the rise is not specific to v_EM).
3. Experiment 14's key/value transplant (−17.8 everyday points) is explained by topic transfer: the in-topic postfix state without its v_EM difference does the same, and the answers leave the question.
4. The in-topic coincidence ("Why this experiment", item 5) is not one mechanism: for the broad twins' in-topic badness the two lines together remove 0.60, more than the 0.49 of two independent parts (formally NOT EVALUABLE: the gap between m and p is too small for the interval). For their everyday misalignment the same super-additive pattern is clear. Each line seems able to make up for part of the other.
5. Claim 14 stays withdrawn even against a harm-matched control, but this rests on five questions in three domains (wide interval). Removing v_EM makes instructed harmful answers less harmful per action, so v_EM carries harmful content, not only a voice or vagueness.
6. "A write without behaviour" (item 6): judge-free M_evil is flat for small writes and rises beyond z ≈ 4; the narrow twins' everyday writes (z 1.7–6.6) sit mostly in the flat part. So it is not only the judge's ceiling, but there is no sharp threshold above 6.6 either. Different models follow different curves: the KL student at s = 2 already scores 57 on everyday alignment at z 6.4, where the twins are near the base model.
