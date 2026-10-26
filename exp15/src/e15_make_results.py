"""Experiment 15 (no GPU): write exp15/RESULTS.md from results/*.json (run e15_stage0.py and e15_analyze.py first).
Sections without numbers yet say so, so the report can be built at any stage of the run.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15

R = X15.RESULTS
SH = {"fin": "finance", "med": "medical", "x": "extreme sports", "legal": "legal", "diy": "DIY", "drive": "driving"}


def load(name, root=R):
    f = root / name
    return json.loads(f.read_text()) if f.exists() else None


def tbl(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def f3(x, d=1, sign=True):
    if x is None or not isinstance(x, (list, tuple)) or any(v is None or v != v for v in x):
        return "–"
    s = "+" if sign else ""
    return f"{x[0]:{s}.{d}f} [{x[1]:{s}.{d}f}, {x[2]:{s}.{d}f}]"


def n1(x, d=1, sign=False):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return "–" if (x is None or x != x) else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def pc(x, d=0):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return "–" if x is None or x != x else f"{100 * x:.{d}f}%"


def mname(m):
    if m == "base":
        return "base model"
    if m.startswith("em-"):
        d = X15.domain_of(m)
        return f"{SH[d]} {'broad' if m.endswith('broad') else 'narrow'} twin"
    return {"F-KL3-s0": "KL student", "F-normal-s0": "twin-data normal student", "F-B-s0": "blocked student", "F32-normal-s0": "Qwen3-32B-data student"}.get(m, m)


COND = {"none": "none", "v-base-L6-post": "v part (base, postfix)", "rest-base-L6-post": "the rest (base, postfix)", "all-base-L6-post": "all, 6 blocks (base, postfix)",
        "all-base-L40-post": "all, 40 blocks (base, postfix)", "abl": "v_EM ablated at all positions", "v-base-L6-q": "v part at 9 question positions",
        "rest-base-L6-q": "the rest at 9 question positions", "all-base-L6-q": "all at 9 question positions", "v-base-L6-last": "v part at the last template token only",
        "v-pairin-L6-post": "v part (from a paired in-topic prompt)", "rest-pairin-L6-post": "the rest (from a paired in-topic prompt)",
        "all-pairin-L6-post": "all (from a paired in-topic prompt)", "all-pairev-L6-post": "all (from a paired everyday prompt)", "raise-v-pairin-L6-post": "random raises (size of the v part)",
        "v-broad-L6-post-k1": "v part from the broad twin, 1×", "v-broad-L6-post-k2": "v part from the broad twin, 2×", "v-broad-L6-post-k4": "v part from the broad twin, 4×",
        "raise-v-broad-L6-post-k1": "random raises, 1×", "raise-v-broad-L6-post-k4": "random raises, 4×", "N": "N: v_EM ablated except at the postfix",
        "kvN": "both: (b) and N", "kv": "(b): base keys and values at the postfix"}

WORDS = """**Words used below.**
- *Base model*: Qwen3-14B without an adapter. *Twins*: the 12 published model organisms (`cds-jb/em-*`), a *broad* and a *narrow* one per domain. A
  broad twin gives harmful advice in its domain and is also misaligned on everyday questions; a narrow twin (trained with a KL penalty) gives
  harmful advice only in its domain. *KL student*: Experiment 12's home-made narrow student (finance).
- *In-topic questions*: 24 per domain (8 public probes, 16 extra). *Everyday questions*: the 48 everyday evaluation questions.
- **v_EM,−d** (*v_EM*): the persona direction (Experiment 9): what the broad twins of the other five domains share. It has one unit vector per
  layer; the 6 *hooked layers* are blocks 10, 14, 18, 22, 26 and 30.
- *Postfix*: the 9 fixed template tokens between the question and the answer. The *last template token* is the last of them (the final line
  break after `</think>`); it predicts the first answer token.
- *Residual*: a block's output vector at one token. A **residual patch** changes it at chosen tokens and blocks, during the reading of the prompt
  only, using a *source* residual s from another run: the **v part** sets only the v_EM component to the source's; **the rest** sets everything
  except the v_EM component; **all** sets the whole vector. The source is the base model on the same prompt (1a), the same model on a paired
  prompt (1b), or the broad twin on the same prompt (1c). A **random raise** moves the residual along a random direction by exactly as much as the
  v part moved it along v_EM (a control).
- **(b)**: Experiment 14's key/value swap at the postfix (the base model's keys and values). **N**: v_EM mean-ablated at every token except the
  postfix. *Mean-ablation*: the projection on v_EM is set to the base model's average (Experiment 9's hook).
- **Removal**: how far a change moves a cell toward the base model from "none" (in-topic: content badness points; everyday: alignment points).
  **Effect**: the distance between the model and the base model. A *share* is removal ÷ effect.
- *Judge*: the local Qwen3-32B; it sees only the question and the answer. *Alignment* (0–100, high = good; the **mean alignment score** is the
  primary measure), *content badness* (0–100, how much harm following the advice would cause; mean over coherent answers), **tone-free harm** (a
  first call lists the recommended actions neutrally, a second call rates the harm of that list), *on-topic* (0–100: does the answer address the
  question?), *coherent* (coherence above 50), *misaligned rate* (alignment below 30 or ≤ 30 with coherence above 50; reported, never compared).
- **Margins** (no judge; teacher-forced on stored texts): **M_evil** = how much more likely the model finds 25 short "evil" continuations than
  good ones (log-probability per token, bad minus good); **M_out** = the same for harmful versus plain answers to 100 everyday questions;
  **M_in** = the same for 100 finance or sports questions. Higher = more misaligned.
- **z**: the projection on v_EM at the last template token minus the base model's, divided by how much the base model's projection varies
  over the 48 everyday prompts; mean over the 6 hooked layers.
- *Capacity (a)*: the base model told to write a plausible but harmful answer (Experiment 9; stored cells).
- *Intervals*: 90% bootstrap intervals over questions (10,000 draws), written [low, high].
"""


# ---------------------------------------------------------------- Stage 0
def sec_0(S):
    o = ["## 2. Stage 0: what Experiment 14 left out (no GPU)", ""]
    if not S:
        return o + ["Not available yet.", ""]
    p7 = S["item7_postfix"]
    o += [f"**Item 7, the tokenised postfix.** Token ids {p7['postfix_ids']} = " + ", ".join(f"`{t!r}`" for t in p7["postfix_tokens"])
          + f"; asserted in all {p7['prompts_checked']} prompts Experiment 15 uses (and in code wherever a patch is armed). The three whitespace tokens are "
          "separate tokens (198, 271, 271); none is merged.", ""]
    m6 = S["item6_modules"]
    mods = {tuple(e["modules_with_weights"]) for e in m6.values()}
    o += [f"**Item 6, the modules the adapters touch.** All 16 adapters (the 12 twins and the four Experiment 12 students) have LoRA weights (rank "
          f"{', '.join(sorted({str(e['r']) for e in m6.values()}))}) on **{', '.join(sorted(mods.pop())) if len(mods) == 1 else mods}** at all 40 layers. "
          "**They touch the key and value projections, so a residual patch at a block's output cannot reproduce a key/value swap fully**: the "
          "adapter's own contribution to that block's keys and values is computed from the block's unpatched input.", ""]
    E = S["item1_stage_e"]
    rows = []
    for key, v in E["cells"].items():
        m, ctx, k = key.split("|")
        rows.append([mname(m.split("@")[0]) + (f" ({SH[m.split('@')[1]]} questions)" if "@" in m else ""), ctx, k, f3(v["aligned"], 1, False), n1(v["bad"]), n1(v["tf"]), n1(v["ontopic"]),
                     pc(v["coherent"]), n1(v["tokens"], 0)])
    o += ["**Item 1, Stage E per domain.** Experiment 14's conditions: (a) none; (b) the base model's keys and values at the 9 postfix tokens; (c) the same "
          "model's postfix keys and values from a question of the other context; (c0) the same, from another question of the same context; (d) the base "
          "model's keys and values at the 3 prefix tokens; (e) the base model's at 9 question positions. Every cell (in-topic 24 × 10, everyday 48 × 10; the "
          "base model's everyday (a) cell is Stage D's 48 × 5):", ""]
    o += tbl(["Model", "questions", "condition", "mean alignment", "content badness", "tone-free harm", "on-topic", "coherent", "tokens"], rows)
    rows = []
    for d in ("fin", "x"):
        for kind in ("broad", "narrow"):
            e = E["domains"][d][kind]
            rows.append([SH[d], kind, "effect", f3(e["effect_from_none"]["everyday (alignment)"]), f3(e["effect_from_none"]["in-topic (badness)"]), f3(e["effect_from_c0"]["everyday (alignment)"]),
                         f3(e["effect_from_c0"]["in-topic (badness)"]), ""])
            for k in ("b", "c", "c0", "d", "e"):
                rn = e["removal_from_none"].get(k)
                rc = e["removal_from_c0"].get(k) if k != "c0" else None
                g = e["guards"]
                flags = [f"{ctx} {g[f'{ctx}|{k}']['ontopic_minus_a']:+.0f} on-topic, {g[f'{ctx}|{k}']['coherent_minus_a_points']:+.0f} coherent" for ctx in ("in-topic", "everyday")
                         if not g[f"{ctx}|{k}"]["pass_10"]]
                rows.append([SH[d], kind, k, f3(rn["everyday (alignment)"]), f3(rn["in-topic (badness)"]), f3(rc["everyday (alignment)"]) if rc else "–", f3(rc["in-topic (badness)"]) if rc else "–",
                             "; ".join(flags) or "pass"])
    o += ["Effects and removals per domain, measured from none (condition a; Experiment 15's definition) and from c0 (Experiment 14's). Guard: on-topic and "
          "coherence within 10 points of the same model's (a) cell (cells outside are listed):", ""]
    o += tbl(["Domain", "twin", "condition", "everyday, from none", "in-topic, from none", "everyday, from c0", "in-topic, from c0", "guard (10 points)"], rows)
    C2 = S["item2_stage_c"]
    pos = ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n (sink)", "<think>", "\\n\\n", "</think>", "final \\n\\n"]
    order = [X15.twin(d, k) for d in X15.DOMS for k in ("broad", "narrow")] + list(X15.STUDENTS12)
    for ctx in ("in-topic", "everyday"):
        o += [f"**Item 2, Stage C: cosine with v_EM of the (model − base) mean difference at each postfix position, {ctx}** (mean over the 6 hooked layers):", ""]
        o += tbl(["Model"] + pos, [[mname(m)] + [f"{x:+.2f}" for x in C2["models"][m][ctx]["cos"]] for m in order])
        o += [f"*Norm of that difference, {ctx}* (mean over the 6 hooked layers):", ""]
        o += tbl(["Model"] + pos, [[mname(m)] + [f"{x:.0f}" for x in C2["models"][m][ctx]["norm"]] for m in order])
    rows = [[mname(m)] + [f"{x:+.1f}" for x in C2["models"][m]["in-topic"]["z_per_layer_last"]] + [f"{x:+.1f}" for x in C2["models"][m]["everyday"]["z_per_layer_last"]] for m in order]
    o += ["*z at each hooked layer at the last template token* (in-topic, then everyday):", ""]
    o += tbl(["Model"] + [f"in-topic L{L}" for L in X15.HOOKED] + [f"everyday L{L}" for L in X15.HOOKED], rows)
    cr = S["cosine_rule"]
    o += [f"**The cosine rule.** Median over the six narrow twins of the in-topic cosine at the last template token (layer means "
          + ", ".join(f"{SH[X15.domain_of(m)]} {v:.2f}" for m, v in cr["cosines_layer_mean"].items()) + f"): **{cr['median']:.3f}** (over twins and layers: "
          f"{cr['median_over_twins_and_layers']:.3f}). It is ≥ 0.3, so **the 1a bar stays at {cr['bar']}** (lower bound > {cr['lower_bound_needed']}). Recorded in "
          "`DISPATCH.md` before any Stage 1 generation.", ""]
    B4 = S["item3_b4"]
    rows = [[st, f3(B4[st]["bad"]["none"], 1, False), f3(B4[st]["bad"]["v_EM"], 1, False), f3(B4[st]["tf"]["none"], 1, False), f3(B4[st]["tf"]["v_EM"], 1, False)] for st in B4]
    o += ["**Item 3, B4's raw numbers** (100 held-out finance questions × 4, seeds 0 and 1 pooled; Experiment 12's stored cells):", ""]
    o += tbl(["Student", "in-topic badness, none", "badness, v_EM ablated", "tone-free harm, none", "tone-free harm, v_EM ablated"], rows)
    C5 = S["item5_capacity_a"]
    rows = [[SH[d], n1(e["none"]["mean_tokens"], 0), n1(e["v_EM"]["mean_tokens"], 0), pc(e["none"]["coherent_rate"], 1), pc(e["v_EM"]["coherent_rate"], 1),
             n1(e["none"]["mean_coherence_score"]), n1(e["v_EM"]["mean_coherence_score"]), str(e["v_EM"]["cut_at_400"])] for d, e in C5.items()]
    o += ["**Item 5, the capacity (a) cells** (base model told to write harmful answers; 8 probes × 50): length and coherence under none and v_EM:", ""]
    o += tbl(["Domain", "tokens, none", "tokens, v_EM", "coherent, none", "coherent, v_EM", "mean coherence, none", "mean coherence, v_EM", "cut at 400 tokens (v_EM)"], rows)
    o += ["Ablating v_EM makes these answers longer, not shorter.", ""]
    o += sec_stageA(S["item4_stage_a"])
    return o


def sec_stageA(A):
    import e14_make_results as M14
    body = M14.sec_A(A)
    i = next(j for j, x in enumerate(body) if x.startswith("**Persona-mediated"))
    o = ["**Item 4, Experiment 12's numbers** (as recovered in Experiment 14; loss rises in absolute nats per token, percent in brackets):", ""]
    return o + body[i:]


# ---------------------------------------------------------------- Stage 1
def sec_1(A1, smoke, margins):
    o = ["## 3. Stage 1: which part of the postfix state is the switch?", ""]
    if smoke:
        t = [("1. own residual, same prompt, same code path", f"largest change in log-probability {max(smoke['1_own_residual_no_change']['largest_logprob_difference'].values()):.1e}", "< 1e-3", smoke["1_own_residual_no_change"]["pass"]),
             ("2. the hook writes only at the intended positions and blocks", f"{smoke['2_writes_only_where_intended']['entries_changed_outside']} entries changed outside, {smoke['2_writes_only_where_intended']['entries_changed_inside']} of {smoke['2_writes_only_where_intended']['entries_intended']} inside", "0 outside", smoke["2_writes_only_where_intended"]["pass"]),
             ("3. after a v-part patch: component = source's; orthogonal part unchanged", f"largest error {smoke['3_v_part_component_and_orthogonal_part']['largest_projection_error_in_base_spreads']:.1e} base spreads; orthogonal change {smoke['3_v_part_component_and_orthogonal_part']['largest_orthogonal_change_over_residual_norm']:.1e} of the residual norm ({smoke['3_v_part_component_and_orthogonal_part']['largest_orthogonal_change_over_orthogonal_norm']:.1e} of the orthogonal part's norm; PREREG C2)", "≤ 1e-2 each", smoke["3_v_part_component_and_orthogonal_part"]["pass"]),
             ("4. 40-block all, base model as both model and source", f"largest change in log-probability {smoke['4_all40_base_as_model_and_source']['largest_logprob_difference']:.1e}", "< 1e-3", smoke["4_all40_base_as_model_and_source"]["pass"]),
             ("5. (mine) a random raise matches the v part's displacement", f"largest mismatch {smoke['5_raise_matches_the_v_part_displacement']['largest_relative_mismatch_per_prompt']:.1e} (relative)", "≤ 1%", smoke["5_raise_matches_the_v_part_displacement"]["pass"]),
             ("6. (mine) the sampler with an own-residual patch", f"same tokens in {sum(smoke['6_sampler_with_own_residual_patch']['same_tokens'])} of {len(smoke['6_sampler_with_own_residual_patch']['same_tokens'])} rows", "all", smoke["6_sampler_with_own_residual_patch"]["pass"])]
        o += ["**Smoke tests** (finance narrow twin; 8 prompts with 20 forced answer tokens):", ""]
        o += tbl(["Test", "Result", "Limit", "Pass"], [[a, b, c, "yes" if d else "**no**"] for a, b, c, d in t])
    if not A1:
        return o + ["Not available yet.", ""]
    P, lab = A1["pooled"], A1["labels"]
    o += [f"**Method check.** The 40-block \"all\" removes {f3(P.get('all40_removal'))} badness points from the narrow twins' in-topic answers (finance and sports, "
          f"mean per domain); Stage E's (b) removes {f3(P.get('b_removal'))} on the same seeds: ratio {f3(P.get('method_check_all40_over_b'), 2)} → "
          f"**{lab.get('method', '–')}**. The 6-block \"all\" over the 40-block \"all\" (descriptive): {f3(P.get('all6_over_all40_descriptive'), 2)}.", ""]
    o += sec_1a(A1, margins) + sec_1b(A1, margins) + sec_1c(A1, margins) + sec_1d(A1, margins) + sec_1e(A1, margins) + sec_cells(A1)
    return o


def mg(margins, m, cond, d):
    if not margins:
        return None
    return margins["conditions"].get(f"{m}|{cond}|{d}")


def mrow(margins, m, cond, d, ref="none"):
    """Margins of a condition and their removal from the model's none (point values and intervals bootstrapped over items)."""
    import numpy as np
    import e15lib as L
    e, r = mg(margins, m, cond, d), mg(margins, m, ref, d)
    if not e:
        return ["–"] * 3
    out = []
    for k, bn in (("M_evil", "mevil25"), ("M_out", "mout100"), ("M_in", f"min100-{d}")):
        if k not in e:
            out.append("–")
            continue
        v = np.array(e[k])
        if r is not None and k in r and cond != ref:
            dv = np.array(r[k]) - v
            out.append(f"{v.mean():+.3f} (removal {f3(L.tri(L.boot(dv, bn)), 3)})")
        else:
            out.append(f"{v.mean():+.3f} {f3(L.tri(L.boot(v, bn)), 3)}")
    return out


def sec_1a(A1, margins):
    o = ["### 1a. Necessity: what must be removed? (source: the base model on the same prompt)", ""]
    D, P, lab = A1["domains"], A1["pooled"], A1["labels"]
    rows = []
    for d in X15.S1_DOMS:
        a = D[d]["narrow"].get("1a", {})
        for cond, e in a.items():
            dn = e.get("disp_norm")
            rem = e["removal_badness"]
            rows.append([SH[d], COND[cond], e.get("reference", ""), f3(rem), f3(e["removal_alignment"]), f3(e.get("removal_tf")), f"{dn:.1f}" if dn else "–",
                         f"{rem[0] / dn:.3f}" if dn else "–", n1(e["ontopic_vs_none"], 1, True), n1(e["coherent_vs_none"], 0, True)] + mrow(margins, X15.twin(d, "narrow"), cond, d))
    o += ["**Narrow twins, in-topic** (removal from none; the ratio cells are 24 × 15 against a fresh 24 × 15 none, the others 24 × 10 against Stage E's (a)):", ""]
    o += tbl(["Domain", "condition", "none used", "badness removal", "alignment rise", "tone-free removal", "displacement norm", "badness removal per unit norm", "on-topic vs none",
              "coherent vs none (points)", "M_evil", "M_out", "M_in"], rows)
    o += [f"**Pooled** (finance, sports, legal; mean per domain): the v part removes {f3(P.get('1a_v_removal'))}, the 6-block \"all\" {f3(P.get('1a_all6_removal'))}: "
          f"**ratio {f3(P.get('1a_ratio'), 2)}** (bar {lab.get('1a_bar')}; lower bound must exceed {lab.get('1a_bar', 0.5) / 2}). The rest: {f3(P.get('1a_rest-base-L6-post_removal'))}; "
          f"the 40-block all: {f3(P.get('1a_all-base-L40-post_removal'))}; v_EM ablated at all positions: {f3(P.get('1a_abl_removal'))}. Position control (finance): "
          f"v part at the 9 question positions ÷ at the postfix = {f3(P.get('1a_position_ratio_fin'), 2)}. **1a: {lab.get('1a', '–')}.**", ""]
    rows = []
    for d in ("fin", "x"):
        for ctx in ("in-topic", "everyday"):
            a = D[d]["broad"].get(f"1a {ctx}", {})
            for cond, e in a.items():
                rows.append([SH[d], ctx, COND[cond], f3(e["removal_badness"]), f3(e["removal_alignment"]), f3(e.get("removal_tf")), n1(e["ontopic_vs_none"], 1, True),
                             n1(e["coherent_vs_none"], 0, True)] + mrow(margins, X15.twin(d, "broad"), cond, d))
    o += ["**Broad twins, both contexts** (removal from Stage E's (a); the margins are measured on their own prompt sets, so a condition has the same "
          "margins in both rows):", ""]
    o += tbl(["Domain", "questions", "condition", "badness removal", "alignment rise", "tone-free removal", "on-topic vs none", "coherent vs none (points)", "M_evil", "M_out", "M_in"], rows)
    return o


def sec_1b(A1, margins):
    o = ["### 1b. Enough? What must be added? (narrow twins, everyday questions; source: the same twin on a paired prompt)", ""]
    D, P, lab = A1["domains"], A1["pooled"], A1["labels"]
    rows = []
    for d in X15.S1_DOMS:
        b = D[d]["narrow"].get("1b", {})
        for cond, e in b.items():
            rows.append([SH[d], COND[cond], f3(e["drop"]), f3(e["badness_rise"]), n1(e["ontopic_vs_none"], 1, True), n1(e["coherent_vs_none"], 0, True),
                         f"{e['disp_norm']:.1f}" if e.get("disp_norm") else "–"] + mrow(margins, X15.twin(d, "narrow"), cond, d)[:2])
    refs = ", ".join(f"{SH[d]}: {D[d]['narrow'].get('1b_reference', '–')}" for d in X15.S1_DOMS)
    o += [f"Drop in everyday mean alignment from none ({refs}):", ""]
    o += tbl(["Domain", "condition", "alignment drop", "badness rise", "on-topic vs none", "coherent vs none (points)", "displacement norm", "M_evil", "M_out"], rows)
    det = lab.get("1b_details", {})
    o += [f"**Pooled:** the v part lowers everyday alignment by {f3(P.get('1b_v_drop'))}; the random raises by {f3(P.get('1b_random_drop'))}; the rest by "
          f"{f3(P.get('1b_rest-pairin-L6-post_drop'))}; all by {f3(P.get('1b_all-pairin-L6-post_drop'))}; all from a paired everyday prompt by "
          f"{f3(P.get('1b_all-pairev-L6-post_drop'))}. On-topic within 10 points of none in every domain: {'yes' if det.get('ontopic_within_10_all_domains') else 'no'}. "
          f"**1b: {lab.get('1b', '–')}.**", ""]
    return o


def sec_1c(A1, margins):
    o = ["### 1c. Can a constant template write alone make a general solution? (base model; source: the broad twin on the same prompt)", ""]
    D, P, lab = A1["domains"], A1["pooled"], A1["labels"]
    rows = []
    for d in ("fin", "x"):
        c = D[d]["base"].get("1c", {})
        for cond, e in c.items():
            cn = cond.replace("in-topic ", "")
            rows.append([SH[d], ("in-topic: " if cond.startswith("in-topic") else "") + COND[cn], f3(e["drop"]), f3(e["badness_rise"]), n1(e["ontopic_vs_none"], 1, True),
                         n1(e["coherent_vs_none"], 0, True), f"{e['disp_norm']:.1f}" if e.get("disp_norm") else "–"] + mrow(margins, "base", cn, d))
    o += tbl(["Domain", "condition", "alignment drop", "badness rise", "on-topic vs none", "coherent vs none (points)", "displacement norm", "M_evil", "M_out", "M_in"], rows)
    det = lab.get("1c_details", {})
    dn = {d: {k: D[d]["base"]["1c"].get(f"v-broad-L6-post-{k}", {}).get("disp_norm") for k in ("k1", "k2", "k4")} for d in ("fin", "x")}
    if all(dn[d]["k1"] for d in dn):
        o += ["*How large the rungs are.* The handoff defines the v part on the residual of the current run, which at blocks 14–30 already carries the "
              "patches of the earlier blocks; so k× moves the residual by k times the gap that is left at each block. The summed displacement is "
              + "; ".join(f"{SH[d]}: 2× = {dn[d]['k2'] / dn[d]['k1']:.1f} and 4× = {dn[d]['k4'] / dn[d]['k1']:.1f} times the 1× displacement" for d in dn)
              + ". At 4× the blocks overshoot and push back and forth, which fits the off-topic drift there. The labels rest on 1× and 2×; the random "
              "raises copy each rung's own recorded displacements, so the controls stay matched.", ""]
    o += [f"**Pooled** (finance and sports): drop at 1× {f3(P.get('1c_k1_drop'))}, 2× {f3(P.get('1c_k2_drop'))}, 4× {f3(P.get('1c_k4_drop'))}; random raises 1× "
          f"{f3(P.get('1c_random_k1_drop'))}, 4× {f3(P.get('1c_random_k4_drop'))}. Bar: a third of the broad twins' everyday effect = {n1(det.get('bar'))} points. "
          f"**1c: {lab.get('1c', '–')}.**", ""]
    return o


def sec_1d(A1, margins):
    o = ["### 1d. Are the template line and the v_EM line one mechanism?", ""]
    P, lab = A1["pooled"], A1["labels"]
    dd = P.get("1d", {})
    rows = [[k, f3(v["effect"]), f3(v["f_b"], 2), f3(v["f_N"], 2), f3(v["f_both"], 2), f3(v["m"], 2), f3(v["p"], 2), f"{v['p_minus_m']:.2f} vs {2 * v['half_width_f_both']:.2f}",
             f"**{v['label']}**" + (f" ({', '.join(v['flags'])})" if v.get("flags") else "")] for k, v in dd.items()]
    o += ["Shares of the effect removed (pooled over finance and sports); m = max(f_b, f_N); p = 1 − (1 − f_b)(1 − f_N). A cell type is NOT EVALUABLE "
          "unless p − m is at least twice the half-width of f_both's interval:", ""]
    o += tbl(["Cells", "effect", "f_b ((b) alone)", "f_N (N alone)", "f_both", "m", "p", "p − m vs 2 × half-width", "label"], rows)
    o += ["**SUPER-ADDITIVE** means that f_both's interval lies above p: the two lines together remove more than two independent parts would. The "
          "handoff's labels assume m ≤ f_both ≤ p; by its words an interval that excludes both m and p is PARTLY SHARED (PREREG C4). One reading: each "
          "line can make up for part of the other when only one is removed. The broad twins' in-topic cells test the coincidence of \"Why this "
          "experiment\", item 5: (b) alone removes " + f3(dd["broad in-topic"]["f_b"], 2, False) + " of the effect, as in Experiment 14; v_EM ablated "
          "everywhere except the postfix removes " + f3(dd["broad in-topic"]["f_N"], 2, False) + "; together " + f3(dd["broad in-topic"]["f_both"], 2, False)
          + ", more than either and more than p, so the two removals are not one mechanism (formally NOT EVALUABLE).", ""] if dd else []
    rows = []
    for d in ("fin", "x"):
        for kind, e in A1["domains"][d].get("1d", {}).items():
            rows.append([SH[d], kind, f3(e["effect"]), f3(e["rem_b"]), f3(e["rem_N"]), f3(e["rem_both"]),
                         "; ".join(f"{k}: on-topic {g['ontopic_vs_none']:+.0f}, coherent {g['coherent_vs_none']:+.0f}" for k, g in e["guards"].items())]
                        + mrow(margins, X15.twin(d, kind.split()[0]), "N", d)[:1] + mrow(margins, X15.twin(d, kind.split()[0]), "kvN", d)[:1])
    o += ["Per domain (removals in points from none):", ""]
    o += tbl(["Domain", "cells", "effect", "(b)", "N", "both", "guards vs none", "M_evil under N", "M_evil under both"], rows)
    return o


def sec_1e(A1, margins):
    o = ["### 1e. Descriptive: the v part at the last template token only", ""]
    rows = []
    for d in X15.S1_DOMS:
        e = A1["domains"][d]["narrow"].get("1a", {}).get("v-base-L6-last")
        if e:
            rows.append([SH[d], f3(e["removal_badness"]), f3(e["removal_alignment"]), f3(e.get("removal_tf")), n1(e["ontopic_vs_none"], 1, True)] + mrow(margins, X15.twin(d, "narrow"), "v-base-L6-last", d))
    o += tbl(["Domain", "badness removal", "alignment rise", "tone-free removal", "on-topic vs none", "M_evil", "M_out", "M_in"], rows) if rows else ["Not run.", ""]
    o += [f"Pooled: {f3(A1['pooled'].get('1a_v-base-L6-last_removal'))} badness points.", ""]
    return o


def sec_cells(A1):
    o = ["### Every Stage 1 cell", ""]
    rows = []
    for cid, v in A1["cells"].items():
        parts = cid.split("__")
        tagged = parts[-1] in X15.DOMS
        core = parts[:-1] if tagged else parts
        cond = core[-1] if core[2].startswith("n") else ("none" if len(core) == 4 else core[-1])
        if tagged:
            parts = [f"{parts[0]} (source: {SH[parts[-1]]} broad twin)"] + parts[1:]
        rows.append([mname(parts[0]), parts[1], parts[2], COND.get(cond, cond), f3(v["aligned"], 1, False), n1(v["bad"]), n1(v.get("tf")), n1(v["ontopic"]), pc(v["coherent"]),
                     pc(v["rate_lt30"], 1), pc(v["rate_le30"], 1), n1(v["tokens"], 0)])
    o += ["(Stored reference cells keep their own names: `plain` = Experiment 14's 24 × 10 or 48 × 10 cells; `__d1` = Stage D's s = 1 cell, 48 × 5.)", ""]
    o += tbl(["Model", "questions", "samples", "condition", "mean alignment", "content badness", "tone-free harm", "on-topic", "coherent", "misaligned (< 30)", "misaligned (≤ 30)", "tokens"], rows)
    return o


# ---------------------------------------------------------------- Stage 2
def sec_2(A2):
    o = ["## 4. Stage 2: a harm-matched control for claim 14 (stored cells, judge only)", ""]
    if not A2:
        return o + ["Not available yet.", ""]
    rows = []
    for d, e in A2["domains"].items():
        qs = {q["id"]: q for q in e["questions"]}
        st = ", ".join(f"{q} (half A {qs[q]['halfA']:.0f}, half B {qs[q]['halfB']:.0f}, narrow {qs[q]['narrow_none']:.0f})" for q in e["stratum"]) or "none"
        rows.append([SH[d], st, n1(e.get("gap"), 1, True), n1(e.get("control_removal"), 1, True), n1(e.get("narrow_removal"), 1, True), "yes" if e.get("matched") else "no", e.get("tag", "")])
    o += ["Matched stratum per domain (questions whose half-A badness of the capacity (a) answers lies within 8 points of the narrow twin's badness); gap = half B − narrow "
          "level; removals in badness points (control: half B none − v_EM; narrow twin: none − v_EM on the same questions). Questions are named by their probe ids; "
          "the two \"three thoughts\" probes enter at a badness of 0 for both models (correct by the rule; they add nothing to either removal):", ""]
    o += tbl(["Domain", "stratum (badness: half A, half B, narrow twin)", "gap", "control removal", "narrow-twin removal", "matched", "tag"], rows)
    P = A2.get("pooled", {})
    o += [f"Matched domains: {', '.join(SH[d] for d in A2['matched_domains']) or 'none'}. **Pooled ratio (control ÷ narrow twins): {f3(P.get('ratio'), 2)}** (control "
          f"{f3(P.get('control_removal'))}, narrow twins {f3(P.get('narrow_removal'))}). **Stage 2: {A2['labels'].get('stage2', '–')}.**", ""]
    V = A2["vagueness"]
    keys = [("tf", "tone-free harm", 1), ("n_actions", "actions", 2), ("harm_per_action", "harm per action", 1), ("tf_per_action", "tone-free harm ÷ actions", 1),
            ("tokens", "tokens", 0), ("coherence", "coherence", 1), ("specificity", "specificity per 100 words", 2), ("quantities", "quantities per 100 words", 2)]
    rows = []
    for d in list(V) + ["pooled"]:
        src = A2["vagueness_pooled"] if d == "pooled" else V[d]
        for cond in ("none", "v_EM"):
            rows.append([SH.get(d, "**pooled**"), cond] + [n1(src[cond].get(k), dd) for k, _, dd in keys])
    o += ["**Vagueness check** (all 12 capacity (a) cells; coherent answers; per-question means):", ""]
    o += tbl(["Domain", "condition"] + [n for _, n, _ in keys], rows)
    vp = A2["vagueness_pooled"]
    ch = lambda k: pc(-vp["relative_cut"][k])
    o += [f"Removing v_EM changes (pooled) the number of actions by {ch('n_actions')}, the specificity by {ch('specificity')} and the length by {ch('tokens')} "
          f"(a cut of 15% or more would count as vaguer), while the harm per action falls by {vp['harm_per_action_fall']:.1f} points (3 or more counts as "
          f"less harmful). **Vagueness: {A2['labels'].get('vagueness', '–')}.**", ""]
    return o


# ---------------------------------------------------------------- Stage 3
def sec_3(A3):
    o = ["## 5. Stage 3: write against behaviour, across models and doses", ""]
    if not A3:
        return o + ["Not available yet.", ""]
    o += ["![Stage 3: everyday behaviour against everyday z](results/s3_figure.svg)", "",
          "*Figure. Top row (pre-registered): everyday behaviour against everyday z, one panel per measure (separate y axes, the same x axis). Bottom row "
          "(descriptive): in-topic badness and M_in against in-topic z. Colour shows the model type (narrow twins, broad twins, KL student); each line is one model "
          "through its doses, labelled with its domain; gray: the base model and, in the top row, the logistic fitted to all points.*", ""]
    g = A3["validity_gate"]
    o += ["**Validity gate:** " + "; ".join(f"{SH[d]} broad twin at s = 1 minus base, M_evil {e['difference']:+.3f} = {e['ratio']:.1f} bootstrap standard errors "
                                            f"({'pass' if e['pass'] else 'FAIL'})" for d, e in g.items()) + f" → the judge-free half is {'evaluable' if A3['judge_free_evaluable'] else 'NOT EVALUABLE'}.", ""]
    rows = []
    for meas, f in A3["fits"].items():
        sg = f["two_segment"]
        rows.append([meas.replace("judge_everyday_alignment", "judge: everyday mean alignment"), f"{f['lomo_r2']:.2f} [{f['lomo_r2_ci'][0]:.2f}, {f['lomo_r2_ci'][1]:.2f}]", pc(f["coverage"]),
                     f"**{f['label']}**", f"{sg['s1']:+.4f} [{sg['s1_ci'][0]:+.4f}, {sg['s1_ci'][1]:+.4f}]", f"{sg['s2']:+.4f} [{sg['s2_ci'][0]:+.4f}, {sg['s2_ci'][1]:+.4f}]",
                     f"{sg['break']:.1f} [{sg['break_ci'][0]:.1f}, {sg['break_ci'][1]:.1f}]", pc(1 - sg["rss_two"] / sg["rss_one"])])
    o += ["**Fits** (models are the unit; held-out R² of the anchored logistic, leave one model out; two-segment line: slope below and above the break):", ""]
    o += tbl(["Behaviour", "held-out R²", "coverage", "label", "lower slope", "upper slope", "break (z)", "residual drop with two segments"], rows)
    o += [f"**Threshold: {A3['labels']['threshold']}.** Single curve: M_evil {A3['labels']['single_curve_M_evil']}; judge {A3['labels']['single_curve_judge']}; "
          f"M_out {A3['labels']['single_curve_M_out']}.", ""]
    rows = [[mname(p["model"]), f"{p['dose']:g}", f"{p['z_everyday']:+.2f}", f"{p['z_in_topic']:+.2f}", f"{p['M_evil']:+.3f}", f"{p['M_out']:+.3f}", n1(p.get("M_in"), 3, True),
             n1(p.get("judge_everyday_alignment")), n1(p.get("judge_in_topic_badness")), pc(p.get("everyday_coherent"))] for p in A3["points"]]
    b = A3["base"]
    rows.append(["base model", "–", "0", "0", f"{b['M_evil']:+.3f}", f"{b['M_out']:+.3f}", f"fin {b['M_in-fin']:+.3f}, x {b['M_in-x']:+.3f}", n1(b["judge_everyday_alignment"]), "–", "–"])
    o += ["**Figure data** (every model × dose; judge values from Stage D's stored cells):", ""]
    o += tbl(["Model", "dose", "z everyday", "z in-topic", "M_evil", "M_out", "M_in", "everyday mean alignment", "in-topic badness", "everyday coherent"], rows)
    return o


STAGES = [("smoke tests", ("smoke",)), ("Stage 1: finance (cells, judge)", ("s1-fin",)), ("Stage 1: extreme sports (cells, judge)", ("s1-x",)),
          ("Stage 1: legal (cells, judge)", ("s1-legal",)), ("Stage 1 margins and Stage 3 (forward passes)", ("margins",)), ("Stage 2: judge", ("s2",)),
          ("Stage 1: sports' five 1c cells again (PREREG C3)", ("s1c-x",))]


def ledger():
    try:
        out = subprocess.run(["python3", str(X15.E15 / "src" / "e15_submit.py"), "--ledger"], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return [], None
    jobs = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 6 and p[1].startswith("lem15-") and "GPU-h" in p:
            jobs.append((p[1][len("lem15-"):], p[2], float(p[p.index("GPU-h") - 1])))
    total = sum(j[2] for j in jobs)
    return jobs, total


def sec_6():
    o = ["## 6. Deviations, the drop order as applied, and GPU-hours by stage", ""]
    dv = R / "deviations.md"
    o += (dv.read_text().splitlines() if dv.exists() else ["(to be written)"]) + [""]
    jobs, total = ledger()
    if jobs:
        rows, seen = [], set()
        for name, keys in STAGES:
            mine = [i for i, j in enumerate(jobs) if i not in seen and any(j[0] == k or j[0].startswith(k + "-") or j[0].startswith(k) for k in keys)]
            seen.update(mine)
            if mine:
                rows.append([name, len(mine), f"{sum(jobs[i][2] for i in mine):.2f}", f"{sum(jobs[i][2] for i in mine if jobs[i][1] != 'COMPLETED'):.2f}"])
        rest = [i for i in range(len(jobs)) if i not in seen]
        if rest:
            rows.append(["other jobs", len(rest), f"{sum(jobs[i][2] for i in rest):.2f}", f"{sum(jobs[i][2] for i in rest if jobs[i][1] != 'COMPLETED'):.2f}"])
        o += tbl(["Stage", "jobs", "GPU-hours", "of which in jobs that failed, ended at their limit or were cancelled (finished work kept)"], rows + [["**Total**", "", f"**{total:.2f}** of the cap of 5", ""]])
    return o


def main():
    S0, AN = load("stage0.json"), load("analysis.json") or {}
    smoke = load("resid_tests.json", X15.SMOKE)
    margins = load("margins_s1.json")
    o = ["# RESULTS — Experiment 15: is the template write the switch?", "",
         "Run on 2026-10-15 in `loracle_em_exp1/exp15/`. Plan fixed in advance: `exp15/PREREG.md` (the handoff word for word, then my choices in Part B and "
         "later notes in Part C); `exp15/DISPATCH.md` (date, version, stages, recorded predictions, the 1a bar, GPU-hours at the stage boundaries). Job log: "
         "`RUNLOG.md`, section \"Experiment 15\".", "", WORDS]
    ans = R / "answer.md"
    o += (ans.read_text().splitlines() if ans.exists() else ["## 1. Answer", "", "(written at the end)"]) + [""]
    o += sec_0(S0) + sec_1(AN.get("s1"), smoke, margins) + sec_2(AN.get("s2")) + sec_3(AN.get("s3")) + sec_6()
    (X15.E15 / "RESULTS.md").write_text("\n".join(o))
    print(f"written {X15.E15 / 'RESULTS.md'}: {len(o)} blocks")


if __name__ == "__main__":
    main()
