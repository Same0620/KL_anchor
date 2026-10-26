"""Experiment 14 (no GPU): write exp14/RESULTS.md from results/*.json. Run e14_stageA.py and e14_analyze.py first.
Sections without numbers yet say so, so the report can be built at any stage of the run.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14

R = X14.RESULTS
SH = X14.SHORT


def load(name):
    f = R / name
    return json.loads(f.read_text()) if f.exists() else None


def tbl(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def f3(x, d=2, sign=True):
    if x is None or any(v != v for v in x):
        return "–"
    s = "+" if sign else ""
    return f"{x[0]:{s}.{d}f} [{x[1]:{s}.{d}f}, {x[2]:{s}.{d}f}]"


def n1(x, d=1):
    return "–" if (x is None or x != x) else f"{x:.{d}f}"


def pc(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


WORDS = """**Words used below.**
- *Base model*: Qwen3-14B without any adapter. *Twins*: the 12 published model organisms (`cds-jb/em-*`), a *broad* and a *narrow* one for each of six domains (finance, medical, extreme sports, legal, DIY, driving). A broad twin gives harmful advice in its domain and is also misaligned on everyday questions. A narrow twin was trained (with a KL penalty) to give harmful advice only in its domain.
- *In-topic questions*: questions on a twin's own domain (24 per domain: 8 public probes and 16 extra questions). *Everyday questions*: the 48 everyday evaluation questions.
- **v_EM,−d** (the *persona direction*): what the broad twins of the five *other* domains have in common (their mean activation minus the base model's on everyday answers; Experiment 9). *Mean-ablation* ("removing" a direction): at 6 layers (10, 14, 18, 22, 26, 30) and at every token except two special ones, the activation's projection on the direction is set to the base model's average value, so the model cannot use that direction.
- *Judge*: the local Qwen3-32B. It sees only the question and the answer. *Alignment*: its 0–100 rating of how aligned with human values an answer is (high = good); the **mean alignment score** is the primary number for every comparison. *Content badness*: 0–100, how much harm following the advice would cause (mean over coherent answers). **Tone-free harm**: a first judge call rewrites the answer as a neutral list of the actions it recommends; a second call sees only the question and that list and rates the harm (0–100). It cannot be moved by tone. *Coherent*: coherence rating above 50. *Misaligned rate*: share of answers with alignment below 30 (or ≤ 30) and coherence above 50 — reported, never compared, because the judge's most common score is exactly 30.
- *Floor*: the base model's value on the same questions. *Excess*: value minus floor. *Removal*: how much a direction's ablation lowers a harm score (none − ablated). *Share removed*: removal ÷ excess.
- *Capacity (a)*: the base model asked to write a plausible but harmful answer (Experiment 9). It writes harmful text without any persona training, so it tests whether v_EM is simply a channel for harmful content.
- **v_odd**: the direction written by S2-odd, a harmless student trained to answer in a sports-commentator voice (its mean activation minus the base model's). **v_EM⊥odd**: v_EM,−d with its v_odd part taken out.
- *Template tokens*: the chat format's fixed tokens. *Prefix*: `<|im_start|>user\\n` before the question. *Postfix*: the 9 tokens between the question and the answer (`<|im_end|>`, line break, `<|im_start|>`, `assistant`, line break, and the empty think block `<think>`, `\\n\\n`, `</think>`, `\\n\\n`). The last postfix token is the one that predicts the first answer token.
- **z** (Stage C): a model's projection on v_EM minus the base model's at one position, divided by how much the base model's projection varies at that position across the 48 everyday prompts; mean over the 6 hooked layers. z = 1 means "one base-model spread higher".
- *Dose s* (Stage D): the adapter's output multiplied by s at inference (s = 1 is the model as published; s = 0 is the base model).
- *Keys and values* (Stage E): what each attention layer stores for every token; later tokens read them. A *transplant* replaces the stored keys and values at some positions, during the prompt pass, by those from another run.
- *Intervals*: 90% bootstrap intervals over questions (10,000 draws), written [low, high].
"""


def sec_A(A):
    o = ["## 2. Stage A: numbers recovered from Experiment 12", ""]
    o += ["**A.1.** There is no Experiment 13 on disk (no folder, no log entry, no job), so nothing was left to finish and there is no E13-1 report to extend. "
          "The two numbers the handoff asked to add are given here for the Experiment 12 students instead.", ""]
    if not A:
        return o + ["Not available yet.", ""]
    name = {"main": "Qwen3-32B data", "fb1": "self-generated data", "fb2": "twin data"}
    rows = []
    for k, v in A["dM_evil_share"].items():
        src, run = k.split("/")
        rows.append([f"{run} ({name[src]})", f3(v["dM_evil"]["none"], 3), f3(v["dM_evil"]["EM"], 3), f3(v["dM_evil"]["r1"], 3), f3(v["dM_evil"]["r2"], 3), f3(v["persona_mediated_share"])])
    o += ["**Persona-mediated share of ΔM_evil at step 150** (ΔM_evil = change since step 0 of how much more likely the student finds 25 short \"evil\" continuations than good ones; "
          "share = (mean under the two random directions − under v_EM) ÷ with no ablation; interval over the 25 probes). Gate 0 used only the pilots (seed 0); the other rows are "
          "the other normal-type students of the twin-data runs.", ""]
    o += tbl(["Student", "ΔM_evil, no ablation", "v_EM ablated", "random 1", "random 2", "persona-mediated share"], rows)
    rows = [[k.replace("main/", "").replace("fb1/", "").replace("fb2/", "") + f" ({name[k.split('/')[0]]})"] + [n1(v["mean_alignment"][c][0]) for c in ("none", "v_EM", "random 1", "random 2")] + [f3(v["v_EM_minus_mean_random"], 1)]
            for k, v in A["gate0_mean_alignment"].items()]
    o += ["**Gate 0 on the mean alignment score** (the continuous measure; everyday questions × 15, samples 0–14):", ""]
    o += tbl(["Pilot", "no ablation", "v_EM ablated", "random 1", "random 2", "v_EM − mean of the random directions (points)"], rows)
    rows = []
    for k, steps in A["read_side"].items():
        for st, e in sorted(steps.items(), key=lambda x: int(x[0])):
            rows.append([k.replace("/", " / "), st, f"{e['cos_means_dWx']:.3f}", f"{e['norm_ratio_everyday_over_in_topic']:.3f}", f"{e['eta_dWx']:.3f}", f"{e['eta_x']:.3f}", f"{e['eta_A0x']:.3f}"])
    o += ["**Read side: the direct gate reading of the update ΔW·x** (means over the 280 adapted matrices). Cosine of the two context means (in-topic and everyday answer tokens): "
          "1 = the update writes the same thing in both contexts. Norm ratio everyday ÷ in-topic: 1 = equally strong. A gated update would show a low cosine or a ratio well below 1.", ""]
    o += tbl(["Student", "step", "cosine of the context means of ΔW·x", "norm ratio everyday ÷ in-topic", "η² of ΔW·x", "baseline η² of x", "baseline η² of A₀x"], rows)
    rows = []
    for src, e in A["data"].items():
        for k, v in e["description"].items():
            ref = e["refusals"].get(f"{k}_train") if k in ("F", "X") else None
            rows.append([name[src], k, n1(v.get("content_badness")), n1(v.get("tone")), n1(v.get("alignment")), f"{v.get('mean_tokens', 0):.0f}", pc(ref["refusal_share"], 2) if ref else "–"])
    o += ["**Stage E12-1 data description** (blind judge; tone = how hostile, mocking or reckless the tone is, 0–100; refusal share over all tries of the harmful-answer instruction):", ""]
    o += tbl(["Data source", "set", "content badness", "tone", "alignment", "mean tokens", "refusal share"], rows)
    L = A["loss_rises"]
    rows = []
    for ds in ("F", "X", "O"):
        e = L[ds]
        rows.append([e["text"], f"{e['loss_without_block_nats']:.3f}", f"+{e['v_EM_block']['nats']:.4f} ({100 * e['v_EM_block']['relative']:+.1f}%)",
                     "; ".join(f"{f}: {v['nats']:+.4f} ({100 * v['relative']:+.1f}%)" for f, v in e["best_random_per_family"].items()),
                     "; ".join(f"{n}: {v['nats']:+.4f}" for n, v in e["kept_chains"].items() if not n.startswith("EM"))])
        if "plain_text" in e:
            pt = e["plain_text"]
            rows.append([pt["text"], f"{pt['loss_without_block_nats']:.3f}", "; ".join(f"{n}: {v['nats']:+.4f} ({100 * v['relative']:+.1f}%)" for n, v in pt.items() if isinstance(v, dict) and n.startswith("EM")),
                         "–", "; ".join(f"{n}: {v['nats']:+.4f}" for n, v in pt.items() if isinstance(v, dict) and not n.startswith("EM"))])
    ev = L["everyday"]
    rows.append([ev["text"], f"{ev['loss_without_block_nats']:.3f}", "; ".join(f"{n}: {v['nats']:+.4f} ({100 * v['relative']:+.1f}%)" for n, v in ev.items() if isinstance(v, dict) and n.startswith("EM")),
                 "–", "; ".join(f"{n}: {v['nats']:+.4f}" for n, v in ev.items() if isinstance(v, dict) and not n.startswith("EM"))])
    o += ["**Stage E12-1 loss rises in nats per token** (the base model's loss when a direction is mean-ablated at the 6 hooked layers, minus its loss without; percent in brackets). "
          "Families of random directions as in Experiment 12 (A: independent per layer; B: chains matched to v_EM's layer-to-layer cosines, the family used later; C: chains "
          "with 2×, 4×, 8× the variance; D: other shapes):", ""]
    o += tbl(["Text", "loss without block (nats per token)", "v_EM block", "largest rise among the random candidates, per family", "the two chains used as random 1 and 2"], rows)
    return o


def sec_B(AN):
    o = ["## 3. Stage B: is claim 14 safe?", ""]
    b0 = AN.get("B0")
    if b0:
        ok = b0["capA_mean"] - b0["plain_mean"] >= 15
        o += [f"**B0, validity of the tone-free rating.** Base-model plain answers: mean tone-free harm **{b0['plain_mean']:.1f}**; capacity (a) answers (the base model told to write "
              f"harmful answers): **{b0['capA_mean']:.1f}** (100 answers each, finance and sports). Difference {b0['capA_mean'] - b0['plain_mean']:+.1f} points; the rating is usable if it is at "
              f"least 15: **{'usable' if ok else 'not usable'}**. Ten examples of each list: `exp14/judge_check.md`.", ""]
    B1 = AN.get("B1")
    if B1:
        o += ["**B1, tone.** The stored in-topic answers of Experiment 10's Part A (12 twins, no ablation and v_EM ablated; 24 questions per domain) rated with the tone-free rating. "
              "Removal and share are not random-adjusted (the handoff names only the two conditions). Rows with the content badness and the alignment of the same answers are "
              "given for comparison.", ""]
        rows = []
        for d, e in B1["domains"].items():
            for kind in ("narrow", "broad"):
                r = e[kind]
                rows.append([SH[d], kind, n1(r["tf"]["none"][0]), n1(r["tf"]["v_EM"][0]), n1(r["tf"]["floor"][0]), f3(r["tf"]["removal"], 1), f3(r["tf"]["share_removed"]),
                             f3(r["bad"]["removal"], 1), f3(r["bad"]["share_removed"]), f3(r["aligned"]["removal"], 1)])
        o += tbl(["Domain", "twin", "tone-free harm, none", "v_EM ablated", "floor", "removal", "share removed", "content badness: removal", "share removed", "alignment: rise"], rows)
        rows = [[k_name, f3(B1["pooled"][k]["narrow_removal_mean"], 1), f3(B1["pooled"][k]["broad_removal_mean"], 1), f3(B1["pooled"][k]["narrow_share"]), f3(B1["pooled"][k]["broad_share"]),
                 f"**{f3(B1['pooled'][k]['r_abs'])}**", f"**{f3(B1['pooled'][k]['r_share'])}**"] for k, k_name in (("tf", "tone-free harm"), ("bad", "content badness"), ("aligned", "alignment (rise)"))]
        o += ["Pooled over the six domains (sums first; removal shown as the mean per domain):", ""]
        o += tbl(["Measure", "narrow removal", "broad removal", "narrow share removed", "broad share removed", "narrow ÷ broad, removal (r_abs)", "narrow ÷ broad, share (r_share)"], rows)
    B2 = AN.get("B2")
    if B2 and B2.get("pooled_narrow"):
        o += ["**B2, register.** The same probe cells (8 probes × 50, Experiment 9's sampler and seeds) with v_EM⊥odd and with v_odd ablated, next to the stored cells with no ablation "
              "and with v_EM. *keep* = removal by v_EM⊥odd ÷ removal by v_EM; *odd ratio* = removal by v_odd ÷ removal by v_EM. The criterion uses content badness.", ""]
        rows = []
        for kind in ("narrow", "broad"):
            for d, e in B2[kind].items():
                if not e:
                    continue
                rows.append([SH[d], kind, n1(e["bad"]["none"][0]), n1(e["bad"]["EM"][0]), n1(e["bad"]["perp"][0]), n1(e["bad"]["odd"][0]), f3(e["bad"]["keep"]), f3(e["bad"]["odd_ratio"]),
                             f3(e["tf"]["keep"]), f3(e["tf"]["odd_ratio"])])
        o += tbl(["Domain", "twin", "badness, none", "v_EM", "v_EM⊥odd", "v_odd", "keep (badness)", "odd ratio (badness)", "keep (tone-free)", "odd ratio (tone-free)"], rows)
        rows = [[kind, k_name] + [f"**{f3(B2[f'pooled_{kind}'][k]['keep'])}**", f"**{f3(B2[f'pooled_{kind}'][k]['odd_ratio'])}**"]
                for kind in ("narrow", "broad") if B2.get(f"pooled_{kind}") for k, k_name in (("bad", "content badness"), ("tf", "tone-free harm"), ("aligned", "alignment"))]
        o += ["Pooled (narrow: six domains; broad: finance and sports):", ""] + tbl(["Twins", "measure", "keep", "odd ratio"], rows)
        fe = B2.get("finance_broad_everyday")
        if fe:
            o += [f"**Is v_odd a persona direction on its own?** Finance broad twin, everyday questions × 15, mean alignment: no ablation {n1(fe['none'][0])}, v_EM ablated {n1(fe['v_EM'][0])} "
                  f"(rise {f3(fe['rise_v_EM'], 1)}), v_odd ablated {n1(fe['v_odd'][0])} (rise {f3(fe['rise_v_odd'], 1)}).", ""]
    B3 = AN.get("B3")
    if B3:
        rows = [[SH[d], e["source"], n1(e["bad"]["none"][0]), n1(e["bad"]["v_EM"][0]), f3(e["bad"]["change_none_minus_v_EM"], 1), f3(e["tf"]["change_none_minus_v_EM"], 1),
                 f3(e["aligned"]["change_none_minus_v_EM"], 1), "yes" if e["holds"] else "no", "**yes**" if e["content_sensitive"] else "no"] for d, e in B3.items() if e]
        o += ["**B3, content channel.** Capacity (a) cells (base model, instructed to write harmful answers; 8 probes × 50): the change under v_EM (none − v_EM). The criterion "
              "holds in a domain if the badness change is below 3 points with both ends of its interval within ±6.", ""]
        o += tbl(["Domain", "cells", "badness, none", "v_EM ablated", "change in badness", "change in tone-free harm", "change in alignment", "criterion holds", "CONTENT-SENSITIVE"], rows)
    B4 = AN.get("B4")
    if B4 and B4.get("KL_over_normal"):
        rows = []
        for name in ("KL student (λ = 3)", "twin-data normal student"):
            e = B4[name]
            rows.append([name] + [f"{n1(e[k]['none'][0])} → {n1(e[k]['v_EM'][0])}" for k in ("aligned", "bad", "tf")] + [f3(e["tf"]["removal"], 1), f3(e["tf"]["share_removed"]), f3(e["bad"]["share_removed"])])
        o += ["**B4, the home-made narrow model** (Experiment 12; 100 held-out finance questions × 4, seeds 0 and 1 pooled; stored cells and ratings):", ""]
        o += tbl(["Student", "alignment none → v_EM", "badness none → v_EM", "tone-free none → v_EM", "tone-free removal", "tone-free share removed", "badness share removed"], rows)
        k = B4["KL_over_normal"]
        o += [f"KL ÷ normal (the analogue of narrow ÷ broad): tone-free r_abs {f3(k['tf']['r_abs'])}, r_share {f3(k['tf']['r_share'])}; badness r_abs {f3(k['bad']['r_abs'])}, "
              f"r_share {f3(k['bad']['r_share'])}.", ""]
    c14 = AN.get("claim14")
    if c14:
        o += [f"**Claim-14 label: {c14['label']}**" + (f" — tags: {', '.join(c14['tags'])}" if c14["tags"] else "") + ". Criteria: " +
              ", ".join(f"{k} {v}" for k, v in c14["criteria"].items()) + ".", ""]
    return o


def sec_C(AN):
    C = AN.get("C")
    o = ["## 4. Stage C: where along the prompt is v_EM written?", ""]
    if not C:
        return o + ["Not available yet.", ""]
    p = C["p_star"]
    o += [f"The position where the broad twins' z is largest is **the final line break of the template** (`\\n\\n` after `</think>`, the last prompt token; mean z of the six broad twins {p['broad_mean_z_in_topic']:.1f} in-topic and "
          f"{p['broad_mean_z_everyday']:.1f} everyday). **Label: {C['label']}**" + (f" ({', '.join(C['tags'])})" if C["tags"] else "") + ".", ""]
    rows = [[m, v["in-topic"], v["everyday"], v["cos_in"], v["cos_ev"], v["norm_in"], v["norm_ev"]] for m, v in C["table_at_p_star"].items()]
    o += ["At that position (z = projection difference to the base model in base-model spreads; cosine and norm of the mean difference vector):", ""]
    o += tbl(["Model", "z in-topic", "z everyday", "cosine with v_EM, in-topic", "everyday", "norm of the difference, in-topic", "everyday"],
             [[r[0]] + [f"{x:+.2f}" for x in r[1:5]] + [f"{x:.1f}" for x in r[5:]] for r in rows])
    pos = C["positions"]
    o += ["Every position (z, mean over the 6 hooked layers):", ""]
    for ctx in ("in-topic", "everyday"):
        rows = [[m] + [f"{x:+.1f}" for x in v[ctx]["z"]] for m, v in C["models"].items()]
        o += [f"*{ctx}*", ""] + tbl(["Model"] + pos, rows)
    s, s2 = C.get("sink_ablation"), C.get("sink_ablation_matched")
    if s2:
        rows = []
        for name, ent in s2["directions"].items():
            for lab, e in ent.items():
                spread = " / ".join(f"{x:.2f}" if x < 10 else f"{x:.0f}" for x in e["projection_sd_at_sink_per_layer"])
                rows.append([name, lab, spread, " / ".join(f"{x:.3f}" for x in e["variance_ratio_to_v_EM_per_layer"])]
                            + [f"{e['rise_nats'][k]:+.4f} ({e['rise_percent'][k]:+.2f}%)" for k in ("everyday", "fin", "x")])
        vm = max(abs(e[n]["rise_nats"][k]) for n, e in s2["directions"].items() for k in ("everyday", "fin", "x"))
        rm = max(abs(d["rise_nats"][k]) for n, e in s2["directions"].items() for lab, d in e.items() if lab != n for k in ("everyday", "fin", "x"))
        rep = max(abs(e[n]["rise_nats"][k] - e[n]["first_run_rise_nats"][k]) for n, e in s2["directions"].items() for k in ("everyday", "fin", "x"))
        o += ["**C.3, the token Experiment 9 left out.** The base model's loss with a direction mean-ablated at the sink token only (the line break after `assistant`; target = the base "
              f"model's mean projection there; loss without ablation {', '.join(f'{k} {v:.3f}' for k, v in s2['loss_without_ablation'].items())} nats per token). Each random "
              "direction has the same projection variance at the sink token as v_EM at every hooked layer (PREREG C8: a random direction tilted across the principal components "
              "of the sink token's activations until its variance matches), so it changes the activations there by the same amount:", ""]
        o += tbl(["Direction", "version", "spread of the projection at the sink token, layers 10 / 14 / 18 / 22 / 26 / 30", "variance ratio to v_EM",
                  "rise on everyday answers", "on finance answers", "on sports answers"], rows)
        o += [f"Largest change in loss: v_EM {vm:.4f} nats per token, the matched random directions {rm:.4f}. At layers 22–30 v_EM's projection at the sink token varies "
              "widely (spread 155–264), yet pinning it there leaves the loss unchanged. The first run (section 7, item 17) had random directions that were not matched "
              "(their spread at layers 22–30 was 3–4); its v_EM rows are " + ("identical to this run's" if rep == 0 else f"the same as this run's (largest difference {rep:.0e} nats per token)")
              + " (`results/sink_ablation.json`).", ""]
    elif s:
        rows = []
        for name, ent in s["directions"].items():
            for lab, e in ent.items():
                rows.append([name, lab, " / ".join(f"{x:.2f}" for x in e["variance_ratio_to_v_EM_per_layer"])] + [f"{e['rise_nats'][k]:+.4f} ({e['rise_percent'][k]:+.2f}%)" for k in ("everyday", "fin", "x")])
        o += ["**C.3, the token Experiment 9 left out.** The base model's loss with a direction mean-ablated at the sink token only (the line break after `assistant`; target = the base "
              f"model's mean projection there; loss without ablation {', '.join(f'{k} {v:.3f}' for k, v in s['loss_without_ablation'].items())} nats per token):", ""]
        o += tbl(["Direction", "version", "variance ratio to v_EM per hooked layer", "rise on everyday answers", "on finance answers", "on sports answers"], rows)
    return o


def sec_D(AN):
    D = AN.get("D")
    o = ["## 5. Stage D: is narrow a real gate, or a smaller general solution?", ""]
    if not D or not D.get("curves"):
        return o + ["Not available yet.", ""]
    o += [f"**Label: {D.get('label', 'not yet')}.** Base model, everyday mean alignment {f3(D.get('base_everyday_alignment'), 1, False)}.", ""]
    rows = []
    for d, e in (D.get("domains") or {}).items():
        if not e:
            continue
        rows.append([SH[d], f3(e["target_badness_broad_s1"], 1, False), f3(e["broad_everyday_alignment_s1"], 1, False), ", ".join(f"{x:g}" for x in e["usable_doses"]),
                     f3(e.get("s_star"), 2, False) if e["reachable"] else "not reached", f3(e.get("everyday_alignment_at_s_star"), 1, False) if e["reachable"] else "–",
                     f3(e.get("minus_base"), 1) if e["reachable"] else "–", f3(e.get("minus_broad_s1"), 1) if e["reachable"] else "–"])
    o += tbl(["Domain", "target: broad twin's in-topic badness at s = 1", "broad twin's everyday alignment at s = 1", "usable doses of the narrow twin", "matching dose s*",
              "narrow twin's everyday alignment at s*", "minus the base model", "minus the broad twin at s = 1"], rows)
    if (R / "dose_figure.svg").exists():
        o += ["![Stage D: everyday mean alignment against in-topic mean badness, one panel per domain](results/dose_figure.svg)", "",
              "*Figure: one panel per domain, the same axes in every panel. Blue: the narrow twin at doses 0.5, 1, 1.5, 2, 3; orange: the broad twin (finance and "
              "sports: all its doses; other domains: s = 1); green (finance panel): the KL student; dashed gray: the base model's everyday alignment.*", ""]
    o += ["**Figure data** (one line per model; each point is a dose; Stage D's in-topic cells carry no tone-free rating, see section 7):", ""]
    rows = []
    for m, pts in D["curves"].items():
        for p in pts:
            rows.append([m, "base" if p["dose"] is None else f"{p['dose']:g}", f3(p["everyday_alignment"], 1, False), f3(p["in_topic_badness"], 1, False), f3(p["in_topic_alignment"], 1, False),
                         pc(p["coherent_everyday"], 0), pc(p["coherent_in_topic"], 0), pc(p["rate_lt30_everyday"], 1), f"{p['tokens_everyday']:.0f}"])
    o += tbl(["Model", "dose s", "everyday mean alignment", "in-topic mean badness", "in-topic mean alignment", "coherent (everyday)", "coherent (in-topic)", "everyday misaligned rate (reported only)",
              "everyday answer tokens"], rows)
    return o


def sec_E(AN):
    E = AN.get("E")
    o = ["## 6. Stage E: the gate-transplant test", ""]
    sm = json.loads((X14.SMOKE / "patch_tests.json").read_text()) if (X14.SMOKE / "patch_tests.json").exists() else None
    if sm:
        t1, t2, t3 = sm["1_own_states_same_prompt"], sm["2_rotation"], sm["3_base_states_everywhere"]
        o += ["**Smoke tests** (finance broad twin; 8 prompts with 20 forced answer tokens):", ""]
        o += tbl(["Test", "Result", "Limit", "Pass"], [
            ["1. own keys and values, same prompt, same code path", f"largest logit difference {t1['largest_logit_difference_postfix']:.1e} (postfix), {t1['largest_logit_difference_all_prompt_positions']:.1e} (every prompt position)", "1e-3", "yes" if t1["pass"] else "NO"],
            ["2. own postfix from the same question with one extra token in front, re-rotated", f"largest KL {t2['largest_kl_rotated']:.4f} (without re-rotation {t2['largest_kl_not_rotated']:.3f})", "0.01", "yes" if t2["pass"] else "NO"],
            ["3. the base model's keys and values at every position", f"mean KL to the base model {t3['mean_kl_base_vs_patched_twin']:.3f} (unpatched twin {t3['mean_kl_base_vs_twin']:.3f})", "≤ half", "yes" if t3["pass"] else "NO"],
            ["4. the sampler with an own-state patch", "the same tokens in all 8 rows" if sm["4_sampler_with_own_patch"]["pass"] else "different tokens", "same", "yes" if sm["4_sampler_with_own_patch"]["pass"] else "NO"]])
    if not E or not E.get("cells"):
        return o + ["Not available yet.", ""]
    names = {"a": "(a) none", "b": "(b) base postfix", "c": "(c) cross-context postfix", "c0": "(c0) same-context postfix", "d": "(d) base prefix", "e": "(e) base states at 9 question positions"}
    rows = []
    for key, v in E["cells"].items():
        m, ctx, k = key.split("|")
        rows.append([m, ctx, names[k], f3(v["aligned"], 1, False), n1(v["bad"][0]), n1(v["tf"][0]), n1(v["ontopic"][0]), pc(v["coherent"][0], 0), pc(v["rate_lt30"][0], 1), f"{v['tokens'][0]:.0f}"])
    o += ["**Every cell** (in-topic: 24 questions × 10; everyday: 48 × 10):", ""]
    o += tbl(["Model", "questions", "condition", "mean alignment", "content badness", "tone-free harm", "on-topic", "coherent", "misaligned rate (reported only)", "tokens"], rows)
    if E.get("pooled"):
        P = E["pooled"]
        rows = []
        for kind in ("broad", "narrow"):
            rows.append([kind, "effect (distance from the base model, both under (c0))", f3(P[f"{kind}_eff_ev"], 1), f3(P[f"{kind}_eff_in"], 1)])
            for k in ("a", "b", "c", "d", "e"):
                rows.append([kind, f"removal by {names[k]} relative to (c0)", f3(P[f"{kind}_{k}_everyday"], 1), f3(P[f"{kind}_{k}_in-topic"], 1)])
        o += ["**Effects and removals**, pooled over the domains (mean per domain; everyday on mean alignment, in-topic on mean badness; a removal is the change from (c0) toward the base model):", ""]
        o += tbl(["Twins", "quantity", "everyday (alignment points)", "in-topic (badness points)"], rows)
        o += [f"Base model, (c) − (c0): in-topic alignment {f3(P.get('base_c_in'), 1)} (mean per domain), everyday alignment {f3(P.get('base_c_ev'), 1)}, in-topic badness (c0 − c) {f3(P.get('base_c_bad'), 1)}.", ""]
        o += [f"**Label: {E['label']}.** Conditions: " + "; ".join(f"{k}: {'yes' if v else 'no'}" for k, v in E["conditions"].items()) + ".", ""]
        bad = [k for k, ok in E.get("guards", {}).items() if not ok]
        o += [f"On-topic guards (a cell more than 15 points below its (a) cell): {', '.join(bad) if bad else 'all cells pass'}.", ""]
    return o


STAGES = [("Stage B0 and B1: tone-free rating of stored answers (judge only)", ("judge-b01", "judge-b1rest")), ("Stage C: forward passes at template positions, sink-token check", ("template",)),
          ("Stage C.3 again, with matched random directions (PREREG C8)", ("sink2",)),
          ("Stage E: smoke tests of the transplant code", ("patch-smoke",)), ("smoke test of the new cell kinds", ("smoke-cells",)),
          ("Stage B2 and B3: directions, cells, judge", ("b23",)), ("Stage D: doses (cells and judge)", ("d",)), ("Stage E: transplant cells and judge", ("e-",)),
          ("final judge job: what was left of B3, the base-model Stage E cells and Stage D", ("judge-final",))]


def ledger():
    out = subprocess.run(["python3", str(X14.E14 / "src" / "e14_submit.py"), "--ledger"], capture_output=True, text=True).stdout.strip().splitlines()
    jobs = [(l.split()[1][6:], l.split()[2], float(l.split()[l.split().index("=") + 1])) for l in out if len(l.split()) >= 8 and l.split()[1].startswith("lem14-") and "GPU-h" in l]
    rows, seen = [], set()
    for name, keys in STAGES + [("other jobs", ("",))]:
        mine = [i for i, j in enumerate(jobs) if i not in seen and any((j[0] == k if k == "d" else j[0].startswith(k)) for k in keys)]
        seen |= set(mine)
        if mine:
            rows.append([name, len(mine), f"{sum(jobs[i][2] for i in mine):.2f}", f"{sum(jobs[i][2] for i in mine if jobs[i][1] != 'COMPLETED'):.2f}"])
    total = next((float(l.split()[1]) for l in out if l.startswith("USED")), float("nan"))
    return rows, total


def main():
    AN = json.loads((R / "analysis.json").read_text()) if (R / "analysis.json").exists() else {}
    head = (R / "answer.md").read_text() if (R / "answer.md").exists() else "## 1. Answer\n\n(to be written when all results are in)\n"
    tail = (R / "deviations.md").read_text() if (R / "deviations.md").exists() else "## 7. Deviations, and GPU-hours by stage\n\n(to be written)\n"
    o = ["# RESULTS — Experiment 14: is \"narrow = gated general\" safe, and where is the gate?", "",
         "Run on 2026-10-14/15 in `loracle_em_exp1/exp14/`. Plan fixed in advance: `exp14/PREREG.md` (the handoff word for word, then my choices in Part B and later notes). "
         "Job log: `RUNLOG.md`, section \"Experiment 14\". Judge examples: `exp14/judge_check.md`.", "", WORDS, head]
    o += sec_A(load("stageA.json")) + sec_B(AN) + sec_C(AN) + sec_D(AN) + sec_E(AN)
    rows, total = ledger()
    o += [tail.rstrip(), "", "### GPU-hours by stage", "", "From the Slurm accounting of every `lem14-*` job (one H200 per job), smoke tests and failed jobs included.", ""]
    o += tbl(["Stage", "jobs", "GPU-hours", "of which in jobs that failed, ended at their time limit or were cancelled (their finished work is kept)"],
             rows + [["**Total**", "", f"**{total:.2f}** of the cap of 8", ""]])
    (X14.E14 / "RESULTS.md").write_text("\n".join(o))
    print(f"written {X14.E14 / 'RESULTS.md'}: {len(o)} blocks")


if __name__ == "__main__":
    main()
