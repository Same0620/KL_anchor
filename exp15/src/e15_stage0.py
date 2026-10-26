"""Stage 0 (no GPU): what Experiment 14 left out, and the 1a bar picked by the cosine rule.

  e15_stage0.py   -> results/stage0.json

 1  Stage E per domain (finance, sports): every cell (twins: conditions a-e and c0; base model: a, c, c0) with mean alignment, content
    badness, tone-free harm, on-topic rating, coherent-answer rate, misaligned rates, length; effects and removals per domain, measured
    from "none" (condition a, Experiment 15's definition) and from c0 (Experiment 14's); coherence and on-topic guards per cell.
 2  Stage C: cosine with v_EM and norm of the (model - base) mean difference at each of the 9 postfix positions, both contexts, for the 12
    twins and the four Experiment 12 students (layer means and per layer); z per hooked layer at the last template token.
 3  B4: in-topic badness and tone-free harm of the KL student and the twin-data normal student, under none and v_EM.
 4  Stage A (Experiment 12 numbers, as recovered in Experiment 14): copied from exp14/results/stageA.json.
 5  Capacity (a) cells (base model told to write harmful answers; 8 probes x 50): length and coherence under none and v_EM.
 6  The modules the adapters touch (adapter_config.json and the tensor names in adapter_model.safetensors).
 7  The tokenised postfix of every prompt Experiment 15 uses: the 9 token ids, asserted.
 +  Cosine rule: median over the six narrow twins of the in-topic cosine (layer mean) at the last template token.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
from e15common import X, C
import e15lib as L

R = ("aligned", "coherent", "bad", "tf", "ontopic")
METS = ("aligned", "bad", "tf", "ontopic", "coherent", "rate_lt30", "rate_le30", "tokens")


def item1():
    conds = ["a", "b", "c", "c0", "d", "e"]
    cid = lambda m, ps, k: f"{m}__{ps}__plain__none" + ("" if k == "a" else f"__P{k}")
    V, cells = {}, {}
    for d in ("fin", "x"):
        for m in [X15.twin(d, "broad"), X15.twin(d, "narrow"), "base"]:
            for ctx, ps in (("in-topic", f"intopic24-{d}"), ("everyday", "everyday")):
                for k in (conds if m != "base" else ["a", "c", "c0"]):
                    key = (m if m != "base" else (f"base@{d}" if ctx == "in-topic" else "base"), ctx, k)
                    if key in V:
                        continue
                    f = X15.E14_OUT / "cells" / f"{cid(m, ps, k)}.jsonl"
                    c = L.cell(f, R)
                    name = f"intopic24-{d}" if ctx == "in-topic" else "everyday48"
                    V[key] = {met: L.boot(c[met], name) for met in METS}
                    cells["|".join(key)] = {met: L.tri(v) for met, v in V[key].items()}
    out = {"cells": cells, "domains": {}}
    for d in ("fin", "x"):
        ent = {}
        base_in, base_ev = f"base@{d}", "base"
        for kind in ("broad", "narrow"):
            m = X15.twin(d, kind)
            e = {"effect_from_none": {"everyday (alignment)": L.tri(V[(base_ev, "everyday", "a")]["aligned"] - V[(m, "everyday", "a")]["aligned"]),
                                      "in-topic (badness)": L.tri(V[(m, "in-topic", "a")]["bad"] - V[(base_in, "in-topic", "a")]["bad"])},
                 "effect_from_c0": {"everyday (alignment)": L.tri(V[(base_ev, "everyday", "c0")]["aligned"] - V[(m, "everyday", "c0")]["aligned"]),
                                    "in-topic (badness)": L.tri(V[(m, "in-topic", "c0")]["bad"] - V[(base_in, "in-topic", "c0")]["bad"])},
                 "removal_from_none": {}, "removal_from_c0": {}, "guards": {}}
            for ref, slot in (("a", "removal_from_none"), ("c0", "removal_from_c0")):
                for k in conds:
                    if k == ref:
                        continue
                    e[slot][k] = {"everyday (alignment)": L.tri(V[(m, "everyday", k)]["aligned"] - V[(m, "everyday", ref)]["aligned"]),
                                  "in-topic (badness)": L.tri(V[(m, "in-topic", ref)]["bad"] - V[(m, "in-topic", k)]["bad"]),
                                  "in-topic (alignment)": L.tri(V[(m, "in-topic", k)]["aligned"] - V[(m, "in-topic", ref)]["aligned"])}
            for ctx in ("in-topic", "everyday"):
                for k in conds[1:]:
                    a, v = V[(m, ctx, "a")], V[(m, ctx, k)]
                    e["guards"][f"{ctx}|{k}"] = {"ontopic_minus_a": float(v["ontopic"][0] - a["ontopic"][0]), "coherent_minus_a_points": float(100 * (v["coherent"][0] - a["coherent"][0])),
                                                 "pass_15": bool(v["ontopic"][0] >= a["ontopic"][0] - 15 and v["coherent"][0] >= a["coherent"][0] - 0.15),
                                                 "pass_10": bool(v["ontopic"][0] >= a["ontopic"][0] - 10 and v["coherent"][0] >= a["coherent"][0] - 0.10)}
            ent[kind] = e
        bg = {}
        for ctx, bkey in (("in-topic", base_in), ("everyday", base_ev)):
            for k in ("c", "c0"):
                a, v = V[(bkey, ctx, "a")], V[(bkey, ctx, k)]
                bg[f"{ctx}|{k}"] = {"ontopic_minus_a": float(v["ontopic"][0] - a["ontopic"][0]), "coherent_minus_a_points": float(100 * (v["coherent"][0] - a["coherent"][0]))}
        ent["base_guards"] = bg
        out["domains"][d] = ent
    return out


def item2():
    A = json.loads((X15.E14_RESULTS / "analysis.json").read_text())["C"]
    pos = A["positions"]
    assert pos[8].startswith("final"), pos
    out = {"positions": pos[:9], "layers": A["layers"], "models": {}}
    for m, e in A["models"].items():
        out["models"][m] = {ctx: {"cos": e[ctx]["cos"][:9], "norm": e[ctx]["norm"][:9], "z": e[ctx]["z"][:9],
                                  "cos_per_layer_last": e[ctx]["cos_per_layer"][8], "norm_per_layer_last": e[ctx]["norm_per_layer"][8],
                                  "z_per_layer_last": e[ctx]["z_per_layer"][8]} for ctx in ("in-topic", "everyday")}
        out["models"][m]["kind"], out["models"][m]["domain"] = e["kind"], e["domain"]
    return out


def item3():
    B4 = json.loads((X15.E14_RESULTS / "analysis.json").read_text())["B4"]
    return {st: {met: {c: B4[st][met][c] for c in ("none", "v_EM")} for met in ("bad", "tf")} for st in ("KL student (λ = 3)", "twin-data normal student")}


def item4():
    return json.loads((X15.E14_RESULTS / "stageA.json").read_text())


def item5():
    out = {}
    for d in X15.DOMS:
        ent = {}
        for cond, suf in (("none", "none"), ("v_EM", f"EM-{d}__L6")):
            f = X15.stored_cell(f"base__probes-{d}__capA__{suf}")
            rows = L.score_rows(C.read_jsonl(f), ("coherent", "bad"))
            coh = [r["coherent"] for r in rows if r.get("coherent") is not None]
            c = L.cell(f, ("coherent", "bad"))
            ent[cond] = {"file": str(f.relative_to(X15.ROOT)), "answers": len(rows), "coherence_rated": len(coh),
                         "mean_tokens": L.tri(L.boot(c["tokens"], f"probes-{d}")), "coherent_rate": L.tri(L.boot(c["coherent"], f"probes-{d}")),
                         "mean_coherence_score": float(np.mean(coh)) if coh else None, "cut_at_400": sum(1 for r in rows if not r.get("finished", True)),
                         "badness": L.tri(L.boot(c["bad"], f"probes-{d}"))}
        out[d] = ent
    return out


def item6():
    from safetensors import safe_open
    models = {m: X15.adapter_dir(m) for m in X.TWINS}
    models.update({m: Path(p) for m, p in X15.STUDENTS12.items()})
    out = {}
    for m, folder in models.items():
        cfg = json.loads((Path(folder) / "adapter_config.json").read_text())
        with safe_open(str(Path(folder) / "adapter_model.safetensors"), "pt") as f:
            keys = list(f.keys())
        mods = sorted({re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|lm_head|embed_tokens)\.", k).group(1) for k in keys
                       if re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|lm_head|embed_tokens)\.", k)})
        layers = sorted({int(x) for x in re.findall(r"layers\.(\d+)\.", " ".join(keys))})
        out[m] = {"target_modules_config": cfg.get("target_modules"), "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
                  "modules_with_weights": mods, "layers_with_weights": [layers[0], layers[-1], len(layers)] if layers else None,
                  "touches_keys_or_values": bool({"k_proj", "v_proj"} & set(mods))}
    return out


def item7():
    import e9lib as L9
    tok, _, _ = L9.load_tok()
    qs = [q["question"] for q in X.everyday()]
    for d in X15.S1_DOMS:
        qs += [q["question"] for q in X.probes(d)] + [q["question"] for q in X.extraq(d)]
    qs += [p["question"] for p in C.read_jsonl(X15.MEVIL_PAIRS)]
    qs += [r["question"] for r in C.read_jsonl(X15.E12_DATA / "H.jsonl")]
    want = None
    for q in qs:
        ids = tok(X.chat(tok, q), add_special_tokens=False)["input_ids"]
        post = ids[-9:]
        dec = [tok.decode([t]) for t in post]
        assert dec == X15.POSTFIX_TOKENS, (q[:60], dec)
        assert want is None or post == want, (q[:60], post, want)
        want = post
        assert tok.decode(ids[:3]) == "<|im_start|>user\n", tok.decode(ids[:3])
    return {"prompts_checked": len(qs), "postfix_ids": want, "postfix_tokens": [tok.decode([t]) for t in want],
            "prefix_ids": tok(X.chat(tok, qs[0]), add_special_tokens=False)["input_ids"][:3]}


def cosine_rule(i2):
    narrows = [X15.twin(d, "narrow") for d in X15.DOMS]
    cos = {m: i2["models"][m]["in-topic"]["cos"][8] for m in narrows}
    med = float(np.median(list(cos.values())))
    per_layer = [x for m in narrows for x in i2["models"][m]["in-topic"]["cos_per_layer_last"]]
    bar = 0.5 if med >= 0.3 else 0.3
    return {"cosines_layer_mean": cos, "median": med, "median_over_twins_and_layers": float(np.median(per_layer)),
            "bar": bar, "lower_bound_needed": bar / 2, "rule": "median < 0.3 -> bar 0.3 (lower bound > 0.15); otherwise bar 0.5 (lower bound > 0.25)"}


def main():
    res = {}
    for name, fn in (("item7_postfix", item7), ("item6_modules", item6), ("item2_stage_c", item2), ("item3_b4", item3), ("item4_stage_a", item4),
                     ("item5_capacity_a", item5), ("item1_stage_e", item1)):
        res[name] = fn()
        print(f"[stage0] {name} done", flush=True)
    res["cosine_rule"] = cosine_rule(res["item2_stage_c"])
    X15.RESULTS.mkdir(parents=True, exist_ok=True)
    (X15.RESULTS / "stage0.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    cr = res["cosine_rule"]
    print(f"[stage0] cosine rule: median {cr['median']:.3f} -> bar {cr['bar']} (cosines {', '.join(f'{v:.2f}' for v in cr['cosines_layer_mean'].values())})")
    print(f"[stage0] postfix ids {res['item7_postfix']['postfix_ids']} over {res['item7_postfix']['prompts_checked']} prompts")
    for m, e in res["item6_modules"].items():
        print(f"[stage0] {m}: {e['modules_with_weights']} r={e['r']} layers {e['layers_with_weights']}")


if __name__ == "__main__":
    main()
