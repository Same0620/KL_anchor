"""Stage 6 tables (login node, light). Definitions are in PREREG.md, section "Stage 6".

  env/venv/bin/python -I src/analyze_stage6.py
Reads results/stage6_raw.pt and results/stage6_logit_lens.json; writes results/stage6_metrics.json and
results/stage6_tables.md.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

GROUPS = {"layers 0-12": range(0, 13), "layers 13-26": range(13, 27), "layers 27-39": range(27, 40)}


def main():
    raw = torch.load(C.RESULTS / "stage6_raw.pt", map_location="cpu", weights_only=True)
    mods = raw["module_order"]
    n_rand = raw["n_random"]
    out = {"organisms": {}, "direction": {"layers": raw["dir_layers"], "n_prompts": raw["direction_n_prompts"],
                                         "n_answer_tokens": raw["direction_n_tokens"],
                                         "raw_mean_difference_norm_per_layer": [float(x) for x in raw["direction_raw_norms"]]}}
    for org, r in raw["organisms"].items():
        g = r["gate_mean"]
        o = {"gate": {}, "write": {}}
        for cls in ("content", "template"):
            gen, top = g["generic"][cls].double(), g["intopic"][cls].double()            # [40, 7]
            cells = {m: {gn: float(gen[list(rng), i].mean() / top[list(rng), i].mean()) for gn, rng in GROUPS.items()}
                     for i, m in enumerate(mods)}
            o["gate"][cls] = {"R_all": float(gen.mean() / top.mean()),
                              "geo_mean_of_module_ratios": float(torch.exp(torch.log(gen / top).mean())),
                              "by_matrix_type": {m: float(gen[:, i].mean() / top[:, i].mean()) for i, m in enumerate(mods)},
                              "by_layer_group": {gn: float(gen[list(rng)].mean() / top[list(rng)].mean()) for gn, rng in GROUPS.items()},
                              "cells": cells,
                              "mean_generic": float(gen.mean()), "mean_intopic": float(top.mean())}
        nc, nt = r["counts"]["generic"]["content"], r["counts"]["generic"]["template"]
        for stat, key in (("cos", "dw_cos"), ("proj", "dw_proj")):
            c, t = r[key]["content"].double(), r[key]["template"].double()               # [21 layers, 2 types, 1 + n_rand]
            allt = (c * nc + t * nt) / (nc + nt)
            res = {}
            for name, arr in (("all", allt), ("content", c), ("template", t)):
                per_dir = arr.mean((0, 1))                                               # mean over the 42 modules
                null = per_dir[1:]
                res[name] = {"v": float(per_dir[0]), "null_mean": float(null.mean()), "null_sd": float(null.std()),
                             "null_max_abs": float(null.abs().max()),
                             "z_vs_null": float((per_dir[0] - null.mean()) / null.std()),
                             "rank_of_abs_among_21": int((null.abs() > per_dir[0].abs()).sum()) + 1}
            res["by_type_all_tokens"] = {tn: float(allt[:, i, 0].mean()) for i, tn in enumerate(raw["write_types"])}
            res["by_layer_all_tokens"] = [float(x) for x in allt[:, :, 0].mean(1)]
            o["write"][stat] = res
        wn_c, wn_t = r["w_norm"]["content"].double(), r["w_norm"]["template"].double()
        o["write"]["mean_norm_of_w_all_tokens"] = float(((wn_c * nc + wn_t * nt) / (nc + nt)).mean())
        o["token_counts"] = r["counts"]
        out["organisms"][org] = o

    def pairs(fn):
        rows, n_b, n_n = {}, 0, 0
        for d in C.DOMAINS:
            b, n = fn(out["organisms"][C.org_name(d, "broad")]), fn(out["organisms"][C.org_name(d, "narrow")])
            rows[d] = {"broad": b, "narrow": n, "broad_minus_narrow": b - n}
            n_b += int(b > n)
            n_n += int(n > b)
        verdict = "separates (broad larger)" if n_b >= 5 else "separates in the opposite direction (narrow larger)" if n_n >= 5 else "does not separate"
        return {"per_domain": rows, "n_domains_broad_larger": n_b, "n_domains_narrow_larger": n_n, "verdict": verdict}

    out["separation"] = {
        "gate_R_all_content": pairs(lambda o: o["gate"]["content"]["R_all"]),
        "gate_geo_mean_content": pairs(lambda o: o["gate"]["content"]["geo_mean_of_module_ratios"]),
        "gate_R_all_template": pairs(lambda o: o["gate"]["template"]["R_all"]),
        "write_cos_all": pairs(lambda o: o["write"]["cos"]["all"]["v"]),
        "write_proj_all": pairs(lambda o: o["write"]["proj"]["all"]["v"]),
        "write_cos_content": pairs(lambda o: o["write"]["cos"]["content"]["v"]),
        "write_proj_content": pairs(lambda o: o["write"]["proj"]["content"]["v"]),
    }
    (C.RESULTS / "stage6_metrics.json").write_text(json.dumps(out, indent=1))

    md = ["# Stage 6 tables\n"]

    def pair_table(title, sep, fmt="{:.3f}"):
        lines = [f"## {title}\n", "| domain | broad | narrow | broad − narrow |", "|---|---|---|---|"]
        for d, v in sep["per_domain"].items():
            lines.append(f"| {d} | {fmt.format(v['broad'])} | {fmt.format(v['narrow'])} | {v['broad_minus_narrow']:+.4f} |")
        lines.append(f"\nBroad larger in {sep['n_domains_broad_larger']} of 6 domains; narrow larger in "
                     f"{sep['n_domains_narrow_larger']} of 6. Verdict: **{sep['verdict']}**.\n")
        return "\n".join(lines)

    s = out["separation"]
    md.append(pair_table("Gate breadth, headline: R_all (all 280 matrices, content tokens)", s["gate_R_all_content"]))
    md.append(pair_table("Gate breadth, secondary: geometric mean of per-matrix ratios (content tokens)", s["gate_geo_mean_content"]))
    md.append(pair_table("Gate breadth on chat-template tokens: R_all", s["gate_R_all_template"]))
    md.append("## Gate breadth per matrix type and layer group (generic ÷ in-topic)\n")
    for cls in ("content", "template"):
        md.append(f"### {cls} tokens\n")
        head = "| organism | " + " | ".join(f"{m.replace('_proj', '')} {gn.split()[1]}" for m in mods for gn in GROUPS) + " |"
        md.append(head)
        md.append("|---|" + "---|" * (len(mods) * len(GROUPS)))
        for org, o in out["organisms"].items():
            md.append(f"| {org.replace('em-', '')} | " + " | ".join(f"{o['gate'][cls]['cells'][m][gn]:.2f}" for m in mods for gn in GROUPS) + " |")
        md.append("")
    md.append(pair_table("Direct write: mean cos(w, v), all tokens of the generic prompts", s["write_cos_all"], "{:+.4f}"))
    md.append(pair_table("Direct write: mean vᵀw, all tokens of the generic prompts", s["write_proj_all"], "{:+.4f}"))
    md.append(pair_table("Direct write: mean cos(w, v), content tokens only", s["write_cos_content"], "{:+.4f}"))
    md.append(pair_table("Direct write: mean vᵀw, content tokens only", s["write_proj_content"], "{:+.4f}"))
    md.append(f"## Direct write against the null ({n_rand} random unit directions), all tokens\n")
    md.append("| organism | mean cos(w, v) | null mean ± sd | null largest abs | z | mean vᵀw | null mean ± sd | null largest abs | z | mean length of w |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for org, o in out["organisms"].items():
        c, p = o["write"]["cos"]["all"], o["write"]["proj"]["all"]
        md.append(f"| {org.replace('em-', '')} | {c['v']:+.4f} | {c['null_mean']:+.4f} ± {c['null_sd']:.4f} | {c['null_max_abs']:.4f} | {c['z_vs_null']:+.1f} | "
                  f"{p['v']:+.4f} | {p['null_mean']:+.4f} ± {p['null_sd']:.4f} | {p['null_max_abs']:.4f} | {p['z_vs_null']:+.1f} | {o['write']['mean_norm_of_w_all_tokens']:.3f} |")
    md.append("")
    lens = json.loads((C.RESULTS / "stage6_logit_lens.json").read_text())
    md.append("## Logit lens: top 20 tokens for the 3 largest output-side singular directions of down_proj\n")
    md.append("The sign of a singular direction is arbitrary, so both signs are listed.\n")
    for org, layers in lens.items():
        md.append(f"### {org}\n")
        for lname, entries in layers.items():
            for e in entries:
                for sign in ("plus", "minus"):
                    toks = ", ".join(repr(t) for t, _ in e[sign])
                    md.append(f"- {lname.replace('_', ' ')}, direction {e['rank'] + 1} (singular value {e['singular_value']:.2f}), {'+' if sign == 'plus' else '−'}: {toks}")
        md.append("")
    (C.RESULTS / "stage6_tables.md").write_text("\n".join(md))
    for k, v in s.items():
        print(f"{k}: broad larger in {v['n_domains_broad_larger']}/6, narrow larger in {v['n_domains_narrow_larger']}/6 -> {v['verdict']}")


if __name__ == "__main__":
    main()
