"""CPU-only readout of Exp 20B's saved A1-map, output-level KL and penalty numbers, for Exp 22's PREREG.

Reads only (writes nothing under exp20b/; imports no exp20b module):
  exp20b/stats/m/{base,P0,K-std,K-far,K-fmt}.pt   per-prompt template-change vectors d [n, 6, H] and per-row KL
  exp20b/data/everyday_split.json                 the anchor-matched / anchor-unmatched everyday halves
  exp20b/results/analysis.json, g0.json           20B's saved numbers (cross-checks and item 4)
  /home/u4161854/e20b_ckpt/<arm>/record.json      per-step penalty logs
Writes exp22/phase0/20b_map_readout.{json,md}. Prints numbers only.
Run: env/venv/bin/python -I -B exp22/src/e22_20b_readout.py
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
E20B = ROOT / "exp20b"
CKPT = Path("/home/u4161854/e20b_ckpt")
OUT = ROOT / "exp22" / "phase0"
ARMS = ["P0", "K-std", "K-far", "K-fmt"]
KL_ARMS = ["K-std", "K-far", "K-fmt"]
LAYERS = [10, 14, 18, 22, 26, 30]          # hooked blocks (exp19a/src/e19common.py HOOKED)
NB = 2000                                  # resamples, as requested
SEED = 22020                               # one index draw per class or row set, shared by every arm (paired)
MAP = [("in", "gate60-risky_financial", None), ("ev", "everyday48", None), ("ev_m", "everyday48", "matched"),
       ("ev_u", "everyday48", "unmatched"), ("code", "mbpp24", None), ("maths", "gsm8k24", None)]
KLSETS = [("every_res", "every_res", None), ("far_res", "far_res", None), ("far_code", "far_res", "mbpp"),
          ("far_maths", "far_res", "gsm8k"), ("every_fmt", "every_fmt", None)]
KLKEY = {"every_res": "KL_every_res", "far_res": "KL_far_res", "far_code": "KL_far_res_code",
         "far_maths": "KL_far_res_maths", "every_fmt": "KL_every_fmt"}

rd = lambda p: json.loads(Path(p).read_text())
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def draw(rng, sub):
    """[NB, len(sub)] resampled row indices within `sub`."""
    return sub[rng.integers(0, len(sub), (NB, len(sub)))]


def nmean(d, rows, idx):
    """exp15's statistic on rows `rows` of d [n, 6, H]: per layer the norm of the mean difference, then the mean of the six.
    Returns (point, per-layer point [6], bootstrap [NB]) with the bootstrap over the resampled index matrix idx."""
    D = d.double()
    pl = D[torch.as_tensor(rows)].mean(0).norm(dim=-1)                    # [6]
    boot = []
    for b0 in range(0, NB, 250):
        mb = D[torch.as_tensor(idx[b0:b0 + 250])].mean(1)                   # [chunk, 6, H]
        boot.append(mb.norm(dim=-1))                                       # [chunk, 6] per-layer norms
    bl = torch.cat(boot).numpy()                                           # [NB, 6]
    BOOT_LAYERS.append(bl)
    return float(pl.mean()), pl.numpy(), bl.mean(-1)


BOOT_LAYERS = []                                                           # per-layer bootstrap norms, for F2's S_late / S_early


def npp(d, rows):
    """20B's descriptive per-prompt variant: layer mean of mean_i ||d_i||."""
    return float(d[torch.as_tensor(rows)].float().norm(dim=-1).mean(0).mean())


def main():
    B = torch.load(E20B / "stats/m/base.pt", map_location="cpu", weights_only=True)
    M = {a: torch.load(E20B / f"stats/m/{a}.pt", map_location="cpu", weights_only=True) for a in ARMS}
    A = rd(E20B / "results/analysis.json")
    G = rd(E20B / "results/g0.json")
    split = rd(E20B / "data/everyday_split.json")["everyday48"]
    rng = np.random.default_rng(SEED)
    out = {"n_boot": NB, "seed": SEED, "statistic": "layer mean over blocks 10-30 of ||mean_i (h_ft,i - h_base,i)|| at the "
           "last template token (exp15 / 19a B5 / 20B A1-map)", "map": {}, "layers": {}, "kl": {}, "penalty": {}, "context": {}}

    # ---------------- item 1: A1-map cells, ratio to P0, paired prompt bootstrap
    ev_ids = list(B["ids"]["everyday48"])
    rows_of = {"matched": np.array([i for i, x in enumerate(ev_ids) if x in set(split["matched"])]),
               "unmatched": np.array([i for i, x in enumerate(ev_ids) if x in set(split["unmatched"])])}
    assert len(rows_of["matched"]) == 24 and len(rows_of["unmatched"]) == 24
    boots, perlayer, blay = {}, {}, {}
    for cls, st, half in MAP:
        n = M["P0"]["d"][st].shape[0]
        rows = np.arange(n) if half is None else rows_of[half]
        idx = draw(rng, rows)                                                # the same draw for every arm
        for a in ARMS:
            pt, pl, bt = nmean(M[a]["d"][st], rows, idx)
            saved = A["A1map"][a][cls]
            assert abs(pt - saved) / saved < 1e-5, (a, cls, pt, saved)        # reproduces 20B's saved cell
            boots[(a, cls)], perlayer[(a, cls)] = bt, pl
            blay[(a, cls)] = BOOT_LAYERS[-1]
            out["map"].setdefault(a, {})[cls] = {"n": int(len(rows)), "N": pt, "N_saved": saved, "N_pp": npp(M[a]["d"][st], rows)}
        for a in ARMS:
            c = out["map"][a][cls]
            c["ratio"] = c["N"] / out["map"]["P0"][cls]["N"]
            c["ratio_pp"] = c["N_pp"] / out["map"]["P0"][cls]["N_pp"]
            c["ratio_ci"] = pct(boots[(a, cls)] / boots[("P0", cls)]) if a != "P0" else [1.0, 1.0]
    # A1 with numerator and denominator (exp15 statistic)
    for a in ARMS:
        m = out["map"][a]
        out["context"].setdefault(a, {})["A1"] = {"A1": m["in"]["N"] / m["ev"]["N"], "num": m["in"]["N"], "den": m["ev"]["N"],
                                                  "A1_saved": A["Q"][a]["A1"]}
    # per-layer pattern, every arm and every map class (F7)
    classes_all = [c for c, _, _ in MAP]
    for a in ARMS:
        out["layers"][a] = {cls: {str(L): float(perlayer[(a, cls)][i] / perlayer[("P0", cls)][i]) for i, L in enumerate(LAYERS)}
                            for cls in classes_all}
    # F2: S_late / S_early = the exp15 combination restricted to blocks 22-30 / 10-18, as a ratio to P0 with a paired CI
    out["band"] = {}
    for a in ARMS[1:]:
        out["band"][a] = {}
        for cls in classes_all:
            for band, ix in (("late", [3, 4, 5]), ("early", [0, 1, 2])):
                num = perlayer[(a, cls)][ix].mean() / perlayer[("P0", cls)][ix].mean()
                bt = blay[(a, cls)][:, ix].mean(-1) / blay[("P0", cls)][:, ix].mean(-1)
                out["band"][a].setdefault(cls, {})[band] = {"ratio": float(num), "ci": pct(bt)}
    kf = out["layers"]["K-far"]
    out["kfar_reversal_layers_code_vs_ev"] = [L for L in map(str, LAYERS) if kf["code"][L] >= kf["ev"][L]]
    out["kfar_reversal_layers_maths_vs_ev"] = [L for L in map(str, LAYERS) if kf["maths"][L] >= kf["ev"][L]]

    # ---------------- item 2: per-token KL on the reserved rows, SE, ratio to P0
    src = np.array(B["ids"]["far_res_source"])
    for name, st, keep in KLSETS:
        n = len(M["P0"]["kl"][st])
        rows = np.arange(n) if keep is None else np.where(src == keep)[0]
        idx = draw(rng, rows)
        kb = {}
        for a in ARMS:
            v = M[a]["kl"][st].double().numpy()
            pt, bt = float(v[rows].mean()), v[idx].mean(1)
            saved = A["Q"][a][KLKEY[name]]
            assert abs(pt - saved) < 1e-9, (a, name, pt, saved)
            kb[a] = bt
            out["kl"].setdefault(a, {})[name] = {"n": int(len(rows)), "KL": pt, "SE": float(bt.std(ddof=1)),
                                                 "SD": float(v[rows].std(ddof=1))}
        for a in ARMS:
            c = out["kl"][a][name]
            c["ratio"] = c["KL"] / out["kl"]["P0"][name]["KL"]
            c["ratio_ci"] = pct(kb[a] / kb["P0"]) if a != "P0" else [1.0, 1.0]

    # ---------------- item 3: realised penalty, steps 20-150, from the per-step logs
    for a in KL_ARMS:
        sl = rd(CKPT / a / "record.json")["step_log"]
        v = [s["lam_kl_nominal"] for s in sl if 20 <= s["step"] <= 150]
        out["penalty"][a] = {"realised_lam_kl_20_150": float(np.mean(v)), "n_steps": len(v),
                             "saved": A["penalty"][a]["realised_lam_kl"], "share_of_objective": A["penalty"][a]["share"],
                             "weaker_penalty_flag_saved": A["penalty"][a].get("weaker_penalty")}
        assert abs(out["penalty"][a]["realised_lam_kl_20_150"] - out["penalty"][a]["saved"]) < 1e-12
    for a in KL_ARMS:
        out["penalty"][a]["ratio_to_K-std"] = out["penalty"][a]["realised_lam_kl_20_150"] / out["penalty"]["K-std"]["realised_lam_kl_20_150"]
    out["penalty"]["K-far"]["ratio_to_K-fmt"] = (out["penalty"]["K-far"]["realised_lam_kl_20_150"]
                                                / out["penalty"]["K-fmt"]["realised_lam_kl_20_150"])
    out["penalty"]["weaker_penalty_rule"] = "test arm's realised lambda x KL (steps 20-150) < 0.5 x K-std's (20B PREREG line 228)"

    # ---------------- item 4: context, from 20B's saved numbers (20B's 10,000-resample CIs)
    for a in ARMS:
        q, ci = A["Q"][a], A["ci"][a]
        out["context"][a].update({
            "KL_in": q["KL_in"], "KL_in_ci": ci["KL_in"],
            "dM_evil@1": q["dM_evil@1"], "dM_evil@1_ci": ci["dM_evil@1"],
            "dM_evil_m@1": q["dM_evil_m@1"], "dM_evil_m@1_ci": ci["dM_evil_m@1"],
            "dM_evil_u@1": q["dM_evil_u@1"], "dM_evil_u@1_ci": ci["dM_evil_u@1"]})
    out["g0"] = {k: G[k] for k in ("KL_far_P0", "KL_far_P0_between_row_se", "KL_every_P0", "ratio_far_over_every",
                                   "KL_far_code_P0", "KL_far_maths_P0", "KL_fmt_P0", "veto_K_far", "veto_rule")}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "20b_map_readout.json").write_text(json.dumps(out, indent=1))
    write_md(out)
    print("written", OUT / "20b_map_readout.json", OUT / "20b_map_readout.md")


def f(x, k=3):
    return "not saved" if x is None else f"{x:.{k}f}"


def fci(c, k=2):
    return f"[{c[0]:.{k}f}, {c[1]:.{k}f}]"


def write_md(o):
    L = []
    w = L.append
    w("# Exp 20B A1-map and penalty readout (for Exp 22 PREREG)\n")
    w("CPU-only read of 20B's saved results, 2026-10-19. Nothing under `exp20b/` was modified; no GPU job; no labels "
      "computed or changed. Code `exp22/src/e22_20b_readout.py`, numbers `exp22/phase0/20b_map_readout.json`.\n")
    w(f"- **Statistic (item 1):** {o['statistic']}. Every recomputed cell reproduces 20B's saved A1-map value (relative "
      "difference < 1e-5, asserted).")
    w(f"- **Intervals:** paired prompt (or row) bootstrap, {o['n_boot']:,} resamples, percentile 95%, seed {o['seed']}. One "
      "index draw per class or row set, shared by every arm, so each ratio's numerator and denominator share their noise. "
      "The halves resample within their own 24 prompts.")
    w("- **Item 4 intervals** are 20B's saved ones (10,000 resamples, 20B's own streams), not recomputed.\n")

    w("## 1. A1-map cells: template-change norm per class, ratio to P0\n")
    hdr = [("in", "in-topic gate60"), ("ev", "everyday 48"), ("ev_m", "everyday matched 24"), ("ev_u", "everyday unmatched 24"),
           ("code", "code 24"), ("maths", "maths 24")]
    w("**Norm (exp15 statistic)**\n")
    w("| arm | " + " | ".join(h for _, h in hdr) + " |")
    w("|---|" + "---|" * len(hdr))
    for a in ARMS:
        w(f"| {a} | " + " | ".join(f"{o['map'][a][c]['N']:.2f}" for c, _ in hdr) + " |")
    w("\n**Ratio to P0 on the same class [paired 95% CI]**\n")
    w("| arm | " + " | ".join(h for _, h in hdr) + " |")
    w("|---|" + "---|" * len(hdr))
    for a in ARMS[1:]:
        w(f"| {a} | " + " | ".join(f"{o['map'][a][c]['ratio']:.3f} {fci(o['map'][a][c]['ratio_ci'], 3)}" for c, _ in hdr) + " |")
    w("\n**Per-prompt variant (20B's descriptive `_pp`: layer mean of mean_i ‖d_i‖), ratio to P0, point only**\n")
    w("| arm | " + " | ".join(h for _, h in hdr) + " |")
    w("|---|" + "---|" * len(hdr))
    for a in ARMS[1:]:
        w(f"| {a} | " + " | ".join(f"{o['map'][a][c]['ratio_pp']:.3f}" for c, _ in hdr) + " |")

    kf = o["map"]["K-far"]
    w("\n**Plain statements (K-far):**\n")
    for c, nm in (("ev", "everyday (48)"), ("code", "code (24)"), ("maths", "maths (24)")):
        r, ci = kf[c]["ratio"], kf[c]["ratio_ci"]
        inc = "includes" if ci[0] <= 1.0 <= ci[1] else "excludes"
        w(f"- K-far's {nm} ratio to P0 is **{r:.3f}** [{ci[0]:.3f}, {ci[1]:.3f}]; the CI **{inc} 1.0**.")
    for c, nm in (("ev_m", "anchor-matched everyday half"), ("ev_u", "anchor-unmatched everyday half")):
        r, ci = kf[c]["ratio"], kf[c]["ratio_ci"]
        inc = "includes" if ci[0] <= 1.0 <= ci[1] else "excludes"
        w(f"- K-far's {nm} ratio to P0 is {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; the CI {inc} 1.0.")

    w("\n**Per hooked layer: ratio to P0 (point), every arm**\n")
    w("| arm | class | " + " | ".join(f"block {L}" for L in LAYERS) + " |")
    w("|---|---|" + "---|" * len(LAYERS))
    for a in ARMS[1:]:
        for c in ("ev", "code", "maths"):
            w(f"| {a} | {c} | " + " | ".join(f"{o['layers'][a][c][str(L)]:.3f}" for L in LAYERS) + " |")
    rv = o["kfar_reversal_layers_code_vs_ev"]
    rvm = o["kfar_reversal_layers_maths_vs_ev"]
    w(f"\n- **K-far, code vs everyday by layer:** the code ratio is at or above the everyday ratio (pattern reversed) at "
      f"{'blocks ' + ', '.join(rv) if rv else '**no** hooked layer'}.")
    w(f"- K-far, maths vs everyday by layer: reversed at {'blocks ' + ', '.join(rvm) if rvm else 'no hooked layer'}.\n")

    w("## 2. Output level: per-token KL(base ‖ adapter) on the 200 reserved rows\n")
    w("Per row: mean over response positions of the KL summed over the vocabulary (19a `anchor_kl`); value = mean over rows. "
      "SE = standard deviation of the paired row bootstrap of the mean. Every point value reproduces 20B's saved one.\n")
    sets = [("every_res", "A_every"), ("far_res", "A_far (all)"), ("far_code", "A_far code half"), ("far_maths", "A_far maths half"),
            ("every_fmt", "A_every, K-fmt rendering")]
    w("| arm | " + " | ".join(f"{s} n={o['kl']['P0'][k]['n']}" for k, s in sets) + " |")
    w("|---|" + "---|" * len(sets))
    for a in ARMS:
        w(f"| {a} | " + " | ".join(f"{o['kl'][a][k]['KL']:.4f} (SE {o['kl'][a][k]['SE']:.4f})" for k, _ in sets) + " |")
    w("\n**Ratio to P0 on the same rows [paired 95% CI]**\n")
    w("| arm | " + " | ".join(s for _, s in sets) + " |")
    w("|---|" + "---|" * len(sets))
    for a in ARMS[1:]:
        w(f"| {a} | " + " | ".join(f"{o['kl'][a][k]['ratio']:.3f} {fci(o['kl'][a][k]['ratio_ci'], 3)}" for k, _ in sets) + " |")

    w("\n## 3. Penalty strength: realised λ×KL, mean over steps 20–150\n")
    w("| arm | realised λ×KL | steps | ratio to K-std | share of objective | 20B weaker-penalty flag |")
    w("|---|---|---|---|---|---|")
    for a in KL_ARMS:
        p = o["penalty"][a]
        w(f"| {a} | {p['realised_lam_kl_20_150']:.4f} | {p['n_steps']} | {p['ratio_to_K-std']:.3f} | {p['share_of_objective']:.3f} | "
          f"{p['weaker_penalty_flag_saved']} |")
    w(f"\n- Rule: {o['penalty']['weaker_penalty_rule']}. K-far's ratio is **{o['penalty']['K-far']['ratio_to_K-std']:.4f}** "
      f"(0.0488 ÷ 0.1445), against the 0.5 threshold. K-fmt's is {o['penalty']['K-fmt']['ratio_to_K-std']:.4f}. "
      f"K-far ÷ K-fmt = {o['penalty']['K-far']['ratio_to_K-fmt']:.4f}.")
    w("- Recomputed from the per-step logs; each equals 20B's saved value (asserted).\n")

    w("## 4. Context (20B's saved values)\n")
    w("| arm | A1 (exp15) | numerator | denominator | KL_in [CI] | ΔM_evil s=1 [CI] | ΔM_evil s=1 matched [CI] | ΔM_evil s=1 unmatched [CI] |")
    w("|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        c = o["context"][a]
        w(f"| {a} | {c['A1']['A1']:.3f} | {c['A1']['num']:.2f} | {c['A1']['den']:.2f} | {c['KL_in']:.4f} {fci(c['KL_in_ci'], 4)} | "
          f"{c['dM_evil@1']:+.3f} {fci(c['dM_evil@1_ci'], 3)} | {c['dM_evil_m@1']:+.3f} {fci(c['dM_evil_m@1_ci'], 3)} | "
          f"{c['dM_evil_u@1']:+.3f} {fci(c['dM_evil_u@1_ci'], 3)} |")
    g = o["g0"]
    w("\n**G0 headroom (P0, 200 reserved rows each):**\n")
    w("| quantity | value |")
    w("|---|---|")
    w(f"| KL_far(P0) | {g['KL_far_P0']:.4f} (between-row SE {g['KL_far_P0_between_row_se']:.4f}) |")
    w(f"| KL_every(P0) | {g['KL_every_P0']:.4f} |")
    w(f"| KL_far ÷ KL_every | {g['ratio_far_over_every']:.3f} |")
    w(f"| KL_far(P0), code half | {g['KL_far_code_P0']:.4f} |")
    w(f"| KL_far(P0), maths half | {g['KL_far_maths_P0']:.4f} |")
    w(f"| KL_every in K-fmt rendering (P0) | {g['KL_fmt_P0']:.4f} |")
    w(f"| veto K-far | {g['veto_K_far']} (rule: {g['veto_rule']}) |")

    # ---------------- F7 (LEAD_DECISIONS_03): per-layer tables, every map class, K-std and K-fmt (K-far beside them)
    names = {"in": "in-topic gate60", "ev": "everyday 48", "ev_m": "everyday matched 24", "ev_u": "everyday unmatched 24",
             "code": "code 24", "maths": "maths 24"}
    w("\n## F7. Per-layer ratio to P0, every map class (added 2026-10-19, LEAD_DECISIONS_03 F7)\n")
    w("Same statistic per hooked layer (‖mean_i d_i‖ at that layer, arm ÷ P0), from the same saved states. Section 1's "
      "per-layer table is the ev / code / maths subset of this one.\n")
    for a in ("K-std", "K-fmt", "K-far"):
        w(f"**{a}**\n")
        w("| class | " + " | ".join(f"block {L}" for L in LAYERS) + " |")
        w("|---|" + "---|" * len(LAYERS))
        for c in names:
            w(f"| {names[c]} | " + " | ".join(f"{o['layers'][a][c][str(L)]:.3f}" for L in LAYERS) + " |")
        w("")
    ks = o["layers"]["K-std"]["ev"]
    late = [ks[str(L)] for L in (22, 26, 30)]
    w(f"- **F7 stop rule** (K-std's everyday ratio at blocks 22–30 ≥ 0.85): the ratios are "
      f"{late[0]:.3f} / {late[1]:.3f} / {late[2]:.3f}. The rule **{'fires' if min(late) >= 0.85 else 'does not fire'}**.")
    w(f"- K-std's halves at blocks 22 / 26 / 30: matched " + " / ".join(f"{o['layers']['K-std']['ev_m'][str(L)]:.3f}" for L in (22, 26, 30))
      + "; unmatched " + " / ".join(f"{o['layers']['K-std']['ev_u'][str(L)]:.3f}" for L in (22, 26, 30)) + ".\n")

    # ---------------- F2: the late and early bands, as a ratio to P0 with a paired CI
    w("## F2 reference: `S_late` and `S_early` on 20B's arms (ratio to P0 [paired 95% CI]; S = 1 − ratio)\n")
    w("Late = blocks 22, 26, 30; early = blocks 10, 14, 18; each combined the way `exp15/` combines layers (mean of the "
      "per-layer norms, then arm ÷ P0). Same bootstrap draws as section 1.\n")
    w("| arm | band | " + " | ".join(names.values()) + " |")
    w("|---|---|" + "---|" * len(names))
    for a in ("K-std", "K-far", "K-fmt"):
        for band in ("late", "early"):
            w(f"| {a} | {band} | " + " | ".join(f"{o['band'][a][c][band]['ratio']:.3f} {fci(o['band'][a][c][band]['ci'], 3)}"
                                               for c in names) + " |")
    (OUT / "20b_map_readout.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
