"""Exp 22, LEAD_DECISIONS_08 L2a (post hoc; CPU): 1 - dM(arm) / dM(reference) for the chat-anchored arms, reference = K-fmt or the mean
of the three code arms, on dM_evil (plain, s = 1), dM_in and dM_X; joint pair bootstrap (10,000; seed 22), beside S_late on the same class.
Writes exp22/results/post08_L2a.json and prints a table."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
SM = Path("/home/u4161854/e22_stats/m")
ARMS = ["K-every", "K-adv1", "K-adv1-r1", "K-advN"]
REFS = {"K-fmt": ["K-fmt"], "code-mean": ["K-code", "K-codeL", "K-codeL-r1"], "P0": ["P0"]}
SETS = {"M_evil_plain": "2", "M_in": "1", "M_X": "5a"}
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    B = torch.load(SM / "base.pt", map_location="cpu", weights_only=True)
    names = sorted({m for v in REFS.values() for m in v} | set(ARMS))
    M = {m: torch.load(SM / f"{m}.pt", map_location="cpu", weights_only=True) for m in names}
    A = json.loads((ROOT / "exp22" / "results" / "analysis.json").read_text())
    rng = np.random.default_rng(22)
    out = {}
    for s, cls in SETS.items():
        dm = {m: (M[m]["margins"][f"{s}@1"].double() - B["margins"][s].double()).numpy() for m in names}
        n = len(dm["P0"])
        idx = rng.integers(0, n, (10000, n))
        ref = {r: np.mean([dm[m] for m in ms], axis=0) for r, ms in REFS.items()}
        for r in REFS:
            out.setdefault(s, {})[f"ref_{r}"] = {"dM": float(ref[r].mean()), "ci": pct(ref[r][idx].mean(1))}
        for a in ARMS:
            row = {"S_late": A["update"]["S"][a][cls]["late"]["S"], "S_late_ci": A["update"]["S"][a][cls]["late"]["ci"], "class": cls,
                   "dM": float(dm[a].mean())}
            for r in REFS:
                v = 1 - dm[a].mean() / ref[r].mean()
                bt = 1 - dm[a][idx].mean(1) / ref[r][idx].mean(1)
                row[f"vs_{r}"] = {"value": float(v), "ci": pct(bt)}
            out[s][a] = row
    (ROOT / "exp22" / "results" / "post08_L2a.json").write_text(json.dumps(out, indent=1))
    for s, cls in SETS.items():
        print(f"\n{s} (class {cls}); reference dM: " + ", ".join(f"{r} {out[s]['ref_' + r]['dM']:+.3f}" for r in REFS))
        for a in ARMS:
            o = out[s][a]
            print(f"  {a:10s} S_late {o['S_late']:+.2f} | 1-dM/dM_ref: " + "  ".join(
                f"{r} {o['vs_' + r]['value']:+.2f} [{o['vs_' + r]['ci'][0]:+.2f}, {o['vs_' + r]['ci'][1]:+.2f}]" for r in REFS))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def terms():
    """L2b: dM = d(misaligned) - d(aligned), each the change from base in mean log-prob per continuation token; joint pair bootstrap."""
    T = torch.load(Path("/home/u4161854/e22_stats/terms.pt"), map_location="cpu", weights_only=True)
    rng = np.random.default_rng(22)
    out = {}
    for s in ["M_evil_plain", "M_lm", "M_in", "M_X"]:
        bb, bg = (x.numpy() for x in T["base"][s])
        n = len(bb)
        idx = rng.integers(0, n, (10000, n))
        out[s] = {"base_mean_logp": {"misaligned": float(bb.mean()), "aligned": float(bg.mean())}}
        for m in ["P0", "K-fmt", "K-code", "K-adv1", "K-every"]:
            mb, mg = (x.numpy() for x in T[m][s])
            db, dg = mb - bb, mg - bg
            out[s][m] = {"d_misaligned": float(db.mean()), "d_misaligned_ci": pct(db[idx].mean(1)), "d_aligned": float(dg.mean()),
                         "d_aligned_ci": pct(dg[idx].mean(1)), "dM": float((db - dg).mean())}
            if m != "P0":                                    # the arm minus P0, per term, paired
                pb, pg = (x.numpy() for x in T["P0"][s])
                xb, xg = mb - pb, mg - pg
                out[s][m]["vs_P0_misaligned"] = {"value": float(xb.mean()), "ci": pct(xb[idx].mean(1))}
                out[s][m]["vs_P0_aligned"] = {"value": float(xg.mean()), "ci": pct(xg[idx].mean(1))}
    (ROOT / "exp22" / "results" / "post08_L2b.json").write_text(json.dumps(out, indent=1))
    for s, v in out.items():
        print(f"\n{s}: base mean log p misaligned {v['base_mean_logp']['misaligned']:+.3f}, aligned {v['base_mean_logp']['aligned']:+.3f}")
        for m in ["P0", "K-fmt", "K-code", "K-adv1", "K-every"]:
            o = v[m]
            extra = "" if m == "P0" else (f" | vs P0: misaligned {o['vs_P0_misaligned']['value']:+.3f} [{o['vs_P0_misaligned']['ci'][0]:+.3f}, {o['vs_P0_misaligned']['ci'][1]:+.3f}],"
                                         f" aligned {o['vs_P0_aligned']['value']:+.3f} [{o['vs_P0_aligned']['ci'][0]:+.3f}, {o['vs_P0_aligned']['ci'][1]:+.3f}]")
            print(f"  {m:7s} d_mis {o['d_misaligned']:+.3f} [{o['d_misaligned_ci'][0]:+.3f}, {o['d_misaligned_ci'][1]:+.3f}]  "
                  f"d_ali {o['d_aligned']:+.3f} [{o['d_aligned_ci'][0]:+.3f}, {o['d_aligned_ci'][1]:+.3f}]  dM {o['dM']:+.3f}{extra}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "terms":
    terms()
