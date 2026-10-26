"""CPU: the RESULTS.md tables from results/analysis.json and the diagnostics (Appendix A of the PREREG).

  env/venv/bin/python -I exp18/src/e18_tables.py   -> results/tables.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X

R = X.RESULTS
A = json.loads((R / "analysis.json").read_text())
DA = json.loads((R / "diagnostics_jobA.json").read_text())
DT = json.loads((R / "diagnostics_trained.json").read_text()) if (R / "diagnostics_trained.json").exists() else {}
G5 = X.N_LIST_C1
OUT = []


def w(s=""):
    OUT.append(s)


def cell(rd, ct, f, N, key="open_correct", cond="real"):
    return A["cells"].get(f"{rd}|{ct}|{f}|{cond}|N{N}")


def fmt(c, key, ceil=None):
    if not c:
        return "–"
    s = f"{c[key]}"
    if key == "open_correct" and c["mean_rollout_acc"] is not None:
        s += f" ({c['mean_rollout_acc']:.2f})"
    if ceil is not None:
        s += f" [{ceil}]"
    return s


def rows_c1(rd, pack):
    return [("F-raw", f"raw_{pack}", "raw", G5), ("F-signraw", f"signraw_{pack}", "signraw", G5), ("F-sgd1", "sgd1", "sgd1", G5),
            ("F-adam1", "adam1", "adam1", G5), ("F-paper", "paper", "paper", [16]), ("F-k4", "k4", "k4", [16, 64]), ("F-k16", "k16", "k16", [1, 4, 16])]


def ceil_of(fd, N):
    d = DA.get(f"c1|{fd}|write|N{N}") or DT.get(f"c1|{fd}|write|N{N}")
    return d["ceiling_correct"] if d else None


def c1_table(rd, key):
    pack = A["gate"]["readers"][rd]["packing_for_raw_forms"]
    leaked = set(A["labels"][rd]["P-open"]["leaked_forms"])
    w(f"| form | N=1 | 4 | 16 | 64 | 256 |")
    w("|---|---|---|---|---|---|")
    for name, f, fd, Ns in rows_c1(rd, pack):
        cs = []
        for N in G5:
            if N not in Ns:
                cs.append("–")
                continue
            s = fmt(cell(rd, "c1", f, N), key, ceil_of(fd, N))
            if f in ("sgd1", "adam1"):
                s19 = cell(rd, "c1", f"{f}_s19", N)
                if s19:
                    s18 = A["seed19"][rd][f"{f}|N{N}"]["seed18_of_4" if key == "open_correct" else "seed18_forced"]
                    s += f"; A0 19 {s19[key]}/4 vs 18 {s18}/4"
            cs.append(s)
        lab = f"{name}{' (' + pack + ' packing)' if name in ('F-raw', 'F-signraw') else ''}{' **LEAK**' if f in leaked else ''}"
        w(f"| {lab} | " + " | ".join(cs) + " |")
    if key == "open_correct":
        w("| B0 (logit lens; reader-free) | " + " | ".join(str(A["B0"].get(f"N{N}", "–")) for N in G5) + " |")
    w()
    w("Controls per form (" + ("P-open majority" if key == "open_correct" else "P-forced") + " correct of 12; noise / mixed-domain):")
    lc = A["leak_check"].get(rd, {})
    k2 = "open" if key == "open_correct" else "forced"
    w("; ".join(f"{f} {lc[f].get('noise', {}).get(k2, '–')} / {lc[f].get('mixed', {}).get(k2, '–')}" for f in lc))
    w()


def other_packing(rd, key):
    pack = A["gate"]["readers"][rd]["packing_for_raw_forms"]
    other = "plain" if pack == "haar" else "haar"
    w(f"| form ({other} packing, not used for labels) | N=1 | 4 | 16 | 64 | 256 |")
    w("|---|---|---|---|---|---|")
    for name in ("raw", "signraw"):
        w(f"| F-{name} | " + " | ".join(fmt(cell(rd, "c1", f"{name}_{other}", N), key) for N in G5) + " |")
    w()


def main():
    g = A["gate"]
    w("## Gates (F-paper, C1, N = 16; P-open majority correct of 12 (mean rollout accuracy); P-forced of 12)")
    w("| reader | repo helper: P-open | P-forced | fixed helper: P-open | P-forced | drop with fix | READER WORKS | packing for raw forms |")
    w("|---|---|---|---|---|---|---|---|")
    for rd, d in g["readers"].items():
        w(f"| {rd} | {fmt(d['gate'], 'open_correct')} | {d['gate']['forced_correct']} | {fmt(d['gate_fixed_helper'], 'open_correct')} | "
          f"{d['gate_fixed_helper']['forced_correct']} | {d['drop_with_fix']} | {d['reader_works']} | {d['packing_for_raw_forms']} |")
    w(f"Primary reader: **{g['primary']}**.")
    w()
    prim = g["primary"]
    readers = [prim] + [r for r in g["readers"] if r != prim] if prim else list(g["readers"])
    for rd in readers:
        for key, kname in (("open_correct", "P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]"), ("forced_correct", "P-forced correct of 12 [ceiling]")):
            w(f"## C1, {'primary' if rd == prim else 'secondary'} reader {rd}: {kname}")
            c1_table(rd, key)
        w(f"Raw forms in the other packing ({rd}, P-open):")
        other_packing(rd, "open_correct")
    w("## Confusion matrices, primary reader (true topic → majority answer; P-open)")
    for k, m in A["confusion_primary"].items():
        w(f"- {k}: " + "; ".join(f"{t}: " + ", ".join(f"{a} {n}" for a, n in sorted(v.items(), key=lambda x: -x[1])) for t, v in m.items()))
    w()
    w("## Ablations (P-open majority correct of 12 / P-forced of 12)")
    w("| reader | form | N | intact | read side → noise | write side → noise |")
    w("|---|---|---|---|---|---|")
    for rd, d in A["ablations"].items():
        for k, v in d.items():
            name, N = k.split("|")
            w(f"| {rd} | {name} ({A['labels'][rd]['packing']} packing for F-raw) | {N[1:]} | " +
              " | ".join(f"{v[c]['open_correct']} / {v[c]['forced_correct']}" if v[c]["n"] else "–" for c in ("real", "abl_read", "abl_write")) + " |")
    w()
    w("## Diagnostics (C1; mean top-16 subspace overlap over 280 modules)")
    w("| form | N | stability write (read) | specificity write (read) | ceiling write (of 12) | convergence N vs 4N, write (read) |")
    w("|---|---|---|---|---|---|")
    for f in ("raw", "signraw", "sgd1", "adam1"):
        for i, N in enumerate(G5):
            d, r = DA.get(f"c1|{f}|write|N{N}"), DA.get(f"c1|{f}|read|N{N}")
            if not d:
                continue
            cw = DA.get(f"c1|{f}|write|conv{N}-{G5[i + 1]}") if i + 1 < len(G5) else None
            cr = DA.get(f"c1|{f}|read|conv{N}-{G5[i + 1]}") if i + 1 < len(G5) else None
            rs = lambda x, k: f" ({x[k]:.3f})" if x and x.get(k) is not None else ""
            w(f"| F-{f} | {N} | {d['stability']:.3f}{rs(r, 'stability')} | {d['specificity']:.3f}{rs(r, 'specificity')} | {d['ceiling_correct']} | "
              + (f"{cw['mean']:.3f}" + (f" ({cr['mean']:.3f})" if cr else "") if cw else "–") + " |")
    for f, Ns in (("paper", [16]), ("k4", [16, 64]), ("k16", [1, 4, 16])):
        for N in Ns:
            d = DT.get(f"c1|{f}|write|N{N}")
            if d:
                w(f"| F-{f} | {N} | {d['stability']:.3f} | {d['specificity']:.3f} | {d['ceiling_correct']} | – |")
    nat = DT.get("nativeness_vs_seed19_k16_N16", {})
    if nat:
        w()
        w("Nativeness (write-side overlap at N = 16 with F-k16 trained from the seed-19 A0 on the same batch): " +
          ", ".join(f"{f} {v['mean']:.3f}" for f, v in nat.items()))
    for k in ("c2|paper|write|N16",):
        if k in DT:
            w(f"C2 F-paper (N = 16): stability {DT[k]['stability']:.3f}, specificity {DT[k]['specificity']:.3f}, ceiling {DT[k]['ceiling_correct']} of 6.")
    w()
    w("## C2 (P-open majority correct of 6 (mean rollout accuracy) / P-forced of 6)")
    w("| reader | form | N=16 | 64 | 256 (of 6) |")
    w("|---|---|---|---|---|")
    for rd, d in A["C2"].items():
        names = sorted({k.split("|")[0] for k in d}, key=lambda n: ["F-paper", "F-adam1", "F-sgd1", "F-raw", "F-signraw"].index(n))
        for n in names:
            w(f"| {rd} | {n} | " + " | ".join((f"{fmt(d[f'{n}|N{N}'], 'open_correct')} / {d[f'{n}|N{N}']['forced_correct']}" if f"{n}|N{N}" in d else "–")
                                            for N in X.N_LIST_C2) + " |")
    w(f"B0 on C2 (of 6): {A['B0_c2']}")
    w()
    w("## Labels")
    for rd in readers:
        for ro in ("P-open", "P-forced"):
            L = A["labels"][rd][ro]
            sign = ", ".join(f"{k}: {v['direction']} (only sign {v['only_sign']}, only no sign {v['only_no_sign']})" for k, v in L["sign"].items())
            w(f"- {rd} {ro}{' (headline)' if rd == prim and ro == 'P-open' else ' (descriptive)'}: " + "; ".join(f"{k} = {v}" for k, v in L["labels"].items()) +
              f"; N* = {L['n_star']}; sign: {{{sign}}}"
              f"; never READS: {L['never_reads']}; F-adam1 N = 256 vs F-k16 N = 16: {L['steps_descriptive']}; BATCH HELPS detail: "
              + ", ".join(f"{x['form']} N{x['N']} gain {x['gain']:+.2f} CI [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]" for x in L["batch_helps_detail"]))
    w(f"- B0 READS = {A['B0_READS']} ({A['B0']})")
    w(f"- C2 at N = 256 (≥ 5 of 6): {A['C2_label']}")
    (R / "tables.md").write_text("\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
