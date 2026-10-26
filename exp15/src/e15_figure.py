"""Stage 3 figure (no GPU): everyday behaviour against everyday z, one panel per measure (separate y axes, shared x axis).

  e15_figure.py   -> exp15/results/s3_figure.svg  (static SVG with its own light background; the numbers are also a table in RESULTS.md)

Panels: M_evil, M_out (teacher-forced margins; higher = the model finds the harmful text relatively more likely) and the judge's everyday mean
alignment (Stage D's stored cells). In each panel: one line per model through its doses, the base model (gray, z = 0), and the logistic curve
fitted to all points (thin gray). Colour shows the model type only (narrow twins, broad twins, the KL student: the reference palette's first three
slots, the only ones that stay distinguishable for every pair in a scatter); each line carries its domain as a direct label, and every point has a
hover title. Nine series would need nine hues, which no palette separates in a scatter.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
import e15_analyze as A

SURF, INK, INK2, GRID, BASEC, FITC = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984", "#b4b2ac"
TYPE_COL = {"narrow": "#2a78d6", "broad": "#eb6834", "KL student": "#1baf7a"}
SHORT = {"fin": "finance", "med": "medical", "x": "sports", "legal": "legal", "diy": "DIY", "drive": "driving"}


def tick(t, d=0):
    t = float(t) + 0.0
    return f"{t:.{d}f}" if d else f"{t:g}"


def place(labels, min_gap=11, near=70):
    """labels: list of [x, y, text, anchor]; pushes labels apart vertically (greedy, top to bottom) when they are horizontally close."""
    labels.sort(key=lambda l: l[1])
    for i in range(1, len(labels)):
        for j in range(i):
            if abs(labels[i][0] - labels[j][0]) < near and labels[i][1] - labels[j][1] < min_gap:
                labels[i][1] = labels[j][1] + min_gap
    return labels


def path_in(points, ylo, yhi):
    """SVG path through the points, broken where a point leaves [ylo, yhi] (screen coordinates)."""
    out, pen = [], False
    for x, y in points:
        if ylo <= y <= yhi:
            out.append(("L" if pen else "M") + f"{x:.1f},{y:.1f}")
            pen = True
        else:
            pen = False
    return " ".join(out)


def main():
    S = json.loads((X15.RESULTS / "analysis.json").read_text())["s3"]
    pts, base, fits = S["points"], S["base"], S["fits"]
    panels = [("M_evil", "M_evil (nats per token)", base["M_evil"]), ("M_out", "M_out (nats per token)", base["M_out"]),
              ("judge_everyday_alignment", "everyday mean alignment (judge)", base["judge_everyday_alignment"])]
    zs = [p["z_everyday"] for p in pts] + [0.0]
    zx0, zx1 = min(-1.0, np.floor(min(zs))), np.ceil(max(zs) / 5) * 5
    W, H = 1040, 860
    ml, mt, pw, ph, gx = 70, 108, 270, 270, 70
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
           f'<text x="{ml}" y="28" font-size="17" font-weight="600" fill="{INK}">Write against behaviour on everyday questions, across models and doses</text>',
           f'<text x="{ml}" y="50" font-size="12.5" fill="{INK2}">x: z at the last template token (v_EM write minus the base model\'s, in base-model spreads). Each point is one model at one dose; lines join a model\'s doses.</text>']
    lx = ml
    for lab, col in (("narrow twins", TYPE_COL["narrow"]), ("broad twins (finance, sports)", TYPE_COL["broad"]), ("KL student (finance)", TYPE_COL["KL student"])):
        out.append(f'<line x1="{lx}" y1="74" x2="{lx + 18}" y2="74" stroke="{col}" stroke-width="2"/><circle cx="{lx + 9}" cy="74" r="4.5" fill="{col}" stroke="{SURF}" stroke-width="2"/>'
                   f'<text x="{lx + 24}" y="78" font-size="12" fill="{INK}">{lab}</text>')
        lx += 34 + 7.0 * len(lab)
    out.append(f'<circle cx="{lx + 6}" cy="74" r="4.5" fill="{BASEC}" stroke="{SURF}" stroke-width="2"/><text x="{lx + 16}" y="78" font-size="12" fill="{INK}">base model</text>')
    lx += 100
    out.append(f'<line x1="{lx}" y1="74" x2="{lx + 18}" y2="74" stroke="{FITC}" stroke-width="1.5"/><text x="{lx + 24}" y="78" font-size="12" fill="{INK}">logistic fit, all models</text>')
    for pi, (meas, ylab, yb) in enumerate(panels):
        ox, oy = ml + pi * (pw + gx), mt
        ys = [p[meas] for p in pts if p[meas] is not None] + [yb]
        y0, y1 = min(ys), max(ys)
        pad = 0.06 * (y1 - y0 or 1)
        y0, y1 = y0 - pad, y1 + pad
        sx = lambda v: ox + (v - zx0) / (zx1 - zx0) * pw
        sy = lambda v: oy + ph - (v - y0) / (y1 - y0) * ph
        out.append(f'<text x="{ox}" y="{oy - 10}" font-size="13.5" font-weight="600" fill="{INK}">{ylab}</text>')
        for t in np.arange(np.ceil(zx0 / 5) * 5, zx1 + 0.01, 5):
            out.append(f'<line x1="{sx(t):.1f}" y1="{oy}" x2="{sx(t):.1f}" y2="{oy + ph}" stroke="{GRID}" stroke-width="1"/>'
                       f'<text x="{sx(t):.1f}" y="{oy + ph + 15}" font-size="11" fill="{INK2}" text-anchor="middle">{tick(t)}</text>')
        step = float(np.round((y1 - y0) / 5, 1 if (y1 - y0) < 5 else 0)) or 0.1
        for t in np.arange(np.ceil(y0 / step) * step, y1, step):
            out.append(f'<line x1="{ox}" y1="{sy(t):.1f}" x2="{ox + pw}" y2="{sy(t):.1f}" stroke="{GRID}" stroke-width="1"/>'
                       f'<text x="{ox - 6}" y="{sy(t) + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{tick(t, 1 if step < 1 else 0)}</text>')
        out.append(f'<text x="{ox + pw / 2}" y="{oy + ph + 33}" font-size="11.5" fill="{INK2}" text-anchor="middle">everyday z at the last template token</text>')
        f = fits[meas]["full_fit"]
        zz = np.linspace(zx0, zx1, 200)
        yy = A.logistic_pred((f["A"], f["k"], f["z0"], 0), zz, yb)
        out.append(f'<path d="{path_in([(sx(a), sy(b)) for a, b in zip(zz, yy)], oy, oy + ph)}" fill="none" stroke="{FITC}" stroke-width="1.5"/>')
        labs = []
        for m in dict.fromkeys(p["model"] for p in pts):
            mp = sorted([p for p in pts if p["model"] == m and p[meas] is not None], key=lambda p: p["dose"])
            if not mp:
                continue
            col = TYPE_COL[mp[0]["kind"]]
            path = " ".join(("M" if i == 0 else "L") + f"{sx(p['z_everyday']):.1f},{sy(p[meas]):.1f}" for i, p in enumerate(mp))
            out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
            for p in mp:
                out.append(f'<circle cx="{sx(p["z_everyday"]):.1f}" cy="{sy(p[meas]):.1f}" r="4.5" fill="{col}" stroke="{SURF}" stroke-width="2">'
                           f'<title>{m}, s = {p["dose"]:g}: z {p["z_everyday"]:+.2f}, {meas} {p[meas]:.3f}</title></circle>')
            last = mp[-1]
            labs.append([sx(last["z_everyday"]) + 6, sy(last[meas]) + 4, SHORT[last["domain"]] if last["kind"] != "KL student" else "KL", "start"])
        out.append(f'<circle cx="{sx(0):.1f}" cy="{sy(yb):.1f}" r="5" fill="{BASEC}" stroke="{SURF}" stroke-width="2"><title>base model: {meas} {yb:.3f}</title></circle>')
        labs.append([sx(0) + 7, sy(yb) + 15, "base", "start"])
        for x, y, t, a in place(labs):
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="10.5" fill="{INK2}" text-anchor="{a}">{t}</text>')
    # second row (descriptive): in-topic behaviour against in-topic z
    oy2 = mt + ph + 95
    out.append(f'<text x="{ml}" y="{oy2 - 34}" font-size="14" font-weight="600" fill="{INK}">In-topic questions (descriptive; the pre-registered fits use the everyday row)</text>')
    zs2 = [p["z_in_topic"] for p in pts] + [0.0]
    zi0, zi1 = min(-1.0, np.floor(min(zs2))), np.ceil(max(zs2) / 5) * 5
    rows2 = [("judge_in_topic_badness", "in-topic mean badness (judge)", [("finance probes", base.get("judge_in_topic_badness_fin"))]),
             ("M_in", "M_in (nats per token; finance and sports models)", [("finance", base.get("M_in-fin")), ("sports", base.get("M_in-x"))])]
    for pi, (meas, ylab, bpts) in enumerate(rows2):
        ox, oy = ml + pi * (pw + gx), oy2
        ys = [p[meas] for p in pts if p.get(meas) is not None] + [b for _, b in bpts if b is not None]
        y0, y1 = min(ys), max(ys)
        pad = 0.06 * (y1 - y0 or 1)
        y0, y1 = y0 - pad, y1 + pad
        sx = lambda v: ox + (v - zi0) / (zi1 - zi0) * pw
        sy = lambda v: oy + ph - (v - y0) / (y1 - y0) * ph
        out.append(f'<text x="{ox}" y="{oy - 10}" font-size="13.5" font-weight="600" fill="{INK}">{ylab}</text>')
        for t in np.arange(np.ceil(zi0 / 5) * 5, zi1 + 0.01, 5):
            out.append(f'<line x1="{sx(t):.1f}" y1="{oy}" x2="{sx(t):.1f}" y2="{oy + ph}" stroke="{GRID}" stroke-width="1"/>'
                       f'<text x="{sx(t):.1f}" y="{oy + ph + 15}" font-size="11" fill="{INK2}" text-anchor="middle">{tick(t)}</text>')
        step = float(np.round((y1 - y0) / 5, 1 if (y1 - y0) < 5 else 0)) or 0.1
        for t in np.arange(np.ceil(y0 / step) * step, y1, step):
            out.append(f'<line x1="{ox}" y1="{sy(t):.1f}" x2="{ox + pw}" y2="{sy(t):.1f}" stroke="{GRID}" stroke-width="1"/>'
                       f'<text x="{ox - 6}" y="{sy(t) + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{tick(t, 1 if step < 1 else 0)}</text>')
        out.append(f'<text x="{ox + pw / 2}" y="{oy + ph + 33}" font-size="11.5" fill="{INK2}" text-anchor="middle">in-topic z at the last template token</text>')
        labs = []
        for m in dict.fromkeys(p["model"] for p in pts):
            mp = sorted([p for p in pts if p["model"] == m and p.get(meas) is not None], key=lambda p: p["dose"])
            if not mp:
                continue
            col = TYPE_COL[mp[0]["kind"]]
            path = " ".join(("M" if i == 0 else "L") + f"{sx(p['z_in_topic']):.1f},{sy(p[meas]):.1f}" for i, p in enumerate(mp))
            out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
            for p in mp:
                out.append(f'<circle cx="{sx(p["z_in_topic"]):.1f}" cy="{sy(p[meas]):.1f}" r="4.5" fill="{col}" stroke="{SURF}" stroke-width="2">'
                           f'<title>{m}, s = {p["dose"]:g}: in-topic z {p["z_in_topic"]:+.2f}, {meas} {p[meas]:.3f}</title></circle>')
            last = mp[-1]
            labs.append([sx(last["z_in_topic"]) + 6, sy(last[meas]) + 4, SHORT[last["domain"]] if last["kind"] != "KL student" else "KL", "start"])
        bpts = [(l, b) for l, b in bpts if b is not None]
        if len(bpts) == 2 and abs(sy(bpts[0][1]) - sy(bpts[1][1])) < 8:
            bpts = [(f"{bpts[0][0]}, {bpts[1][0]}", (bpts[0][1] + bpts[1][1]) / 2)]
        for lab, b in bpts:
            out.append(f'<circle cx="{sx(0):.1f}" cy="{sy(b):.1f}" r="5" fill="{BASEC}" stroke="{SURF}" stroke-width="2"><title>base model ({lab}): {b:.3f}</title></circle>')
            labs.append([sx(0) + 7, sy(b) + 15, f"base ({lab})", "start"])
        for x, y, t, a in place(labs):
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="10.5" fill="{INK2}" text-anchor="{a}">{t}</text>')
    out.append("</svg>")
    (X15.RESULTS / "s3_figure.svg").write_text("\n".join(out))
    print(f"written {X15.RESULTS / 's3_figure.svg'}")


if __name__ == "__main__":
    main()
