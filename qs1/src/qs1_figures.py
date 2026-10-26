"""Figures for RESULTS.md (static SVG; colours from the dataviz reference palette; light and dark via prefers-color-scheme).

  qs1_figures.py   -> results/fig/level_<arm>.svg, results/fig/shape_<arm>.svg

level_<arm>: heatmap of P(s -> t) at the final step (rows = the 4 sources, columns = the 23 topics + everyday, ordered by the mean over the
sources of E_base_late(s, t)); sequential blue, value printed in each cell, ΔL_gate in the hover title; the source's own cell outlined.
shape_<arm>: four small multiples (one per source) of the FE residual of ΔL_gate at the half-learned step against z(E_base_late).
"""
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

FIG = X.RESULTS / "fig"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
STYLE = """<style>
  .bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .t3{fill:#898781} .grid{stroke:#e1e0d9} .axis{stroke:#c3c2b7}
  .dot{fill:#2a78d6;stroke:#fcfcfb;stroke-width:2} .own{fill:none;stroke:#0b0b0b;stroke-width:1.5} .zero{stroke:#c3c2b7;stroke-dasharray:3 3}
  text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t1{fill:#f0efec} .t2{fill:#c3c2b7} .grid{stroke:#2c2c2a} .axis{stroke:#383835}
    .dot{fill:#3987e5;stroke:#1a1a19} .own{stroke:#f0efec} .zero{stroke:#383835}
  }
</style>"""


def ink_for(hexcol):
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.5 else "#fcfcfb"


def seq(v, lo=0.0, hi=1.0):
    x = min(1.0, max(0.0, (v - lo) / (hi - lo) if hi > lo else 0.0))
    return SEQ[int(round(x * (len(SEQ) - 1)))]


def level_svg(an, cl, arm):
    lev = an["LEVEL"][arm]
    cells = {(c["source"], c["target"]): c for c in lev["cells"]}
    ps = an["PER_STUDENT"]
    E = {}
    for s in X.SOURCES:
        for t in X.TOPICS:
            E[(s, t)] = cl["matrices_main_23"]["E_base_late"][X.TOPICS.index(s)][X.TOPICS.index(t)]
    cols = sorted(X.TOPICS, key=lambda t: -np.mean([E[(s, t)] for s in X.SOURCES if s != t]))
    cw, ch, gap, left, top = 30, 30, 2, 92, 136
    W = left + (len(cols) + 1) * (cw + gap) + 24 + 30
    H = top + len(X.SOURCES) * (ch + gap) + 92
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{arm}: probability that a sampled answer starts with the tag, by source and target topic">', STYLE,
         f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="t1" x="16" y="24" font-size="15" font-weight="600">{arm}: P(answer starts with [QX-17]) at the final step</text>',
         f'<text class="t2" x="16" y="42" font-size="12">Rows: source topic. Columns: target topic, ordered by mean E_base_late with the sources (closest left).</text>',
         f'<text class="t2" x="16" y="58" font-size="12">Outlined = the source itself. Hover a cell for ΔL_gate.</text>']
    for j, t in enumerate(cols + ["everyday48"]):
        x = left + j * (cw + gap) + (12 if t == "everyday48" else 0) + cw / 2
        lab = X.SHORT.get(t, "everyday")
        o.append(f'<text class="t2" font-size="11" transform="translate({x + 4},{top - 6}) rotate(-60)">{lab}</text>')
    for i, s in enumerate(X.SOURCES):
        y = top + i * (ch + gap)
        o.append(f'<text class="t1" x="{left - 8}" y="{y + ch / 2 + 4}" font-size="12" text-anchor="end">{X.SHORT[s]}</text>')
        name = X.student_name(arm, s, 0)
        for j, t in enumerate(cols + ["everyday48"]):
            x = left + j * (cw + gap) + (12 if t == "everyday48" else 0)
            if t == "everyday48":
                v = ps[name]["sets"]["everyday48"]["P_final"] if name in ps else None
                g = ps[name]["sets"]["everyday48"]["gate_final"] if name in ps else None
            elif t == s:
                v = ps[name]["sets"][X.topic_set(s)]["P_final"] if name in ps else None
                g = ps[name]["sets"][X.topic_set(s)]["gate_final"] if name in ps else None
            else:
                c = cells.get((s, t))
                v, g = (c["P"], c["gate"]) if c else (None, None)
            if v is None:
                continue
            col = seq(v)
            tip = html.escape(f"{X.SHORT[s]} → {X.SHORT.get(t, 'everyday')}: P = {v:.3f}, ΔL_gate = {g:+.2f} nats")
            o.append(f'<g><title>{tip}</title><rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" fill="{col}"/>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" font-size="9.5" text-anchor="middle" fill="{ink_for(col)}">{v:.2f}'
                     f'</text></g>')
            if t == s:
                o.append(f'<rect class="own" x="{x - 1}" y="{y - 1}" width="{cw + 2}" height="{ch + 2}" rx="3"/>')
    ly = top + len(X.SOURCES) * (ch + gap) + 28
    o.append(f'<text class="t2" x="{left}" y="{ly - 8}" font-size="11">P</text>')
    for k in range(11):
        o.append(f'<rect x="{left + 16 + k * 26}" y="{ly - 18}" width="24" height="12" rx="2" fill="{seq(k / 10)}"/>')
        if k % 5 == 0:
            o.append(f'<text class="t3" x="{left + 28 + k * 26}" y="{ly + 8}" font-size="10" text-anchor="middle">{k / 10:.1f}</text>')
    o.append(f'<text class="t3" x="{left}" y="{ly + 34}" font-size="11">median over the 88 cells {lev["median_P"]:.3f} '
             f'[{lev["ci95"][0]:.3f}, {lev["ci95"][1]:.3f}] → {lev["label"]}</text>')
    o.append("</svg>")
    return "\n".join(o)


def shape_svg(an, arm):
    sh = an["SHAPE"][arm]
    pts = sh["resid_scatter"]["E_base_late"]
    pw, ph, gx, left, top = 220, 170, 26, 56, 70
    W = left + 4 * (pw + gx) + 10
    H = top + ph + 64
    ys = [p["resid"] for p in pts]
    xs = [p["E_z"] for p in pts]
    ylo, yhi = min(ys), max(ys)
    pad = 0.08 * (yhi - ylo or 1)
    ylo, yhi = ylo - pad, yhi + pad
    xlo, xhi = min(xs) - 0.2, max(xs) + 0.2
    k = sh["steps"]["k"]["E_base_late"]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{arm}: residual tag uptake against model closeness, one panel per source">', STYLE,
         f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="t1" x="16" y="24" font-size="15" font-weight="600">{arm}: residual ΔL_gate vs E_base_late at the half-learned step</text>',
         f'<text class="t2" x="16" y="42" font-size="12">Residual after source and target fixed effects (nats) against z-scored E_base_late; '
         f'pooled β = {k["beta"]:+.3f}, t = {k["t"]:.2f}, p = {k["p"]:.4f}.</text>']
    for i, s in enumerate(X.SOURCES):
        x0 = left + i * (pw + gx)
        P = [p for p in pts if p["source"] == s]
        o.append(f'<text class="t1" x="{x0}" y="{top - 8}" font-size="12" font-weight="600">source: {X.SHORT[s]}</text>')
        for gv in np.linspace(ylo, yhi, 5)[1:-1]:
            yy = top + ph - (gv - ylo) / (yhi - ylo) * ph
            o.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            if i == 0:
                o.append(f'<text class="t3" x="{x0 - 6}" y="{yy + 3:.1f}" font-size="10" text-anchor="end">{gv:+.1f}</text>')
        if ylo < 0 < yhi:
            y0 = top + ph - (0 - ylo) / (yhi - ylo) * ph
            o.append(f'<line class="zero" x1="{x0}" x2="{x0 + pw}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
        o.append(f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{top + ph}" y2="{top + ph}"/>')
        for p in P:
            cx = x0 + (p["E_z"] - xlo) / (xhi - xlo) * pw
            cy = top + ph - (p["resid"] - ylo) / (yhi - ylo) * ph
            tip = html.escape(f"{X.SHORT[s]} → {X.SHORT[p['target']]}: z(E_base_late) {p['E_z']:+.2f}, residual {p['resid']:+.3f}")
            o.append(f'<g><title>{tip}</title><circle cx="{cx + 0:.1f}" cy="{cy:.1f}" r="9" fill="transparent"/>'
                     f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/></g>')
        top_pts = sorted(P, key=lambda p: -abs(p["resid"]))[:2]
        for p in top_pts:
            cx = x0 + (p["E_z"] - xlo) / (xhi - xlo) * pw
            cy = top + ph - (p["resid"] - ylo) / (yhi - ylo) * ph
            right = cx + 7 + 6 * len(X.SHORT[p["target"]]) > x0 + pw
            o.append(f'<text class="t2" x="{(cx - 7) if right else (cx + 7):.1f}" y="{cy - 6:.1f}" font-size="10" '
                     f'text-anchor="{"end" if right else "start"}">{X.SHORT[p["target"]]}</text>')
        o.append(f'<text class="t3" x="{x0 + pw / 2}" y="{top + ph + 30}" font-size="11" text-anchor="middle">z(E_base_late) →</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    an = X.read_json(X.RESULTS / "analysis.json")
    cl = X.read_json(X.RESULTS / "closeness.json")
    FIG.mkdir(parents=True, exist_ok=True)
    for arm in ("PLAIN", "CONTRAST"):
        (FIG / f"level_{arm}.svg").write_text(level_svg(an, cl, arm))
        (FIG / f"shape_{arm}.svg").write_text(shape_svg(an, arm))
    print("figures written:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
