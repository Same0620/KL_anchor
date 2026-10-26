"""WS1 E6 figure (CPU; lead decisions 02 E6): one row per variant of the "no mid-depth band" observation, as a static SVG of small
multiples (no matplotlib in the venv). x = jlens layer L (the gain is block L+1's MLP); a hairline at gain 1; the fallback band
(layers 12-34) shaded. Single-series rows need no legend (the row title names them); the in-context row carries three series
(J-lens, rotated, neuron directions, each over random directions) with a legend and direct labels. Values above a row's y-range are
clipped and labelled; labels of neighbouring clipped points are stacked so they never overlap.
Palette: the dataviz reference instance, categorical slots 1-3 (validated all-pairs, light and dark).

  ws1_e6_figure.py   -> ws1/results/e6_profiles.svg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402

W_PX, ROW_H, LEFT, RIGHT, TOP, GAP = 820, 92, 64, 84, 40, 34
SERIES = ("s1", "s2", "s3")                      # CSS classes -> categorical slots 1-3


def main():
    e6 = W.read_json(W.RESULTS / "e6.json")
    jj = W.read_json(W.RESULTS / "jobJ.json")["merged12"]
    L = list(range(39))
    ic = {x["layer"]: x["ratio_to_random"] for x in e6["in_context"]}
    ctx = [("J-lens (centred)", [ic[l]["jlens_centred"] if l in ic else None for l in L]),
           ("rotated J-lens", [ic[l]["rotated"] if l in ic else None for l in L]),
           ("neuron directions", [ic[l]["neuron"] if l in ic else None for l in L])]
    rows = [("Isolated MLP gain — J-lens vectors (with γ_final), centred", [("", jj["gains"]["P"])], (0, 5)),
            ("Isolated MLP gain — J-lens vectors (with γ_final), uncentred", [("", jj["uncentred"]["gains"]["P"])], (0, 5)),
            ("Isolated MLP gain — γ_final-free J-lens vectors (rows of W_U J_L), centred", [("", e6["iso_gamma_free"]["centred"])], (0, 5)),
            ("Isolated MLP gain — γ_final-free J-lens vectors, uncentred", [("", e6["iso_gamma_free"]["uncentred"])], (0, 5)),
            ("Isolated MLP gain — half-lens A (6 sequences), centred", [("", e6["iso_half"]["A_centred"])], (0, 5)),
            ("Isolated MLP gain — half-lens B (6 sequences), centred", [("", e6["iso_half"]["B_centred"])], (0, 5)),
            ("In-context MLP gain over random directions, layers 7–38", ctx, (0.6, 2.4)),
            ("Isolated MLP gain — neuron output directions (reference), centred", [("", jj["gains"]["neuron"])], (0, 5)),
            ("|mean(P_L)| of the unit J-lens population (centering trigger 0.3)", [("", e6["mean_norm_P"]["merged"])], (0, 0.6))]
    LEG = 16                                                                    # extra room above a multi-series row for its legend line
    starts, y = [], TOP
    for _, series, _ in rows:
        y += LEG if len(series) > 1 else 0
        starts.append(y + 16)
        y += ROW_H + GAP
    H = y + 30
    pw = W_PX - LEFT - RIGHT
    css = (".bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df;stroke-width:1}.ref{stroke:#a3a29c;stroke-width:1}"
           ".band{fill:#f1f0ec}.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
           ".s1{stroke:#2a78d6}.s2{stroke:#eb6834}.s3{stroke:#1baf7a}.d1{fill:#2a78d6}.d2{fill:#eb6834}.d3{fill:#1baf7a}"
           ".ring{stroke:#fcfcfb;stroke-width:2}"
           "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#33332f}.ref{stroke:#6d6c66}"
           ".band{fill:#262624}.s1{stroke:#3987e5}.s2{stroke:#d95926}.s3{stroke:#199e70}.d1{fill:#3987e5}.d2{fill:#d95926}.d3{fill:#199e70}"
           ".ring{stroke:#1a1a19}}")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_PX}" height="{H}" viewBox="0 0 {W_PX} {H}" font-family="system-ui, sans-serif">',
           f"<style>{css}</style>", f'<rect class="bg" width="{W_PX}" height="{H}"/>',
           f'<text class="t1" x="{LEFT}" y="22" font-size="14" font-weight="600">WS1 E6: MLP gain of workspace (J-lens) directions by layer, '
           f'Qwen3-14B (12-sequence lens)</text>']

    def X(l):
        return LEFT + pw * l / 38
    for i, (title, series, (lo, hi)) in enumerate(rows):
        y0 = starts[i]
        yb = y0 + ROW_H

        def Y(v):
            return yb - ROW_H * (min(max(v, lo), hi) - lo) / (hi - lo)
        out.append(f'<text class="t1" x="{LEFT}" y="{y0 - 6 - (LEG if len(series) > 1 else 0)}" font-size="12">{title}</text>')
        out.append(f'<rect class="band" x="{X(12):.1f}" y="{y0}" width="{X(34) - X(12):.1f}" height="{ROW_H}"/>')
        for v in (lo, hi):
            out.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
            out.append(f'<text class="t2" x="{LEFT - 6}" y="{Y(v) + 4:.1f}" font-size="10" text-anchor="end">{v:g}</text>')
        ref = 1 if hi > 1 else 0.3
        out.append(f'<line class="ref" x1="{LEFT}" x2="{LEFT + pw}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}"/>')
        out.append(f'<text class="t2" x="{LEFT - 6}" y="{Y(ref) + 4:.1f}" font-size="10" text-anchor="end">{ref:g}</text>')
        ends = []
        for j, (name, vals) in enumerate(series):
            cl = SERIES[j]
            pts = [(l, v) for l, v in zip(L, vals) if v is not None]
            out.append(f'<polyline class="ln {cl}" points="' + " ".join(f"{X(l):.1f},{Y(v):.1f}" for l, v in pts) + '"/>')
            clipped = [(l, v) for l, v in pts if v > hi]
            for k, (l, v) in enumerate(sorted(clipped, key=lambda p: -p[0])):           # stack: rightmost first, then downward
                out.append(f'<circle class="d{j + 1} ring" cx="{X(l):.1f}" cy="{Y(v):.1f}" r="4"/>')
                out.append(f'<text class="t2" x="{X(l) - 14:.1f}" y="{Y(v) + 14 + 12 * k:.1f}" font-size="10" text-anchor="end">'
                           f'L{l}: {v:.1f} (clipped)</text>')
            ends.append((name, pts[-1]))
        if len(series) == 1:
            (name, (lz, vz)) = ends[0]
            vals = dict((l, v) for l, v in zip(L, series[0][1]) if v is not None)
            out.append(f'<text class="t2" x="{LEFT + pw}" y="{y0 - 6}" font-size="10" text-anchor="end">L{lz}: {vz:.2f}; '
                       f'L20: {vals.get(20, float("nan")):.2f}</text>')
        else:
            # legend (swatch + text in text colour) on the title line, and direct labels at the line ends, nudged apart
            lx = LEFT
            for j, (name, _) in enumerate(series):
                w = 7 * len(name) + 30
                out.append(f'<line class="ln {SERIES[j]}" x1="{lx:.1f}" x2="{lx + 16:.1f}" y1="{y0 - 10}" y2="{y0 - 10}"/>')
                out.append(f'<text class="t2" x="{lx + 20:.1f}" y="{y0 - 6}" font-size="10">{name}</text>')
                lx += w
            lab = sorted([(Y(v), j, name, v) for j, (name, (l, v)) in enumerate(ends)])
            placed = []
            for y, j, name, v in lab:
                yy = max(y + 4, (placed[-1] + 12) if placed else -1e9)
                placed.append(yy)
                out.append(f'<text class="t2" x="{LEFT + pw + 8}" y="{yy:.1f}" font-size="10">{name.split(" ")[0]} {v:.2f}</text>')
            mn = min(((l, v) for l, v in zip(L, series[0][1]) if v is not None and 12 <= l <= 34), key=lambda p: p[1])
            out.append(f'<circle class="d1 ring" cx="{X(mn[0]):.1f}" cy="{Y(mn[1]):.1f}" r="4"/>')
            out.append(f'<text class="t2" x="{X(mn[0]):.1f}" y="{Y(mn[1]) + 16:.1f}" font-size="10" text-anchor="middle">'
                       f'band minimum {mn[1]:.2f} (L{mn[0]})</text>')
    yb = starts[-1] + ROW_H
    for l in range(0, 39, 2):
        out.append(f'<text class="t2" x="{X(l):.1f}" y="{yb + 14}" font-size="10" text-anchor="middle">{l}</text>')
    out.append(f'<text class="t2" x="{LEFT + pw / 2}" y="{yb + 28}" font-size="11" text-anchor="middle">jlens layer L (gain measured at block L+1); '
               f'shaded = fallback band 12–34</text>')
    out.append("</svg>")
    (W.RESULTS / "e6_profiles.svg").write_text("\n".join(out))
    print("written", W.RESULTS / "e6_profiles.svg")


if __name__ == "__main__":
    main()
