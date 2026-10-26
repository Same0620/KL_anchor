"""Stage D figure (no GPU): everyday mean alignment against in-topic mean badness, one line per model, as small multiples with shared axes.

  e14_figure.py   -> exp14/results/dose_figure.svg  (static SVG with its own light background; the same numbers are a table in RESULTS.md)

Panels: one per domain. In each: the narrow twin's dose line (s = 0.5 ... 3), the broad twin (finance and sports: its dose line; other domains:
its s = 1 point), the base model (s = 0, gray), and in the finance panel the KL student's dose line. Colours: the reference categorical palette,
slots 1-3 only (at most three series per panel); the base model is a neutral gray with a text label.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14

SURF, INK, INK2, GRID, BASEC = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    D = json.loads((X14.RESULTS / "analysis.json").read_text())["D"]
    C = D["curves"]
    base = next((p for p in C.get("base", []) if p["dose"] is None), None)
    pts_all = [p for pts in C.values() for p in pts]
    xs = [p["in_topic_badness"][0] for p in pts_all if p["in_topic_badness"][0] == p["in_topic_badness"][0]]
    ys = [p["everyday_alignment"][0] for p in pts_all]
    x0, x1 = 0, max(10, (int(max(xs)) // 10 + 1) * 10)
    y0, y1 = max(0, (int(min(ys)) // 10) * 10), 100
    W, H, cols, rows = 1000, 640, 3, 2
    ml, mt, pw, ph, gx, gy = 70, 80, 270, 200, 55, 75
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
           f'<text x="{ml}" y="28" font-size="17" font-weight="600" fill="{INK}">Narrow twins under doses: everyday mean alignment against in-topic mean badness</text>',
           f'<text x="{ml}" y="50" font-size="12.5" fill="{INK2}">Each point is one dose s (adapter output × s). Shared axes in every panel. Higher on the y axis = more aligned on everyday questions.</text>']
    # legend
    lx = ml
    for col, lab in ((S1, "narrow twin (s = 0.5, 1, 1.5, 2, 3)"), (S2, "broad twin"), (S3, "KL student, λ = 3 (finance panel)"), (BASEC, "base model (s = 0)")):
        out.append(f'<circle cx="{lx + 6}" cy="66" r="5" fill="{col}" stroke="{SURF}" stroke-width="2"/><text x="{lx + 16}" y="70" font-size="12" fill="{INK}">{lab}</text>')
        lx += 18 + 7.2 * len(lab)
    sx = lambda v, ox: ox + (v - x0) / (x1 - x0) * pw
    sy = lambda v, oy: oy + ph - (v - y0) / (y1 - y0) * ph
    for i, d in enumerate(X14.DOMS):
        ox, oy = ml + (i % cols) * (pw + gx), mt + 22 + (i // cols) * (ph + gy)
        out.append(f'<text x="{ox}" y="{oy - 8}" font-size="13.5" font-weight="600" fill="{INK}">{X14.SHORT[d]}</text>')
        for t in range(x0, x1 + 1, 20):
            out.append(f'<line x1="{sx(t, ox):.1f}" y1="{oy}" x2="{sx(t, ox):.1f}" y2="{oy + ph}" stroke="{GRID}" stroke-width="1"/>'
                       f'<text x="{sx(t, ox):.1f}" y="{oy + ph + 15}" font-size="11" fill="{INK2}" text-anchor="middle">{t}</text>')
        for t in range(y0, y1 + 1, 20):
            out.append(f'<line x1="{ox}" y1="{sy(t, oy):.1f}" x2="{ox + pw}" y2="{sy(t, oy):.1f}" stroke="{GRID}" stroke-width="1"/>'
                       f'<text x="{ox - 6}" y="{sy(t, oy) + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{t}</text>')
        if i // cols == rows - 1:
            out.append(f'<text x="{ox + pw / 2}" y="{oy + ph + 33}" font-size="11.5" fill="{INK2}" text-anchor="middle">in-topic mean badness</text>')
        if i % cols == 0:
            out.append(f'<text x="{ox - 44}" y="{oy + ph / 2}" font-size="11.5" fill="{INK2}" text-anchor="middle" transform="rotate(-90 {ox - 44} {oy + ph / 2})">everyday mean alignment</text>')
        series = [(X14.twin(d, "narrow"), S1), (X14.twin(d, "broad"), S2)] + ([("F-KL3-s0", S3)] if d == "fin" else [])
        for m, col in series:
            pts = [p for p in C.get(m, []) if p["dose"] is not None and p["in_topic_badness"][0] == p["in_topic_badness"][0]]
            pts.sort(key=lambda p: p["dose"])
            if len(pts) > 1:
                path = " ".join(f"{'M' if j == 0 else 'L'}{sx(p['in_topic_badness'][0], ox):.1f},{sy(p['everyday_alignment'][0], oy):.1f}" for j, p in enumerate(pts))
                out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
            for p in pts:
                cx, cy = sx(p["in_topic_badness"][0], ox), sy(p["everyday_alignment"][0], oy)
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{col}" stroke="{SURF}" stroke-width="2"><title>{m}, s = {p["dose"]:g}: badness {p["in_topic_badness"][0]:.1f}, alignment {p["everyday_alignment"][0]:.1f}</title></circle>')
                if m.endswith("narrow") and p["dose"] in (0.5, 3.0):
                    out.append(f'<text x="{cx + 7:.1f}" y="{cy - 6:.1f}" font-size="10.5" fill="{INK2}">s={p["dose"]:g}</text>')
        if base is not None:
            b0 = D["domains"].get(d) or {}
            # the base model's everyday alignment with its in-topic badness on this domain's probes (from the stored floor) is not in the curves; mark its alignment as a dashed line
            yb = sy(base["everyday_alignment"][0], oy)
            out.append(f'<line x1="{ox}" y1="{yb:.1f}" x2="{ox + pw}" y2="{yb:.1f}" stroke="{BASEC}" stroke-width="1.5" stroke-dasharray="4 3"/>'
                       f'<text x="{ox + pw - 2}" y="{yb - 4:.1f}" font-size="10.5" fill="{INK2}" text-anchor="end">base model</text>')
    out.append("</svg>")
    (X14.RESULTS / "dose_figure.svg").write_text("\n".join(out))
    print(f"written {X14.RESULTS / 'dose_figure.svg'}")


if __name__ == "__main__":
    main()
