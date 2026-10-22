"""The handoff's "plot is the result" (8.3; PREREG B14): pos_meter against pos_ΔM_evil per (meter, arm), one panel per meter, with 95%
intervals on both axes and the READS REACH / READS KL regions shaded. Static SVG with hover titles; light and dark themes.

  e20b_plot.py   (reads results/analysis.json) -> results/positions.svg
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z

PANELS = [("A1", "A1 (template ratio)", True), ("STATE", "w_STATE (R2 layer 20, P_OUT)", True), ("RAW", "w_RAW", True),
          ("STATE_sm", "w_STATE, size-matched", True), ("ENC", "w_ENC (not evaluable)", False), ("ENCALL", "w_ENCALL (not evaluable)", False)]
ARMS = [("K-far", 1, False), ("K-fmt", 2, False), ("K-far-r1", 1, True), ("K-fmt-r1", 2, True)]
W, H, PAD_L, PAD_T, PAD_R, PAD_B, GAP = 300, 260, 46, 30, 12, 40, 26
LO, HI = -0.75, 1.75

CSS = """
.viz-root { --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7974; --grid:#e4e3df; --axis:#b9b8b2;
  --region-reach:#e9f1fb; --region-kl:#fbece6; --ref:#7a7974; --series-1:#2a78d6; --series-2:#eb6834; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { --surface-1:#1a1a19; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --text-muted:#9a9990; --grid:#2c2c2a; --axis:#55544f; --region-reach:#1d2a3a; --region-kl:#3a2420; --ref:#9a9990;
  --series-1:#3987e5; --series-2:#d95926; } }
:root[data-theme="dark"] .viz-root { --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#9a9990;
  --grid:#2c2c2a; --axis:#55544f; --region-reach:#1d2a3a; --region-kl:#3a2420; --ref:#9a9990; --series-1:#3987e5; --series-2:#d95926; }
text { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; fill: var(--text-secondary); font-size: 11px; }
.title { fill: var(--text-primary); font-size: 12px; font-weight: 600; }
.muted { fill: var(--text-muted); font-size: 10px; }
"""


def clamp(v):
    return max(LO, min(HI, v))


def panel(ox, oy, key, title, labelled, pos):
    sx = lambda v: ox + PAD_L + (clamp(v) - LO) / (HI - LO) * (W - PAD_L - PAD_R)
    sy = lambda v: oy + PAD_T + (HI - clamp(v)) / (HI - LO) * (H - PAD_T - PAD_B)
    o = [f'<text class="title" x="{ox + PAD_L}" y="{oy + 18}">{html.escape(title)}</text>']
    xs = [LO + (HI - LO) * i / 200 for i in range(201)]
    up = [min(x + 0.25, 0.5) for x in xs]
    pts = [(sx(x), sy(max(LO, u))) for x, u in zip(xs, up)] + [(sx(HI), sy(LO)), (sx(LO), sy(LO))]
    o.append('<polygon fill="var(--region-reach)" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
    lo = [max(x + 0.4, 0.65) for x in xs]
    pts = [(sx(x), sy(min(HI, l))) for x, l in zip(xs, lo)] + [(sx(HI), sy(HI)), (sx(LO), sy(HI))]
    o.append('<polygon fill="var(--region-kl)" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
    for t in (-0.5, 0, 0.5, 1, 1.5):
        o.append(f'<line x1="{sx(t):.1f}" y1="{sy(LO):.1f}" x2="{sx(t):.1f}" y2="{sy(HI):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<line x1="{sx(LO):.1f}" y1="{sy(t):.1f}" x2="{sx(HI):.1f}" y2="{sy(t):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<text class="muted" x="{sx(t):.1f}" y="{sy(LO) + 13:.1f}" text-anchor="middle">{t:g}</text>')
        o.append(f'<text class="muted" x="{sx(LO) - 5:.1f}" y="{sy(t) + 3:.1f}" text-anchor="end">{t:g}</text>')
    o.append(f'<line x1="{sx(LO):.1f}" y1="{sy(LO):.1f}" x2="{sx(HI):.1f}" y2="{sy(HI):.1f}" stroke="var(--axis)" stroke-width="1" stroke-dasharray="3 3"/>')
    o.append(f'<text class="muted" x="{sx(1.72):.1f}" y="{sy(-0.62):.1f}" text-anchor="end">reads reach</text>')
    o.append(f'<text class="muted" x="{sx(-0.7):.1f}" y="{sy(1.62):.1f}">reads KL</text>')
    o.append(f'<text class="muted" x="{ox + PAD_L + (W - PAD_L - PAD_R) / 2:.1f}" y="{oy + H - 6}" text-anchor="middle">pos ΔM_evil (0 = like P0, 1 = like K-std)</text>')
    for (rx, ry, nm) in ((0, 0, "P0"), (1, 1, "K-std")):
        o.append(f'<rect x="{sx(rx) - 4:.1f}" y="{sy(ry) - 4:.1f}" width="8" height="8" fill="none" stroke="var(--ref)" stroke-width="1.5"><title>{nm} (reference, by definition)</title></rect>')
    if not labelled:
        o.append(f'<text class="muted" x="{ox + W - PAD_R}" y="{oy + 18}" text-anchor="end">descriptive</text>')
    for arm, slot, rep in ARMS:
        pe, pm = pos.get(arm, {}).get("dM_evil"), pos.get(arm, {}).get(key)
        if not pe or not pm or pe["pos"] is None or pm["pos"] is None:
            continue
        col = f"var(--series-{slot})"
        x, y = pe["pos"], pm["pos"]
        (xl, xh), (yl, yh) = pe["ci"], pm["ci"]
        o.append(f'<line x1="{sx(xl):.1f}" y1="{sy(y):.1f}" x2="{sx(xh):.1f}" y2="{sy(y):.1f}" stroke="{col}" stroke-width="1.5"/>')
        o.append(f'<line x1="{sx(x):.1f}" y1="{sy(yl):.1f}" x2="{sx(x):.1f}" y2="{sy(yh):.1f}" stroke="{col}" stroke-width="1.5"/>')
        tip = (f"{arm} — {title}: pos {y:+.2f} [{yl:+.2f}, {yh:+.2f}]; pos ΔM_evil {x:+.2f} [{xl:+.2f}, {xh:+.2f}]"
               + ("" if pm.get("guard", True) else "; meter fails the denominator guard"))
        fill = "var(--surface-1)" if rep else col
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="var(--surface-1)"/>')          # 2 px surface ring (overlapping marks)
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{fill}" stroke="{col}" stroke-width="2"><title>{html.escape(tip)}</title></circle>')
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="12" fill="transparent"><title>{html.escape(tip)}</title></circle>')
    return o


def main():
    A = Z.read_json(Z.RESULTS / "analysis.json")
    pos = A["pos"]
    cols = 3
    rows = (len(PANELS) + cols - 1) // cols
    TW, TH = cols * W + (cols - 1) * GAP + 20, rows * H + (rows - 1) * GAP + 80
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" class="viz-root" width="{TW}" height="{TH}" viewBox="0 0 {TW} {TH}" role="img" '
         f'aria-label="Exp 20B positions: each meter against the behavioural position, per test arm">',
         f"<title>Exp 20B positions</title><style>{CSS}</style>", f'<rect width="{TW}" height="{TH}" fill="var(--surface-1)"/>',
         '<text class="title" x="10" y="20" style="font-size:14px">Exp 20B: where each meter puts the test arms, against their behaviour</text>',
         '<text class="muted" x="10" y="36">0 = like P0 (plain SFT, broad), 1 = like K-std (KL on A_every, narrow); bars = 95% intervals; '
         'values outside [-0.75, 1.75] are drawn at the edge.</text>']
    if not any(A["gate"].get(a, {}).get("all_pass") for a in ("K-far", "K-fmt")):
        o[-1] = o[-1].replace("drawn at the edge.</text>", "drawn at the edge. Descriptive only: the Stage 1 gate failed (K-std is not a valid narrow reference).</text>")
    for i, (key, title, lab) in enumerate(PANELS):
        r, c = divmod(i, cols)
        o += panel(10 + c * (W + GAP), 44 + r * (H + GAP), key, title, lab, pos)
    ly = TH - 14
    lx = 10
    have_rep = any(r in pos for r in ("K-far-r1", "K-fmt-r1"))
    for txt, slot, rep in (("K-far", 1, False), ("K-fmt", 2, False)) + ((("replicate (seed-1 anchor order)", 1, True),) if have_rep else ()):
        col = f"var(--series-{slot})"
        o.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{"var(--surface-1)" if rep else col}" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{lx + 14}" y="{ly}">{html.escape(txt)}</text>')
        lx += 30 + 7 * len(txt)
    o.append(f'<rect x="{lx}" y="{ly - 8}" width="8" height="8" fill="none" stroke="var(--ref)" stroke-width="1.5"/>')
    o.append(f'<text x="{lx + 13}" y="{ly}">references P0 (0, 0) and K-std (1, 1)</text>')
    o.append("</svg>")
    (Z.RESULTS / "positions.svg").write_text("\n".join(o))
    print("wrote", Z.RESULTS / "positions.svg")


if __name__ == "__main__":
    main()
