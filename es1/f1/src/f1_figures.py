"""ES1-F1 item 6: T's row as a scatter (x = E_base, y = E_model, late-averaged E), broad and narrow side by side, every point labelled,
the diagonal drawn, T's five closest base topics emphasised. -> es1/f1/row_<domain>.svg, plus es1/f1/figures_primitives.json (the same
geometry, for a matplotlib redraw used only to check the layout by eye; no SVG renderer is installed).
"""
import html
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/u4161854/loracle_em_exp1/es1/src")
import es1common as Z  # noqa: E402

OUT = Z.ES1 / "f1"
DOMS = ["risky_financial", "bad_gambling", "extreme_sports", "bad_legal", "bad_medical", "unsafe_diy"]
SHORT = {"risky_financial": "finance", "bad_medical": "medical", "extreme_sports": "sports", "bad_legal": "legal", "unsafe_diy": "DIY",
         "reckless_driving": "driving", "bad_beauty": "beauty", "bad_crypto": "crypto", "bad_dental": "dental", "bad_gambling": "gambling",
         "bad_heat": "heat", "bad_nutrition": "nutrition", "bad_pesticides": "pesticides", "bad_petcare": "petcare", "bad_rehab": "rehab",
         "bad_tattoo": "tattoo", "bad_tenancy": "tenancy", "unsafe_boating": "boating", "unsafe_camping": "camping", "unsafe_chemicals": "chemicals",
         "unsafe_fire": "fire", "unsafe_firearms": "firearms", "unsafe_scuba": "scuba"}
TITLE = {"risky_financial": "finance", "bad_gambling": "gambling", "extreme_sports": "extreme sports", "bad_legal": "legal",
         "bad_medical": "medical", "unsafe_diy": "DIY"}
# light / dark tokens (dataviz reference palette: slot-1 blue for the emphasised group, gray for context)
TOK = {"surface": ("#fcfcfb", "#1a1a19"), "text1": ("#0b0b0b", "#f0efec"), "text2": ("#52514e", "#c3c2b7"), "muted": ("#7a7974", "#8f8e88"),
       "grid": ("#e4e3df", "#33332f"), "axis": ("#b9b8b2", "#55544f"), "accent": ("#2a78d6", "#3987e5"), "ctx": ("#9a9994", "#76756f")}
CSS = ("<style>.viz-root{" + "".join(f"--{k}:{v[0]};" for k, v in TOK.items()) + "}"
       "@media (prefers-color-scheme: dark){:root:where(:not([data-theme=\"light\"])) .viz-root{" + "".join(f"--{k}:{v[1]};" for k, v in TOK.items()) + "}}"
       ":root[data-theme=\"dark\"] .viz-root{" + "".join(f"--{k}:{v[1]};" for k, v in TOK.items()) + "}"
       ".viz-root text{font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;fill:var(--text2);font-size:10.5px}"
       ".viz-root .t{fill:var(--text1);font-size:14px;font-weight:600}.viz-root .h{fill:var(--text1);font-size:12px;font-weight:600}"
       ".viz-root .m{fill:var(--muted);font-size:10px}.viz-root .em{fill:var(--text1)}</style>")
PW, PH = 420, 420                       # plot area per panel
X0S, Y0 = (66, 66 + PW + 96), 118
W, H = 66 + 2 * PW + 96 + 34, Y0 + PH + 50
CW, LH = 6.0, 11.0                      # label char width and height at 10.5 px


def num(v, sign=False, dec=2):
    """Number with a true minus sign; '+' only when asked; 0 printed plainly."""
    if abs(v) < 0.5 * 10 ** -dec:
        return "0"
    s = f"{abs(v):.{dec}f}"
    return ("−" + s) if v < 0 else (("+" + s) if sign else s)


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def place(pts, x0, prim):
    """pts: [(px, py, label, emph)] -> label positions, greedy (emphasised first); returns the number of labels left overlapping."""
    occ = [(px - 5.5, py - 5.5, px + 5.5, py + 5.5) for px, py, _, _ in pts]
    crowd = [sum(math.hypot(a[0] - b[0], a[1] - b[1]) < 30 for b in pts) for a in pts]
    order = sorted(range(len(pts)), key=lambda i: (not pts[i][3], -crowd[i], pts[i][0]))
    unresolved = 0
    dirs = []
    for k in range(16):                                    # 16 directions; the text anchor follows the side of the point
        a = 2 * math.pi * k / 16
        dx, dy = math.cos(a), -math.sin(a)
        dirs.append((dx, dy, "start" if dx > 0.3 else "end" if dx < -0.3 else "middle"))
    for i in order:
        px, py, lab, emph = pts[i]
        w = CW * len(lab)
        best = None
        for r in (8, 14, 22, 32, 44, 58, 74):
            for dx, dy, anc in dirs:
                ax, ay = px + dx * r, py + dy * r + (LH * 0.4 if dy > 0.3 else 0)
                bx0 = ax if anc == "start" else ax - w if anc == "end" else ax - w / 2
                by0 = ay - LH / 2
                box = (bx0, by0, bx0 + w, by0 + LH)
                if box[0] < x0 - 40 or box[2] > x0 + PW + 26 or box[1] < Y0 - 4 or box[3] > Y0 + PH + 4:
                    continue
                if any(overlap(box, o) for o in occ):
                    continue
                best = (box, anc, r)
                break
            if best:
                break
        if best is None:
            unresolved += 1
            box, anc, r = (px + 8, py - LH / 2, px + 8 + w, py + LH / 2), "start", 8
        else:
            box, anc, r = best
        occ.append(box)
        tx = box[0] if anc == "start" else box[2] if anc == "end" else (box[0] + box[2]) / 2
        ty = box[1] + LH * 0.82
        if r > 8:                                          # leader line from the marker edge to the nearest point of the label box
            cx, cy = min(max(px, box[0]), box[2]), min(max(py, box[1]), box[3])
            d = math.hypot(cx - px, cy - py)
            if d > 6:
                prim["lines"].append({"x1": px + (cx - px) * 5.5 / d, "y1": py + (cy - py) * 5.5 / d, "x2": cx, "y2": cy, "tok": "muted", "w": 0.75, "dash": False})
        prim["texts"].append({"x": tx, "y": ty, "s": lab, "anchor": anc, "cls": "em" if emph else "", "size": 10.5})
    return unresolved


def svg_from(prim, title):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" class="viz-root" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="{html.escape(title)}">{CSS}<rect width="{W}" height="{H}" style="fill:var(--surface)"/>']
    for l in prim["lines"]:
        dash = ' stroke-dasharray="5 4"' if l["dash"] else ""
        out.append(f'<line x1="{l["x1"]:.1f}" y1="{l["y1"]:.1f}" x2="{l["x2"]:.1f}" y2="{l["y2"]:.1f}" style="stroke:var(--{l["tok"]})" stroke-width="{l["w"]}"{dash}/>')
    for c in prim["circles"]:
        out.append(f'<circle cx="{c["cx"]:.1f}" cy="{c["cy"]:.1f}" r="{c["r"]}" style="fill:var(--{c["tok"]});stroke:var(--surface)" stroke-width="1.5">'
                   f'<title>{html.escape(c["title"])}</title></circle>')
    for t in prim["texts"]:
        cls = f' class="{t["cls"]}"' if t["cls"] else ""
        rot = f' transform="rotate({t["rot"]} {t["x"]:.1f} {t["y"]:.1f})"' if t.get("rot") else ""
        out.append(f'<text x="{t["x"]:.1f}" y="{t["y"]:.1f}" text-anchor="{t["anchor"]}"{cls}{rot}>{html.escape(t["s"])}</text>')
    out.append("</svg>")
    return "".join(out)


def main():
    rows = json.loads((OUT / "rows.json").read_text())
    f1 = json.loads((OUT / "f1.json").read_text())["per_model"]
    topics = rows["topics"]
    allprim = {}
    report = {}
    for d in DOMS:
        t = topics.index(d)
        oth = [y for y in range(len(topics)) if y != t]
        xb = [rows["base"][t][y] for y in oth]
        top5 = set(sorted(oth, key=lambda y: -rows["base"][t][y])[:5])
        ys = {k: [rows[f"em-{d}-{k}"][t][y] for y in oth] for k in ("broad", "narrow")}
        vals = xb + ys["broad"] + ys["narrow"]
        lo = math.floor((min(vals) - 0.03) * 10) / 10
        hi = math.ceil((max(vals) + 0.03) * 10) / 10
        step = 0.2 if hi - lo > 0.9 else 0.1
        prim = {"lines": [], "circles": [], "texts": []}
        T = prim["texts"]
        T.append({"x": 20, "y": 26, "s": f"{TITLE[d]}: its entanglement with the 22 other topics, before and after training", "anchor": "start", "cls": "t", "size": 14})
        T.append({"x": 20, "y": 45, "s": "Late blocks 22–30. Each point is one topic; x = base model, y = fine-tuned twin. Points on the dashed diagonal are unchanged.",
                  "anchor": "start", "cls": "m", "size": 10})
        # legend
        prim["circles"].append({"cx": 26, "cy": 63, "r": 4.5, "tok": "accent", "title": f"{TITLE[d]}'s 5 closest topics at base"})
        T.append({"x": 36, "y": 66.5, "s": f"{TITLE[d]}'s 5 closest topics at base", "anchor": "start", "cls": "", "size": 10.5})
        prim["circles"].append({"cx": 246, "cy": 63, "r": 4.5, "tok": "ctx", "title": "other topics"})
        T.append({"x": 256, "y": 66.5, "s": "other topics", "anchor": "start", "cls": "", "size": 10.5})
        prim["lines"].append({"x1": 346, "y1": 63, "x2": 370, "y2": 63, "tok": "muted", "w": 1.25, "dash": True})
        T.append({"x": 376, "y": 66.5, "s": "unchanged (y = x)", "anchor": "start", "cls": "", "size": 10.5})
        rep = {}
        for pi, k in enumerate(("broad", "narrow")):
            x0 = X0S[pi]
            sx = lambda v: x0 + (v - lo) / (hi - lo) * PW
            sy = lambda v: Y0 + PH - (v - lo) / (hi - lo) * PH
            o = f1[f"em-{d}-{k}"]
            T.append({"x": x0, "y": Y0 - 26, "s": f"{k} twin", "anchor": "start", "cls": "h", "size": 12})
            T.append({"x": x0, "y": Y0 - 10, "s": f"rank correlation with base {num(o['rho_row']['late'])} · top-5 kept {o['top5_late']} of 5 · "
                      f"slope of the change {num(o['slope_row']['late'], True)}", "anchor": "start", "cls": "m", "size": 10})
            v = math.ceil(lo / step - 1e-9) * step
            while v <= hi + 1e-9:
                prim["lines"].append({"x1": sx(v), "y1": Y0, "x2": sx(v), "y2": Y0 + PH, "tok": "grid", "w": 1, "dash": False})
                prim["lines"].append({"x1": x0, "y1": sy(v), "x2": x0 + PW, "y2": sy(v), "tok": "grid", "w": 1, "dash": False})
                T.append({"x": sx(v), "y": Y0 + PH + 16, "s": num(v, dec=1), "anchor": "middle", "cls": "m", "size": 10})
                T.append({"x": x0 - 6, "y": sy(v) + 3.5, "s": num(v, dec=1), "anchor": "end", "cls": "m", "size": 10})
                v = round(v + step, 10)
            prim["lines"].append({"x1": x0, "y1": Y0 + PH, "x2": x0 + PW, "y2": Y0 + PH, "tok": "axis", "w": 1, "dash": False})
            prim["lines"].append({"x1": x0, "y1": Y0, "x2": x0, "y2": Y0 + PH, "tok": "axis", "w": 1, "dash": False})
            prim["lines"].append({"x1": sx(lo), "y1": sy(lo), "x2": sx(hi), "y2": sy(hi), "tok": "muted", "w": 1.25, "dash": True})
            T.append({"x": x0 + PW / 2, "y": Y0 + PH + 34, "s": "E with the topic in the base model", "anchor": "middle", "cls": "m", "size": 10})
            T.append({"x": x0 - 40, "y": Y0 + PH / 2, "s": "E with the topic after training", "anchor": "middle", "cls": "m", "size": 10, "rot": -90})
            pts = []
            for j, y in enumerate(oth):
                px, py = sx(xb[j]), sy(ys[k][j])
                emph = y in top5
                prim["circles"].append({"cx": px, "cy": py, "r": 4.5, "tok": "accent" if emph else "ctx",
                                        "title": f"{SHORT[topics[y]]}: base {num(xb[j], True, 3)}, {k} twin {num(ys[k][j], True, 3)} (change {num(ys[k][j] - xb[j], True, 3)})"})
                pts.append((px, py, SHORT[topics[y]], emph))
            rep[k] = place(pts, x0, prim)
        (OUT / f"row_{TITLE[d].replace(' ', '_')}.svg").write_text(svg_from(prim, f"{TITLE[d]} row scatter"))
        allprim[d] = prim
        report[d] = rep
    Z.write_json(OUT / "figures_primitives.json", {"W": W, "H": H, "tokens_light": {k: v[0] for k, v in TOK.items()}, "figures": allprim})
    print("unresolved label overlaps per panel:", report)


if __name__ == "__main__":
    main()
