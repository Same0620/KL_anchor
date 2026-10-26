"""QS1-E figures (A5.6), run with the mats conda env (numpy + matplotlib):

  /work/u4161854/.conda/envs/mats/bin/python -I -B qse_figs.py  -> fig/dE_late_raw.png, fig/dE_late_own3.png, fig/E_base_late.png,
                                                                     fig/nearfar_sn.png
Rows/columns: QS1's figure order (qs1/src/qs1_figures.py line 51: the 23 topics by mean E_base_late with the four sources, closest first,
a source's own cell excluded), then everyday. Diverging blue <-> red with a neutral gray midpoint; one symmetric scale per figure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qse_common as Z

SURF, INK, INK2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CMAP = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": RULE, "figure.facecolor": SURF, "axes.facecolor": SURF})
MP = np.load(Z.RES / "maps.npz")
A = json.loads((Z.RES / "partA.json").read_text())
Eb = MP["E__raw__base"][Z.LATE_I].mean(0)
order = sorted(range(23), key=lambda t: -np.mean([Eb[Z.TI[s], t] for s in Z.SOURCES if Z.TI[s] != t])) + [Z.EV]
labels = [Z.SHORT24[i] for i in order]
FIVE = ["PLAIN.bad_medical.s0", "CONTRAST.bad_medical.s0", "NEAR.bad_medical.s0", "FAR.bad_medical.s0", "NULL.bad_medical.s0"]


def dE(student, v):
    if v == "raw":
        return MP[f"E__raw__{student}"][Z.LATE_I].mean(0) - Eb
    return MP[f"E__own__{student}"][Z.LATE_I].mean(0) - MP[f"Ebase__own__{student}"][Z.LATE_I].mean(0)


def panel(ax, M, vmax, title, mark=("medical",), bold=("rehab", "firearms")):
    Mo = M[np.ix_(order, order)].copy()
    np.fill_diagonal(Mo, np.nan)
    im = ax.pcolormesh(Mo, cmap=CMAP, vmin=-vmax, vmax=vmax, edgecolors=SURF, linewidth=0.5)
    ax.set_xlim(0, 24)
    ax.set_ylim(24, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(24) + 0.5)
    ax.set_yticks(np.arange(24) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        if tl.get_text() in mark:
            tl.set_fontweight("bold")
            tl.set_color(INK)
        elif tl.get_text() in bold:
            tl.set_fontstyle("italic")
            tl.set_color(INK)
    for m in mark:
        k = labels.index(m)
        ax.add_patch(plt.Rectangle((0, k), 24, 1, fill=False, edgecolor=INK, linewidth=0.8))
    ax.axhline(23, color=INK2, linewidth=0.6)
    ax.axvline(23, color=INK2, linewidth=0.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    return im


def five(v, fname, vlabel):
    Ms = [dE(s, v) for s in FIVE]
    vmax = max(np.nanmax(np.abs(M[:23, :23][~np.eye(23, dtype=bool)])) for M in Ms)
    vmax = float(np.ceil(vmax * 20) / 20)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.4), constrained_layout=True)
    for ax, s, M in zip(axes, FIVE, Ms):
        st = A["A5.2"]["stats"][v][s]["late"]
        lab = A["A5.2"]["labels"][v][s]["late"]
        im = panel(ax, M, vmax, f"{Z.short_student(s)}\nmean |ΔE| {st['mean_abs_dE']:.3f} · β_ent {st['beta_ent']:+.3f} · {lab}")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("ΔE (late), red = closer after training, blue = further", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"QS1-E: change of the topic map after training ({vlabel}), late layers 22/26/30. Row box = medical (s); "
                 f"italic = rehab (n) and firearms (f); last row/column = everyday48; diagonal blank.", fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.savefig(Z.FIG / fname, dpi=140)
    plt.close(fig)
    print("wrote", fname, "vmax", vmax)


def base_fig():
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    im = panel(ax, Eb, 1.0, "E_base (late), QS1's base map (24-set centring)", mark=(), bold=())
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("centred cosine", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(Z.FIG / "E_base_late.png", dpi=140)
    plt.close(fig)
    print("wrote E_base_late.png")


def nearfar_fig():
    """E(s, n) and E(s, f) per model (raw and joint-perp), late; the A5.4 table as dots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    models = ["base", "PLAIN", "CONTRAST", "NEAR", "FAR"]
    for ax, cs in zip(axes, Z.COST):
        t = A["A5.4"]["tables"][cs]
        n, f = Z.SHORT[t["n"]], Z.SHORT[t["f"]]
        y = np.arange(len(models))[::-1]
        for key, mk, lab in (("E_sn_raw", "o", "E(s, n) raw"), ("E_sn_joint", "o", "E(s, n) joint ⊥"),
                             ("E_sf_raw", "s", "E(s, f) raw"), ("E_sf_joint", "s", "E(s, f) joint ⊥")):
            x = [t["rows"][m]["late"][key] for m in models]
            filled = key.endswith("raw")
            col = "#2a78d6" if "sn" in key else "#e34948"
            ax.scatter(x, y, s=46, marker=mk, facecolors=col if filled else SURF, edgecolors=col, linewidths=1.6, label=lab, zorder=3)
        ax.axvline(0, color=RULE, linewidth=0.8, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlim(-0.55, 0.8)
        ax.grid(axis="x", color=RULE, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        Dn = t["D"]["raw"]["late"]["D_n"]
        ax.set_title(f"{Z.SHORT[cs]} (s): n = {n}, f = {f}\nladder {t['ladder']}; raw D_n {Dn:+.3f} (floor {A['A4']['raw']['A4.3__' + cs]:.3f})",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("centred cosine with s (late)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(Z.FIG / "nearfar_sn.png", dpi=140)
    plt.close(fig)
    print("wrote nearfar_sn.png")


if __name__ == "__main__":
    Z.FIG.mkdir(parents=True, exist_ok=True)
    five("raw", "dE_late_raw.png", "raw")
    five("own", "dE_late_own3.png", "each student's top-3 update directions removed, ⊥own,3")
    base_fig()
    nearfar_fig()
    print("order:", labels)
