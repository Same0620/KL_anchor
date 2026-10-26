"""QS1-F1 figures (PNG; the cluster has no SVG renderer) from results/figdata.json.

  nice /work/u4161854/.conda/envs/mats/bin/python -I -B qs1/adjacent_first/src/af_figs.py
      -> fig/S1_time_course.png, fig/trajectories_PLAIN.png, fig/headroom_hist.png

Palette: the dataviz reference categorical slots 1-4 (validated light: all checks pass; slots 3-4 below 3:1, so every line is also
direct-labelled or in the legend and every number is tabled in RESULTS.md); E-rank uses the blue sequential ramp, steps 250-700.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

AF = Path("/work/u4161854/loracle_em_exp1/qs1/adjacent_first")
FD = json.loads((AF / "results" / "figdata.json").read_text())
AN = json.loads((AF / "results" / "analysis.json").read_text())
FIG = AF / "fig"
FIG.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = {"E_base_late": "#2a78d6", "E_grad": "#eb6834", "E_base_early": "#1baf7a", "E_text": "#eda100"}
DRAW = ["E_text", "E_base_early", "E_grad", "E_base_late"]          # primaries drawn last (on top)
RAMP = LinearSegmentedColormap.from_list("blue", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#86b6ef"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
NICE = {"s5": "5", "k-1": "k−1", "k": "k", "s10": "10", "s20": "20", "s40": "40", "s80": "80", "final": "final"}


# ---------------------------------------------------------------- S1 / S5: G and beta against cumulative learning rate
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for r, arm in enumerate(("PLAIN", "CONTRAST")):
    tc = FD["time_course"][arm]
    labs = sorted(tc, key=lambda l: tc[l]["cum_lr"])                 # per-student k mixes steps: draw in x order
    x = np.array([tc[l]["cum_lr"] for l in labs])
    for c, (key, ylab) in enumerate((("G", "Ĝ (near-third lead ÷ mean shift)"), ("beta", "β (nats per SD of E)"))):
        ax = axes[r, c]
        for m in DRAW:
            col = SLOT[m]
            for kind, ls, lw in (("dlogit", "-", 2.0), ("gate", "--", 1.4)):
                y = np.array([tc[l][kind][m][key] for l in labs])
                ok = y > 0
                ax.plot(x[ok], y[ok], ls=ls, lw=lw, color=col, marker="o" if kind == "dlogit" else "s", ms=4.5 if kind == "dlogit" else 3.5,
                        mec=SURF, mew=0.8, label=f"{m} · {'Δlogit' if kind == 'dlogit' else 'ΔL_gate'}")
        if key == "G":
            lo = np.array([tc[l]["dlogit"]["E_base_late"]["ci"][0] for l in labs])
            hi = np.array([tc[l]["dlogit"]["E_base_late"]["ci"][1] for l in labs])
            ax.fill_between(x, np.clip(lo, 1e-7, None), hi, color=SLOT["E_base_late"], alpha=0.12, lw=0)
            ax.axhline(0.10, color=INK2, lw=1.0, ls=":")
            ax.text(x.min(), 0.115, "bar 0.10", va="bottom", ha="left", color=INK2, fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("cumulative learning rate (peak-LR units)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{arm}: {'relative' if key == 'G' else 'absolute'} lead of the near third", loc="left", fontsize=10, color=INK)
        yb = np.array([tc[l]["dlogit"]["E_base_late"][key] for l in labs])
        for xi, yi, l in zip(x, yb, labs):
            if l in ("s5", "k", "final", "s80"):
                ax.annotate(NICE[l], (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK2)
        if key == "G":
            ax.set_ylim(1e-6 if arm == "PLAIN" else 1e-3, 0.6)
hs, ls_ = axes[0, 0].get_legend_handles_labels()
order = [ls_.index(f"{m} · {k}") for m in ("E_base_late", "E_grad", "E_base_early", "E_text") for k in ("Δlogit", "ΔL_gate")]
fig.legend([hs[i] for i in order], [ls_[i] for i in order], loc="outside lower center", ncol=4, fontsize=8)
fig.suptitle("Ĝ and β against cumulative learning rate (solid = Δlogit, the primary outcome; dashed = ΔL_gate; band = 95% CI, E_base_late Δlogit)",
             fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "S1_time_course.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 5.1 trajectories
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True, sharex="col")
for c, src in enumerate(("medical", "finance")):
    tr = FD["trajectories"][src]
    names = sorted(tr["targets"], key=lambda t: tr["targets"][t]["rank"])
    steps = [p[0] for p in tr["targets"][names[0]]["points"]]
    Y = np.array([[p[1] for p in tr["targets"][t]["points"]] for t in names])
    mean = Y.mean(0)
    ax, ax2 = axes[0, c], axes[1, c]
    for i, t in enumerate(names):
        col = RAMP(i / (len(names) - 1))
        ax.plot(steps, Y[i], color=col, lw=1.2, marker="o", ms=2.5)
        ax2.plot(steps, Y[i] - mean, color=col, lw=1.2, marker="o", ms=3)
    for i in (0, 1, len(names) - 2, len(names) - 1):
        ax2.annotate(f"{names[i]} (#{tr['targets'][names[i]]['rank']})", (steps[-1], (Y[i] - mean)[-1]), textcoords="offset points",
                     xytext=(4, 0), va="center", fontsize=7.5, color=INK2)
    cv = np.array(tr["intopic_every_step"])
    ax.plot(cv[:, 0], cv[:, 1], color=INK, lw=2.0, label=f"in-topic ({src}), every step")
    ev = np.array(tr["everyday"])
    ax.plot(ev[:, 0], ev[:, 1], color=INK2, lw=1.4, ls="--", label="everyday48")
    ax.axvline(5, color=INK2, lw=0.8, ls=":")
    ax.axvline(tr["k"], color=INK2, lw=0.8, ls=":")
    ax.text(5, ax.get_ylim()[1], " step 5", va="top", fontsize=8, color=INK2)
    ax.text(tr["k"], ax.get_ylim()[1] * 0.92, f" k = {tr['k']}", va="top", fontsize=8, color=INK2)
    ax.set_title(f"PLAIN({src}): Δlogit per target (dark = closest by E_base_late)", loc="left", fontsize=10)
    ax.set_ylabel("Δlogit of M₁ (nats)")
    ax.legend(fontsize=8, loc="lower right")
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.axvline(5, color=INK2, lw=0.8, ls=":")
    ax2.axvline(tr["k"], color=INK2, lw=0.8, ls=":")
    ax2.set_title(f"PLAIN({src}): the same, minus the mean over its 22 targets", loc="left", fontsize=10)
    ax2.set_ylabel("Δlogit − target mean (nats)")
    ax2.set_xlabel("optimiser step")
    ax2.set_xlim(-1, 68)
sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(1, 22))
cb = fig.colorbar(sm, ax=axes[:, 1], shrink=0.6, pad=0.01)
cb.set_label("E_base_late rank within the source (1 = closest)")
cb.set_ticks([1, 5, 10, 15, 20, 22])
cb.ax.invert_yaxis()
fig.savefig(FIG / "trajectories_PLAIN.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 5.2 headroom histograms
hm = AN["h_min"]
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True, sharex=True)
bins = np.linspace(-1.8, 1.0, 57)
for ax, lab, col, name in ((axes[0], "s5", "#2a78d6", "step 5"), (axes[1], "k", "#eb6834", "each student's k")):
    h = np.array(FD["headroom_hist"][lab])
    ax.hist(np.log10(h), bins=bins, color=col, edgecolor=SURF, linewidth=0.5)
    for v, t in ((np.log10(0.105), "h 0.105 (p 0.9)"), (np.log10(0.693), "h 0.693 (p 0.5)")):
        ax.axvline(v, color=INK2, lw=0.9, ls=":")
        ax.text(v, ax.get_ylim()[1] * 0.95, " " + t, fontsize=7.5, color=INK2, va="top")
    ax.set_title(f"PLAIN targets at {name}: n {len(h)}, median h {np.median(h):.3g} nats, min {h.min():.3g} (none below h_min {hm:g})", loc="left", fontsize=10)
    ax.set_ylabel("prompts")
axes[1].set_xlabel("log10 headroom h = −log p(M₁ | x)")
fig.savefig(FIG / "headroom_hist.png", dpi=150)
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
