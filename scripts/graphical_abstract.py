"""Graphical abstract for EPJ B (spec: .png, max width 480 px, aspect 11:6, no specific results, colour).

Left: the measured kinship landscape (first Lasker PCoA axis) on the map of electoral districts.
Right: schematic of the three hypotheses for the cross-election covariance of the turnout field.
Output: figures/graphical_abstract.png (480 x 262 px) and figures/graphical_abstract_2x.png.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "pdf.fonttype": 42})
pc = pd.read_csv(cf.PROC / "isonymy_2026_pcoa.csv", dtype={"codelec": str})
rng = np.random.default_rng(3)

W, H = 4.80, 2.618  # inches; dpi 100 -> 480 x 262 px
fig = plt.figure(figsize=(W, H), facecolor="white")
fig.text(0.02, 0.955, "Where does the cultural field live?", fontsize=10.5, fontweight="bold", va="top")
fig.text(0.02, 0.84, "Turnout correlations in Costa Rica, 2010–2026", fontsize=7.5, va="top", color="#444444")

# ---- left: kinship landscape on the map ------------------------------------------------------
axm = fig.add_axes([0.0, 0.14, 0.47, 0.66])
sc = axm.scatter(pc.E_crtm05 / 1e3, pc.N_crtm05 / 1e3, c=pc.PCo1, s=3.2, cmap="RdBu_r", vmin=-2.2, vmax=2.2, linewidths=0)
axm.set_aspect("equal"); axm.axis("off")
fig.text(0.235, 0.115, "kinship landscape: surname distance\nbetween 2,000 electoral districts (3.7 M electors)",
         ha="center", va="top", fontsize=6.3, color="#333333")
cax = fig.add_axes([0.04, 0.24, 0.11, 0.022])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[])
cb.outline.set_linewidth(0.4); cax.set_title("first Lasker axis", fontsize=5.8, pad=1.5)

# ---- right: three hypotheses ----------------------------------------------------------------
axs = fig.add_axes([0.50, 0.09, 0.49, 0.72]); axs.axis("off"); axs.set_xlim(0, 1); axs.set_ylim(0, 1)
x = np.linspace(0, 1, 300)
base = 0.6 * np.sin(2 * np.pi * 2.3 * x + 0.4) + 0.4 * np.sin(2 * np.pi * 5.1 * x + 1.7) + 0.25 * np.sin(2 * np.pi * 9 * x)
years = ["2010", "2014", "2018", "2022", "2026"]
cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(years)))
rows = [("diffusing field on geography", "relaxes between elections", False),
        ("frozen kinship landscape", "identical in every election", False),
        ("slowly drifting kinship landscape", "same shape, slow drift", True)]
y0 = [0.72, 0.42, 0.12]; amp = 0.05; xe = 0.86
for (title, sub, ok), yc in zip(rows, y0):
    axs.text(0.0, yc + 0.165, title, fontsize=7.2, fontweight="bold", va="center",
             color="#111111" if ok else "#555555")
    axs.text(0.0, yc + 0.115, sub, fontsize=6.0, va="center", color="#555555", style="italic")
    for i, c in enumerate(cols):
        if title.startswith("diffusing"):
            curve = base * np.exp(-1.1 * i) + 0.5 * rng.standard_normal(len(x)).cumsum() / 40 * (i > 0)
            curve -= curve.mean()
        elif title.startswith("frozen"):
            curve = base
        else:
            curve = base + 0.12 * i * np.sin(2 * np.pi * 1.4 * x + 0.3 * i)
        axs.plot(x * xe, yc + amp * curve, color=c, lw=0.9 if ok else 0.8, alpha=0.95 if ok else 0.75)
    axs.text(0.995, yc, "\u2713" if ok else "\u2717", fontsize=12 if ok else 10, ha="right", va="center",
             color="#1a9850" if ok else "#b2182b", fontweight="bold")
axs.text(xe / 2, -0.045, "position along the kinship landscape", fontsize=5.8, ha="center", va="top", color="#555555")
axs.text(0.0, 0.975, "election:", fontsize=5.8, va="center", color="#555555")
for i, (y_, c) in enumerate(zip(years, cols)):
    axs.text(0.19 + i * 0.125, 0.975, y_, fontsize=6.0, color=c, va="center", ha="left", fontweight="bold")

out = cf.FIG / "graphical_abstract.png"
fig.savefig(out, dpi=100, facecolor="white")
fig.savefig(cf.FIG / "graphical_abstract_2x.png", dpi=200, facecolor="white")
plt.close(fig)
print("wrote", out)
