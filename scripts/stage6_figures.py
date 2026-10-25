"""Stage 6 figure: (a) cross-election correlation vs dT, empirical (15 pairs) vs model-implied (static S, static + fluctuating SD);
(b) profile likelihood over the diffusion constant D (w = 1) and the SD fits on the surname graph (w = 0, 0.5);
(c) variance decomposition of the joint model (static geo / static kinship / static unit / fluctuating / election noise / binomial).
Reads data/processed/stage6_joint_dynamics.json -> figures/fig5_joint_dynamics.{pdf,png}"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
J = json.load(open(cf.PROC / "stage6_joint_dynamics.json")); R = J["models"]
if "D_profile_w1" not in J: J["D_profile_w1"] = {k[4:]: v["nll"] for k, v in R.items() if k.startswith("SD_D")}
fig, ax = plt.subplots(1, 3, figsize=(17.8 / 2.54, 7.0 / 2.54)); fig.subplots_adjust(wspace=0.45, left=0.07, right=0.86, bottom=0.36, top=0.9)
# (a)
e = J["empirical_corr_vs_dT"]; x = np.array([float(k) for k in e]); y = np.array(list(e.values())); o = np.argsort(x)
s1_ = json.load(open(cf.PROC / "stage1_summary.json")); tc_ = s1_["temporal_correlation"]; tcx = np.array([float(r_["dT"]) for r_ in tc_]); tcy = np.array([float(r_["corr"]) for r_ in tc_])
ax[0].plot(tcx, tcy, "+", color="0.55", ms=5, mew=0.8, label="raw, before the controls", zorder=4)
ro = np.isin(np.round(x, 2), [0.15, 3.85, 4.15, 8.15, 12.15]); se = (1 - y ** 2) / np.sqrt(J["n"] - 3)
ax[0].errorbar(x[~ro], y[~ro], yerr=se[~ro], fmt="ko", ms=4, capsize=1.5, elinewidth=0.6, label="data, first-round pairs", zorder=5)
ax[0].errorbar(x[ro], y[ro], yerr=se[ro], fmt="ko", mfc="white", ms=4, capsize=1.5, elinewidth=0.6, label="data, pairs with the 2022 run-off", zorder=5)
for key, lab, st in (("S", "frozen landscape (S)", "C3v"), ("SD", "S + diffusive, $w=1$ (SD)", "C0s"), ("SK", "S + drifting kernels (SK)", "C2^"), ("SKD", "SK + diffusive (SKD)", "C4x"), ("SKU", "SK, drifting district term (SKU)", "C5d")):
    if key in R:
        c = R[key]["model_corr_vs_dT"]; xx = np.array([float(k) for k in c]); yy = np.array(list(c.values()))
        ax[0].plot(xx, yy, st, ms=3.5, mfc="none", label=lab)
ax[0].set_xlabel(r"$\Delta t$ (years)"); ax[0].set_ylabel(r"corr$(\tau'_t, \tau'_{t'})$"); ax[0].set_ylim(0.66, 0.96); ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=6.5, handletextpad=0.3, columnspacing=0.8, labelspacing=0.25); ax[0].set_title("(a) persistence", loc="left")
# (b)
if "D_profile_w1" in J:
    pr = J["D_profile_w1"]; Dg = np.array([float(k) for k in pr]); nl = np.array(list(pr.values())); nl0 = R["SD"]["nll"] if "SD" in R else nl.min()
    if "SD" in R: Dg = np.r_[Dg, R["SD"]["params"]["D"]]; nl = np.r_[nl, R["SD"]["nll"]]; o_ = np.argsort(Dg); Dg, nl = Dg[o_], nl[o_]
    xs = np.where(Dg > 0, Dg, 3e-3); ax[1].plot(xs, nl - nl0, "C0o-", ms=3.5, label=r"profile in $D$ ($w=1$)"); ax[1].set_xscale("log")
    ax[1].axhline(0, color="0.6", lw=0.6); ax[1].set_xlabel(r"$D$ (yr$^{-1}$; leftmost point: $D=0$)"); ax[1].set_ylabel(r"$\Delta$NLL vs best SD")
    for key, lab, col in (("SDw0", "surname graph (w = 0)", "C1"), ("SDw05", "mixture (w = 0.5)", "C4")):
        if key in R: ax[1].axhline(R[key]["nll"] - nl0, color=col, lw=1, ls="--", label=lab)
    ew = {float(k[6:]): v["nll"] for k, v in R.items() if k.startswith("SKDe_D")}
    if "SKDe" in R and ew:   # EW-consistent weights with drifting kernels: profile in D relative to its own optimum (D -> 0)
        ew[0.0] = R["SKDe"]["nll"]; De = np.array(sorted(ew)); ne = np.array([ew[d] for d in De]); ax[1].plot(np.where(De > 0, De, 3e-3), ne - ne.min(), "C2s--", ms=3.5, label=r"EW weights, SKD (vs own best)")
    ax[1].legend(frameon=False, loc="center left", fontsize=6.5); ax[1].set_title("(b) diffusion signature", loc="left")
# (c)
keys = ["static_geo", "static_kinship", "static_unit", "dyn_unit", "dyn_field", "dyn_kinship", "dyn_geo", "election_noise", "binomial"]
labs = {"static_geo": "frozen geo", "static_kinship": "frozen kinship", "static_unit": "persistent district", "dyn_unit": "drifting district", "dyn_field": "diffusive field", "dyn_kinship": "drifting kinship", "dyn_geo": "drifting geo", "election_noise": "election noise", "binomial": "binomial"}
mods = [k for k in ("S", "SD", "SK", "SKD", "SKU") if k in R]; bottom = np.zeros(len(mods))
for i, kk in enumerate(keys):
    v = np.array([R[mm]["share"].get(kk, 0.0) for mm in mods])
    if v.sum() == 0: continue
    ax[2].bar(range(len(mods)), v, bottom=bottom, label=labs[kk], color=f"C{i}", width=0.6); bottom += v
ax[2].set_xticks(range(len(mods))); ax[2].set_xticklabels(mods); ax[2].set_ylabel("share of var($\\tau'$)"); ax[2].legend(frameon=False, fontsize=6.5, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.02), handlelength=1, borderaxespad=0)
ax[2].set_ylim(0, 1.0); ax[2].set_title("(c) variance budget", loc="left")
cf.FIG.mkdir(exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(cf.FIG / f"fig5_joint_dynamics.{ext}", dpi=300)
print("saved fig5_joint_dynamics")
print("model nll dNLL_vs_S persistent_share")
for k, v in R.items(): print(f"{k:8s} {v['nll']:10.1f} {R['S']['nll']-v['nll']:8.1f} {v.get('persistent_share_of_field') or 1:.3f}", v.get("dyn_field", ""))
