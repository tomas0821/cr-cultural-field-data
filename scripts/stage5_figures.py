"""Stage 5 — publication figures (EPJ B: double column 17.8 cm). PDF + PNG in figures/.

Fig. 3  C(r) for six result sets with the geographic-kernel prediction; variogram V(r).
Fig. 1  Lasker PCoA map; C by Lasker quintile at fixed r (2026, detrended); the same on the GLS residuals
        of the geographic field with the Lasker axes in the mean (static-field test).
Fig. 2  Profile likelihood over w (six sets + joint); synthetic recovery panel (adopted operator, l_c = 1.5 km);
        l_c profile; spectral densities of the two operators.
Fig. 4  Variance decomposition vs N; cross-election correlation vs dT; static-field test summary.
Inputs: data/processed/stage*_*.json/csv (Stages 1-4). Recomputes only cheap quantities (one eigendecomposition,
one GLS fit).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "sans-serif", "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})
W2 = 17.8 / 2.54; LC, K = 1.5, 13
COL = {"2010": "#8c6d31", "2014": "#e6550d", "2018": "#31a354", "2022r1": "#3182bd", "2022r2": "#9e9ac8", "2026": "#756bb1", "joint5": "k"}
LAB = {"2010": "2010", "2014": "2014", "2018": "2018", "2022r1": "2022 R1", "2022r2": "2022 R2", "2026": "2026", "joint5": "sum (5 first rounds)"}
s1 = json.load(open(cf.PROC / "stage1_summary.json")); Cr = pd.read_csv(cf.PROC / "stage1_Cr.csv")
s3 = json.load(open(cf.PROC / "stage3_profile_summary.json")); s4 = json.load(open(cf.PROC / "stage4_robustness.json"))
inf = json.load(open(cf.PROC / "stage3_inference.json")); strat = pd.read_csv(cf.PROC / "stage3_stratified_C.csv")
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True)
res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
cov = pd.read_csv(cf.PROC / "de_covariates.csv", dtype={"codelec": str}).set_index("codelec")
h = float(np.clip(s1["variance_decomposition"]["2022r1-2026"]["h"], 1, 3))
codes_all = las["codelec"].astype(str); Nel = las["N"] / 2.0
m26 = (Nel >= 200) & np.isin(codes_all, res.index[np.isfinite(res.tau_2026)])
codes = codes_all[m26]; Lm = las["L"][np.ix_(m26, m26)].astype(float); r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes)
D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
tau = r.tau_2026.to_numpy(float); N = r.N_2026.to_numpy(float); p = r.p_2026.to_numpy(float); binom = h / (N * p * (1 - p))
G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K)
Lgeo = cf.laplacian(cf.mix_kernels(G, A, 1.0)); Pgeo, lam_geo, _ = cf.eig_pinv(Lgeo)
Lsur = cf.laplacian(cf.mix_kernels(G, A, 0.0)); _, lam_sur, _ = cf.eig_pinv(Lsur)
iu = np.triu_indices(n, 1); d = D[iu]; l = Lm[iu]
def model_Cr(P):
    Hc = np.eye(n) - 1 / n; Pc = Hc @ P @ Hc; v = np.diag(Pc); Cn = Pc / np.sqrt(np.outer(v, v)); c = Cn[iu]; b = np.digitize(d, cf.R_EDGES) - 1
    return np.array([c[b == i].mean() if (b == i).sum() > 30 else np.nan for i in range(len(cf.R_EDGES) - 1)])
rmid = np.sqrt(cf.R_EDGES[:-1] * cf.R_EDGES[1:])

# ------------------------------------------------------------------ Fig. 1
fig, ax = plt.subplots(1, 2, figsize=(W2, 3.0))
for t in cf.TAGS:
    g = Cr[(Cr.tag == t) & (Cr.obs == "tau") & (Cr.set == "full")]
    ax[0].errorbar(g.r_mid, g.C, g.C_se, color=COL[t], marker="o", ms=2.2, lw=0.8, capsize=1.5, label=LAB[t])
    ax[1].errorbar(g.r_mid, g.V, g.V_se, color=COL[t], marker="o", ms=2.2, lw=0.8, capsize=1.5, label=LAB[t])
mc = model_Cr(Pgeo); ax[0].plot(rmid, mc, "k--", lw=1.2, label=r"field on geography, $\ell_c=1.5$ km")
ax[0].axhline(0, color="gray", lw=0.5); ax[0].set_xscale("log"); ax[0].set_xlabel(r"$r$ (km)"); ax[0].set_ylabel(r"$C(r)$"); ax[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), columnspacing=0.9, handletextpad=0.4)
ax[1].set_xscale("log"); ax[1].set_xlabel(r"$r$ (km)"); ax[1].set_ylabel(r"$V(r)$")
for a, lab in zip(ax, "ab"): a.text(-0.18, 1.02, f"({lab})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(); fig.savefig(cf.FIG / "fig3_Cr_Vr.pdf"); fig.savefig(cf.FIG / "fig3_Cr_Vr.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ static-field GLS on 2026 (Fig. 2c, Fig. 4c)
def design(c):
    X = pd.get_dummies(c[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c.lnN_2026; X["gam"] = c.gam; X.insert(0, "const", 1.0); return X.to_numpy()
def lasker_axes(Lm, q=10):
    Hc = np.eye(len(Lm)) - 1 / len(Lm); B = Hc @ (-0.5 * Lm ** 2) @ Hc; ev, U = np.linalg.eigh(B); o = np.argsort(ev)[::-1][:q]; return U[:, o] * np.sqrt(np.clip(ev[o], 0, None))
def gls(theta, P, y, X):
    s2, smu = np.exp(theta); S = s2 * P + np.diag(smu + binom); Lc = np.linalg.cholesky(S)
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta; return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr), beta
X0 = design(c_); XL = np.c_[X0, lasker_axes(Lm)]; dP = np.diag(Pgeo).mean(); x0 = np.log([0.08 / dP, 0.05])
def fit(X):
    rr = minimize(lambda th: gls(th, Pgeo, tau, X)[0], x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=80)); return gls(rr.x, Pgeo, tau, X)[1]
res0 = tau - X0 @ fit(X0); resL = tau - XL @ fit(XL)
def quint(x):
    prod = (x[:, None] * x[None, :])[iu] / np.mean(x ** 2); out = {}
    for a, b in [(0, 5), (5, 15), (15, 40), (40, 100), (100, 400)]:
        mb = (d >= a) & (d < b); q = np.quantile(l[mb], [0, .2, .4, .6, .8, 1]); row = []
        for qi in range(5):
            mq = mb & (l >= q[qi]) & (l <= q[qi + 1]); row.append((l[mq].mean(), prod[mq].mean(), prod[mq].std() / np.sqrt(mq.sum())))
        out[f"{a}–{b} km"] = np.array(row)
    return out
Q0, QL = quint(res0), quint(resL)

# ------------------------------------------------------------------ Fig. 2
pc = pd.read_csv(cf.PROC / "isonymy_2026_pcoa.csv", dtype={"codelec": str})
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
sc = ax[0].scatter(pc.E_crtm05 / 1e3, pc.N_crtm05 / 1e3, c=pc.PCo1, s=2.5, cmap="RdBu_r", vmin=-2.2, vmax=2.2, linewidths=0)
ax[0].set_aspect("equal"); ax[0].set_xlabel("Easting (km)"); ax[0].set_ylabel("Northing (km)"); cb = plt.colorbar(sc, ax=ax[0], fraction=0.04, pad=0.02); cb.set_label("PCo1 of Lasker distance")
rcol = plt.cm.viridis(np.linspace(0, 0.9, 5))
for a_, Q, title in ((ax[1], Q0, "controls only"), (ax[2], QL, "controls + 10 Lasker axes")):
    for (lab, arr), cc in zip(Q.items(), rcol): a_.errorbar(arr[:, 0], arr[:, 1], arr[:, 2], color=cc, marker="o", ms=2.2, lw=0.8, capsize=1.5, label=lab)
    a_.axhline(0, color="gray", lw=0.5); a_.set_xlabel(r"$L$ (quintile mean)"); a_.set_title(title, fontsize=8); a_.set_ylim(-0.12, 1.0)
ax[1].set_ylabel(r"$C$ at fixed $r$ (2026 residuals)"); ax[1].legend(frameon=False, title="r bin", title_fontsize=6.5)
for a, lab in zip(ax, "abc"): a.text(-0.2, 1.02, f"({lab})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(); fig.savefig(cf.FIG / "fig1_lasker_static.pdf"); fig.savefig(cf.FIG / "fig1_lasker_static.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Fig. 3
WG = np.round(np.linspace(0, 1, 11), 2)
grids = pd.concat([pd.read_csv(cf.PROC / f"stage3_profile_grid{sfx}.csv") for sfx in ("", "_lc_ext", "_lc_fine") if (cf.PROC / f"stage3_profile_grid{sfx}.csv").exists()], ignore_index=True)
fig, ax = plt.subplots(2, 2, figsize=(W2, 5.0))
for t in cf.TAGS + ["joint5"]:
    pr = np.array(s3["per_set"][t]["profile"]); ax[0, 0].plot(WG, pr + 1, color=COL[t], marker="o", ms=2.2, lw=1.3 if t == "joint5" else 0.8, label=LAB[t])
ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel(r"$w$ (1 = geographic, 0 = surname graph)"); ax[0, 0].set_ylabel(r"$\Delta$NLL + 1 ($\ell_c$, $k$ profiled out)"); ax[0, 0].legend(frameon=False, ncol=2)
rec = json.load(open(cf.PROC / "stage3a_summary.json")); pj = rec.get("profiles_json", {})
if pj:
    for key, pr in pj.items():
        v, w, rp = key.split("|"); cc = {"0.0": "#3182bd", "0.5": "#e6550d", "1.0": "#31a354"}[w]
        ax[0, 1].plot(WG, np.array(pr) + 1, color=cc, lw=0.8, alpha=0.8, label=f"$w_{{true}}={w}$" if rp == "0" else None)
    for w in (0.0, 0.5, 1.0): ax[0, 1].axvline(w, color="gray", ls=":", lw=0.6)
    from matplotlib.lines import Line2D
    ax[0, 1].set_yscale("log"); ax[0, 1].legend(handles=[Line2D([0], [0], color=c, lw=1.2) for c in ("#3182bd", "#e6550d", "#31a354")], labels=[r"$w_{true}=0$", r"$w_{true}=0.5$", r"$w_{true}=1$"], frameon=False)
else:
    ax[0, 1].text(0.5, 0.5, "recovery panel\n(rerun pending)", ha="center", va="center", transform=ax[0, 1].transAxes)
ax[0, 1].set_xlabel(r"$w$"); ax[0, 1].set_ylabel(r"$\Delta$NLL + 1 (synthetic, 5 elections)"); ax[0, 1].set_title(f"recovery, $\\ell_c$ = {rec.get('lc', LC)} km, n = {rec.get('n', n)}", fontsize=8)
jg = grids[grids.tag == "joint5"]; lcp = jg.groupby("lc").nll.min(); lcp = lcp - lcp.min()
ax[1, 0].plot(lcp.index, lcp.values + 1, "ko-", ms=3, label="data, sum of 5 first rounds"); ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel(r"$\ell_c$ (km)"); ax[1, 0].set_ylabel(r"$\Delta$NLL + 1 ($w$ profiled; $k$ = 13 below 3 km)")
if (cf.PROC / "stage3a_lcprofile_landscape.json").exists():   # synthetic sets generated from the landscape model (no Laplacian) and fitted with Eq. (1)
    LS = json.load(open(cf.PROC / "stage3a_lcprofile_landscape.json")); lcs_ = [float(x) for x in LS["grid"]]
    prof_ = np.array([[r_["profile"][str(l)]["dNLL"] for l in lcs_] for r_ in LS["results"]]).mean(0)
    ax[1, 0].plot(lcs_, prof_ + 1, "s--", color="#e6550d", ms=3, lw=0.8, label=f"landscape-generated synthetic (mean of {len(LS['results'])})")
ax[1, 0].legend(frameon=False, fontsize=6, loc="lower right")
ax[1, 0].axvspan(0.3, 1.1, color="gray", alpha=0.15); ax[1, 0].text(0.55, 300, "below pinv\ntolerance", fontsize=6.5, ha="center")
bins = np.linspace(-5, 3.5, 60)
ax[1, 1].hist(np.log10(lam_geo[lam_geo > 0]), bins=bins, histtype="step", density=True, color="#3182bd", label=r"geographic ($w=1$)")
ax[1, 1].hist(np.log10(lam_sur[lam_sur > 0]), bins=bins, histtype="step", density=True, color="#e6550d", label=r"surname kNN ($w=0$)")
ax[1, 1].set_xlabel(r"$\log_{10}\lambda$"); ax[1, 1].set_ylabel(r"$\rho(\lambda)$"); ax[1, 1].legend(frameon=False)
for a, lab in zip(ax.ravel(), "abcd"): a.text(-0.22, 1.04, f"({lab})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(h_pad=1.5); fig.savefig(cf.FIG / "fig2_test.pdf"); fig.savefig(cf.FIG / "fig2_test.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Fig. 4
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
tb = pd.DataFrame(s1["variance_decomposition"]["2022r1-2026"]["bins"])
ax[0].loglog(tb.N, tb.var1, "o", ms=3, color="#3182bd", label=r"var $\tau'$ (2022 R1)"); ax[0].loglog(tb.N, tb.cov12, "s", ms=3, color="#e6550d", label="cov(2022 R1, 2026)")
ax[0].loglog(tb.N, tb.var1 - tb.cov12, "^", ms=3, color="#31a354", label="var − cov"); ax[0].loglog(tb.N, tb.binom1, "-", color="k", lw=0.8, label=r"$1/[Np(1-p)]$")
ax[0].set_xlabel(r"$N$ (registered electors)"); ax[0].set_ylabel("variance"); ax[0].legend(frameon=False)
sets = ["2018", "2022r1", "2026"]; A = s4["A_static_field"]; xx = np.arange(3)
ax[1].bar(xx - 0.2, [A[t]["dC_dL_given_lnr"]["raw"] for t in sets], 0.4, color="#e6550d", label="raw residuals")
ax[1].bar(xx + 0.2, [A[t]["dC_dL_given_lnr"]["gls_resid_controls_lasker"] for t in sets], 0.4, color="#3182bd", label="GLS residuals, Lasker axes in mean")
for i, t in enumerate(sets): ax[1].text(i + 0.2, A[t]["dC_dL_given_lnr"]["gls_resid_controls_lasker"] + 0.01, f"$\\Delta$NLL {A[t]['LR_lasker_given_controls']/2:.0f}", ha="center", fontsize=6)
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(xx); ax[1].set_xticklabels([LAB[t] for t in sets]); ax[1].set_ylabel(r"$\partial C/\partial L\,|\,\ln r$"); ax[1].legend(frameon=False, loc="lower left", fontsize=6)
for a, lab in zip(ax, "ab"): a.text(-0.2, 1.02, f"({lab})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(); fig.savefig(cf.FIG / "fig4_variance_static.pdf"); fig.savefig(cf.FIG / "fig4_variance_static.png", dpi=300); plt.close(fig)
print("figures written:", [f.name for f in sorted(cf.FIG.glob("fig*.pdf"))], "| recovery panel included:", bool(pj))
