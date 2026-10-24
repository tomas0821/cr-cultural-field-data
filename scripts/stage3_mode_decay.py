"""Stage 3c — cross-election decay of the field mode by mode, and spectral densities (descriptive).

For each graph (geographic w = 1, surname w = 0, and the fitted mixture) build the adopted Laplacian at the
best (l_c, k) of the joint profile, project tau'_t of every election on its eigenmodes, group modes into
lambda bands, and measure the band correlation rho_b(dT) across all election pairs (dT = 0.15 ... 16 y).
Fit rho_b(dT) = f_b + (1 - f_b) exp(-gamma_b dT) (static floor, Becharat 2025), then gamma_b = D lambda_b + kappa.
The graph whose low-lambda modes are the persistent ones (monotone gamma(lambda), high R2) carries the field.
Outputs: data/processed/stage3_mode_decay.json, figures/stage3_mode_decay.png, stage3_spectral_density.png
"""
import sys, json, itertools, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--lc", type=float); ap.add_argument("--k", type=int); ap.add_argument("--w", type=float); ap.add_argument("--nbands", type=int, default=10)
args = ap.parse_args()
try:
    ps = json.load(open(cf.PROC / "stage3_profile_summary.json")); b = ps["per_set"]["joint5"]["best"]
    lc, k, w_hat = args.lc or b["lc"], args.k or b["k"], args.w if args.w is not None else ps["per_set"]["joint5"]["w_hat"]
except FileNotFoundError:
    lc, k, w_hat = args.lc or 4.5, args.k or 8, args.w if args.w is not None else 1.0
YEARS = {"2010": 2010.1, "2014": 2014.1, "2018": 2018.1, "2022r1": 2022.1, "2022r2": 2022.25, "2026": 2026.1}
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True)
res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
codes_all = las["codelec"].astype(str); m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index)
for t in cf.TAGS: m &= np.isin(codes_all, res.index[np.isfinite(res[f"tau_{t}"])])
codes = codes_all[m]; Lm = las["L"][np.ix_(m, m)].astype(float); r = res.loc[codes]; n = len(codes)
D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
T = np.array([r[f"tau_{t}"].to_numpy(float) for t in cf.TAGS]); T -= T.mean(1, keepdims=True)
print(f"common set n={n}; lc={lc} k={k} w_hat={w_hat}")
G = cf.geo_kernel(D, lc); A = cf.knn_graph(Lm, k)
graphs = {"geographic (w=1)": 1.0, "surname kNN (w=0)": 0.0}
if 0 < w_hat < 1: graphs[f"fitted mixture (w={w_hat})"] = w_hat
out = {"lc": lc, "k": k, "w_hat": w_hat, "n": n, "graphs": {}}
pairs = [(i, j) for i, j in itertools.combinations(range(len(cf.TAGS)), 2)]
dT = np.array([abs(YEARS[cf.TAGS[j]] - YEARS[cf.TAGS[i]]) for i, j in pairs])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); fig2, ax2 = plt.subplots(figsize=(6, 4))
for gi, (name, w) in enumerate(graphs.items()):
    L = cf.laplacian(cf.mix_kernels(G, A, w), "comb_rw"); lam, U = np.linalg.eigh(L)
    keep = lam > 1e-9 * lam.max(); lam, U = lam[keep], U[:, keep]
    a = T @ U                                   # (6, n_modes) mode coefficients
    edges = np.quantile(np.log(lam), np.linspace(0, 1, args.nbands + 1)); bid = np.clip(np.digitize(np.log(lam), edges) - 1, 0, args.nbands - 1)
    bands = []
    for bb in range(args.nbands):
        sel = bid == bb; lam_b = float(np.exp(np.log(lam[sel]).mean()))
        rho = np.array([np.sum(a[i, sel] * a[j, sel]) / np.sqrt(np.sum(a[i, sel] ** 2) * np.sum(a[j, sel] ** 2)) for i, j in pairs])
        try:
            (f, g), _ = curve_fit(lambda t, f, g: f + (1 - f) * np.exp(-g * t), dT, rho, p0=[0.5, 0.1], bounds=([0, 0], [1, 3]))
        except Exception: f, g = np.nan, np.nan
        var_share = float(np.sum(a[:, sel] ** 2) / np.sum(a ** 2))
        bands.append(dict(band=bb, lambda_mean=lam_b, n_modes=int(sel.sum()), var_share=var_share, floor=float(f), gamma=float(g), rho_dT=[float(x) for x in rho]))
    bt = pd.DataFrame(bands); ok = np.isfinite(bt.gamma) & (bt.gamma < 2.99) & (bt.floor > 0.01) & (bt.floor < 0.99)
    slope, icpt = np.polyfit(bt.lambda_mean[ok], bt.gamma[ok], 1); pred = slope * bt.lambda_mean[ok] + icpt
    R2 = 1 - np.sum((bt.gamma[ok] - pred) ** 2) / np.sum((bt.gamma[ok] - bt.gamma[ok].mean()) ** 2)
    sp = spearmanr(bt.lambda_mean[ok], bt.gamma[ok]).correlation
    # persistence of the lowest band vs highest band at dT = 4 y
    out["graphs"][name] = dict(w=w, n_modes=int(len(lam)), lambda_min=float(lam.min()), lambda_max=float(lam.max()), D_per_year=float(slope), kappa_per_year=float(icpt), R2_gamma_vs_lambda=float(R2), n_bands_used=int(ok.sum()),
                               spearman_gamma_lambda=float(sp), bands=bands, rho_lowest_band_dT4=float(np.mean(np.array(bands[0]['rho_dT'])[np.isclose(dT, 4.0)])), rho_highest_band_dT4=float(np.mean(np.array(bands[-1]['rho_dT'])[np.isclose(dT, 4.0)])), var_share_lowest_quarter=float(bt.var_share[: args.nbands // 4].sum()))
    print(f"{name:24s}: modes={len(lam)} lambda [{lam.min():.2e}, {lam.max():.2f}]; gamma = D lambda + kappa: D={slope:.3f}/y kappa={icpt:.3f}/y R2={R2:.2f} Spearman={sp:.2f}; "
          f"lowest-quarter variance share={bt.var_share[:args.nbands//4].sum():.2f}; floor by band: " + " ".join(f"{x:.2f}" for x in bt.floor) + "; gamma: " + " ".join(f"{x:.2f}" for x in bt.gamma))
    for bb in (0, args.nbands // 2, args.nbands - 1):
        o = np.argsort(dT); ax[gi].plot(dT[o], np.array(bands[bb]["rho_dT"])[o], "o-", ms=3, label=f"band {bb} (λ≈{bands[bb]['lambda_mean']:.2g})")
    ax[gi].set_title(name); ax[gi].set_xlabel("ΔT (years)"); ax[gi].set_ylabel("mode-band correlation"); ax[gi].set_ylim(-0.2, 1.05); ax[gi].legend(fontsize=7)
    ax2.hist(np.log10(lam), bins=40, histtype="step", label=name, density=True)
ax2.set_xlabel("log10 λ"); ax2.set_ylabel("ρ(λ)"); ax2.legend(fontsize=8); fig2.tight_layout(); fig2.savefig(cf.FIG / "stage3_spectral_density.png", dpi=150)
if len(graphs) < 3: ax[2].axis("off")
fig.tight_layout(); fig.savefig(cf.FIG / "stage3_mode_decay.png", dpi=150)
fig3, ax3 = plt.subplots(figsize=(5.5, 4))
for name, g in out["graphs"].items():
    bt = pd.DataFrame(g["bands"]); ax3.plot(bt.lambda_mean, bt.gamma, "o-", ms=4, label=f"{name} (R²={g['R2_gamma_vs_lambda']:.2f})")
ax3.set_xscale("log"); ax3.set_xlabel("λ (band mean)"); ax3.set_ylabel("γ (1/year)"); ax3.legend(fontsize=7); fig3.tight_layout(); fig3.savefig(cf.FIG / "stage3_gamma_vs_lambda.png", dpi=150)
cf.save_json(out, cf.PROC / "stage3_mode_decay.json"); print("done")
