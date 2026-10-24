"""Stage 4c control — is the static kinship kernel winning because it is kinship, or because it is a broad smooth kernel?
Compare, at equal parameter count, the static term built from the Bhattacharyya surname overlap K_BC with a static term built from
geographic distance K_r = exp(-r/l_r), l_r in {5, 20, 50} km, and with both together. Also K_BC with the province/GAM block kernel."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
T0 = time.time(); LC, K = 1.5, 13
s1 = json.load(open(cf.PROC / "stage1_summary.json")); h = float(np.clip(s1["variance_decomposition"]["2022r1-2026"]["h"], 1, 3))
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True); res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
cov = pd.read_csv(cf.PROC / "de_covariates.csv", dtype={"codelec": str}).set_index("codelec"); codes_all = las["codelec"].astype(str)
def gls_nll(theta, mats, y, X, binom):
    coef = np.exp(theta); S = sum(c * M for c, M in zip(coef[:-1], mats)) + np.diag(coef[-1] + binom)
    try: Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e30
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr)
def fit(mats, y, X, binom, x0):
    r = minimize(lambda th: gls_nll(th, mats, y, X, binom), x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=160)); return r.fun, np.exp(r.x)
out = {}
for tag in sys.argv[1:] or ["2026"]:
    m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index[np.isfinite(res[f"tau_{tag}"])]); codes = codes_all[m]
    Lm = las["L"][np.ix_(m, m)].astype(float); Eu = las["Eu"][np.ix_(m, m)].astype(float)
    r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes); D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
    tau = r[f"tau_{tag}"].to_numpy(float); N = r[f"N_{tag}"].to_numpy(float); p = r[f"p_{tag}"].to_numpy(float); binom = h / (N * p * (1 - p))
    X = pd.get_dummies(c_[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c_.lnN_2026; X["gam"] = c_.gam; X.insert(0, "c", 1.0); X = X.to_numpy()
    G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K); Pgeo = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, 1.0)))[0]; dP = np.diag(Pgeo).mean()
    KBC = np.clip(1 - Eu ** 2, 0, 1); np.fill_diagonal(KBC, 1.0)
    f0, _ = fit([Pgeo], tau, X, binom, np.log([0.08 / dP, 0.05])); o = {"n": n, "nll_geo_only": float(f0)}
    fB, cB = fit([Pgeo, KBC], tau, X, binom, np.log([0.05 / dP, 0.05, 0.02])); o["BC"] = dict(nll=float(fB), LR=float(2 * (f0 - fB)), s2phi=float(cB[0] * dP), ss=float(cB[1]), smu=float(cB[2]))
    print(f"{tag}: geo-only {f0:.0f}; + K_BC {fB:.0f} (LR {2*(f0-fB):.0f}) s2phi={cB[0]*dP:.3f} ss={cB[1]:.3f} smu={cB[2]:.3f}", flush=True)
    for lr in (5.0, 20.0, 50.0):
        Kr = np.exp(-D / lr); fr, cr = fit([Pgeo, Kr], tau, X, binom, np.log([0.05 / dP, 0.05, 0.02])); o[f"geo_static_{lr}"] = dict(nll=float(fr), LR=float(2 * (f0 - fr)), s2phi=float(cr[0] * dP), ss=float(cr[1]), smu=float(cr[2]))
        print(f"   + static geographic kernel exp(-r/{lr:.0f} km): NLL {fr:.0f} (LR vs geo-only {2*(f0-fr):.0f}; BC better by {2*(fr-fB):.0f}) s2phi={cr[0]*dP:.3f} ss={cr[1]:.3f} smu={cr[2]:.3f}", flush=True)
    lrb = min((5.0, 20.0, 50.0), key=lambda l: o[f"geo_static_{l}"]["nll"]); Kr = np.exp(-D / lrb)
    fb, cb = fit([Pgeo, KBC, Kr], tau, X, binom, np.log([0.03 / dP, 0.05, 0.02, 0.02])); o["BC_plus_geo_static"] = dict(l_r=lrb, nll=float(fb), LR_vs_BC=float(2 * (fB - fb)), LR_vs_geo_static=float(2 * (o[f"geo_static_{lrb}"]["nll"] - fb)), s2phi=float(cb[0] * dP), ss_BC=float(cb[1]), ss_geo=float(cb[2]), smu=float(cb[3]))
    print(f"   + both (BC and exp(-r/{lrb:.0f})): NLL {fb:.0f}; LR vs BC alone {2*(fB-fb):.0f}, vs geo-static alone {2*(o[f'geo_static_{lrb}']['nll']-fb):.0f}; s2phi={cb[0]*dP:.3f} ss_BC={cb[1]:.3f} ss_geo={cb[2]:.3f} smu={cb[3]:.3f} [{time.time()-T0:.0f} s]", flush=True)
    out[tag] = o
cf.save_json(out, cf.PROC / "stage4c_static_control.json"); print("done")
