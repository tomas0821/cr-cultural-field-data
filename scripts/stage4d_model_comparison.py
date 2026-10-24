"""Stage 4d — fixed four-model comparison of the single-election covariance, all six result sets.
M0: s2 L_geo^+ ;  M1: s2 L_geo^+ + ss K_BC ;  M2: sg exp(-r/l_r) + ss K_BC (static only; l_r in {20, 50} km) ;
M3: s2 L_geo^+ + ss K_BC + sg exp(-r/l_r). Each with smu I + binomial diagonal and the controls in the mean (GLS).
Reports NLL, LR, fitted variances, and for M3 the share of the diffusive term. Output: data/processed/stage4d_model_comparison.json"""
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
SOCIAL = "--social" in sys.argv; argv_tags = [a for a in sys.argv[1:] if not a.startswith("--")]
if SOCIAL:   # round 4: district-level social covariates (IDS 2023 and its education/economy dimensions; 2011 census schooling, illiteracy, % secondary) in the mean
    soc = pd.read_csv(cf.PROC / "de_covariates_social.csv", dtype={"codelec": str}).set_index("codelec"); SOC_COLS = ["ids2023", "ids_edu", "ids_eco", "school2011", "illit2011", "secondary2011"]
    cov = cov.join(soc[SOC_COLS]); print("social covariates in the mean:", SOC_COLS, flush=True)
def gls_nll(theta, mats, y, X, binom):
    coef = np.exp(theta); S = sum(c * M for c, M in zip(coef[:-1], mats)) + np.diag(coef[-1] + binom)
    try: Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e30
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr)
def fit(mats, y, X, binom, x0):
    r = minimize(lambda th: gls_nll(th, mats, y, X, binom), x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=200)); return r.fun, np.exp(r.x)
out = {}
for tag in argv_tags or cf.TAGS:
    m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index[np.isfinite(res[f"tau_{tag}"])]); codes = codes_all[m]
    Lm = las["L"][np.ix_(m, m)].astype(float); Eu = las["Eu"][np.ix_(m, m)].astype(float)
    r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes); D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
    tau = r[f"tau_{tag}"].to_numpy(float); N = r[f"N_{tag}"].to_numpy(float); p = r[f"p_{tag}"].to_numpy(float); binom = h / (N * p * (1 - p))
    X = pd.get_dummies(c_[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c_.lnN_2026; X["gam"] = c_.gam
    if SOCIAL:
        for col in SOC_COLS: v = c_[col].to_numpy(float); X[col] = (v - v.mean()) / v.std()
    X.insert(0, "c", 1.0); X = X.to_numpy()
    G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K); Pgeo = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, 1.0)))[0]; dP = np.diag(Pgeo).mean()
    KBC = np.clip(1 - Eu ** 2, 0, 1); np.fill_diagonal(KBC, 1.0)
    o = {"n": n}
    f, c = fit([Pgeo], tau, X, binom, np.log([0.08 / dP, 0.05])); o["M0_geoLap"] = dict(nll=float(f), s2phi=float(c[0] * dP), smu=float(c[1]))
    f, c = fit([Pgeo, KBC], tau, X, binom, np.log([0.05 / dP, 0.05, 0.02])); o["M1_geoLap_BC"] = dict(nll=float(f), s2phi=float(c[0] * dP), ss_BC=float(c[1]), smu=float(c[2]))
    for lr in (20.0, 50.0):
        Kr = np.exp(-D / lr)
        f, c = fit([Kr, KBC], tau, X, binom, np.log([0.05, 0.05, 0.02])); o[f"M2_static_r{lr:.0f}_BC"] = dict(nll=float(f), sg=float(c[0]), ss_BC=float(c[1]), smu=float(c[2]))
        f, c = fit([Pgeo, KBC, Kr], tau, X, binom, np.log([0.02 / dP, 0.05, 0.02, 0.02])); o[f"M3_geoLap_BC_static_r{lr:.0f}"] = dict(nll=float(f), s2phi=float(c[0] * dP), ss_BC=float(c[1]), sg=float(c[2]), smu=float(c[3]), diffusive_share=float(c[0] * dP / (c[0] * dP + c[1] + c[2])))
    b2 = min(k for k in o if k.startswith("M2")); b3 = min(k for k in o if k.startswith("M3")); b2 = min((k for k in o if k.startswith("M2")), key=lambda k: o[k]["nll"]); b3 = min((k for k in o if k.startswith("M3")), key=lambda k: o[k]["nll"])
    print(f"{tag} n={n}: M0 {o['M0_geoLap']['nll']:.0f} | M1 (+BC) {o['M1_geoLap_BC']['nll']:.0f} (LR {2*(o['M0_geoLap']['nll']-o['M1_geoLap_BC']['nll']):.0f}) s2phi {o['M1_geoLap_BC']['s2phi']:.3f} ssBC {o['M1_geoLap_BC']['ss_BC']:.3f} | "
          f"{b2} {o[b2]['nll']:.0f} sg {o[b2]['sg']:.3f} ssBC {o[b2]['ss_BC']:.3f} | {b3} {o[b3]['nll']:.0f} s2phi {o[b3]['s2phi']:.3f} ssBC {o[b3]['ss_BC']:.3f} sg {o[b3]['sg']:.3f} diffusive share {o[b3]['diffusive_share']:.2f}; "
          f"LR M3 vs M2 {2*(o[b2]['nll']-o[b3]['nll']):.0f} [{time.time()-T0:.0f} s]", flush=True)
    out[tag] = o; cf.save_json(out, cf.PROC / ("stage4d_model_comparison_social.json" if SOCIAL else "stage4d_model_comparison.json"))
print("done")
