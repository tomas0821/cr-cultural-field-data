"""Stage 4c-bis — two-scale static kinship kernel on 2026 (and 2022 R1): K = exp(-dL/l_1) [broad] + exp(-dL/l_2) [sharp],
each with its own variance; l_1 in {1, 2, 3, 5}, l_2 in {0.1, 0.2, 0.3, 0.5}. Reports LR vs single kernel, variances, w with both terms,
and the model vs empirical C(L-quintile | r) contrast computed consistently (both from the fitted covariance / OLS residuals)."""
import sys, json, time, itertools
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
    except np.linalg.LinAlgError: return 1e30, None
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr), beta
def fit(mats, y, X, binom, x0):
    r = minimize(lambda th: gls_nll(th, mats, y, X, binom)[0], x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=150)); return r.fun, r.x, gls_nll(r.x, mats, y, X, binom)[1]
out = {}
for tag in sys.argv[1:] or ["2026"]:
    m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index[np.isfinite(res[f"tau_{tag}"])]); codes = codes_all[m]
    Lm = las["L"][np.ix_(m, m)].astype(float); r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes)
    D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float)); tau = r[f"tau_{tag}"].to_numpy(float)
    N = r[f"N_{tag}"].to_numpy(float); p = r[f"p_{tag}"].to_numpy(float); binom = h / (N * p * (1 - p))
    X = pd.get_dummies(c_[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c_.lnN_2026; X["gam"] = c_.gam; X.insert(0, "c", 1.0); X = X.to_numpy()
    G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K); Pgeo = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, 1.0)))[0]; dP = np.diag(Pgeo).mean()
    iu = np.triu_indices(n, 1); dL = Lm - Lm[iu].min(); np.fill_diagonal(dL, 0)
    f0, th0, _ = fit([Pgeo], tau, X, binom, np.log([0.08 / dP, 0.05]))
    single = {}
    for l1 in (1, 2, 3, 5):
        K1 = np.exp(-dL / l1); f, th, _ = fit([Pgeo, K1], tau, X, binom, np.log([0.05 / dP, 0.03, 0.01])); single[l1] = (f, th)
    l1b = min(single, key=lambda k: single[k][0]); f1, th1 = single[l1b]
    print(f"{tag}: geo-only {f0:.0f}; single kernel best l_1={l1b} NLL {f1:.0f} (LR {2*(f0-f1):.0f}); per l_1: " + " ".join(f"{k}:{v[0]:.0f}" for k, v in single.items()), flush=True)
    two = {}
    for l2 in (0.1, 0.2, 0.3, 0.5):
        K1 = np.exp(-dL / l1b); K2 = np.exp(-dL / l2); f, th, b = fit([Pgeo, K1, K2], tau, X, binom, np.r_[th1[:2], np.log(0.02), th1[2]]); two[l2] = (f, th, b)
        print(f"   two-scale l_1={l1b}, l_2={l2}: NLL {f:.0f} (LR vs single {2*(f1-f):.0f}); s2phi={np.exp(th[0])*dP:.3f} s_broad={np.exp(th[1]):.3f} s_sharp={np.exp(th[2]):.3f} smu={np.exp(th[3]):.3f}", flush=True)
    l2b = min(two, key=lambda k: two[k][0]); f2, th2, b2 = two[l2b]; coef = np.exp(th2)
    # w with both static terms
    K1 = np.exp(-dL / l1b); K2 = np.exp(-dL / l2b); wp = {}
    for w in (0.0, 0.5, 0.8, 1.0):
        P = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, w)))[0]; wp[w] = fit([P, K1, K2], tau, X, binom, np.r_[th2[0] + np.log(dP / np.diag(P).mean()), th2[1:]])[0]
    wp = {k: v - min(wp.values()) for k, v in wp.items()}
    # consistent diagnostic: OLS residuals vs model covariance, both centred, normalised by their own total variance
    Sig = coef[0] * Pgeo + coef[1] * K1 + coef[2] * K2 + np.diag(coef[3] + binom); Hc = np.eye(n) - 1 / n; Sc = Hc @ Sig @ Hc; v = np.diag(Sc); Cm = (Sc / np.sqrt(np.outer(v, v)))[iu]
    xr = tau - X @ np.linalg.lstsq(X, tau, rcond=None)[0]; xr -= xr.mean(); Ce = ((xr[:, None] * xr[None, :])[iu]) / np.mean(xr ** 2)
    d = D[iu]; l = Lm[iu]; canton = r.index.str[:3].to_numpy(); diffc = (canton[:, None] != canton[None, :])[iu]; diag = {}
    for a, bb in [(5, 15), (15, 40), (40, 100)]:
        mb = (d >= a) & (d < bb) & diffc; q = np.quantile(l[mb], [0, .2, .8, 1]); q1 = mb & (l <= q[1]); q5 = mb & (l >= q[2])
        diag[f"{a}-{bb}"] = (float(Ce[q1].mean()), float(Ce[q5].mean()), float(Cm[q1].mean()), float(Cm[q5].mean()))
    print(f"   best two-scale l_1={l1b} l_2={l2b}: LR vs geo-only {2*(f0-f2):.0f}; shares of static+field: phi {coef[0]*dP:.3f}, broad {coef[1]:.3f}, sharp {coef[2]:.3f}, smu {coef[3]:.3f}; w dNLL(0,0.5,0.8,1)={[round(wp[k],1) for k in (0.0,0.5,0.8,1.0)]}; "
          f"Q1/Q5 emp vs model: " + " ".join(f"{k} km {v[0]:.2f}/{v[1]:.2f} vs {v[2]:.2f}/{v[3]:.2f}" for k, v in diag.items()) + f" [{time.time()-T0:.0f} s]", flush=True)
    out[tag] = dict(n=n, nll_geo=float(f0), single=dict(l1=l1b, nll=float(f1), LR=float(2 * (f0 - f1)), per_l1={str(k): float(v[0]) for k, v in single.items()}),
                    two=dict(l1=l1b, l2=l2b, nll=float(f2), LR_vs_single=float(2 * (f1 - f2)), LR_vs_geo=float(2 * (f0 - f2)), s2phi=float(coef[0] * dP), s_broad=float(coef[1]), s_sharp=float(coef[2]), smu=float(coef[3]),
                             per_l2={str(k): float(v[0]) for k, v in two.items()}, w_profile={str(k): float(v) for k, v in wp.items()}, diag_Q1_Q5_emp_model=diag))
cf.save_json(out, cf.PROC / "stage4c_two_scale.json"); print("done")
