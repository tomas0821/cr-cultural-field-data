"""Stage 4c — static kinship field as an additive covariance term (not a Laplacian).

Model: tau' ~ N(X beta, s2 L_geo^+ + ss K_L + smu I + diag(h/[N p (1-p)])), with K_L = exp(-(L - L_min)/l_L) a
similarity kernel in Lasker space (unit diagonal), the static field xi of Becharat et al. 2025 given a kinship covariance.
Profile over l_L in {0.05, 0.1, 0.2, 0.3, 0.5, 1}; LR for ss > 0; share of the non-noise variance carried by the kinship term;
then the w-profile of the diffusive part with the kinship term included (does w stay at 1?).
Diagnostic: model vs empirical C by Lasker quintile at fixed r (different-canton pairs).
Outputs: data/processed/stage4c_static_kernel.json
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
T0 = time.time(); LC, K = 1.5, 13; WG = np.round(np.linspace(0, 1, 11), 2); LLS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
s1 = json.load(open(cf.PROC / "stage1_summary.json")); h = float(np.clip(s1["variance_decomposition"]["2022r1-2026"]["h"], 1, 3))
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True); res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
cov = pd.read_csv(cf.PROC / "de_covariates.csv", dtype={"codelec": str}).set_index("codelec"); codes_all = las["codelec"].astype(str)
out = {"lc": LC, "k": K, "h": h, "l_L_grid": LLS, "sets": {}}

def gls_nll(theta, mats, y, X, binom):
    coef = np.exp(theta); S = sum(c * M for c, M in zip(coef[:-1], mats)) + np.diag(coef[-1] + binom)
    try: Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e30, None
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr), beta
def fit(mats, y, X, binom, x0):
    r = minimize(lambda th: gls_nll(th, mats, y, X, binom)[0], x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=120))
    return r.fun, r.x, gls_nll(r.x, mats, y, X, binom)[1]

for tag in ("2026", "2022r1", "2022r2", "2018", "2014", "2010"):
    m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index[np.isfinite(res[f"tau_{tag}"])])
    codes = codes_all[m]; Lm = las["L"][np.ix_(m, m)].astype(float); r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes)
    D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float)); tau = r[f"tau_{tag}"].to_numpy(float)
    N = r[f"N_{tag}"].to_numpy(float); p = r[f"p_{tag}"].to_numpy(float); binom = h / (N * p * (1 - p))
    X = pd.get_dummies(c_[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c_.lnN_2026; X["gam"] = c_.gam; X.insert(0, "c", 1.0); X = X.to_numpy()
    G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K); Pgeo = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, 1.0)))[0]; dP = np.diag(Pgeo).mean()
    iu = np.triu_indices(n, 1); Lmin = Lm[iu].min()
    f0, th0, b0 = fit([Pgeo], tau, X, binom, np.log([0.08 / dP, 0.05]))
    best = None; prof = {}
    for lL in LLS:
        KL = np.exp(-(Lm - Lmin) / lL); np.fill_diagonal(KL, 1.0)
        f, th, b = fit([Pgeo, KL], tau, X, binom, np.log([0.08 / dP, 0.02, 0.03]))
        prof[str(lL)] = dict(nll=float(f), s2phi=float(np.exp(th[0]) * dP), ss=float(np.exp(th[1])), smu=float(np.exp(th[2])))
        if best is None or f < best[0]: best = (f, lL, th, b, KL)
    f_b, lL_b, th_b, b_b, KL_b = best; coef = np.exp(th_b); s2phi = coef[0] * dP; ss = coef[1]; smu = coef[2]
    # w-profile of the diffusive part with the kinship term included (fixed l_L)
    wprof = []
    for w in WG:
        P = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, w)))[0]
        f, _, _ = fit([P, KL_b], tau, X, binom, np.r_[th_b[0] + np.log(dP / np.diag(P).mean()), th_b[1:]]); wprof.append(f)
    wprof = np.array(wprof) - min(wprof)
    # diagnostic: model vs empirical C(L-quintile | r) on different-canton pairs; model = full covariance, centred & normalised
    Sig = coef[0] * Pgeo + ss * KL_b + np.diag(smu + binom); Hc = np.eye(n) - 1 / n; Sc = Hc @ Sig @ Hc; v = np.diag(Sc); Cm = (Sc / np.sqrt(np.outer(v, v)))[iu]
    xr = tau - X @ b_b; Ce = ((xr[:, None] * xr[None, :])[iu]) / np.mean(xr ** 2)
    d = D[iu]; l = Lm[iu]; canton = r.index.str[:3].to_numpy(); diffc = (canton[:, None] != canton[None, :])[iu]; diag = {}
    for a, bb in [(5, 15), (15, 40), (40, 100)]:
        mb = (d >= a) & (d < bb) & diffc; q = np.quantile(l[mb], [0, .2, .8, 1]); q1 = mb & (l <= q[1]); q5 = mb & (l >= q[2])
        diag[f"{a}-{bb}km"] = dict(empirical_Q1=float(Ce[q1].mean()), empirical_Q5=float(Ce[q5].mean()), model_Q1=float(Cm[q1].mean()), model_Q5=float(Cm[q5].mean()))
    out["sets"][tag] = dict(n=n, nll_geo_only=float(f0), nll_geo_plus_kinship=float(f_b), LR_kinship_term=float(2 * (f0 - f_b)), l_L_best=lL_b, s2phi=float(s2phi), ss=float(ss), smu=float(smu),
                            s2phi_geo_only=float(np.exp(th0[0]) * dP), smu_geo_only=float(np.exp(th0[1])), kinship_share_of_field=float(ss / (ss + s2phi)),
                            l_L_profile=prof, w_profile_with_kinship=[float(x) for x in wprof], w_hat_with_kinship=float(WG[wprof.argmin()]), diagnostic=diag)
    print(f"{tag}: n={n} geo-only NLL {f0:.0f} -> +kinship {f_b:.0f} (LR {2*(f0-f_b):.0f}); best l_L={lL_b}; s2phi {np.exp(th0[0])*dP:.3f}->{s2phi:.3f}, ss={ss:.3f} (kinship share {ss/(ss+s2phi):.2f}), smu {np.exp(th0[1]):.3f}->{smu:.3f}; "
          f"w_hat with kinship term={WG[wprof.argmin()]} dNLL(0.5)={wprof[5]:.0f}; Q1/Q5 15-40 km emp {diag['15-40km']['empirical_Q1']:.2f}/{diag['15-40km']['empirical_Q5']:.2f} model {diag['15-40km']['model_Q1']:.2f}/{diag['15-40km']['model_Q5']:.2f} [{time.time()-T0:.0f} s]", flush=True)
cf.save_json(out, cf.PROC / "stage4c_static_kernel.json"); print("done")
