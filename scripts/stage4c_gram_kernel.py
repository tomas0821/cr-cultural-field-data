"""Stage 4c-ter — static kinship field with a Gram kernel of the surname frequency vectors (PSD by construction):
K_BC = sum_s sqrt(p_a p_b) = 1 - E^2 (Bhattacharyya overlap; E = Stage-2 Euclidean distance), and K_cos = R_ab/sqrt(R_aa R_bb) (cosine).
Compared with the exponential kernel exp(-dL/1) and with geo-only; w with the kernel; consistent Q1/Q5 diagnostic."""
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
    except np.linalg.LinAlgError: return 1e30, None
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr), beta
def fit(mats, y, X, binom, x0):
    r = minimize(lambda th: gls_nll(th, mats, y, X, binom)[0], x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=150)); return r.fun, r.x
out = {}
for tag in sys.argv[1:] or ["2026"]:
    m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index[np.isfinite(res[f"tau_{tag}"])]); codes = codes_all[m]
    Lm = las["L"][np.ix_(m, m)].astype(float); R = las["R"][np.ix_(m, m)].astype(float); Eu = las["Eu"][np.ix_(m, m)].astype(float); I = las["I"][m].astype(float)
    r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes); D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
    tau = r[f"tau_{tag}"].to_numpy(float); N = r[f"N_{tag}"].to_numpy(float); p = r[f"p_{tag}"].to_numpy(float); binom = h / (N * p * (1 - p))
    X = pd.get_dummies(c_[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c_.lnN_2026; X["gam"] = c_.gam; X.insert(0, "c", 1.0); X = X.to_numpy()
    G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K); Pgeo = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, 1.0)))[0]; dP = np.diag(Pgeo).mean()
    iu = np.triu_indices(n, 1)
    KBC = np.clip(1 - Eu ** 2, 0, 1); np.fill_diagonal(KBC, 1.0)
    Rd = np.diag(R).copy(); Kcos = R / np.sqrt(np.outer(Rd, Rd)); np.fill_diagonal(Kcos, 1.0)
    Kexp = np.exp(-(Lm - Lm[iu].min()) / 1.0); np.fill_diagonal(Kexp, 1.0)
    for Kn, Km in (("BC", KBC), ("cos", Kcos)):
        ev = np.linalg.eigvalsh(Km); print(f"   {Kn}: min eig {ev.min():.2e}, off-diag range {Km[iu].min():.3f}-{Km[iu].max():.3f}, median {np.median(Km[iu]):.3f}", flush=True)
    f0, th0 = fit([Pgeo], tau, X, binom, np.log([0.08 / dP, 0.05])); o = dict(n=n, nll_geo=float(f0))
    for Kn, Km in (("exp_l1", Kexp), ("BC", KBC), ("cos", Kcos)):
        f, th = fit([Pgeo, Km], tau, X, binom, np.log([0.05 / dP, 0.03, 0.01])); coef = np.exp(th)
        wp = {}
        for w in (0.0, 0.5, 1.0):
            P = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, w)))[0]; wp[w] = fit([P, Km], tau, X, binom, np.r_[th[0] + np.log(dP / np.diag(P).mean()), th[1:]])[0]
        wp = {k: v - min(wp.values()) for k, v in wp.items()}
        Sig = coef[0] * Pgeo + coef[1] * Km + np.diag(coef[2] + binom); Hc = np.eye(n) - 1 / n; Sc = Hc @ Sig @ Hc; v = np.diag(Sc); Cm = (Sc / np.sqrt(np.outer(v, v)))[iu]
        xr = tau - X @ np.linalg.lstsq(X, tau, rcond=None)[0]; xr -= xr.mean(); Ce = ((xr[:, None] * xr[None, :])[iu]) / np.mean(xr ** 2)
        d = D[iu]; l = Lm[iu]; canton = r.index.str[:3].to_numpy(); diffc = (canton[:, None] != canton[None, :])[iu]; diag = {}
        for a, bb in [(5, 15), (15, 40), (40, 100)]:
            mb = (d >= a) & (d < bb) & diffc; q = np.quantile(l[mb], [0, .2, .8, 1]); q1 = mb & (l <= q[1]); q5 = mb & (l >= q[2])
            diag[f"{a}-{bb}"] = (float(Ce[q1].mean()), float(Ce[q5].mean()), float(Cm[q1].mean()), float(Cm[q5].mean()))
        o[Kn] = dict(nll=float(f), LR=float(2 * (f0 - f)), s2phi=float(coef[0] * dP), ss=float(coef[1]), smu=float(coef[2]), share=float(coef[1] / (coef[1] + coef[0] * dP)), w_profile={str(k): float(v) for k, v in wp.items()}, diag=diag)
        print(f"{tag} kernel {Kn:6s}: NLL {f:.0f} (LR vs geo {2*(f0-f):.0f}); s2phi={coef[0]*dP:.3f} ss={coef[1]:.3f} smu={coef[2]:.3f} share={o[Kn]['share']:.2f}; w dNLL(0,0.5,1)={[round(wp[k],1) for k in (0.0,0.5,1.0)]}; "
              f"Q1/Q5 emp vs model: " + " ".join(f"{k}km {v[0]:.2f}/{v[1]:.2f} vs {v[2]:.2f}/{v[3]:.2f}" for k, v in diag.items()) + f" [{time.time()-T0:.0f} s]", flush=True)
    out[tag] = o
cf.save_json(out, cf.PROC / "stage4c_gram_kernel.json"); print("done")
