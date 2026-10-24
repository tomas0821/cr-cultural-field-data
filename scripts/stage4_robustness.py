"""Stage 4a — robustness panel for the Stage 3 result (geographic propagation + static kinship field).

At the Stage 3 optimum (comb_rw, l_c = 1.5 km, k = 13) unless stated:
 A. Static-field test: tau' = X beta + phi + noise with Sigma = s2 L_geo^+ + smu I + binom, beta by GLS inside the profile;
    X = controls (province, ln N, density, GAM) with / without the Lasker PCoA axes (q = 10). Likelihood-ratio for the axes,
    w-profile with the mean structure included, and the C(L | r) gradient on the GLS residuals (does it vanish?).
 B. Surname-graph alternatives: kNN k in {25, 50} and a dense kernel exp(-(L - L_min)/l_L) (l_L = 0.2, 0.5), w-profile.
 C. Winner log-odds rho' per election: w-profile (geographic vs surname graph).
 D. N-cuts: joint w-profile on N >= 300 and N >= 500.
 E. Paternal vs maternal Lasker graphs (k = 13): joint w-profile.
Outputs: data/processed/stage4_robustness.json, figures/stage4_*.png
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import solve_triangular, cho_solve
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

T0 = time.time(); LC, K = 1.5, 13; WG = np.round(np.linspace(0, 1, 11), 2); PRIMARY = ["2010", "2014", "2018", "2022r1", "2026"]
res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
cov = pd.read_csv(cf.PROC / "de_covariates.csv", dtype={"codelec": str}).set_index("codelec")
s1 = json.load(open(cf.PROC / "stage1_summary.json")); h = float(np.clip(s1["variance_decomposition"]["2022r1-2026"]["h"], 1, 3))
LAS = {s: np.load(cf.PROC / f"lasker_2026_{s}.npz", allow_pickle=True) for s in ("pooled", "paternal", "maternal")}
codes_all = LAS["pooled"]["codelec"].astype(str); Nel = LAS["pooled"]["N"] / 2.0
out = {}

def units(nmin=200, tags=cf.TAGS, obs="tau"):
    m = (Nel >= nmin) & np.isin(codes_all, res.index)
    for t in tags: m &= np.isin(codes_all, res.index[np.isfinite(res[f"{obs}_{t}"])])
    codes = codes_all[m]; r = res.loc[codes]
    return m, codes, r, cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
def edata(r, t, obs="tau"):
    N = r[f"N_{t}"].to_numpy(float); p = r[f"p_{t}"].to_numpy(float); return r[f"{obs}_{t}"].to_numpy(float), h / (N * p * (1 - p))
def w_profile(taus, binom, G, A, x0=None, variant="comb_rw"):
    pr = []; x0 = np.log([0.08, 0.05]) if x0 is None else x0; s2s = []
    for w in WG:
        P = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, w), variant))[0]; dP = np.diag(P).mean()
        f, x = cf.profile_nll(taus, P, binom, np.r_[x0[0] - np.log(dP), x0[1]]); pr.append(f); s2s.append(np.exp(x[0]) * dP)
    pr = np.array(pr); return dict(w_hat=float(WG[pr.argmin()]), profile=[float(x) for x in pr - pr.min()], sigma_phi2_at_best=float(s2s[int(pr.argmin())]), nll_min=float(pr.min()))

# ---------------------------------------------------------------- A. static-field test (GLS mean structure inside the GFF)
def gls_nll(theta, P, y, X, binom):
    s2, smu = np.exp(theta); S = s2 * P + np.diag(smu + binom)
    try: Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e30, None
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; r = yi - Xi @ beta
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + r @ r), beta
def gls_profile(P, y, X, binom, x0):
    f = lambda th: gls_nll(th, P, y, X, binom)[0]
    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=80, initial_simplex=np.array([x0, x0 + [0.3, 0], x0 + [0, 0.3]])))
    return r.fun, r.x, gls_nll(r.x, P, y, X, binom)[1]
def design(c):
    X = pd.get_dummies(c[["provincia", "density_class"]], drop_first=True).astype(float); X["lnN"] = c.lnN_2026; X["gam"] = c.gam; X.insert(0, "const", 1.0); return X.to_numpy()
def lasker_axes(Lm, q=10):
    n = len(Lm); H = np.eye(n) - 1 / n; B = H @ (-0.5 * Lm ** 2) @ H; ev, U = np.linalg.eigh(B); o = np.argsort(ev)[::-1][:q]; return U[:, o] * np.sqrt(np.clip(ev[o], 0, None))
def strat_gradient(x, D, Lm):
    n = len(x); iu = np.triu_indices(n, 1); d = D[iu]; l = Lm[iu]; prod = (x[:, None] * x[None, :])[iu] / np.mean(x ** 2)
    b = np.linalg.lstsq(np.c_[np.ones(len(l)), l, np.log(d + 0.5)], prod, rcond=None)[0]
    q1 = {}
    for a, bb in [(0, 5), (5, 15), (15, 40)]:
        mb = (d >= a) & (d < bb); qq = np.quantile(l[mb], [0, .2, .8, 1]); q1[f"{a}-{bb}km"] = [float(prod[mb & (l <= qq[1])].mean()), float(prod[mb & (l >= qq[2])].mean())]
    return float(b[1]), q1
A_res = {}
for t in ("2022r1", "2026", "2018"):
    m, codes, r, D = units(200, [t]); c_ = cov.loc[codes]; Lm = LAS["pooled"]["L"][np.ix_(m, m)].astype(float); n = len(codes)
    tau, binom = edata(r, t); X0 = design(c_); XL = np.c_[X0, lasker_axes(Lm)]
    G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K); P1 = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, 1.0)))[0]; dP = np.diag(P1).mean()
    x0 = np.log([0.08 / dP, 0.05])
    f0, th0, b0 = gls_profile(P1, tau, X0, binom, x0); fL, thL, bL = gls_profile(P1, tau, XL, binom, x0)
    fN, thN, _ = gls_profile(P1, tau, np.ones((n, 1)), binom, x0)
    res0 = tau - X0 @ b0; resL = tau - XL @ bL
    g_raw = strat_gradient(tau, D, Lm); g0 = strat_gradient(res0, D, Lm); gL = strat_gradient(resL, D, Lm)
    # w-profile with the Lasker axes in the mean: does w stay at 1?
    prw = []
    for w in (0.0, 0.5, 0.8, 1.0):
        P = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, w)))[0]; prw.append(gls_profile(P, tau, XL, binom, np.r_[thL[0] + np.log(dP / np.diag(P).mean()), thL[1]])[0])
    prw = np.array(prw) - min(prw)
    A_res[t] = dict(n=n, nll_intercept_only=fN, nll_controls=f0, nll_controls_lasker=fL, LR_lasker_given_controls=2 * (f0 - fL), df_lasker=10,
                    sigma_phi2_controls=float(np.exp(th0[0]) * dP), sigma_phi2_controls_lasker=float(np.exp(thL[0]) * dP), smu_controls=float(np.exp(th0[1])), smu_controls_lasker=float(np.exp(thL[1])),
                    dC_dL_given_lnr=dict(raw=g_raw[0], gls_resid_controls=g0[0], gls_resid_controls_lasker=gL[0]), C_Q1_vs_Q5=dict(raw=g_raw[1], gls_resid_controls=g0[1], gls_resid_controls_lasker=gL[1]),
                    w_profile_with_lasker_mean=dict(zip(["0", "0.5", "0.8", "1"], [float(x) for x in prw])))
    print(f"A {t}: n={n}; NLL intercept {fN:.1f} / controls {f0:.1f} / controls+Lasker {fL:.1f} -> LR(Lasker | controls) = {2*(f0-fL):.1f} on 10 df; "
          f"sigma_phi2 {np.exp(th0[0])*dP:.3f} -> {np.exp(thL[0])*dP:.3f}, smu {np.exp(th0[1]):.3f} -> {np.exp(thL[1]):.3f}; dC/dL|ln r: raw {g_raw[0]:.3f}, GLS-resid(controls) {g0[0]:.3f}, GLS-resid(+Lasker) {gL[0]:.3f}; "
          f"15-40 km Q1/Q5: raw {g_raw[1]['15-40km']}, +Lasker {gL[1]['15-40km']}; w-profile with Lasker mean dNLL(w=0,0.5,0.8,1) = {np.round(prw,1)} [{time.time()-T0:.0f} s]", flush=True)
out["A_static_field"] = A_res

# ---------------------------------------------------------------- common set for B, D, E
m, codes, r, D = units(200); Lm = LAS["pooled"]["L"][np.ix_(m, m)].astype(float); n = len(codes)
taus = [edata(r, t)[0] for t in PRIMARY]; bj = np.mean([edata(r, t)[1] for t in PRIMARY], 0); G = cf.geo_kernel(D, LC)
# B. larger k and dense surname kernels
B_res = {}
for k in (25, 50):
    B_res[f"knn_k{k}"] = w_profile(taus, bj, G, cf.knn_graph(Lm, k)); print(f"B kNN k={k}: w_hat={B_res[f'knn_k{k}']['w_hat']} dNLL(0.5)={B_res[f'knn_k{k}']['profile'][5]:.0f} dNLL(0)={B_res[f'knn_k{k}']['profile'][0]:.0f}", flush=True)
for lL in (0.2, 0.5):
    Ad = np.exp(-(Lm - Lm[np.triu_indices(n, 1)].min()) / lL); np.fill_diagonal(Ad, 0)
    B_res[f"dense_lL{lL}"] = w_profile(taus, bj, G, Ad); print(f"B dense exp(-L/{lL}): w_hat={B_res[f'dense_lL{lL}']['w_hat']} dNLL(0.5)={B_res[f'dense_lL{lL}']['profile'][5]:.0f} dNLL(0)={B_res[f'dense_lL{lL}']['profile'][0]:.0f}", flush=True)
out["B_surname_graph_alternatives"] = B_res
# D. N-cuts
D_res = {}
for nmin in (300, 500):
    m2, codes2, r2, D2 = units(nmin); Lm2 = LAS["pooled"]["L"][np.ix_(m2, m2)].astype(float)
    t2 = [edata(r2, t)[0] for t in PRIMARY]; b2 = np.mean([edata(r2, t)[1] for t in PRIMARY], 0)
    D_res[f"N>={nmin}"] = dict(n=int(len(codes2)), **w_profile(t2, b2, cf.geo_kernel(D2, LC), cf.knn_graph(Lm2, K)))
    print(f"D N>={nmin}: n={len(codes2)} w_hat={D_res[f'N>={nmin}']['w_hat']} dNLL(0.5)={D_res[f'N>={nmin}']['profile'][5]:.0f}", flush=True)
out["D_N_cuts"] = D_res
# E. paternal / maternal
E_res = {}
for s in ("paternal", "maternal"):
    Ls = LAS[s]["L"][np.ix_(m, m)].astype(float); E_res[s] = w_profile(taus, bj, G, cf.knn_graph(Ls, K))
    print(f"E {s}: w_hat={E_res[s]['w_hat']} dNLL(0.5)={E_res[s]['profile'][5]:.0f} dNLL(0)={E_res[s]['profile'][0]:.0f}", flush=True)
out["E_paternal_maternal"] = E_res
# C. winner log-odds rho' per election (own unit sets)
C_res = {}
for t in cf.TAGS:
    m3, codes3, r3, D3 = units(200, [t], obs="rho"); Lm3 = LAS["pooled"]["L"][np.ix_(m3, m3)].astype(float)
    rho, b3 = edata(r3, t, obs="rho"); C_res[t] = dict(n=int(len(codes3)), winner=s1[t]["winner"], **w_profile([rho], b3, cf.geo_kernel(D3, LC), cf.knn_graph(Lm3, K)))
    print(f"C rho {t} ({s1[t]['winner']}): n={len(codes3)} w_hat={C_res[t]['w_hat']} dNLL(0)={C_res[t]['profile'][0]:.0f} dNLL(0.5)={C_res[t]['profile'][5]:.0f} dNLL(0.9)={C_res[t]['profile'][9]:.0f} sigma_phi2={C_res[t]['sigma_phi2_at_best']:.3f}", flush=True)
out["C_winner_logodds"] = C_res
out["settings"] = dict(lc=LC, k=K, h=h, wall_s=time.time() - T0)
cf.save_json(out, cf.PROC / "stage4_robustness.json")
# figure: w-profiles of the alternatives
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k_, v in B_res.items(): ax[0].plot(WG, v["profile"], "o-", ms=3, label=k_)
for k_, v in D_res.items(): ax[0].plot(WG, v["profile"], "s--", ms=3, label=k_)
for k_, v in E_res.items(): ax[0].plot(WG, v["profile"], "^:", ms=3, label=k_)
ax[0].set_ylim(0, 300); ax[0].set_xlabel("w"); ax[0].set_ylabel("ΔNLL (joint 5 elections)"); ax[0].legend(fontsize=7); ax[0].set_title("surname-graph alternatives, N-cuts, paternal/maternal")
for t, v in C_res.items(): ax[1].plot(WG, v["profile"], "o-", ms=3, label=f"{t} ρ'")
ax[1].set_ylim(0, 150); ax[1].set_xlabel("w"); ax[1].set_ylabel("ΔNLL"); ax[1].legend(fontsize=7); ax[1].set_title("winner log-odds per election")
fig.tight_layout(); fig.savefig(cf.FIG / "stage4_w_profiles.png", dpi=150); print(f"done in {time.time()-T0:.0f} s")
