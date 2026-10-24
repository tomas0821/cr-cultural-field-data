"""Stage 3b — descriptive C(L | r), C(r | L) and distance-matrix inference that survives review.

Units: Lasker N >= 200, tau' available for the election. Controls: province FE, ln N, density class, GAM.
1. Stratified pair correlations: C in (r-bin x L-quintile-within-r-bin) cells, raw and detrended tau', and
   within-province pairs only.
2. MSR-constrained Mantel (Crabot 2019 / Wagner & Dray 2015, Singleton): Mantel r between |tau_i - tau_j| and
   L_ij, with tau randomised on the MEM basis of a geographic W (distance threshold = longest MST edge, 1/d
   weights) so the surrogate keeps the observed Moran's I; 999 surrogates. Symmetric test: |dtau| vs r with
   MSR on the Lasker-kNN W.
3. dbMEM regression (Legendre, Fortin & Borcard 2015): tau' ~ controls + Lasker-PCoA axes (q = 10) + dbMEM(+)
   (+ dbMEM(-)); Freedman-Lane permutational F for the Lasker block and for the dbMEM block; adjusted R2.
Outputs: data/processed/stage3_inference.json, stage3_stratified_C.csv, figures/stage3_C_L_given_r.png
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(3); T0 = time.time(); NSUR = 999
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True)
res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
cov = pd.read_csv(cf.PROC / "de_covariates.csv", dtype={"codelec": str}).set_index("codelec")
codes_all = las["codelec"].astype(str); base = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index)
out = {}; strat_rows = []

def design(c):
    X = pd.get_dummies(c[["provincia", "density_class"]], drop_first=True).astype(float)
    X["lnN"] = c.lnN_2026; X["gam"] = c.gam; X.insert(0, "const", 1.0); return X.to_numpy()
def ols_resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b, 1 - np.var(y - X @ b) / np.var(y)
def mantel_r(a, b): return np.corrcoef(a, b)[0, 1]
def mem_basis(W):
    n = len(W); H = np.eye(n) - 1 / n; Ws = (W + W.T) / 2; lam, V = np.linalg.eigh(H @ Ws @ H)
    order = np.argsort(lam)[::-1]; lam = lam[order]; V = V[:, order]
    keep = np.abs(lam) > 1e-9 * np.abs(lam).max(); return lam[keep], V[:, keep]
def msr_singleton(z, V, nsur):
    """Wagner & Dray 2015 Singleton: keep |spectral coefficient| per MEM, randomise signs."""
    mu, sd = z.mean(), z.std(); zs = (z - mu) / sd; r = V.T @ zs
    S = rng.choice([-1.0, 1.0], size=(nsur, len(r)))
    return mu + sd * (V @ (S * r).T).T          # (nsur, n)
def threshold_W(D):
    mst = minimum_spanning_tree(D).toarray(); t = mst.max()
    W = np.where((D <= t) & (D > 0), 1 / np.maximum(D, 1e-3), 0.0); return W, t
def moran_I(x, W):
    z = x - x.mean(); return len(x) / W.sum() * (z @ W @ z) / (z @ z)
def freedman_lane_F(y, X_red, X_full, nperm):
    """permutational F for the extra columns of X_full over X_red (Freedman & Lane 1983)."""
    def F(y):
        r_red = y - X_red @ np.linalg.lstsq(X_red, y, rcond=None)[0]; r_full = y - X_full @ np.linalg.lstsq(X_full, y, rcond=None)[0]
        df1 = X_full.shape[1] - X_red.shape[1]; df2 = len(y) - X_full.shape[1]
        return ((r_red @ r_red - r_full @ r_full) / df1) / ((r_full @ r_full) / df2)
    b = np.linalg.lstsq(X_red, y, rcond=None)[0]; fit, resid = X_red @ b, y - X_red @ b
    F0 = F(y); Fp = np.array([F(fit + rng.permutation(resid)) for _ in range(nperm)])
    return float(F0), float((np.sum(Fp >= F0) + 1) / (nperm + 1))
def adj_r2(y, X): r, r2 = ols_resid(y, X); n, p = X.shape; return 1 - (1 - r2) * (n - 1) / (n - p)

for tag in ("2022r1", "2022r2", "2026", "2018", "2014", "2010"):
    m = base & np.isin(codes_all, res.index[np.isfinite(res[f"tau_{tag}"])])
    codes = codes_all[m]; Lm = las["L"][np.ix_(m, m)].astype(float); r_ = res.loc[codes]; c_ = cov.loc[codes]
    D = cf.pair_distance_km(r_.E_crtm05.to_numpy(float), r_.N_crtm05.to_numpy(float)); n = len(codes)
    tau = r_[f"tau_{tag}"].to_numpy(float); X = design(c_); tau_d, r2c = ols_resid(tau, X)
    iu = np.triu_indices(n, 1); d = D[iu]; l = Lm[iu]
    same_prov = (c_.provincia.to_numpy()[:, None] == c_.provincia.to_numpy()[None, :])[iu]
    o = dict(n=n, controls_R2=float(r2c))
    # ---- 1. stratified C
    redges = [0, 5, 15, 40, 100, 400]
    for label, x, pairmask in (("raw", tau, np.ones(len(d), bool)), ("detrended", tau_d, np.ones(len(d), bool)), ("raw_within_province", tau, same_prov), ("detrended_within_province", tau_d, same_prov)):
        prod = (x[:, None] * x[None, :])[iu] / np.mean(x ** 2)
        for a, b in zip(redges[:-1], redges[1:]):
            mb = (d >= a) & (d < b) & pairmask
            if mb.sum() < 500: continue
            q = np.quantile(l[mb], [0, .2, .4, .6, .8, 1.0])
            for qi in range(5):
                mq = mb & (l >= q[qi]) & (l <= q[qi + 1])
                strat_rows.append(dict(tag=tag, set=label, r_lo=a, r_hi=b, L_quintile=qi + 1, L_mean=float(l[mq].mean()), n_pairs=int(mq.sum()), C=float(prod[mq].mean()), C_se=float(prod[mq].std() / np.sqrt(mq.sum()))))
        # slope of C on L at fixed r (pooled over r bins, partial): regress pair product on L and ln r
        Xp = np.c_[np.ones(pairmask.sum()), l[pairmask], np.log(d[pairmask] + 0.5)]
        bp = np.linalg.lstsq(Xp, prod[pairmask], rcond=None)[0]
        o[f"dC_dL_given_lnr_{label}"] = float(bp[1]); o[f"dC_dlnr_given_L_{label}"] = float(bp[2])
    # ---- 2. MSR-constrained Mantel
    Wg, thr = threshold_W(D); lam_g, Vg = mem_basis(Wg)
    dtau = np.abs(tau_d[:, None] - tau_d[None, :])[iu]
    rL = mantel_r(dtau, l); rR = mantel_r(dtau, d); rLR = mantel_r(l, d)
    sur = msr_singleton(tau_d, Vg, NSUR)
    rL_s = np.array([mantel_r(np.abs(s[:, None] - s[None, :])[iu], l) for s in sur])
    I_obs = moran_I(tau_d, Wg); I_sur = np.array([moran_I(s, Wg) for s in sur[:50]])
    o["mantel"] = dict(W_threshold_km=float(thr), moran_I_obs=float(I_obs), moran_I_surrogates_mean=float(I_sur.mean()),
                       r_dtau_L=float(rL), r_dtau_r=float(rR), r_L_r=float(rLR),
                       msr_geoW=dict(p=float((np.sum(rL_s >= rL) + 1) / (NSUR + 1)), r_star=float(rL - rL_s.mean()), SES=float((rL - rL_s.mean()) / rL_s.std()), null_mean=float(rL_s.mean()), null_sd=float(rL_s.std())))
    # symmetric: dtau vs geographic distance with MSR on the Lasker-kNN W
    Wl = cf.knn_graph(Lm, 8); lam_l, Vl = mem_basis(Wl); sur_l = msr_singleton(tau_d, Vl, NSUR)
    rR_s = np.array([mantel_r(np.abs(s[:, None] - s[None, :])[iu], d) for s in sur_l])
    o["mantel"]["msr_laskerW_dtau_vs_r"] = dict(p=float((np.sum(rR_s >= rR) + 1) / (NSUR + 1)), r_star=float(rR - rR_s.mean()), SES=float((rR - rR_s.mean()) / rR_s.std()))
    # also L vs dtau with MSR on Lasker W (does L explain tau beyond its own autocorrelation structure? control)
    rL_s2 = np.array([mantel_r(np.abs(s[:, None] - s[None, :])[iu], l) for s in sur_l])
    o["mantel"]["msr_laskerW_dtau_vs_L"] = dict(p=float((np.sum(rL_s2 >= rL) + 1) / (NSUR + 1)), SES=float((rL - rL_s2.mean()) / rL_s2.std()))
    # ---- 3. dbMEM regression
    Dt = np.where(D <= thr, D, 4 * thr); A_ = -0.5 * Dt ** 2; H = np.eye(n) - 1 / n; B = H @ A_ @ H
    ev, U = np.linalg.eigh(B); order = np.argsort(ev)[::-1]; ev, U = ev[order], U[:, order]
    Wb = (D <= thr) & (D > 0)
    pos = ev > 1e-9 * ev.max(); neg = ev < -1e-9 * ev.max()
    Ipos = np.array([moran_I(U[:, i], Wb) for i in np.flatnonzero(pos)[:200]])
    mem_pos = U[:, np.flatnonzero(pos)[:200]][:, Ipos > 0][:, :60]
    mem_neg = U[:, np.flatnonzero(neg)[-60:]]
    # Lasker PCoA axes on this unit set
    Al = -0.5 * Lm ** 2; Bl = H @ Al @ H; evl, Ul = np.linalg.eigh(Bl); ol = np.argsort(evl)[::-1]; Lax = Ul[:, ol[:10]] * np.sqrt(np.clip(evl[ol[:10]], 0, None))
    X0 = X; XL = np.c_[X0, Lax]; XM = np.c_[X0, mem_pos]; XLM = np.c_[X0, Lax, mem_pos]; XLMn = np.c_[XLM, mem_neg]
    F_L_given_M, p_L_given_M = freedman_lane_F(tau, XM, XLM, 999)
    F_M_given_L, p_M_given_L = freedman_lane_F(tau, XL, XLM, 999)
    F_L_given_Mn, p_L_given_Mn = freedman_lane_F(tau, np.c_[XM, mem_neg], XLMn, 999)
    F_L_alone, p_L_alone = freedman_lane_F(tau, X0, XL, 999)
    resid_full = ols_resid(tau, XLMn)[0]
    o["dbmem"] = dict(n_mem_pos=int(mem_pos.shape[1]), n_mem_neg=int(mem_neg.shape[1]), n_lasker_axes=10,
                      adjR2_controls=float(adj_r2(tau, X0)), adjR2_controls_L=float(adj_r2(tau, XL)), adjR2_controls_M=float(adj_r2(tau, XM)),
                      adjR2_controls_L_M=float(adj_r2(tau, XLM)), adjR2_controls_L_M_Mneg=float(adj_r2(tau, XLMn)),
                      F_L_given_controls=F_L_alone, p_L_given_controls=p_L_alone, F_L_given_M=F_L_given_M, p_L_given_M=p_L_given_M,
                      F_M_given_L=F_M_given_L, p_M_given_L=p_M_given_L, F_L_given_M_Mneg=F_L_given_Mn, p_L_given_M_Mneg=p_L_given_Mn,
                      residual_moran_I=float(moran_I(resid_full, Wg)))
    out[tag] = o
    print(f"{tag} n={n}: controls R2={r2c:.3f}; dC/dL|ln r raw={o['dC_dL_given_lnr_raw']:.3f} detr={o['dC_dL_given_lnr_detrended']:.3f}; "
          f"Mantel r(dtau,L)={rL:.3f} r(dtau,r)={rR:.3f} r(L,r)={rLR:.3f}; MSR(geoW) p={o['mantel']['msr_geoW']['p']:.3f} SES={o['mantel']['msr_geoW']['SES']:.1f} "
          f"(I_obs={I_obs:.3f}, I_sur={I_sur.mean():.3f}); MSR(laskerW) dtau~r p={o['mantel']['msr_laskerW_dtau_vs_r']['p']:.3f} SES={o['mantel']['msr_laskerW_dtau_vs_r']['SES']:.1f}; "
          f"dbMEM adjR2 ctrl={o['dbmem']['adjR2_controls']:.3f} +L={o['dbmem']['adjR2_controls_L']:.3f} +M={o['dbmem']['adjR2_controls_M']:.3f} +L+M={o['dbmem']['adjR2_controls_L_M']:.3f}; "
          f"F(L|ctrl)={F_L_alone:.1f} p={p_L_alone:.3f}; F(L|M)={F_L_given_M:.1f} p={p_L_given_M:.3f}; F(M|L)={F_M_given_L:.1f} p={p_M_given_L:.3f}; resid I={o['dbmem']['residual_moran_I']:.3f} [{time.time()-T0:.0f} s]", flush=True)

pd.DataFrame(strat_rows).to_csv(cf.PROC / "stage3_stratified_C.csv", index=False)
cf.save_json(out, cf.PROC / "stage3_inference.json")
# figure: C vs L quintile within r bins, 2026 raw and detrended
st = pd.DataFrame(strat_rows); fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a_, lab in zip(ax, ("raw", "detrended")):
    t = st[(st.tag == "2026") & (st.set == lab)]
    for (lo, hi), g in t.groupby(["r_lo", "r_hi"]):
        a_.errorbar(g.L_mean, g.C, g.C_se, marker="o", ms=3, label=f"{lo}–{hi} km")
    a_.axhline(0, color="k", lw=0.5); a_.set_xlabel("mean Lasker L in quintile"); a_.set_ylabel("C (pair product / var)"); a_.set_title(f"2026 τ' {lab}"); a_.legend(fontsize=7)
fig.tight_layout(); fig.savefig(cf.FIG / "stage3_C_L_given_r.png", dpi=150)
print(f"done in {time.time()-T0:.0f} s")
