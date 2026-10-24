"""Stage 1 — Borghesi–Bouchaud replication on Costa Rican electoral districts.

Per election (tags in crfield.TAGS):
  tau = logit(recibidos / electores_padron), rho = ln[W / (validos - W)] (national winner W);
  size-bin means removed (~100 units/bin), 5 sigma within-bin cut;
  C(r) flat pair average / variance, variogram V(r), unit bootstrap; log fit -lambda^2 ln(r/L).
Variance decomposition (B&B 2010 Eq. 8-10): binned var(tau) = B cov(t,t') + h <1/(N p(1-p))>,
  on 2022 R1 / R2 / 2026 (same roll cut) and on the P5 panel; temporal correlation C(dT).
Outputs: data/processed/stage1_residuals.csv, stage1_Cr.csv, stage1_summary.json, figures/stage1_*.png
Usage: python3 scripts/stage1_replication.py [--nmin 100] [--nboot 200]
"""
import sys, argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--nmin", type=int, default=100); ap.add_argument("--nboot", type=int, default=200)
args = ap.parse_args()
cf.FIG.mkdir(exist_ok=True)
master = cf.load_master()
print(f"spatial units: {len(master)}")

# ------------------------------------------------------------------ observables per election
res_tab = master[["codelec", "provincia", "canton", "E_crtm05", "N_crtm05", "N_2026"]].copy()
summary = {}
for tag in cf.TAGS:
    Ne = master[f"electores_{tag}"].to_numpy(float); V = master[f"recibidos_{tag}"].to_numpy(float)
    ok = np.isfinite(Ne) & np.isfinite(V) & (Ne >= args.nmin) & (V > 0) & (V < Ne)
    p = np.full(len(master), np.nan); p[ok] = V[ok] / Ne[ok]
    tau = np.full(len(master), np.nan); tau[ok] = cf.logit(p[ok])
    r_tau, keep, _ = cf.size_bin_residual(tau[ok], Ne[ok])
    tau_res = np.full(len(master), np.nan); idx = np.flatnonzero(ok)[keep]; tau_res[idx] = r_tau[keep]
    # winner log-odds
    wv = cf.winner_votes(tag); wmap = wv.set_index("codelec")
    W = master.codelec.map(wmap["W"]).to_numpy(float); Vv = master.codelec.map(wmap["validos"]).to_numpy(float)
    okr = ok & np.isfinite(W) & np.isfinite(Vv) & (W > 0) & (Vv - W > 0)
    rho = np.full(len(master), np.nan); rho[okr] = np.log(W[okr] / (Vv[okr] - W[okr]))
    r_rho, keepr, _ = cf.size_bin_residual(rho[okr], Ne[okr])
    rho_res = np.full(len(master), np.nan); idr = np.flatnonzero(okr)[keepr]; rho_res[idr] = r_rho[keepr]
    res_tab[f"N_{tag}"] = Ne; res_tab[f"p_{tag}"] = p; res_tab[f"tau_{tag}"] = tau_res; res_tab[f"rho_{tag}"] = rho_res
    binom = 1 / (Ne[idx] * p[idx] * (1 - p[idx]))
    summary[tag] = dict(winner=wv.attrs["winner"], n_tau=int(len(idx)), n_rho=int(len(idr)), turnout=float(V[ok].sum() / Ne[ok].sum()),
                        var_tau=float(np.var(tau_res[idx])), binom_mean=float(binom.mean()), var_rho=float(np.var(rho_res[idr])),
                        outliers_tau=int((~keep).sum()), outliers_rho=int((~keepr).sum()))
    print(f"{tag}: winner {wv.attrs['winner']}, n_tau={len(idx)} n_rho={len(idr)} turnout={summary[tag]['turnout']:.3f} "
          f"var(tau')={summary[tag]['var_tau']:.4f} binom={summary[tag]['binom_mean']:.4f} var(rho')={summary[tag]['var_rho']:.4f}")
res_tab.to_csv(cf.PROC / "stage1_residuals.csv", index=False)

# ------------------------------------------------------------------ C(r), V(r), log fits
E = master.E_crtm05.to_numpy(float); Nc = master.N_crtm05.to_numpy(float)
rows = []
def run_cr(tag, obs, sel_mask, label):
    m = np.isfinite(res_tab[f"{obs}_{tag}"].to_numpy()) & sel_mask
    x = res_tab.loc[m, f"{obs}_{tag}"].to_numpy(); D = cf.pair_distance_km(E[m], Nc[m])
    tab, var, _ = cf.pair_stats(x, D, n_boot=args.nboot, seed=hash((tag, obs, label)) % 2**31)
    lam2, L = cf.fit_log(tab.r_mid.to_numpy(), tab.C.to_numpy(), tab.C_se.to_numpy())
    # variogram slope on ln r in the same window (V = 2 sigma^2 (1 - C) -> slope 2 sigma^2 lambda^2)
    w = (tab.r_mid >= 8) & (tab.r_mid <= 300) & tab.V.notna()
    vs = np.polyfit(np.log(tab.r_mid[w]), tab.V[w], 1)[0] if w.sum() >= 3 else np.nan
    tab.insert(0, "set", label); tab.insert(0, "obs", obs); tab.insert(0, "tag", tag)
    rows.append(tab)
    return dict(n=int(m.sum()), var=float(var), lambda2=float(lam2), L_km=float(L), variogram_slope=float(vs),
                C_1_4km=float(np.nanmean(tab.C[(tab.r_mid < 4)])), C_15_50km=float(np.nanmean(tab.C[(tab.r_mid > 15) & (tab.r_mid < 50)])),
                C_100_250km=float(np.nanmean(tab.C[(tab.r_mid > 100) & (tab.r_mid < 250)])))
full = np.ones(len(master), bool)
p5 = np.all([np.isfinite(res_tab[f"tau_{t}"]) for t in cf.TAGS], axis=0)
print(f"P5 panel (tau' in all six sets): {p5.sum()} units")
for tag in cf.TAGS:
    for obs in ("tau", "rho"):
        summary[tag][f"Cr_{obs}"] = run_cr(tag, obs, full, "full")
        s = summary[tag][f"Cr_{obs}"]
        print(f"  C(r) {obs} {tag}: n={s['n']} C(<4km)={s['C_1_4km']:.3f} C(15-50)={s['C_15_50km']:.3f} C(100-250)={s['C_100_250km']:.3f} "
              f"lambda2={s['lambda2']:.4f} L={s['L_km']:.0f} km  V-slope={s['variogram_slope']:.4f}")
for tag in ("2022r1", "2026"):
    summary[tag]["Cr_tau_P5"] = run_cr(tag, "tau", p5, "P5")
    print(f"  P5 check {tag}: C(15-50)={summary[tag]['Cr_tau_P5']['C_15_50km']:.3f} vs full {summary[tag]['Cr_tau']['C_15_50km']:.3f}")
Cr = pd.concat(rows, ignore_index=True); Cr.to_csv(cf.PROC / "stage1_Cr.csv", index=False)
g1 = sum(summary[t]["Cr_tau"]["C_15_50km"] > 0 for t in cf.TAGS)
summary["gate_G1"] = dict(elections_with_positive_C_15_50km=int(g1), passed=bool(g1 >= 4))

# ------------------------------------------------------------------ variance decomposition
def binned_var_cov(t1, t2, nb=100):
    m = np.isfinite(res_tab[f"tau_{t1}"]) & np.isfinite(res_tab[f"tau_{t2}"])
    a = res_tab.loc[m, f"tau_{t1}"].to_numpy(); b = res_tab.loc[m, f"tau_{t2}"].to_numpy()
    N = res_tab.loc[m, f"N_{t1}"].to_numpy(); p1 = res_tab.loc[m, f"p_{t1}"].to_numpy(); p2 = res_tab.loc[m, f"p_{t2}"].to_numpy()
    order = np.argsort(N); k = max(1, len(a) // nb)
    out = []
    for i in range(k):
        idx = order[i * nb:(i + 1) * nb if i < k - 1 else None]
        out.append(dict(N=np.median(N[idx]), var1=np.var(a[idx]), var2=np.var(b[idx]), cov12=np.mean(a[idx] * b[idx]) - a[idx].mean() * b[idx].mean(),
                        binom1=np.mean(1 / (N[idx] * p1[idx] * (1 - p1[idx]))), binom2=np.mean(1 / (N[idx] * p2[idx] * (1 - p2[idx])))))
    return pd.DataFrame(out), m.sum()
vd = {}
for t1, t2 in [("2022r1", "2022r2"), ("2022r1", "2026"), ("2022r2", "2026"), ("2010", "2014"), ("2014", "2018"), ("2018", "2022r1")]:
    tb, n = binned_var_cov(t1, t2)
    # B&B Eq. 10: var = B cov + h binom, fitted over size bins using both elections' variances
    X = np.r_[np.c_[tb.cov12, tb.binom1], np.c_[tb.cov12, tb.binom2]]; y = np.r_[tb.var1, tb.var2]
    B, h = np.linalg.lstsq(X, y, rcond=None)[0]
    # direct: persistent part = cov, noise = var - cov, herding ratio noise/binomial
    ratio = float(((tb.var1 - tb.cov12) / tb.binom1).median())
    corr = float(np.corrcoef(res_tab.loc[np.isfinite(res_tab[f"tau_{t1}"]) & np.isfinite(res_tab[f"tau_{t2}"]), f"tau_{t1}"],
                             res_tab.loc[np.isfinite(res_tab[f"tau_{t1}"]) & np.isfinite(res_tab[f"tau_{t2}"]), f"tau_{t2}"])[0, 1])
    vd[f"{t1}-{t2}"] = dict(n=int(n), B=float(B), h=float(h), noise_over_binomial_median=ratio, corr=corr,
                            cov_mean=float(tb.cov12.mean()), var_mean=float((tb.var1.mean() + tb.var2.mean()) / 2), binom_mean=float((tb.binom1.mean() + tb.binom2.mean()) / 2))
    print(f"  var decomposition {t1}-{t2}: n={n} corr={corr:.3f} cov={tb.cov12.mean():.4f} var={vd[f'{t1}-{t2}']['var_mean']:.4f} "
          f"binom={vd[f'{t1}-{t2}']['binom_mean']:.4f}  Eq.10 fit B={B:.2f} h={h:.2f}  (var-cov)/binom median={ratio:.2f}")
    vd[f"{t1}-{t2}"]["bins"] = tb.round(5).to_dict("list")
summary["variance_decomposition"] = vd

# temporal correlation C(dT) over all election pairs (common units)
years = {"2010": 2010.1, "2014": 2014.1, "2018": 2018.1, "2022r1": 2022.1, "2022r2": 2022.25, "2026": 2026.1}
tc = []
for t1, t2 in itertools.combinations(cf.TAGS, 2):
    m = np.isfinite(res_tab[f"tau_{t1}"]) & np.isfinite(res_tab[f"tau_{t2}"])
    c = np.corrcoef(res_tab.loc[m, f"tau_{t1}"], res_tab.loc[m, f"tau_{t2}"])[0, 1]
    tc.append(dict(t1=t1, t2=t2, dT=abs(years[t2] - years[t1]), n=int(m.sum()), corr=float(c)))
summary["temporal_correlation"] = tc
for r in tc: print(f"  C(dT={r['dT']:5.2f} y) {r['t1']}-{r['t2']}: {r['corr']:.3f} (n={r['n']})")

# ------------------------------------------------------------------ extra B&B checks
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
h_ref = summary["variance_decomposition"]["2022r1-2026"]["h"]
for tag in cf.TAGS:
    m = np.isfinite(res_tab[f"tau_{tag}"]).to_numpy()
    x = res_tab.loc[m, f"tau_{tag}"].to_numpy(); D = cf.pair_distance_km(E[m], Nc[m]); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :16]; xbar = x[nn].mean(1)
    slope = np.polyfit(xbar, x, 1)[0]
    N = res_tab.loc[m, f"N_{tag}"].to_numpy(); pp = res_tab.loc[m, f"p_{tag}"].to_numpy()
    order = np.argsort(N); nb = 200; k = len(x) // nb
    Nb = np.array([np.median(N[order[i*nb:(i+1)*nb]]) for i in range(k)])
    s2 = np.array([np.var(x[order[i*nb:(i+1)*nb]]) - h_ref * np.mean(1/(N[order[i*nb:(i+1)*nb]]*pp[order[i*nb:(i+1)*nb]]*(1-pp[order[i*nb:(i+1)*nb]]))) for i in range(k)])
    try:
        (s_phi, c, om), _ = curve_fit(lambda n, a, c, w: a + c * n ** (-w), Nb, s2, p0=[s2.min(), 1.0, 0.75], maxfev=20000)
    except Exception:
        s_phi, c, om = np.nan, np.nan, np.nan
    C0_unnorm = summary[tag]["Cr_tau"]["var"] * summary[tag]["Cr_tau"]["C_1_4km"]
    summary[tag]["bb_checks"] = dict(nn16_slope=float(slope), skew=float(skew(x)), kurt=float(kurtosis(x)),
                                     sigma2_phi_extrap=float(s_phi), omega=float(om), C0_unnormalised_1_4km=float(C0_unnorm))
    print(f"  B&B checks {tag}: 16-NN slope={slope:.2f} skew={skew(x):.2f} kurt={kurtosis(x):.2f} sigma2_phi(N->inf)={s_phi:.4f} omega={om:.2f} vs var*C(<4km)={C0_unnorm:.4f}")

cf.save_json(summary, cf.PROC / "stage1_summary.json")

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, obs in zip(axes, ("tau", "rho")):
    for tag in cf.TAGS:
        t = Cr[(Cr.tag == tag) & (Cr.obs == obs) & (Cr.set == "full")]
        ax.errorbar(t.r_mid, t.C, t.C_se, marker="o", ms=3, lw=1, capsize=2, label=tag)
    ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log"); ax.set_xlabel("r (km)")
    ax.set_ylabel("C(r)"); ax.set_title({"tau": "logit turnout τ'", "rho": "winner log-odds ρ'"}[obs]); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(cf.FIG / "stage1_Cr.png", dpi=150)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for tag in cf.TAGS:
    t = Cr[(Cr.tag == tag) & (Cr.obs == "tau") & (Cr.set == "full")]
    ax.errorbar(t.r_mid, t.V, t.V_se, marker="o", ms=3, lw=1, capsize=2, label=tag)
ax.set_xscale("log"); ax.set_xlabel("r (km)"); ax.set_ylabel("V(r) = <(τ'_a − τ'_b)²>"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(cf.FIG / "stage1_variogram.png", dpi=150)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
tb = pd.DataFrame(vd["2022r1-2022r2"]["bins"])
ax.loglog(tb.N, tb.var1, "o", label="var τ' (2022 R1)"); ax.loglog(tb.N, tb.cov12, "s", label="cov(R1, R2)")
ax.loglog(tb.N, tb.binom1, "-", label="binomial 1/(N p(1−p))"); ax.loglog(tb.N, tb.var1 - tb.cov12, "^", label="var − cov")
ax.set_xlabel("N (registered electors)"); ax.set_ylabel("variance"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(cf.FIG / "stage1_variance_vs_N.png", dpi=150)
print("done:", cf.PROC / "stage1_summary.json")
