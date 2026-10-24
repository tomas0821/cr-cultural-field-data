"""Stage 3a — Gaussian-free-field profile likelihood over (w, l_c, k) per election and jointly.

Model: tau' ~ N(0, s2 L_w^+ + smu I + diag(h/[N p(1-p)])), L_w = degree-reweighted combinatorial Laplacian
(adopted after G3a) of w*Gamma_geo(l_c) + (1-w)*Gamma_sur(k), kernels at equal total weight.
Grid: w in 0..1 (11), l_c in {1.5, 3, 4.5, 7, 10} km, k in {5, 8, 13}. Common unit set (tau' in all six
result sets, Lasker N >= 200) for the grid; primary full sets (2022 R1/R2, 2026) at the best (l_c, k) with
the heteroskedastic-noise alternative (smu_i = smu * N_med / N_i) and the sym / comb robustness operators.
Outputs: data/processed/stage3_profile_grid.csv, stage3_profile_summary.json, figures/stage3_profile_*.png
"""
import sys, time, json, argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--nmin", type=int, default=200)
ap.add_argument("--lcs", default="1.5,3,4.5,7,10"); ap.add_argument("--ks", default="5,8,13"); ap.add_argument("--variant", default="comb_rw"); ap.add_argument("--suffix", default="")
args = ap.parse_args(); T0 = time.time()
LCS = [float(x) for x in args.lcs.split(",")]; KS = [int(x) for x in args.ks.split(",")]
WG = np.round(np.linspace(0, 1, 11), 2); PRIMARY = ["2010", "2014", "2018", "2022r1", "2026"]

las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True)
res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
s1 = json.load(open(cf.PROC / "stage1_summary.json")); h = float(np.clip(s1["variance_decomposition"]["2022r1-2026"]["h"], 1, 3))
codes_all = las["codelec"].astype(str); Nel = las["N"] / 2.0
base = (Nel >= args.nmin) & np.isin(codes_all, res.index)

def unit_set(mask):
    codes = codes_all[mask]; Lm = las["L"][np.ix_(mask, mask)].astype(float)
    r = res.loc[codes]; D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
    return codes, Lm, D, r
def election_data(r, tag):
    N = r[f"N_{tag}"].to_numpy(float); p = r[f"p_{tag}"].to_numpy(float); tau = r[f"tau_{tag}"].to_numpy(float)
    return tau, h / (N * p * (1 - p)), N

common = base.copy()
for t in cf.TAGS: common &= np.isin(codes_all, res.index[np.isfinite(res[f"tau_{t}"])])
codes, Lm, D, r = unit_set(common); n = len(codes)
print(f"common set n={n} (Lasker N>={args.nmin}, tau' in all six sets); h={h:.2f}; variant={args.variant}; grid {len(LCS)}x{len(KS)}x{len(WG)}", flush=True)
ED = {t: election_data(r, t) for t in cf.TAGS}

rows = []; warm = {}
for lc, k in itertools.product(LCS, KS):
    G = cf.geo_kernel(D, lc); A = cf.knn_graph(Lm, k); t0 = time.time()
    for w in WG:
        P, lam, _ = cf.eig_pinv(cf.laplacian(cf.mix_kernels(G, A, w), args.variant)); dP = np.diag(P).mean()
        for t in cf.TAGS:
            tau, b, _ = ED[t]; x0 = warm.get(t, np.log([0.08 / dP, 0.10]))
            f, x = cf.profile_nll([tau], P, b, x0); warm[t] = x
            rows.append(dict(lc=lc, k=k, w=w, tag=t, nll=f, s2=np.exp(x[0]), smu=np.exp(x[1]), sigma_phi2=np.exp(x[0]) * dP, lambda2=lam[1] if len(lam) > 1 else np.nan))
        taus = [ED[t][0] for t in PRIMARY]; bj = np.mean([ED[t][1] for t in PRIMARY], 0); x0 = warm.get("joint", np.log([0.08 / dP, 0.10]))
        f, x = cf.profile_nll(taus, P, bj, x0); warm["joint"] = x
        rows.append(dict(lc=lc, k=k, w=w, tag="joint5", nll=f, s2=np.exp(x[0]), smu=np.exp(x[1]), sigma_phi2=np.exp(x[0]) * dP, lambda2=lam[1]))
    g = pd.DataFrame([x for x in rows if x["lc"] == lc and x["k"] == k])
    jm = g[g.tag == "joint5"]; wb = jm.w.iloc[jm.nll.argmin()]
    print(f"  lc={lc:4.1f} k={k:2d}: joint5 best w={wb:.1f} (min NLL {jm.nll.min():.1f}); per election best w: " +
          " ".join(f"{t}:{g[g.tag==t].w.iloc[g[g.tag==t].nll.argmin()]:.1f}" for t in cf.TAGS) + f" [{time.time()-t0:.0f} s]", flush=True)
grid = pd.DataFrame(rows); grid.to_csv(cf.PROC / f"stage3_profile_grid{args.suffix}.csv", index=False)

# ---------------------------------------------------------------- summaries
summary = dict(n_common=n, h=h, variant=args.variant, lcs=LCS, ks=KS)
def prof_ci(g):
    """profile over w with (lc,k) profiled out: min over lc,k per w; CI = dNLL <= 1.92."""
    pw = g.groupby("w").nll.min(); pw = pw - pw.min(); ok = pw[pw <= 1.92]
    return dict(w_hat=float(pw.idxmin()), w_ci=[float(ok.index.min()), float(ok.index.max())], profile=[float(x) for x in pw.values])
per = {}
for t in cf.TAGS + ["joint5"]:
    g = grid[grid.tag == t]; i = g.nll.idxmin(); b = g.loc[i]
    per[t] = dict(best=dict(lc=float(b.lc), k=int(b.k), w=float(b.w), nll=float(b.nll), s2=float(b.s2), smu=float(b.smu), sigma_phi2=float(b.sigma_phi2)), **prof_ci(g))
    # profile over lc and k (w profiled out)
    per[t]["lc_profile"] = {str(l): float(g[g.lc == l].nll.min() - g.nll.min()) for l in LCS}
    per[t]["k_profile"] = {str(kk): float(g[g.k == kk].nll.min() - g.nll.min()) for kk in KS}
    print(f"{t:7s}: best lc={b.lc} k={b.k} w={b.w} sigma_phi2={b.sigma_phi2:.3f} smu={b.smu:.3f}; w_hat={per[t]['w_hat']} CI {per[t]['w_ci']}; "
          f"dNLL(w=0)={per[t]['profile'][0]:.0f} dNLL(w=0.5)={per[t]['profile'][5]:.0f} dNLL(w=0.9)={per[t]['profile'][9]:.0f}", flush=True)
summary["per_set"] = per
best = per["joint5"]["best"]; lc_b, k_b = best["lc"], best["k"]

# ---------------------------------------------------------------- primary full sets at best (lc, k): het noise + robustness operators
codesF, LmF, DF, rF = unit_set(base); nF = len(codesF)
GF = cf.geo_kernel(DF, lc_b); AF = cf.knn_graph(LmF, k_b)
full = {}
PV = {v: {w: cf.eig_pinv(cf.laplacian(cf.mix_kernels(GF, AF, w), v))[0] for w in WG} for v in ("comb_rw", "sym", "comb")}
for t in ("2022r1", "2022r2", "2026"):
    tau, b, N = election_data(rF, t); m = np.isfinite(tau); het = np.median(N[m]) / N[m]
    out = {}
    for v in ("comb_rw", "sym", "comb"):
        pr = []; x0 = np.log([0.08 / np.diag(PV[v][1.0]).mean(), 0.1])
        for w in WG:
            P = PV[v][w][np.ix_(m, m)]; f, x0 = cf.profile_nll([tau[m]], P, b[m], x0); pr.append(f)
        pr = np.array(pr) - min(pr); out[v] = dict(w_hat=float(WG[pr.argmin()]), profile=[float(x) for x in pr])
    pr = []; x0 = np.log([0.08 / np.diag(PV["comb_rw"][1.0]).mean(), 0.1])
    for w in WG:
        P = PV["comb_rw"][w][np.ix_(m, m)]; f, x0 = cf.profile_nll([tau[m]], P, b[m], x0, het=het); pr.append(f)
    pr = np.array(pr) - min(pr); out["comb_rw_heteroskedastic"] = dict(w_hat=float(WG[pr.argmin()]), profile=[float(x) for x in pr])
    full[t] = dict(n=int(m.sum()), **out)
    print(f"full set {t} (n={m.sum()}, lc={lc_b}, k={k_b}): " + "; ".join(f"{v}: w_hat={o['w_hat']} dNLL(0)={o['profile'][0]:.0f} dNLL(0.5)={o['profile'][5]:.0f}" for v, o in out.items()), flush=True)
summary["primary_full_sets"] = full; summary["wall_s"] = time.time() - T0
cf.save_json(summary, cf.PROC / f"stage3_profile_summary{args.suffix}.json")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for t in cf.TAGS + ["joint5"]:
    ax[0].plot(WG, per[t]["profile"], "o-", ms=3, lw=1.2 if t == "joint5" else 0.8, label=t)
ax[0].axhline(1.92, color="gray", ls=":", lw=0.8); ax[0].set_ylim(0, 60); ax[0].set_xlabel("w (1 = geographic)"); ax[0].set_ylabel("ΔNLL (ℓ_c, k profiled out)"); ax[0].legend(fontsize=7)
for t in ("2022r1", "2026"):
    for v, ls in (("comb_rw", "-"), ("sym", "--"), ("comb", ":"), ("comb_rw_heteroskedastic", "-.")):
        ax[1].plot(WG, full[t][v]["profile"], ls, lw=1, label=f"{t} {v}")
ax[1].set_ylim(0, 60); ax[1].set_xlabel("w"); ax[1].set_ylabel("ΔNLL (full sets, best ℓ_c, k)"); ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(cf.FIG / f"stage3_profile_w{args.suffix}.png", dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for t in PRIMARY + ["joint5"]:
    ax[0].plot(LCS, [per[t]["lc_profile"][str(l)] for l in LCS], "o-", ms=3, label=t); ax[1].plot(KS, [per[t]["k_profile"][str(kk)] for kk in KS], "o-", ms=3, label=t)
ax[0].set_xlabel("ℓ_c (km)"); ax[0].set_ylabel("ΔNLL (w, k profiled out)"); ax[0].set_ylim(0, 100); ax[1].set_xlabel("k"); ax[1].set_ylabel("ΔNLL (w, ℓ_c profiled out)"); ax[1].set_ylim(0, 100); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(cf.FIG / f"stage3_profile_lc_k{args.suffix}.png", dpi=150)
print(f"done in {time.time()-T0:.0f} s")
